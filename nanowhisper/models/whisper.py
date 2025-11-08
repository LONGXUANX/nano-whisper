import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Optional, TypedDict, Union

import torch
import torch.distributed as dist
from torch import nn
from transformers import WhisperConfig
from transformers.models.whisper.modeling_whisper import sinusoids

from nanowhisper.layers.attention import Attention, AttentionType
from nanowhisper.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanowhisper.layers.activation import get_act_fn

from nanowhisper.layers.linear import QKVParallelLinear, ColumnParallelLinear, RowParallelLinear


def cast_overflow_tensors(
    tensors: torch.Tensor,
    offset: float = 1000,
) -> torch.Tensor:
    if tensors.isinf().any() or tensors.isnan().any():
        clamp_value = torch.finfo(tensors.dtype).max - offset
        tensors = torch.clamp(tensors, min=-clamp_value, max=clamp_value)
    return tensors


class WhisperPositionalEmbedding(nn.Embedding):

    def __init__(self, num_positions: int, embedding_dim: int):
        super().__init__(num_positions, embedding_dim)

    def forward(self, position_ids):
        return self.weight[position_ids]


class WhisperAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        attn_type: AttentionType = AttentionType.DECODER,
        prefix: str = "",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        if self.total_num_heads >= tp_size:
            # Number of heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_heads % tp_size == 0
        else:
            # Number of heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_heads == 0
        self.num_kv_heads = max(1, self.total_num_heads // tp_size)
        self.head_dim = self.embed_dim // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.attn_type = attn_type

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: "
                f"{self.embed_dim} and `num_heads`: {num_heads}).")
        self.scaling = self.head_dim**-0.5

        self._init_qkv(embed_dim, bias, prefix=prefix)
        self.out_proj = RowParallelLinear(
            input_size=embed_dim,
            output_size=embed_dim,
            bias=bias,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            self.attn_type,
        )

    def _init_qkv(
        self,
        embed_dim: int,
        bias: bool = True,
        prefix: str = "",
    ) -> None:
        self.qkv_proj = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_heads,
            bias=bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        attn_output = self.attn(q, k, v)

        output = self.out_proj(attn_output)

        return output


class WhisperCrossAttention(WhisperAttention):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        prefix: str = "",
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=bias,
            prefix=prefix,
            attn_type=AttentionType.ENCODER_DECODER,
        )

    def _init_qkv(
        self,
        embed_dim: int,
        bias: bool = True,
        prefix: str = "",
    ) -> None:
        self.q_proj = ColumnParallelLinear(
            input_size=embed_dim,
            output_size=embed_dim,
            bias=bias,
        )
        self.kv_proj = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=self.head_dim,
            total_num_heads=0,
            total_num_kv_heads=self.total_num_heads,
            bias=bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
    ):
        q = self.q_proj(hidden_states)

        # Encoder hidden states are only computed once during prefill phase.
        # Afterwards, the keys and values should be available in the kv-cache.
        if encoder_hidden_states is not None:
            kv = self.kv_proj(encoder_hidden_states)
            k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
        else:
            k = v = None

        attn_output = self.attn(q, k, v)

        output = self.out_proj(attn_output)

        return output


class WhisperMLP(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        act_fn: str,
        prefix: str = "",
    ):
        super().__init__()

        self.activation_fn = get_act_fn(act_fn)
        self.fc1 = ColumnParallelLinear(
            input_size=embed_dim,
            output_size=ffn_dim,
            bias=True,
        )
        self.fc2 = RowParallelLinear(
            input_size=ffn_dim,
            output_size=embed_dim,
            bias=True,
        )

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class WhisperEncoderLayer(nn.Module):

    def __init__(self, *, config: WhisperConfig, prefix: str = ""):
        super().__init__()

        self.embed_dim = config.d_model
        self.self_attn = WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            attn_type=AttentionType.ENCODER,
            prefix=f"{prefix}.self_attn",
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.mlp = WhisperMLP(
            embed_dim=config.d_model,
            ffn_dim=config.encoder_ffn_dim,
            act_fn=config.activation_function,
            prefix=f"{prefix}.mlp",
        )
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        hidden_states = cast_overflow_tensors(hidden_states)

        return hidden_states


class WhisperDecoderLayer(nn.Module):

    def __init__(self, *, config: WhisperConfig, prefix: str = ""):
        super().__init__()

        self.self_attn = WhisperAttention(
            embed_dim=config.d_model,
            num_heads=config.decoder_attention_heads,
            attn_type=AttentionType.DECODER,
            prefix=f"{prefix}.self_attn",
        )
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.encoder_attn = WhisperCrossAttention(
            embed_dim=config.d_model,
            num_heads=config.decoder_attention_heads,
            prefix=f"{prefix}.encoder_attn",
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.mlp = WhisperMLP(
            embed_dim=config.d_model,
            ffn_dim=config.decoder_ffn_dim,
            act_fn=config.activation_function,

            prefix=f"{prefix}.mlp",
        )
        self.final_layer_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
    ):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        hidden_states = self.encoder_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class WhisperEncoder(nn.Module):

    def __init__(self, *, config: WhisperConfig, prefix: str = ""):
        super().__init__()
        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.embed_scale = (math.sqrt(embed_dim)
                            if config.scale_embedding else 1.0)

        self.conv1 = nn.Conv1d(self.num_mel_bins,
                               embed_dim,
                               kernel_size=3,
                               padding=1)
        self.conv2 = nn.Conv1d(embed_dim,
                               embed_dim,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.embed_positions = nn.Embedding(self.max_source_positions,
                                            embed_dim)
        self.layers = nn.ModuleList([
            WhisperEncoderLayer(config=config, prefix=f"{prefix}.layers.{idx}")
            for idx in range(config.encoder_layers)
        ])
        self.layer_norm = nn.LayerNorm(config.d_model)

        with torch.no_grad():
            self.embed_positions.weight.copy_(
                sinusoids(*self.embed_positions.weight.shape))

    # @torch.compile
    def forward(self, input_features: Union[torch.Tensor, list[torch.Tensor]]):
        hidden_states = []
        for features in input_features:
            embeds = nn.functional.gelu(self.conv1(features))
            embeds = nn.functional.gelu(self.conv2(embeds))
            embeds = embeds.permute(1, 0)
            embeds = embeds + self.embed_positions.weight[:embeds.size(0), :]
            hidden_states.append(embeds)
        hidden_states = torch.cat(hidden_states)

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)

        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class WhisperDecoder(nn.Module):

    def __init__(self, *, config: WhisperConfig, prefix: str = ""):
        super().__init__()
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_target_positions
        self.max_source_positions = config.max_source_positions
        self.embed_scale = (math.sqrt(config.d_model)
                            if config.scale_embedding else 1.0)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model,
                                         self.padding_idx)
        self.embed_positions = WhisperPositionalEmbedding(
            self.max_target_positions, config.d_model)
        self.layers = nn.ModuleList([
            WhisperDecoderLayer(config=config, prefix=f"{prefix}.layers.{idx}")
            for idx in range(config.decoder_layers)
        ])
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        input_ids,
        positions: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
    ):
        inputs_embeds = self.get_input_embeddings(input_ids)
        positions = self.embed_positions(positions)
        hidden_states = inputs_embeds + positions

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )

        hidden_states = self.layer_norm(hidden_states)
        return hidden_states

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.embed_tokens(input_ids)


class WhisperModel(nn.Module):

    def __init__(self, *, config: WhisperConfig, prefix: str = ""):
        super().__init__()
        self.encoder = WhisperEncoder(config=config,
                                      prefix=f"{prefix}.encoder")
        self.decoder = WhisperDecoder(config=config,
                                      prefix=f"{prefix}.decoder")

    def forward(
        self,
        input_features: Optional[Union[torch.Tensor, list[torch.Tensor]]],
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
    ) -> torch.Tensor:
        encoder_outputs = self.get_encoder_outputs(input_features)
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            positions=positions,
            encoder_hidden_states=encoder_outputs,
        )
        return decoder_outputs

    def get_encoder_outputs(
        self,
        input_features: Optional[Union[torch.Tensor, list[torch.Tensor]]],
    ) -> Optional[torch.Tensor]:
        if input_features is None:
            return None
        return self.encoder(input_features)


class WhisperForConditionalGeneration(nn.Module):
    def __init__(self, config: WhisperConfig, prefix: str = ""):
        super().__init__()
        self.config = config
        self.model = WhisperModel(config=config, prefix=prefix)
        self.unpadded_vocab_size = config.vocab_size
        self.proj_out = ParallelLMHead(config.vocab_size,
                                       config.d_model)
        self.proj_out.weight = self.model.decoder.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        audio_tensor: Optional[Union[torch.Tensor, list[torch.Tensor]]]=None,
    ) -> torch.Tensor:
        decoder_outputs = self.model(
            input_features=audio_tensor,
            input_ids=input_ids,
            positions=positions,
        )
        return decoder_outputs
    
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.proj_out(hidden_states)
        return logits


 
