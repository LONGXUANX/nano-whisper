import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanowhisper.utils.context import get_context

class AttentionType:
    """
    Attention type.
    Use string to be compatible with `torch.compile`.
    """
    # Decoder attention between previous layer Q/K/V
    DECODER = "decoder"
    # Encoder attention between previous layer Q/K/V for encoder-decoder
    ENCODER = "encoder"
    # Attention between dec. Q and enc. K/V for encoder-decoder
    ENCODER_DECODER = "encoder_decoder"

@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
    block_size: tl.constexpr,  # The size of the subblock for each processing
    offset: tl.constexpr       # The start offset of the sub-block
):
    idx = tl.program_id(0)  # current batch
    offsets = tl.arange(0, block_size) + offset  # The index range of the current subblock

    # Handle Key offset and loading
    key_offsets = idx * key_stride + offsets
    key = tl.load(key_ptr + key_offsets, mask=offsets < D, other=0.0)  # No more than D

    # Handle value offset and loading
    value_offsets = idx * value_stride + offsets
    value = tl.load(value_ptr + value_offsets, mask=offsets < D, other=0.0)  # No more than D

    # Handle storage offsets in the Cache
    slot = tl.load(slot_mapping_ptr + idx)  # get slot
    cache_offsets = slot * D + offsets
    tl.store(k_cache_ptr + cache_offsets, key, mask=offsets < D)
    tl.store(v_cache_ptr + cache_offsets, value, mask=offsets < D)

def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor
):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim  # Total dimension

    # Chunk size, a power of two (e.g., 1024)
    block_size = 1024
    num_blocks = (D + block_size - 1) // block_size  # Calculate the number of blocks needed

    # Each sub-block is traversed and the Triton kernel is called block by block
    for block_idx in range(num_blocks):
        offset = block_idx * block_size  # The start offset of the sub-block
        current_block_size = min(block_size, D - offset)     # The current sub-block size is computed dynamically
        store_kvcache_kernel[(N,)](
            key, key.stride(0),
            value, value.stride(0),
            k_cache, v_cache,
            slot_mapping,
            D,
            current_block_size,
            offset
        )

class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        attn_type,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.attn_type = attn_type

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        if k is not None and v is not None:
            k = k.view(-1, self.num_kv_heads, self.head_dim)
            v = v.view(-1, self.num_kv_heads, self.head_dim)
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            updated_slot_mapping = None
            if (self.attn_type != AttentionType.ENCODER) and (k is not None) and (
                    v is not None):
                if self.attn_type == AttentionType.ENCODER_DECODER:
                    # Update cross-attention KV cache (prefill-only)
                    updated_slot_mapping = context.cross_slot_mapping
                else:
                    # Update self-attention KV cache (prefill/decode)
                    updated_slot_mapping = context.slot_mapping
            if updated_slot_mapping is not None:
                store_kvcache(k, v, k_cache, v_cache, updated_slot_mapping)
        if context.is_prefill:
            # if context.block_tables is not None:    # prefix cache
            #     k, v = k_cache, v_cache
            if self.attn_type == AttentionType.ENCODER:
                o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.encoder_seq_lens, cu_seqlens_q=context.encoder_cu_seqlens,
                                       max_seqlen_k=context.encoder_seq_lens, cu_seqlens_k=context.encoder_cu_seqlens,
                                       softmax_scale=self.scale, causal=False)
            elif self.attn_type == AttentionType.DECODER:
                o = flash_attn_varlen_func(q, k, v,
                                        max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                        max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                        softmax_scale=self.scale, causal=True)
            elif self.attn_type == AttentionType.ENCODER_DECODER:
                o = flash_attn_varlen_func(q, k, v,
                                        max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                        max_seqlen_k=context.encoder_seq_lens, cu_seqlens_k=context.encoder_cu_seqlens,
                                        softmax_scale=self.scale, causal=False)
        else:    # decode
            if self.attn_type == AttentionType.DECODER:
                o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                            cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                            softmax_scale=self.scale, causal=True)
            elif self.attn_type == AttentionType.ENCODER_DECODER:
                o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                            cache_seqlens=context.cross_context_lens, block_table=context.cross_block_tables, 
                                            softmax_scale=self.scale, causal=True)
            elif self.attn_type == AttentionType.ENCODER:
                raise NotImplementedError("decode with encoder attention is wrong!!!")
        o = o.view(-1, self.num_heads * self.head_dim)
        return o

