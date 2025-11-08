import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open

def _create_fake_bias_for_k_proj(weights):
    """
    add fake bias to k_proj, so that qkv_proj initializes bias correctly
    """
    for name, weight in weights:
        if name.endswith(".k_proj.weight"):
            bias = torch.zeros(weight.size(0))
            bias_name = name.replace("weight", "bias")
            yield (name, weight)
            yield (bias_name, bias)
        else:
            yield (name, weight)

def map_name(name: str, hf_to_vllm_mapper: dict) -> str:
    for old, new in hf_to_vllm_mapper.items():
        name = name.replace(old, new)
    return name


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(
    model: nn.Module,
    path: str,
):
    stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".self_attn.qkv_proj", ".self_attn.q_proj", "q"),
            (".self_attn.qkv_proj", ".self_attn.k_proj", "k"),
            (".self_attn.qkv_proj", ".self_attn.v_proj", "v"),
            (".encoder_attn.kv_proj", ".encoder_attn.k_proj", "k"),
            (".encoder_attn.kv_proj", ".encoder_attn.v_proj", "v"),
        ]
    hf_to_vllm_mapper = {
        ".fc1.": ".mlp.fc1.",
        ".fc2.": ".mlp.fc2."
    }
    skip_prefixes=["proj_out."]
    # 1. safetensors
    all_weights = []
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                all_weights.append((weight_name, f.get_tensor(weight_name)))

    # 2. add fake bias
    all_weights = list(_create_fake_bias_for_k_proj(all_weights))

    # 3. Do weight name mapping from HF to nanowhisper
    all_weights = [(map_name(name, hf_to_vllm_mapper), tensor) for name, tensor in all_weights]

    # 4. skip prefix
    if skip_prefixes is not None:
        all_weights = [
            (name, weight)
            for name, weight in all_weights
            if not any(name.startswith(prefix) for prefix in skip_prefixes)
        ]

    params_dict = dict(model.named_parameters())
    loaded_params: set[str] = set()
    for name, loaded_weight in all_weights:
        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in name:
                continue
            name = name.replace(weight_name, param_name)
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue

            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, loaded_weight, shard_id)
            break
        else:
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
        loaded_params.add(name)
    return loaded_params

