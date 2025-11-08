import torch
from nanowhisper.models.whisper import WhisperForConditionalGeneration
from nanowhisper.utils.loader_whisper import load_model
from nanowhisper.config import Config
import os
import torch.distributed as dist

def compare_model_with_pt(model, pt_path, verbose=True):
    # 读取保存的参考权重
    ref_state = torch.load(pt_path)
    ref_state = {"model." + k: v for k, v in ref_state.items()}
    # 当前模型参数
    curr_state = {name: param.detach().cpu() for name, param in model.named_parameters()}
    names_ref = set(ref_state.keys())
    names_curr = set(curr_state.keys())

    # 检查参数名集合是否一致
    if names_ref != names_curr:
        print("参数名不一致!")
        print("参考模型多的：", names_ref - names_curr)
        print("当前模型多的：", names_curr - names_ref)
        return False

    # 逐一比较参数内容
    all_match = True
    for name in names_ref:
        tensor_ref = ref_state[name]
        tensor_curr = curr_state[name]
        if not torch.equal(tensor_ref, tensor_curr):
            all_match = False
            print(f"参数 {name} 不一致！最大差值: {(tensor_ref - tensor_curr).abs().max().item()}")
        elif verbose:
            print(f"参数 {name} 一致")
    return all_match

def main():
    pt_path = "nano-whisper/test/whisper_model.pt"
    model_path = os.path.expanduser("/model/pretrained_models/whisper-large-v3-turbo")
    config = Config(model=model_path)
    dist.init_process_group("nccl", "tcp://localhost:2333", world_size=1, rank=0)
    torch.cuda.set_device(0)
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(config.hf_config.torch_dtype)
    torch.set_default_device("cuda")
    model = WhisperForConditionalGeneration(config.hf_config)
    load_model(model, config.model)
    if(compare_model_with_pt(model, pt_path)):
        print("融合后的模型权重完全匹配！")
    else:
        print("融合后的模型权重不匹配！")

if __name__ == "__main__":
    main()
