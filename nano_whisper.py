import torch
import os
from nanowhisper import LLM, SamplingParams
from transformers import AutoTokenizer
from nanowhisper import AudioPromptProcessor
import time
import copy

def main():
    requests = 32
    audio_path_30s = "../audio.wav"
    audio_path_16s = "../audio.wav"
    audio_path_6s = "../audio.wav"
    model_path = os.path.expanduser("../whisper-model")
    sampling_params = SamplingParams(temperature=0.0, max_tokens=500)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    audio_prompt_processor = AudioPromptProcessor(
        model_path=model_path,
        # Divided into two types of mel encoders: transformers and nanowhisper
        # Bugs in mel encoder in transformers, bad features in 16s, normal in 30s and 6s
        # nanowhisper encoder works
        mel_encoder_type="transformers",
        feature_size=128,
        sampling_rate=16000,
        dtype=torch.float16,
        device="cuda"
    )
    # eager mode
    # llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)
    # cuda graph mode
    llm = LLM(model_path, enforce_eager=False, tensor_parallel_size=1)

    # warm up
    prompts_warmup = [
        {
            "prompt": "<|startoftranscript|><|zh|><|transcribe|><|notimestamps|>",
            "multi_modal_data": audio_path_30s,
        },
        {
            "prompt": "<|startoftranscript|><|zh|><|transcribe|><|notimestamps|>",
            "multi_modal_data": audio_path_30s,
        }
    ]
    processed_prompts = [audio_prompt_processor.process_prompt(prompt) for prompt in prompts_warmup] * 128
    outputs = llm.generate(processed_prompts, sampling_params,)
    del processed_prompts
    torch.cuda.synchronize()

    # Formal testing
    prompts_new = [
        {
            "prompt": "<|startoftranscript|><|zh|><|transcribe|><|notimestamps|>",
            "multi_modal_data": audio_path_6s,
        },
    ]
    processed_prompts = [audio_prompt_processor.process_prompt(prompt) for prompt in prompts_new]
    # deepcopy
    expanded_prompts = [copy.deepcopy(processed_prompts[0]) for _ in range(requests)]

    # forward
    start_time = time.time()
    outputs = llm.generate(expanded_prompts, sampling_params,)
    end_time = time.time()
    for output in outputs:
        print(f"Completion: {output['text'][:-13]!r}")
    print(f"Time taken for repeated prompts: {end_time - start_time} seconds")
    print(f"Average latency: {1000 * (end_time - start_time) / requests} ms")


if __name__ == "__main__":
    main()
