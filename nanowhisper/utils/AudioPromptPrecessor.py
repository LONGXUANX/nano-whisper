import torch
import librosa
import gc
import time
from transformers import AutoTokenizer
from transformers import WhisperFeatureExtractor
from .. import FeatureExtractor, pad_or_trim


class AudioPromptProcessor:
    def __init__(self, model_path, mel_encoder_type="transformers",
                 feature_size=128, sampling_rate=16000, dtype=torch.float16, device="cuda"):
        """
        Initialize the AudioPromptProcessor class
        param model_path: The model path for loading the Tokenizer
        param mel_encoder_type: Mel encoder type (transformers or nanowhisper)
        param feature_size: Feature extractor size
        param sampling_rate: The audio sample rate
        param dtype: The data type of the PyTorch tensor
        param device: Device type ("cuda" or "cpu")
        """
        self.mel_encoder_type = mel_encoder_type
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if mel_encoder_type == "transformers":
            self.audio_feature_extractor = WhisperFeatureExtractor(feature_size=feature_size)
        elif mel_encoder_type == "nanowhisper":
            self.audio_feature_extractor = FeatureExtractor(feature_size=feature_size)
        else:
            raise ValueError("mel_encoder_type must be 'transformers' or 'nanowhisper'")
        self.sampling_rate = sampling_rate
        self.dtype = dtype
        self.device = device

    def process_prompt(self, prompt_data):
        """
        Multi-modal input is processed, and encoded text and audio tensors are returned
        prompt_data: A dictionary containing the prompt text and audio paths
        {
        "prompt": "This is the prompt text ",
        "multi_modal_data": "Audio file path"
        }
        return: A dictionary containing text token_ids and audio feature tensors
        {
        "prompt": token_ids, # token_ids for text
        "multi_modal_data": audio_tensor # audio feature tensor
        }
        """
        # 1. encode text
        token_ids = self.tokenizer.encode(prompt_data["prompt"], add_special_tokens=False)

        # 2. load audio
        audio, sample_rate = librosa.load(prompt_data["multi_modal_data"], sr=self.sampling_rate)

        # 3. feature extraction
        if self.mel_encoder_type == "transformers":
            input_features_tmp = self.audio_feature_extractor(audio, sampling_rate=self.sampling_rate).input_features[0]
        else:
            input_features_tmp = pad_or_trim(self.audio_feature_extractor(audio))

        # 4. to PyTorch Tensor
        audio_tensor = torch.tensor(input_features_tmp, dtype=self.dtype).to(self.device, non_blocking=True)

        # 5. return
        return {
            "prompt": token_ids,  # Encoded token_ids
            "multi_modal_data": audio_tensor  # The processed audio feature tensor
        }
