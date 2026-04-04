"""
Module xử lý VAD (Voice Activity Detection) dùng Silero VAD — GPU-accelerated nếu có
"""

import numpy as np
import torch
from silero_vad import load_silero_vad


class VADProcessor:
    def __init__(self, threshold: float, sample_rate: int = 16000):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\033[93m[INFO] Loading Silero VAD ({self.device})...\033[0m")
        self.model = load_silero_vad()
        if self.device == "cuda":
            self.model = self.model.to("cuda")
        self.threshold = threshold
        self.sample_rate = sample_rate
        print(f"\033[92m[SUCCESS] Silero VAD ready ({self.device})\033[0m")

    def is_speech(self, audio_chunk: np.ndarray) -> tuple[bool, float]:
        """
        Kiểm tra chunk có chứa giọng nói không.
        Returns: (is_speech, probability)
        """
        chunk_tensor = torch.from_numpy(audio_chunk.astype(np.float32))
        if self.device == "cuda":
            chunk_tensor = chunk_tensor.to("cuda")
        with torch.no_grad():
            prob = self.model(chunk_tensor, self.sample_rate).item()
        return prob > self.threshold, prob

    def reset_states(self):
        """Reset internal RNN state of Silero VAD to forget past contexts."""
        if hasattr(self.model, 'reset_states'):
            self.model.reset_states()
