"""
Module audio processing: resample, convert stereo→mono — optimized
"""

import numpy as np
from math import gcd


def stereo_to_mono(audio: np.ndarray, channels: int) -> np.ndarray:
    """Convert stereo (or multi-channel) to mono."""
    if channels > 1:
        return audio.reshape(-1, channels).mean(axis=1)
    return audio.flatten()


def int16_to_float32(audio: np.ndarray) -> np.ndarray:
    """Convert int16 → float32 in range [-1.0, 1.0]."""
    return audio.astype(np.float32) / 32768.0


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio to desired sample rate.
    Use linear interpolation — fast enough for realtime.
    """
    if orig_sr == target_sr:
        return audio
    new_len = int(len(audio) * target_sr / orig_sr)
    return np.interp(
        np.linspace(0, len(audio) - 1, new_len),
        np.arange(len(audio)),
        audio
    ).astype(np.float32)


def compute_rms(audio_f32: np.ndarray) -> float:
    """Compute RMS level (0.0 → 1.0) from float32 audio."""
    if len(audio_f32) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio_f32 ** 2)))


def process_raw_bytes(raw: bytes, channels: int,
                      orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Full pipe line: bytes → int16 → mono → float32 → resample 16kHz
    """
    audio_int16 = np.frombuffer(raw, dtype=np.int16)
    mono = stereo_to_mono(audio_int16, channels)
    f32  = int16_to_float32(mono)
    return resample(f32, orig_sr, target_sr)
