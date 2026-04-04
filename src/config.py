"""
Project configuration — auto detect GPU
"""

import torch


def _detect_device() -> tuple[str, str]:
    """Auto-detect GPU. Returns (device, compute_type)."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"\033[92m[SUCCESS] GPU detected: {name}\033[0m")
        return "cuda", "float16"
    print("\033[93m[WARNING] No CUDA GPU — using CPU (int8)\033[0m")
    return "cpu", "int8"


_device, _compute = _detect_device()


class Config:
    # ── Audio ──────────────────────────────────────────
    TARGET_SAMPLE_RATE: int   = 16000   # Hz — Silero VAD & Whisper config
    CHUNK_MS: int             = 64      # ms per chunk audio from system
    VAD_CHUNK_SAMPLES: int    = 512     # Silero VAD require 512 / 1024 / 1536 samples

    # ── VAD (Voice Activity Detection) ─────────────────
    VAD_THRESHOLD: float      = 0.15    # 0.0→1.0 | lower = more sensitive
    SILENCE_DURATION_S: float = 0.8     # seconds of silence to cut off sentence

    # ── Streaming / Interim ──────────────────────────
    INTERIM_INTERVAL_S: float = 0.5    # reduce from 0.6 → 0.5 for smoother realtime
    INTERIM_MIN_S: float      = 0.3    # need at least N seconds of audio before sending
    INTERIM_WINDOW_S: float   = 1.5    # only transcribe N seconds NEAREST → always fast

    # ── STT (Speech-to-Text) ───────────────────────────
    WHISPER_MODEL: str        = "base"  # tiny | base | small | medium | large-v3
    WHISPER_DEVICE: str       = _device   # auto: cuda if GPU, fallback cpu
    WHISPER_COMPUTE: str      = _compute  # auto: float16 on GPU, int8 on CPU
    LANGUAGE: str             = "vi"    # vi | en | None (auto-detect)
    BEAM_SIZE: int            = 1       # smaller = faster

    # ── Output ─────────────────────────────────────────
    SAVE_TRANSCRIPT: bool     = True    # save transcript to file
    TRANSCRIPT_FILE: str      = "transcript.txt"
    SHOW_TIMESTAMP: bool      = True    # show timestamp each sentence
