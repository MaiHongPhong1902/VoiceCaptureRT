"""
Module STT — Optimized latency + anti-hallucination mạnh — GPU-accelerated
"""

import re
import time
from collections import Counter
import numpy as np
from faster_whisper import WhisperModel


class STTProcessor:
    def __init__(self, model_size: str, device: str, compute_type: str,
                 language: str, beam_size: int):
        self.device = device
        self.compute_type = compute_type
        self.model_size = model_size
        self.language  = language
        self.beam_size = beam_size
        self.last_latency_ms = 0.0
        
        self.load_model(model_size)

    def load_model(self, model_size: str):
        if hasattr(self, 'model') and self.model_size == model_size and getattr(self, '_loaded_compute_type', None) == self.compute_type:
            return
            
        print(f"\033[93m[INFO] Loading faster-whisper [{model_size}] on {self.device} ({self.compute_type})...\033[0m")
        self.model = WhisperModel(model_size, device=self.device, compute_type=self.compute_type)
        self.model_size = model_size
        self._loaded_compute_type = self.compute_type
        print(f"\033[92m[SUCCESS] faster-whisper [{model_size}] ready ({self.device})\033[0m")

    @staticmethod
    def _is_hallucination(text: str) -> bool:
        if not text or len(text.strip()) < 2:
            return True

        words = text.split()

        # 1. loop[word] >= 3 times
        for i in range(len(words) - 2):
            if words[i] == words[i+1] == words[i+2]:
                return True

        return False

    @staticmethod
    def _dedup_sentences(text: str) -> str:
        """Remove repeated sentences, keep the first occurrence."""
        sentences = re.split(r'([.!?。])', text)
        seen = set()
        result = []
        i = 0
        while i < len(sentences):
            s = sentences[i].strip()
            #   Merge punctuation if any
            punct = sentences[i+1] if i+1 < len(sentences) and re.match(r'^[.!?。]$', sentences[i+1]) else ''
            if punct:
                i += 2
            else:
                i += 1

            if not s:
                continue

            key = s.lower().strip()
            if key not in seen:
                seen.add(key)
                result.append(s + punct)

        return ' '.join(result).strip() if result else text

    def transcribe(self, audio: np.ndarray) -> str:
        lang = None if self.language in ("auto", "", None) else self.language
        
        # If auto-detect and restrict_langs is provided, manually find best matching language
        restrict = getattr(self, 'restrict_langs', [])
        if lang is None and len(restrict) > 0:
            try:
                detect_info = self.model.detect_language(audio.astype(np.float32))
                # detect_info is a tuple: (language, top_prob, all_probs_list)
                if len(detect_info) >= 3:
                    probs_dict = dict(detect_info[2])
                    best_lang = None
                    best_prob = -1
                    for r_lang in restrict:
                        p = probs_dict.get(r_lang, 0)
                        if p > best_prob:
                            best_prob = p
                            best_lang = r_lang
                    if best_lang:
                        lang = best_lang
            except Exception as e:
                pass # Fallback to standard auto-detect

        t0 = time.perf_counter()
        segments, info = self.model.transcribe(
            audio.astype(np.float32),
            language=lang,
            beam_size=self.beam_size,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.5,
            compression_ratio_threshold=2.4,
            vad_filter=False,
        )

        text = " ".join(s.text for s in segments).strip()
        self.last_latency_ms = (time.perf_counter() - t0) * 1000

        if self._is_hallucination(text):
            return ""

        # Dedup any remaining repeated sentences
        text = self._dedup_sentences(text)

        return text
