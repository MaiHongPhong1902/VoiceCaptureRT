"""
Transcriber single-model (base) — chính xác cho cả live + final, tối ưu realtime queue
"""

import re
import threading
import queue
import time
import difflib
from datetime import datetime
from pathlib import Path
import gc
import torch

import numpy as np
import pyaudiowpatch as pyaudio

from src.config import Config
from src.audio_device import get_loopback_device
from src.audio_utils import process_raw_bytes, compute_rms
from src.vad import VADProcessor
from src.stt import STTProcessor
from src.translator import Translator
from src.diarizer import SpeakerDiarizer


class TranscriberWS:
    def __init__(self, config: Config, msg_queue: queue.Queue):
        self.config    = config
        self.msg_queue = msg_queue

        # Loopback device
        self.loopback_device, self.system_sr = get_loopback_device()
        self.channels    = self.loopback_device["maxInputChannels"]
        self.chunk_size  = int(self.system_sr * config.CHUNK_MS / 1000)
        self.silence_limit = int(config.SILENCE_DURATION_S * 1000 / config.CHUNK_MS)

        # Single model — base cho cả interim + final
        self.vad = VADProcessor(config.VAD_THRESHOLD, config.TARGET_SAMPLE_RATE)
        self.stt = STTProcessor(
            config.WHISPER_MODEL, config.WHISPER_DEVICE,
            config.WHISPER_COMPUTE, config.LANGUAGE, config.BEAM_SIZE,
        )
        self.translator = Translator()
        self.diarizer = SpeakerDiarizer()

        # Language
        self.source_lang = config.LANGUAGE or "vi"
        self.target_lang = ""

        # Audio state
        self.audio_buffer:   list[np.ndarray] = []
        self.silence_chunks: int  = 0
        self.vad_active:     bool = False

        # VAD accumulator
        self.vad_accum = np.array([], dtype=np.float32)
        self.VAD_CHUNK = config.VAD_CHUNK_SAMPLES

        # Timing
        self.last_interim_time = 0.0
        self.last_level_time   = 0.0   # throttle audio level sends

        # Queues
        self.task_queue: queue.Queue = queue.Queue()
        self.ui_queue:   queue.Queue = queue.Queue()

        # Limits
        self.INTERIM_WINDOW = int(30.0 * config.TARGET_SAMPLE_RATE) # Tối đa giữ 30s
        self.MAX_SPEECH_S   = 15.0

        # Auto-save
        self.transcript_path = Path(config.TRANSCRIPT_FILE)
        self.save_transcript_enabled = config.SAVE_TRANSCRIPT
        self.log_terminal_enabled = True
        if self.save_transcript_enabled:
            self._init_transcript_file()

    def _init_transcript_file(self):
        self.transcript_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.transcript_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Session: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*60}\n\n")
        print(f"\033[96m[INFO] Transcript -> {self.transcript_path}\033[0m")

    def set_save_transcript(self, enabled: bool):
        if getattr(self, 'save_transcript_enabled', None) == enabled: return
        self.save_transcript_enabled = enabled
        status = 'ON' if enabled else 'OFF'
        if self.log_terminal_enabled:
            print(f"\033[93m[INFO] Save Transcript -> {status}\033[0m")

    def set_log_terminal(self, enabled: bool):
        if getattr(self, 'log_terminal_enabled', None) == enabled: return
        self.log_terminal_enabled = enabled
        status = 'ON' if enabled else 'OFF'
        if self.log_terminal_enabled:
            print(f"\033[96m[INFO] Terminal Logs -> {status}\033[0m")

    def set_diarization(self, enabled: bool):
        if getattr(self.diarizer, 'enabled', None) == enabled: return
        self.diarizer.set_enabled(enabled)
        status = 'ON' if enabled else 'OFF'
        if self.log_terminal_enabled:
            print(f"\033[93m[INFO] Diarization -> {status}\033[0m")

    def _save_line(self, ts, text, translated="", speaker=""):
        if not self.save_transcript_enabled:
            return
        try:
            with open(self.transcript_path, "a", encoding="utf-8") as f:
                prefix = f"[{ts}] [{speaker}] " if speaker else f"[{ts}] "
                f.write(f"{prefix}{text}\n")
                if translated:
                    f.write(f"{' ' * len(prefix)}→ {translated}\n")
        except Exception:
            pass

    def _push(self, msg: dict):
        self.msg_queue.put(msg)

    def _translate(self, text: str) -> str:
        if not text or not self.target_lang or self.source_lang == self.target_lang:
            return ""
        return self.translator.translate(text, self.source_lang, self.target_lang)

    @staticmethod
    def _split_sentences(text: str):
        match = None
        for m in re.finditer(r'[.!?。]\s*', text):
            match = m
        if match:
            pos = match.end()
            return text[:pos].strip(), text[pos:].strip()
        return "", text.strip()

    # ─── STT WORKER ───────────────────────────────────
    def _extract_unprinted(self, full_text: str, printed_text: str) -> str:
        if not printed_text:
            return full_text
            
        def clean_word(w):
            return re.sub(r'[^\w\s]', '', w).lower()
            
        f_words = full_text.split()
        p_words = printed_text.split()
        
        f_clean = [clean_word(w) for w in f_words]
        p_clean = [clean_word(w) for w in p_words]
        
        s = difflib.SequenceMatcher(None, p_clean, f_clean)
        match_end = 0
        for block in s.get_matching_blocks():
            if block.size > 0:
                end_in_f = block.b + block.size
                if end_in_f > match_end:
                    match_end = end_in_f
                    
        return " ".join(f_words[match_end:]).strip()

    def _stt_worker(self):
        prev_interim = ""
        printed_text = ""

        while True:
            task = self.task_queue.get()
            if task is None:
                self.ui_queue.put(None)
                break

            task_type, audio_data = task
            audio_np = np.concatenate(audio_data).astype(np.float32)

            if task_type == "interim":
                while not self.task_queue.empty():
                    next_task = self.task_queue.queue[0]
                    if next_task[0] == "interim":
                        _, next_audio = self.task_queue.get_nowait()
                        audio_np = np.concatenate(next_audio).astype(np.float32)
                        self.task_queue.task_done()
                    else:
                        break

                if len(audio_np) > self.INTERIM_WINDOW:
                    audio_np = audio_np[-self.INTERIM_WINDOW:]

                text = self.stt.transcribe(audio_np)
                latency = self.stt.last_latency_ms
                unprinted = self._extract_unprinted(text, printed_text)

                if unprinted != prev_interim:
                    speaker = self.diarizer.identify_speaker(audio_np) if self.diarizer.enabled else ""
                    self.ui_queue.put(("interim", unprinted, "", speaker))
                    prev_interim = unprinted

                # Always send latency metric
                self._push({"type": "stt_latency", "ms": round(latency, 1), "mode": "interim"})

            elif task_type == "final":
                self._push({"type": "status", "state": "processing"})
                text = self.stt.transcribe(audio_np)
                latency = self.stt.last_latency_ms
                
                unprinted = self._extract_unprinted(text, printed_text)
                self.ui_queue.put(("interim", "", "", ""))
                
                if unprinted:
                    ts = datetime.now().strftime("%H:%M:%S")
                    speaker = self.diarizer.identify_speaker(audio_np) if self.diarizer.enabled else ""
                    self.ui_queue.put(("final", unprinted, ts, speaker))
                else:
                    self._push({"type": "status", "state": "listening"})
                
                self._push({"type": "stt_latency", "ms": round(latency, 1), "mode": "final"})
                prev_interim = ""
                printed_text = ""
                
                # Partially release RAM/VRAM after a full sentence to prevent memory leaks
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            self.task_queue.task_done()

    # ─── UI / TRANSLATE WORKER ────────────────────────
    def _ui_worker(self):
        prev_confirmed = ""
        while True:
            task = self.ui_queue.get()
            if task is None:
                break

            task_type, text, ts, speaker = task

            if task_type == "interim":
                # Drain extra interim UI tasks to avoid blocking on Translate API
                while not self.ui_queue.empty():
                    next_task = self.ui_queue.queue[0]
                    if next_task[0] == "interim":
                        _, text, _, _ = self.ui_queue.get_nowait()
                        self.ui_queue.task_done()
                    else:
                        break

                if not text:
                    self._push({"type": "interim", "confirmed": "", "pending": "", "text": "", "translated": "", "speaker": ""})
                else:
                    confirmed, pending = self._split_sentences(text)
                    if confirmed and len(confirmed) >= len(prev_confirmed):
                        prev_confirmed = confirmed
                    confirmed = prev_confirmed

                    full = (confirmed + " " + pending).strip() if confirmed else pending
                    translated = self._translate(full)
                    self._push({
                        "type": "interim", "confirmed": confirmed,
                        "pending": pending, "text": full, "translated": translated,
                        "speaker": speaker
                    })
            elif task_type == "final":
                translated = self._translate(text)
                try:
                    if self.log_terminal_enabled:
                        spk_fmt = f"[{speaker}] " if speaker else ""
                        print(f"\033[96m[{ts}] {spk_fmt}[SOURCE] {text}\033[0m")
                        if translated:
                            print(f"\033[95m{' ' * 10}[TARGET] {translated}\033[0m")
                except Exception:
                    pass
                self._push({"type": "transcript", "text": text,
                             "translated": translated, "timestamp": ts, "speaker": speaker})
                self._save_line(ts, text, translated, speaker)
                self._push({"type": "status", "state": "listening"})
                prev_confirmed = ""

            self.ui_queue.task_done()


    def _process_vad_chunk(self, chunk_512: np.ndarray):
        is_speech, prob = self.vad.is_speech(chunk_512)

        # Send audio level + VAD prob (throttle to ~30fps)
        now_level = time.time()
        if now_level - self.last_level_time >= 0.033:
            rms = compute_rms(chunk_512)
            self._push({"type": "audio_level", "rms": round(rms, 4), "vad_prob": round(prob, 3)})
            self.last_level_time = now_level

        if is_speech and not self.vad_active:
            self.vad_active        = True
            self.last_interim_time = time.time()
            self._speech_start     = time.time()
            self._push({"type": "vad", "active": True})
        elif not is_speech and self.vad_active and not self.audio_buffer:
            self.vad_active = False
            self._push({"type": "vad", "active": False})
            self.vad.reset_states()

        if is_speech:
            self.audio_buffer.append(chunk_512)
            self.silence_chunks = 0

            now       = time.time()
            audio_len = len(self.audio_buffer) * self.VAD_CHUNK / self.config.TARGET_SAMPLE_RATE

            if hasattr(self, '_speech_start') and now - self._speech_start >= self.MAX_SPEECH_S:
                self.task_queue.put(("final", list(self.audio_buffer)))
                self.audio_buffer      = []
                self.silence_chunks    = 0
                self._speech_start     = now
                self.last_interim_time = now
                self.vad.reset_states()
                return

            if (now - self.last_interim_time >= self.config.INTERIM_INTERVAL_S
                    and audio_len >= self.config.INTERIM_MIN_S):
                self.task_queue.put(("interim", list(self.audio_buffer)))
                self.last_interim_time = now
        else:
            if self.audio_buffer:
                self.silence_chunks += 1
                self.audio_buffer.append(chunk_512)
                if self.silence_chunks >= self.silence_limit:
                    self.task_queue.put(("final", list(self.audio_buffer)))
                    self._push({"type": "vad", "active": False})
                    self.audio_buffer   = []
                    self.silence_chunks = 0
                    self.vad_active     = False
                    self.vad.reset_states()

    def _audio_callback(self, in_data, frame_count, time_info, status):
        chunk_16k = process_raw_bytes(
            in_data, self.channels,
            self.system_sr, self.config.TARGET_SAMPLE_RATE
        )
        # Append to accumulator and process all available VAD chunks
        self.vad_accum = np.concatenate([self.vad_accum, chunk_16k])
        while len(self.vad_accum) >= self.VAD_CHUNK:
            chunk_512      = self.vad_accum[:self.VAD_CHUNK]
            self.vad_accum = self.vad_accum[self.VAD_CHUNK:]
            self._process_vad_chunk(chunk_512)
        return (in_data, pyaudio.paContinue)

    def set_source_lang(self, lang: str):
        self.source_lang = lang
        
        # If language is a specialized restrict auto-detect mode
        if lang.startswith("auto_"):
            langs_part = lang.split("_")[1:] # ['en', 'vi']
            self.stt.language = None
            self.stt.restrict_langs = langs_part
        else:
            self.stt.language = None if lang == "auto" else lang
            self.stt.restrict_langs = []
            
        if self.log_terminal_enabled:
            print(f"\033[96m[SETTING] Source -> {lang}\033[0m")

    def set_target_lang(self, lang: str):
        self.target_lang = lang if lang != "none" else ""
        if self.log_terminal_enabled:
            print(f"\033[95m[SETTING] Target -> {lang if lang != 'none' else 'OFF'}\033[0m")

    def set_model(self, model_size: str):
        self.config.WHISPER_MODEL = model_size
        self.stt.load_model(model_size)
        self._push({
            "type": "info", "device": self.loopback_device["name"],
            "model": model_size, "source": self.source_lang,
            "target": self.target_lang or "none", "sr": self.system_sr,
            "compute_device": self.config.WHISPER_DEVICE,
            "compute_type": self.stt.compute_type,
        })

    def set_model_config(self, compute_type: str, beam_size: int):
        need_reload = self.stt.compute_type != compute_type
        self.config.WHISPER_COMPUTE = compute_type
        self.config.BEAM_SIZE = beam_size
        self.stt.beam_size = beam_size
        self.stt.compute_type = compute_type
        if need_reload:
            self.stt.load_model(self.config.WHISPER_MODEL)
            if self.log_terminal_enabled:
                print(f"\033[93m[SETTING] Model Config -> compute: {compute_type}, beam: {beam_size}\033[0m")
        self._push({
            "type": "info", "device": self.loopback_device["name"],
            "model": self.config.WHISPER_MODEL, "source": self.source_lang,
            "target": self.target_lang or "none", "sr": self.system_sr,
            "compute_device": self.config.WHISPER_DEVICE,
            "compute_type": self.stt.compute_type,
        })

    def run(self):
        t_stt = threading.Thread(target=self._stt_worker, daemon=True)
        t_ui  = threading.Thread(target=self._ui_worker,  daemon=True)
        t_stt.start()
        t_ui.start()

        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16, channels=self.channels,
            rate=self.system_sr, input=True,
            input_device_index=self.loopback_device["index"],
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback,
        )

        print(f"\033[93m[AUDIO] [{self.loopback_device['name']}]\033[0m")
        print(f"\033[90m[INFO]  Model: {self.config.WHISPER_MODEL} | Interval: {self.config.INTERIM_INTERVAL_S}s | Silence: {self.config.SILENCE_DURATION_S}s\033[0m")
        self._push({"type": "status", "state": "listening"})
        self._push({
            "type": "info", "device": self.loopback_device["name"],
            "model": self.config.WHISPER_MODEL, "source": self.source_lang,
            "target": self.target_lang or "none", "sr": self.system_sr,
            "compute_device": self.config.WHISPER_DEVICE,
            "compute_type": self.config.WHISPER_COMPUTE,
        })

        try:
            while stream.is_active():
                time.sleep(0.1)
        except Exception:
            pass
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            self.task_queue.put(None)
            t_stt.join(timeout=3)
            t_ui.join(timeout=3)
