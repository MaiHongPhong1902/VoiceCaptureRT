"""
Main module: connect all components
"""

import threading
import queue
import time
from datetime import datetime

import numpy as np
import pyaudiowpatch as pyaudio

from src.config import Config
from src.audio_device import get_loopback_device
from src.audio_utils import process_raw_bytes
from src.vad import VADProcessor
from src.stt import STTProcessor
from src.transcript_writer import TranscriptWriter


class Transcriber:
    def __init__(self, config: Config):
        self.config = config

        # Tìm loopback device
        self.loopback_device, self.system_sr = get_loopback_device()
        self.channels = self.loopback_device["maxInputChannels"]
        self.chunk_size = int(self.system_sr * config.CHUNK_MS / 1000)
        self.silence_limit = int(config.SILENCE_DURATION_S * 1000 / config.CHUNK_MS)

        # Load models
        self.vad = VADProcessor(config.VAD_THRESHOLD, config.TARGET_SAMPLE_RATE)
        self.stt = STTProcessor(
            config.WHISPER_MODEL,
            config.WHISPER_DEVICE,
            config.WHISPER_COMPUTE,
            config.LANGUAGE,
            config.BEAM_SIZE,
        )

        # Transcript writer
        self.writer = None
        if config.SAVE_TRANSCRIPT:
            self.writer = TranscriptWriter(config.TRANSCRIPT_FILE, config.SHOW_TIMESTAMP)

        # State
        self.audio_buffer: list[np.ndarray] = []   # buffer giọng nói (16kHz)
        self.silence_chunks = 0
        self.audio_queue: queue.Queue = queue.Queue()

        # ── Accumulator: collect resampled audio to 512 samples for VAD ──
        self.vad_accum = np.array([], dtype=np.float32)
        self.VAD_CHUNK = config.VAD_CHUNK_SAMPLES  # = 512

    # ─── TRANSCRIBE WORKER ────────────────────────────
    def _transcribe_worker(self):
        while True:
            audio_data = self.audio_queue.get()
            if audio_data is None:
                break

            audio_np = np.concatenate(audio_data)
            text = self.stt.transcribe(audio_np)

            if text:
                ts = datetime.now().strftime("%H:%M:%S")
                if self.config.SHOW_TIMESTAMP:
                    print(f"[{ts}] 🗣️  {text}")
                else:
                    print(f"🗣️  {text}")

                if self.writer:
                    self.writer.write(text)

            self.audio_queue.task_done()

    # ─── Process VAD chunk 512 samples ───────────────────
    def _process_vad_chunk(self, chunk_512: np.ndarray):
        """Run VAD on chunk 512 samples then update buffer."""
        is_speech, _ = self.vad.is_speech(chunk_512)

        if is_speech:
            self.audio_buffer.append(chunk_512)
            self.silence_chunks = 0
        else:
            if self.audio_buffer:
                self.silence_chunks += 1
                self.audio_buffer.append(chunk_512)  #  padding end of sentence     

                if self.silence_chunks >= self.silence_limit:
                    # End of sentence → send to transcribe
                    self.audio_queue.put(list(self.audio_buffer))
                    self.audio_buffer = []
                    self.silence_chunks = 0

    # ─── AUDIO CALLBACK ───────────────────────────────
    def _audio_callback(self, in_data, frame_count, time_info, status):
        # bytes → float32 mono 16kHz
        chunk_16k = process_raw_bytes(
            in_data, self.channels,
            self.system_sr, self.config.TARGET_SAMPLE_RATE
        )

        # Add to accumulator
        self.vad_accum = np.concatenate([self.vad_accum, chunk_16k])

        # Process each batch correct VAD_CHUNK samples
        while len(self.vad_accum) >= self.VAD_CHUNK:
            chunk_512 = self.vad_accum[:self.VAD_CHUNK]
            self.vad_accum = self.vad_accum[self.VAD_CHUNK:]
            self._process_vad_chunk(chunk_512)

        return (in_data, pyaudio.paContinue)

    # ─── RUN ──────────────────────────────────────────
    def run(self):
        # Start transcribe thread
        worker = threading.Thread(target=self._transcribe_worker, daemon=True)
        worker.start()

        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.system_sr,
            input=True,
            input_device_index=self.loopback_device["index"],
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback,
        )

        print(f"\n waiting for audio ready")
        print(f"   Model  : {self.config.WHISPER_MODEL} | Language: {self.config.LANGUAGE}")
        print(f"   Device : {self.loopback_device['name']}")
        print(f"   SR     : {self.system_sr}Hz → 16000Hz")
        print(f"   VAD    : threshold={self.config.VAD_THRESHOLD} | silence={self.config.SILENCE_DURATION_S}s | chunk={self.VAD_CHUNK}samples")
        print(f"\n  press Ctrl+C to stop.\n")
        print("-" * 50)

        try:
            while stream.is_active():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n stopping")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            self.audio_queue.put(None)
            worker.join(timeout=5)
            print(" stopped")
