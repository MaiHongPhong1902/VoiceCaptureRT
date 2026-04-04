[🇻🇳 Tiếng Việt](README.md) | [🇺🇸 English](README_en.md) | [🇨🇳 中文](README_zh.md)

# 🎙️ VoiceCapture — Realtime Subtitles & Translation

VoiceCapture is an application that captures system audio (via WASAPI loopback on Windows) and converts speech to text (Speech-to-Text) in real-time.
The project is integrated with a Web UI that displays subtitles intuitively, smoothly, and allows automatic translation into multiple languages.

## 🌟 Key Features

- **Direct System Audio Capture:** Accurately captures audio playing from your PC (Videos, Games, Meetings, podcasts, etc.) using the `PyAudioWPatch` library.
- **Ultra-fast & Accurate Voice Recognition:** Powered by the `faster-whisper` model.
- **Voice Activity Detection (VAD):** Integrates the `Silero VAD` model to eliminate noise, smartly chunk sentences, and achieve ultra-low latency.
- **Automatic Translation:** Translates transcript segments in real-time into various languages directly on the screen.
- **Direct AI Model Customization:** Ability to switch between `faster-whisper` model versions (`tiny`, `base`, `small`, `medium`, `large-v3`) directly on the UI to optimize hardware usage.
- **Anti-overlap Buffer:** Automatically closes a sentence (Final) when the speaker continues continuously beyond the limit (15s), 100% preventing text flickering and VRAM congestion.
- **Modern Web UI Dashboard:** Optimal Subtitle control panel fully monitoring real-time metrics like Audio Level (responsive VU Meter), VAD probability in %, latency, and hardware configuration.
- **Hardware Auto-Fallback:** Automatically detects and utilizes NVIDIA GPU (CUDA + float16) for peak speeds. Integrates a complete fix for Unicode Crash issues on MacOS / Windows Terminals. If no GPU is available, the system falls back seamlessly to CPU (int8) mode without errors.
- **Smart & Smooth Logging:** The entire history of both original and translated text is continuously appended to a `transcript.txt` log file in real-time.

## 📂 Folder Structure

```text
Voice/
├── main_ui.py               # Main entry point (starts the Web UI)
├── main.py                  # Old entry point (Terminal debug mode only)
├── requirements.txt         # Required base Python libraries
├── transcript.txt           # File containing all conversation logs
├── static/                  
│   └── index.html           # Frontend Web UI (Subtitles and control panel)
└── src/
    ├── config.py            # Settings manager & CUDA GPU auto-detect
    ├── transcriber_ws.py    # WebSocket Orchestrator (audio, STT, UI push)
    ├── ws_server.py         # FastAPI WebSocket Server (realtime bridge)
    ├── audio_device.py      # Automatically scans & binds WASAPI Loopback 
    ├── audio_utils.py       # Raw audio processing & 16kHz resampling
    ├── vad.py               # Voice Activity Detection (Silero VAD)
    ├── stt.py               # Calls faster-whisper & hallucination prevention
    ├── translator.py        # Multi-language translation module
    └── transcriber.py       # Old CLI-only backend handler
```

## ⚙️ Requirements and Installation

The application functions optimally on the **Windows** operating system due to its deep WASAPI integration.

1. **Install essential Python libraries:**
   Open a terminal in the root folder and run:

   ```bash
   pip install -r requirements.txt
   ```

2. **Install PyTorch (Required if using a GPU):**
   To achieve ultra-smooth, low-latency performance on `faster-whisper` and `Silero VAD`, you should install a CUDA-compatible PyTorch version (e.g., `cu124`).

   ```bash
   pip uninstall -y torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

   *(Point to a different index url if your PC uses a different CUDA architecture version).*

## 🚀 Usage Guide

Just run the following command in the root folder:

```bash
python main_ui.py
```

- After the NLP models finish loading (usually 2->5s depending on the hard drive), the Terminal will display a link, and the browser will auto-open `http://127.0.0.1:8765`.
- **Note:** The interface starts in "Listening" state. Open any YouTube Video, Zoom Meeting, or audio player. Subtitles will instantly appear.
- Use the Dropdowns next to "Source" and "Target" to swap languages as needed.

## 🛠 Advanced Configuration (`src/config.py`)

Flexibly assign all aspects of the Backend to your needs:

| Parameter | Default | Description |
|---------|----------|--------------------|
| `WHISPER_MODEL` | `"base"` | `faster-whisper` version (`tiny`, `base`, `small`, `medium`, `large-v3`) |
| `LANGUAGE` | `"vi"` | Default STT language. Set to `"auto"` for automatic detection |
| `VAD_THRESHOLD` | `0.3` | Voice sensitivity ratio. From `0`->`1`, lower makes VAD pick up heavier noise |
| `SILENCE_DURATION_S` | `0.8` | Silence duration (seconds) for the system to automatically close a sentence (`final`) |
| `INTERIM_INTERVAL_S` | `0.5` | The audio stream cycle sent to recognition, affects the FPS of the `pending` text |
