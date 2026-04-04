"""
Real-time System Audio Speech-to-Text
Capture âm thanh hệ thống (WASAPI Loopback) → Silero VAD → faster-whisper
"""

from src.transcriber import Transcriber
from src.config import Config

def main():
    print("=" * 50)
    print("  🎙️  System Audio → Speech-to-Text  ")
    print("=" * 50)

    config = Config()
    transcriber = Transcriber(config)
    transcriber.run()

if __name__ == "__main__":
    main()
