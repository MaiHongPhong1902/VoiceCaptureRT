"""
Module lưu transcript ra file
"""

import os
from datetime import datetime


class TranscriptWriter:
    def __init__(self, filepath: str, show_timestamp: bool = True):
        self.filepath = filepath
        self.show_timestamp = show_timestamp
        # Tạo/ghi đè file với header
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# Transcript — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        print(f"📄 Transcript sẽ lưu vào: {os.path.abspath(filepath)}")

    def write(self, text: str):
        if not text:
            return
        if self.show_timestamp:
            ts = datetime.now().strftime("%H:%M:%S")
            line = f"[{ts}] {text}\n"
        else:
            line = f"{text}\n"

        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(line)
