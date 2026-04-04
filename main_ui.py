"""
Entry point với UI mode
"""

import threading
import queue
import webbrowser
import asyncio
import sys
import socket
import subprocess
import time
import os
from urllib.request import urlopen
from urllib.error import URLError

import uvicorn

from src.config import Config
from src.transcriber_ws import TranscriberWS
from src.ws_server import app, set_message_queue, set_transcriber


def _is_port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def _is_ui_alive(url: str) -> bool:
    try:
        with urlopen(url, timeout=1.0) as resp:
            return 200 <= resp.status < 500
    except URLError:
        return False


def _get_port_owner_pid(host: str, port: int) -> int | None:
    try:
        out = subprocess.check_output(
            ["netstat", "-ano", "-p", "tcp"],
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
    except Exception:
        return None

    host_port = f"{host}:{port}"
    any_host_port = f"0.0.0.0:{port}"
    loopback_v6_port = f"[::1]:{port}"
    any_v6_port = f"[::]:{port}"

    for line in out.splitlines():
        row = line.strip()
        if "LISTENING" not in row:
            continue
        if not (
            host_port in row
            or any_host_port in row
            or loopback_v6_port in row
            or any_v6_port in row
        ):
            continue

        parts = row.split()
        if not parts:
            continue
        try:
            return int(parts[-1])
        except Exception:
            continue
    return None


def _kill_pid(pid: int) -> bool:
    if pid <= 0 or pid == os.getpid():
        return False
    try:
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/F", "/T"],
            check=False,
            capture_output=True,
            text=True,
        )
        return True
    except Exception:
        return False


def main():
    host = "127.0.0.1"
    port = 8765
    ui_url = f"http://{host}:{port}"

    print("\033[96m" + "=" * 50)
    print("  [SYSTEM] Audio -> Speech-to-Text [UI]  ")
    print("=" * 50 + "\033[0m")

    # On Windows, selector loop is more stable for websocket disconnect handling.
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Auto-recover from stale/busy port by killing current owner before startup.
    if _is_port_in_use(host, port):
        print(f"\033[93m[WARNING] Port {port} is in use. Auto-releasing...\033[0m")
        owner_pid = _get_port_owner_pid(host, port)
        if owner_pid:
            print(f"\033[96m[SYSTEM] Killing PID {owner_pid}...\033[0m")
            _kill_pid(owner_pid)
        else:
            print("\033[93m[WARNING] PID owner not found via netstat. Waiting for port to release.\033[0m")

        deadline = time.time() + 5.0
        while time.time() < deadline and _is_port_in_use(host, port):
            time.sleep(0.2)

        if _is_port_in_use(host, port):
            print(f"\033[91m[ERROR] Failed to release port {port}. Please try again.\033[0m")
            if _is_ui_alive(ui_url):
                print(f"\033[96m[INFO] Current instance alive at: {ui_url}\033[0m")
                webbrowser.open(ui_url)
            return
        print(f"\033[92m[SUCCESS] Port {port} released.\033[0m")

    config = Config()
    config.CHUNK_MS           = 32    # 32ms → VAD latency tối thiểu

    msg_queue = queue.Queue()
    set_message_queue(msg_queue)

    # Khởi động transcriber trong background thread
    transcriber = TranscriberWS(config, msg_queue)
    set_transcriber(transcriber)       # để ws_server gọi set_language()

    t = threading.Thread(target=transcriber.run, daemon=True)
    t.start()

    # Mở browser sau 1.5 giây
    threading.Timer(1.5, lambda: webbrowser.open(ui_url)).start()

    print(f"\033[92m[READY] UI running at: {ui_url}\033[0m")
    print(f"\033[90m  Press Ctrl+C to stop.\033[0m\n")

    try:
        uvicorn.run(app, host=host, port=port, log_level="error")
    except OSError as e:
        if getattr(e, "errno", None) == 10048:
            print(f"\033[91m[ERROR] Port {port} is occupied after startup. Try again later.\033[0m")
            return
        raise


if __name__ == "__main__":
    main()
