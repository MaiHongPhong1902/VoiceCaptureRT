"""
Module find and manage WASAPI loopback device
"""

import pyaudiowpatch as pyaudio


def get_loopback_device() -> tuple[dict, int]:
    """
    Auto find loopback device of speaker default.
    Returns: (loopback_device_info, sample_rate)
    """
    p = pyaudio.PyAudio()
    try:
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_idx = wasapi_info["defaultOutputDevice"]
        default_speakers = p.get_device_info_by_index(default_idx)
        speaker_name = default_speakers["name"]
        sample_rate = int(default_speakers["defaultSampleRate"])
        print(f"\033[96m[AUDIO] Speaker: {speaker_name}  ({sample_rate} Hz)\033[0m")

        for loopback in p.get_loopback_device_info_generator():
            if speaker_name in loopback["name"]:
                print(f"\033[92m[AUDIO] Loopback: {loopback['name']}\033[0m")
                return loopback, sample_rate

        raise RuntimeError(
            "\033[91m[ERROR] Not found loopback device!\n"
            "   Run: python -m pyaudiowpatch  to see device list.\033[0m"
        )
    finally:
        p.terminate()


def list_all_devices():
    """List all audio device for debug."""
    p = pyaudio.PyAudio()
    print("\033[96m\n [INFO] List all audio device:\033[0m")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        direction = "IN" if info["maxInputChannels"] > 0 else "OUT"
        print(f"  [{i:2d}] [{direction}] {info['name']}")
    p.terminate()
