[🇻🇳 Tiếng Việt](README.md) | [🇺🇸 English](README_en.md) | [🇨🇳 中文](README_zh.md)

# 🎙️ VoiceCapture — Realtime Subtitles & Translation

VoiceCapture là ứng dụng thu âm thanh hệ thống (thông qua WASAPI loopback trên Windows) và chuyển đổi giọng nói thành văn bản (Speech-to-Text) theo thời gian thực.
Dự án được kết hợp với một giao diện Web UI hiển thị Subtitle trực quan, mượt mà và cho phép dịch tự động sang nhiều ngôn ngữ khác nhau.

## 🌟 Tính năng nổi bật

- **Thu âm hệ thống trực tiếp:** Bắt chính xác âm thanh đang phát ra từ PC (Video, Game, Meeting, podcast, v.v) bằng thư viện `PyAudioWPatch`.
- **Nhận diện giọng nói siêu tốc & chính xác:** Sử dụng mô hình `faster-whisper`.
- **Phát hiện giọng nói phân vùng (VAD):** Tích hợp mô hình `Silero VAD` giúp loại bỏ tiếng ồn, chia câu thông minh, tạo ra độ trễ (latency) siêu thấp.
- **Tự động dịch thuật (Translation):** Dịch đoạn transcript real-time sang nhiều ngôn ngữ khác nhau ngay trên màn hình.
- **Tuỳ biến Model AI trực tiếp:** Có khả năng chuyển đổi trực tiếp trên UI giữa các phiên bản model `faster-whisper` (`tiny`, `base`, `small`, `medium`, `large-v3`) để tối ưu theo phần cứng.
- **Chống tràn bộ đệm (Anti-overlap):** Tự động ngắt chốt câu (Final) khi nhận thấy người nói liên tục kéo dài quá giới hạn (15s), ngăn chặn 100% tình trạng "nhấp nháy chữ" và nghẽn VRAM.
- **Giao diện Web UI hiện đại:** Bảng điều khiển Subtitle tối ưu, theo dõi đầy đủ chỉ số real-time như Audio Level (VU Meter nhạy bén), VAD probablity tính bằng %, độ trễ, cấu hình hardware.
- **Tự động nhận diện phần cứng (Auto Fallback):** Tự động detect và sử dụng GPU NVIDIA (CUDA + float16) để đạt tốc độ tốt nhất. Tích hợp Fix triệt để lỗi Unicode Crash trên MacOS / Windows Terminal. Mất cắm GPU, hệ thống tự động fallback chạy trên CPU (int8) mà không gây lỗi.
- **Lưu file thông minh & mượt mà:** Toàn bộ lịch sử nội dung bản nguồn lẫn bản dịch được append liên tục theo tiến trình vào file `transcript.txt` theo thời gian thực.

## 📂 Cấu trúc thư mục

```text
Voice/
├── main_ui.py               # Entry point chính (có giao diện Web UI)
├── main.py                  # Entry point cũ (chỉ chạy dạng Terminal debug)
├── requirements.txt         # Các thư viện Python gốc cần thiết
├── transcript.txt           # File chứa toàn bộ log nội dung cuộc trò chuyện
├── static/                  
│   └── index.html           # Frontend Web UI (hiển thị Subtitle, control panel)
└── src/
    ├── config.py            # Quản lý thiết lập & auto-detect CUDA GPU
    ├── transcriber_ws.py    # Orchestrator cho WebSocket (luồng âm thanh, STT, đẩy ra UI)
    ├── ws_server.py         # Server FastAPI WebSocket (giao tiếp real-time với frontend)
    ├── audio_device.py      # Tự động quét và móc hệ thống WASAPI Loopback 
    ├── audio_utils.py       # Xử lý audio thô, dải tần, resample (16kHz)
    ├── vad.py               # Quản lý nhận diện vùng chứa giọng nói (Silero VAD)
    ├── stt.py               # Module gọi faster-whisper và chống ảo giác (Anti-hallucination)
    ├── translator.py        # Module hỗ trợ dịch thuật tự động đa ngôn ngữ
    └── transcriber.py       # Trình xử lý backend thuần CLI (cũ)
```

## ⚙️ Yêu cầu và Cài đặt

Ứng dụng hoạt động tốt nhất trên hệ điều hành **Windows** do khai thác sâu vào WASAPI.

1. **Cài đặt thư viện Python thiết yếu:**
   Mở terminal tại thư mục gốc và chạy:

   ```bash
   pip install -r requirements.txt
   ```

2. **Cài đặt PyTorch (Bắt buộc nếu muốn dùng GPU):**
   Để ứng dụng chạy siêu mượt với độ trễ thấp trên thư viện `faster-whisper` và `Silero VAD`, bạn nên cài bản PyTorch tương thích CUDA (Ví dụ: chuẩn `cu124`).

   ```bash
   pip uninstall -y torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

   *(Trỏ index url khác tương ứng phiên bản CUDA máy tính bạn nếu dùng cấu trúc card khác).*

## 🚀 Hướng dẫn sử dụng

Chỉ cần chạy lệnh sau tại thư mục gốc:

```bash
python main_ui.py
```

- Sau khi các mô hình NLP tải hoàn tất (tầm 2->5s tuỳ ổ cứng), Terminal sẽ thông báo link, và trình duyệt tự động mở đến `http://127.0.0.1:8765`.
- **Lưu ý:** Giao diện bắt đầu hiện trạng thái "Listening". Bạn mở bất kỳ một Video YouTube, Zoom Meeting, hay trình phát âm thanh ra. Subtitle sẽ lập tức xuất hiện.
- Ấn Dropdown bên cạnh chữ Transcript và Dịch sang để chuyển đổi tuỳ ý.

## 🛠 Cấu hình nâng cao (`src/config.py`)

Chỉ định linh hoạt mọi mặt của Backend theo nhu cầu:

| Tham số | Mặc định | Mô tả ý nghĩa xử lý |
|---------|----------|--------------------|
| `WHISPER_MODEL` | `"base"` | Phiên bản `faster-whisper` (`tiny`, `base`, `small`, `medium`, `large-v3`) |
| `LANGUAGE` | `"vi"` | Ngôn ngữ mặc định STT. Bạn có thể set là `"auto"` để tự động dò |
| `VAD_THRESHOLD` | `0.3` | Tỉ lệ nhạy giọng nói. Từ `0`->`1`, để càng thấp VAD sẽ càng bắt tiếng ồn mạnh |
| `SILENCE_DURATION_S` | `0.8` | Khoảng im lặng (tính bằng giây) để hệ thống tự động ngắt chốt một câu nói (`final`) |
| `INTERIM_INTERVAL_S` | `0.5` | Chu kỳ stream audio sang nhận diện ảnh hưởng fps hiển thị text dạng `pending` |
