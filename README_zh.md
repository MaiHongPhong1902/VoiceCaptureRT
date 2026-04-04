[🇻🇳 Tiếng Việt](README.md) | [🇺🇸 English](README_en.md) | [🇨🇳 中文](README_zh.md)

# 🎙️ VoiceCapture — 实时字幕与翻译系统

VoiceCapture 是一款能够捕获系统音频（通过 Windows 上的 WASAPI 环回）病实时将语音转换为文本 (STT) 的应用程序。
该项目与一个 Web 界面相结合，能够直观、流畅地显示字幕，并允许将字幕自动翻译成多种不同的语言。

## 🌟 主要功能

- **直接内录捕获系统音频:** 依靠 `PyAudioWPatch` 库精确捕获电脑播放的声音（视频、游戏、会议、播客等）。
- **极速与极其准确的语音识别:** 由 `faster-whisper` 模型提供支持。
- **语音活动检测 (VAD):** 整合了 `Silero VAD` 模型功能以消除噪音、智能切分句子并实现超低延迟。
- **自动翻译功能 (Translation):** 直接在屏幕上将文字记录片段实时翻译成各种各样的语言。
- **直接自定义 AI 模型:** 可以在 UI 本地实时在 `faster-whisper` 模型版本（`tiny`、`base`、`small`、`medium`、`large-v3`）之间切换，以优化硬件利用。
- **防溢出缓冲区 (Anti-overlap):** 当说话者不断说出超过设定的最长语音限制 (15s) 时会自动截断一个句子 (Final)，100% 防止文本闪烁和防止 VRAM 显存阻塞。
- **现代美观的 Web UI 仪表板:** 最佳字幕控制面板全面支持实时指标监控，如音频输入振幅（高灵敏度的 VU Meter）、VAD 检测百分比、模型加载延迟和硬件配置。
- **自动硬件回退 (Auto Fallback):** 自动侦测并利用 NVIDIA GPU（CUDA + float16）实现极速运转。整合了针对 MacOS / Windows 终端上的 Unicode 崩溃问题的强力修复程序。如果没有可用的 GPU，系统将顺畅地回退运行到 CPU（int8）端，完全不引发错误。
- **平滑智能储存历史记录:** 所有原文和译文的历史记录都会根据当前进度不断地实时追加到 `transcript.txt` 记录文件中。

## 📂 文件夹结构

```text
Voice/
├── main_ui.py               # 核心主入口点 (带 Web UI 的图形界面)
├── main.py                  # 旧主入口点 (仅有终端调试界面)
├── requirements.txt         # 所需的基本 Python 依赖库
├── transcript.txt           # 包含所有对话日志的默认保存文件
├── static/                  
│   └── index.html           # 前端 Web UI (用于显示字幕及控制面板)
└── src/
    ├── config.py            # 设置管理器与自动检测 CUDA GPU
    ├── transcriber_ws.py    # WebSocket 编排器 (音频流水线、STT、UI推送)
    ├── ws_server.py         # FastAPI WebSocket 服务器 (负责处理实时通信)
    ├── audio_device.py      # 自动扫描并绑定 WASAPI 环回设备 
    ├── audio_utils.py       # 原始音频处理流程和 16kHz 重采样
    ├── vad.py               # 语音活动检测管理 (Silero VAD)
    ├── stt.py               # 调用 faster-whisper 以及防止文字由于错觉不断重复
    ├── translator.py        # 多语言自动翻译处理模块
    └── transcriber.py       # (旧) 纯控制台后端处理程序
```

## ⚙️ 依赖需求与安装指南

因为对 WASAPI 的深度集成，该应用程序在 **Windows** 操作系统可最理想平滑地运行。

1. **安装必要的 Python 库依赖:**
   在根目录打开终端运行：

   ```bash
   pip install -r requirements.txt
   ```

2. **安装 PyTorch (如果您有显卡并打算用 GPU 加速则必须安装):**
   为了在 `faster-whisper` 和 `Silero VAD` 上实现极其流畅、低延迟的操作性能表现，你应当安装具有兼容 CUDA 特性的 PyTorch 版本 (例如：标准 `cu124`)。

   ```bash
   pip uninstall -y torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

   *(如果你的电脑用着其他版本的 CUDA 计算架构，请务必将其指向相匹配的 index url).*

## 🚀 使用指南

只需在根文件夹下运行以下命令：

```bash
python main_ui.py
```

- 当 NLP 语音模型全数加载完成后 (取决于你的硬盘读取速度，一般大约需要 2 至 5 秒左右)，终端面板会发送提示链接，浏览器会自动访问开启 `http://127.0.0.1:8765`。
- **请注意:** 该 UI 界面开始的时候会处于 "聆听中" 的工作状态。你无论开启任意一个 YouTube 的视频、Zoom 会议或者别的音频播放器。字幕马上会在你面前显现。
- 点击语言旁边选项下的下拉菜单可根据您的语言需求将其调换。

## 🛠 高级配置 (`src/config.py`)

根据具体需求灵活配置后端系统的各个方面：

| 参数名 | 默认值 | 意思与说明 |
|---------|----------|--------------------|
| `WHISPER_MODEL` | `"base"` | `faster-whisper` 使用版本 (`tiny`, `base`, `small`, `medium`, `large-v3`) |
| `LANGUAGE` | `"vi"` | STT 的默认原生识别语言。将其设成 `"auto"` 会自动检测语种 |
| `VAD_THRESHOLD` | `0.3` | 对声音探测的敏感度。取值在 `0`->`1` 间, 越低代表即使有强烈的背景噪音也能捕获 |
| `SILENCE_DURATION_S` | `0.8` | 当安静了一段多少秒数的空隙时让系统自行切断结算整装句子一回 (`final`) |
| `INTERIM_INTERVAL_S` | `0.5` | 发送给后台用于进行推断文字并展现中间过程文字状态的音频流发送间隔周期时长，这会直接影响你查阅动态显示的帧率体验 |
