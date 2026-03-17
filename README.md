# LiveTrans

**English** | [中文](README_zh.md)

Real-time audio translation tool for Windows. Captures system audio via WASAPI loopback, runs speech recognition (ASR), translates through LLM APIs, and displays results in a transparent overlay window.

Perfect for watching foreign-language videos, livestreams, and meetings — no player modifications needed, works with any system audio.

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![Windows](https://img.shields.io/badge/Platform-Windows-0078d4)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

- **Real-time translation**: System audio → ASR → LLM translation → subtitle overlay, fully automatic
- **Multiple ASR engines**: faster-whisper, FunASR SenseVoice (optimized for Japanese), FunASR Nano, Qwen3-ASR (GGUF)
- **Flexible translation backend**: Compatible with any OpenAI-format API (DeepSeek, Grok, Qwen, GPT, etc.)
- **Low-latency VAD**: 32ms audio chunks + Silero VAD with adaptive silence detection
- **Transparent overlay**: Always-on-top, click-through, draggable — doesn't interfere with your workflow
- **CUDA acceleration**: GPU-accelerated ASR inference
- **Automatic model management**: First-launch setup wizard, supports ModelScope / HuggingFace dual sources
- **Translation benchmark**: Built-in benchmark tool for comparing model performance

## Screenshots

**English → Chinese** (Twitch livestream)

![English to Chinese](screenshot/en-to-cn.png)

**Japanese → Chinese** (Japanese livestream)

![Japanese to Chinese](screenshot/jp-to-cn.png)

## Requirements

- **OS**: Windows 10/11
- **Python**: 3.10+
- **GPU** (recommended): NVIDIA GPU with CUDA 12.6 (for ASR acceleration)
- **Network**: Access to a translation API (DeepSeek, OpenAI, etc.)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/TheDeathDragon/LiveTranslate.git
cd LiveTranslate
```

### 2. Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install PyTorch (with CUDA)

Choose the install command based on your CUDA version. See [PyTorch official site](https://pytorch.org/get-started/locally/):

```bash
# CUDA 12.6 (recommended)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126

# CPU only (no NVIDIA GPU)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
pip install funasr --no-deps
```

> **Note**: FunASR is installed with `--no-deps` because its dependency `editdistance` requires a C++ compiler. The pure-Python alternative `editdistance-s` is included in `requirements.txt` as a drop-in replacement.

### 5. Launch

```bash
.venv\Scripts\python.exe main.py
```

Or double-click `start.bat`.

## First Launch

1. A **setup wizard** will appear on first launch — choose your model download source (ModelScope for China, HuggingFace for international) and model cache path
2. Silero VAD and SenseVoice ASR models will be downloaded automatically (~1GB)
3. The main UI appears once downloads complete

## Configuring the Translation API

Click **Settings** on the overlay → **Translation** tab:

| Parameter | Description |
|-----------|-------------|
| API Base | API endpoint, e.g. `https://api.deepseek.com/v1` |
| API Key | Your API key |
| Model | Model name, e.g. `deepseek-chat` |
| Proxy | `none` (direct) / `system` (system proxy) / custom proxy URL |

Works with any OpenAI-compatible API, including:
- [DeepSeek](https://platform.deepseek.com/)
- [xAI Grok](https://console.x.ai/)
- [Alibaba Qwen](https://dashscope.aliyuncs.com/)
- [OpenAI GPT](https://platform.openai.com/)
- Self-hosted [Ollama](https://ollama.ai/), [vLLM](https://github.com/vllm-project/vllm), etc.

## Usage

1. Play a video or livestream with foreign-language audio
2. Launch LiveTrans — the overlay appears automatically
3. Recognized text and translations are displayed in real time

### Overlay Controls

- **Pause/Resume**: Pause or resume translation
- **Clear**: Clear current subtitles
- **Click-through**: Mouse clicks pass through the subtitle window
- **Always on top**: Keep overlay above all windows
- **Auto-scroll**: Automatically scroll to the latest subtitle
- **Model selector**: Switch between configured translation models
- **Target language**: Change the translation target language

### Settings Panel

Open via the **Settings** button on the overlay or the system tray menu:

- **VAD/ASR**: ASR engine selection, VAD mode, sensitivity parameters
- **Translation**: API configuration, system prompt, multi-model management
- **Benchmark**: Translation speed and quality benchmarks
- **Cache**: Model cache path management

## Architecture

```
Audio (WASAPI 32ms) → VAD (Silero) → ASR (Whisper/SenseVoice/Nano/Qwen3) → LLM Translation → Overlay
```

```
main.py                 Entry point & pipeline orchestration
├── audio_capture.py    WASAPI loopback audio capture
├── vad_processor.py    Silero VAD speech detection
├── asr_engine.py       faster-whisper ASR backend
├── asr_sensevoice.py   FunASR SenseVoice backend
├── asr_funasr_nano.py  FunASR Nano backend
├── asr_qwen3.py        Qwen3-ASR backend (ONNX + GGUF)
├── qwen_asr_gguf/      Qwen3-ASR inference engine
├── translator.py       OpenAI-compatible translation client
├── model_manager.py    Model detection, download & cache management
├── subtitle_overlay.py PyQt6 transparent overlay window
├── control_panel.py    Settings panel UI
├── dialogs.py          Setup wizard & model download dialogs
├── log_window.py       Real-time log viewer
├── benchmark.py        Translation benchmark
└── config.yaml         Default configuration
```

## Known Limitations

- Windows only (depends on WASAPI loopback)
- ASR model first load takes a few seconds (GPU) to tens of seconds (CPU)
- Translation quality depends on the LLM API used
- Recognition degrades in noisy environments or with overlapping speakers

## Acknowledgements

- [CapsWriter-Offline](https://github.com/HaujetZhao/CapsWriter-Offline) — Qwen3-ASR integration architecture and hotword system reference
- [Qwen3-ASR-GGUF](https://github.com/HaujetZhao/Qwen3-ASR-GGUF) — ONNX + GGUF hybrid inference engine for Qwen3-ASR
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — GGUF model inference runtime

## License

[MIT License](LICENSE)
