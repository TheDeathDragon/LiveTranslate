# LiveTrans

**English** | [中文](README_zh.md)

Real-time audio translation for Windows. Captures system audio (WASAPI loopback) and optional microphone input, runs ASR, translates via LLM API, and displays results in a transparent overlay.

Works with any system audio — videos, livestreams, voice chat. No player modifications needed.

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![Windows](https://img.shields.io/badge/Platform-Windows-0078d4)
![License](https://img.shields.io/badge/License-MIT-green)

## Screenshot

![LiveTrans](screenshot/en.png)

## Video

[![Install & Demo](https://img.shields.io/badge/Bilibili-Install%20%26%20Demo-00A1D6?logo=bilibili)](https://www.bilibili.com/video/BV1K2Awz6Euw)

## Features

- **Real-time pipeline**: System audio → VAD → ASR → LLM translation → overlay
- **Multiple ASR engines**: faster-whisper, SenseVoice, FunASR Nano, Qwen3-ASR (GGUF)
- **Any OpenAI-compatible API**: DeepSeek, Grok, Qwen, GPT, Ollama, vLLM, etc.
- **Microphone mix-in**: Optionally mix microphone input with system audio for ASR
- **Low-latency VAD**: 32ms chunks + Silero VAD with adaptive silence detection
- **Transparent overlay**: Always-on-top, click-through, draggable, 14 color themes
- **CUDA acceleration**: GPU-accelerated ASR inference
- **Auto model management**: Setup wizard, ModelScope / HuggingFace dual sources
- **Built-in benchmark**: Compare translation model speed and quality

## Roadmap

- **Streamer Mode**: Transparent OBS overlay window that translates the streamer's speech in real-time, with multi-language simultaneous output — usable as on-screen subtitles or a live chat translation panel

## Requirements

- **OS**: Windows 10/11
- **Python**: 3.10+
- **GPU** (recommended): NVIDIA + CUDA 12.6 (Blackwell GPUs like RTX 50xx require CUDA 12.8)
- **Network**: Access to a translation API

## Quick Start

```bash
git clone https://github.com/TheDeathDragon/LiveTranslate.git
cd LiveTranslate
```

Double-click **`install.bat`** — the installer will:
1. Detect Python 3.10+ (auto-install via winget if missing)
2. Create a virtual environment
3. Auto-detect NVIDIA GPU and let you choose CUDA / CPU PyTorch
4. Install all dependencies

Then double-click **`start.bat`** to launch.

To update, double-click **`update.bat`** — it will pull the latest code and update dependencies (auto-installs Git via winget if missing).

<details>
<summary>Manual install</summary>

```bash
python -m venv .venv
.venv\Scripts\activate

# PyTorch (choose one)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126  # CUDA
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128  # CUDA (RTX 50xx)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu    # CPU only

# Dependencies
pip install -r requirements.txt
pip install funasr --no-deps

# Launch
.venv\Scripts\python.exe main.py
```

> FunASR uses `--no-deps` because `editdistance` requires a C++ compiler. `editdistance-s` in `requirements.txt` is a pure-Python drop-in replacement.

</details>

## First Launch

1. Setup wizard appears — choose download source (ModelScope / HuggingFace) and cache path
2. Silero VAD + SenseVoice models download automatically (~1GB)
3. Main UI appears when ready

## Translation API

Settings → Translation tab:

| Parameter | Example |
|-----------|---------|
| API Base | `https://api.deepseek.com/v1` |
| API Key | Your key |
| Model | `deepseek-chat` |
| Proxy | `none` / `system` / custom URL |

## Architecture

```
Audio (WASAPI 32ms) → VAD (Silero) → ASR → LLM Translation → Overlay
         ↑ optional mic mix-in
```

```
main.py                 Entry point & pipeline
├── audio_capture.py    WASAPI loopback + mic mix-in
├── vad_processor.py    Silero VAD
├── asr_engine.py       faster-whisper backend
├── asr_sensevoice.py   SenseVoice backend
├── asr_funasr_nano.py  FunASR Nano backend
├── asr_qwen3.py        Qwen3-ASR backend (ONNX + GGUF)
├── translator.py       OpenAI-compatible client
├── model_manager.py    Model download & cache
├── subtitle_overlay.py PyQt6 overlay
├── control_panel.py    Settings UI
├── dialogs.py          Wizard & download dialogs
└── benchmark.py        Translation benchmark
```

## Acknowledgements

- [CapsWriter-Offline](https://github.com/HaujetZhao/CapsWriter-Offline) — Qwen3-ASR integration reference
- [Qwen3-ASR-GGUF](https://github.com/HaujetZhao/Qwen3-ASR-GGUF) — ONNX + GGUF inference engine
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — GGUF runtime

## Star History

<a href="https://www.star-history.com/?repos=TheDeathDragon%2FLiveTranslate&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/image?repos=TheDeathDragon/LiveTranslate&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/image?repos=TheDeathDragon/LiveTranslate&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/image?repos=TheDeathDragon/LiveTranslate&type=date&legend=top-left" />
 </picture>
</a>

## License

[MIT License](LICENSE)
