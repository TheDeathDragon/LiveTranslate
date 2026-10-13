# LiveTranslate

**English** | [中文](README_zh.md)

Real-time audio translation for Windows. Captures system audio (WASAPI loopback) and optional microphone input, runs ASR, translates via LLM API, and displays results in a transparent overlay.

Works with any system audio — videos, livestreams, voice chat. No player modifications needed.

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![Windows](https://img.shields.io/badge/Platform-Windows-0078d4)
![License](https://img.shields.io/badge/License-MIT-green)

## Screenshot

![LiveTranslate](screenshot/en.png)

## Video

[![Install & Demo](https://img.shields.io/badge/Bilibili-Install%20%26%20Demo-00A1D6?logo=bilibili)](https://www.bilibili.com/video/BV1K2Awz6Euw)

## Features

- **Real-time pipeline**: System audio → VAD → ASR → LLM translation → overlay
- **Multiple ASR engines**: faster-whisper, SenseVoice, FunASR Nano, Anime-Whisper
- **Remote ASR**: offload speech recognition to a GPU machine over HTTP — see [REMOTE_ASR.md](REMOTE_ASR.md)
- **Any OpenAI-compatible API**: DeepSeek, Grok, Qwen, GPT, Ollama, vLLM, etc.
- **Streaming translation display**: Real-time character-by-character translation output
- **Per-model settings**: Streaming, structured output (JSON), context history, disable thinking
- **Microphone mix-in**: Optionally mix microphone input with system audio for ASR
- **Browser-only audio**: Chrome or Sogou application capture, excluding other apps and microphone input
- **Low-latency VAD**: 32ms chunks + Silero VAD with adaptive silence detection
- **Transparent overlay**: Always-on-top, click-through, draggable, 14 color themes
- **CUDA acceleration**: GPU-accelerated ASR inference
- **Auto model management**: Setup wizard, ModelScope / HuggingFace dual sources
- **Built-in benchmark**: Compare translation model speed and quality

## Changelog

See [English Changelog](i18n/CHANGELOG_en.md) | [中文更新日志](i18n/CHANGELOG_zh.md)

## Requirements

- **OS**: Windows 10/11
- **Python**: 3.10–3.12 (or use the portable build)
- **GPU** (recommended): NVIDIA + CUDA 12.6 (Blackwell GPUs like RTX 50xx require CUDA 12.8)
- **Network**: Access to a translation API

## Quick Start

### Portable build (no Python required, recommended for non-developers)

Download `LiveTranslate-portable-*.zip` from [Releases](https://github.com/TheDeathDragon/LiveTranslate/releases), unzip, and double-click **`start.bat`**. The first run auto-downloads a portable Python 3.12 and installs GPU-aware dependencies — no Python installation needed.

### From source

```bash
git clone https://github.com/TheDeathDragon/LiveTranslate.git
cd LiveTranslate
```

Double-click **`install.bat`** — the installer will:
1. Detect Python 3.10–3.12 (auto-install via winget if missing)
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

# Launch
.venv\Scripts\python.exe main.py
```

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

## Browser-only audio

In Settings → VAD / ASR → Audio, select **Chrome audio only** or **Sogou browser audio only** to watch a stream while using a separate voice-chat app. These modes disable microphone capture and never fall back to system audio. The existing system/default-device and microphone modes remain available.

The capture target is the browser's actual Windows audio-session process (including Sogou audio children), with its process tree. A paused session remains selectable; closing and reopening the browser reconnects automatically. If more than one browser audio process is active, capture stops with a log message: close or pause the other instance, then restart capture. This does not isolate one tab or window; tabs sharing the selected audio process are included. Voice chat in the same browser may therefore also be included.

Requires Windows application-loopback support. Verified on fully patched Windows 10 22H2 (build 19045); unavailable APIs fail without capturing another audio source. It uses the existing dependencies and adds no external audio-routing tool.

Deterministic browser selection, Qt settings and failure-path checks:

```bash
python -m pytest tests/test_browser_audio.py
```

An optional native check plays two quiet synthetic tones in separate processes on the default output and verifies that only the selected process is captured. It requires an audible Windows output device and does not run by default (PowerShell):

```powershell
$env:LIVETRANSLATE_TEST_NATIVE_AUDIO = "1"
python -m pytest tests/test_browser_audio_native.py -s
```

## Live subtitle preview (opt-in)

Enable **Live subtitle preview** in the ASR/VAD settings to revise the current caption while speech continues. Enable **Translate provisional subtitles** separately for each translation model. Both options default to off; cloud preview translation makes additional requests and may incur charges.

The existing interim interval also controls preview checks (0.5–10 seconds). Preview starts after buffering at least one second or the chosen interval, whichever is longer. A 0.7-second interval favors speed; 1.5 seconds gives the first preview more context and reduces revisions. This is a check interval, not a guaranteed subtitle latency. Keep the source language fixed when it is known.

Preview uses an immutable copy of the current audio buffer. It does not trim audio or commit half-sentences. Final VAD segments retain priority; pending ASR and translation previews are replaced by newer previews. Provisional text changes in place, including in the standalone subtitle window, without its minimum-display delay or transition animations. Only final text enters transcript files, exported message history, and translation context.

For a rolling display, set the standalone subtitle window to two sentences. Live captions show the preceding confirmed sentence and the current provisional sentence on separate lines. The visible translation remains while its replacement is being computed; a failed final translation does not retain provisional text as a confirmed result. Current sentences remain revisable, including changes to their word order.

Pause, Clear, and model/language/device changes invalidate pending previews. An identical completed preview translation can be reused for the final result. Failed translations are not retried by the preview pipeline. This remains repeated inference with an existing ASR model, rather than a native streaming decoder.

Regression checks require the existing app dependencies and pytest:

```bash
python -m pytest tests/test_live_preview.py tests/test_preview_widgets.py tests/test_preview_integration.py
```

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
├── asr_funasr.py       Unified FunASR model selector backend
├── asr_sensevoice.py   SenseVoice backend
├── asr_funasr_nano.py  FunASR Nano backend
├── asr_anime_whisper.py Anime-Whisper backend (ja anime/galgame)
├── asr_remote.py        Remote Whisper client (→ asr_server.py, see REMOTE_ASR.md)
├── translator.py       OpenAI-compatible client (streaming, JSON schema, context)
├── model_manager.py    Model download & cache
├── subtitle_overlay.py PyQt6 overlay
├── control_panel.py    Settings UI (7 tabs)
├── dialogs.py          Wizard, download & model config dialogs
└── benchmark.py        Translation benchmark
```

## Acknowledgements

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — Whisper inference via CTranslate2
- [FunASR](https://github.com/modelscope/FunASR) — SenseVoice / Fun-ASR-Nano
- [Anime-Whisper](https://huggingface.co/litagin/anime-whisper) — Japanese anime/galgame ASR
- [Silero VAD](https://github.com/snakers4/silero-vad) — Voice activity detection

## Star History

<a href="https://www.star-history.com/?repos=TheDeathDragon%2FLiveTranslate&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=TheDeathDragon/LiveTranslate&type=date&theme=dark&legend=top-left&sealed_token=0t-kzcN9leqL-yaHdKcBdHLdLlE6NNHa48RpsLvUM3u2fOiZOYqWfjgplXxAtjk1ZJciSAbzI3gZ4PQqqHrOv4abM1CpOomUymVX6J1zPN-3Ygu0-Xr8Kpj3Xt8jWS05B4tTpuNSmoYqyHipPvKC7lxGfLcOF_zctjqvOka-j9gYWct0oQyJnGjdcZxY" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=TheDeathDragon/LiveTranslate&type=date&legend=top-left&sealed_token=0t-kzcN9leqL-yaHdKcBdHLdLlE6NNHa48RpsLvUM3u2fOiZOYqWfjgplXxAtjk1ZJciSAbzI3gZ4PQqqHrOv4abM1CpOomUymVX6J1zPN-3Ygu0-Xr8Kpj3Xt8jWS05B4tTpuNSmoYqyHipPvKC7lxGfLcOF_zctjqvOka-j9gYWct0oQyJnGjdcZxY" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=TheDeathDragon/LiveTranslate&type=date&legend=top-left&sealed_token=0t-kzcN9leqL-yaHdKcBdHLdLlE6NNHa48RpsLvUM3u2fOiZOYqWfjgplXxAtjk1ZJciSAbzI3gZ4PQqqHrOv4abM1CpOomUymVX6J1zPN-3Ygu0-Xr8Kpj3Xt8jWS05B4tTpuNSmoYqyHipPvKC7lxGfLcOF_zctjqvOka-j9gYWct0oQyJnGjdcZxY" />
 </picture>
</a>

## License

[MIT License](LICENSE)
