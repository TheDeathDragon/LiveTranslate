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
- **Multiple ASR engines**: faster-whisper, SenseVoice, FunASR Nano, Anime-Whisper, CrispASR, sherpa-onnx, parakeet.cpp, Remote Whisper
- **Remote ASR**: offload speech recognition to a GPU machine over HTTP — see [REMOTE_ASR.md](REMOTE_ASR.md)
- **Any OpenAI-compatible API**: DeepSeek, Grok, Qwen, GPT, Ollama, vLLM, etc.
- **Streaming translation display**: Real-time character-by-character translation output
- **Per-model settings**: Streaming, structured output (JSON), context history, disable thinking
- **Microphone mix-in**: Optionally mix microphone input with system audio for ASR
- **Low-latency VAD**: 32ms chunks + Silero VAD or optional FireRedVAD with adaptive silence detection
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
uv venv --python 3.12 .venv
.venv\Scripts\activate

# PyTorch (choose one)
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126  # CUDA
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128  # CUDA (RTX 50xx)
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu    # CPU only

# Dependencies
uv sync --locked --inexact --no-install-package torch --no-install-package torchaudio
uv pip install funasr --no-deps
uv pip install "sherpa-onnx>=1.13.3" "sherpa-onnx-bin>=1.13.3"

# Launch
.venv\Scripts\python.exe main.py
```

> FunASR uses `--no-deps` because `editdistance` requires a C++ compiler. `editdistance-s` in `pyproject.toml` is a pure-Python drop-in replacement.
> `uv sync` installs the optional FireRedVAD Python package; its model is downloaded separately only if you choose to use that VAD backend.

</details>

## First Launch

1. Setup wizard appears — choose download source (ModelScope / HuggingFace) and cache path
2. Silero VAD + SenseVoice models download automatically (~1GB)
3. Main UI appears when ready

## FireRedVAD Models

FireRedVAD is optional and is not downloaded during first launch. To use it, download the official model under `models/FireRedVAD`:

```powershell
modelscope download --model xukaituo/FireRedVAD --local_dir ./models/FireRedVAD
# or
huggingface-cli download FireRedTeam/FireRedVAD --local-dir ./models/FireRedVAD
```

Make sure these files exist:

```text
models/FireRedVAD/Stream-VAD/cmvn.ark
models/FireRedVAD/Stream-VAD/model.pth.tar
```

Then open Settings → VAD/ASR, choose `FireRedVAD`, click Refresh, and select the local Stream-VAD model. The first integration uses FireRedVAD only as a streaming speech-confidence backend; LiveTranslate still owns segmentation, silence handling, backtrack splitting, incremental ASR, and ASR queueing. FireRed AED is not connected.

## sherpa-onnx Models

LiveTranslate supports sherpa-onnx local ONNX models through Python `OfflineRecognizer` and `OnlineRecognizer` APIs. Online models are currently decoded as a VAD-segment wrapper, not as true partial streaming ASR. `install.ps1` installs the CPU wheel by default. CUDA requires replacing it with a CUDA wheel, for example:

```powershell
powershell -ExecutionPolicy Bypass -File install.ps1 -SherpaOnnxRuntime cuda12
```

Download sherpa-onnx ASR model archives from the official sherpa-onnx releases, extract them anywhere under `models/`, then open Settings → VAD/ASR, choose `sherpa-onnx (ONNX)`, click Refresh, and select the local model directory. Online transducer scans accept `encoder.onnx`/`decoder.onnx`/`joiner.onnx` and int8 variants such as `encoder.int8.onnx`/`decoder.int8.onnx`/`joiner.int8.onnx`. PR #3671 Nemotron packages are published with names like `sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11`; unofficial snapshots must still have ONNX files accepted by the installed sherpa-onnx/ONNX Runtime version. The `onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4` layout is not treated as a sherpa-onnx model in this path.

## parakeet.cpp Models

parakeet.cpp is optional and is not downloaded during first launch. It uses local GGUF models plus a native parakeet.cpp runtime. The first integration uses LiveTranslate's existing VAD segments and ASR worker process; parakeet.cpp streaming EOU is not connected yet.

Optional installer commands:

```powershell
powershell -ExecutionPolicy Bypass -File install.ps1 -InstallParakeetCpp -ParakeetCppBackend cpu
powershell -ExecutionPolicy Bypass -File install.ps1 -DownloadParakeetCppModel -ParakeetCppModel tdt_ctc-110m-q4_k
```

Manual layout:

```text
models/parakeet.cpp/runtime/v0.3.2/<cpu|cuda|vulkan>/
models/parakeet.cpp/models/tdt_ctc-110m-q4_k.gguf
```

Then open Settings → VAD/ASR, choose `parakeet.cpp (GGUF)`, click Refresh for both model and runtime, and select them. CUDA runtime may require the matching `cudart-parakeet-bin-win-cuda-x64.zip` asset next to the runtime DLLs.

## Translation API

Settings → Translation tab:

| Parameter | Example |
|-----------|---------|
| API Base | `https://api.deepseek.com/v1` |
| API Key | Your key |
| Model | `deepseek-chat` |
| Proxy | `none` / `system` / custom URL |

Real API keys are stored in `user_settings.json`, which is git-ignored. Keep `config.yaml` free of real credentials.

## Architecture

```
Audio (WASAPI 32ms) → VAD (Silero / FireRedVAD / Energy) → ASR → LLM Translation → Overlay
         ↑ optional mic mix-in
```

```
main.py                 Entry point & pipeline
├── audio_capture.py    WASAPI loopback + mic mix-in
├── vad_processor.py    VAD state machine (Silero / FireRedVAD / energy / disabled)
├── vad_firered.py      FireRedVAD streaming frame adapter
├── asr_engine.py       faster-whisper backend
├── asr_funasr.py       Unified FunASR model selector backend
├── asr_sensevoice.py   SenseVoice backend
├── asr_funasr_nano.py  FunASR Nano backend
├── asr_anime_whisper.py Anime-Whisper backend (ja anime/galgame)
├── asr_crispasr.py     CrispASR ggml runtime backend
├── asr_sherpa_onnx.py  sherpa-onnx OfflineRecognizer/OnlineRecognizer backend
├── asr_parakeet_cpp.py parakeet.cpp C API GGUF backend
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
- CrispASR — ggml C++ ASR runtime hub with GGUF/bin single-file models, used through its Python binding in the ASR worker
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — ONNX ASR runtime used through `OfflineRecognizer` and segment-wrapped `OnlineRecognizer`
- [parakeet.cpp](https://github.com/mudler/parakeet.cpp) — NVIDIA NeMo Parakeet GGUF inference through the C API in the ASR worker
- [Silero VAD](https://github.com/snakers4/silero-vad) — Voice activity detection
- [FireRedVAD](https://github.com/FireRedTeam/FireRedVAD) — Optional streaming VAD confidence backend

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
