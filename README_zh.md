# LiveTranslate

[English](README.md) | **中文**

Windows 实时音频翻译工具。捕获系统音频（WASAPI loopback）和可选的麦克风输入，语音识别后调用 LLM API 翻译，结果显示在透明悬浮字幕窗口上。

适用于看外语视频、直播、语音对话等场景——无需修改播放器，全局音频捕获即开即用。

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![Windows](https://img.shields.io/badge/Platform-Windows-0078d4)
![License](https://img.shields.io/badge/License-MIT-green)

## 截图

![LiveTranslate](screenshot/zh.png)

## 安装视频

[![安装演示](https://img.shields.io/badge/Bilibili-安装演示-00A1D6?logo=bilibili)](https://www.bilibili.com/video/BV1K2Awz6Euw) 适用于看外语视频、直播、ASMR等场景，也可以语音输入实时并行翻译多种语音

## 功能特性

- **实时翻译管线**：系统音频 → VAD → ASR → LLM 翻译 → 字幕显示
- **多 ASR 引擎**：faster-whisper、SenseVoice、FunASR Nano、Anime-Whisper、CrispASR、sherpa-onnx、parakeet.cpp、Remote Whisper
- **远程 ASR**：通过 HTTP 把语音识别放到 GPU 机器上跑 —— 见 [REMOTE_ASR.md](REMOTE_ASR.md)
- **兼容任意 OpenAI 格式 API**：DeepSeek、Grok、Qwen、GPT、Ollama、vLLM 等
- **流式翻译显示**：翻译结果逐字实时显示
- **模型独立配置**：流式传输、结构化输出(JSON)、上下文历史、禁用思考
- **麦克风混音**：可选将麦克风输入混合到系统音频一起识别
- **低延迟 VAD**：32ms 音频块 + Silero VAD，可选 FireRedVAD，自适应静音检测
- **透明悬浮窗**：始终置顶、鼠标穿透、可拖拽，14 种配色主题
- **CUDA 加速**：ASR 模型 GPU 推理
- **模型自动管理**：首次启动向导，支持 ModelScope / HuggingFace 双源
- **内置基准测试**：对比翻译模型速度和质量

## 更新日志

查看 [中文更新日志](i18n/CHANGELOG_zh.md) | [English Changelog](i18n/CHANGELOG_en.md)

## 系统要求

- **操作系统**：Windows 10/11
- **Python**：3.10–3.12（绿色版免装）
- **GPU**（推荐）：NVIDIA 显卡 + CUDA 12.6（RTX 50 系列等 Blackwell 架构需要 CUDA 12.8）
- **网络**：需要访问翻译 API

## 快速开始

### 绿色版（免装 Python，推荐新手）

从 [Releases](https://github.com/TheDeathDragon/LiveTranslate/releases) 下载 `LiveTranslate-portable-*.zip`，解压后双击 **`start.bat`** 即可。首次运行会自动下载便携版 Python 3.12 并按显卡安装依赖，无需预装任何 Python。

### 从源码安装

```bash
git clone https://github.com/TheDeathDragon/LiveTranslate.git
cd LiveTranslate
```

双击 **`install.bat`** 一键安装——脚本会自动：
1. 检测 Python 3.10–3.12（未安装则通过 winget 自动安装）
2. 创建虚拟环境
3. 检测 NVIDIA 显卡，选择 CUDA / CPU 版 PyTorch
4. 安装全部依赖

安装完成后双击 **`start.bat`** 启动。

更新时双击 **`update.bat`**——自动拉取最新代码并更新依赖（未安装 Git 会通过 winget 自动安装）。

<details>
<summary>手动安装</summary>

```bash
uv venv --python 3.12 .venv
.venv\Scripts\activate

# PyTorch（三选一）
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126  # CUDA
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128  # CUDA（RTX 50 系列）
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu    # 仅 CPU

# 依赖
uv sync --locked --inexact --no-install-package torch --no-install-package torchaudio
uv pip install funasr --no-deps
uv pip install "sherpa-onnx>=1.13.3" "sherpa-onnx-bin>=1.13.3"

# 启动
.venv\Scripts\python.exe main.py
```

> FunASR 使用 `--no-deps` 安装，因为 `editdistance` 需要 C++ 编译器。`pyproject.toml` 中已包含纯 Python 替代品 `editdistance-s`。
> `uv sync` 会安装可选的 FireRedVAD Python 包；FireRedVAD 模型只在你选择使用该 VAD 后端时单独下载。

</details>

## 首次使用

1. 弹出设置向导——选择下载源（ModelScope 适合国内，HuggingFace 适合海外）和缓存路径
2. 自动下载 Silero VAD + SenseVoice 模型（约 1GB）
3. 下载完成后进入主界面

## FireRedVAD 模型

FireRedVAD 是可选 VAD 后端，首次启动不会强制下载。使用前请把官方模型下载到 `models/FireRedVAD`：

```powershell
modelscope download --model xukaituo/FireRedVAD --local_dir ./models/FireRedVAD
# 或
huggingface-cli download FireRedTeam/FireRedVAD --local-dir ./models/FireRedVAD
```

确保存在：

```text
models/FireRedVAD/Stream-VAD/cmvn.ark
models/FireRedVAD/Stream-VAD/model.pth.tar
```

然后在 设置 → VAD/ASR → VAD 模式 中选择 `FireRedVAD`，点击刷新并选择本地 Stream-VAD 模型。当前集成只把 FireRedVAD 用作 streaming speech confidence 后端；分段、静音、回溯切分、增量 ASR 和 ASR queue 仍由 LiveTranslate 的 VADProcessor 负责。FireRed AED 暂未接入。

## sherpa-onnx 模型

LiveTranslate 通过 Python `OfflineRecognizer` 和 `OnlineRecognizer` API 接入 sherpa-onnx 本地 ONNX 模型。Online 模型当前是对 VAD 切段做整段 segment-wrapper 解码，不是真正逐 chunk partial streaming ASR。`install.ps1` 默认安装 CPU wheel；CUDA 版需要用 CUDA wheel 替换，例如：

```powershell
powershell -ExecutionPolicy Bypass -File install.ps1 -SherpaOnnxRuntime cuda12
```

从 sherpa-onnx 官方 releases 下载 ASR 模型压缩包，解压到 `models/` 下任意子目录；然后在 设置 → VAD/ASR 中选择 `sherpa-onnx (ONNX)`，点击刷新并选择本地模型目录。Online transducer 扫描支持 `encoder.onnx`/`decoder.onnx`/`joiner.onnx`，也支持 `encoder.int8.onnx`/`decoder.int8.onnx`/`joiner.int8.onnx`。PR #3671 对应的 Nemotron 包名类似 `sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11`；非官方 snapshot 的 ONNX 文件仍必须能被当前安装的 sherpa-onnx/ONNX Runtime 接受。`onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4` 这类结构不属于此 sherpa-onnx 路线。

## parakeet.cpp 模型

parakeet.cpp 是可选 ASR 后端，首次启动不会强制下载。它使用本地 GGUF 模型和 parakeet.cpp native runtime。当前第一阶段仍使用 LiveTranslate 现有 VAD 切段和 ASR worker，暂不接入 parakeet.cpp streaming EOU。

可选安装命令：

```powershell
powershell -ExecutionPolicy Bypass -File install.ps1 -InstallParakeetCpp -ParakeetCppBackend cpu
powershell -ExecutionPolicy Bypass -File install.ps1 -DownloadParakeetCppModel -ParakeetCppModel tdt_ctc-110m-q4_k
```

手动放置路径：

```text
models/parakeet.cpp/runtime/v0.3.2/<cpu|cuda|vulkan>/
models/parakeet.cpp/models/tdt_ctc-110m-q4_k.gguf
```

然后在 设置 → VAD/ASR 中选择 `parakeet.cpp (GGUF)`，分别刷新并选择本地 GGUF 模型和 runtime。CUDA runtime 可能需要把匹配的 `cudart-parakeet-bin-win-cuda-x64.zip` 一并解压到 runtime DLL 附近。

## 配置翻译 API

设置 → 翻译标签页：

| 参数 | 示例 |
|------|------|
| API Base | `https://api.deepseek.com/v1` |
| API Key | 你的密钥 |
| Model | `deepseek-chat` |
| 代理 | `none` / `system` / 自定义地址 |

真实 API Key 会保存在已被 git 忽略的 `user_settings.json` 中，不要把真实凭据写入 `config.yaml`。

## 架构

```
Audio (WASAPI 32ms) → VAD (Silero / FireRedVAD / Energy) → ASR → LLM Translation → Overlay
         ↑ 可选麦克风混音
```

```
main.py                 主入口，管线编排
├── audio_capture.py    WASAPI loopback + 麦克风混音
├── vad_processor.py    VAD 状态机 (Silero / FireRedVAD / energy / disabled)
├── vad_firered.py      FireRedVAD streaming frame 适配器
├── asr_engine.py       faster-whisper 后端
├── asr_funasr.py       统一 FunASR 模型选择后端
├── asr_sensevoice.py   SenseVoice 后端
├── asr_funasr_nano.py  FunASR Nano 后端
├── asr_anime_whisper.py Anime-Whisper 后端 (日语动画/Galgame)
├── asr_crispasr.py     CrispASR ggml runtime 后端
├── asr_sherpa_onnx.py  sherpa-onnx OfflineRecognizer/OnlineRecognizer 后端
├── asr_parakeet_cpp.py parakeet.cpp C API GGUF 后端
├── asr_remote.py        远程 Whisper 客户端 (→ asr_server.py, 见 REMOTE_ASR.md)
├── translator.py       OpenAI 兼容翻译客户端 (流式/JSON/上下文)
├── model_manager.py    模型下载与缓存管理
├── subtitle_overlay.py PyQt6 透明悬浮窗
├── control_panel.py    设置面板 UI (7 个标签页)
├── dialogs.py          设置向导、下载、模型配置对话框
└── benchmark.py        翻译基准测试
```

## 致谢

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — 基于 CTranslate2 的 Whisper 推理
- [FunASR](https://github.com/modelscope/FunASR) — SenseVoice / Fun-ASR-Nano
- [Anime-Whisper](https://huggingface.co/litagin/anime-whisper) — 日语动画/Galgame 专用 ASR
- CrispASR — ggml C++ ASR runtime hub，使用 GGUF/bin 单文件模型，LiveTranslate 通过 ASR worker 内的 Python binding 调用
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — 通过 `OfflineRecognizer` 和 segment-wrapper `OnlineRecognizer` 调用的 ONNX ASR runtime
- [parakeet.cpp](https://github.com/mudler/parakeet.cpp) — 通过 ASR worker 内 C API 调用的 NVIDIA NeMo Parakeet GGUF 推理后端
- [Silero VAD](https://github.com/snakers4/silero-vad) — 语音活动检测
- [FireRedVAD](https://github.com/FireRedTeam/FireRedVAD) — 可选 streaming VAD confidence 后端

## Star History

<a href="https://www.star-history.com/?repos=TheDeathDragon%2FLiveTranslate&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/image?repos=TheDeathDragon/LiveTranslate&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/image?repos=TheDeathDragon/LiveTranslate&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/image?repos=TheDeathDragon/LiveTranslate&type=date&legend=top-left" />
 </picture>
</a>

## 许可证

[MIT License](LICENSE)
