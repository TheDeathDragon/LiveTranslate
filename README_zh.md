# LiveTrans

[English](README.md) | **中文**

Windows 实时音频翻译工具。捕获系统音频（WASAPI loopback）和可选的麦克风输入，语音识别后调用 LLM API 翻译，结果显示在透明悬浮字幕窗口上。

适用于看外语视频、直播、语音对话等场景——无需修改播放器，全局音频捕获即开即用。

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![Windows](https://img.shields.io/badge/Platform-Windows-0078d4)
![License](https://img.shields.io/badge/License-MIT-green)

## 截图

![LiveTrans](screenshot/zh.png)

## 安装视频

[![安装演示](https://img.shields.io/badge/Bilibili-安装演示-00A1D6?logo=bilibili)](https://www.bilibili.com/video/BV1K2Awz6Euw) 适用于看外语视频、直播、ASMR等场景，也可以语音输入实时并行翻译多种语音

## 功能特性

- **实时翻译管线**：系统音频 → VAD → ASR → LLM 翻译 → 字幕显示
- **多 ASR 引擎**：faster-whisper、SenseVoice、FunASR Nano、Qwen3-ASR（GGUF）
- **兼容任意 OpenAI 格式 API**：DeepSeek、Grok、Qwen、GPT、Ollama、vLLM 等
- **麦克风混音**：可选将麦克风输入混合到系统音频一起识别
- **低延迟 VAD**：32ms 音频块 + Silero VAD，自适应静音检测
- **透明悬浮窗**：始终置顶、鼠标穿透、可拖拽，14 种配色主题
- **CUDA 加速**：ASR 模型 GPU 推理
- **模型自动管理**：首次启动向导，支持 ModelScope / HuggingFace 双源
- **内置基准测试**：对比翻译模型速度和质量

## 开发计划

- **主播模式**：为 OBS 提供透明窗口，实时翻译主播语音，支持多语言同时输出——可用作画面字幕或直播间留言板

## 系统要求

- **操作系统**：Windows 10/11
- **Python**：3.10+
- **GPU**（推荐）：NVIDIA 显卡 + CUDA 12.6（RTX 50 系列等 Blackwell 架构需要 CUDA 12.8）
- **网络**：需要访问翻译 API

## 快速开始

```bash
git clone https://github.com/TheDeathDragon/LiveTranslate.git
cd LiveTranslate
```

双击 **`install.bat`** 一键安装——脚本会自动：
1. 检测 Python 3.10+（未安装则通过 winget 自动安装）
2. 创建虚拟环境
3. 检测 NVIDIA 显卡，选择 CUDA / CPU 版 PyTorch
4. 安装全部依赖

安装完成后双击 **`start.bat`** 启动。

更新时双击 **`update.bat`**——自动拉取最新代码并更新依赖（未安装 Git 会通过 winget 自动安装）。

<details>
<summary>手动安装</summary>

```bash
python -m venv .venv
.venv\Scripts\activate

# PyTorch（三选一）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126  # CUDA
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128  # CUDA（RTX 50 系列）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu    # 仅 CPU

# 依赖
pip install -r requirements.txt
pip install funasr --no-deps

# 启动
.venv\Scripts\python.exe main.py
```

> FunASR 使用 `--no-deps` 安装，因为 `editdistance` 需要 C++ 编译器。`requirements.txt` 中已包含纯 Python 替代品 `editdistance-s`。

</details>

## 首次使用

1. 弹出设置向导——选择下载源（ModelScope 适合国内，HuggingFace 适合海外）和缓存路径
2. 自动下载 Silero VAD + SenseVoice 模型（约 1GB）
3. 下载完成后进入主界面

## 配置翻译 API

设置 → 翻译标签页：

| 参数 | 示例 |
|------|------|
| API Base | `https://api.deepseek.com/v1` |
| API Key | 你的密钥 |
| Model | `deepseek-chat` |
| 代理 | `none` / `system` / 自定义地址 |

## 架构

```
Audio (WASAPI 32ms) → VAD (Silero) → ASR → LLM Translation → Overlay
         ↑ 可选麦克风混音
```

```
main.py                 主入口，管线编排
├── audio_capture.py    WASAPI loopback + 麦克风混音
├── vad_processor.py    Silero VAD
├── asr_engine.py       faster-whisper 后端
├── asr_sensevoice.py   SenseVoice 后端
├── asr_funasr_nano.py  FunASR Nano 后端
├── asr_qwen3.py        Qwen3-ASR 后端 (ONNX + GGUF)
├── translator.py       OpenAI 兼容翻译客户端
├── model_manager.py    模型下载与缓存管理
├── subtitle_overlay.py PyQt6 透明悬浮窗
├── control_panel.py    设置面板 UI
├── dialogs.py          设置向导、下载对话框
└── benchmark.py        翻译基准测试
```

## 致谢

- [CapsWriter-Offline](https://github.com/HaujetZhao/CapsWriter-Offline) — Qwen3-ASR 集成架构参考
- [Qwen3-ASR-GGUF](https://github.com/HaujetZhao/Qwen3-ASR-GGUF) — ONNX + GGUF 混合推理引擎
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — GGUF 模型推理运行时

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
