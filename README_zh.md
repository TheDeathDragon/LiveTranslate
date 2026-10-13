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
- **多 ASR 引擎**：faster-whisper、SenseVoice、FunASR Nano、Anime-Whisper
- **远程 ASR**：通过 HTTP 把语音识别放到 GPU 机器上跑 —— 见 [REMOTE_ASR.md](REMOTE_ASR.md)
- **兼容任意 OpenAI 格式 API**：DeepSeek、Grok、Qwen、GPT、Ollama、vLLM 等
- **流式翻译显示**：翻译结果逐字实时显示
- **模型独立配置**：流式传输、结构化输出(JSON)、上下文历史、禁用思考
- **麦克风混音**：可选将麦克风输入混合到系统音频一起识别
- **仅捕捉浏览器声音**：支持 Chrome 和搜狗浏览器，排除其他应用及麦克风输入
- **低延迟 VAD**：32ms 音频块 + Silero VAD，自适应静音检测
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
python -m venv .venv
.venv\Scripts\activate

# PyTorch（三选一）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126  # CUDA
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128  # CUDA（RTX 50 系列）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu    # 仅 CPU

# 依赖
pip install -r requirements.txt

# 启动
.venv\Scripts\python.exe main.py
```

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

## 仅捕捉浏览器声音

在“设置 → VAD / 语音识别 → 音频”中选择“仅 Chrome 声音”或“仅搜狗浏览器声音”，就可以一边看直播，一边使用独立的语音聊天软件。这两个模式会关闭麦克风捕捉，失败时也不会切回系统音频。原有系统音频、输出设备和麦克风模式仍可使用。

程序优先捕捉 Windows 音频会话中实际发声的浏览器进程及其子进程，包括搜狗的音频子进程。暂停后会保留音频会话；关闭并重新打开浏览器后会自动连接。如果同时存在多个发声的浏览器进程，捕捉会停止并记录原因。请关闭或暂停其他实例，再重新开始捕捉。这个模式不能单独隔离一个标签页或窗口：共用所选音频进程的标签页会一起捕捉，浏览器内的语音聊天也可能被包含。

此功能需要 Windows 的应用音频捕捉接口，已在安装了系统更新的 Windows 10 22H2（内部版本 19045）上验证。接口不可用时会停止捕捉，不会改抓其他音源。不需要新增依赖或安装音频路由工具。

浏览器选择、Qt 设置和失败处理的确定性检查：

```bash
python -m pytest tests/test_browser_audio.py
```

另有一项默认不运行的原生检查。它会让两个独立进程在默认输出设备播放轻声测试音，验证只捕捉所选进程；需要 Windows 上可播放声音的输出设备。使用 PowerShell 运行：

```powershell
$env:LIVETRANSLATE_TEST_NATIVE_AUDIO = "1"
python -m pytest tests/test_browser_audio_native.py -s
```

## 实时字幕预览

在语音识别设置中开启“实时字幕预览”，程序会在说话过程中更新当前字幕。你还需要为翻译模型单独开启“翻译临时字幕”。两个开关默认关闭。云端预译会增加请求次数，可能产生费用。

检查间隔沿用增量识别的设置，可选择 0.5 秒至 10 秒。首次预览至少积累 1 秒音频；间隔超过 1 秒时，首次也会等待所选间隔。0.7 秒偏向速度，1.5 秒能为首次预览提供更多上下文，减少修订。这个数值表示检查间隔，实际字幕延迟需要实测。已知源语言时，建议固定语言。

预览使用当前音频的独立副本，不会裁掉音频或提前保存半句话。程序优先处理完整片段，等待中的识别和翻译预览只保留最新一份。透明字幕窗口会直接修订当前文字，跳过最短显示等待和切换动画。只有确认结果写入记录文件、导出历史和翻译上下文。

想使用滚动字幕时，将透明字幕窗口的句数设为 2。实时模式会分行显示前一句确认结果和当前临时字幕。等待新译文时保留当前译文，避免清空闪烁。最终翻译失败时不会将临时结果冒充确认结果。当前句仍允许修订，包括词序变化。

暂停、清空和切换模型、语言或设备时，程序会使旧预览失效。最终原文与已完成预译一致时，程序可以复用译文。预览翻译失败后不会自动重试。预览仍使用现有识别模型反复推理，并未改成原生流式解码。

开发检查需要现有程序依赖和 pytest：

```bash
python -m pytest tests/test_live_preview.py tests/test_preview_widgets.py tests/test_preview_integration.py
```

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
├── asr_funasr.py       统一 FunASR 模型选择后端
├── asr_sensevoice.py   SenseVoice 后端
├── asr_funasr_nano.py  FunASR Nano 后端
├── asr_anime_whisper.py Anime-Whisper 后端 (日语动画/Galgame)
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
- [Silero VAD](https://github.com/snakers4/silero-vad) — 语音活动检测

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
