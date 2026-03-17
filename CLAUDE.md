# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LiveTrans is a real-time audio translation system for video players on Windows. It captures system audio via WASAPI loopback, runs speech recognition, and translates via LLM APIs, displaying results in a transparent overlay.

**Current phase**: Phase 0 Python prototype (Phase 1 will be a C++ DirectShow Audio Tap Filter).

## Running

```bash
# Must use the project venv (system Python lacks dependencies)
.venv/Scripts/python.exe main.py
```

Linter: `ruff` (installed globally). Run `python -m ruff check --select F,E,W --ignore E501,E402 *.py` to lint. E402 is intentionally ignored because `main.py` requires torch before PyQt6.

## Architecture

The pipeline runs in a background thread: **Audio Capture (32ms chunks) -> VAD -> ASR -> Translation (async) -> Overlay**

```
main.py (LiveTransApp)
  |-- model_manager.py     Centralized model detection, download, cache utils
  |-- audio_capture.py     WASAPI loopback via pyaudiowpatch, auto-reconnects on device change
  |-- vad_processor.py     Silero VAD / energy-based / disabled modes, progressive silence + backtrack split
  |-- asr_engine.py        faster-whisper (Whisper) backend
  |-- asr_sensevoice.py    FunASR SenseVoice backend (better for Japanese)
  |-- asr_funasr_nano.py   FunASR Nano backend
  |-- asr_qwen3.py         Qwen3-ASR backend (ONNX Encoder + GGUF Decoder)
  |-- qwen_asr_gguf/       Qwen3-ASR inference engine (from Qwen3-ASR-GGUF project)
  |-- translator.py        OpenAI-compatible API client, streaming, make_openai_client()
  |-- subtitle_overlay.py  PyQt6 transparent overlay (2-row header: controls + model/lang combos)
  |-- control_panel.py     Settings UI (5 tabs: VAD/ASR, Translation, Style, Benchmark, Cache)
  |-- dialogs.py           Setup wizard, model download/load dialogs, ModelEditDialog
  |-- benchmark.py         Translation benchmark (BENCH_SENTENCES, run_benchmark())
  |-- log_window.py        Real-time log viewer
```

### Threading Model

- **Main thread**: Qt event loop (all UI)
- **Pipeline thread**: `_pipeline_loop` in `LiveTransApp` - reads audio, runs VAD/ASR/translation
- **ASR loading**: Background thread via `_switch_asr_engine` (heavy model load, ~3-8s)
- Cross-thread UI updates use **Qt signals** (e.g., `add_message_signal`, `update_translation_signal`)
- ASR readiness tracked by `_asr_ready` flag; pipeline drops segments while loading

### Configuration

- `config.yaml` - Base configuration (audio, ASR, translation, subtitle defaults)
- `user_settings.json` - Runtime settings persisted by control panel (models, VAD params, ASR engine choice, optional `cache_path`). Takes priority over config.yaml on load.

### Model Config

Each model in `user_settings.json` has: `name`, `api_base`, `api_key`, `model`, `proxy` ("none"/"system"/custom URL), optional `no_system_role` (bool, for APIs that reject system messages like Qwen-MT).

Proxy handling: `proxy="none"` uses `httpx.Client(trust_env=False)` to bypass system proxy; `proxy="system"` uses default httpx behavior (env vars).

### Overlay UI (subtitle_overlay.py)

DragHandle is a 2-row header bar:
- **Row 1**: Draggable title + action buttons (Paused/Running, Clear, Settings, Monitor, Quit)
- **Row 2**: Checkboxes (Click-through, Top-most, Auto-scroll) + Model combo + Target Language combo

Style system:
- `DEFAULT_STYLE` and `STYLE_PRESETS` defined in `subtitle_overlay.py` — 14 presets including terminal themes (Dracula, Nord, Monokai, Solarized, Gruvbox, Tokyo Night, Catppuccin, One Dark, Everforest, Kanagawa)
- Default style is high-contrast (pure black background, white translation text, 14pt)
- Original and translation text have independent `font_family` fields (`original_font_family`, `translation_font_family`)
- `SubtitleOverlay.apply_style(style)` updates container/header backgrounds, window opacity, and rebuilds all message HTML
- Style dict stored in `user_settings.json` under `"style"` key; forwarded via `settings_changed` signal → `main.py` → `overlay.apply_style()`
- Backward compat: old `font_family` key auto-migrated to split fields in `apply_style()`

Key overlay features:
- **Top-most**: Toggles `WindowStaysOnTopHint`; requires `setWindowFlags()` + `show()` to take effect
- **Click-through**: Uses Win32 `WS_EX_TRANSPARENT` on the scroll area while keeping header interactive
- **Auto-scroll**: Controls whether new messages/translations auto-scroll to bottom
- **Model combo**: Populated from `user_settings.json` models list; switching emits `model_switch_requested` signal
- **Target Language combo**: Emits `target_language_changed`; synced from settings on startup

### Settings UX

- **Auto-save with debounce**: All control panel settings (combos, spinboxes) auto-save after 300ms debounce via `_auto_save()` → `_do_auto_save()`. No manual Save button needed.
- **Slider special handling**: VAD/Energy sliders update labels in real-time but only trigger save on `sliderReleased` (mouse) or immediately for keyboard input (`isSliderDown()` check).
- **Apply Prompt button**: Kept because TextEdit shouldn't trigger on every keystroke. Also persists to disk.
- **Cache path**: Default `./models/` (not `~/.cache`). Applied at startup in `main.py` before `import torch` via `model_manager.apply_cache_env()`.

### Startup Flow

1. `main.py` reads `user_settings.json` and calls `apply_cache_env()` before `import torch`
2. First launch (no `user_settings.json`) → `SetupWizardDialog`: choose hub + path + download Silero+SenseVoice
3. Non-first launch but models missing → `ModelDownloadDialog`: auto-download missing models
4. All models ready → create main UI (overlay, panel, pipeline)
5. Runtime ASR engine switch: if uncached → `ModelDownloadDialog`, then `_ModelLoadDialog` for GPU loading

### Key Patterns

- `torch` must be imported before PyQt6 to avoid DLL conflicts on Windows (PyTorch 2.9.0+ bug, see `main.py` and [pytorch#166628](https://github.com/pytorch/pytorch/issues/166628))
- Cache env vars set at module level in `main.py` before `import torch` to ensure `TORCH_HOME` is respected
- Deferred initialization: ASR model loading and settings application happen via `QTimer.singleShot(100)` after UI is shown to prevent startup freeze
- `make_openai_client()` in `translator.py` is the single shared function for proxy-aware OpenAI client creation (used by both translator and benchmark)
- `create_app_icon()` in `main.py` generates the app icon; set globally via `app.setWindowIcon()` so all windows inherit it
- Model cache detection (`is_asr_cached`, `get_local_model_path`) checks both ModelScope and HuggingFace paths to avoid redundant downloads when switching hubs
- Settings log output (`_apply_settings`) filters out `models` and `system_prompt` to avoid leaking API keys
- FunASR Nano: `asr_funasr_nano.py` does `os.chdir(model_dir)` before `AutoModel()` so relative paths in config.yaml (e.g. `Qwen3-0.6B`) resolve locally instead of triggering HuggingFace Hub network requests
- `Translator` defaults to 10s timeout via `make_openai_client()` to prevent API calls from hanging indefinitely
- Log window is created at startup but hidden; shown via tray menu "Show Log"
- Audio chunk duration is 32ms (512 samples at 16kHz), matching Silero VAD's native window size for minimal latency
- VAD adaptive silence mode: tracks recent pause durations, sets silence threshold to P75 × 1.2, auto-adjusts between 0.3s~2.0s
- VAD progressive silence: buffer越长接受越短的停顿切分 (<3s=full, 3-6s=half, 6-10s=quarter of silence_limit)
- VAD backtrack split: max duration时回溯confidence history找最低谷切分，remainder保留到下一段；三级策略（绝对低谷→相对低谷20%→低于均值兜底）
- FunASR `disable_pbar=True` required in all `generate()` calls — tqdm crashes in GUI process when flushing stderr
- ASR engine lifecycle: each engine exposes `unload()` (move to CPU + release) and `to_device(device)` (in-place migration). Device switching uses `to_device()` for PyTorch engines (SenseVoice/FunASR) and full reload for ctranslate2 (Whisper). Release order: `unload()` → `del` → `gc.collect()` → `torch.cuda.empty_cache()`
- Qwen3-ASR: ONNX Encoder (DirectML) + llama.cpp GGUF Decoder (Vulkan). Model files auto-downloaded from GitHub Release. llama.cpp DLLs go in `qwen_asr_gguf/inference/bin/`. No PyTorch dependency at inference time. `to_device()` returns False (must reload). Context injection via `_context` field for continuous recognition accuracy.
- Whisper (ctranslate2) only accepts `device="cuda"` not `"cuda:0"`; device index passed via `device_index` param. Parsed from combo text like `"cuda:0 (RTX 4090)"` in `_switch_asr_engine`
- VAD speech density filter: `_flush_segment()` discards segments where <25% of chunks are above confidence threshold (noise rejection)
- ASR text density filter: segments ≥2s producing ≤3 alnum characters are discarded as noise
- Settings file uses atomic write (write to `.tmp` then `os.replace`) to prevent corruption on crash
- `stop()` joins pipeline thread before flushing VAD to prevent concurrent `_process_segment` calls
- Cancelled ASR download/failed load restores `_asr_ready` if old engine is still available
- `Translator._build_system_prompt` catches format errors in user prompt templates, falls back to DEFAULT_PROMPT

## Language & Style

- Respond in Chinese
- Code comments in English only where critical
- Commit messages without Co-Authored-By
