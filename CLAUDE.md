# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LiveTranslate is a real-time audio translation system for video players on Windows. It captures system audio via WASAPI loopback, runs speech recognition, and translates via LLM APIs, displaying results in a transparent overlay.

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
main.py (LiveTranslateApp)
  |-- model_manager.py     Centralized model detection, download, cache utils
  |-- pipeline_controller.py Audio capture + VAD + ASR queue controller, incremental ASR coordination
  |-- audio_capture.py     WASAPI loopback via pyaudiowpatch, auto-reconnects on device change
  |-- vad_processor.py     Silero VAD / FireRedVAD / energy-based / disabled modes, progressive silence + backtrack split
  |-- vad_firered.py      FireRedVAD streaming rolling-frame confidence adapter
  |-- asr_client.py        Main-process ASR worker manager (spawn, Pipe IPC, timeouts)
  |-- asr_worker.py        ASR subprocess entrypoint; loads and owns one backend/model
  |-- asr_engine.py        faster-whisper (Whisper) backend
  |-- asr_sensevoice.py    FunASR SenseVoice backend (better for Japanese)
  |-- asr_funasr_nano.py   FunASR Nano backend
  |-- asr_anime_whisper.py Anime-Whisper backend (litagin/anime-whisper, ja anime/galgame)
  |-- asr_crispasr.py     CrispASR ggml runtime backend (GGUF/bin single-file models)
  |-- asr_sherpa_onnx.py  sherpa-onnx OfflineRecognizer/OnlineRecognizer backend (local ONNX model dirs)
  |-- asr_parakeet_cpp.py parakeet.cpp C API backend (local GGUF model + native runtime)
  |-- translator.py        OpenAI-compatible API client, streaming, JSON schema, context history
  |-- subtitle_overlay.py  PyQt6 transparent overlay (2-row header: controls + model/lang combos)
  |-- subtitle_window.py   Standalone subtitle window for OBS capture (outlined text, animations)
  |-- subtitle_settings.py Subtitle window settings UI (grid layout, text line editor)
  |-- control_panel.py     Settings UI (8 tabs: ASR, VAD, Translation, Style, Subtitle, Benchmark, Cache, Changelog)
  |-- dialogs.py           Setup wizard, model download/load dialogs, ModelEditDialog
  |-- benchmark.py         Translation benchmark (BENCH_SENTENCES, run_benchmark())
  |-- log_window.py        Real-time log viewer
```

### Threading / Process Model

- **Main thread**: Qt event loop (all UI)
- **Capture thread**: `_capture_loop` in `PipelineController` reads audio and runs VAD
- **ASR queue thread**: `_asr_loop` in `PipelineController` drains VAD segments and calls ASR via `ASRService`
- **ASR worker process**: `asr_worker.py` owns the concrete ASR backend/model and runs inference over `multiprocessing.Pipe`
- **ASR loading**: `_switch_asr_engine()` stops the current worker, then starts the target worker in a background thread; if target loading fails, it restarts the previous worker from its saved config
- Cross-thread UI updates use **Qt signals** (e.g., `add_message_signal`, `update_translation_signal`)
- ASR readiness tracked by `ASRService`; `PipelineController` drops segments while no ready worker exists

### Configuration

- `config.yaml` - Base configuration (audio, ASR, translation, subtitle defaults)
- `user_settings.json` - Runtime settings persisted by control panel (models, VAD params, ASR engine choice, optional `cache_path`). Takes priority over config.yaml on load.

### Model Config

Each model in `user_settings.json` has: `name`, `api_base`, `api_key`, `model`, `proxy` ("none"/"system"/custom URL), and optional per-model flags:

- `no_system_role` (bool): Merges system prompt into user message for APIs that reject system role (e.g. Qwen-MT)
- `no_think` (bool, default True): Passes `extra_body={"enable_thinking": False}` to disable reasoning for thinking models (Qwen3 etc.)
- `streaming` (bool, default True): Per-model streaming toggle; streaming mode yields partial text via `translate_iter()` generator
- `json_response` (bool, default False): Uses `response_format={"type": "json_schema"}` with schema `{"t": "string"}` for structured output; mutually exclusive with streaming UI display
- `context_turns` (int, default 0): Number of recent (source, translation) pairs to include as multi-turn context in messages
- `input_price`/`output_price` (float, per 1M tokens): Token pricing for cost estimation displayed in MonitorBar
- `overrides` (dict, optional): Per-request OpenAI chat-completion params that override defaults — supported keys `temperature`, `top_p`, `max_tokens`, `frequency_penalty`, `presence_penalty`, `seed`. Only keys present in the dict are sent; absent keys fall back to constructor defaults (temperature/max_tokens) or are omitted entirely (others)
- `extra_body` (dict, optional): Provider-specific params (e.g. `thinking_budget`, `reasoning_effort`) merged into the request's `extra_body`. Combined with `no_think`'s `enable_thinking=false` when both are set
- Proxy handling: `proxy="none"` uses `httpx.Client(trust_env=False)` to bypass system proxy; `proxy="system"` uses default httpx behavior (env vars)
- `Translator._build_request_kwargs()` is the single assembly point — it injects `overrides`, merges `extra_body` with `no_think`, and stamps `response_format` for `json_response`. All three code paths (`_translate_sync`, `_translate_streaming`, streaming `translate_iter`) go through it
- ModelEditDialog "Advanced Parameters" group uses `[Override] checkbox + value widget` pattern — unchecked rows don't write to the `overrides` dict, preserving backward-compat. `extra_body` is a multi-line JSON text field validated on OK
- Editing the active model in ModelEditDialog triggers `model_changed` signal to immediately apply changes

### Subtitle Window (subtitle_window.py)

Standalone transparent window for OBS capture, separate from the main overlay:
- `SubtitleWindow`: Frameless, transparent, middle-click draggable, auto-hides after timeout, default position (100, 100)
- `_SubtitleTextWidget`: QPainterPath-based outlined text rendering per line, with automatic word-wrap and entry/exit animations
- Text rendered to cached QPixmap; animations only blit the cached image (no per-frame path rendering)
- Each line has independent font, color, outline, alignment, animation settings
- Long text auto-wraps at punctuation/word boundaries, expanding widget height for multiple lines
- `desired_height()` returns height based on wrapped line count (never 0), preventing window collapse when empty
- Window height animates smoothly (150ms OutCubic) when wrap count changes, keeping vertical center stable
- Animation order: exit animation → height change → entry animation
- Settings UI in `subtitle_settings.py`: grid layout, text lines as list with double-click edit dialog

### Overlay UI (subtitle_overlay.py)

DragHandle is a 2-row header bar:
- **Row 1**: Draggable title + action buttons (Hide, Subtitle, Paused/Running, Clear, Full/Compact, Settings, Quit)
- **Row 2**: Checkboxes (Click-through, Top-most, Auto-scroll, Taskbar) + Model combo + Target Language combo

MonitorBar displays: ASR device, CPU/RAM/GPU usage, ASR/TL counts, token usage with cost estimation (¥/$ based on UI language).

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
- **Compact mode animation**: Toggles between full and minimumHeight with 200ms size animation; uses `frameGeometry()` for actual window size to avoid Windows MINMAXINFO mismatch; skips animation when height difference < 10px
- **Position persistence**: `moveEvent`/`resizeEvent` with 500ms debounce save `overlay_x/y/w/h` to `user_settings.json`; restored on startup
- **Reset positions**: `reset_positions` signal from ControlPanel; subtitle window → (100, 100), overlay → bottom-right of screen

### Settings UX

- **Auto-save with debounce**: All control panel settings (combos, spinboxes) auto-save after 300ms debounce via `_auto_save()` → `_do_auto_save()`. No manual Save button needed.
- **Slider special handling**: VAD/Energy sliders update labels in real-time but only trigger save on `sliderReleased` (mouse) or immediately for keyboard input (`isSliderDown()` check).
- **Prompt auto-apply**: System prompt TextEdit uses 600ms debounce via `textChanged` signal — no manual Apply button needed.
- **Cache path**: Default `./models/` (not `~/.cache`). Applied at startup in `main.py` before `import torch` via `model_manager.apply_cache_env()`.
- **Whisper group visibility**: Shows/hides download group when switching ASR engine; window auto-resizes via `sizeHint()`.
- **QDoubleSpinBox precision**: All float values `round()` to 2 decimal places on save to avoid floating-point drift (e.g. `0.9999999999999992`).

### Startup Flow

1. `main.py` reads `user_settings.json` and calls `apply_cache_env()` before `import torch`
2. First launch (no `user_settings.json`) → `SetupWizardDialog`: choose hub + path + download Silero+SenseVoice
3. Non-first launch but models missing → `ModelDownloadDialog`: auto-download missing models
4. All models ready → create main UI (overlay, panel, pipeline)
5. Runtime ASR engine switch: if uncached → `ModelDownloadDialog`; then `_ModelLoadDialog` while the app shuts down the current worker and loads the target worker. If target loading fails, the app tries to restore the previous worker.

### Incremental ASR

Continuous speech is processed incrementally to reduce latency (enabled by `incremental_asr` setting):

- **Pipeline loop** checks `_do_interim_asr()` every ~1s while VAD is accumulating speech (buffer > `interim_interval`)
- **Sentence splitting** via `pysbd` library (rule-based, 23 languages, ~0.08ms/call), with comma fallback:
  - pysbd handles sentence-ending punctuation (。！？!?.) and language-specific rules (Mr./Dr., abbreviations)
  - CJK comma `、` fallback at 25-char threshold (Japanese clause separator)
  - Western comma `,，;；` fallback at 60-char threshold for long unsplit sentences
  - Both fallbacks require `before > 15 chars` and `after > 3 chars` to avoid fragments
- **Trim with safety margin**: After committing sentences, proportional audio trim + 0.3s margin to reduce echo; keeps ≥0.5s remainder
- **Echo dedup**: `_strip_committed_overlap()` removes re-recognized text by matching committed tail against new text prefix
- **Short utterance buffering**: Fragments ≤8 alnum chars buffered in `_interim_pending`, prepended to next sentence

### VAD Behavior

- **FireRedVAD mode**: `vad_firered.py` lazy-imports `fireredvad` and adapts LiveTranslate 32ms float32 chunks into FireRedVAD 25ms/10ms streaming frames. FireRedVAD only supplies speech confidence; `VADProcessor` still owns segmentation, silence handling, backtrack splitting, incremental ASR buffer state, and ASR queue contract.
- **Progressive silence**: Buffer越长接受越短的停顿切分 (<3s=full, 3-6s=half, 6-10s=quarter of silence_limit)
- **Adaptive silence**: Tracks recent pause durations, sets threshold to P75 × 1.2, auto-adjusts between 0.3s~2.0s
- **Backtrack split**: Max duration时回溯smoothed confidence history找最低谷切分，remainder保留到下一段
- **Speech density filter**: `_flush_segment()` discards segments where <25% of chunks are above confidence threshold
- **Short segment merge**: Segments below `min_speech_duration` are NOT discarded — VAD does a soft-reset (`_is_speaking=False`) but keeps the buffer, which naturally merges with the next speech onset
- **Trimmed segment handling**: `_was_trimmed` flag (set by `trim_front`) ensures interim ASR remainders are `force_flush()`ed instead of being dropped by min_speech check

### Key Patterns

- `torch` must be imported before PyQt6 to avoid DLL conflicts on Windows (PyTorch 2.9.0+ bug, see `main.py` and [pytorch#166628](https://github.com/pytorch/pytorch/issues/166628))
- Cache env vars set at module level in `main.py` before `import torch` to ensure `TORCH_HOME` is respected
- Deferred initialization: ASR model loading and settings application happen via `QTimer.singleShot(100)` after UI is shown to prevent startup freeze
- `make_openai_client()` in `translator.py` is the single shared function for proxy-aware OpenAI client creation (used by both translator and benchmark)
- `create_app_icon()` in `main.py` generates the app icon; set globally via `app.setWindowIcon()` so all windows inherit it
- Model cache detection (`is_asr_cached`, `get_local_model_path`) checks both ModelScope and HuggingFace paths to avoid redundant downloads when switching hubs
- Settings log output (`_apply_settings`) filters out `models` and `system_prompt` to avoid leaking API keys
- FunASR Nano: `asr_funasr_nano.py` does `os.chdir(model_dir)` before `AutoModel()` inside the ASR worker process, so relative paths in config.yaml (e.g. `Qwen3-0.6B`) resolve locally instead of triggering HuggingFace Hub network requests, without changing the GUI process cwd
- `Translator` defaults to 10s timeout via `make_openai_client()` to prevent API calls from hanging indefinitely
- Log window is created at startup but hidden; shown via tray menu "Show Log"
- Audio chunk duration is 32ms (512 samples at 16kHz), matching Silero VAD's native window size for minimal latency. FireRedVAD keeps this capture cadence and internally rolls 25ms windows with 10ms shifts.
- FireRedVAD model discovery expects the official `Stream-VAD` directory containing `cmvn.ark` and `model.pth.tar`; the model is optional and not part of first-launch required downloads.
- FireRedVAD input scaling is explicit: LiveTranslate keeps ASR/VAD buffers as float32 PCM, while `vad_firered.py` clips to [-1, 1] and multiplies by 32768 before calling `detect_frame()`.
- FunASR `disable_pbar=True` required in all `generate()` calls — tqdm crashes in GUI process when flushing stderr
- ASR engine lifecycle: the GUI process never instantiates `ASREngine`, `FunASREngine`, or `AnimeWhisperEngine` directly. It owns an `ASRClient`; each worker process owns one concrete backend/model. Engine/model/device changes shut down the current worker first, then start a new worker. On target load failure, the saved previous worker config is used to restore ASR.
- CrispASR is treated as a ggml C++ runtime hub with GGUF/bin single-file weights. The GUI process never imports `crispasr`; `asr_crispasr.py` imports the Python binding only inside the ASR worker process and adapts results to the common ASR dict.
- sherpa-onnx is treated as a local ONNX ASR runtime. The GUI process never imports `sherpa_onnx`; `asr_sherpa_onnx.py` imports it lazily inside the ASR worker after any CUDA device mapping is finalized. Offline families use `OfflineRecognizer`; `online_transducer` uses `OnlineRecognizer` as a VAD segment wrapper, not true partial streaming.
- parakeet.cpp is treated as a worker-only native C API backend. The GUI process never loads the parakeet DLL; `asr_parakeet_cpp.py` uses `ctypes` only inside the ASR worker, loads one local `.gguf` once, and returns the common ASR dict.
- parakeet.cpp model scanning only accepts known official GGUF filename prefixes or explicit sidecar metadata (`family=parakeet_cpp` / `architecture=parakeet`). CrispASR scanning excludes recognized parakeet GGUF files to avoid `.gguf` list pollution.
- parakeet.cpp C API returned strings must be released with `parakeet_capi_free_string()`, and loaded contexts must be released with `parakeet_capi_free()`.
- parakeet.cpp runtime DLL paths are added only inside the worker with `os.add_dll_directory()`. CUDA runtime may require the matching `cudart-parakeet-bin-win-cuda-x64.zip` DLLs beside the parakeet runtime.
- CrispASR packaging has two parts that must stay aligned: `pyproject.toml` installs the pure-Python binding from the CrispASR release tag via `uv sync`, while `install.ps1` and the portable `bootstrap.ps1` download the matching prebuilt Windows `libcrispasr` DLL runtime from GitHub Releases. Do not replace this with plain `crispasr` from PyPI; that package name may not exist in the configured registry and the binding alone does not include `crispasr.dll`.
- CrispASR native runtime install prefers `libcrispasr-windows-x86_64-cuda.tar.gz` on NVIDIA systems and falls back to `libcrispasr-windows-x86_64.tar.gz`. DLLs are copied next to the installed `crispasr` package so the worker can load them without a global PATH edit.
- Whisper (ctranslate2) only accepts `device="cuda"` not `"cuda:0"`; device index passed via `device_index` param. Parsed from combo text like `"cuda:0 (RTX 4090)"` in `_switch_asr_engine`
- ASR text density filter: segments ≥2s producing ≤3 alnum characters are discarded as noise
- Settings file uses atomic write (write to `.tmp` then `os.replace`) to prevent corruption on crash
- `PipelineController.stop()` joins capture/ASR queue threads before flushing VAD to prevent concurrent segment processing
- Cancelled ASR download leaves the current worker running; failed target worker load attempts to restore the previous worker config
- `Translator._build_system_prompt` catches format errors in user prompt templates, falls back to DEFAULT_PROMPT
- Translation prompt presets: `PROMPT_PRESETS` in `translator.py` (daily/esports/anime), selectable via control panel combo
- `translate_iter()` is a generator that yields accumulated partial text for streaming UI; `translate()` is the blocking equivalent
- Streaming UI: `update_streaming_signal` → `ChatMessage.update_streaming()` with 50ms QTimer throttle; `set_translation()` finalizes
- `RepetitionError` raised when model output contains repetition loops (pattern length 8+); caught in `_translate_async`, shows user-facing warning
- Changelog: `i18n/CHANGELOG_{lang}.md` files rendered as HTML in Settings → Changelog tab via `_load_latest_changelog()`

## Language & Style

- Respond in Chinese
- Code comments in English only where critical
- Commit messages without Co-Authored-By
