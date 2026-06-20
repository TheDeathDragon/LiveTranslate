# Changelog

## 2026-06-20
- New ASR engine: sherpa-onnx (ONNX OfflineRecognizer), reusing the existing VAD segmentation, ASR worker subprocess, and translation pipeline
- Settings panel can select a local sherpa-onnx model directory, provider, and thread count; local scanning supports SenseVoice / Paraformer / Moonshine / Whisper directory layouts, plus online transducer snapshots with `encoder.onnx` / `decoder.onnx` / `joiner.onnx` / `tokens.txt`
- sherpa-onnx online transducer models use `OnlineRecognizer` as a VAD segment wrapper in this phase; partial streaming ASR is not enabled yet
- sherpa-onnx does not use a built-in downloader in this phase: extract official models anywhere under `models/`, then refresh and select them in settings
- Installer now installs the CPU sherpa-onnx runtime by default, with `-SherpaOnnxRuntime cuda11/cuda12` for CUDA wheels
- New "Remote Whisper" ASR engine: offload speech recognition to a separate GPU machine (ships `asr_server.py` server), so a box without a GPU can still transcribe in real time
- New "WebID / ID Verify" translation prompt preset, tuned for video identity-verification calls
- ASR now runs in an isolated subprocess, so recognition failures no longer drag down the UI process
- Subtitle window mouse click-through (#28): a toggle in the subtitle settings plus a "Subtitle Click-through" tray shortcut; when on, clicks pass to the window behind (middle-click drag is disabled while on — turn it off to reposition)

## 2026-06-19
- Fix CrispASR startup failure with `No module named 'crispasr'`: added the CrispASR Python binding to `pyproject.toml`, pinned to GitHub Releases `v0.7.2`
- Installer now downloads the prebuilt Windows `libcrispasr` DLL runtime from Releases and places it next to the installed `crispasr` package
- NVIDIA systems prefer the CUDA CrispASR runtime and automatically fall back to the CPU runtime if download or installation fails
- Portable first-run bootstrap now installs both the CrispASR binding and native DLLs, preventing release builds from missing `crispasr.dll`

## 2026-05-10
- New "Export to file" menu: original / translation / combined formats, accessible from overlay right-click menu and tray menu
- New "Transcript persistence" (enabled by default): each session creates 3 files under `transcripts/` (original / translation / combined), appended in real time per segment — no longer bounded by the 50-message overlay cap
- Settings panel "Cache" tab: added "Transcript persistence" group with toggle and open-folder button
- Memory ceiling protection: tray notification shown once when RSS exceeds 4096MB, advising restart (ASR backends keep native-side workspaces/caches that Python GC and `torch.cuda.empty_cache` may not reclaim)
- New `MEM[asr#/tick]` log lines: per-ASR-call RSS / GPU (alloc/reserved) / audio duration / overlay message count / VAD buffer length for memory diagnostics

## 2026-04-20
- Removed Qwen3-ASR engine (ONNX + GGUF hybrid had compatibility issues; model files and llama.cpp runtime dependencies cleaned up)
- Model config: new "Advanced Parameters" group — `temperature`, `top_p`, `max_tokens`, `frequency_penalty`, `presence_penalty`, `seed`, each gated by an independent "Override" checkbox (unchecked = use server default)
- Model config: new `extra_body` (JSON) field for provider-specific parameters (e.g. `thinking_budget`, `reasoning_effort`), validated on save
- Fix: Anime-Whisper download dialog was a silent no-op when the model was not cached
- Fix: settings panel "Changelog" tab showed blank (regex expected H3 but files used H2 headings — broken since the tab was added in March)

## 2026-04-18
- New ASR engine: Anime-Whisper (litagin/anime-whisper), Japanese-only, specialized for anime / galgame speech (breaths, sighs, non-verbal sounds)
- Fix HF cache detection: aborted downloads leaving empty dirs no longer trigger false "cached" state

## 2026-03-31
- Pipeline thread split: capture+VAD+ASR was a single thread; now capture and ASR run on separate threads, so long-segment ASR no longer blocks live RMS/VAD bar updates
- ASR scheduling uses a bounded queue (16 segments); oldest interim segments are dropped when full to prevent backlog-induced latency

## 2026-03-26
- Default translation prompt improvements: added ASR error-correction rules (fix typos/homophones from context) and fluency rules (avoid word-for-word literal translation)

## 2026-03-25
- Style tab: new "Reset window positions" button — subtitle window returns to (100,100), overlay returns to bottom-right of screen
- Subtitle window default position changed from bottom-center to (100,100); minimum height adjusted to 200px
- Window position restore now validates against the visible screen area (`availableGeometry` excludes the taskbar); height changes clamp to screen bounds to prevent windows from being pushed off-screen

## 2026-03-24
- Subtitle window: auto word-wrap for long text (no more split segments), smooth height animation, pixmap render cache
- Overlay & subtitle window: position/size persistence across restarts
- Overlay: compact mode toggle animation
- Settings: removed valid-key whitelist restriction

## 2026-03-23
- Rebranded LiveTrans → LiveTranslate
- Model config: streaming toggle, structured output, context count, disable thinking (default on)
- Streaming translation display in overlay
- Prompt improvements: no alternatives, instant apply
- Repetition loop detection and user warning
- ASR engine labels: Accurate / Fast
- Changelog tab in settings panel
