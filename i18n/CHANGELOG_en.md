# Changelog

## 2026-11-17
- New Qwen3-ASR 1.7B engine: native transformers implementation with 30 languages plus auto-detect, ~4.1GB of VRAM, better Japanese and Korean accuracy than Whisper
- New BuzzASR Korean engine: Korean-specific, weights converted to CTranslate2 float16 and served through the existing Whisper inference path

## 2026-08-17
- Incremental ASR sentence segmentation switched from PySBD to yasbd-lib (#37): API-compatible with no behavior change, adds native rules for Korean and 16 more languages (Korean previously fell back to English rules), and is much faster on long text
- Fixed empty translations showing up as untranslated same-language text with DeepSeek (#38): DeepSeek defaults to thinking mode ON and only accepts thinking.type=disabled to turn it off, while the previously sent enable_thinking=false is a Qwen-style flag it ignores; the model edit dialog now has a "Disable thinking" style selector (auto-detect / DeepSeek·Volcano Ark·GLM / Qwen·DashScope·SiliconFlow / self-hosted vLLM·SGLang / OpenAI·Grok reasoning_effort / do not send), auto-detect no longer sends unknown parameters to official OpenAI-style endpoints that reject them, and a diagnostic warning is logged when reasoning burns the whole token budget and returns an empty completion
- Fixed settings windows overflowing the screen at 150%+ DPI scaling, which left the OK/Cancel buttons unreachable (#39): the model edit dialog and control panel tabs now scroll when needed, and window heights are clamped to the screen's usable area
- Install and update no longer write download caches to the system drive: the portable build keeps the uv cache and managed Python inside the app folder, and the install/update scripts keep pip's cache and temp files inside the project folder, cleaned up after a successful install

## 2026-07-11
- Fixed Fun-ASR-Nano first load: the Qwen3-0.6B weight download could be killed by the 180s worker startup timeout (#32); weights are now fetched up-front in the model download phase, so worker startup no longer waits on large downloads
- Fixed model detection misses caused by the ModelScope 1.38+ cache location change (#32, #33): all cache layouts across SDK versions are recognized, so upgrading the SDK no longer re-downloads existing models
- Qwen3-0.6B weights can now be downloaded from ModelScope; ModelScope-hub users are no longer forced through HuggingFace (#33)

## 2026-06-20
- New "Remote Whisper" ASR engine: offload speech recognition to a separate GPU machine (ships `asr_server.py` server), so a box without a GPU can still transcribe in real time
- New "WebID / ID Verify" translation prompt preset, tuned for video identity-verification calls
- ASR now runs in an isolated subprocess: the worker auto-restarts on crash/timeout and recycles when memory grows past a threshold, so a recognition failure no longer drags down the UI
- Subtitle window mouse click-through (#28): a toggle in the subtitle settings plus a "Subtitle Click-through" tray shortcut; when on, clicks pass to the window behind (middle-click drag is disabled while on — turn it off to reposition)

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
