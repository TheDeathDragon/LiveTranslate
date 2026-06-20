# 更新日志

## 2026-06-20
- 新增 ASR 引擎: sherpa-onnx (ONNX OfflineRecognizer), 复用现有 VAD 切段、ASR worker 子进程和翻译管线
- 设置面板可选择 sherpa-onnx 本地模型目录、provider 和线程数; 模型扫描支持 SenseVoice / Paraformer / Moonshine / Whisper 目录结构，以及包含 `encoder.onnx` / `decoder.onnx` / `joiner.onnx` / `tokens.txt` 的 online transducer snapshot
- sherpa-onnx online transducer 模型当前通过 `OnlineRecognizer` 对 VAD 切段做整段识别; 暂未启用 partial streaming ASR
- sherpa-onnx 默认不内置下载器: 将官方模型解压到 `models/` 下任意子目录后在设置中刷新选择
- 安装脚本默认安装 CPU 版 sherpa-onnx runtime, 可通过 `-SherpaOnnxRuntime cuda11/cuda12` 安装 CUDA wheel
- 新增「远程 Whisper」ASR 引擎: 把语音识别外包到带 GPU 的另一台机器 (附 `asr_server.py` 服务端), 无 GPU 的机器也能实时识别
- 新增「WebID 身份核验」翻译提示词预设: 针对视频核验场景调优用词
- ASR 改为子进程隔离运行: 识别故障不再拖垮界面进程
- 字幕窗口支持鼠标穿透 (#28): 字幕设置里新增开关, 托盘菜单可「字幕鼠标穿透」快速切换, 开启后点击直达背后窗口 (此时中键拖动失效, 先关穿透再移动)

## 2026-06-19
- 修复 CrispASR 启动时报 `No module named 'crispasr'`: `pyproject.toml` 新增 CrispASR Python binding 依赖, 固定到 GitHub Releases `v0.7.2`
- 安装脚本新增 CrispASR 原生运行时安装: 从 Releases 下载预编译 `libcrispasr` Windows DLL, 放入已安装的 `crispasr` 包目录
- NVIDIA 环境优先安装 CUDA 版 CrispASR runtime, 下载或安装失败时自动回退 CPU runtime
- Portable 首次启动 bootstrap 同步安装 CrispASR binding 与原生 DLL, 避免发布包缺少 `crispasr.dll`

## 2026-05-10
- 新增「导出到文件」: 支持原文 / 译文 / 原文+译文 三种格式, 悬浮窗右键菜单与托盘菜单均可触发
- 新增「转录持久化」(默认开启): 每次会话自动在 `transcripts/` 下创建 3 份文件 (原文 / 译文 / 全部), 每段识别结果实时追加写入, 不再受悬浮窗 50 条上限丢失早期文本
- 设置面板「缓存」标签新增「转录持久化」分组: 开关 + 打开转录文件夹按钮
- 新增内存占用上限保护: RSS 超过 4096MB 时弹托盘通知提示重启 (ASR 后端存在 native 侧工作区/缓存, Python GC 与 `torch.cuda.empty_cache` 不一定能回收)
- 日志新增 `MEM[asr#/tick]` 行: 每次真实 ASR 调用后输出 RSS / GPU(alloc/reserved) / 音频时长 / overlay 消息数 / VAD buffer 长度, 便于排查内存问题

## 2026-04-20
- 移除 Qwen3-ASR 引擎 (ONNX + GGUF 方案兼容性较差, 相关模型文件和 llama.cpp 运行时依赖一并清理)
- 模型配置新增「高级参数」: `temperature` / `top_p` / `max_tokens` / `frequency_penalty` / `presence_penalty` / `seed`, 每项独立「覆盖」开关, 未勾选时使用服务端默认值
- 模型配置新增 `extra_body` (JSON): 供应商专有参数透传, 如 `thinking_budget`、`reasoning_effort` 等, 保存时自动校验 JSON 格式
- 修复 Anime-Whisper 模型未缓存时下载对话框静默无动作的问题
- 修复设置面板「更新日志」Tab 空白 (正则匹配 H3 但文件用 H2, 3 月加入功能起就失效)

## 2026-04-18
- 新增 ASR 引擎: Anime-Whisper (litagin/anime-whisper), 日语动画/Galgame 特化, 擅长识别喘息/叹息等非语言发声
- 修复 HF 缓存检测误判: 下载中断留下的空目录不再被认为"已缓存"

## 2026-03-31
- 管道线程拆分: 原 capture+VAD+ASR 单线程改为 capture 与 ASR 两条独立线程, 长句 ASR 不再阻塞 RMS/VAD 条实时更新
- ASR 调度新增有界队列 (16 段), 队列满时丢弃最旧的 interim 段, 避免堆积导致延迟

## 2026-03-26
- 默认翻译提示词优化: 新增 ASR 纠错规则 (识别错别字/同音字时尽量按上下文修正)、流畅性规则 (避免逐字直译)

## 2026-03-25
- 样式标签新增「重置窗口位置」按钮: 字幕窗口回到 (100,100), 主悬浮窗回到屏幕右下角
- 字幕窗口默认位置从屏幕底部居中改为 (100,100), 最小高度调整为 200px
- 窗口位置恢复时校验是否在可见屏幕范围内 (使用 availableGeometry 排除任务栏), 高度变化后 clamp 到屏幕边界

## 2026-03-24
- 字幕窗口: 超长文本自动换行显示(不再分段), 背景高度平滑动画, 文字渲染 Pixmap 缓存
- 主悬浮窗/字幕窗口: 位置和大小自动记忆, 重启后恢复
- 主悬浮窗: 精简模式切换动画
- 设置: 移除配置文件 key 白名单限制

## 2026-03-23
- 品牌更名 LiveTrans → LiveTranslate
- 模型配置新增: 流式传输、结构化输出、上下文数、禁用思考(默认开启)
- 翻译结果流式逐字显示
- 提示词优化: 禁止多候选翻译, 编辑即时生效
- 模型重复输出检测与提示
- ASR 引擎标注[准确]/[快速]
- 设置面板新增更新日志 Tab
