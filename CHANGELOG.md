# Changelog

## [0.2.0] — 2026-03-18

### New Features

- **Dynamic batching for Qwen3 models**: New `--batch-window-ms` and `--max-batch-size` CLI flags enable batched inference for `qwen3_tts` models. Workers collect multiple requests within the window and dispatch them as a single `batch_generate` call, improving throughput under concurrent load
- **`TTSEngine.generate_batch()`**: New method on `TTSEngine` that runs a batch of queued requests in one forward pass, with parallel audio encoding via `ThreadPoolExecutor` and graceful fallback to sequential for streaming or non-batch-aware requests
- **`_BatchParams` dataclass**: Carries per-request inference parameters from the HTTP handler through to the batching worker, enabling the worker to reconstruct batch inputs without re-parsing the original request
- **`supports_batch` property**: `TTSEngine` exposes a `supports_batch` flag (true for `qwen3_tts` models with `batch_generate`); `RequestQueue.start_workers()` uses this to select between standard and batching worker coroutines per engine
- **MLX Metal cache release**: `_release_mlx_cache()` calls `mx.metal.clear_cache()` after every inference request (both full and streaming), bounding per-worker memory growth under sustained concurrent load
- **`quantize_fish_s2.py` script**: New script to quantize Fish Audio S2 Pro BF16 → 8-bit in-memory via `mlx.nn.quantize()`, with correct weight key handling to avoid `apply_quantization` predicate mismatch
- **Default seed**: `TTSRequest.seed` now defaults to `42` instead of `None`, giving deterministic output by default without requiring the caller to set a seed

### Changes

- `RequestQueue.__init__` now accepts `batch_window_ms` and `max_batch_size` parameters; `start_workers()` routes each engine to either `_worker` or `_batch_worker` based on `supports_batch` and `batch_window_ms`
- `ServerConfig` gains `batch_window_ms` (default `0`) and `max_batch_size` (default `8`) fields; both are wired through `parse_config()` with env var fallbacks `TTS_BATCH_WINDOW_MS` / `TTS_MAX_BATCH_SIZE`
- `server.py` attaches `_BatchParams` to each `_run_inference` closure so the batching worker can extract parameters without accessing the original request object
- Flow-matching parameters (`cfg_scale`, `flow_steps`, `sigma`) are now skipped for `qwen3_tts` model type in `_run_model`, preventing unsupported kwargs from being passed to Qwen3's `generate`

### Fixes

- Fixed `_run_model` passing flow-matching kwargs (`cfg_scale`, `ddpm_steps`, `sigma`) to Qwen3 models, which do not accept them

---

## 更新日志

## [0.2.0] — 2026-03-18

### 新功能

- **Qwen3 模型动态批处理**：新增 `--batch-window-ms` 和 `--max-batch-size` 启动参数，为 `qwen3_tts` 模型启用批量推理。Worker 在时间窗口内收集多个请求，合并为一次 `batch_generate` 调用，显著提升高并发吞吐量
- **`TTSEngine.generate_batch()`**：新方法，将一批队列请求合并为单次前向传播，支持并行音频编码（`ThreadPoolExecutor`），对流式或不支持批处理的请求自动回退为顺序执行
- **`_BatchParams` 数据类**：将每个请求的推理参数从 HTTP 处理函数传递到批处理 Worker，使 Worker 无需重新解析原始请求即可重建批量输入
- **`supports_batch` 属性**：`TTSEngine` 新增 `supports_batch` 标志（对具有 `batch_generate` 的 `qwen3_tts` 模型为 `True`）；`RequestQueue.start_workers()` 据此为每个引擎选择标准或批处理 Worker 协程
- **MLX Metal 缓存释放**：每次推理请求（全量和流式）结束后调用 `_release_mlx_cache()` 执行 `mx.metal.clear_cache()`，防止持续并发负载下单 Worker 内存持续增长
- **`quantize_fish_s2.py` 脚本**：新增脚本，通过 `mlx.nn.quantize()` 将 Fish Audio S2 Pro BF16 量化为 8-bit，正确处理权重键名以避免 `apply_quantization` 谓词匹配失败
- **默认随机种子**：`TTSRequest.seed` 默认值从 `None` 改为 `42`，无需调用方显式设置即可获得确定性输出

### 变更

- `RequestQueue.__init__` 新增 `batch_window_ms` 和 `max_batch_size` 参数；`start_workers()` 根据 `supports_batch` 和 `batch_window_ms` 为每个引擎选择 `_worker` 或 `_batch_worker`
- `ServerConfig` 新增 `batch_window_ms`（默认 `0`）和 `max_batch_size`（默认 `8`）字段，通过 `parse_config()` 接入环境变量 `TTS_BATCH_WINDOW_MS` / `TTS_MAX_BATCH_SIZE`
- `server.py` 为每个 `_run_inference` 闭包附加 `_BatchParams`，使批处理 Worker 无需访问原始请求对象即可提取参数
- `_run_model` 中流匹配参数（`cfg_scale`、`flow_steps`、`sigma`）现在对 `qwen3_tts` 模型类型跳过传递，避免向 Qwen3 的 `generate` 传入不支持的关键字参数

### 修复

- 修复 `_run_model` 向 Qwen3 模型传递流匹配参数（`cfg_scale`、`ddpm_steps`、`sigma`）导致报错的问题，Qwen3 不接受这些参数

---

## [0.1.0] — 2026-03-14

### New Features

- **Multi-model support**: Added support for Ming-omni-tts-0.5B (MLX-native, ~2.8 GB) alongside Fish Audio S2 Pro BF16
- **VoiceDesign / `instruct` field**: New `instruct` parameter in `TTSRequest` for voice style description, enabling Qwen3-TTS VoiceDesign models; auto-applies default style for `qwen3_tts` model type
- **Flow-matching model parameters**: New `cfg_scale`, `flow_steps`, and `sigma` fields in `TTSRequest` for Ming-omni and other flow-matching architectures
- **Extended sampling controls**: Added `top_k` parameter alongside existing `top_p` and `temperature`
- **Model type detection**: `TTSEngine` now reads `model_type` from the loaded model to apply model-specific inference logic
- **`trust_remote_code=True`**: `load_model` now passes `trust_remote_code=True` to support models with custom code (e.g. Ming-omni)

### Changes

- Default `temperature` changed from `0.8` → `0.0` (deterministic by default)
- Default `top_p` changed from `0.8` → `0.7`
- `TTSEngine._run_model` now passes `top_p`, `top_k`, `cfg_scale`, `flow_steps` (`ddpm_steps`), `sigma`, and `instruct` to `model.generate`

### Fixes

- Fixed startup failure when running with system Python (conda `base`): documented that `uv run` or `.venv` activation is required

---

## 更新日志

## [0.1.0] — 2026-03-14

### 新功能

- **多模型支持**：新增对 Ming-omni-tts-0.5B（MLX 原生格式，约 2.8GB）的支持，与 Fish Audio S2 Pro BF16 并列可用
- **VoiceDesign / `instruct` 字段**：`TTSRequest` 新增 `instruct` 参数，用于描述声音风格，支持 Qwen3-TTS VoiceDesign 类模型；`qwen3_tts` 模型类型会自动应用默认风格
- **流匹配模型参数**：`TTSRequest` 新增 `cfg_scale`、`flow_steps`、`sigma` 字段，适配 Ming-omni 等流匹配架构
- **扩展采样控制**：新增 `top_k` 参数，与现有 `top_p`、`temperature` 配合使用
- **模型类型检测**：`TTSEngine` 启动时读取已加载模型的 `model_type`，据此应用模型专属推理逻辑
- **`trust_remote_code=True`**：`load_model` 调用时传入 `trust_remote_code=True`，支持含自定义代码的模型（如 Ming-omni）

### 变更

- 默认 `temperature` 从 `0.8` 改为 `0.0`（默认确定性输出）
- 默认 `top_p` 从 `0.8` 改为 `0.7`
- `TTSEngine._run_model` 现在将 `top_p`、`top_k`、`cfg_scale`、`flow_steps`（映射为 `ddpm_steps`）、`sigma`、`instruct` 传递给 `model.generate`

### 修复

- 修复使用系统 Python（conda `base` 环境）启动时报 `ModuleNotFoundError: No module named 'mlx_audio'` 的问题：文档中明确说明需使用 `uv run` 或激活 `.venv`
