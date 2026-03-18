# MLX Audio API

[English](#english) | [中文](#chinese)

---

<a name="english"></a>

## English

A local text-to-speech inference service built on top of [mlx-audio](https://github.com/Blaizzy/mlx-audio) — the core inference engine that handles MLX-based TTS model loading and audio generation. This project wraps mlx-audio with a production-ready HTTP API layer: multi-worker concurrency, request queuing, zero-shot voice cloning, streaming output, and Bearer token auth. Optimized for Apple Silicon (M1/M2/M3).

### What's New in v0.2.0 (2026-03-18)

- **Dynamic batching for Qwen3 models** — new `--batch-window-ms` / `--max-batch-size` flags let workers collect multiple requests and dispatch them as a single `batch_generate` call, improving throughput under concurrent load
- **MLX Metal cache release** — `mx.metal.clear_cache()` is called after every inference request, bounding per-worker memory growth under sustained load
- **`scripts/quantize_fish_s2.py`** — new script to quantize Fish Audio S2 Pro BF16 → 8-bit locally
- **Default seed `42`** — `TTSRequest.seed` now defaults to `42` for deterministic output without explicit configuration
- **Flow-matching params isolated** — `cfg_scale`, `flow_steps`, `sigma` are no longer passed to Qwen3 models (they don't accept them)

### Requirements

- macOS (Apple Silicon — M1/M2/M3)
- Python 3.10+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager

### Quick Start

**1. Install dependencies**

```bash
uv sync

# Activate the virtual environment (optional — you can also use `uv run` prefix instead)
source .venv/bin/activate
```

> Always use `uv run` or activate `.venv` before running. Using the system Python (e.g. conda `base`) will fail with `ModuleNotFoundError: No module named 'mlx_audio'`.

**2. Download a model**

```bash
# Fish Audio S2 Pro BF16 (high quality, 80+ languages, ~9.6 GB)
# https://huggingface.co/mlx-community/fish-audio-s2-pro-bf16
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('mlx-community/fish-audio-s2-pro-bf16', local_dir='./models/fish-audio-s2-pro-bf16')
"

# Fish Audio S2 Pro 8-bit (pre-quantized, ~6.3 GB — recommended over BF16 for lower memory)
# https://huggingface.co/cs2764/fish-audio-s2-pro-8bit-mlx
hf download cs2764/fish-audio-s2-pro-8bit-mlx --local-dir ./models/fish-audio-s2-pro-8bit

# Qwen3-TTS-12Hz-1.7B-VoiceDesign 8bit (lightweight, voice style control, ~2.3 GB)
# https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit
hf download mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit --local-dir ./models/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit

# Qwen3-TTS-12Hz-1.7B-Base 8bit (lightweight, voice cloning via references, ~2.3 GB)
# https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit
hf download mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit --local-dir ./models/Qwen3-TTS-12Hz-1.7B-Base-8bit
```

**3. Start the server**

```bash
# Fish Audio S2 Pro BF16
uv run python -m src --model-path ./models/fish-audio-s2-pro-bf16 --num-workers 4 --timeout 600

# Fish Audio S2 Pro 8bit (locally quantized, ~6.3 GB, lower memory usage)
uv run python -m src --model-path ./models/fish-audio-s2-pro-8bit --num-workers 4 --timeout 600

# Qwen3 Base (voice cloning, consistent timbre)
uv run python -m src --model-path ./models/Qwen3-TTS-12Hz-1.7B-Base-8bit --num-workers 8 --timeout 600

# Qwen3 VoiceDesign (voice style via instruct)
uv run python -m src --model-path ./models/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit --num-workers 8 --timeout 600
```

**Audiobook / high-throughput mode (Qwen3 Base + voice cloning)**

Recommended for audiobook workloads: consistent timbre via reference audio, high concurrency via multi-worker:

```bash
uv run python -m src \
  --model-path ./models/Qwen3-TTS-12Hz-1.7B-Base-8bit \
  --num-workers 16 \
  --timeout 600 \
  --max-queue-size 64
```

**Dynamic batching mode (Qwen3 only)**

Collects multiple requests within a time window and dispatches them as a single batch, improving throughput when many short requests arrive simultaneously:

```bash
uv run python -m src \
  --model-path ./models/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit \
  --num-workers 4 \
  --batch-window-ms 50 \
  --max-batch-size 8 \
  --timeout 600
```

> Note: dynamic batching pads all sequences in a batch to the longest one. For audiobook workloads with uneven text lengths, plain multi-worker mode (no batching) is usually faster.

Server listens on `http://0.0.0.0:8013` by default. Verify with:

```bash
curl http://127.0.0.1:8013/v1/health
# {"status":"ok"}
```

### Configuration

| CLI arg | Env var | Default | Description |
|---|---|---|---|
| `--model-path` | `TTS_MODEL_PATH` | `./models/fish-audio-s2-pro-bf16` | Model directory |
| `--host` | `TTS_HOST` | `0.0.0.0` | Bind address |
| `--port` | `TTS_PORT` | `8013` | Port |
| `--num-workers` | — | `1` | Parallel inference processes |
| `--api-key` | `TTS_API_KEY` | None | Enable Bearer token auth |
| `--max-queue-size` | `TTS_MAX_QUEUE_SIZE` | `32` | Max queued requests (503 when full) |
| `--timeout` | `TTS_TIMEOUT` | `300` | Per-request timeout in seconds |
| `--batch-window-ms` | `TTS_BATCH_WINDOW_MS` | `0` | Max wait time (ms) to fill a batch (Qwen3 only, 0=disabled) |
| `--max-batch-size` | `TTS_MAX_BATCH_SIZE` | `8` | Max requests per batch (Qwen3 only) |

### API Usage

**Basic TTS**

```bash
curl -X POST http://127.0.0.1:8013/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, welcome to the speech synthesis service."}' \
  --output output.wav
```

**Output formats** — `wav` (default), `mp3`, `pcm`

```bash
curl -X POST http://127.0.0.1:8013/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "format": "mp3"}' \
  --output output.mp3
```

**Voice cloning**

```python
import base64, requests

audio_b64 = base64.b64encode(open('samples/0流萤.wav', 'rb').read()).decode()
ref_text  = open('samples/0流萤.txt').read().strip()

resp = requests.post('http://127.0.0.1:8013/v1/tts', json={
    'text': 'Clone this voice and say something.',
    'references': [{'audio': audio_b64, 'text': ref_text}],
})
open('cloned.wav', 'wb').write(resp.content)
```

**VoiceDesign (Qwen3-TTS-12Hz-1.7B-VoiceDesign)**

> **Note**: The `instruct` field is **required** when using `Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit`. Omitting it will result in a default/random voice style being applied.

Use the `instruct` field to describe the desired voice style in English. No reference audio needed.

```bash
curl -X POST http://127.0.0.1:8013/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, how can I help you today?", "instruct": "A cheerful young female voice with high pitch"}' \
  --output output.wav
```

Some example `instruct` values:
- `"A cheerful young female voice with high pitch"`
- `"A calm middle-aged male voice with deep tone"`
- `"An energetic young male voice, fast pace"`
- `"A gentle elderly female voice, slow and clear"`

**Streaming output**

```python
import requests

resp = requests.post('http://127.0.0.1:8013/v1/tts',
    json={'text': 'Streaming test.', 'streaming': True, 'format': 'wav'},
    stream=True)

with open('streamed.wav', 'wb') as f:
    for chunk in resp.iter_content(4096):
        if chunk:
            f.write(chunk)
```

**With authentication**

```bash
uv run python -m src --model-path ./models/fish-audio-s2-pro-bf16 --api-key <YOUR_API_KEY>

curl -X POST http://127.0.0.1:8013/v1/tts \
  -H "Authorization: Bearer <YOUR_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"text": "Authenticated request"}' \
  --output output.wav
```

### TTS Request Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text` | string | required | Text to synthesize |
| `format` | string | `"wav"` | Output format: `wav`, `mp3`, `pcm` |
| `references` | array | `[]` | Inline base64 reference audio for voice cloning |
| `reference_id` | string | null | Server-side preloaded reference voice ID |
| `instruct` | string | null | Voice style description (Qwen3 VoiceDesign models) |
| `streaming` | bool | false | Chunked streaming output (WAV only) |
| `seed` | int | `42` | Random seed for reproducible output |
| `temperature` | float | `0.0` | Sampling temperature (0.0–1.0) |
| `top_p` | float | `0.7` | Top-p sampling (0.1–1.0) |
| `top_k` | int | `30` | Top-k sampling (0–1000) |
| `repetition_penalty` | float | `1.1` | Repetition penalty (0.9–2.0) |
| `max_new_tokens` | int | `1024` | Max tokens to generate |
| `chunk_length` | int | `200` | Text chunk length (100–300) |
| `cfg_scale` | float | `2.0` | CFG scale for flow-matching models |
| `flow_steps` | int | `10` | Diffusion steps for flow-matching models |
| `sigma` | float | `0.25` | Sigma for flow-matching models |

### Reference Voice Management

```bash
# Add
curl -X POST http://127.0.0.1:8013/v1/references/add \
  -F "id=my-voice" -F "audio=@samples/0流萤.wav" -F "text=reference text"

# List
curl http://127.0.0.1:8013/v1/references/list

# Use
curl -X POST http://127.0.0.1:8013/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Use preset voice", "reference_id": "my-voice"}' --output output.wav

# Delete
curl -X DELETE http://127.0.0.1:8013/v1/references/delete \
  -H "Content-Type: application/json" \
  -d '{"reference_id": "my-voice"}'
```

### Supported Models

| Model | HuggingFace | Size | Notes |
|---|---|---|---|
| `fish-audio-s2-pro-bf16` | [mlx-community/fish-audio-s2-pro-bf16](https://huggingface.co/mlx-community/fish-audio-s2-pro-bf16) | 9.6 GB | High quality, 80+ languages, voice cloning |
| `fish-audio-s2-pro-8bit` | [cs2764/fish-audio-s2-pro-8bit-mlx](https://huggingface.co/cs2764/fish-audio-s2-pro-8bit-mlx) | 6.3 GB | Same quality, lower memory, pre-quantized |
| `Qwen3-TTS-12Hz-1.7B-Base-8bit` | [mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit) | 2.3 GB | Lightweight, voice cloning via `references` (ICL), consistent timbre |
| `Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit` | [mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit) | 2.3 GB | Lightweight, voice style via `instruct` field |

**Download pre-quantized S2 Pro 8-bit (recommended):**

```bash
# https://huggingface.co/cs2764/fish-audio-s2-pro-8bit-mlx
hf download cs2764/fish-audio-s2-pro-8bit-mlx --local-dir ./models/fish-audio-s2-pro-8bit
```

**Or quantize locally from BF16:**

```bash
uv run python scripts/quantize_fish_s2.py
```

**Convert other HuggingFace models:**

```bash
uv run python scripts/convert_model.py --hf-path <hf-repo-id> --mlx-path ./models/<name>
```

**Upgrade dependencies (especially mlx-audio from git):**

```bash
bash scripts/upgrade_deps.sh            # upgrade all dependencies
bash scripts/upgrade_deps.sh --mlx-only # upgrade mlx-audio only
```

### Testing

All test scripts require a running server. They are located in `tests/`.

**Concurrent API test** — sequential + concurrent requests with optional voice cloning:

```bash
# Default (with voice cloning, uses samples/0流萤.wav)
uv run python tests/test_concurrent_api.py

# No reference audio (plain TTS, faster)
uv run python tests/test_concurrent_api.py --no-ref

# Qwen3 VoiceDesign mode
uv run python tests/test_concurrent_api.py --instruct "A cheerful young female voice with high pitch" --no-ref

# Custom API URL, auth key, output dir
uv run python tests/test_concurrent_api.py --url http://127.0.0.1:8013/v1/tts --api-key <KEY> --output-dir ./my_outputs
```

**Long-text concurrent API test** — same as above but every test string is ≥ 200 characters:

```bash
uv run python tests/test_concurrent_api_long.py
uv run python tests/test_concurrent_api_long.py --no-ref
uv run python tests/test_concurrent_api_long.py --instruct "A calm professional male voice" --no-ref
```

**Text length scaling test** — sequential, one request at a time. Text length steps from 50 to 1000 characters (step 50). Records latency per request and prints a summary table:

```bash
uv run python tests/test_text_length_scaling.py
uv run python tests/test_text_length_scaling.py --no-ref
uv run python tests/test_text_length_scaling.py --min-len 100 --max-len 500 --step 100
```

### Project Structure

```
mlx-audio-api/
├── src/
│   ├── __main__.py          # Entry point; single/multi-process startup
│   ├── server.py            # FastAPI app factory, all routes, lifespan
│   ├── tts_engine.py        # TTSEngine: mlx_audio wrapper, audio encoding, batch inference
│   ├── request_queue.py     # RequestQueue + asyncio worker pool (standard + batching workers)
│   ├── reference_manager.py # File-based reference voice storage
│   ├── auth.py              # Bearer token middleware
│   ├── models.py            # Pydantic request/response models
│   └── config.py            # ServerConfig + argparse/env var parsing
├── tests/                   # API integration test scripts
├── scripts/
│   ├── convert_model.py     # HuggingFace → MLX conversion utility
│   ├── quantize_fish_s2.py  # Fish Audio S2 Pro BF16 → 8-bit quantization
│   └── upgrade_deps.sh      # Upgrade all dependencies (especially mlx-audio from git)
├── models/                  # Local model storage (gitignored)
├── samples/                 # Reference audio samples
├── references/              # Server-side stored voices (runtime, gitignored)
└── api_reference.md         # Full API documentation
```

---

<a name="chinese"></a>

## 中文

基于 [mlx-audio](https://github.com/Blaizzy/mlx-audio) 构建的本地 TTS 推理服务。mlx-audio 是核心推理引擎，负责 MLX 模型加载和音频生成；本项目在其基础上封装了生产级 HTTP API 层：多 worker 并发、请求队列、零样本声音克隆、流式输出和 Bearer Token 认证。专为 Apple Silicon（M1/M2/M3）优化。

### v0.2.0 更新内容（2026-03-18）

- **Qwen3 模型动态批处理** — 新增 `--batch-window-ms` / `--max-batch-size` 参数，Worker 在时间窗口内收集多个请求合并为一次 `batch_generate` 调用，提升高并发吞吐量
- **MLX Metal 缓存释放** — 每次推理结束后调用 `mx.metal.clear_cache()`，防止持续并发负载下内存持续增长
- **`scripts/quantize_fish_s2.py`** — 新增脚本，本地将 Fish Audio S2 Pro BF16 量化为 8-bit
- **默认随机种子 `42`** — `TTSRequest.seed` 默认值改为 `42`，无需显式设置即可获得确定性输出
- **流匹配参数隔离** — `cfg_scale`、`flow_steps`、`sigma` 不再传递给 Qwen3 模型（Qwen3 不接受这些参数）

### 环境要求

- macOS（Apple Silicon，M1/M2/M3 系列）
- Python 3.10+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) 包管理器

### 快速开始

**1. 安装依赖**

```bash
uv sync

# 激活虚拟环境（可选，后续命令也可直接用 uv run 前缀）
source .venv/bin/activate
```

> 请始终使用 `uv run` 或激活 `.venv` 后再运行。直接使用系统 Python（如 conda `base` 环境）会因缺少 `mlx_audio` 而报错。

**2. 下载模型**

```bash
# Fish Audio S2 Pro BF16（高质量，80+ 语言，约 9.6GB）
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('mlx-community/fish-audio-s2-pro-bf16', local_dir='./models/fish-audio-s2-pro-bf16')
"

# Fish Audio S2 Pro 8-bit（预量化版，约 6.3GB，推荐，显存占用更低）
# https://huggingface.co/cs2764/fish-audio-s2-pro-8bit-mlx
hf download cs2764/fish-audio-s2-pro-8bit-mlx --local-dir ./models/fish-audio-s2-pro-8bit

# Qwen3-TTS-12Hz-1.7B-VoiceDesign 8bit（轻量，支持声音风格控制，约 2.3GB）
hf download mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit --local-dir ./models/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit

# Qwen3-TTS-12Hz-1.7B-Base 8bit（轻量，通过 references 声音克隆，约 2.3GB）
hf download mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit --local-dir ./models/Qwen3-TTS-12Hz-1.7B-Base-8bit
```

**3. 启动服务**

```bash
# Fish Audio S2 Pro BF16
uv run python -m src --model-path ./models/fish-audio-s2-pro-bf16 --num-workers 4 --timeout 600

# Fish Audio S2 Pro 8bit（本地量化版，约 6.3GB，显存占用更低）
uv run python -m src --model-path ./models/fish-audio-s2-pro-8bit --num-workers 4 --timeout 600

# Qwen3 Base（声音克隆，音色稳定一致）
uv run python -m src --model-path ./models/Qwen3-TTS-12Hz-1.7B-Base-8bit --num-workers 8 --timeout 600

# Qwen3 VoiceDesign（通过 instruct 控制声音风格）
uv run python -m src --model-path ./models/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit --num-workers 8 --timeout 600
```

**有声书 / 高吞吐模式（Qwen3 Base + 声音克隆）**

```bash
uv run python -m src \
  --model-path ./models/Qwen3-TTS-12Hz-1.7B-Base-8bit \
  --num-workers 16 \
  --timeout 600 \
  --max-queue-size 64
```

**动态批处理模式（仅 Qwen3）**

在时间窗口内收集多个请求合并为一次批量推理，适合大量短文本并发场景：

```bash
uv run python -m src \
  --model-path ./models/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit \
  --num-workers 4 \
  --batch-window-ms 50 \
  --max-batch-size 8 \
  --timeout 600
```

> 注意：动态批处理会将批内所有序列 padding 到最长那条。有声书等文本长度不均匀的场景，建议使用普通多 worker 模式（不开批处理）。

服务默认监听 `http://0.0.0.0:8013`，启动后验证：

```bash
curl http://127.0.0.1:8013/v1/health
# {"status":"ok"}
```

### 启动参数

| 参数 | 环境变量 | 默认值 | 说明 |
|---|---|---|---|
| `--model-path` | `TTS_MODEL_PATH` | `./models/fish-audio-s2-pro-bf16` | 模型目录路径 |
| `--host` | `TTS_HOST` | `0.0.0.0` | 监听地址 |
| `--port` | `TTS_PORT` | `8013` | 监听端口 |
| `--num-workers` | — | `1` | 并行推理进程数 |
| `--api-key` | `TTS_API_KEY` | 无 | 启用 Bearer Token 认证 |
| `--max-queue-size` | `TTS_MAX_QUEUE_SIZE` | `32` | 最大队列长度，超出返回 503 |
| `--timeout` | `TTS_TIMEOUT` | `300` | 单请求超时秒数 |
| `--batch-window-ms` | `TTS_BATCH_WINDOW_MS` | `0` | 凑批最大等待时间（ms），仅 Qwen3 生效，0=禁用 |
| `--max-batch-size` | `TTS_MAX_BATCH_SIZE` | `8` | 每批最大请求数，仅 Qwen3 生效 |

### API 使用

**基础 TTS**

```bash
curl -X POST http://127.0.0.1:8013/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "你好，欢迎使用语音合成服务。"}' \
  --output output.wav
```

**指定格式**（`wav` / `mp3` / `pcm`）

```bash
curl -X POST http://127.0.0.1:8013/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "format": "mp3"}' \
  --output output.mp3
```

**声音克隆**

```python
import base64, requests

audio_b64 = base64.b64encode(open('samples/0流萤.wav', 'rb').read()).decode()
ref_text  = open('samples/0流萤.txt').read().strip()

resp = requests.post('http://127.0.0.1:8013/v1/tts', json={
    'text': '用克隆的声音说这句话。',
    'references': [{'audio': audio_b64, 'text': ref_text}],
})
open('cloned.wav', 'wb').write(resp.content)
```

**VoiceDesign 声音风格（Qwen3-TTS-12Hz-1.7B-VoiceDesign）**

> **注意**：使用 `Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit` 时，`instruct` 字段为**必填项**。不填将使用默认/随机声音风格。

```bash
curl -X POST http://127.0.0.1:8013/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "你好，有什么可以帮助你的？", "instruct": "A cheerful young female voice with high pitch"}' \
  --output output.wav
```

`instruct` 示例：
- `"A cheerful young female voice with high pitch"`（活泼年轻女声）
- `"A calm middle-aged male voice with deep tone"`（沉稳中年男声）
- `"An energetic young male voice, fast pace"`（充满活力的年轻男声）
- `"A gentle elderly female voice, slow and clear"`（温柔清晰的老年女声）

**流式输出**

```python
import requests

resp = requests.post('http://127.0.0.1:8013/v1/tts',
    json={'text': '这是流式输出测试。', 'streaming': True, 'format': 'wav'},
    stream=True)

with open('streamed.wav', 'wb') as f:
    for chunk in resp.iter_content(4096):
        if chunk:
            f.write(chunk)
```

**带认证**

```bash
uv run python -m src --model-path ./models/fish-audio-s2-pro-bf16 --api-key <YOUR_API_KEY>

curl -X POST http://127.0.0.1:8013/v1/tts \
  -H "Authorization: Bearer <YOUR_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"text": "认证请求测试"}' \
  --output output.wav
```

### 请求参数说明

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `text` | string | 必填 | 要合成的文本 |
| `format` | string | `"wav"` | 输出格式：`wav`、`mp3`、`pcm` |
| `references` | array | `[]` | 内联 Base64 参考音频（声音克隆） |
| `reference_id` | string | null | 服务端预加载的参考音色 ID |
| `instruct` | string | null | 声音风格描述（Qwen3 VoiceDesign 模型） |
| `streaming` | bool | false | 流式返回（仅支持 WAV） |
| `seed` | int | `42` | 随机种子，固定值可复现结果 |
| `temperature` | float | `0.0` | 采样温度（0.0–1.0） |
| `top_p` | float | `0.7` | Top-p 采样（0.1–1.0） |
| `top_k` | int | `30` | Top-k 采样（0–1000） |
| `repetition_penalty` | float | `1.1` | 重复惩罚（0.9–2.0） |
| `max_new_tokens` | int | `1024` | 最大生成 token 数 |
| `chunk_length` | int | `200` | 文本分块长度（100–300） |
| `cfg_scale` | float | `2.0` | CFG 系数（流匹配模型） |
| `flow_steps` | int | `10` | 扩散步数（流匹配模型） |
| `sigma` | float | `0.25` | Sigma（流匹配模型） |

### 参考音色管理

```bash
# 添加
curl -X POST http://127.0.0.1:8013/v1/references/add \
  -F "id=my-voice" -F "audio=@samples/0流萤.wav" -F "text=参考音频的文本内容"

# 列出
curl http://127.0.0.1:8013/v1/references/list

# 使用
curl -X POST http://127.0.0.1:8013/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "用预设音色说话", "reference_id": "my-voice"}' --output output.wav

# 删除
curl -X DELETE http://127.0.0.1:8013/v1/references/delete \
  -H "Content-Type: application/json" \
  -d '{"reference_id": "my-voice"}'
```

### 可用模型

| 模型 | HuggingFace | 大小 | 特点 |
|---|---|---|---|
| `fish-audio-s2-pro-bf16` | [mlx-community/fish-audio-s2-pro-bf16](https://huggingface.co/mlx-community/fish-audio-s2-pro-bf16) | 9.6GB | 高质量，支持 80+ 语言，支持声音克隆 |
| `fish-audio-s2-pro-8bit` | [cs2764/fish-audio-s2-pro-8bit-mlx](https://huggingface.co/cs2764/fish-audio-s2-pro-8bit-mlx) | 6.3GB | 同等质量，显存占用更低，已预量化可直接下载 |
| `Qwen3-TTS-12Hz-1.7B-Base-8bit` | [mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit) | 2.3GB | 轻量快速，通过 `references` 声音克隆（ICL），音色稳定一致 |
| `Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit` | [mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit) | 2.3GB | 轻量快速，通过 `instruct` 控制声音风格 |

**直接下载预量化 S2 Pro 8-bit（推荐）：**

```bash
# https://huggingface.co/cs2764/fish-audio-s2-pro-8bit-mlx
hf download cs2764/fish-audio-s2-pro-8bit-mlx --local-dir ./models/fish-audio-s2-pro-8bit
```

**或从 BF16 本地量化：**

```bash
uv run python scripts/quantize_fish_s2.py
```

**转换其他 HuggingFace 模型：**

```bash
uv run python scripts/convert_model.py --hf-path <hf-repo-id> --mlx-path ./models/<name>
```

**升级依赖（尤其是 git 源的 mlx-audio）：**

```bash
bash scripts/upgrade_deps.sh            # 升级所有依赖
bash scripts/upgrade_deps.sh --mlx-only # 仅升级 mlx-audio
```

### 测试

所有测试脚本均需先启动服务，位于 `tests/` 目录。

**并发 API 测试**（顺序请求 + 并发请求，支持声音克隆）：

```bash
uv run python tests/test_concurrent_api.py
uv run python tests/test_concurrent_api.py --no-ref
uv run python tests/test_concurrent_api.py --instruct "A cheerful young female voice with high pitch" --no-ref
```

**长文本并发 API 测试**（每条测试文本均不低于 200 字）：

```bash
uv run python tests/test_concurrent_api_long.py
uv run python tests/test_concurrent_api_long.py --no-ref
```

**文本长度扩展测试**（顺序，文本长度从 50 字到 1000 字，步长 50，记录延迟并输出汇总表格）：

```bash
uv run python tests/test_text_length_scaling.py
uv run python tests/test_text_length_scaling.py --no-ref
uv run python tests/test_text_length_scaling.py --min-len 100 --max-len 500 --step 100
```

### 项目结构

```
mlx-audio-api/
├── src/
│   ├── __main__.py          # 入口，支持单/多进程启动
│   ├── server.py            # FastAPI 应用和路由
│   ├── tts_engine.py        # MLX 推理引擎封装，含批量推理
│   ├── request_queue.py     # 请求队列和 worker 管理（标准 + 批处理 worker）
│   ├── reference_manager.py # 参考音色存储管理
│   ├── auth.py              # Bearer Token 认证中间件
│   ├── models.py            # Pydantic 请求/响应模型
│   └── config.py            # 配置解析（CLI + 环境变量）
├── tests/                   # API 集成测试脚本
├── scripts/
│   ├── convert_model.py     # HuggingFace → MLX 转换脚本
│   ├── quantize_fish_s2.py  # Fish Audio S2 Pro BF16 → 8-bit 量化脚本
│   └── upgrade_deps.sh      # 升级所有依赖（尤其是 git 源的 mlx-audio）
├── models/                  # 本地模型目录（gitignored）
├── samples/                 # 参考音频示例
├── references/              # 服务端存储的参考音色（运行时，gitignored）
└── api_reference.md         # 完整 API 文档
```
