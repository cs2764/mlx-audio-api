# MLX Audio API

[English](#english) | [中文](#chinese)

---

<a name="english"></a>

## English

A local text-to-speech inference service built on top of [mlx-audio](https://github.com/Blaizzy/mlx-audio) — the core inference engine that handles MLX-based TTS model loading and audio generation. This project wraps mlx-audio with a production-ready HTTP API layer: multi-worker concurrency, request queuing, zero-shot voice cloning, streaming output, and Bearer token auth. Optimized for Apple Silicon (M1/M2/M3).

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
# Fish Audio S2 Pro BF16 (recommended, high quality, ~9.6 GB)
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('mlx-community/fish-audio-s2-pro-bf16', local_dir='./models/fish-audio-s2-pro-bf16')
"

# Ming-omni-tts-0.5B (lightweight, ~2.8 GB, already in MLX format)
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('inclusionAI/Ming-omni-tts-0.5B', local_dir='./models/Ming-omni-tts-0.5B-mlx')
"
```

**3. Start the server**

```bash
# Single worker (default)
uv run python -m src --model-path ./models/fish-audio-s2-pro-bf16

# Multi-worker (recommended for production)
uv run python -m src --model-path ./models/fish-audio-s2-pro-bf16 --num-workers 4
```

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

**VoiceDesign (Qwen3 models)**

Use the `instruct` field to describe the desired voice style:

```bash
curl -X POST http://127.0.0.1:8013/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello!", "instruct": "A cheerful young female voice with high pitch"}' \
  --output output.wav
```

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
| `seed` | int | null | Random seed for reproducible output |
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

### Concurrency Performance

Benchmarked on Apple M3 Max with Fish Audio S2 Pro BF16 (~50-char text with voice cloning):

| workers | 4 concurrent | 16 concurrent | 32 concurrent | avg latency (32) |
|---------|-------------|--------------|--------------|-----------------|
| 1       | 5.0s        | 23.8s        | 63.3s        | 33.7s           |
| 4       | 4.0s        | 15.2s        | 29.0s        | 14.8s           |
| 8       | 4.2s        | **12.1s** ✅  | **27.2s** ✅  | **13.6s** ✅     |
| 16      | 4.0s        | 13.5s        | 25.5s        | 14.4s           |

Recommended: `--num-workers 4` for S2 Pro BF16 (memory-constrained), `--num-workers 8` for lighter models.

### Supported Models

| Model | Size | Notes |
|---|---|---|
| `fish-audio-s2-pro-bf16` | 9.6 GB | High quality, 80+ languages |
| `Ming-omni-tts-0.5B-mlx` | 2.8 GB | Lightweight, fast, MLX-native |

**Quantize S2 Pro to 8-bit:**

```bash
uv run python -m mlx_audio.convert \
  --hf-path ./models/fish-audio-s2-pro-bf16 \
  --mlx-path ./models/fish-audio-s2-pro-8bit \
  -q --q-bits 8
```

**Convert other HuggingFace models:**

```bash
uv run python scripts/convert_model.py --hf-path <hf-repo-id> --mlx-path ./models/<name>
```

### Testing

**Concurrent API test** — requires a running server, tests sequential + concurrent requests with optional voice cloning:

```bash
# Default (with voice cloning, uses samples/0流萤.wav)
uv run python tests/test_concurrent_api.py

# No reference audio (plain TTS, faster)
uv run python tests/test_concurrent_api.py --no-ref

# Skip sequential baseline, go straight to concurrent
uv run python tests/test_concurrent_api.py --skip-sequential

# Custom API URL, auth key, output dir
uv run python tests/test_concurrent_api.py --url http://127.0.0.1:8013/v1/tts --api-key <KEY> --output-dir ./my_outputs
```

**Performance benchmark** (Ming-omni-tts-0.5B, 32-concurrent):

```bash
uv run python tests/test_performance.py
uv run python tests/test_performance.py --skip-sequential --no-ref
```

### Project Structure

```
mlx-audio-api/
├── src/
│   ├── __main__.py          # Entry point; single/multi-process startup
│   ├── server.py            # FastAPI app factory, all routes, lifespan
│   ├── tts_engine.py        # TTSEngine: mlx_audio wrapper, audio encoding
│   ├── request_queue.py     # RequestQueue + asyncio worker pool
│   ├── reference_manager.py # File-based reference voice storage
│   ├── auth.py              # Bearer token middleware
│   ├── models.py            # Pydantic request/response models
│   └── config.py            # ServerConfig + argparse/env var parsing
├── tests/                   # Unit and property-based tests
├── scripts/
│   └── convert_model.py     # HuggingFace → MLX conversion utility
├── models/                  # Local model storage (gitignored)
├── samples/                 # Reference audio samples
├── references/              # Server-side stored voices (runtime, gitignored)
└── api_reference.md         # Full API documentation
```

---

<a name="chinese"></a>

## 中文

基于 [mlx-audio](https://github.com/Blaizzy/mlx-audio) 构建的本地 TTS 推理服务。mlx-audio 是核心推理引擎，负责 MLX 模型加载和音频生成；本项目在其基础上封装了生产级 HTTP API 层：多 worker 并发、请求队列、零样本声音克隆、流式输出和 Bearer Token 认证。专为 Apple Silicon（M1/M2/M3）优化。

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
# Fish Audio S2 Pro BF16（推荐，高质量，约 9.6GB）
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('mlx-community/fish-audio-s2-pro-bf16', local_dir='./models/fish-audio-s2-pro-bf16')
"

# Ming-omni-tts-0.5B（轻量，约 2.8GB，已是 MLX 格式）
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('inclusionAI/Ming-omni-tts-0.5B', local_dir='./models/Ming-omni-tts-0.5B-mlx')
"
```

**3. 启动服务**

```bash
# 单 worker（默认）
uv run python -m src --model-path ./models/fish-audio-s2-pro-bf16

# 多 worker（推荐生产环境）
uv run python -m src --model-path ./models/fish-audio-s2-pro-bf16 --num-workers 4
```

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

**VoiceDesign 声音风格（Qwen3 等模型）**

通过 `instruct` 字段描述声音风格：

```bash
curl -X POST http://127.0.0.1:8013/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "你好！", "instruct": "A cheerful young female voice with high pitch"}' \
  --output output.wav
```

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
| `seed` | int | null | 随机种子，固定值可复现结果 |
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

### 并发性能

以下为 Apple M3 Max 上 Fish Audio S2 Pro BF16 的实测数据（约 50 字文本，含声音克隆）：

| workers | 4 并发总耗时 | 16 并发总耗时 | 32 并发总耗时 | 32 并发平均延迟 |
|---------|------------|-------------|-------------|--------------|
| 1       | 5.0s       | 23.8s       | 63.3s       | 33.7s        |
| 4       | 4.0s       | 15.2s       | 29.0s       | 14.8s        |
| 8       | 4.2s       | **12.1s** ✅ | **27.2s** ✅ | **13.6s** ✅  |
| 16      | 4.0s       | 13.5s       | 25.5s       | 14.4s        |

推荐：S2 Pro BF16 使用 `--num-workers 4`（内存受限），轻量模型可用 `--num-workers 8`。

### 可用模型

| 模型 | 大小 | 特点 |
|---|---|---|
| `fish-audio-s2-pro-bf16` | 9.6GB | 高质量，支持 80+ 语言 |
| `Ming-omni-tts-0.5B-mlx` | 2.8GB | 轻量快速，MLX 原生格式 |

**量化 S2 Pro 为 8bit：**

```bash
uv run python -m mlx_audio.convert \
  --hf-path ./models/fish-audio-s2-pro-bf16 \
  --mlx-path ./models/fish-audio-s2-pro-8bit \
  -q --q-bits 8
```

**转换其他 HuggingFace 模型：**

```bash
uv run python scripts/convert_model.py --hf-path <hf-repo-id> --mlx-path ./models/<name>
```

### 测试

**并发 API 测试**（需先启动服务，测试顺序请求 + 并发请求，支持声音克隆）：

```bash
# 默认运行（带声音克隆，使用 samples/0流萤.wav）
uv run python tests/test_concurrent_api.py

# 不使用参考音频（纯 TTS，速度更快）
uv run python tests/test_concurrent_api.py --no-ref

# 跳过顺序测试，直接跑并发
uv run python tests/test_concurrent_api.py --skip-sequential

# 自定义 API 地址、认证、输出目录
uv run python tests/test_concurrent_api.py --url http://127.0.0.1:8013/v1/tts --api-key <KEY> --output-dir ./my_outputs
```

**性能基准测试**（Ming-omni-tts-0.5B，32 并发）：

```bash
uv run python tests/test_performance.py
uv run python tests/test_performance.py --skip-sequential --no-ref
```

### 项目结构

```
mlx-audio-api/
├── src/
│   ├── __main__.py          # 入口，支持单/多进程启动
│   ├── server.py            # FastAPI 应用和路由
│   ├── tts_engine.py        # MLX 推理引擎封装
│   ├── request_queue.py     # 请求队列和 worker 管理
│   ├── reference_manager.py # 参考音色存储管理
│   ├── auth.py              # Bearer Token 认证中间件
│   ├── models.py            # Pydantic 请求/响应模型
│   └── config.py            # 配置解析（CLI + 环境变量）
├── tests/                   # 单元和属性测试
├── scripts/
│   └── convert_model.py     # HuggingFace → MLX 转换脚本
├── models/                  # 本地模型目录（gitignored）
├── samples/                 # 参考音频示例
├── references/              # 服务端存储的参考音色（运行时，gitignored）
└── api_reference.md         # 完整 API 文档
```
