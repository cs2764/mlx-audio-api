# TTS Inference API 参考文档

> **版本**: 1.1.0  
> **基础 URL**: `http://<host>:8013`  
> **协议**: HTTP REST  
> **支持格式**: `application/json`、`multipart/form-data`

---

## 目录

- [快速开始](#快速开始)
- [认证](#认证)
- [API 端点](#api-端点)
  - [健康检查](#1-健康检查)
  - [文本转语音 (TTS)](#2-文本转语音-tts)
  - [参考音色管理](#3-参考音色管理)
- [语音克隆](#语音克隆)
- [VoiceDesign 模式](#voicedesign-模式)
- [流式输出](#流式输出)
- [错误处理](#错误处理)
- [代码示例](#代码示例)
- [性能建议](#性能建议)

---

## 快速开始

最简单的 TTS 请求只需要一个 `text` 字段：

```python
import requests

response = requests.post(
    "http://127.0.0.1:8013/v1/tts",
    json={"text": "你好，欢迎使用语音合成系统。"},
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

---

## 认证

如果服务端配置了 `--api-key`，所有请求必须携带 Bearer Token：

```
Authorization: Bearer <YOUR_API_KEY>
```

未配置 API Key 时无需认证。

---

## API 端点

### 1. 健康检查

| 属性 | 值 |
|---|---|
| **URL** | `/v1/health` |
| **方法** | `GET` 或 `POST` |

**响应示例：**

```json
{"status": "ok"}
```

---

### 2. 文本转语音 (TTS)

| 属性 | 值 |
|---|---|
| **URL** | `/v1/tts` |
| **方法** | `POST` |
| **Content-Type** | `application/json` |
| **响应类型** | `audio/wav`、`audio/mpeg`、`application/octet-stream` |

#### 请求体

```json
{
  "text": "要合成的文本内容",
  "format": "wav",
  "references": [],
  "reference_id": null,
  "instruct": null,
  "streaming": false,
  "seed": null,
  "normalize": true,
  "chunk_length": 200,
  "max_new_tokens": 1024,
  "temperature": 0.0,
  "top_p": 0.7,
  "top_k": 30,
  "repetition_penalty": 1.1,
  "cfg_scale": 2.0,
  "flow_steps": 10,
  "sigma": 0.25
}
```

#### 参数说明

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `text` | `string` | ✅ | - | 要合成的文本，不能为空或纯空白 |
| `format` | `string` | ❌ | `"wav"` | 输出格式：`wav`、`mp3`、`pcm` |
| `references` | `array` | ❌ | `[]` | 内联参考音频列表，用于声音克隆（见[语音克隆](#语音克隆)） |
| `reference_id` | `string\|null` | ❌ | `null` | 服务端预加载的参考音色 ID |
| `instruct` | `string\|null` | ❌ | `null` | VoiceDesign 声音描述（Qwen3 等模型专用，见[VoiceDesign 模式](#voicedesign-模式)） |
| `streaming` | `bool` | ❌ | `false` | 是否启用流式返回（仅支持 `wav` 格式） |
| `seed` | `int\|null` | ❌ | `null` | 随机种子，固定值可复现结果 |
| `normalize` | `bool` | ❌ | `true` | 是否对文本进行标准化（数字、英文等） |
| `chunk_length` | `int` | ❌ | `200` | 文本分块长度（100-300） |
| `max_new_tokens` | `int` | ❌ | `1024` | 最大生成 token 数，0 为不限制 |
| `temperature` | `float` | ❌ | `0.0` | 采样温度（0.0-1.0），0 为确定性输出 |
| `top_p` | `float` | ❌ | `0.7` | Top-p 采样（0.1-1.0） |
| `top_k` | `int` | ❌ | `30` | Top-k 采样（0-1000） |
| `repetition_penalty` | `float` | ❌ | `1.1` | 重复惩罚因子（0.9-2.0） |
| `cfg_scale` | `float\|null` | ❌ | `2.0` | Flow-matching 引导强度（0.5-10.0），部分模型专用 |
| `flow_steps` | `int\|null` | ❌ | `10` | Flow-matching 步数（1-100），部分模型专用 |
| `sigma` | `float\|null` | ❌ | `0.25` | Flow-matching sigma（0.0-1.0），部分模型专用 |

#### 成功响应

- **状态码**: `200`
- **Content-Type**: 取决于 `format`
  - `wav` → `audio/wav`
  - `mp3` → `audio/mpeg`
  - `pcm` → `application/octet-stream`
- **Body**: 二进制音频数据

#### 错误响应

| 状态码 | 原因 |
|---|---|
| `400` | 参数不合法（文本为空、streaming 使用非 WAV 格式等） |
| `401` | API Key 无效或缺失 |
| `404` | `reference_id` 不存在 |
| `408` | 推理超时 |
| `503` | 请求队列已满 |
| `500` | 模型推理失败 |

---

### 3. 参考音色管理

#### 3.1 添加参考音色

| 属性 | 值 |
|---|---|
| **URL** | `/v1/references/add` |
| **方法** | `POST` |
| **Content-Type** | `multipart/form-data` |

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `id` | `string` | ✅ | 音色 ID（字母数字、`-`、`_`、空格，最长 255 字符） |
| `audio` | `file` | ✅ | 参考音频文件（推荐 WAV） |
| `text` | `string` | ✅ | 参考音频对应的文本 |

```bash
curl -X POST http://127.0.0.1:8013/v1/references/add \
  -F "id=my-voice" \
  -F "audio=@reference.wav" \
  -F "text=这是参考音频的文本内容"
```

**响应：**

```json
{"success": true, "message": "Reference 'my-voice' added successfully", "reference_id": "my-voice"}
```

#### 3.2 列出参考音色

| 属性 | 值 |
|---|---|
| **URL** | `/v1/references/list` |
| **方法** | `GET` |

```json
{"success": true, "reference_ids": ["my-voice", "narrator-01"], "message": "Found 2 reference(s)"}
```

#### 3.3 删除参考音色

| 属性 | 值 |
|---|---|
| **URL** | `/v1/references/delete` |
| **方法** | `DELETE` |
| **Content-Type** | `application/json` |

```json
{"reference_id": "my-voice"}
```

#### 3.4 重命名参考音色

| 属性 | 值 |
|---|---|
| **URL** | `/v1/references/update` |
| **方法** | `POST` |
| **Content-Type** | `application/json` |

```json
{"old_reference_id": "my-voice", "new_reference_id": "narrator-main"}
```

---

## 语音克隆

支持零样本声音克隆，有两种方式：

### 方式一：内联参考音频

将参考音频 Base64 编码后放入请求体（适合一次性使用）：

```python
import base64, requests

audio_b64 = base64.b64encode(open("reference.wav", "rb").read()).decode()

response = requests.post("http://127.0.0.1:8013/v1/tts", json={
    "text": "用克隆的声音说出这段话。",
    "references": [{"audio": audio_b64, "text": "参考音频的文本内容"}],
})
open("cloned.wav", "wb").write(response.content)
```

### 方式二：服务端预加载音色

先上传，后通过 `reference_id` 引用（适合反复使用同一音色）：

```python
# 上传一次
import requests
requests.post("http://127.0.0.1:8013/v1/references/add",
    files={"audio": open("reference.wav", "rb")},
    data={"id": "my-voice", "text": "参考文本"})

# 每次请求直接引用
response = requests.post("http://127.0.0.1:8013/v1/tts", json={
    "text": "用预设的声音说话。",
    "reference_id": "my-voice",
})
```

> 当 `references` 和 `reference_id` 同时提供时，`reference_id` 优先。

---

## VoiceDesign 模式

`Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16` 等 VoiceDesign 类模型不使用参考音频，而是通过 `instruct` 字段用自然语言描述声音风格：

```bash
curl -X POST http://127.0.0.1:8013/v1/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, welcome to our service.",
    "instruct": "A cheerful young female voice with high pitch"
  }' \
  --output output.wav
```

- 若使用 Qwen3 模型且未传 `instruct`，服务端自动使用默认值 `"A sexy female voice"`
- `instruct` 使用英文描述效果最佳
- 示例描述：
  - `"A calm middle-aged male voice with deep tone"`
  - `"An energetic young male voice, fast pace"`
  - `"A gentle elderly female voice, slow and clear"`

---

## 流式输出

启用 `streaming: true` 时以 chunked transfer 方式逐块返回音频，**仅支持 WAV 格式**：

```python
import requests

response = requests.post(
    "http://127.0.0.1:8013/v1/tts",
    json={"text": "这是流式输出测试。", "streaming": True},
    stream=True,
)

with open("streamed.wav", "wb") as f:
    for chunk in response.iter_content(4096):
        if chunk:
            f.write(chunk)
```

---

## 错误处理

| 状态码 | 含义 | 常见原因 |
|---|---|---|
| `200` | 成功 | 正常返回音频数据 |
| `400` | 请求参数错误 | 文本为空、streaming 使用非 WAV 格式、参数越界 |
| `401` | 认证失败 | API Key 无效或缺失 |
| `404` | 资源不存在 | `reference_id` 不存在 |
| `408` | 请求超时 | 推理时间超过 `--timeout` 设置 |
| `409` | 资源冲突 | 添加已存在的参考 ID |
| `500` | 服务器内部错误 | 模型推理失败 |
| `503` | 服务繁忙 | 请求队列已满（超过 `--max-queue-size`） |

**错误响应格式：**

```json
{"success": false, "message": "text must not be empty or whitespace-only"}
```

---

## 代码示例

### Python 客户端封装

```python
import base64
import requests
from pathlib import Path
from typing import Optional


class TTSClient:
    def __init__(self, base_url="http://127.0.0.1:8013", api_key=None, timeout=300):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def health_check(self) -> bool:
        try:
            return requests.get(f"{self.base_url}/v1/health", timeout=5).status_code == 200
        except Exception:
            return False

    def tts(
        self,
        text: str,
        *,
        output_path: Optional[str] = None,
        format: str = "wav",
        ref_audio_path: Optional[str] = None,
        ref_text: Optional[str] = None,
        reference_id: Optional[str] = None,
        instruct: Optional[str] = None,
        temperature: float = 0.0,
        top_p: float = 0.7,
        top_k: int = 30,
        seed: Optional[int] = None,
        streaming: bool = False,
    ) -> bytes:
        payload = {
            "text": text,
            "format": format,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "streaming": streaming,
        }
        if seed is not None:
            payload["seed"] = seed
        if instruct:
            payload["instruct"] = instruct
        elif reference_id:
            payload["reference_id"] = reference_id
        elif ref_audio_path:
            audio_b64 = base64.b64encode(Path(ref_audio_path).read_bytes()).decode()
            payload["references"] = [{"audio": audio_b64, "text": ref_text or ""}]

        resp = requests.post(
            f"{self.base_url}/v1/tts",
            json=payload,
            headers=self.headers,
            timeout=self.timeout,
            stream=streaming,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"TTS failed: HTTP {resp.status_code} - {resp.text}")

        audio = b"".join(resp.iter_content(4096)) if streaming else resp.content
        if output_path:
            Path(output_path).write_bytes(audio)
        return audio


# 使用示例
if __name__ == "__main__":
    client = TTSClient("http://127.0.0.1:8013")

    # 基础 TTS
    client.tts("你好，世界！", output_path="hello.wav")

    # 声音克隆
    client.tts("用克隆的声音说话", output_path="cloned.wav",
               ref_audio_path="reference.wav", ref_text="参考文本")

    # VoiceDesign（Qwen3）
    client.tts("Hello world", output_path="designed.wav",
               instruct="A calm male voice with deep tone")
```

### cURL

```bash
# 基础 TTS
curl -X POST http://127.0.0.1:8013/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "你好，世界！"}' --output output.wav

# MP3 格式
curl -X POST http://127.0.0.1:8013/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "format": "mp3"}' --output output.mp3

# VoiceDesign
curl -X POST http://127.0.0.1:8013/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello", "instruct": "A cheerful young female voice"}' --output output.wav

# 带认证
curl -X POST http://127.0.0.1:8013/v1/tts \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR_API_KEY>" \
  -d '{"text": "认证请求"}' --output output.wav
```

### JavaScript / Node.js

```javascript
async function tts(text, options = {}) {
  const response = await fetch("http://127.0.0.1:8013/v1/tts", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, ...options }),
  });
  if (!response.ok) throw new Error(`TTS failed: ${response.status}`);
  return response.arrayBuffer();
}

// 基础用法
const audio = await tts("Hello world");

// VoiceDesign
const audio2 = await tts("Hello", { instruct: "A cheerful young female voice" });
```

---

## 性能建议

### 参数调优

| 目标 | 推荐设置 |
|---|---|
| **确定性输出（可复现）** | `temperature: 0.0`，固定 `seed` |
| **自然多变** | `temperature: 0.7`，`top_p: 0.8`，`top_k: 50` |
| **长文本合成** | `chunk_length: 250`，`max_new_tokens: 2048` |
| **短句快速合成** | `chunk_length: 100`，`max_new_tokens: 512` |

### 并发配置（Fish Audio S2 Pro BF16，M3 Max 实测）

| workers | 16 并发总耗时 | 32 并发总耗时 | 32 并发平均延迟 |
|---------|-------------|-------------|--------------|
| 1       | 23.8s       | 63.3s       | 33.7s        |
| 4       | 15.2s       | 29.0s       | 14.8s        |
| 8       | **12.1s** ✅ | **27.2s** ✅ | **13.6s** ✅  |

推荐 `--num-workers 4` 作为生产配置，兼顾吞吐和内存（S2 Pro BF16 约 9.6GB/worker）。

### 其他建议

- 单次请求文本建议不超过 **500 字**，超长文本建议客户端分段发送
- 采样率由模型决定（S2 Pro 为 **44100 Hz**），客户端播放时注意匹配
- 队列满时返回 `503`，可通过 `--max-queue-size` 调整容量
