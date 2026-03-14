from dataclasses import dataclass
import argparse
import os


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8013
    model_path: str = "./models/fish-audio-s2-pro-bf16"
    api_key: str | None = None
    max_queue_size: int = 32
    timeout: float = 300.0
    num_workers: int = 1


def parse_config(args=None) -> ServerConfig:
    """Parse CLI args with env var fallbacks.
    Priority: CLI > env var > default
    Env vars are used in multi-process mode where child workers can't read CLI args.
    """
    parser = argparse.ArgumentParser(description="TTS Inference API Server")
    parser.add_argument("--host", default=os.environ.get("TTS_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("TTS_PORT", "8013")))
    parser.add_argument("--model-path", default=os.environ.get("TTS_MODEL_PATH", "./models/fish-audio-s2-pro-bf16"))
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-queue-size", type=int, default=int(os.environ.get("TTS_MAX_QUEUE_SIZE", "32")))
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("TTS_TIMEOUT", "300.0")))
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Number of parallel inference workers (each loads its own model instance)")

    parsed = parser.parse_args(args)

    # Priority: CLI > env var > None
    api_key = parsed.api_key or os.environ.get("TTS_API_KEY") or None

    return ServerConfig(
        host=parsed.host,
        port=parsed.port,
        model_path=parsed.model_path,
        api_key=api_key,
        max_queue_size=parsed.max_queue_size,
        timeout=parsed.timeout,
        num_workers=parsed.num_workers,
    )
