"""Entry point: python -m src --model-path ./models/fish-audio-s2-pro-bf16"""

import logging
import os

import uvicorn

from .config import parse_config
from .server import create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def app_factory():
    """App factory for uvicorn --factory mode (multi-process).
    Each worker process loads exactly one engine instance.
    """
    config = parse_config()
    config.num_workers = 1  # each process has exactly one engine
    return create_app(config)


def main():
    config = parse_config()

    num_workers = config.num_workers
    if num_workers > 1:
        # Multi-process mode: pass CLI args via env so each worker can reconstruct config
        os.environ.setdefault("TTS_MODEL_PATH", config.model_path)
        os.environ.setdefault("TTS_HOST", config.host)
        os.environ.setdefault("TTS_PORT", str(config.port))
        os.environ.setdefault("TTS_MAX_QUEUE_SIZE", str(config.max_queue_size))
        os.environ.setdefault("TTS_TIMEOUT", str(config.timeout))
        if config.api_key:
            os.environ.setdefault("TTS_API_KEY", config.api_key)

        print(f"Starting in multi-process mode with {num_workers} workers")
        uvicorn.run(
            "src.__main__:app_factory",
            host=config.host,
            port=config.port,
            workers=num_workers,
            factory=True,
        )
    else:
        app = create_app(config)
        uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
