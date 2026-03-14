#!/usr/bin/env python3
"""Model conversion script: converts fishaudio/s2-pro to MLX 8-bit quantized format."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Convert and quantize Fish Audio model to MLX format"
    )
    parser.add_argument("--hf-path", default="fishaudio/s2-pro", help="HuggingFace model path")
    parser.add_argument(
        "--mlx-path",
        default="./models/fish-audio-s2-pro-bf16",
        help="Output MLX model path",
    )
    parser.add_argument("--q-bits", type=int, default=8, help="Quantization bits (default: 8)")
    args = parser.parse_args()

    mlx_path = Path(args.mlx_path)

    # Skip if already exists
    if mlx_path.exists():
        print(f"Model already exists at {mlx_path}, skipping conversion.")
        return

    print(f"Converting {args.hf_path} → {mlx_path} ({args.q_bits}-bit quantization)...")

    cmd = [
        sys.executable,
        "-m",
        "mlx_audio.convert",
        "--hf-path",
        args.hf_path,
        "--mlx-path",
        str(mlx_path),
        "-q",
        "--q-bits",
        str(args.q_bits),
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Model conversion failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Model conversion failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Model successfully converted and saved to {mlx_path}")


if __name__ == "__main__":
    main()
