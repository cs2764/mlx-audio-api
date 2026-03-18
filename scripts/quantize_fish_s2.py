"""
Quantize fish-audio-s2-pro-bf16 to 8-bit in-memory and save.

Strategy:
  1. Load the bf16 model via mlx_audio (handles sanitize + weight loading)
  2. Quantize in-memory with nn.quantize(group_size=64, bits=8)
  3. Save weights WITHOUT "model." prefix so apply_quantization's predicate
     (which checks f"{layer_path}.scales" in weights) can match correctly.
     Model.load_weights() prepends nothing — it strips "model." on load, so
     the file must store keys as e.g. "embeddings.scales", "layers.0...".
  4. Copy non-weight files, update config.json with quantization block.

Why NOT add "model." prefix:
  base_load_model calls apply_quantization(model, config, weights) where
  `weights` is the raw dict from load_weights(). The predicate checks
  f"{p}.scales" in weights, where p is the DualARTransformer layer path
  (no "model." prefix). If weights have "model." prefix, nothing matches,
  no layers get converted to QuantizedLinear/QuantizedEmbedding, and
  load_weights(strict=True) fails with "Missing N parameters: embeddings.weight".

  Model.load_weights() strips "model." before delegating to DualARTransformer,
  so the file keys must NOT have the "model." prefix.
"""

import json
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

SRC = Path("models/fish-audio-s2-pro-bf16")
DST = Path("models/fish-audio-s2-pro-8bit")
GROUP_SIZE = 64
BITS = 8


def main():
    print("Loading bf16 model...")
    from mlx_audio.tts.utils import load_model

    model = load_model(str(SRC))
    print("Model loaded.")

    # Quantize DualARTransformer in-memory
    print(f"Quantizing (group_size={GROUP_SIZE}, bits={BITS})...")
    nn.quantize(model.model, group_size=GROUP_SIZE, bits=BITS)
    mx.eval(model.model.parameters())
    print("Quantization done.")

    # Flatten weights — NO "model." prefix.
    # Keys will be e.g. "embeddings.weight", "embeddings.scales",
    # "layers.0.attention.wqkv.scales", etc.
    # apply_quantization checks f"{p}.scales" in weights where p has no prefix.
    flat = dict(tree_flatten(model.model.parameters()))
    print(f"Total weight tensors: {len(flat)}")

    # Spot-check: confirm quantized keys are present
    emb_keys = [k for k in flat if "embeddings" in k]
    print(f"Embedding keys (sample): {emb_keys[:6]}")

    # Save
    DST.mkdir(parents=True, exist_ok=True)
    out_file = DST / "model.safetensors"
    print(f"Saving weights to {out_file} ...")
    mx.save_safetensors(str(out_file), flat)
    print("Weights saved.")

    # Remove stale index file if present (single-shard now)
    index_file = DST / "model.safetensors.index.json"
    if index_file.exists():
        index_file.unlink()
        print("Removed stale model.safetensors.index.json")

    # Copy non-weight files from src (tokenizer, codec, special tokens, etc.)
    skip = {"model.safetensors", "model.safetensors.index.json", "config.json"}
    for f in SRC.iterdir():
        if f.name not in skip and f.is_file():
            dst_f = DST / f.name
            shutil.copy2(f, dst_f)
            print(f"Copied {f.name}")

    # Update config.json — add quantization block
    with open(SRC / "config.json") as fh:
        config = json.load(fh)
    config["quantization"] = {"group_size": GROUP_SIZE, "bits": BITS}
    with open(DST / "config.json", "w") as fh:
        json.dump(config, fh, indent=2)
    print("config.json updated with quantization block.")

    print("\nDone. Verifying load (strict=True)...")
    # Patch base_load_model to use strict=True for verification
    from mlx_audio.tts.utils import load_model as _load
    model2 = _load(str(DST))
    print("Verification load succeeded.")


if __name__ == "__main__":
    main()
