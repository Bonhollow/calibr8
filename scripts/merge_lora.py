#!/usr/bin/env python3
"""
Manually merge LoRA weights into the quantized Qwen3-4B base model.
Produces a standalone 2.4 GB model loadable via load("path/to/model").
"""

import gc
import json
import shutil
from pathlib import Path

import mlx.core as mx
from mlx import nn
from mlx_lm import load
from mlx_lm.tuner.lora import LoRALinear

BASE_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit-g32"
ADAPTER_PATH = Path(__file__).parent.parent / "adapters" / "confidence_qwen_v1"
OUTPUT_PATH = Path(__file__).parent.parent / "models" / "calibr8"

GROUP_SIZE = 32
BITS = 4


def get_lora_alpha(path: Path) -> float:
    config_path = path / "adapter_config.json"
    with open(config_path) as f:
        cfg = json.load(f)
    return float(cfg.get("lora_alpha", 32))


def main():
    print("Loading base model + adapter...")
    model, tokenizer = load(BASE_MODEL, adapter_path=str(ADAPTER_PATH))
    lora_alpha = get_lora_alpha(ADAPTER_PATH)
    print(f"  LoRA alpha: {lora_alpha}")

    print("Merging LoRA weights into quantized base model...")
    merged_count = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            inner = module.linear
            lora_a = module.lora_a
            lora_b = module.lora_b
            rank = lora_b.shape[0]

            # Get quantized components
            w_q = inner.weight
            s = inner.scales
            b = inner.biases

            # Dequantize to float16
            w_fp = mx.dequantize(w_q, s, b, group_size=GROUP_SIZE, bits=BITS)

            # Compute LoRA delta: B^T @ A^T * alpha/rank
            # Stored: lora_a = (in_features, rank), lora_b = (rank, out_features)
            A = lora_a.T  # (rank, in_features)
            B = lora_b.T  # (out_features, rank)
            lora_delta = (B @ A) * (lora_alpha / rank)

            # Merge
            w_merged = w_fp + lora_delta.astype(w_fp.dtype)

            # Re-quantize
            w_q_new, s_new, b_new = mx.quantize(w_merged, group_size=GROUP_SIZE, bits=BITS)

            # Store back
            inner.weight = w_q_new
            inner.scales = s_new
            inner.biases = b_new

            merged_count += 1
            if merged_count % 10 == 0:
                print(f"  Merged {merged_count} LoRA layers...")

    print(f"  Merged {merged_count} LoRA layers total")

    # Replace LoRALinear modules with their inner QuantizedLinear
    print("Removing LoRA wrappers from model graph...")
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            parts = name.split(".")
            child_name = parts[-1]
            parent = model
            for part in parts[:-1]:
                if isinstance(parent, (list, nn.Sequential)):
                    parent = parent[int(part)]
                else:
                    parent = getattr(parent, part)
            setattr(parent, child_name, module.linear)

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    print(f"Saving merged model to {OUTPUT_PATH}...")
    model.save_weights(str(OUTPUT_PATH / "model.safetensors"))

    # Copy config files from base model cache
    import glob as globlib
    base_cache = Path.home() / ".cache" / "huggingface" / "hub"
    snap_dirs = list(base_cache.glob(f"models--{BASE_MODEL.replace('/', '--')}/snapshots/*"))
    if snap_dirs:
        snap = snap_dirs[0]
        for f in ["config.json", "tokenizer.json", "tokenizer_config.json",
                   "generation_config.json", "chat_template.jinja"]:
            src = snap / f
            if src.exists():
                shutil.copy2(src, OUTPUT_PATH / f)

    total_size = sum(f.stat().st_size for f in OUTPUT_PATH.glob("**/*") if f.is_file()) / 1e9
    print(f"\nDone! Merged model saved to {OUTPUT_PATH}")
    print(f"Size: {total_size:.2f} GB")
    print(f"\nLoad with:\n  model, tokenizer = load('{OUTPUT_PATH}')")


if __name__ == "__main__":
    main()
