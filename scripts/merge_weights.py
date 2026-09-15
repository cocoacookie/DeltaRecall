#!/usr/bin/env python3
# Copyright 2026 DeltaRecall Authors.
# SPDX-License-Identifier: Apache-2.0
"""Merge a frozen Qwen3.5 backbone with a DeltaRecall memory adapter into one
self-contained checkpoint directory.

DeltaRecall is plug-and-play: at runtime you keep the backbone frozen and only
load the small ``memory_*`` adapter with ``install_gras_hmc`` +
``load_memory_adapter``. This helper is for the alternative "ship one folder"
workflow -- it writes the base weights plus the adapter's ``memory_*`` tensors
into a single directory, and copies the adapter/calibration alongside so the
recipe is reproducible.

Example (Qwen3.5-9B):

    python scripts/merge_weights.py \
      --base-model  Qwen/Qwen3.5-9B \
      --adapter     model/deltarecall-9b/memory_adapter.pt \
      --calibration calibration/deltarecall-9b-calibration.json \
      --output-path merged_ckpt/DeltaRecall-9B
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", required=True, help="HF repo id or local path to the frozen Qwen3.5 backbone.")
    p.add_argument("--adapter", required=True, help="DeltaRecall memory_adapter.pt to merge.")
    p.add_argument("--calibration", default=None, help="Optional calibration JSON to copy next to the merged model.")
    p.add_argument("--output-path", required=True, help="Destination directory for the merged checkpoint.")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_path)
    out.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16, trust_remote_code=True
    )
    model_sd = model.state_dict()

    # The adapter stores only the per-layer memory_* tensors keyed by layer index.
    payload = torch.load(args.adapter, map_location="cpu", weights_only=True)
    layers = payload.get("layers", payload)
    merged, missing = 0, []
    for layer_idx, tensors in layers.items():
        for name, value in tensors.items():
            key = f"model.layers.{layer_idx}.self_attn.{name}"
            if key in model_sd:
                model_sd[key] = value.to(model_sd[key].dtype)
                merged += 1
            else:
                missing.append(key)
    if missing:
        # Backbones expose the memory params only after install_gras_hmc; a merged
        # folder therefore also needs the adapter file for install-time loading.
        print(f"[warn] {len(missing)} adapter keys have no matching backbone tensor; "
              f"example: {missing[0]}")

    try:
        AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True).save_pretrained(out)
    except Exception as exc:  # tokenizer is optional for a weights-only merge
        print(f"[warn] tokenizer not saved: {exc}")

    model.save_pretrained(out, state_dict=model_sd, safe_serialization=True, max_shard_size="4GB")

    # Keep the plug-and-play artifacts next to the merged folder for reproducibility.
    shutil.copy2(args.adapter, out / "memory_adapter.pt")
    if args.calibration:
        shutil.copy2(args.calibration, out / "head_calibration.json")

    (out / "deltarecall_merge.json").write_text(json.dumps({
        "base_model": args.base_model,
        "adapter": Path(args.adapter).name,
        "calibration": Path(args.calibration).name if args.calibration else None,
        "merged_tensors": merged,
    }, indent=2) + "\n")
    print(f"[ok] merged {merged} memory tensors -> {out}")


if __name__ == "__main__":
    main()
