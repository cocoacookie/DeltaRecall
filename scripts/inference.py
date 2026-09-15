#!/usr/bin/env python3
# Copyright 2026 DeltaRecall Authors.
# SPDX-License-Identifier: Apache-2.0
"""Minimal plug-and-play inference demo for DeltaRecall.

This mirrors the real evaluation path used in the paper:

  1. load a frozen Qwen3.5 backbone,
  2. ``install_gras_hmc`` on the full-attention layers,
  3. ``load_memory_adapter`` (the trained ``memory_*`` weights only),
  4. optionally inject a semantic-calibrated head-weight file,
  5. chunked ``context-first`` prefill -> per-token question replay -> decode.

Example (Qwen3.5-9B, best 16K recipe S=128 / K=4096 / W=10174):

    CUDA_VISIBLE_DEVICES=0 python scripts/inference.py \
      --base-model Qwen/Qwen3.5-9B \
      --adapter    model/deltarecall-9b/memory_adapter.pt \
      --calibration calibration/deltarecall-9b-calibration.json \
      --select-budget 4096 --sliding-window 10174 --num-attention-sinks 128 \
      --prompt "When was the concept of AI introduced?"
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from deltarecall import install_gras_hmc, load_memory_adapter
from deltarecall.run_qwen35_longbench_v2 import text_position_ids


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", required=True, help="HF repo id or local path to the frozen Qwen3.5 backbone.")
    p.add_argument("--adapter", required=True, help="Path to a DeltaRecall memory_adapter.pt (memory_* weights only).")
    p.add_argument("--calibration", default=None, help="Optional semantic-calibrated head-weight JSON.")
    p.add_argument("--prompt", default="When was the concept of AI introduced?")
    p.add_argument("--select-budget", type=int, default=4096, help="Selective Cache capacity K per full-attention layer.")
    p.add_argument("--sliding-window", type=int, default=10174, help="Recent window length W.")
    p.add_argument("--num-attention-sinks", type=int, default=128, help="Attention-sink count S.")
    p.add_argument("--memory-projection-rank", type=int, default=64)
    p.add_argument("--memory-output-scale", type=float, default=0.1)
    p.add_argument("--recurrent-token-limit", type=int, default=2048)
    p.add_argument("--prefill-chunk-size", type=int, default=2048)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--shard-model", action="store_true", help="device_map=auto across visible GPUs (27B at long context).")
    return p.parse_args()


def load_head_calibration(path):
    if not path:
        return None
    raw = json.loads(Path(path).read_text())
    weights = raw.get("weights", raw)
    # Keys are absolute full-attention layer indices, e.g. "3","7",...,"31".
    return {int(k): v for k, v in weights.items()}


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    load_kwargs = dict(dtype=torch.bfloat16, trust_remote_code=True)
    if args.shard_model:
        model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map="auto", **load_kwargs).eval()
        input_device = model.get_input_embeddings().weight.device
    else:
        model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs).to("cuda").eval()
        input_device = torch.device("cuda")
    for config in (model.config, model.model.config):
        config._attn_implementation = "sdpa"

    # (1-3) Plug the three-resolution memory onto the frozen full-attention layers.
    wrappers = install_gras_hmc(
        model,
        select_budget=args.select_budget,
        sliding_window=args.sliding_window,
        num_attn_sinks=args.num_attention_sinks,
        demand_mode="recurrent",
        recurrent_token_limit=args.recurrent_token_limit,
        memory_output_scale=args.memory_output_scale,
        train_memory=True,  # materialize the trainable memory params so the adapter can load
        memory_projection_rank=args.memory_projection_rank,
        selector_mode="gras",
    )
    load_memory_adapter(wrappers, args.adapter)

    # (4) Optional semantic-calibrated head aggregation.
    calibration = load_head_calibration(args.calibration)
    if calibration is not None:
        for wrapper in wrappers:
            weights = calibration.get(wrapper.layer_idx)
            if weights is None:
                continue
            if len(weights) != wrapper.num_heads:
                raise ValueError(
                    f"calibration for layer {wrapper.layer_idx} has {len(weights)} weights, "
                    f"expected {wrapper.num_heads}"
                )
            wrapper.selector_head_weights = torch.tensor(weights, device=input_device, dtype=torch.float32)

    model.eval()
    for wrapper in wrappers:
        wrapper.eval()
        wrapper.train_memory = False  # inference: no student/teacher training path

    prompt = args.prompt
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    inputs = tokenizer(prompt, return_tensors="pt").to(input_device)
    history_length = int(inputs.input_ids.shape[1])

    for wrapper in wrappers:
        wrapper.reset_state()

    generated = []
    with torch.inference_mode():
        chunk = args.prefill_chunk_size
        if chunk > 0 and history_length > chunk:
            for wrapper in wrappers:
                wrapper.defer_compression = True
            outputs = None
            for start in range(0, history_length, chunk):
                end = min(start + chunk, history_length)
                outputs = model(
                    input_ids=inputs.input_ids[:, start:end],
                    position_ids=text_position_ids(start, end - start, input_device),
                    past_key_values=None if outputs is None else outputs.past_key_values,
                    cache_position=torch.arange(start, end, device=input_device),
                    use_cache=True, return_dict=True, logits_to_keep=1,
                )
            for wrapper in wrappers:
                wrapper.compress_deferred(outputs.past_key_values)
        else:
            outputs = model(
                input_ids=inputs.input_ids,
                position_ids=text_position_ids(0, history_length, input_device),
                use_cache=True, return_dict=True, logits_to_keep=1,
            )

        next_token = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
        for step in range(args.max_new_tokens):
            token_id = int(next_token.item())
            generated.append(token_id)
            if token_id == tokenizer.eos_token_id:
                break
            outputs = model(
                input_ids=next_token,
                position_ids=text_position_ids(history_length + step, 1, input_device),
                past_key_values=outputs.past_key_values,
                cache_position=torch.tensor([history_length + step], device=input_device, dtype=torch.long),
                use_cache=True, return_dict=True, logits_to_keep=1,
            )
            next_token = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)

    print(tokenizer.decode(generated, skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
