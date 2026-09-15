#!/usr/bin/env python3
"""Run one LongBench v2 record with the Qwen3.5 hybrid cache."""

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .qwen35_gras_hmc import install_gras_hmc


def build_prompt(record):
    return "\n".join(
        [
            "Please read the following text and answer the question below.",
            "",
            "<text>",
            record["context"].strip(),
            "</text>",
            "",
            f"What is the correct answer to this question: {record['question'].strip()}",
            "Choices:",
            f"(A) {record['choice_A'].strip()}",
            f"(B) {record['choice_B'].strip()}",
            f"(C) {record['choice_C'].strip()}",
            f"(D) {record['choice_D'].strip()}",
            "",
            'Format your response as follows: "The correct answer is (insert answer here)".',
        ]
    )


def truncate_context(record, tokenizer, max_context_tokens):
    if max_context_tokens <= 0:
        return record
    context_ids = tokenizer.encode(record["context"], add_special_tokens=False)
    if len(context_ids) <= max_context_tokens:
        return record
    left = max_context_tokens // 2
    shortened = dict(record)
    shortened["context"] = tokenizer.decode(
        context_ids[:left] + context_ids[-(max_context_tokens - left) :],
        skip_special_tokens=True,
    )
    return shortened


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["fullkv", "gras"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sliding-window", type=int, default=4096)
    parser.add_argument("--num-attention-sinks", type=int, default=128)
    parser.add_argument("--select-budget", type=int, default=512)
    parser.add_argument("--layer-budget-file", default=None)
    parser.add_argument("--demand-mode", choices=["proxy", "recurrent"], default="proxy")
    parser.add_argument(
        "--recurrent-token-limit", type=int, default=2048,
        help="Chunk size for recurrent demand replay",
    )
    parser.add_argument("--memory-output-scale", type=float, default=0.0)
    parser.add_argument("--max-context-tokens", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--no-chat-template", action="store_true")
    return parser.parse_args()


def extract_answer(text):
    match = re.search(r"The correct answer is \(([A-D])\)", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"The correct answer is ([A-D])\b", text, re.IGNORECASE)
    return match.group(1).upper() if match else None


def text_position_ids(start, length, device):
    positions = torch.arange(start, start + length, device=device, dtype=torch.long)
    return positions.view(1, 1, -1).expand(3, 1, -1)


def tensor_bytes(value):
    return int(value.numel() * value.element_size()) if isinstance(value, torch.Tensor) else 0


def native_linear_state_bytes(cache):
    total = 0
    for layer in cache.layers:
        for name in ("conv_states", "recurrent_states"):
            states = getattr(layer, name, None)
            if states:
                total += sum(tensor_bytes(value) for value in states.values() if value is not None)
    return total


def load_budgets(path):
    if not path:
        return None
    data = json.loads(Path(path).read_text())
    return [int(value) for value in data.get("budgets", data)]


def main():
    args = parse_args()
    layer_budgets = load_budgets(args.layer_budget_file)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda").eval()

    # Use the native hybrid model only; the adapter replaces full-attention
    # modules after the base weights have loaded.
    for config in (model.config, model.model.config):
        config._attn_implementation = "sdpa"
    wrappers = []
    if args.method == "gras":
        wrappers = install_gras_hmc(
            model,
            select_budget=args.select_budget,
            sliding_window=args.sliding_window,
            num_attn_sinks=args.num_attention_sinks,
            layer_budgets=layer_budgets,
            demand_mode=args.demand_mode,
            recurrent_token_limit=args.recurrent_token_limit,
            memory_output_scale=args.memory_output_scale,
        )

    records = json.loads(Path(args.data).read_text())
    record = truncate_context(records[args.index], tokenizer, args.max_context_tokens)
    user_prompt = build_prompt(record)
    prompt = user_prompt
    if tokenizer.chat_template and not args.no_chat_template:
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": args.enable_thinking,
        }
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            **template_kwargs,
        )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    prompt_length = int(inputs.input_ids.shape[1])
    generated = []

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    with torch.inference_mode():
        outputs = model(
            **inputs,
            position_ids=text_position_ids(0, prompt_length, inputs.input_ids.device),
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
        next_token = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
        for step in range(args.max_new_tokens):
            token_id = int(next_token.item())
            generated.append(token_id)
            if token_id == tokenizer.eos_token_id:
                break
            outputs = model(
                input_ids=next_token,
                position_ids=text_position_ids(
                    prompt_length + step, 1, next_token.device
                ),
                past_key_values=outputs.past_key_values,
                cache_position=torch.tensor(
                    [prompt_length + step], device=next_token.device, dtype=torch.long
                ),
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
            next_token = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    cache = outputs.past_key_values
    exact_bytes = 0
    layers = {}
    for layer_index, layer_cache in enumerate(cache.layers):
        keys = getattr(layer_cache, "keys", None)
        values = getattr(layer_cache, "values", None)
        if keys is None or values is None:
            continue
        exact_bytes += tensor_bytes(keys) + tensor_bytes(values)
        layers[str(layer_index)] = {"cache_length": int(keys.shape[-2])}

    if wrappers:
        for wrapper in wrappers:
            item = layers.setdefault(str(wrapper.layer_idx), {})
            item["select_budget"] = wrapper.select_budget
            item["select_count"] = len(wrapper.select_positions)
            item["window_count"] = len(wrapper.window_positions)
            if wrapper.last_demand is not None and wrapper.last_demand.numel():
                demand = wrapper.last_demand.reshape(-1)
                item["demand_mean"] = float(demand.mean())
                item["demand_p90"] = float(torch.quantile(demand, 0.9))
            item["select_positions"] = wrapper.select_positions
            item["window_positions"] = wrapper.window_positions

    generated_text = tokenizer.decode(generated, skip_special_tokens=True)
    predicted = extract_answer(generated_text)
    gras_state_bytes = sum(tensor_bytes(wrapper.mem_state) for wrapper in wrappers)
    gate_metadata_bytes = sum(
        tensor_bytes(wrapper.beta_cache) + tensor_bytes(wrapper.alpha_cache)
        for wrapper in wrappers
    )
    result = {
        "method": args.method,
        "model": args.model,
        "data_index": args.index,
        "record_id": record["_id"],
        "domain": record["domain"],
        "sub_domain": record["sub_domain"],
        "difficulty": record["difficulty"],
        "length_category": record["length"],
        "gold_answer": record["answer"],
        "question": record["question"],
        "prompt_length": prompt_length,
        "generated_text": generated_text,
        "predicted_answer": predicted,
        "correct": predicted == record["answer"],
        "sliding_window": args.sliding_window if args.method == "gras" else None,
        "num_attention_sinks": args.num_attention_sinks if args.method == "gras" else None,
        "select_budget": args.select_budget if args.method == "gras" else 0,
        "layer_budget_file": args.layer_budget_file if args.method == "gras" else None,
        "layer_budgets": layer_budgets if args.method == "gras" else None,
        "demand_mode": args.demand_mode if args.method == "gras" else None,
        "memory_output_scale": args.memory_output_scale if args.method == "gras" else 0.0,
        "enable_thinking": args.enable_thinking,
        "full_attention_layers": [
            int(index) for index, layer in enumerate(model.model.layers)
            if layer.block_type == "full_attention"
        ],
        "prefill_decode_seconds": elapsed,
        "exact_kv_bytes": exact_bytes,
        "native_linear_state_bytes": native_linear_state_bytes(cache),
        "gras_state_bytes": gras_state_bytes,
        "gate_metadata_bytes": gate_metadata_bytes,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "layers": layers,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    output_path.with_suffix(".prompt.txt").write_text(prompt)
    print(json.dumps({
        key: result[key]
        for key in (
            "method", "data_index", "record_id", "gold_answer",
            "predicted_answer", "correct", "prompt_length",
            "exact_kv_bytes", "native_linear_state_bytes", "gras_state_bytes",
            "gate_metadata_bytes", "peak_cuda_bytes", "prefill_decode_seconds",
        )
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
