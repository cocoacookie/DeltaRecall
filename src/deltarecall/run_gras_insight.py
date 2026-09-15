#!/usr/bin/env python3
"""Controlled token/head-level Insight probe for GRAS-HMC.

The experiment keeps the Qwen3.5 backbone fixed and records the recurrent
GRAS demand before top-k selection.  Each context contains one true answer
span, optional answer-looking distractors, and ordinary filler.  We report
where high-demand tokens fall (answer, evidence neighbourhood, distractor,
or background) together with the actual greedy answer under the compressed
cache.  This is the online analogue of the semantic-head diagnostic in
CompressKV, while making the semantic label explicit at token level.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .qwen35_gras_hmc import install_gras_hmc, load_memory_adapter
from .run_qwen35_longbench_v2 import text_position_ids


INPUT_DEVICE = "cuda"
TRUE_CODE = "GRAS-NIAH-7429"
TRUE_NEEDLE = f"The unique secret code is {TRUE_CODE}."
FILLER = "This is ordinary background prose with no answer-bearing information."


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--context-tokens", default="4096,8192")
    p.add_argument("--positions", default="0.1,0.3,0.5,0.7,0.9")
    p.add_argument("--distractors", default="0,3")
    p.add_argument("--select-budget", type=int, default=512)
    p.add_argument("--sliding-window", type=int, default=2048)
    p.add_argument("--num-attention-sinks", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--replicates", type=int, default=1)
    p.add_argument("--replicate-indices", default=None)
    p.add_argument("--prefill-chunk-size", type=int, default=2048)
    p.add_argument("--memory-output-scale", type=float, default=0.1)
    p.add_argument("--memory-adapter", default=None)
    p.add_argument("--memory-projection-rank", type=int, default=0)
    p.add_argument("--shard-model", action="store_true")
    p.add_argument("--compact-output", action="store_true")
    p.add_argument(
        "--save-evidence-trace",
        action="store_true",
        help="store complete per-token, per-head scores for the contiguous evidence span",
    )
    return p.parse_args()


def char_span_positions(prompt: str, text: str, offsets: list[list[int]]) -> list[int]:
    start = prompt.find(text)
    if start < 0:
        return []
    end = start + len(text)
    return [i for i, (left, right) in enumerate(offsets) if right > start and left < end]


def build_prompt(
    position: float,
    target_tokens: int,
    distractors: int,
    variant: int = 0,
    block_count: int | None = None,
):
    # Blocks are deliberately repetitive so that lexical surface frequency is
    # separated from the semantic answer label.
    block_count = block_count or max(32, int(target_tokens / 18))
    insert_at = min(block_count - 1, max(0, int(round(position * (block_count - 1)))))
    blocks: list[str] = []
    true_block = (
        f"Evidence record {insert_at}: {TRUE_NEEDLE} "
        "This record is the authoritative source for the question."
    )
    decoy_blocks = {
        insert_at: true_block,
    }
    for j in range(distractors):
        # Place decoys away from the true span and use distinct codes so that
        # exact-answer correctness remains unambiguous.
        idx = int((j + 1) * block_count / (distractors + 1))
        if idx == insert_at:
            idx = (idx + 1) % block_count
        code = f"DECOY-{j + 1:02d}-9813"
        decoy_blocks[idx] = (
            f"Reference record {idx}: The unique secret code is {code}. "
            "This is a distractor record and must not be used as the answer."
        )
    filler_variants = [
        FILLER,
        "Routine service notes contain ordinary status updates and no answer-bearing facts.",
        "This archival paragraph describes normal operations; it is irrelevant to the query.",
        "The following maintenance details are background only and should be ignored.",
        "A generic report records uneventful activity without any useful code or entity.",
    ]
    filler = filler_variants[variant % len(filler_variants)]
    for i in range(block_count):
        blocks.append(decoy_blocks.get(i, f"Background section {i}: {filler} Batch {variant % 11}."))
    context = "\n".join(blocks)
    user = (
        "Please read the following text and answer the question.\n\n"
        f"<text>\n{context}\n</text>\n\n"
        "What is the unique secret code? Answer with the code only."
    )
    return user, insert_at, true_block


def history_boundary(prompt, offsets):
    marker = "\n</text>\n\nWhat is the unique secret code?"
    boundary = prompt.rindex(marker) + len("\n</text>")
    return next(
        index
        for index, (_, end) in enumerate(offsets)
        if end > boundary
    )


def encode_target_length_case(
    tokenizer,
    position: float,
    target_tokens: int,
    distractors: int,
    variant: int,
):
    block_count = max(32, int(target_tokens / 18))
    best = None
    for _ in range(6):
        user, insert_at, true_block = build_prompt(
            position,
            target_tokens,
            distractors,
            variant,
            block_count=block_count,
        )
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        encoded = tokenizer(
            prompt,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = encoded.offset_mapping
        current_length = history_boundary(prompt, offsets)
        error = abs(current_length - target_tokens)
        candidate = (
            error,
            user,
            insert_at,
            true_block,
            prompt,
            encoded.input_ids,
            offsets,
            current_length,
        )
        if best is None or error < best[0]:
            best = candidate
        if error <= 32:
            break
        next_count = max(
            32,
            int(round(block_count * target_tokens / max(current_length, 1))),
        )
        if next_count == block_count:
            break
        block_count = next_count
    return best[1:]


def generate(
    model,
    tokenizer,
    input_ids,
    history_length: int,
    max_new_tokens: int,
    prefill_chunk_size: int,
    wrappers,
):
    device = input_ids.device
    outputs = None
    if prefill_chunk_size > 0 and history_length > prefill_chunk_size:
        for wrapper in wrappers:
            wrapper.defer_compression = True
        for start in range(0, history_length, prefill_chunk_size):
            end = min(start + prefill_chunk_size, history_length)
            outputs = model(
                input_ids=input_ids[None, start:end],
                position_ids=text_position_ids(start, end - start, device),
                past_key_values=None if outputs is None else outputs.past_key_values,
                cache_position=torch.arange(start, end, device=device),
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
        for wrapper in wrappers:
            wrapper.compress_deferred(outputs.past_key_values)
    else:
        outputs = model(
            input_ids=input_ids[None, :history_length],
            position_ids=text_position_ids(0, history_length, device),
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
    for position in range(history_length, input_ids.numel()):
        outputs = model(
            input_ids=input_ids[position].view(1, 1),
            position_ids=text_position_ids(position, 1, device),
            cache_position=torch.tensor([position], device=device, dtype=torch.long),
            past_key_values=outputs.past_key_values,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
    next_id = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
    generated: list[int] = []
    for step in range(max_new_tokens):
        token_id = int(next_id.item())
        generated.append(token_id)
        if token_id == tokenizer.eos_token_id:
            break
        outputs = model(
            input_ids=next_id,
            position_ids=text_position_ids(input_ids.numel() + step, 1, device),
            cache_position=torch.tensor(
                [input_ids.numel() + step], device=device, dtype=torch.long
            ),
            past_key_values=outputs.past_key_values,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
        next_id = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
    return tokenizer.decode(generated, skip_special_tokens=True), generated


def percentile_rank(values: list[float], value: float) -> float:
    if not values:
        return 0.0
    return 100.0 * sum(v <= value for v in values) / len(values)


def summarize_layer(
    tokenizer,
    token_ids: list[int],
    head_scores: torch.Tensor,
    offset: int,
    labels: list[str],
    answer_positions: set[int],
    evidence_positions: set[int],
    budget: int,
    compact_output: bool,
    save_evidence_trace: bool,
):
    # head_scores is [candidate_tokens, heads].
    if head_scores.ndim != 2:
        raise ValueError(f"unexpected score shape {tuple(head_scores.shape)}")
    n_tokens, n_heads = head_scores.shape
    scores = head_scores.numpy()
    global_positions = [offset + i for i in range(n_tokens)]
    mean_scores = head_scores.mean(dim=-1).tolist()
    top_k = min(budget, n_tokens)
    top_indices = sorted(range(n_tokens), key=lambda i: mean_scores[i], reverse=True)[:top_k]

    def mass(label_set: set[str]):
        total = sum(max(0.0, mean_scores[i]) for i in range(n_tokens))
        return (
            sum(max(0.0, mean_scores[i]) for i in range(n_tokens) if labels[global_positions[i]] in label_set)
            / total
            if total > 0
            else 0.0
        )

    top_labels = [labels[global_positions[i]] for i in top_indices]
    top_counts = {name: top_labels.count(name) for name in ("A", "E", "D", "B", "S")}
    all_scores = [float(v) for v in mean_scores]
    label_means = {}
    for name in ("A", "E", "D", "B"):
        vals = [mean_scores[i] for i in range(n_tokens) if labels[global_positions[i]] == name]
        label_means[name] = sum(vals) / len(vals) if vals else None

    heads = []
    for h in range(n_heads):
        vals = [float(scores[i, h]) for i in range(n_tokens)]
        head_top = sorted(range(n_tokens), key=lambda i: vals[i], reverse=True)[: min(20, n_tokens)]
        head_labels = [labels[global_positions[i]] for i in head_top]
        head_row = {
            "head": h,
            "mean": sum(vals) / len(vals) if vals else 0.0,
            "max": max(vals) if vals else 0.0,
            "answer_mean": (
                sum(vals[i] for i in range(n_tokens) if global_positions[i] in answer_positions)
                / max(1, sum(global_positions[i] in answer_positions for i in range(n_tokens)))
            ),
            "evidence_mean": (
                sum(vals[i] for i in range(n_tokens) if global_positions[i] in evidence_positions)
                / max(1, sum(global_positions[i] in evidence_positions for i in range(n_tokens)))
            ),
            "background_mean": (
                sum(vals[i] for i in range(n_tokens) if labels[global_positions[i]] == "B")
                / max(1, sum(labels[global_positions[i]] == "B" for i in range(n_tokens)))
            ),
            "sink_mean": (
                sum(vals[i] for i in range(n_tokens) if labels[global_positions[i]] == "S")
                / max(1, sum(labels[global_positions[i]] == "S" for i in range(n_tokens)))
            ),
            "decoy_mean": (
                sum(vals[i] for i in range(n_tokens) if labels[global_positions[i]] == "D")
                / max(1, sum(labels[global_positions[i]] == "D" for i in range(n_tokens)))
            ),
            "top_label_counts": {
                name: head_labels.count(name)
                for name in ("A", "E", "D", "B", "S")
            },
        }
        if not compact_output:
            head_row.update(
                {
                    "top_positions": [global_positions[i] for i in head_top],
                    "top_scores": [vals[i] for i in head_top],
                    "top_text": [
                        tokenizer.decode(
                            [token_ids[global_positions[i]]],
                            skip_special_tokens=True,
                        )
                        for i in head_top
                    ],
                }
            )
        heads.append(head_row)
    result = {
        "offset": offset,
        "num_candidate_tokens": n_tokens,
        "num_heads": n_heads,
        "top_k": top_k,
        "top_k_label_counts": top_counts,
        "top_k_mass_A": sum(1 for i in top_indices if labels[global_positions[i]] == "A") / max(1, top_k),
        "top_k_mass_A_or_E": sum(1 for i in top_indices if labels[global_positions[i]] in ("A", "E")) / max(1, top_k),
        "top_k_mass_D": sum(1 for i in top_indices if labels[global_positions[i]] == "D") / max(1, top_k),
        "score_mass_A": mass({"A"}),
        "score_mass_A_or_E": mass({"A", "E"}),
        "score_mass_D": mass({"D"}),
        "label_means": label_means,
        "answer_max_percentile": percentile_rank(all_scores, max((mean_scores[i] for i in range(n_tokens) if global_positions[i] in answer_positions), default=0.0)),
        "heads": heads,
    }
    if save_evidence_trace:
        trace_positions = [
            position
            for position in sorted(evidence_positions)
            if offset <= position < offset + n_tokens
        ]
        trace_indices = [position - offset for position in trace_positions]
        result["evidence_trace"] = {
            "positions": trace_positions,
            "tokens": [
                tokenizer.decode([token_ids[position]], skip_special_tokens=True)
                for position in trace_positions
            ],
            "labels": [
                "A" if position in answer_positions else "E"
                for position in trace_positions
            ],
            "scores_by_head": {
                str(head): [float(scores[index, head]) for index in trace_indices]
                for head in range(n_heads)
            },
        }
    return result


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    global INPUT_DEVICE
    if args.shard_model:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        ).eval()
        INPUT_DEVICE = model.get_input_embeddings().weight.device
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, trust_remote_code=True
        ).to("cuda").eval()
        INPUT_DEVICE = torch.device("cuda")
    for config in (model.config, model.model.config):
        config._attn_implementation = "sdpa"
    wrappers = install_gras_hmc(
        model,
        select_budget=args.select_budget,
        sliding_window=args.sliding_window,
        num_attn_sinks=args.num_attention_sinks,
        demand_mode="recurrent",
        memory_output_scale=args.memory_output_scale,
        train_memory=args.memory_adapter is not None,
        memory_projection_rank=args.memory_projection_rank,
        sink_warm_start=True,
        selection_include_sinks=False,
        memory_write_mode="all",
    )
    if args.memory_adapter:
        load_memory_adapter(wrappers, args.memory_adapter)

    lengths = [int(x) for x in args.context_tokens.split(",") if x.strip()]
    positions = [float(x) for x in args.positions.split(",") if x.strip()]
    distractor_counts = [int(x) for x in args.distractors.split(",") if x.strip()]
    replicates = (
        [int(value) for value in args.replicate_indices.split(",") if value.strip()]
        if args.replicate_indices
        else list(range(args.replicates))
    )
    output_path = Path(args.output)
    jsonl_path = output_path.with_suffix(".jsonl")
    completed = {}
    if jsonl_path.is_file():
        for line in jsonl_path.read_text().splitlines():
            try:
                row = json.loads(line)
                key = (
                    int(row["target_tokens"]),
                    float(row["position"]),
                    int(row["distractors"]),
                    int(row["replicate"]),
                )
                completed[key] = row
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
    total = len(lengths) * len(positions) * len(distractor_counts) * len(replicates)
    run_index = 0
    for target_tokens in lengths:
        for position in positions:
            for distractors in distractor_counts:
                for replicate in replicates:
                    run_index += 1
                    key = (target_tokens, position, distractors, replicate)
                    if key in completed:
                        continue
                    variant = args.seed + replicate
                    (
                        _user,
                        _,
                        true_block,
                        prompt,
                        input_ids,
                        offsets,
                        history_length,
                    ) = encode_target_length_case(
                        tokenizer,
                        position,
                        target_tokens,
                        distractors,
                        variant,
                    )
                    token_ids = input_ids
                    prompt_ids = torch.tensor(
                        input_ids,
                        dtype=torch.long,
                        device=INPUT_DEVICE,
                    )
                    answer_positions = set(char_span_positions(prompt, TRUE_NEEDLE, offsets))
                    evidence_span = set(char_span_positions(prompt, true_block, offsets))
                    if not evidence_span and answer_positions:
                        center = min(answer_positions)
                        evidence_span = set(range(max(0, center - 24), min(len(token_ids), center + len(answer_positions) + 24)))
                    labels = ["B"] * len(token_ids)
                    for i in range(min(args.num_attention_sinks, len(labels))):
                        labels[i] = "S"
                    for i in evidence_span:
                        if i < len(labels) and labels[i] != "S":
                            labels[i] = "E"
                    for i in answer_positions:
                        if i < len(labels) and labels[i] != "S":
                            labels[i] = "A"
                    # Mark answer-looking decoys by exact code spans.
                    for j in range(distractors):
                        decoy = f"DECOY-{j + 1:02d}-9813"
                        for i in char_span_positions(prompt, decoy, offsets):
                            if labels[i] not in ("A", "S"):
                                labels[i] = "D"
                    for wrapper in wrappers:
                        wrapper.reset_state()
                    started = time.perf_counter()
                    with torch.inference_mode():
                        answer, generated_ids = generate(
                            model,
                            tokenizer,
                            prompt_ids,
                            history_length,
                            args.max_new_tokens,
                            args.prefill_chunk_size,
                            wrappers,
                        )
                    layer_rows = []
                    for wrapper in wrappers:
                        if wrapper.prefill_head_scores is None:
                            continue
                        raw = wrapper.prefill_head_scores[0]
                        layer = summarize_layer(
                            tokenizer,
                            token_ids,
                            raw,
                            wrapper.prefill_score_offset,
                            labels,
                            answer_positions,
                            evidence_span,
                            args.select_budget,
                            args.compact_output,
                            args.save_evidence_trace,
                        )
                        layer.update({
                            "layer": wrapper.layer_idx,
                            "selected_hit_A": bool(set(wrapper.select_positions) & answer_positions),
                            "selected_hit_A_or_E": bool(set(wrapper.select_positions) & (answer_positions | evidence_span)),
                        })
                        if not args.compact_output:
                            layer["selected_positions"] = wrapper.select_positions
                        layer_rows.append(layer)
                    row = {
                        "run_index": run_index,
                        "runs_total": total,
                        "target_tokens": target_tokens,
                        "prompt_tokens": len(token_ids),
                        "history_tokens": history_length,
                        "query_tokens": len(token_ids) - history_length,
                        "position": position,
                        "distractors": distractors,
                        "replicate": replicate,
                        "answer_positions": sorted(answer_positions),
                        "evidence_positions": sorted(evidence_span),
                        "answer": answer,
                        "correct": TRUE_CODE in answer,
                        "generated_token_ids": generated_ids,
                        "seconds": time.perf_counter() - started,
                        "layers": layer_rows,
                    }
                    completed[key] = row
                    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
                    with jsonl_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    print(json.dumps({
                        "run": f"{run_index}/{total}",
                        "tokens": len(token_ids),
                        "position": position,
                        "distractors": distractors,
                        "replicate": replicate,
                        "answer": answer,
                        "correct": TRUE_CODE in answer,
                        "seconds": row["seconds"],
                    }, ensure_ascii=False), flush=True)
                    del prompt_ids
                    gc.collect()
                    torch.cuda.empty_cache()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [completed[key] for key in sorted(completed)]
    if len(rows) != total:
        raise RuntimeError(f"incomplete insight shard: {len(rows)} != {total}")
    payload = {
        "model": args.model,
        "memory_adapter": args.memory_adapter,
        "memory_projection_rank": args.memory_projection_rank,
        "select_budget": args.select_budget,
        "sliding_window": args.sliding_window,
        "num_attention_sinks": args.num_attention_sinks,
        "context_tokens": lengths,
        "positions": positions,
        "distractors": distractor_counts,
        "replicate_indices": replicates,
        "prefill_protocol": "context-first",
        "prefill_chunk_size": args.prefill_chunk_size,
        "signal": "recurrent beta times residual norm, head-level before mean",
        "labels": {"A": "true answer span", "E": "authoritative evidence neighbourhood", "D": "answer-looking distractor", "B": "ordinary background", "S": "attention sink prefix"},
        "rows": rows,
    }
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(output_path)
    print(json.dumps({"output": str(output_path), "rows": len(rows), "correct": sum(r["correct"] for r in rows)}))


if __name__ == "__main__":
    main()
