#!/usr/bin/env python3
"""Train the small Qwen3.5 GRAS-HMC memory readouts.

The pretrained Qwen3.5 backbone stays frozen.  A teacher pass uses the native
full-attention layers; a student pass uses sink-plus-window attention and the
all-write recurrent memory.  Only the memory scale, gates, decay and optional
low-rank readout are optimized.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
import random
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import pyarrow.parquet as pq
from transformers import AutoModelForCausalLM, AutoTokenizer

from .qwen35_gras_hmc import install_gras_hmc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, help="Parquet file or directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--loss-tokens", type=int, default=128)
    parser.add_argument("--sliding-window", type=int, default=2048)
    parser.add_argument("--num-attention-sinks", type=int, default=128)
    parser.add_argument("--memory-projection-rank", type=int, default=64)
    parser.add_argument("--memory-output-scale", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-examples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--training-protocol", choices=["sequence", "context-first-answer"], default="sequence")
    parser.add_argument("--head-calibration", default=None)
    parser.add_argument("--shard-model", action="store_true",
                        help="Load frozen backbone with device_map=auto across all visible GPUs "
                             "(needed for 27B at 32K seq on a single 80GB card).")
    parser.add_argument("--ce-weight", type=float, default=0.0)
    parser.add_argument("--min-history-tokens", type=int, default=0)
    parser.add_argument("--max-retained-fraction", type=float, default=1.0,
                        help="Require sampled budgets to evict history during training")
    parser.add_argument(
        "--budget-options", default=None,
        help="Comma-separated K:W pairs; sample one per step for budget augmentation",
    )
    return parser.parse_args()


def distributed_context():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, torch.device("cuda", local_rank)


def parquet_paths(path):
    source = Path(path)
    if source.is_dir():
        return sorted(source.glob("*.parquet"))
    if "*" in path:
        return [Path(item) for item in sorted(glob.glob(path))]
    return [source]


def tokenized_examples(tokenizer, path, seq_len, min_len, max_examples):
    examples = []
    tail_len = min(512, seq_len // 4)
    for parquet_path in parquet_paths(path):
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches():
            for row in batch.to_pylist():
                prompt = tokenizer.apply_chat_template(
                    row["messages"], tokenize=False, add_generation_prompt=False
                )
                token_ids = tokenizer(prompt, add_special_tokens=False).input_ids
                if len(token_ids) < min_len:
                    continue
                if len(token_ids) > seq_len:
                    token_ids = token_ids[: seq_len - tail_len] + token_ids[-tail_len:]
                examples.append(torch.tensor(token_ids, dtype=torch.long))
                if len(examples) >= max_examples:
                    return examples
    return examples


def position_ids(length, device):
    return torch.arange(length, device=device, dtype=torch.long).view(1, 1, -1).expand(3, 1, -1)


def answer_examples(tokenizer, path, seq_len, min_len, max_examples):
    """Keep the query and answer intact; crop only the supplied history."""
    examples = []
    for parquet_path in parquet_paths(path):
        for batch in pq.ParquetFile(parquet_path).iter_batches(batch_size=64):
            for row in batch.to_pylist():
                messages = row["messages"]
                if messages[-1]["role"] != "assistant":
                    continue
                prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
                marker = prompt.rfind("\n\nQuestion:")
                if marker < 0:
                    continue
                answer = messages[-1]["content"]
                history = tokenizer(prompt[:marker], add_special_tokens=False).input_ids
                query = tokenizer(prompt[marker:], add_special_tokens=False).input_ids
                target = tokenizer(answer + tokenizer.eos_token, add_special_tokens=False).input_ids
                capacity = seq_len - len(query) - len(target)
                if capacity <= min_len or len(history) < min_len:
                    continue
                if len(history) > capacity:
                    # This retrieval recipe uses an evidence-containing crop,
                    # with answer depth varied independently of the KV budget.
                    sink_length = 128
                    body_capacity = capacity - sink_length
                    location = prompt[:marker].lower().find(answer.strip().lower()) if answer.strip() else -1
                    if location >= 0:
                        anchor = len(tokenizer(prompt[:location], add_special_tokens=False).input_ids)
                    else:
                        anchor = random.randint(sink_length, len(history))
                    start = max(sink_length, min(len(history) - body_capacity,
                        anchor - random.randint(0, max(0, body_capacity - len(target)))))
                    history = history[:sink_length] + history[start:start + body_capacity]
                ids = torch.tensor(history + query + target, dtype=torch.long)
                examples.append((ids, len(history), len(target)))
                if len(examples) >= max_examples:
                    return examples
    return examples


def trainable_parameters(wrappers):
    parameters = []
    for wrapper in wrappers:
        for name, parameter in wrapper.named_parameters():
            if name.startswith("memory_"):
                parameter.requires_grad_(True)
                parameters.append(parameter)
    return parameters


def reduce_gradients(parameters, world_size):
    if world_size == 1:
        return
    for parameter in parameters:
        if parameter.grad is not None:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(world_size)


def save_adapter(path, wrappers, losses, args, rank):
    if rank != 0:
        return
    payload = {
        "layers": {
            str(wrapper.layer_idx): {
                name: value.detach().cpu()
                for name, value in wrapper.state_dict().items()
                if name.startswith("memory_")
            }
            for wrapper in wrappers
        },
        "config": vars(args),
        "losses": losses,
        "memory_scales": {
            str(wrapper.layer_idx): float(wrapper.memory_output_scale.detach().cpu())
            for wrapper in wrappers
        },
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    output.with_suffix(".json").write_text(json.dumps(payload["config"] | {
        "losses": losses,
        "memory_scales": payload["memory_scales"],
    }, indent=2) + "\n")


def main():
    args = parse_args()
    budget_options = None
    if args.budget_options:
        budget_options = [
            tuple(int(value) for value in item.split(":", 1))
            for item in args.budget_options.split(",")
        ]
    rank, world_size, local_rank, device = distributed_context()
    # Give each DDP rank a distinct stream so it contributes a different
    # example/budget instead of repeating rank 0's random choices.
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    loader = answer_examples if args.training_protocol == "context-first-answer" else tokenized_examples
    examples = loader(
        tokenizer,
        args.data,
        args.seq_len,
        max(args.min_history_tokens, args.num_attention_sinks + args.sliding_window + args.loss_tokens),
        args.max_examples,
    )
    if not examples:
        raise RuntimeError("no sufficiently long training examples found")

    eligible_examples = {}
    if budget_options and args.max_retained_fraction < 1.0:
        for budget in budget_options:
            capacity = args.num_attention_sinks + sum(budget)
            eligible = [i for i, example in enumerate(examples)
                        if capacity <= args.max_retained_fraction * (
                            example[1] if args.training_protocol == "context-first-answer" else len(example))]
            if not eligible:
                raise ValueError(f"budget {budget} has no history long enough to train compression")
            eligible_examples[budget] = eligible
    if rank == 0:
        print(json.dumps({"examples": len(examples), "seq_len": args.seq_len,
                          "budget_eligible_examples": {f"{k}:{w}": len(ids)
                                                       for (k, w), ids in eligible_examples.items()}}), flush=True)

    if args.shard_model:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        input_device = next(model.parameters()).device
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device)
        input_device = device
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    for config in (model.config, model.model.config):
        config._attn_implementation = "sdpa"
    wrappers = install_gras_hmc(
        model,
        select_budget=512,
        sliding_window=args.sliding_window,
        num_attn_sinks=args.num_attention_sinks,
        demand_mode="recurrent",
        recurrent_token_limit=2048,
        memory_output_scale=args.memory_output_scale,
        train_memory=True,
        memory_projection_rank=args.memory_projection_rank,
    )
    if args.head_calibration:
        weights = json.loads(Path(args.head_calibration).read_text())["weights"]
        for wrapper in wrappers:
            if str(wrapper.layer_idx) in weights:
                wrapper.selector_head_weights = torch.tensor(weights[str(wrapper.layer_idx)], device=device, dtype=torch.float32)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parameters = trainable_parameters(wrappers)
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.0)
    losses = []
    start_time = time.perf_counter()

    for step in range(args.steps):
        budget = random.choice(budget_options) if budget_options else None
        example_index = (random.choice(eligible_examples[budget]) if eligible_examples
                         else (step * world_size + rank) % len(examples))
        example = examples[example_index]
        answer_length = None
        if args.training_protocol == "context-first-answer":
            example, context_length, answer_length = example
            for wrapper in wrappers:
                wrapper.training_context_length = context_length
        input_ids = example.unsqueeze(0).to(input_device)
        history_tokens = context_length if answer_length is not None else input_ids.shape[1]
        positions = position_ids(input_ids.shape[1], input_device)
        logits_to_keep = answer_length + 1 if answer_length is not None else args.loss_tokens
        if budget is not None:
            select_budget, sliding_window = budget
            for wrapper in wrappers:
                wrapper.select_budget = select_budget
                wrapper.sliding_window = sliding_window
                wrapper._local_mask = None

        model.eval()
        with torch.no_grad():
            teacher = model(
                input_ids=input_ids,
                position_ids=positions,
                use_cache=False,
                return_dict=True,
                logits_to_keep=logits_to_keep,
            ).logits.float()

        model.train()
        student = model(
            input_ids=input_ids,
            position_ids=positions,
            use_cache=False,
            return_dict=True,
            logits_to_keep=logits_to_keep,
        ).logits.float()
        if answer_length is not None:
            teacher = teacher[:, :-1]
            student = student[:, :-1]
        teacher_probs = teacher.softmax(dim=-1)
        student_log_probs = student.log_softmax(dim=-1)
        loss = (teacher_probs * (teacher_probs.clamp_min(1e-8).log() - student_log_probs)).sum(dim=-1).mean()
        if answer_length is not None and args.ce_weight:
            loss = loss + args.ce_weight * F.cross_entropy(
                student.reshape(-1, student.shape[-1]), input_ids[:, -answer_length:].reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        reduce_gradients(parameters, world_size)
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if rank == 0:
            capacity = args.num_attention_sinks + wrappers[0].select_budget + wrappers[0].sliding_window
            print(json.dumps({"step": step + 1, "loss": losses[-1],
                              "history_tokens": history_tokens,
                              "select_budget": wrappers[0].select_budget,
                              "sliding_window": wrappers[0].sliding_window,
                              "retained_fraction": min(capacity, history_tokens) / history_tokens,
                              "evicted_tokens": max(0, history_tokens - capacity)}, ensure_ascii=False), flush=True)

    save_adapter(args.output, wrappers, losses, args, rank)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    if rank == 0:
        print(json.dumps({
            "steps": args.steps,
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "seconds": time.perf_counter() - start_time,
            "output": args.output,
            "world_size": world_size,
            "local_rank": local_rank,
        }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
