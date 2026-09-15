# DeltaRecall: Write-Demand-Driven Plug-and-Play Hybrid Memory for Long-Context LLMs

DeltaRecall attaches a **three-resolution memory** to the *frozen* full-attention
layers of a pretrained hybrid Transformer (Qwen3.5), leaving the model's native
linear-attention / Gated DeltaNet layers untouched. It is plug-and-play: the
backbone weights are never modified, and only a small per-layer memory adapter
(the `memory_*` parameters) is trained and shipped.

This repo provides the inference, training, and evaluation code for DeltaRecall,
together with the best 9B / 27B adapters and their head-calibration files.

## :bell: News

- **:fire: \[2026-09] DeltaRecall released.**

## Introduction

A full key–value (KV) cache stores every historical token and grows linearly
with context; a recent window or a single compressed state has fixed cost but
covers only local access *or* global history in isolation. DeltaRecall organizes
three complementary memories at **every full-attention layer**:

- **Active Cache** — attention sinks and the recent window (continuous local access);
- **Selective Cache** — a few remote exact KV entries the compressed state cannot reconstruct (addressable evidence);
- **Compressive Memory** — a fixed-size Gated DeltaNet state that covers the growing history.

**Key insight.** The compressor already exposes its *write demand* when it
absorbs a token. We combine the Gated DeltaNet write gate `β` and the
value-prediction residual `e` into a native, query-free demand `s = β·‖e‖₂`,
which — after intra-layer normalization and semantic-calibrated head
aggregation — becomes the routing score that decides *which* remote tokens stay
exact in the Selective Cache. The signal is available at prefill, needs no
question and no auxiliary scorer.

On Qwen3.5-9B/27B, DeltaRecall attains the best Overall among compression methods
across the LongBench v2 budget settings, and keeps its advantage on multi-turn
SCBench and ultra-long RULER.

## Usage

### Installation

**Default environment:** Python 3.11, PyTorch 2.5+, a Qwen3.5-capable
`transformers` (the `qwen3_5` backbone must be importable via
`trust_remote_code=True`).

```bash
# 1. Clone and enter the repo
git clone https://github.com/<your-org>/DeltaRecall.git
cd DeltaRecall

# 2. Install DeltaRecall in editable mode
pip install -e ".[train,eval]"
# or: pip install -r requirements.txt
```

### Model Zoo

Adapters store only the per-layer `memory_*` parameters (rank-64 low-rank
readout, per-head write gate/decay, and a learned fusion scale). The frozen
Qwen3.5 backbone is downloaded separately from its official source.

|  base model |                                         adapter                                       |                                           calibration                                          |
| :---------: |  :----------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------: |
|  Qwen3.5-9B |   [`model/deltarecall-9b/memory_adapter.pt`](model/deltarecall-9b/memory_adapter.pt)  |  [`calibration/deltarecall-9b-calibration.json`](calibration/deltarecall-9b-calibration.json)  |
| Qwen3.5-27B |  [`model/deltarecall-27b/memory_adapter.pt`](model/deltarecall-27b/memory_adapter.pt) | [`calibration/deltarecall-27b-calibration.json`](calibration/deltarecall-27b-calibration.json) |

### Plug-and-Play Inference

DeltaRecall needs no weight merging — attach the memory at load time:

```python
from transformers import AutoModelForCausalLM
from deltarecall import install_gras_hmc, load_memory_adapter

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-9B", trust_remote_code=True, dtype="bfloat16",
).to("cuda").eval()
model.config._attn_implementation = model.model.config._attn_implementation = "sdpa"

wrappers = install_gras_hmc(
    model,
    select_budget=4096, sliding_window=10174, num_attn_sinks=128,
    demand_mode="recurrent", memory_projection_rank=64, memory_output_scale=0.1,
    train_memory=True,          # materialize memory_* params so the adapter can load
)
load_memory_adapter(wrappers, "model/deltarecall-9b/memory_adapter.pt")
for w in wrappers:              # switch to inference
    w.eval(); w.train_memory = False
```

Or run the bundled demo end to end (chunked context-first prefill + decode):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/inference.py \
  --base-model Qwen/Qwen3.5-9B \
  --adapter    model/deltarecall-9b/memory_adapter.pt \
  --calibration calibration/deltarecall-9b-calibration.json \
  --select-budget 4096 --sliding-window 10174 --num-attention-sinks 128 \
  --prompt "When was the concept of AI introduced?"
```

For Qwen3.5-27B, add `--shard-model` (device\_map=auto across GPUs) and use
`--select-budget 6016 --sliding-window 2048`.

### (Optional) Merge into a Standalone Checkpoint

If you prefer to ship a single folder instead of backbone + adapter:

```bash
python scripts/merge_weights.py \
  --base-model  Qwen/Qwen3.5-9B \
  --adapter     model/deltarecall-9b/memory_adapter.pt \
  --calibration calibration/deltarecall-9b-calibration.json \
  --output-path merged_ckpt/DeltaRecall-9B
```

### Training

The backbone stays frozen; a teacher pass uses native full attention while a
student pass uses the sink+window attention plus the all-write recurrent memory,
and only the `memory_*` parameters are optimized (KL self-distillation with a
small cross-entropy term). Capacity randomization samples `(K, W)` budgets so one
adapter serves multiple exact-history ranges.

```bash
python -m deltarecall.train_qwen35_memory \
  --model Qwen/Qwen3.5-9B \
  --data  <training_data_dir_or_parquet> \
  --output model/deltarecall-9b/memory_adapter.pt \
  --steps 160 --seq-len 32768 --sliding-window 2048 \
  --memory-projection-rank 64 --memory-output-scale 0.1 \
  --budget-options 1024:2944,2048:6016,3072:9088,4096:10174 \
  --training-protocol context-first-answer --ce-weight 0.25 --max-retained-fraction 0.85
```

Key arguments:

- `--budget-options` — comma-separated `K:W` pairs sampled per step (capacity randomization).
- `--memory-projection-rank` — low-rank readout rank (default 64).
- `--head-calibration` — optional head-weight file applied to the selector during training.
- `--training-protocol` — `context-first-answer` (evidence-cropped) or `sequence`.

### Head Calibration

Semantic-calibrated head weights are fit offline from controlled write-demand
diagnostics (`run_gras_insight.py` → `fit_gras_head_calibration.py` →
`fit_gras_hard_negative_calibration.py`). The shipped files are already fit; a
calibration JSON maps each absolute full-attention layer index to a per-head
weight vector (9B: 8 layers × 16 heads; 27B: 16 layers × 24 heads).

### Evaluation

`run_qwen35_longbench_v2.py` scores a single LongBench v2 record with the
plug-and-play cache and reports the real persistent-byte breakdown (exact KV,
native linear state, recurrent memory, gate metadata). It supports `fullkv` and
`gras`; loop over `--index` to cover the benchmark and aggregate `correct`:

```bash
# LongBench v2, single record (9B, best 16K recipe)
python -m deltarecall.run_qwen35_longbench_v2 \
  --method gras --model Qwen/Qwen3.5-9B --data <longbench_v2>/data.json \
  --index 0 --output results/longbench_9b_16k/rec_0000.json \
  --max-context-tokens 131072 --max-new-tokens 128 \
  --num-attention-sinks 128 --select-budget 4096 --sliding-window 10174 \
  --demand-mode recurrent --memory-output-scale 0.1

# FullKV reference on the same record
python -m deltarecall.run_qwen35_longbench_v2 \
  --method fullkv --model Qwen/Qwen3.5-9B --data <longbench_v2>/data.json \
  --index 0 --output results/longbench_9b_fullkv/rec_0000.json
```

> The plug-and-play adapter path (`install_gras_hmc` + `load_memory_adapter`) is
> the recommended way to reproduce the paper numbers on other benchmarks
> (SCBench, RULER); wire the same three-cache setup into that benchmark's own
> prompt/eval loop. The external cache-compression baselines used in the paper
> (window / AHN / Compactor / CompressKV) are not included in this release.


## Acknowledgments

DeltaRecall builds on the plug-and-play memory paradigm of
[AHN](https://github.com/ByteDance-Seed/AHN) and the Gated DeltaNet recurrence
from [flash-linear-attention](https://github.com/fla-org/flash-linear-attention).
We thank the developers of [🤗 transformers](https://github.com/huggingface/transformers)
for the Qwen3.5 backbone integration.

## License

Released under the [Apache License 2.0](LICENSE). The Qwen3.5 backbone weights
and any benchmark datasets remain under their respective original licenses.
