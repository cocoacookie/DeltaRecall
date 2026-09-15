#!/usr/bin/env python3
"""Fit offline head-role weights from the controlled GRAS calibration split.

Layers and per-layer head count are derived from the insight data itself
(the full-attention layers wrapped by install_gras_hmc), so this works for
any model geometry (Qwen3.5-9B: 8 layers x 16 heads; 27B: 16 layers x 24 heads).
Defaults reproduce the original 9B artifact byte-for-byte.
"""

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).parent


def answer_is_candidate(row, layer):
    return any(layer["offset"] <= p < layer["offset"] + layer["num_candidate_tokens"] for p in row["answer_positions"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default=str(ROOT / "gras_insight_500.json"))
    parser.add_argument("--output", default=str(ROOT / "gras_head_calibration.json"))
    parser.add_argument("--layers", default=None,
                        help="comma-separated layer ids; default = all layers present in the data")
    args = parser.parse_args()
    ROWS = json.loads(Path(args.rows).read_text())["rows"]
    if args.layers:
        LAYERS = [int(x) for x in args.layers.split(",") if x.strip()]
    else:
        LAYERS = sorted({int(layer["layer"]) for row in ROWS for layer in row["layers"]})
    weights = {}
    summary = {}
    for layer_id in LAYERS:
        calibration, heldout = [], []
        for row in ROWS:
            layer = next(x for x in row["layers"] if x["layer"] == layer_id)
            if not answer_is_candidate(row, layer):
                continue
            (calibration if row["replicate"] < 15 else heldout).append(layer["heads"])
        head_count = len(calibration[0])
        raw_weights = []
        head_stats = []
        for head in range(head_count):
            deltas = [
                (heads[head]["answer_mean"] + heads[head]["evidence_mean"]) / 2.0
                - heads[head]["background_mean"]
                for heads in calibration
            ]
            mean_delta = statistics.mean(deltas)
            scale = statistics.pstdev(deltas) + 1e-6
            weight = max(0.0, mean_delta / scale)
            raw_weights.append(weight)
            head_stats.append({"head": head, "mean_semantic_minus_background": mean_delta, "std": scale, "raw_weight": weight})
        if sum(raw_weights) == 0:
            raw_weights = [1.0] * head_count
        normalized = [x / sum(raw_weights) for x in raw_weights]
        for stat, value in zip(head_stats, normalized):
            stat["weight"] = value
        weights[str(layer_id)] = normalized

        raw_delta, calibrated_delta = [], []
        for heads in heldout:
            per_head = [h["answer_mean"] - h["background_mean"] for h in heads]
            raw_delta.append(statistics.mean(per_head))
            calibrated_delta.append(sum(w * d for w, d in zip(normalized, per_head)))
        summary[str(layer_id)] = {
            "calibration_contexts": len(calibration),
            "heldout_contexts": len(heldout),
            "raw_heldout_mean_A_minus_B": statistics.mean(raw_delta),
            "calibrated_heldout_mean_A_minus_B": statistics.mean(calibrated_delta),
            "raw_heldout_A_gt_B_fraction": statistics.mean(float(x > 0) for x in raw_delta),
            "calibrated_heldout_A_gt_B_fraction": statistics.mean(float(x > 0) for x in calibrated_delta),
            "nonzero_heads": sum(x > 0 for x in normalized),
            "heads": head_stats,
        }
    payload = {
        "source": Path(args.rows).name,
        "split": "replicate 0-14 calibration; 15-24 heldout; selector-eligible spans only",
        "rule": "positive standardized ((answer+evidence)/2 - background) head weight, normalized per layer",
        "weights": weights,
        "summary": summary,
    }
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        layer: {
            "raw_A_gt_B": round(values["raw_heldout_A_gt_B_fraction"], 4),
            "calibrated_A_gt_B": round(values["calibrated_heldout_A_gt_B_fraction"], 4),
            "raw_delta": round(values["raw_heldout_mean_A_minus_B"], 4),
            "calibrated_delta": round(values["calibrated_heldout_mean_A_minus_B"], 4),
        }
        for layer, values in summary.items()
    }, indent=2))


if __name__ == "__main__":
    main()
