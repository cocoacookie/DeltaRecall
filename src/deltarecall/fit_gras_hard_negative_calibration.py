#!/usr/bin/env python3
"""Add a fixed distractor penalty to the semantic head calibration weights.

Layers and per-layer head count are derived from the base calibration file and
the insight rows, so this works for any model geometry. Defaults reproduce the
original 9B artifact byte-for-byte.
"""

import argparse
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).parent


def normalize_with_floor(values, uniform_floor, min_effective_heads):
    count = len(values)
    if count < min_effective_heads:
        raise ValueError(
            f"head_count={count} is below min_effective_heads={min_effective_heads}"
        )
    total = sum(values)
    normalized = (
        [value / total for value in values]
        if total > 0 and all(math.isfinite(value) for value in values)
        else [1.0 / count] * count
    )

    def blend(alpha):
        return [
            (1.0 - alpha) * value + alpha / count
            for value in normalized
        ]

    def effective(weights):
        return 1.0 / sum(value * value for value in weights)

    alpha = min(max(uniform_floor, 0.0), 1.0)
    weights = blend(alpha)
    if effective(weights) < min_effective_heads:
        low, high = alpha, 1.0
        for _ in range(48):
            middle = (low + high) / 2.0
            if effective(blend(middle)) >= min_effective_heads:
                high = middle
            else:
                low = middle
        alpha = high
        weights = blend(alpha)
    return weights, alpha, effective(weights)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda", dest="penalty", type=float, default=.75)
    parser.add_argument("--calibration", default=str(ROOT / "gras_head_calibration.json"))
    parser.add_argument("--rows", default=str(ROOT / "gras_insight_500.json"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--uniform-floor", type=float, default=0.05)
    parser.add_argument("--min-effective-heads", type=float, default=8.0)
    args = parser.parse_args()
    calibration = json.loads(Path(args.calibration).read_text())
    rows = json.loads(Path(args.rows).read_text())["rows"]
    LAYERS = [int(x) for x in calibration["weights"].keys()]
    weights = {}
    diagnostics = {}
    for layer in LAYERS:
        base = calibration["weights"][str(layer)]
        head_count = len(base)
        d_rates = []
        for head in range(head_count):
            d_rates.append(statistics.mean(
                next(x for x in row["layers"] if x["layer"] == layer)["heads"][head]["top_label_counts"].get("D", 0) / 20.0
                for row in rows if row["replicate"] < 15
            ))
        max_rate = max(d_rates) or 1.0
        adjusted = [
            weight * max(0.0, 1.0 - args.penalty * rate / max_rate)
            for weight, rate in zip(base, d_rates)
        ]
        final, floor_mix, effective_heads = normalize_with_floor(
            adjusted,
            args.uniform_floor,
            args.min_effective_heads,
        )
        weights[str(layer)] = final
        diagnostics[str(layer)] = {
            "distractor_rate": d_rates,
            "base": base,
            "pre_floor": adjusted,
            "uniform_mix": floor_mix,
            "effective_heads": effective_heads,
            "nonzero_heads": sum(value > 0 for value in final),
            "adjusted": final,
        }
    output = Path(args.output) if args.output else ROOT / f"gras_head_calibration_hardneg_lambda{args.penalty:g}.json"
    output.write_text(json.dumps({
        "source": str(Path(args.calibration)),
        "rule": (
            f"semantic calibration weight times max(0, 1 - {args.penalty:g} * "
            "per-head distractor top-20 rate / layer maximum), then mixed "
            "toward uniform until the effective-head constraint is met"
        ),
        "penalty": args.penalty,
        "uniform_floor": args.uniform_floor,
        "min_effective_heads": args.min_effective_heads,
        "weights": weights,
        "diagnostics": diagnostics,
    }, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "output": str(output),
        "penalty": args.penalty,
        "effective_heads": {
            layer: round(item["effective_heads"], 4)
            for layer, item in diagnostics.items()
        },
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
