#!/usr/bin/env python3
"""Build a fixed GRAS per-layer budget from calibration result files."""

import argparse
import json
from pathlib import Path

from .gras_hmc_selector import allocate_layer_budgets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--total-budget", type=int, required=True)
    parser.add_argument("--min-budget", type=int, default=128)
    parser.add_argument("--max-budget", type=int, default=1024)
    parser.add_argument("--align", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    layer_values = {}
    for input_path in args.inputs:
        result = json.loads(Path(input_path).read_text())
        for layer_id, layer in result["layers"].items():
            if "demand_mean" not in layer or "demand_p90" not in layer:
                continue
            value = 0.5 * float(layer["demand_mean"]) + 0.5 * float(layer["demand_p90"])
            layer_values.setdefault(int(layer_id), []).append(value)

    if not layer_values:
        raise ValueError("calibration results contain no demand summaries")
    layer_ids = sorted(layer_values)
    demand = [sum(layer_values[index]) / len(layer_values[index]) for index in layer_ids]
    budgets = allocate_layer_budgets(
        demand,
        total_budget=args.total_budget,
        min_budget=args.min_budget,
        max_budget=args.max_budget,
        align=args.align,
    )
    output = {
        "budgets": budgets,
        "demand": demand,
        "layer_ids": layer_ids,
        "total_budget": args.total_budget,
        "min_budget": args.min_budget,
        "max_budget": args.max_budget,
        "align": args.align,
        "inputs": args.inputs,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"output": str(output_path), "budgets": budgets}, ensure_ascii=False))


if __name__ == "__main__":
    main()
