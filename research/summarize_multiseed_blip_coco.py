#!/usr/bin/env python3
"""Aggregate multi-seed BLIP canonical results with paired deltas and confidence intervals."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any


METRICS = (
    "retrieval_mean_r_at_1",
    "retrieval_mean_r_at_5",
    "retrieval_mean_r_at_10",
    "linear_probe_accuracy",
    "cka_vs_base",
    "mean_cosine_drift",
    "ood_mean_r_at_1",
    "ood_mean_retention_r_at_1",
    "ood_worst_retention_r_at_1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    # Two-sided Student-t critical values avoid the overconfident normal
    # approximation for the intentionally small canonical seed counts.
    t95 = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        21: 2.080,
        22: 2.074,
        23: 2.069,
        24: 2.064,
        25: 2.060,
        26: 2.056,
        27: 2.052,
        28: 2.048,
        29: 2.045,
        30: 2.042,
    }
    critical = t95.get(len(values) - 1, 1.96)
    return critical * stdev(values) / math.sqrt(len(values))


def main() -> None:
    args = parse_args()
    runs: list[tuple[int, list[dict[str, Any]]]] = []
    seed_dirs = sorted(args.root.glob("seed-*"), key=lambda path: int(path.name.split("-", 1)[1]))
    for seed_dir in seed_dirs:
        comparison = seed_dir / "run" / "comparison.json"
        if not comparison.exists():
            continue
        seed = int(seed_dir.name.split("-", 1)[1])
        runs.append((seed, json.loads(comparison.read_text(encoding="utf-8"))))
    if not runs:
        raise FileNotFoundError(f"No seed-*/run/comparison.json files under {args.root}")

    by_variant: dict[str, dict[str, list[float]]] = {}
    paired: dict[str, dict[str, list[float]]] = {}
    for seed, rows in runs:
        row_map = {row["variant"]: row for row in rows}
        base = row_map["base"]
        for variant, row in row_map.items():
            metric_map = by_variant.setdefault(variant, {})
            delta_map = paired.setdefault(variant, {})
            for metric in METRICS:
                if metric not in row:
                    continue
                metric_map.setdefault(metric, []).append(float(row[metric]))
                if variant != "base" and metric in base:
                    delta_map.setdefault(metric, []).append(float(row[metric]) - float(base[metric]))

    summary: dict[str, Any] = {"seed_count": len(runs), "seeds": [seed for seed, _ in runs], "variants": {}}
    for variant, metrics in by_variant.items():
        entry: dict[str, Any] = {"metrics": {}, "paired_delta_vs_base": {}}
        for metric, values in metrics.items():
            entry["metrics"][metric] = {
                "mean": mean(values),
                "std": stdev(values) if len(values) > 1 else 0.0,
                "ci95_half_width": ci95(values),
                "values": values,
            }
        for metric, values in paired.get(variant, {}).items():
            entry["paired_delta_vs_base"][metric] = {
                "mean": mean(values),
                "std": stdev(values) if len(values) > 1 else 0.0,
                "ci95_half_width": ci95(values),
                "values": values,
            }
        summary["variants"][variant] = entry

    output = args.output or args.root / "multiseed-summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "metric", "mean", "std", "ci95_half_width", "paired_delta_vs_base_mean"])
        for variant, entry in summary["variants"].items():
            for metric, stats in entry["metrics"].items():
                delta = entry["paired_delta_vs_base"].get(metric, {}).get("mean")
                writer.writerow([variant, metric, stats["mean"], stats["std"], stats["ci95_half_width"], delta])

    print(json.dumps({"seeds": summary["seeds"], "output": str(output), "csv": str(csv_path)}, indent=2))


if __name__ == "__main__":
    main()
