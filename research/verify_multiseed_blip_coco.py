#!/usr/bin/env python3
"""Verify that every requested BLIP multi-seed run produced complete small artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_VARIANTS = ("base", "lora-r8", "lora-r16", "lora-r8-hard-negative")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--seeds", default="42,1337,2026")
    parser.add_argument("--variants", default=",".join(EXPECTED_VARIANTS))
    return parser.parse_args()


def verify_seed(seed_root: Path, variants: list[str]) -> dict[str, Any]:
    run = seed_root / "run"
    required = [
        run / "run_manifest.json",
        run / "comparison.json",
        run / "comparison.csv",
        run / "comparison.md",
    ]
    for variant in variants:
        required.append(run / variant / "metrics.json")
        required.append(run / variant / "fused_representations.npz")
    missing = [str(path.relative_to(seed_root)) for path in required if not path.exists()]

    variants_seen: list[str] = []
    comparison_path = run / "comparison.json"
    if comparison_path.exists():
        rows = json.loads(comparison_path.read_text(encoding="utf-8"))
        variants_seen = [str(row.get("variant")) for row in rows]
        for variant in variants:
            if variant not in variants_seen:
                missing.append(f"comparison.json:variant:{variant}")

    return {
        "seed": int(seed_root.name.split("-", 1)[1]),
        "complete": not missing,
        "missing": missing,
        "variants_seen": variants_seen,
    }


def main() -> None:
    args = parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    reports = [verify_seed(args.root / f"seed-{seed}", variants) for seed in seeds]
    result = {
        "root": str(args.root),
        "requested_seeds": seeds,
        "requested_variants": variants,
        "complete": all(report["complete"] for report in reports),
        "complete_seed_count": sum(report["complete"] for report in reports),
        "seeds": reports,
    }
    print(json.dumps(result, indent=2))
    if not result["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
