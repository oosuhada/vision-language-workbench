#!/usr/bin/env python3
"""Run the canonical BLIP study across deterministic seeds on one CUDA runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("research/canonical/blip_coco_harder_multiseed.json"))
    parser.add_argument("--seeds", default="42,1337,2026")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/canonical-blip-coco-harder-multiseed-v2"))
    parser.add_argument("--cache", type=Path, default=Path("/content/blip-coco-cache"))
    parser.add_argument("--variants", default="base,lora-r8,lora-r16,lora-r8-hard-negative")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    template = json.loads(args.config.read_text(encoding="utf-8"))
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique.")

    args.output_root.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        config = dict(template)
        config["seed"] = seed
        config["study"] = f"{template['study']}-seed-{seed}"
        seed_root = args.output_root / f"seed-{seed}"
        config_path = seed_root / "resolved-config.json"
        output_path = seed_root / "run"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        command = [
            sys.executable,
            "research/canonical_blip_coco.py",
            "--config",
            str(config_path),
            "--output",
            str(output_path),
            "--cache",
            str(args.cache),
            "--variants",
            args.variants,
        ]
        print("RUN", seed, " ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
