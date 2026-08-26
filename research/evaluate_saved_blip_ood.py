#!/usr/bin/env python3
"""Evaluate the exact saved adapters from a canonical BLIP run on OOD inputs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.canonical_blip_coco import (
    LORA_TARGET,
    evaluate_ood_variant,
    evaluate_variant,
    load_json,
    load_model,
    make_zip,
    prepare_coco_subset,
    result_row,
    write_comparison,
    write_json,
)
from workbench.drift import compare_snapshots
from workbench.lora_research import LoRAPolicy, inject_lora
from workbench.representations import load_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("research/canonical/blip_coco_small.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("/content/blip-coco-cache"))
    parser.add_argument(
        "--adapter",
        action="append",
        default=[],
        metavar="VARIANT=PATH",
        help="Exact adapter_model.pt to reload; may be repeated.",
    )
    return parser.parse_args()


def parse_adapters(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Adapter must be VARIANT=PATH, got {value!r}.")
        name, path = value.split("=", 1)
        result[name] = Path(path).resolve()
    return result


def load_exact_adapter(
    model: torch.nn.Module,
    variant: dict[str, Any],
    adapter_path: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    if not adapter_path.is_file():
        raise FileNotFoundError(adapter_path)
    policy = LoRAPolicy(
        name=str(variant["name"]),
        rank=int(variant["rank"]),
        alpha=float(variant["alpha"]),
        dropout=0.0,
        target_regex=(LORA_TARGET,),
    )
    budget = inject_lora(model, policy)
    model.to(device=device, dtype=dtype)
    state = torch.load(adapter_path, map_location="cpu", weights_only=True)
    current_keys = set(model.state_dict())
    unexpected = sorted(set(state) - current_keys)
    if unexpected:
        raise RuntimeError(f"Adapter has unexpected keys: {unexpected[:5]}")
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected adapter keys: {incompatible.unexpected_keys}")
    loaded = sum(value.numel() for value in state.values())
    if loaded != budget["adapter_parameters"]:
        raise RuntimeError(f"Adapter parameter mismatch: file={loaded}, policy={budget['adapter_parameters']}")
    budget.update({"source_adapter": str(adapter_path), "loaded_adapter_parameters": int(loaded)})
    return budget


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    adapters = parse_adapters(args.adapter)
    expected_names = {row["name"] for row in config["variants"]}
    unknown = set(adapters) - expected_names
    if unknown:
        raise ValueError(f"Unknown variants: {sorted(unknown)}")
    if not torch.cuda.is_available():
        raise RuntimeError("Saved-adapter OOD evaluation requires CUDA.")
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = prepare_coco_subset(config, args.cache.resolve(), output)
    probe = [row for row in records if row["split"] == "probe"]
    variant_map = {row["name"]: row for row in config["variants"]}
    requested = ["base", *adapters]
    all_metrics: dict[str, dict[str, Any]] = {}

    for name in requested:
        model, processor = load_model(config, device, dtype)
        training = None
        if name != "base":
            training = load_exact_adapter(model, variant_map[name], adapters[name], device, dtype)
        metrics = evaluate_variant(name, model, processor, probe, output, config, device, dtype, training)
        metrics["ood"] = evaluate_ood_variant(name, model, processor, probe, output, config, device, dtype, metrics)
        write_json(output / name / "metrics.json", metrics)
        all_metrics[name] = metrics
        del model
        torch.cuda.empty_cache()
        print(f"COMPLETE_SAVED_ADAPTER variant={name}", flush=True)

    base_snapshot = load_snapshot(output / "base" / "fused_representations.npz")
    rows: list[dict[str, Any]] = []
    for name in requested:
        drift = None
        if name != "base":
            drift = compare_snapshots(base_snapshot, load_snapshot(output / name / "fused_representations.npz"))
            write_json(output / name / "drift_vs_base.json", drift)
            all_metrics[name]["drift_vs_base"] = drift
            write_json(output / name / "metrics.json", all_metrics[name])
        rows.append(result_row(all_metrics[name], drift))
    write_comparison(output, rows)
    write_json(
        output / "saved_adapter_ood_manifest.json",
        {
            "utc_timestamp": datetime.now(timezone.utc).isoformat(),
            "gpu": torch.cuda.get_device_name(device),
            "dtype": str(dtype),
            "model_id": config["model_id"],
            "model_revision": config["model_revision"],
            "probe_samples": len(probe),
            "condition_count_per_variant": sum(len(v) for v in config["ood_corruptions"].values()),
            "adapters": {name: str(path) for name, path in adapters.items()},
        },
    )
    print("RESULT_ZIP " + str(make_zip(output)), flush=True)
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
