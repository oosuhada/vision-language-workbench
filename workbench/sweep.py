"""Cartesian experiment sweeps built on top of existing LAVIS manifests."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import yaml

from .manifest import ExperimentManifest
from .planner import ExperimentPlan, build_plan


@dataclass(frozen=True)
class SweepSpec:
    name: str
    mode: str
    cfg_path: str
    options: dict[str, Any]
    matrix: dict[str, tuple[Any, ...]]
    tags: tuple[str, ...] = ()
    notes: str = ""

    @classmethod
    def load(cls, path: str | Path) -> "SweepSpec":
        source = Path(path)
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Sweep spec must contain a mapping/object.")

        name = str(raw.get("name", "")).strip()
        mode = str(raw.get("mode", "")).strip().lower()
        cfg_path = str(raw.get("cfg_path", "")).strip()
        matrix = raw.get("matrix") or {}
        options = raw.get("options") or {}

        if not name:
            raise ValueError("Sweep field 'name' is required.")
        if mode not in {"train", "evaluate"}:
            raise ValueError("Sweep field 'mode' must be 'train' or 'evaluate'.")
        if not cfg_path:
            raise ValueError("Sweep field 'cfg_path' is required.")
        if not isinstance(options, dict):
            raise ValueError("Sweep field 'options' must be a mapping/object.")
        if not isinstance(matrix, dict) or not matrix:
            raise ValueError("Sweep field 'matrix' must be a non-empty mapping/object.")

        normalized: dict[str, tuple[Any, ...]] = {}
        for key, values in matrix.items():
            if not isinstance(values, list) or not values:
                raise ValueError(f"Sweep matrix '{key}' must be a non-empty list.")
            normalized[str(key)] = tuple(values)

        tags = raw.get("tags") or []
        if not isinstance(tags, list):
            raise ValueError("Sweep field 'tags' must be a list.")

        return cls(
            name=name,
            mode=mode,
            cfg_path=cfg_path,
            options=dict(options),
            matrix=normalized,
            tags=tuple(str(tag) for tag in tags),
            notes=str(raw.get("notes", "")),
        )

    def variants(self) -> list[ExperimentManifest]:
        keys = sorted(self.matrix)
        variants: list[ExperimentManifest] = []
        for index, values in enumerate(itertools.product(*(self.matrix[key] for key in keys)), start=1):
            options = dict(self.options)
            options.update(dict(zip(keys, values, strict=True)))
            fingerprint = hashlib.sha256(
                json.dumps(options, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
            ).hexdigest()[:8]
            variants.append(
                ExperimentManifest(
                    name=f"{self.name}-{index:03d}-{fingerprint}",
                    mode=self.mode,
                    cfg_path=self.cfg_path,
                    options=options,
                    tags=self.tags + ("sweep", self.name),
                    notes=self.notes,
                )
            )
        return variants


def build_sweep_plans(spec: SweepSpec, repo_root: str | Path = ".") -> list[ExperimentPlan]:
    return [build_plan(manifest, repo_root) for manifest in spec.variants()]


def materialize_sweep(spec: SweepSpec, output_dir: str | Path) -> list[Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for manifest in spec.variants():
        path = target / f"{manifest.name}.yaml"
        path.write_text(
            yaml.safe_dump(manifest.canonical_dict(), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        written.append(path)
    return written
