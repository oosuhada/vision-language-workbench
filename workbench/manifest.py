"""Manifest parsing for repeatable LAVIS train/evaluate experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentManifest:
    """A small declarative wrapper around an existing LAVIS config."""

    name: str
    mode: str
    cfg_path: str
    options: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    notes: str = ""

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentManifest":
        manifest_path = Path(path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")

        if manifest_path.suffix.lower() == ".json":
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

        if not isinstance(raw, dict):
            raise ValueError("Experiment manifest must contain a mapping/object.")

        name = str(raw.get("name", "")).strip()
        mode = str(raw.get("mode", "")).strip().lower()
        cfg_path = str(raw.get("cfg_path", "")).strip()

        if not name:
            raise ValueError("Manifest field 'name' is required.")
        if mode not in {"train", "evaluate"}:
            raise ValueError("Manifest field 'mode' must be 'train' or 'evaluate'.")
        if not cfg_path:
            raise ValueError("Manifest field 'cfg_path' is required.")

        options = raw.get("options") or {}
        if not isinstance(options, dict):
            raise ValueError("Manifest field 'options' must be a mapping/object.")

        tags = raw.get("tags") or []
        if not isinstance(tags, list):
            raise ValueError("Manifest field 'tags' must be a list.")

        return cls(
            name=name,
            mode=mode,
            cfg_path=cfg_path,
            options=dict(options),
            tags=tuple(str(tag) for tag in tags),
            notes=str(raw.get("notes", "")),
        )

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode,
            "cfg_path": self.cfg_path,
            "options": self.options,
            "tags": list(self.tags),
            "notes": self.notes,
        }
