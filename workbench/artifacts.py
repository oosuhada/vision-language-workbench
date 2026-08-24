"""Discover LAVIS output artifacts and build resume plans without replacing LAVIS runners."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .manifest import ExperimentManifest
from .planner import ExperimentPlan, build_plan


@dataclass(frozen=True)
class Artifact:
    path: str
    kind: str
    size_bytes: int
    modified_ns: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_output_dir(manifest: ExperimentManifest, repo_root: str | Path = ".") -> Path | None:
    root = Path(repo_root).resolve()
    override = manifest.options.get("run.output_dir")
    if override:
        output = Path(str(override))
        return output if output.is_absolute() else (root / output).resolve()

    cfg_path = (root / manifest.cfg_path).resolve()
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        return None
    run = raw.get("run") or {}
    if not isinstance(run, dict) or not run.get("output_dir"):
        return None
    output = Path(str(run["output_dir"]))
    return output if output.is_absolute() else (root / output).resolve()


def _kind(path: Path) -> str:
    name = path.name.lower()
    if "checkpoint" in name or path.suffix.lower() in {".pth", ".pt", ".ckpt"}:
        return "checkpoint"
    if path.suffix.lower() in {".json", ".jsonl", ".csv"}:
        return "result"
    if path.suffix.lower() in {".log", ".txt"}:
        return "log"
    return "artifact"


def discover_artifacts(manifest: ExperimentManifest, repo_root: str | Path = ".") -> list[Artifact]:
    output_dir = resolve_output_dir(manifest, repo_root)
    if output_dir is None or not output_dir.exists():
        return []

    artifacts: list[Artifact] = []
    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue
        stat = path.stat()
        artifacts.append(
            Artifact(
                path=str(path),
                kind=_kind(path),
                size_bytes=stat.st_size,
                modified_ns=stat.st_mtime_ns,
            )
        )
    return sorted(artifacts, key=lambda item: (item.modified_ns, item.path), reverse=True)


def find_latest_checkpoint(output_dir: str | Path) -> Path | None:
    root = Path(output_dir)
    if not root.exists():
        return None
    candidates = [
        path
        for path in root.rglob("*")
        if path.is_file() and ("checkpoint" in path.name.lower() or path.suffix.lower() in {".pth", ".ckpt"})
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)))


def build_resume_plan(
    manifest: ExperimentManifest,
    repo_root: str | Path = ".",
    checkpoint: str | Path | None = None,
) -> ExperimentPlan:
    if manifest.mode != "train":
        raise ValueError("Resume planning is only available for train manifests.")

    root = Path(repo_root).resolve()
    output_dir = resolve_output_dir(manifest, root)
    selected = Path(checkpoint).resolve() if checkpoint else (find_latest_checkpoint(output_dir) if output_dir else None)
    if selected is None or not selected.exists():
        raise FileNotFoundError("No checkpoint was found. Pass --checkpoint or run the experiment first.")

    options = dict(manifest.options)
    options["run.resume_ckpt_path"] = str(selected)
    resumed = ExperimentManifest(
        name=f"{manifest.name}-resume",
        mode=manifest.mode,
        cfg_path=manifest.cfg_path,
        options=options,
        tags=manifest.tags + ("resume",),
        notes=manifest.notes,
    )
    return build_plan(resumed, root)
