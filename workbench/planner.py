"""Turn a workbench manifest into the exact original LAVIS command."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shlex
import sys

from .manifest import ExperimentManifest


@dataclass(frozen=True)
class ExperimentPlan:
    name: str
    mode: str
    command: tuple[str, ...]
    manifest_hash: str
    config_hash: str

    @property
    def shell_command(self) -> str:
        return shlex.join(self.command)

    def as_dict(self) -> dict[str, str | list[str]]:
        return {
            "name": self.name,
            "mode": self.mode,
            "command": list(self.command),
            "shell_command": self.shell_command,
            "manifest_hash": self.manifest_hash,
            "config_hash": self.config_hash,
        }


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_hash(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return _sha256_bytes(canonical.encode("utf-8"))


def build_plan(manifest: ExperimentManifest, repo_root: str | Path = ".") -> ExperimentPlan:
    root = Path(repo_root).resolve()
    cfg_path = (root / manifest.cfg_path).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"LAVIS config does not exist: {cfg_path}")

    entrypoint = root / ("train.py" if manifest.mode == "train" else "evaluate.py")
    if not entrypoint.exists():
        raise FileNotFoundError(f"LAVIS entrypoint does not exist: {entrypoint}")

    command = [sys.executable, str(entrypoint), "--cfg-path", str(cfg_path)]
    if manifest.options:
        command.append("--options")
        for key in sorted(manifest.options):
            value = manifest.options[key]
            if isinstance(value, bool):
                rendered = "true" if value else "false"
            elif value is None:
                rendered = "null"
            else:
                rendered = str(value)
            command.append(f"{key}={rendered}")

    return ExperimentPlan(
        name=manifest.name,
        mode=manifest.mode,
        command=tuple(command),
        manifest_hash=_json_hash(manifest.canonical_dict()),
        config_hash=_sha256_bytes(cfg_path.read_bytes()),
    )
