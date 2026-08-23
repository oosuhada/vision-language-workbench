"""Append-only local run provenance for workbench experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any

from .manifest import ExperimentManifest
from .planner import ExperimentPlan


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    experiment: str
    mode: str
    status: str
    started_at: str
    finished_at: str
    exit_code: int
    manifest_hash: str
    config_hash: str
    command: list[str]
    tags: list[str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class RunLedger:
    """A JSONL ledger intentionally kept separate from model output files."""

    def __init__(self, path: str | Path = "artifacts/run-ledger.jsonl") -> None:
        self.path = Path(path)

    def append(self, record: RunRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.as_dict(), ensure_ascii=False) + "\n")

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines[-limit:] if line.strip()]


def execute_plan(
    manifest: ExperimentManifest,
    plan: ExperimentPlan,
    ledger: RunLedger,
    cwd: str | Path = ".",
) -> RunRecord:
    started_at = _utc_now()
    run_id = f"{started_at.replace(':', '').replace('+00:00', 'Z')}-{plan.manifest_hash[:8]}"
    result = subprocess.run(list(plan.command), cwd=Path(cwd), check=False)
    finished_at = _utc_now()
    record = RunRecord(
        run_id=run_id,
        experiment=manifest.name,
        mode=manifest.mode,
        status="succeeded" if result.returncode == 0 else "failed",
        started_at=started_at,
        finished_at=finished_at,
        exit_code=result.returncode,
        manifest_hash=plan.manifest_hash,
        config_hash=plan.config_hash,
        command=list(plan.command),
        tags=list(manifest.tags),
    )
    ledger.append(record)
    return record
