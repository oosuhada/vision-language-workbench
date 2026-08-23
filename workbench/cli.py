"""Command-line interface for planning LAVIS experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .catalog import search_catalog
from .ledger import RunLedger, execute_plan
from .manifest import ExperimentManifest
from .planner import build_plan


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vl-workbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Render a manifest into an exact LAVIS command.")
    plan.add_argument("manifest")
    plan.add_argument("--repo-root", default=".")
    plan.add_argument("--json", action="store_true", dest="as_json")

    run = subparsers.add_parser("run", help="Execute a manifest through the original LAVIS entrypoint and record it.")
    run.add_argument("manifest")
    run.add_argument("--repo-root", default=".")
    run.add_argument("--ledger", default="artifacts/run-ledger.jsonl")

    history = subparsers.add_parser("history", help="Show recent workbench run records.")
    history.add_argument("--ledger", default="artifacts/run-ledger.jsonl")
    history.add_argument("--limit", type=int, default=20)

    catalog = subparsers.add_parser("catalog", help="Search bundled LAVIS model/project configs without importing models.")
    catalog.add_argument("query", nargs="?", default="")
    catalog.add_argument("--kind", choices=["all", "model", "project"], default="all")
    catalog.add_argument("--repo-root", default=".")
    catalog.add_argument("--limit", type=int, default=30)

    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "plan":
        manifest = ExperimentManifest.load(Path(args.manifest))
        plan = build_plan(manifest, args.repo_root)
        if args.as_json:
            print(json.dumps(plan.as_dict(), indent=2, ensure_ascii=False))
        else:
            print(plan.shell_command)
    elif args.command == "run":
        manifest = ExperimentManifest.load(Path(args.manifest))
        plan = build_plan(manifest, args.repo_root)
        record = execute_plan(manifest, plan, RunLedger(args.ledger), cwd=args.repo_root)
        print(json.dumps(record.as_dict(), indent=2, ensure_ascii=False))
    elif args.command == "history":
        records = RunLedger(args.ledger).recent(args.limit)
        print(json.dumps(records, indent=2, ensure_ascii=False))
    elif args.command == "catalog":
        entries = search_catalog(args.query, args.kind, args.repo_root, args.limit)
        print(json.dumps([entry.as_dict() for entry in entries], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
