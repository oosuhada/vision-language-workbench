"""Command-line interface for planning LAVIS experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .catalog import search_catalog
from .drift import compare_snapshots
from .hard_negatives import mine_hard_negatives, write_hard_negative_jsonl
from .artifacts import build_resume_plan, discover_artifacts, resolve_output_dir
from .ledger import RunLedger, execute_plan
from .lavis_features import extract_lavis_snapshot
from .manifest import ExperimentManifest
from .planner import build_plan
from .representations import load_snapshot, probe_snapshot
from .sweep import SweepSpec, build_sweep_plans, materialize_sweep


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

    sweep = subparsers.add_parser("sweep-plan", help="Expand a matrix sweep into concrete LAVIS commands.")
    sweep.add_argument("spec")
    sweep.add_argument("--repo-root", default=".")

    materialize = subparsers.add_parser("sweep-materialize", help="Write matrix sweep variants as normal experiment manifests.")
    materialize.add_argument("spec")
    materialize.add_argument("--output-dir", default="artifacts/generated-experiments")

    artifacts = subparsers.add_parser("artifacts", help="Discover files produced in a LAVIS experiment output directory.")
    artifacts.add_argument("manifest")
    artifacts.add_argument("--repo-root", default=".")

    resume = subparsers.add_parser("resume-plan", help="Build a train command using the latest or an explicit LAVIS checkpoint.")
    resume.add_argument("manifest")
    resume.add_argument("--repo-root", default=".")
    resume.add_argument("--checkpoint")

    extract = subparsers.add_parser("extract-features", help="Save frozen LAVIS representations from a JSONL probe dataset.")
    extract.add_argument("dataset_jsonl")
    extract.add_argument("--output", required=True)
    extract.add_argument("--model-name", required=True)
    extract.add_argument("--model-type", required=True)
    extract.add_argument("--mode", choices=["image", "text", "multimodal"], required=True)
    extract.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    extract.add_argument("--feature-field")
    extract.add_argument("--checkpoint")
    extract.add_argument("--device", default="auto")
    extract.add_argument("--batch-size", type=int, default=8)

    probe = subparsers.add_parser("probe", help="Measure linear separability and geometry of a representation snapshot.")
    probe.add_argument("snapshot")
    probe.add_argument("--test-fraction", type=float, default=0.2)
    probe.add_argument("--ridge", type=float, default=1.0)
    probe.add_argument("--seed", type=int, default=42)

    drift = subparsers.add_parser("drift", help="Compare representation geometry between two checkpoints/snapshots.")
    drift.add_argument("baseline")
    drift.add_argument("current")
    drift.add_argument("--top-k", type=int, default=10)

    negatives = subparsers.add_parser("hard-negatives", help="Mine nearest incorrect samples from saved embedding spaces.")
    negatives.add_argument("anchor")
    negatives.add_argument("--candidates")
    negatives.add_argument("--policy", choices=["different-id", "different-label", "same-label"], default="different-label")
    negatives.add_argument("--top-k", type=int, default=5)
    negatives.add_argument("--chunk-size", type=int, default=512)
    negatives.add_argument("--output")

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
    elif args.command == "sweep-plan":
        spec = SweepSpec.load(args.spec)
        plans = build_sweep_plans(spec, args.repo_root)
        print(json.dumps([plan.as_dict() for plan in plans], indent=2, ensure_ascii=False))
    elif args.command == "sweep-materialize":
        spec = SweepSpec.load(args.spec)
        paths = materialize_sweep(spec, args.output_dir)
        print(json.dumps([str(path) for path in paths], indent=2, ensure_ascii=False))
    elif args.command == "artifacts":
        manifest = ExperimentManifest.load(args.manifest)
        output_dir = resolve_output_dir(manifest, args.repo_root)
        payload = {
            "output_dir": str(output_dir) if output_dir else None,
            "artifacts": [artifact.as_dict() for artifact in discover_artifacts(manifest, args.repo_root)],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    elif args.command == "resume-plan":
        manifest = ExperimentManifest.load(args.manifest)
        plan = build_resume_plan(manifest, args.repo_root, args.checkpoint)
        print(json.dumps(plan.as_dict(), indent=2, ensure_ascii=False))
    elif args.command == "extract-features":
        result = extract_lavis_snapshot(
            args.dataset_jsonl,
            args.output,
            model_name=args.model_name,
            model_type=args.model_type,
            mode=args.mode,
            pooling=args.pooling,
            feature_field=args.feature_field,
            checkpoint=args.checkpoint,
            device=args.device,
            batch_size=args.batch_size,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.command == "probe":
        result = probe_snapshot(
            load_snapshot(args.snapshot),
            test_fraction=args.test_fraction,
            ridge=args.ridge,
            seed=args.seed,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.command == "drift":
        result = compare_snapshots(load_snapshot(args.baseline), load_snapshot(args.current), top_k=args.top_k)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.command == "hard-negatives":
        anchor = load_snapshot(args.anchor)
        candidates = load_snapshot(args.candidates) if args.candidates else None
        result = mine_hard_negatives(anchor, candidates, top_k=args.top_k, policy=args.policy, chunk_size=args.chunk_size)
        if args.output:
            result["written_to"] = str(write_hard_negative_jsonl(result, args.output))
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
