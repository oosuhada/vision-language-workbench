"""Experiment orchestration extensions for the bundled LAVIS codebase."""

from .manifest import ExperimentManifest
from .planner import ExperimentPlan, build_plan
from .catalog import CatalogEntry, search_catalog
from .sweep import SweepSpec, build_sweep_plans
from .artifacts import Artifact, build_resume_plan, discover_artifacts
from .representations import RepresentationSnapshot, load_snapshot, probe_snapshot
from .drift import compare_snapshots, linear_cka
from .hard_negatives import mine_hard_negatives
from .lora_research import LoRALinear, LoRAPolicy, compare_lora_policies, discover_lora_targets, inject_lora

__all__ = [
    "CatalogEntry",
    "Artifact",
    "ExperimentManifest",
    "ExperimentPlan",
    "RepresentationSnapshot",
    "SweepSpec",
    "build_plan",
    "build_resume_plan",
    "build_sweep_plans",
    "compare_snapshots",
    "discover_artifacts",
    "load_snapshot",
    "linear_cka",
    "mine_hard_negatives",
    "probe_snapshot",
    "LoRALinear",
    "LoRAPolicy",
    "compare_lora_policies",
    "discover_lora_targets",
    "inject_lora",
    "search_catalog",
]
