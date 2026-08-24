"""Experiment orchestration extensions for the bundled LAVIS codebase."""

from .manifest import ExperimentManifest
from .planner import ExperimentPlan, build_plan
from .catalog import CatalogEntry, search_catalog
from .sweep import SweepSpec, build_sweep_plans
from .artifacts import Artifact, build_resume_plan, discover_artifacts
from .representations import RepresentationSnapshot, load_snapshot, probe_snapshot

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
    "discover_artifacts",
    "load_snapshot",
    "probe_snapshot",
    "search_catalog",
]
