"""Fast filesystem catalog for bundled LAVIS model and project configs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class CatalogEntry:
    kind: str
    name: str
    path: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def build_catalog(repo_root: str | Path = ".") -> list[CatalogEntry]:
    root = Path(repo_root).resolve()
    entries: list[CatalogEntry] = []

    model_root = root / "lavis" / "configs" / "models"
    if model_root.exists():
        for path in sorted(model_root.rglob("*.yaml")):
            entries.append(
                CatalogEntry(
                    kind="model",
                    name=path.stem,
                    path=str(path.relative_to(root)),
                )
            )

    project_root = root / "lavis" / "projects"
    if project_root.exists():
        for path in sorted(project_root.rglob("*.yaml")):
            entries.append(
                CatalogEntry(
                    kind="project",
                    name=path.stem,
                    path=str(path.relative_to(root)),
                )
            )

    return entries


def search_catalog(
    query: str = "",
    kind: str = "all",
    repo_root: str | Path = ".",
    limit: int = 50,
) -> list[CatalogEntry]:
    query_lower = query.strip().lower()
    results: list[CatalogEntry] = []
    for entry in build_catalog(repo_root):
        if kind != "all" and entry.kind != kind:
            continue
        haystack = f"{entry.name} {entry.path}".lower()
        if query_lower and query_lower not in haystack:
            continue
        results.append(entry)
        if len(results) >= limit:
            break
    return results
