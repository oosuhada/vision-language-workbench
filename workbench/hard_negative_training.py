"""Turn mined negatives into LAVIS-ready annotations and training losses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row must be an object: {path}")
            rows.append(value)
    return rows


def materialize_hard_negative_annotations(
    source_jsonl: str | Path,
    mined_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    negatives_per_anchor: int = 1,
) -> dict[str, Any]:
    """Join source examples with mined ids into LAVIS retrieval annotations.

    Source rows need ``id``, ``image`` and ``text``. The output keeps the
    positive image/caption pair and attaches one or more mined negative captions
    plus their cosine hardness, so a LAVIS dataset can consume them directly.
    """
    if negatives_per_anchor < 1:
        raise ValueError("negatives_per_anchor must be >= 1.")
    source = _load_jsonl(source_jsonl)
    mined = _load_jsonl(mined_jsonl)
    by_id = {str(row["id"]): row for row in source}
    output_rows: list[dict[str, Any]] = []
    missing: set[str] = set()

    for item in mined:
        anchor_id = str(item.get("anchor_id", ""))
        anchor = by_id.get(anchor_id)
        if anchor is None:
            missing.add(anchor_id)
            continue
        negatives: list[dict[str, Any]] = []
        for candidate in (item.get("negatives") or [])[:negatives_per_anchor]:
            candidate_id = str(candidate.get("id", ""))
            negative = by_id.get(candidate_id)
            if negative is None:
                missing.add(candidate_id)
                continue
            negatives.append(
                {
                    "id": candidate_id,
                    "caption": str(negative["text"]),
                    "label": negative.get("label"),
                    "cosine_similarity": float(candidate.get("cosine_similarity", 0.0)),
                }
            )
        if not negatives:
            continue
        output_rows.append(
            {
                "image": str(anchor["image"]),
                "caption": str(anchor["text"]),
                "image_id": anchor_id,
                "label": anchor.get("label"),
                "hard_negatives": negatives,
            }
        )

    destination = Path(output_jsonl)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "source_samples": len(source),
        "mined_anchors": len(mined),
        "written_samples": len(output_rows),
        "negatives_per_anchor": negatives_per_anchor,
        "missing_ids": sorted(missing),
        "output": str(destination),
    }


def hard_negative_margin_loss(
    image_features: torch.Tensor,
    positive_text_features: torch.Tensor,
    negative_text_features: torch.Tensor,
    *,
    margin: float = 0.2,
    hardness: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pairwise cosine margin loss for externally mined hard negatives.

    Supports negative shape ``[B, D]`` or ``[B, K, D]``. Optional hardness
    weights can be ``[B]`` or ``[B, K]`` and are normalized by their mean so
    loss scale remains comparable across mining policies.
    """
    if image_features.ndim != 2 or positive_text_features.ndim != 2:
        raise ValueError("image_features and positive_text_features must be [B, D].")
    if image_features.shape != positive_text_features.shape:
        raise ValueError("Image and positive text features must share shape.")
    if negative_text_features.ndim not in {2, 3}:
        raise ValueError("negative_text_features must be [B, D] or [B, K, D].")
    image = F.normalize(image_features, dim=-1)
    positive = F.normalize(positive_text_features, dim=-1)
    negative = F.normalize(negative_text_features, dim=-1)
    positive_similarity = (image * positive).sum(dim=-1)
    if negative.ndim == 2:
        if negative.shape != image.shape:
            raise ValueError("Single negative features must match image feature shape.")
        negative_similarity = (image * negative).sum(dim=-1)
    else:
        if negative.shape[0] != image.shape[0] or negative.shape[2] != image.shape[1]:
            raise ValueError("Batched negatives must have shape [B, K, D].")
        negative_similarity = torch.einsum("bd,bkd->bk", image, negative)
        positive_similarity = positive_similarity[:, None]
    losses = F.relu(float(margin) - positive_similarity + negative_similarity)
    if hardness is not None:
        weights = hardness.to(device=losses.device, dtype=losses.dtype)
        if weights.shape != losses.shape:
            raise ValueError("hardness must match the per-negative loss shape.")
        weights = weights / weights.mean().clamp_min(1e-8)
        losses = losses * weights
    return losses.mean()
