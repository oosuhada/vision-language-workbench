"""Embedding-space hard-negative mining for contrastive/fine-tuning datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .representations import RepresentationSnapshot


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def _valid_mask(
    anchor: RepresentationSnapshot,
    candidates: RepresentationSnapshot,
    anchor_index: int,
    *,
    policy: str,
) -> np.ndarray:
    mask = np.ones(len(candidates.ids), dtype=bool)
    anchor_id = str(anchor.ids[anchor_index])
    mask &= candidates.ids.astype(str) != anchor_id

    if policy == "different-id":
        return mask
    if anchor.labels is None or candidates.labels is None:
        raise ValueError(f"negative policy '{policy}' requires labels in both snapshots.")
    anchor_label = str(anchor.labels[anchor_index])
    candidate_labels = candidates.labels.astype(str)
    if policy == "different-label":
        mask &= candidate_labels != anchor_label
    elif policy == "same-label":
        mask &= candidate_labels == anchor_label
    else:
        raise ValueError("policy must be different-id, different-label, or same-label.")
    return mask


def mine_hard_negatives(
    anchor: RepresentationSnapshot,
    candidates: RepresentationSnapshot | None = None,
    *,
    top_k: int = 5,
    policy: str = "different-label",
    chunk_size: int = 512,
) -> dict[str, Any]:
    """Mine nearest invalid matches by cosine similarity.

    ``candidates`` may be a different modality snapshot captured on the same
    semantic probe ids, enabling image->text or text->image negative mining.
    Similarity is computed in chunks so only ``chunk_size x candidate_count``
    scores are resident at once.
    """
    candidates = candidates or anchor
    if anchor.embeddings.shape[1] != candidates.embeddings.shape[1]:
        raise ValueError("Anchor and candidate embeddings must share dimensions for cosine mining.")
    if top_k < 1 or chunk_size < 1:
        raise ValueError("top_k and chunk_size must be >= 1.")
    a = _normalize(anchor.embeddings)
    c = _normalize(candidates.embeddings)
    mined: list[dict[str, Any]] = []
    top1_scores: list[float] = []

    for start in range(0, len(a), chunk_size):
        stop = min(start + chunk_size, len(a))
        scores = a[start:stop] @ c.T
        for local_index, anchor_index in enumerate(range(start, stop)):
            valid = _valid_mask(anchor, candidates, anchor_index, policy=policy)
            valid_indices = np.flatnonzero(valid)
            if len(valid_indices) == 0:
                mined.append({
                    "anchor_id": str(anchor.ids[anchor_index]),
                    "anchor_label": str(anchor.labels[anchor_index]) if anchor.labels is not None else None,
                    "negatives": [],
                })
                continue
            valid_scores = scores[local_index, valid_indices]
            count = min(top_k, len(valid_indices))
            if count == len(valid_indices):
                selected_local = np.argsort(-valid_scores)
            else:
                partial = np.argpartition(-valid_scores, count - 1)[:count]
                selected_local = partial[np.argsort(-valid_scores[partial])]
            selected = valid_indices[selected_local]
            negatives = []
            for candidate_index in selected:
                score = float(scores[local_index, candidate_index])
                negatives.append({
                    "id": str(candidates.ids[candidate_index]),
                    "label": str(candidates.labels[candidate_index]) if candidates.labels is not None else None,
                    "cosine_similarity": score,
                })
            if negatives:
                top1_scores.append(float(negatives[0]["cosine_similarity"]))
            mined.append({
                "anchor_id": str(anchor.ids[anchor_index]),
                "anchor_label": str(anchor.labels[anchor_index]) if anchor.labels is not None else None,
                "negatives": negatives,
            })

    return {
        "policy": policy,
        "anchor_samples": int(len(anchor.ids)),
        "candidate_samples": int(len(candidates.ids)),
        "top_k": int(top_k),
        "mean_top1_similarity": float(np.mean(top1_scores)) if top1_scores else None,
        "p95_top1_similarity": float(np.quantile(top1_scores, 0.95)) if top1_scores else None,
        "items": mined,
    }


def write_hard_negative_jsonl(result: dict[str, Any], output: str | Path) -> Path:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for item in result["items"]:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return destination
