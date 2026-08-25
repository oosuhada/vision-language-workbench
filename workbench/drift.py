"""Checkpoint-to-checkpoint representation drift analysis."""

from __future__ import annotations

from typing import Any

import numpy as np

from .representations import RepresentationSnapshot, class_separation, embedding_anisotropy


def _center(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    return values - values.mean(axis=0, keepdims=True)


def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    """Linear centered-kernel alignment between two representation spaces."""
    x = _center(left)
    y = _center(right)
    if x.shape[0] != y.shape[0]:
        raise ValueError("CKA requires the same number of matched samples.")
    cross = x.T @ y
    numerator = float(np.sum(cross * cross))
    x_norm = float(np.linalg.norm(x.T @ x, ord="fro"))
    y_norm = float(np.linalg.norm(y.T @ y, ord="fro"))
    denominator = x_norm * y_norm
    return numerator / denominator if denominator > 1e-12 else 0.0


def _aligned_rows(
    baseline: RepresentationSnapshot,
    current: RepresentationSnapshot,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    current_index = {sample_id: index for index, sample_id in enumerate(current.ids.tolist())}
    baseline_rows: list[int] = []
    current_rows: list[int] = []
    ids: list[str] = []
    labels: list[str] = []
    for index, sample_id in enumerate(baseline.ids.tolist()):
        other = current_index.get(sample_id)
        if other is None:
            continue
        baseline_rows.append(index)
        current_rows.append(other)
        ids.append(sample_id)
        if baseline.labels is not None and current.labels is not None:
            if baseline.labels[index] != current.labels[other]:
                raise ValueError(f"Label mismatch for sample id={sample_id}")
            labels.append(str(baseline.labels[index]))
    if len(ids) < 2:
        raise ValueError("Need at least two shared sample ids for drift analysis.")
    label_array = np.asarray(labels, dtype=str) if labels else None
    return (
        baseline.embeddings[np.asarray(baseline_rows)],
        current.embeddings[np.asarray(current_rows)],
        np.asarray(ids, dtype=str),
        label_array,
    )


def _row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape != right.shape:
        raise ValueError("Per-sample cosine drift requires equal embedding dimensions.")
    left_norm = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-12)
    right_norm = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
    return np.sum(left_norm * right_norm, axis=1)


def compare_snapshots(
    baseline: RepresentationSnapshot,
    current: RepresentationSnapshot,
    *,
    top_k: int = 10,
) -> dict[str, Any]:
    """Compare two snapshots captured on the same stable probe ids."""
    left, right, ids, labels = _aligned_rows(baseline, current)
    result: dict[str, Any] = {
        "matched_samples": int(len(ids)),
        "baseline_dimensions": int(left.shape[1]),
        "current_dimensions": int(right.shape[1]),
        "linear_cka": linear_cka(left, right),
        "anisotropy": {
            "baseline": embedding_anisotropy(left),
            "current": embedding_anisotropy(right),
        },
        "baseline_metadata": baseline.metadata,
        "current_metadata": current.metadata,
    }
    result["anisotropy"]["delta"] = result["anisotropy"]["current"] - result["anisotropy"]["baseline"]

    if left.shape[1] == right.shape[1]:
        cosine = _row_cosine(left, right)
        drift = 1.0 - cosine
        order = np.argsort(-drift)[: max(0, top_k)]
        result["sample_cosine"] = {
            "mean_similarity": float(cosine.mean()),
            "mean_drift": float(drift.mean()),
            "median_drift": float(np.median(drift)),
            "p95_drift": float(np.quantile(drift, 0.95)),
        }
        result["most_drifted_samples"] = [
            {"id": str(ids[index]), "cosine_similarity": float(cosine[index]), "cosine_drift": float(drift[index])}
            for index in order
        ]

    if labels is not None and len(np.unique(labels)) >= 2:
        before = class_separation(left, labels)
        after = class_separation(right, labels)
        result["class_geometry"] = {
            "baseline": before,
            "current": after,
            "separation_margin_delta": after["separation_margin"] - before["separation_margin"],
        }
        if left.shape[1] == right.shape[1]:
            centroid_cosines: list[float] = []
            per_class: dict[str, float] = {}
            for label in np.unique(labels):
                left_centroid = left[labels == label].mean(axis=0)
                right_centroid = right[labels == label].mean(axis=0)
                similarity = float(
                    left_centroid @ right_centroid
                    / max(float(np.linalg.norm(left_centroid) * np.linalg.norm(right_centroid)), 1e-12)
                )
                centroid_cosines.append(similarity)
                per_class[str(label)] = 1.0 - similarity
            result["class_centroid_drift"] = {
                "mean_cosine_drift": float(np.mean(1.0 - np.asarray(centroid_cosines))),
                "per_class": per_class,
            }
    return result
