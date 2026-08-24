"""Representation probing utilities for saved multimodal embeddings."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RepresentationSnapshot:
    embeddings: np.ndarray
    ids: np.ndarray
    labels: np.ndarray | None
    metadata: dict[str, Any]


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2-D embedding matrix, got shape={values.shape}.")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def save_snapshot(
    path: str | Path,
    embeddings: np.ndarray,
    ids: list[str] | np.ndarray,
    labels: list[str] | np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(embeddings, dtype=np.float32)
    id_array = np.asarray(ids, dtype=str)
    if matrix.ndim != 2:
        raise ValueError("embeddings must be a 2-D matrix.")
    if len(id_array) != matrix.shape[0]:
        raise ValueError("ids length must match embedding rows.")
    payload: dict[str, Any] = {
        "embeddings": matrix,
        "ids": id_array,
        "metadata": np.asarray(json.dumps(metadata or {}, sort_keys=True)),
    }
    if labels is not None:
        label_array = np.asarray(labels, dtype=str)
        if len(label_array) != matrix.shape[0]:
            raise ValueError("labels length must match embedding rows.")
        payload["labels"] = label_array
    np.savez_compressed(output, **payload)


def load_snapshot(path: str | Path) -> RepresentationSnapshot:
    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        embeddings = np.asarray(data["embeddings"], dtype=np.float64)
        ids = np.asarray(data["ids"], dtype=str)
        labels = np.asarray(data["labels"], dtype=str) if "labels" in data else None
        raw_metadata = str(data["metadata"].item()) if "metadata" in data else "{}"
    metadata = json.loads(raw_metadata)
    if embeddings.ndim != 2 or len(ids) != embeddings.shape[0]:
        raise ValueError(f"Invalid representation snapshot: {source}")
    if labels is not None and len(labels) != embeddings.shape[0]:
        raise ValueError(f"Invalid labels in representation snapshot: {source}")
    return RepresentationSnapshot(embeddings, ids, labels, metadata if isinstance(metadata, dict) else {"value": metadata})


def embedding_anisotropy(embeddings: np.ndarray) -> float:
    unit = _normalize_rows(embeddings)
    n = unit.shape[0]
    if n < 2:
        return 0.0
    summed = unit.sum(axis=0)
    return (float(summed @ summed) - float(n)) / float(n * (n - 1))


def class_separation(embeddings: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    unit = _normalize_rows(embeddings)
    labels = np.asarray(labels, dtype=str)
    classes = np.unique(labels)
    if len(classes) < 2:
        raise ValueError("Class separation requires at least two labels.")
    within_sum = 0.0
    within_pairs = 0
    centroids: list[np.ndarray] = []
    for label in classes:
        group = unit[labels == label]
        centroid = group.mean(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
        centroids.append(centroid)
        count = group.shape[0]
        if count >= 2:
            group_sum = group.sum(axis=0)
            within_sum += (float(group_sum @ group_sum) - float(count)) / 2.0
            within_pairs += count * (count - 1) // 2
    within = within_sum / within_pairs if within_pairs else 0.0
    centroid_matrix = np.stack(centroids)
    centroid_sim = centroid_matrix @ centroid_matrix.T
    between = float(centroid_sim[~np.eye(len(classes), dtype=bool)].mean())
    return {
        "within_class_cosine": float(within),
        "between_class_centroid_cosine": between,
        "separation_margin": float(within - between),
    }


def _stratified_split(labels: np.ndarray, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1.")
    rng = np.random.default_rng(seed)
    train: list[int] = []
    test: list[int] = []
    for label in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == label))
        if len(indices) < 2:
            train.extend(indices.tolist())
            continue
        test_count = min(max(1, int(round(len(indices) * test_fraction))), len(indices) - 1)
        test.extend(indices[:test_count].tolist())
        train.extend(indices[test_count:].tolist())
    if not train or not test:
        raise ValueError("Need at least two samples per class for a probe split.")
    return np.asarray(train, dtype=int), np.asarray(test, dtype=int)


def ridge_linear_probe(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    test_fraction: float = 0.2,
    ridge: float = 1.0,
    seed: int = 42,
) -> dict[str, Any]:
    x = _normalize_rows(embeddings)
    y = np.asarray(labels, dtype=str)
    classes, encoded = np.unique(y, return_inverse=True)
    if len(classes) < 2:
        raise ValueError("Linear probing requires at least two classes.")
    train_idx, test_idx = _stratified_split(y, test_fraction, seed)
    x_train = np.concatenate([x[train_idx], np.ones((len(train_idx), 1))], axis=1)
    x_test = np.concatenate([x[test_idx], np.ones((len(test_idx), 1))], axis=1)
    target = np.eye(len(classes), dtype=np.float64)[encoded[train_idx]]
    gram = x_train.T @ x_train
    regularizer = np.eye(gram.shape[0], dtype=np.float64) * float(ridge)
    regularizer[-1, -1] = 0.0
    weights = np.linalg.pinv(gram + regularizer) @ x_train.T @ target
    return {
        "classes": classes.tolist(),
        "train_samples": int(len(train_idx)),
        "test_samples": int(len(test_idx)),
        "ridge": float(ridge),
        "train_accuracy": float(np.mean((x_train @ weights).argmax(axis=1) == encoded[train_idx])),
        "test_accuracy": float(np.mean((x_test @ weights).argmax(axis=1) == encoded[test_idx])),
    }


def probe_snapshot(snapshot: RepresentationSnapshot, *, test_fraction: float = 0.2, ridge: float = 1.0, seed: int = 42) -> dict[str, Any]:
    result: dict[str, Any] = {
        "samples": int(snapshot.embeddings.shape[0]),
        "dimensions": int(snapshot.embeddings.shape[1]),
        "anisotropy": embedding_anisotropy(snapshot.embeddings),
        "metadata": snapshot.metadata,
    }
    if snapshot.labels is not None:
        result["class_count"] = int(len(np.unique(snapshot.labels)))
        result["class_separation"] = class_separation(snapshot.embeddings, snapshot.labels)
        result["linear_probe"] = ridge_linear_probe(snapshot.embeddings, snapshot.labels, test_fraction=test_fraction, ridge=ridge, seed=seed)
    return result
