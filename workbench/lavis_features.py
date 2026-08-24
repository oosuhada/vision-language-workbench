"""Feature snapshot extraction through bundled LAVIS implementations."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .representations import save_snapshot


@dataclass(frozen=True)
class ProbeItem:
    sample_id: str
    image: str | None = None
    text: str | None = None
    label: str | None = None


def load_probe_items(path: str | Path) -> list[ProbeItem]:
    source = Path(path)
    items: list[ProbeItem] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        sample_id = str(raw.get("id", "")).strip()
        if not sample_id:
            raise ValueError(f"Missing id at {source}:{line_number}")
        items.append(ProbeItem(sample_id, raw.get("image"), raw.get("text"), str(raw["label"]) if raw.get("label") is not None else None))
    if not items:
        raise ValueError(f"Probe dataset is empty: {source}")
    return items


def _auto_device(torch_module: Any) -> str:
    if torch_module.cuda.is_available():
        return "cuda"
    mps = getattr(torch_module.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _feature_tensor(output: Any, mode: str, feature_field: str | None) -> Any:
    candidates = {"image": ("image_features", "image_embeds"), "text": ("text_features", "text_embeds"), "multimodal": ("multimodal_embeds",)}[mode]
    fields = (feature_field,) if feature_field else candidates
    for field in fields:
        value = getattr(output, field, None)
        if value is None and isinstance(output, dict):
            value = output.get(field)
        if value is not None:
            return value
    raise ValueError(f"LAVIS output has none of requested feature fields: {fields}")


def _pool_features(tensor: Any, pooling: str) -> Any:
    if tensor.ndim == 2:
        return tensor
    if tensor.ndim != 3:
        raise ValueError(f"Expected rank-2/3 features, got shape={tuple(tensor.shape)}")
    return tensor[:, 0] if pooling == "cls" else tensor.mean(dim=1)


def extract_lavis_snapshot(
    dataset_jsonl: str | Path,
    output_path: str | Path,
    *,
    model_name: str,
    model_type: str,
    mode: str,
    pooling: str = "cls",
    feature_field: str | None = None,
    checkpoint: str | Path | None = None,
    device: str = "auto",
    batch_size: int = 8,
) -> dict[str, Any]:
    if mode not in {"image", "text", "multimodal"} or pooling not in {"cls", "mean"}:
        raise ValueError("Invalid mode or pooling.")
    import numpy as np
    from PIL import Image
    import torch
    from lavis.models import load_model_and_preprocess

    resolved_device = _auto_device(torch) if device == "auto" else device
    model, vis_processors, txt_processors = load_model_and_preprocess(model_name, model_type, is_eval=True, device=resolved_device)
    checkpoint_path = None
    if checkpoint is not None:
        checkpoint_file = Path(checkpoint).expanduser().resolve()
        payload = torch.load(checkpoint_file, map_location="cpu")
        state_dict = payload.get("model", payload) if isinstance(payload, dict) else payload
        incompatible = model.load_state_dict(state_dict, strict=False)
        checkpoint_path = str(checkpoint_file)
        print(json.dumps({"checkpoint_load": {"missing_keys": len(incompatible.missing_keys), "unexpected_keys": len(incompatible.unexpected_keys)}}))

    items = load_probe_items(dataset_jsonl)
    dataset_root = Path(dataset_jsonl).resolve().parent
    chunks: list[np.ndarray] = []
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        samples: dict[str, Any] = {}
        if mode in {"image", "multimodal"}:
            processor = vis_processors.get("eval") if vis_processors else None
            if processor is None:
                raise ValueError("Selected model has no eval visual processor.")
            images = []
            for item in batch:
                if not item.image:
                    raise ValueError(f"Sample {item.sample_id} requires image input.")
                image_path = Path(item.image)
                if not image_path.is_absolute():
                    image_path = dataset_root / image_path
                images.append(processor(Image.open(image_path).convert("RGB")))
            samples["image"] = torch.stack(images).to(resolved_device)
        if mode in {"text", "multimodal"}:
            processor = txt_processors.get("eval") if txt_processors else None
            texts = []
            for item in batch:
                if item.text is None:
                    raise ValueError(f"Sample {item.sample_id} requires text input.")
                texts.append(processor(item.text) if processor else item.text)
            samples["text_input"] = texts
        with torch.no_grad():
            output = model.extract_features(samples, mode=mode)
            features = _pool_features(_feature_tensor(output, mode, feature_field), pooling)
        chunks.append(features.detach().float().cpu().numpy())

    embeddings = np.concatenate(chunks, axis=0)
    labels = [item.label for item in items]
    metadata = {"model_name": model_name, "model_type": model_type, "mode": mode, "pooling": pooling, "feature_field": feature_field, "checkpoint": checkpoint_path, "device": resolved_device, "dataset": str(Path(dataset_jsonl).resolve())}
    save_snapshot(output_path, embeddings, [item.sample_id for item in items], labels=labels if all(label is not None for label in labels) else None, metadata=metadata)
    return {"output": str(Path(output_path)), "samples": len(items), "dimensions": int(embeddings.shape[1]), **metadata}
