#!/usr/bin/env python3
"""Run the first small, real-data BLIP representation/LoRA study.

The runner deliberately downloads only a deterministic COCO 2017 validation
subset and never writes model weights into the repository.  Small metrics and
representation snapshots can be copied into a tracked results directory after
the run has been validated.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import time
from typing import Any, Iterable
import urllib.request
import zipfile

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import BlipForImageTextRetrieval, BlipProcessor

# Direct ``python research/...py`` execution otherwise exposes only the
# research directory on sys.path, not the repository package root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workbench.drift import compare_snapshots
from workbench.hard_negative_training import hard_negative_margin_loss
from workbench.hard_negatives import mine_hard_negatives, write_hard_negative_jsonl
from workbench.lora_research import LoRAPolicy, inject_lora
from workbench.representations import load_snapshot, probe_snapshot, save_snapshot


CAPTIONS_URL = "https://huggingface.co/datasets/merve/coco/resolve/main/annotations/captions_val2017.json"
INSTANCES_URL = "https://huggingface.co/datasets/merve/coco/resolve/main/annotations/instances_val2017.json"
# The official COCO image host currently presents a mismatched TLS certificate
# from Colab.  This Hugging Face dataset mirror stores the same val2017 files
# beside the annotation mirror and keeps certificate verification enabled.
COCO_IMAGE_URL = "https://huggingface.co/datasets/merve/coco/resolve/main/val2017/{image_id:012d}.jpg"
LORA_TARGET = r"^text_encoder\.encoder\.layer\.\d+\.attention\.self\.(query|value)$"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("research/canonical/blip_coco_small.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/canonical-blip-coco-small-v1"))
    parser.add_argument("--cache", type=Path, default=Path("/content/blip-coco-cache"))
    parser.add_argument("--variants", default="base,lora-r8,lora-r16")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def download(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "vision-language-workbench/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            temporary.replace(destination)
            return
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_coco_subset(config: dict[str, Any], cache: Path, output: Path) -> list[dict[str, Any]]:
    annotations = cache / "annotations"
    captions_path = annotations / "captions_val2017.json"
    instances_path = annotations / "instances_val2017.json"
    download(CAPTIONS_URL, captions_path)
    download(INSTANCES_URL, instances_path)
    captions = load_json(captions_path)
    instances = load_json(instances_path)

    category_to_id = {row["name"]: int(row["id"]) for row in instances["categories"]}
    selected_names = list(config["categories"])
    selected_ids = {category_to_id[name] for name in selected_names}
    id_to_name = {category_to_id[name]: name for name in selected_names}
    image_categories: dict[int, set[int]] = defaultdict(set)
    for annotation in instances["annotations"]:
        image_categories[int(annotation["image_id"])].add(int(annotation["category_id"]))
    image_captions: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for annotation in captions["annotations"]:
        image_captions[int(annotation["image_id"])].append((int(annotation["id"]), str(annotation["caption"])))

    candidates: dict[str, list[int]] = {name: [] for name in selected_names}
    for image_id, categories in image_categories.items():
        present = categories & selected_ids
        if len(present) == 1 and image_id in image_captions:
            candidates[id_to_name[next(iter(present))]].append(image_id)

    rng = random.Random(int(config["seed"]))
    records: list[dict[str, Any]] = []
    needed = int(config["train_per_class"]) + int(config["probe_per_class"])
    for class_index, name in enumerate(selected_names):
        ids = sorted(candidates[name])
        rng.shuffle(ids)
        if len(ids) < needed:
            raise RuntimeError(f"COCO class {name!r} has {len(ids)} eligible images, need {needed}.")
        for offset, image_id in enumerate(ids[:needed]):
            split = "train" if offset < int(config["train_per_class"]) else "probe"
            caption = sorted(image_captions[image_id])[0][1]
            image_path = cache / "images" / f"{image_id:012d}.jpg"
            download(COCO_IMAGE_URL.format(image_id=image_id), image_path)
            with Image.open(image_path) as image:
                image.verify()
            records.append(
                {
                    "sample_id": f"coco-val2017-{image_id:012d}",
                    "image_id": image_id,
                    "image_path": str(image_path),
                    "image_url": COCO_IMAGE_URL.format(image_id=image_id),
                    "caption": caption,
                    "label": name,
                    "label_index": class_index,
                    "split": split,
                }
            )
    manifest_path = output / "dataset_manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            public_record = {key: value for key, value in record.items() if key != "image_path"}
            handle.write(json.dumps(public_record, sort_keys=True) + "\n")
    return records


def batched(values: list[Any], size: int) -> Iterable[list[Any]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def core_embeddings(
    model: BlipForImageTextRetrieval,
    processor: BlipProcessor,
    records: list[dict[str, Any]],
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    image_rows: list[np.ndarray] = []
    text_rows: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in batched(records, batch_size):
            images = [Image.open(row["image_path"]).convert("RGB") for row in batch]
            encoded = processor(images=images, text=[row["caption"] for row in batch], padding=True, return_tensors="pt")
            pixel_values = encoded["pixel_values"].to(device=device, dtype=dtype)
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            with autocast_context(device, dtype):
                vision = model.vision_model(pixel_values=pixel_values, return_dict=True).last_hidden_state[:, 0]
                text = model.text_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state[:, 0]
                image_features = F.normalize(model.vision_proj(vision), dim=-1)
                text_features = F.normalize(model.text_proj(text), dim=-1)
            image_rows.append(image_features.float().cpu().numpy())
            text_rows.append(text_features.float().cpu().numpy())
            for image in images:
                image.close()
    image_matrix = np.concatenate(image_rows)
    text_matrix = np.concatenate(text_rows)
    fused = image_matrix + text_matrix
    fused /= np.maximum(np.linalg.norm(fused, axis=1, keepdims=True), 1e-12)
    return image_matrix, text_matrix, fused


def retrieval_metrics(image: np.ndarray, text: np.ndarray, temperature: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    similarities = image @ text.T
    n = similarities.shape[0]
    target = np.arange(n)
    rows: dict[str, float] = {}
    for direction, matrix in (("image_to_text", similarities), ("text_to_image", similarities.T)):
        order = np.argsort(-matrix, axis=1)
        ranks = np.asarray([int(np.flatnonzero(order[index] == target[index])[0]) + 1 for index in range(n)])
        for k in (1, 5, 10):
            rows[f"{direction}_r_at_{k}"] = float(np.mean(ranks <= min(k, n)))
        rows[f"{direction}_median_rank"] = float(np.median(ranks))
    for k in (1, 5, 10):
        rows[f"mean_r_at_{k}"] = float(
            (rows[f"image_to_text_r_at_{k}"] + rows[f"text_to_image_r_at_{k}"]) / 2.0
        )
    scaled = similarities / float(temperature)
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    predicted = probabilities.argmax(axis=1)
    records = [
        {
            "index": int(index),
            "target_index": int(index),
            "predicted_index": int(predicted[index]),
            "confidence": float(probabilities[index, predicted[index]]),
            "correct": bool(predicted[index] == index),
            "nll": float(-np.log(max(probabilities[index, index], 1e-12))),
        }
        for index in range(n)
    ]
    rows["image_to_text_nll"] = float(np.mean([row["nll"] for row in records]))
    return {"samples": n, **rows}, records


def train_lora(
    model: BlipForImageTextRetrieval,
    processor: BlipProcessor,
    train_records: list[dict[str, Any]],
    config: dict[str, Any],
    variant: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    hard_negatives: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    policy = LoRAPolicy(
        name=str(variant["name"]),
        rank=int(variant["rank"]),
        alpha=float(variant["alpha"]),
        dropout=float(config["lora_dropout"]),
        target_regex=(LORA_TARGET,),
    )
    budget = inject_lora(model, policy)
    # The generic injector creates fresh Linear modules on CPU; place the
    # adapters beside the already-loaded base weights before optimization.
    model.to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    generator = torch.Generator().manual_seed(int(config["seed"]))
    losses: list[float] = []
    contrastive_losses: list[float] = []
    hard_negative_losses: list[float] = []
    model.train()
    started = time.perf_counter()
    for epoch in range(int(config["epochs"])):
        permutation = torch.randperm(len(train_records), generator=generator).tolist()
        ordered = [train_records[index] for index in permutation]
        for batch in batched(ordered, int(config["batch_size"])):
            images = [Image.open(row["image_path"]).convert("RGB") for row in batch]
            encoded = processor(images=images, text=[row["caption"] for row in batch], padding=True, return_tensors="pt")
            pixel_values = encoded["pixel_values"].to(device=device, dtype=dtype)
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), autocast_context(device, dtype):
                vision = model.vision_model(pixel_values=pixel_values, return_dict=True).last_hidden_state[:, 0]
                image_features = F.normalize(model.vision_proj(vision), dim=-1)
            with autocast_context(device, dtype):
                text = model.text_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state[:, 0]
                text_features = F.normalize(model.text_proj(text), dim=-1)
                logits = image_features @ text_features.T / float(config["contrastive_temperature"])
                targets = torch.arange(len(batch), device=device)
                contrastive_loss = (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2.0
                loss = contrastive_loss
                hard_negative_loss = None
                if hard_negatives is not None:
                    mined = [hard_negatives[row["sample_id"]] for row in batch]
                    negative_tokens = processor(
                        text=[row["caption"] for row in mined], padding=True, return_tensors="pt"
                    )
                    negative_text = model.text_encoder(
                        input_ids=negative_tokens["input_ids"].to(device),
                        attention_mask=negative_tokens["attention_mask"].to(device),
                        return_dict=True,
                    ).last_hidden_state[:, 0]
                    negative_features = F.normalize(model.text_proj(negative_text), dim=-1)
                    hardness = torch.as_tensor(
                        [row["cosine_similarity"] for row in mined], device=device, dtype=image_features.dtype
                    ).clamp_min(0)
                    hard_negative_loss = hard_negative_margin_loss(
                        image_features,
                        text_features,
                        negative_features,
                        margin=float(variant["hard_negative_margin"]),
                        hardness=hardness,
                    )
                    loss = loss + float(variant["hard_negative_weight"]) * hard_negative_loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            contrastive_losses.append(float(contrastive_loss.detach().cpu()))
            if hard_negative_loss is not None:
                hard_negative_losses.append(float(hard_negative_loss.detach().cpu()))
            for image in images:
                image.close()
            hard_text = f" hard_negative={hard_negative_losses[-1]:.6f}" if hard_negative_losses else ""
            print(
                f"TRAIN variant={variant['name']} epoch={epoch + 1} step={len(losses)} "
                f"loss={losses[-1]:.6f} contrastive={contrastive_losses[-1]:.6f}{hard_text}",
                flush=True,
            )
    budget.update(
        {
            "epochs": int(config["epochs"]),
            "steps": len(losses),
            "samples": len(train_records),
            "learning_rate": float(config["learning_rate"]),
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "mean_loss": float(np.mean(losses)),
            "mean_contrastive_loss": float(np.mean(contrastive_losses)),
            "mean_hard_negative_loss": float(np.mean(hard_negative_losses)) if hard_negative_losses else None,
            "hard_negative": bool(hard_negatives is not None),
            "hard_negative_config": {
                key: variant[key]
                for key in ("mining_policy", "hard_negative_margin", "hard_negative_weight")
                if key in variant
            },
            "training_seconds": time.perf_counter() - started,
        }
    )
    return budget


def mine_training_negatives(
    model: BlipForImageTextRetrieval,
    processor: BlipProcessor,
    train_records: list[dict[str, Any]],
    output: Path,
    config: dict[str, Any],
    variant: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, dict[str, Any]]:
    image, text, _ = core_embeddings(model, processor, train_records, device, dtype, int(config["batch_size"]))
    ids = [row["sample_id"] for row in train_records]
    labels = [row["label"] for row in train_records]
    variant_dir = output / str(variant["name"])
    image_path = variant_dir / "mining_image_representations.npz"
    text_path = variant_dir / "mining_text_representations.npz"
    metadata = {"variant": variant["name"], "split": "train", "purpose": "hard-negative-mining"}
    save_snapshot(image_path, image, ids, labels, {**metadata, "modality": "image"})
    save_snapshot(text_path, text, ids, labels, {**metadata, "modality": "text"})
    result = mine_hard_negatives(
        load_snapshot(image_path),
        load_snapshot(text_path),
        top_k=1,
        policy=str(variant["mining_policy"]),
        chunk_size=512,
    )
    write_json(variant_dir / "hard_negative_mining_summary.json", {key: value for key, value in result.items() if key != "items"})
    write_hard_negative_jsonl(result, variant_dir / "hard_negatives.jsonl")
    records_by_id = {row["sample_id"]: row for row in train_records}
    mined: dict[str, dict[str, Any]] = {}
    for item in result["items"]:
        if not item["negatives"]:
            raise RuntimeError(f"No hard negative for {item['anchor_id']}")
        negative = item["negatives"][0]
        candidate = records_by_id[str(negative["id"])]
        mined[str(item["anchor_id"])] = {
            "id": str(negative["id"]),
            "caption": candidate["caption"],
            "cosine_similarity": float(negative["cosine_similarity"]),
        }
    if len(mined) != len(train_records):
        raise RuntimeError("Hard-negative mining did not cover every training pair.")
    return mined


def save_adapter(model: torch.nn.Module, output: Path, training: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    adapter_state = {name: value.detach().cpu() for name, value in model.state_dict().items() if "lora_" in name}
    torch.save(adapter_state, output / "adapter_model.pt")
    write_json(output / "adapter_config.json", training["policy"])


def load_model(config: dict[str, Any], device: torch.device, dtype: torch.dtype):
    processor = BlipProcessor.from_pretrained(config["model_id"], revision=config["model_revision"])
    model = BlipForImageTextRetrieval.from_pretrained(
        config["model_id"], revision=config["model_revision"], torch_dtype=dtype
    ).to(device)
    return model, processor


def evaluate_variant(
    name: str,
    model: BlipForImageTextRetrieval,
    processor: BlipProcessor,
    probe_records: list[dict[str, Any]],
    output: Path,
    config: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    training: dict[str, Any] | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    image, text, fused = core_embeddings(model, processor, probe_records, device, dtype, int(config["batch_size"]))
    ids = [row["sample_id"] for row in probe_records]
    labels = [row["label"] for row in probe_records]
    variant_dir = output / name
    metadata = {"variant": name, "model": config["model_id"], "revision": config["model_revision"], "split": "probe"}
    save_snapshot(variant_dir / "image_representations.npz", image, ids, labels, {**metadata, "modality": "image"})
    save_snapshot(variant_dir / "text_representations.npz", text, ids, labels, {**metadata, "modality": "text"})
    save_snapshot(variant_dir / "fused_representations.npz", fused, ids, labels, {**metadata, "modality": "fused"})
    probe = probe_snapshot(load_snapshot(variant_dir / "fused_representations.npz"), test_fraction=0.25, ridge=1.0, seed=int(config["seed"]))
    retrieval, confidence = retrieval_metrics(image, text, float(config["contrastive_temperature"]))
    for row, sample in zip(confidence, probe_records):
        row["sample_id"] = sample["sample_id"]
        row["label"] = sample["label"]
        row["variant"] = name
    with (variant_dir / "confidence_records.jsonl").open("w", encoding="utf-8") as handle:
        for row in confidence:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    metrics = {
        "variant": name,
        "probe_samples": len(probe_records),
        "retrieval": retrieval,
        "representation": probe,
        "training": training,
        "evaluation_seconds": time.perf_counter() - started,
    }
    write_json(variant_dir / "metrics.json", metrics)
    return metrics


def result_row(metrics: dict[str, Any], drift: dict[str, Any] | None) -> dict[str, Any]:
    training = metrics.get("training") or {}
    representation = metrics["representation"]
    retrieval = metrics["retrieval"]
    return {
        "variant": metrics["variant"],
        "trainable_parameters": int(training.get("trainable_parameters_after_injection", 0)),
        "trainable_parameter_pct": float(training.get("adapter_fraction_of_base", 0.0) * 100.0),
        "retrieval_mean_r_at_1": retrieval["mean_r_at_1"],
        "retrieval_mean_r_at_5": retrieval["mean_r_at_5"],
        "retrieval_mean_r_at_10": retrieval["mean_r_at_10"],
        "linear_probe_accuracy": representation["linear_probe"]["test_accuracy"],
        "class_separation": representation["class_separation"]["separation_margin"],
        "anisotropy": representation["anisotropy"],
        "cka_vs_base": 1.0 if drift is None else drift["linear_cka"],
        "mean_cosine_drift": 0.0 if drift is None else drift["sample_cosine"]["mean_drift"],
        "evaluation_seconds": metrics["evaluation_seconds"],
    }


def write_comparison(output: Path, rows: list[dict[str, Any]]) -> None:
    write_json(output / "comparison.json", rows)
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    headers = ["Variant", "Trainable %", "R@1", "R@5", "R@10", "Probe Acc", "Separation", "Anisotropy", "CKA", "Cosine Drift"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append(
            "| {variant} | {trainable_parameter_pct:.4f}% | {retrieval_mean_r_at_1:.4f} | {retrieval_mean_r_at_5:.4f} | "
            "{retrieval_mean_r_at_10:.4f} | {linear_probe_accuracy:.4f} | {class_separation:.4f} | {anisotropy:.4f} | "
            "{cka_vs_base:.4f} | {mean_cosine_drift:.4f} |".format(**row)
        )
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_zip(output: Path) -> Path:
    zip_path = output.parent / f"{output.name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output.parent))
    return zip_path


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    requested = [value.strip() for value in args.variants.split(",") if value.strip()]
    if not torch.cuda.is_available():
        raise RuntimeError("Canonical training must run on a CUDA GPU.")
    random.seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    capability = torch.cuda.get_device_capability(device)
    dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = prepare_coco_subset(config, args.cache.resolve(), output)
    train_records = [row for row in records if row["split"] == "train"]
    probe_records = [row for row in records if row["split"] == "probe"]
    expected_train = len(config["categories"]) * int(config["train_per_class"])
    expected_probe = len(config["categories"]) * int(config["probe_per_class"])
    assert len(train_records) == expected_train and len(probe_records) == expected_probe
    assert not ({row["sample_id"] for row in train_records} & {row["sample_id"] for row in probe_records})

    manifest = {
        "study": config["study"],
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": os.environ.get("GIT_COMMIT", "unknown"),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": package_version("transformers"),
        "pillow": package_version("Pillow"),
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "gpu_capability": list(capability),
        "dtype": str(dtype),
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "dataset": "COCO 2017 validation subset",
        "train_samples": len(train_records),
        "probe_samples": len(probe_records),
        "categories": config["categories"],
        "config": config,
        "requested_variants": requested,
        "train_probe_overlap": 0,
    }
    write_json(output / "run_manifest.json", manifest)
    print("ENV " + json.dumps({key: manifest[key] for key in ("gpu", "dtype", "python", "torch", "transformers")}), flush=True)
    print(f"DATA train={len(train_records)} probe={len(probe_records)} overlap=0", flush=True)

    all_metrics: dict[str, dict[str, Any]] = {}
    if "base" in requested:
        model, processor = load_model(config, device, dtype)
        all_metrics["base"] = evaluate_variant("base", model, processor, probe_records, output, config, device, dtype, None)
        del model
        torch.cuda.empty_cache()
        print("COMPLETE variant=base", flush=True)

    base_snapshot = load_snapshot(output / "base" / "fused_representations.npz")
    variant_map = {row["name"]: row for row in config["variants"]}
    for name in requested:
        if name == "base":
            continue
        variant = variant_map[name]
        model, processor = load_model(config, device, dtype)
        hard_negatives = None
        if variant.get("hard_negative"):
            hard_negatives = mine_training_negatives(
                model, processor, train_records, output, config, variant, device, dtype
            )
        training = train_lora(
            model,
            processor,
            train_records,
            config,
            variant,
            device,
            dtype,
            hard_negatives=hard_negatives,
        )
        save_adapter(model, output / "adapters" / name, training)
        metrics = evaluate_variant(name, model, processor, probe_records, output, config, device, dtype, training)
        current = load_snapshot(output / name / "fused_representations.npz")
        drift = compare_snapshots(base_snapshot, current, top_k=10)
        write_json(output / name / "drift_vs_base.json", drift)
        metrics["drift_vs_base"] = drift
        write_json(output / name / "metrics.json", metrics)
        all_metrics[name] = metrics
        del model
        torch.cuda.empty_cache()
        print(f"COMPLETE variant={name}", flush=True)

    rows = []
    for name in requested:
        metrics = all_metrics[name]
        rows.append(result_row(metrics, None if name == "base" else metrics["drift_vs_base"]))
    write_comparison(output, rows)
    zip_path = make_zip(output)
    print("RESULT_ZIP " + str(zip_path), flush=True)
    print((output / "comparison.md").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
