# Vision Language Workbench

Vision Language Workbench keeps the proven LAVIS model, processor, dataset,
task, runner, training, and evaluation code intact while adding a lightweight
experiment layer for repeatable local and GPU runs.

The goal is not to reimplement BLIP, BLIP-2, ALBEF, CLIP, VQA, captioning, or
retrieval. Those capabilities already exist in the included LAVIS codebase.
This project adds a small workflow around them so an experiment can be defined,
reviewed, executed, and traced without editing shell scripts for every run.

## What is added

- YAML/JSON experiment manifests.
- Deterministic command planning for `train.py` and `evaluate.py`.
- Config and manifest fingerprints for reproducibility.
- A run ledger that records the exact plan and exit status.
- Frozen representation snapshots and linear/geometry probes.
- A compact CLI: `vl-workbench`.

## Quick start

Create a manifest:

```yaml
name: clip-coco-retrieval-eval
mode: evaluate
cfg_path: projects/clip/exp_coco_ret_eval.yaml
options:
  run.num_workers: 4
tags:
  - retrieval
  - clip
```

Inspect the exact command without launching a model:

```bash
vl-workbench plan experiments/clip-coco.yaml
```

Execute the same plan and record it:

```bash
vl-workbench run experiments/clip-coco.yaml
```

Review recent execution provenance without opening model logs:

```bash
vl-workbench history --limit 10

# expand 4 concrete LAVIS runs without writing duplicate scripts
vl-workbench sweep-plan experiments/clip-coco-retrieval-sweep.yaml

# optionally materialize the variants as ordinary manifests
vl-workbench sweep-materialize experiments/clip-coco-retrieval-sweep.yaml

# inspect the output directory declared by the underlying LAVIS config
vl-workbench artifacts experiments/blip-caption-resume-example.yaml

# continue training from the newest checkpoint without hand-editing LAVIS YAML
vl-workbench resume-plan experiments/blip-caption-resume-example.yaml
```

The ledger stores only compact run metadata under `artifacts/` and is ignored by
Git, so model outputs and large generated files do not pollute repository history.

## Measured results

The first canonical study uses the real
[COCO 2017](https://cocodataset.org/#download) validation images and human
captions with the pretrained
[`Salesforce/blip-itm-base-coco`](https://huggingface.co/Salesforce/blip-itm-base-coco)
checkpoint. A deterministic, leakage-free subset contains 64 training pairs
and 80 held-out probe pairs across `person`, `car`, `dog`, and `cat`. Mean
retrieval recall is the average of image-to-text and text-to-image recall.

| Variant | Trainable parameters | Trainable % | Mean R@1 | Mean R@5 | Mean R@10 | Linear probe | Class separation | Anisotropy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pretrained Base | 0 | 0.0000% | 0.95625 | 1.00000 | 1.00000 | 1.00000 | -0.22854 | 0.41491 |
| LoRA Q/V r=8 | 294,912 | 0.1318% | 0.93750 | 1.00000 | 1.00000 | 1.00000 | -0.22623 | 0.39830 |
| LoRA Q/V r=16 | 589,824 | 0.2636% | 0.94375 | 1.00000 | 1.00000 | 1.00000 | -0.22369 | 0.38666 |
| LoRA Q/V r=8 + mined hard negative | 294,912 | 0.1318% | 0.95000 | 1.00000 | 1.00000 | 1.00000 | -0.22647 | 0.39887 |

LoRA used identical data and budget for both ranks: 2 epochs, 16 optimizer
steps, batch size 8, learning rate `1e-4`, and a symmetric contrastive loss.
R@1 decreased by 0.01875 for r=8 and 0.01250 for r=16 versus Base; these are
the canonical outcomes and were not retuned. Fused-space CKA versus Base was
0.999953 (r=8) and 0.999850 (r=16), with mean cosine drift 0.000300 and
0.000862 respectively. The result indicates that this tiny Q/V-only update
slightly reduces retrieval R@1 while barely moving the held-out representation
space, although its image-to-text NLL improves from 1.16840 to 1.10865/1.06734.

For the hard-negative variant, the pretrained train embeddings mine one
nearest wrong caption per image (`different-id`; mean cosine 0.35994), and the
existing margin loss is added with margin 0.2 and weight 1.0 under the same r=8
budget. It recovers 0.01250 R@1 over ordinary r=8 but remains 0.00625 below
Base. Its 16-step mean hard-negative loss was 0.09550. No outcome-driven
hyperparameter retry was performed.

The exact saved canonical adapters were then evaluated on five lazy image
corruptions at severity 1–3 (15 matched conditions, 80 samples each):

| Variant | Clean R@1 | Mean OOD R@1 | Mean retention | Worst retention | Worst condition |
|---|---:|---:|---:|---:|---|
| Pretrained Base | 0.95625 | 0.91625 | 0.95817 | 0.78431 | occlusion s3 |
| LoRA Q/V r=8 | 0.93750 | 0.91250 | 0.97333 | 0.80000 | occlusion s3 |
| LoRA Q/V r=16 | 0.94375 | 0.91458 | 0.96909 | 0.80795 | occlusion s3 |
| LoRA Q/V r=8 + mined hard negative | 0.95000 | 0.91542 | 0.96360 | 0.78947 | occlusion s3 |

LoRA improves relative retention because it starts from a lower clean score,
but Base still has the highest absolute mean OOD R@1. Occlusion is materially
harder than blur, noise, JPEG, or low light for every variant. Clean
representations re-extracted from the saved adapters match the canonical NPZ
files exactly (maximum absolute difference 0.0).

This is an actual A100/bfloat16 run, not a synthetic smoke test. Its pinned
environment, sample IDs, confidence records, and representation NPZ files are
stored under [`results/canonical-blip-coco-small-v1`](results/canonical-blip-coco-small-v1).

Search bundled LAVIS configs without importing any heavyweight model package:

```bash
vl-workbench catalog blip2 --kind model
vl-workbench catalog retrieval --kind project
```

This catalog is filesystem-based, so it stays fast and does not trigger checkpoint
downloads or GPU initialization.

The underlying execution still goes through the original LAVIS entrypoints:
`train.py` and `evaluate.py`.

## Representation research

Intermediate LAVIS representations can be captured as reusable NumPy snapshots
and analyzed without repeatedly loading model weights. The extractor calls the
bundled model's existing `extract_features()` implementation rather than
recreating an encoder.

Probe data is JSONL with stable ids and optional labels:

```json
{"id":"dog-001","image":"images/dog.jpg","text":"a black dog running","label":"dog"}
{"id":"cat-001","image":"images/cat.jpg","text":"a tabby cat sitting","label":"cat"}
```

```bash
vl-workbench extract-features probes.jsonl \
  --model-name blip_feature_extractor \
  --model-type base \
  --mode image \
  --pooling cls \
  --output artifacts/blip-base-image.npz

vl-workbench probe artifacts/blip-base-image.npz
```

`extract-features` also accepts `--checkpoint`, allowing the identical probe
set to be captured before and after fine-tuning. `probe` reports frozen ridge
linear-probe accuracy, within-class cosine cohesion, between-class centroid
similarity, separation margin, and embedding anisotropy.

Compare a base snapshot against a fine-tuned checkpoint snapshot:

```bash
vl-workbench drift \
  artifacts/blip-base-image.npz \
  artifacts/blip-finetuned-image.npz \
  --top-k 20
```

Drift analysis aligns samples by stable ids and reports linear CKA, per-sample
cosine drift, anisotropy change, class-separation change, class-centroid drift,
and the samples whose representations moved the most. CKA remains valid even
when the compared feature dimensions differ; direct cosine metrics are emitted
when dimensions match.

## Hard-negative mining

Use the same saved representation space to find examples the current model is
most likely to confuse:

```bash
vl-workbench hard-negatives artifacts/blip-base-image.npz \
  --policy different-label \
  --top-k 5 \
  --output artifacts/hard-negatives.jsonl
```

For cross-modal contrastive mining, pass a second snapshot (for example image
anchors against text candidates) and use stable ids to exclude the true pair:

```bash
vl-workbench hard-negatives artifacts/images.npz \
  --candidates artifacts/text.npz \
  --policy different-id \
  --top-k 10
```

Mining is cosine-nearest-neighbor based and chunked to avoid allocating the
entire similarity matrix. The resulting JSONL can be consumed by later
fine-tuning data builders without changing the bundled LAVIS encoders.

## LoRA placement research

The bundled xInstructBLIP code already uses PEFT LoRA for its LLM path. The
workbench adds a generic research policy for any LAVIS `nn.Linear` tree so LoRA
rank and placement are explicit experiment variables instead of fixed choices.

Policies under `research/lora/` can target module suffixes or regular
expressions. `compare_lora_policies()` reports matched layers and exact adapter
parameter cost before training. `inject_lora()` installs zero-initialized
low-rank residuals while preserving the underlying LAVIS architecture.

This supports controlled studies such as Q/V rank 8 vs 16, projection-only
LoRA, or vision/Q-Former/LLM placement at a known trainable-parameter budget.

## Hard negatives in the training loop

Mined neighbors can now be joined back to the original probe JSONL as concrete
LAVIS retrieval annotations with `materialize_hard_negative_annotations()`.
The registered `hard_negative_retrieval` dataset returns the normal positive
caption plus a mined `hard_negative_text` and its cosine hardness score.

`hard_negative_margin_loss()` then adds an explicit cosine ranking objective on
top of features produced by the existing LAVIS encoders. It supports one or
multiple negatives per image and optional hardness weighting, allowing studies
of random vs mined negatives without rewriting BLIP/ALBEF encoders.

For end-to-end training, `blip_retrieval_hard_negative` subclasses the original
`BlipRetrieval`: the full upstream ITC/ITM forward pass is reused, then the
already-computed positive image/text embeddings are paired with the mined
negative caption and an extra weighted margin term is added to the loss. The
model config exposes `hard_negative_margin` and `hard_negative_weight` as normal
experiment variables.

## Repository shape

```text
lavis/                  LAVIS models, datasets, processors, tasks, runners
projects/               LAVIS project configs retained for direct reuse
run_scripts/            Existing runnable training/evaluation examples
train.py                Original LAVIS training entrypoint
evaluate.py             Original LAVIS evaluation entrypoint
workbench/              Oosu experiment orchestration and provenance layer
experiments/            Small reusable manifests for this workbench
```

## Origin and licensing

This repository was created from a clean source snapshot rather than a fork and
contains no upstream Git history. The underlying LAVIS code remains under its
BSD 3-Clause license. See `LICENSE.txt` and `THIRD_PARTY_NOTICE.md`.
