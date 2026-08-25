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
