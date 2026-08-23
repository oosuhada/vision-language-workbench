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
```

The ledger stores only compact run metadata under `artifacts/` and is ignored by
Git, so model outputs and large generated files do not pollute repository history.

The underlying execution still goes through the original LAVIS entrypoints:
`train.py` and `evaluate.py`.

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
