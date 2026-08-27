# Harder multi-seed BLIP canonical study v2

This is a measured Google Colab A100 result over seeds 42, 1337, and 2026, not
a smoke fixture. Each seed uses 128 training pairs and 256 disjoint held-out
probe pairs (32/64 per class for `person`, `car`, `dog`, and `cat`) with the
pinned `Salesforce/blip-itm-base-coco` revision. Training remains two epochs,
batch size 8, learning rate `1e-4`, and the configured Q/V-only LoRA policies.

The first attempt exposed a real subset-selection bug: excluding every image
containing two study categories left only 81 eligible `dog` images for the 96
required. Commit `b244cbe` assigns valid candidates from the scarcest class
first with global image-id disjointness, preserving the requested balanced
counts and zero train/probe overlap. Commit `1eb76fd` only parallelizes image
cache downloads to avoid paid A100 idle time. No model, seed, training budget,
or evaluation condition was changed.

## Execution provenance

The run was operated in Google Colab with an `NVIDIA A100-SXM4-40GB`. The
interactive browser session was used to observe runtime allocation, the first
allocation failure, cache progress, and the seed-by-seed execution. Four
screenshots from that session are now committed alongside the machine-readable
artifacts so the human execution record and the reproducible result record stay
together.

| Stage | Screenshot |
|---|---|
| A100 runtime selected | ![A100 runtime selected](screenshots/01-a100-runtime-selection.png) |
| Multi-seed experiment running | ![A100 experiment running](screenshots/02-a100-experiment-running.png) |
| Seeds 42/1337/2026 validated and ZIP prepared | ![Three seeds validated](screenshots/03-three-seeds-validated-zip.png) |
| Runtime disconnected and deleted after artifact capture | ![Runtime disconnected and deleted](screenshots/04-runtime-disconnected-deleted.png) |

The durable execution trail is:

1. `b244cbe` — corrected balanced class allocation after the real
   `dog: 81 available, 96 required` failure while keeping requested sample
   counts and zero train/probe overlap.
2. `1eb76fd` — parallelized COCO image-cache downloads only, avoiding paid A100
   idle time without touching model or evaluation settings.
3. `b4e0b6c` — added seed-completeness verification and direct paired
   `r8+HN − r8` analysis.
4. Seeds `42`, `1337`, and `2026` completed with all four variants.
5. `f86ab7f` — published the measured canonical v2 results.
6. `a9281b3` — linked this source experiment to its downstream evaluation
   record in `multimodal-eval-workbench`.

The first canonical study's full eight-image execution record remains under
`../canonical-blip-coco-small-v1/screenshots/`; the root README surfaces only
the representative images needed to tell the end-to-end story.

## Three-seed results

Values are mean ± sample standard deviation; the JSON summary also records
Student-t 95% confidence intervals for n=3.

| Variant | Params % | Clean R@1 | Mean OOD R@1 | Retention | Worst retention | CKA vs Base | Cosine drift | Probe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 0.0000 | 0.85612 ± 0.01567 | 0.75303 ± 0.01991 | 0.87951 ± 0.00962 | 0.41030 ± 0.04440 | 1.000000 | 0.000000 | 0.78646 ± 0.03253 |
| LoRA r=8 | 0.1318 | 0.84766 ± 0.01367 | 0.74966 ± 0.01778 | 0.88431 ± 0.00666 | 0.41134 ± 0.03636 | 0.999477 | 0.002537 | 0.78646 ± 0.03253 |
| LoRA r=16 | 0.2636 | 0.84701 ± 0.01591 | 0.74556 ± 0.01665 | 0.88019 ± 0.00495 | 0.41620 ± 0.03852 | 0.998371 | 0.008515 | 0.78646 ± 0.03253 |
| LoRA r=8 + hard negative | 0.1318 | 0.84766 ± 0.01367 | 0.75050 ± 0.01700 | 0.88532 ± 0.00574 | 0.41284 ± 0.03741 | 0.999543 | 0.002040 | 0.78646 ± 0.03253 |

## Paired seed interpretation

| Contrast | Clean R@1 delta (95% CI) | OOD R@1 delta (95% CI) | Retention delta (95% CI) |
|---|---:|---:|---:|
| r8 − Base | -0.00846 ± 0.01704 | -0.00337 ± 0.00751 | +0.00480 ± 0.01183 |
| r16 − Base | -0.00911 ± 0.01837 | -0.00747 ± 0.01745 | +0.00069 ± 0.01488 |
| r8+HN − r8 | 0.00000 ± 0.00000 | +0.00084 ± 0.00214 | +0.00100 ± 0.00256 |

Both LoRA ranks lose clean R@1 on all three seeds, so the clean-regression
direction from the first canonical study reproduces. The n=3 confidence
intervals still include zero. Hard-negative training does not recover clean
R@1 here: it ties ordinary r8 on every seed. Its small OOD and retention gains
are inconclusive at three seeds.

At occlusion severity 4, r8 has the highest mean retention (0.51829); at
severity 5, r16 is highest (0.41620). The r8+HN two-severity average is only
fractionally highest, and intervals overlap widely, so there is no defensible
single high-severity winner. Representation CKA remains near one for every
LoRA variant while r16 produces the largest cosine drift.

## Contents

- `multiseed-summary.json` / `.csv`: aggregate mean, standard deviation, and 95% CI.
- `paired-comparisons.json`: paired seed deltas, including r8+HN minus r8.
- `occlusion-severity-4-5.json` / `.csv`: high-severity OOD results.
- `seed-*`: resolved config, run manifest, comparison, compact variant metrics, and OOD metrics.

Large adapters, representation NPZ files, model weights, and the COCO cache are
intentionally excluded from Git. The downloaded canonical ZIP SHA-256 is
`5fc52a37700b83c08c7ce53bd1f361f1699a8263a78b7c902f6465ab60905867`.

## Research lineage

`research-lineage.json` makes the cross-repository research chain explicit.
This directory is the measurement source; its downstream calibration, OOD,
Pareto, and replication analysis lives in
`oosuhada/multimodal-eval-workbench` at
`results/canonical-blip-coco-harder-multiseed-v2` (evaluation result commit
`1d4f784`). The manifest also records which v1 observations replicated,
remained inconclusive, or failed to reproduce in this harder three-seed study.
