# Canonical BLIP/COCO small study

This directory contains measured artifacts from the first complete real-data
research loop. It is separate from synthetic and unit-test fixtures.

- Model: `Salesforce/blip-itm-base-coco`
- Checkpoint revision: `98cf803d8586e8e8b3ec527885ac50c422d514ae`
- Data: deterministic COCO 2017 val subset, seed 42
- Train: 64 image-caption pairs (16 per class)
- Held-out probe: 80 pairs (20 per class)
- Train/probe ID overlap: 0
- GPU: NVIDIA A100-SXM4-40GB
- Precision: bfloat16
- Environment: Python 3.13.15, torch 2.11.0+cu128, transformers 4.57.6
- Source commit executed in Colab: `66dd6ee12b2e80ddd11ae5b5cd16b53c3b08efa1`
- Downloaded zip SHA-256: `5f88b8d04045a4d3271a14b30fca5df928a17709f4bbd82b6a50bff3b5e1187f`

Retrieval R@K is averaged over image-to-text and text-to-image retrieval. The
frozen ridge linear probe uses the fused image/text representation with a
deterministic stratified 75/25 probe split. Model weights and LoRA adapter
weights are intentionally not committed.

| Variant | Trainable | Mean R@1 | Mean R@5 | Mean R@10 | Probe | CKA vs Base | Mean cosine drift |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | 0 (0.0000%) | 0.95625 | 1.00000 | 1.00000 | 1.00000 | 1.000000 | 0.000000 |
| LoRA Q/V r=8 | 294,912 (0.1318%) | 0.93750 | 1.00000 | 1.00000 | 1.00000 | 0.999953 | 0.000300 |
| LoRA Q/V r=16 | 589,824 (0.2636%) | 0.94375 | 1.00000 | 1.00000 | 1.00000 | 0.999850 | 0.000862 |
| LoRA Q/V r=8 + mined hard negative | 294,912 (0.1318%) | 0.95000 | 1.00000 | 1.00000 | 1.00000 | 0.999956 | 0.000275 |

Both LoRA variants target 24 text-encoder self-attention Q/V linear modules.
They use rank-scaled alpha (16 for r=8; 32 for r=16), dropout 0.05, batch size
8, two epochs, 16 steps, learning rate 1e-4, and seed 42. The first normal run
is canonical; no seed, split, rank, or epoch was changed in response to the
measured regression.

The hard-negative extension uses the same seed, 64 train pairs, 80 held-out
pairs, rank, optimizer, and 16-step budget as ordinary r=8. Base embeddings
mine one nearest different-ID caption for every training image. Mining coverage
is 64/64, mean top-1 cosine similarity is 0.35994, and the 95th percentile is
0.42436. The downloaded extension zip has SHA-256
`4dfd23834f47f9ccd870e581ad203d8498616ff91302ed7b82a8711b20216107`.
Its adapter was validated locally but, like the other adapters, is not committed.

## Exact-checkpoint OOD extension

The saved adapters from the first canonical runs were reloaded and evaluated
without further training. Gaussian blur, Gaussian noise, JPEG compression,
low light, and centered occlusion are generated lazily at severity 1–3; no
corrupted image copies are committed. Each of the 15 conditions uses all 80
held-out IDs. Results are under `ood/`.

The OOD zip SHA-256 is
`708c8016341248afe864581d5513661e0c5e7a09af0e6e60e6bbee6226bb79d6`.
For all four variants, clean representations from the reloaded checkpoints
match the canonical snapshots exactly (maximum absolute difference 0.0).

## Research conclusion and next study

This first canonical run is best interpreted as a trade-off study, not as a
single-model winner. Base preserves the strongest clean and absolute OOD R@1.
LoRA improves relative robustness and calibration, while r=8 hard-negative
training recovers most of the ordinary r=8 clean-performance loss at the same
trainable-parameter budget. The very high CKA values (`>= 0.99985`) show that
these short PEFT runs make only small global changes to the measured fused
representation space.

Several metrics are already saturated in this small study: every variant has
R@5/R@10 = `1.0` and frozen linear-probe accuracy = `1.0`. The next canonical
study should therefore make the retrieval problem harder rather than merely
increase epochs. Priority changes are:

1. enlarge the held-out retrieval candidate pool;
2. increase the density/diversity of confusable mined negatives;
3. emphasize occlusion, which is the worst severity-3 OOD condition for every
   measured variant; and
4. repeat the main Base / r=8 / r=16 / r=8+HN comparison across multiple seeds
   before making statistical claims about calibration or robustness gains.

Until that harder multi-seed study exists, `base` remains the reference for
absolute retrieval quality rather than being replaced by a LoRA checkpoint.
