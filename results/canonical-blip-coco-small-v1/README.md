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

Both LoRA variants target 24 text-encoder self-attention Q/V linear modules.
They use rank-scaled alpha (16 for r=8; 32 for r=16), dropout 0.05, batch size
8, two epochs, 16 steps, learning rate 1e-4, and seed 42. The first normal run
is canonical; no seed, split, rank, or epoch was changed in response to the
measured regression.
