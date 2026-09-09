# Results — First Real Training Pass (250-clip DFD subset)

Status: first non-smoke-test training run. Still a small local CPU run, not
the GPU-scale final training — treat these numbers as directional, not final.

## Setup

- **Dataset**: 250-video DFD training-phase subset (`docs/dataset.md`
  Subset 2), disjoint from the 50-video smoke-test set.
- **Split**: 200 train / 50 val, video-level, class-balanced (100/100 real+fake
  train, 25/25 val), `seed=100`.
- **Spatial/frequency**: 10 epochs, batch 16, Adam lr=1e-4, degradation-aware
  augmentation on (train split only; val stays clean).
- **Temporal**: 10 epochs, batch 16, built on spatial embeddings re-extracted
  from this run's retrained spatial checkpoint (not the smoke-test one).
- **Fusion**: 10 epochs, batch 16, both the softmax-gating attention variant
  and the matched-capacity naive-concat baseline, built on this run's
  retrained spatial/frequency/temporal checkpoints.
- **Total wall-clock, restart to final eval**: 1h 26m 46s (CPU only).

## Results (held-out 50-video validation split)

Spatial/frequency are evaluated at frame-level (their native training
granularity, n=483 val frames); temporal/fusion at video-level (n=50 val
videos, one prediction per clip).

| Component | n (val) | Accuracy | F1 | AUC |
|---|---|---|---|---|
| Spatial (frame-level) | 483 | 0.7702 | 0.7702 | 0.8075 |
| Frequency (frame-level) | 483 | 0.6315 | 0.6993 | 0.7526 |
| Temporal (video-level) | 50 | 0.8600 | 0.8511 | 0.8640 |
| Fusion — attention (video-level) | 50 | 0.8400 | 0.8333 | 0.8752 |
| Fusion — concat (video-level) | 50 | 0.8400 | 0.8333 | 0.8704 |

Metrics computed via `src/eval/metrics.py` (hand-rolled accuracy/F1/AUC,
verified to exact match against scikit-learn in an isolated environment,
including a tie-heavy AUC case).

## Honest caveats — do not over-read these numbers

- **Frequency is the weakest stream** (63.2% accuracy) and converged slowly
  during training (train acc 53.9%→60.5% over 10 epochs, well below the
  un-augmented smoke test's 72%). Augmentation appears to hit this
  from-scratch, no-pretraining CNN harder than the ImageNet-pretrained
  spatial stream. Worth investigating before leaning on frequency-stream
  results in the paper's ablations — e.g. whether it needs more epochs, a
  lower augmentation-probability schedule, or architectural changes.
- **Fusion does not clearly beat temporal alone here.** Temporal's solo AUC
  (0.864) sits close to both fusion variants (0.875 attention, 0.870
  concat); attention and concat are identical on accuracy/F1 (84.0%/0.833
  each), with attention only ahead by 0.005 AUC. At n=50 validation videos
  this gap is not a meaningful difference — same caveat as the smoke test's
  attention-vs-concat comparison, just now on real data instead of a
  10-video toy split. Not evidence attention fusion is the better
  mechanism yet.
- **n=50 validation videos is small** for any of these numbers to be
  considered stable — expect meaningful movement with a different
  train/val split or seed.
- This is still a **local CPU run on a 250-video subset**, not the GPU-scale
  training against the full acquired data the project plan anticipates.
