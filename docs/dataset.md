# Dataset — Local Prototyping and Training-Phase Subsets

Status: two DFD subsets in use — a 50-video smoke-test set (Step 5 onward) and
a 250-video set for the real training phase, sampled disjoint from each other
so the 50 can later serve as an additional held-out check.

## Source

**DFD (Deep Fake Detection) dataset** — Google/Jigsaw's original+manipulated
sequence release, part of the FaceForensics++ family.

- Kaggle mirror: `sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset`
- Full dataset: 3,432 `.mp4` files, ~24.2 GB (364 original / 3,068 manipulated) —
  too large for local prototyping, hence the selective slices below.
- **Clip length**: full-length acted scenes, **not trimmed to a fixed duration**
  — unlike DFDC/SDFVD, DFD clips vary from a few seconds to roughly a minute
  (e.g. `01__walking_and_outside_surprised.mp4` is ~45s at 1,085 frames/24fps).
  Frame-sampling in preprocessing uniformly samples N frames per video
  regardless of length, so this doesn't block the pipeline — just means more
  actual footage per file than originally scoped for DFDC.

## Subset 1: smoke-test set (50 videos)

25 real + 25 fake videos, selected via `random.seed(42)` from:
- `DFD_original sequences/` (real, 363-file pool)
- `DFD_manipulated_sequences/` (fake, 3,068-file pool)

Deterministic — rerunnable if the local copy is ever lost.

- **Total size**: 405 MB (25 real: 235.1 MB / 25 fake: 188.8 MB, estimated
  pre-download; actual on-disk total 405 MB)
- **Working copy (unzipped, gitignored)**: `data/raw/dfd_subset/` — 50 `.mp4`
  files. Regenerate from the frozen zip below if `data/raw/` is ever cleared.
- **Manifest (tracked in git)**: `data/splits/dfd_subset_manifest.csv` — label,
  file path, byte size for all 50 selected files.
- **Frozen copy (outside the repo, not tracked by git)**:
  `~/datasets/deepfake-detection-msaf/dfd_subset_50clips.zip` — 405 MB, contains
  the 50 videos plus a copy of the manifest CSV.

## Subset 2: training-phase set (250 videos)

125 real + 125 fake videos, selected via `random.Random(7).sample(...)` (real)
and `random.Random(8).sample(...)` (fake) from the same real/fake pools as
Subset 1, **after explicitly excluding the 50 files already used in Subset 1**
— zero overlap, confirmed by set-intersection check at selection time. This is
deliberate: Subset 1's 50 videos can later serve as an additional held-out
check that was never seen during training-phase work, rather than silently
overlapping with it.

- **Total size**: target was capped at 2 GB; landed at **1,844.9 MB (1.8449 GB)**
  computed pre-download from actual per-file sizes (125 real: 954.4 MB / 125
  fake: 890.4 MB) — 155 MB margin under the cap. Actual on-disk total after
  download: 1.8 GB, zip size 1.843 GB (1,843,396,174 bytes).
- **Diversity check**: spans all 28 real subject IDs and 27/28 fake
  source-subject IDs in DFD — not accidentally clustered on a few actors.
- **Working copy (unzipped, gitignored)**: `data/raw/dfd_subset_250/` — 250
  `.mp4` files.
- **Manifest (tracked in git)**: `data/splits/dfd_subset_250_manifest.csv` —
  label, file path, byte size for all 250 selected files.
- **Frozen copy (outside the repo, not tracked by git)**:
  `~/datasets/deepfake-detection-msaf/dfd_subset_250clips.zip` — 1.843 GB,
  contains the 250 videos plus a copy of the manifest CSV. This is a
  **separate zip alongside** `dfd_subset_50clips.zip`, not a replacement —
  the original 50-clip zip is untouched.

Both frozen zips live outside the repo tree so neither is ever at risk of
being committed; they're the archival backup in case the Kaggle listing
changes or the local `data/raw/` copies are deleted.

## Derived artifacts with a checkpoint dependency

`data/interim/spatial_embeddings/{video_id}.npy` (used by the temporal stream)
is **not derived from the raw dataset directly** — it's the output of running
`results/checkpoints/spatial_baseline_smoke_test/spatial_baseline_smoke_test.pt`
(frozen) over the face crops in `data/interim/frames/`. If that spatial
checkpoint is ever retrained (different data, different architecture, even
just a different random seed), these cached embeddings go stale and must be
regenerated via `src/features/extract_spatial_embeddings.py` before the
temporal stream is trained again. Nothing currently checks this
automatically — worth remembering as more streams start depending on each
other's checkpoints.

## Confirmed

`data/raw/` is covered by `.gitignore` (line 2) — none of the video files or
this zip's contents will be accidentally committed.
