# Dataset — Local Prototyping Subset

Status: in use for the spatial-only baseline smoke test (Step 5).

## Source

**DFD (Deep Fake Detection) dataset** — Google/Jigsaw's original+manipulated
sequence release, part of the FaceForensics++ family.

- Kaggle mirror: `sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset`
- Full dataset: 3,432 `.mp4` files, ~24.2 GB (364 original / 3,068 manipulated) —
  too large for local prototyping, hence the selective slice below.

## Selection

25 real + 25 fake videos, selected via `random.seed(42)` from:
- `DFD_original sequences/` (real, 363-file pool)
- `DFD_manipulated_sequences/` (fake, 3,068-file pool)

Deterministic — rerunnable if the local copy is ever lost (see below).

- **Total size**: 405 MB (25 real: 235.1 MB / 25 fake: 188.8 MB, estimated
  pre-download; actual on-disk total 405 MB)
- **Clip length**: full-length acted scenes, **not trimmed to a fixed duration**
  — unlike DFDC/SDFVD, DFD clips vary from a few seconds to roughly a minute
  (e.g. `01__walking_and_outside_surprised.mp4` is ~45s at 1,085 frames/24fps).
  Frame-sampling in preprocessing uniformly samples N frames per video
  regardless of length, so this doesn't block the pipeline — just means more
  actual footage per file than originally scoped for DFDC.

## Where the data lives

- **Working copy (unzipped, gitignored)**: `data/raw/dfd_subset/` — 50 `.mp4`
  files. Regenerate from the frozen zip below if `data/raw/` is ever cleared.
- **Manifest (tracked in git)**: `data/splits/dfd_subset_manifest.csv` — label,
  file path, byte size for all 50 selected files. This is the reproducibility
  record; the selection itself is also regeneratable from the `seed=42` sample
  against the same Kaggle dataset listing.
- **Frozen copy (outside the repo, not tracked by git)**:
  `~/datasets/deepfake-detection-msaf/dfd_subset_50clips.zip` — 405 MB, contains
  the 50 videos plus a copy of the manifest CSV. Kept outside the repo tree so
  it's never at risk of being committed; this is the archival backup in case
  the Kaggle listing changes or the local `data/raw/` copy is deleted.

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
