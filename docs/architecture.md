# Architecture — Multi-Scale Attention-Based Deepfake Detection

Status: draft for review. Nothing below is implemented yet.

## 1. Overview

Four components, trained end-to-end (streams can also be pretrained/frozen individually for ablation):

```
video clip
   │
   ├─► face detect + crop (per frame) ──► [Spatial Stream]   ──► spatial embedding
   │                                     │
   ├─► DCT / wavelet transform (per frame) ─► [Frequency Stream] ──► frequency embedding
   │
   └─► frame embedding sequence ─────────► [Temporal Stream]  ──► temporal embedding
                                                                    │
                                        [Attention Fusion Module] ◄─┘
                                                 │
                                        real / fake (binary)
```

Spatial and frequency streams operate per-frame; the temporal stream operates over the
sequence of (fused or spatial) frame embeddings. All three feed the fusion module, which
outputs the final prediction.

## 2. Spatial stream

- **Input**: face-cropped RGB frame, fixed size (default **224×224**, matches standard
  CNN backbones).
- **Backbone (candidates, to decide during baseline)**: EfficientNet-B0, XceptionNet, or
  a small ResNet variant. Start with EfficientNet-B0 for the CPU/MPS smoke test —
  cheapest to run locally; swap for a heavier backbone once on GPU.
- **What it learns**: blending artifacts at face boundaries, texture/skin
  inconsistency, GAN fingerprint patterns.
- **Output**: a fixed-length embedding per frame (e.g. 1280-d for EfficientNet-B0
  pre-pool, or pooled to a smaller vector).
- **Lives in**: `src/models/spatial/`

## 3. Frequency stream

- **Input**: same cropped face region, transformed via 2D DCT (block-wise, e.g. 8×8 as
  used in JPEG) or a wavelet transform (e.g. Haar/DB via `pywt`).
- **What it learns**: compression re-encoding artifacts and high-frequency forgery
  traces that survive video compression — this is the direct extension point over the
  base paper's compression-aware features.
- **Architecture (candidate)**: a small CNN over the DCT/wavelet coefficient map
  (structurally similar to the spatial CNN but shallower, since the input is already a
  transformed low-dimensional signal).
- **Output**: a fixed-length embedding per frame, same dimensionality convention as the
  spatial stream so fusion can align them.
- **Lives in**: `src/models/frequency/`

## 4. Temporal stream

- **Input**: sequence of per-frame embeddings (spatial, or spatial+frequency
  concatenated) across a clip window — default window **16 frames**, uniformly sampled.
- **Architecture (candidate)**: a small Transformer encoder (2–4 layers) or a
  Bi-LSTM as a cheaper local-dev fallback. Start with Bi-LSTM for the CPU/MPS smoke
  test; move to Transformer once on GPU where attention over 16–32 tokens is cheap.
- **What it learns**: unnatural motion, flicker, or frame-to-frame inconsistency that a
  single-frame model can't see.
- **Output**: one embedding per clip (not per frame).
- **Lives in**: `src/models/temporal/`

## 5. Attention-based fusion module

- **Input**: spatial embedding(s), frequency embedding(s), temporal embedding — not
  naively concatenated.
- **Mechanism (candidate)**: learned attention weights over the three streams (and
  optionally over frames within a stream) so the model can down-weight a stream that's
  uninformative for a given clip (e.g. rely more on frequency when spatial artifacts are
  compression-smoothed away).
- **Output**: single fused embedding → linear layer → sigmoid → real/fake probability.
- **Lives in**: `src/models/fusion/`

## 6. Degradation-aware augmentation (training-time, not a stream)

Applied on the fly during training, not precomputed/stored:

- JPEG re-compression at random quality levels
- Video re-encoding (H.264 at varying CRF)
- Gaussian noise
- Downscaling (and upscaling back, to simulate low-res source video)
- Motion blur

Lives in `src/features/` (augmentation pipeline) and is invoked from `src/train/`.

## 7. Data flow through the repo

```
data/raw/          → original downloaded videos (untouched)
data/interim/       → face-cropped frames (+ optionally cached DCT/wavelet maps)
data/processed/     → final tensors / manifests ready for the dataloader
data/splits/        → train/val/test video-ID lists (tracked in git; data itself is not)
```

Frequency-domain features are cheap enough to compute **on the fly** in the dataloader
rather than precomputing and caching — see footprint discussion below. We'll only start
caching them if profiling shows the transform is a training bottleneck.

## 8. Expected disk footprint in `data/interim/`

Estimates for the **DFDC sample subset** (20 real + 20 fake, 5–10s clips):

| Item | Per video (est.) | 40 videos (est.) |
|---|---|---|
| Raw source `.mp4` (`data/raw/`) | ~2–8 MB (sample-set videos are already fairly compressed/short) | well under the 3 GB budget — likely 100–300 MB total |
| Face-cropped frames, 16 frames × 224×224 JPEG (`data/interim/`) | ~250–350 KB | ~10–14 MB |
| Cached frequency maps, *if* precomputed (16 × compressed DCT block map) | ~500 KB–1 MB | ~20–40 MB |
| **Total `data/interim/` (frames + optional freq cache)** | **~0.3–1.3 MB** | **~30–55 MB** |

Takeaway: `data/interim/` is negligible next to `data/raw/` for this subset size — the
3 GB budget is effectively bounded by the raw video downloads, not by anything we
derive from them. These are estimates; we'll confirm actual bytes once the sample set
is downloaded and adjust this table if real numbers diverge meaningfully.

## 9. Open decisions (flag if you want a different default)

- Spatial backbone: EfficientNet-B0 (local) → swap later, vs. committing to Xception now
- Temporal module: Bi-LSTM (local) vs. Transformer from the start
- Frequency transform: block DCT vs. wavelet (DCT is simpler to implement first and
  matches the base paper's compression framing more directly)
- Frame window size: 16 frames default — fine for smoke test, likely increases for real
  training
