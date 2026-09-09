# Contribution Notes — Base Paper vs. This Work

Status: draft for review. Tracks real progress against the plan — update as work
actually happens, don't let this drift into aspirational copy.

Base paper facts below were pulled from the public abstract/methodology summary via
IEEE Xplore and ResearchGate (full text is paywalled — I do not have the PDF, so exact
numeric results are not verified). Anything not explicitly sourced is marked
**assumed, not verified**.

Sources:
- [Compression-Aware Hybrid Framework for Deep Fake Detection in Low-Quality Video — IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/11095666/) (DOI 10.1109/ACCESS.2025.3592358)
- [Same paper — ResearchGate](https://www.researchgate.net/publication/393985836_Compression-Aware_Hybrid_Framework_for_deep_fake_Detection_in_Low-Quality_Video)

## Inherited from base paper

**"Compression-Aware Hybrid Framework for Deep Fake Detection in Low-Quality Video"**
(IEEE Access, 2025)

- **General problem framing**: deepfake detection specifically under the compression
  regimes typical of social media re-upload, where compression artifacts degrade
  detector performance. This is the problem statement we inherit wholesale.
- **Frequency-domain feature extraction as a core idea**: the base paper uses
  **wavelet transforms** as a hand-crafted frequency-domain feature. We inherit the
  general principle — frequency-domain signal survives/reveals forgery traces that
  compression partially destroys in the spatial domain — and build our frequency
  stream on it (DCT and/or wavelet, see [architecture.md](architecture.md) §3).
- **Spatiotemporal feature extraction**: the base paper uses a **Conv3D-based
  spatiotemporal descriptor** — i.e. space and time are captured jointly by one 3D
  convolutional module, not as separate streams. We inherit the idea that temporal
  information matters for this problem, but *diverge* on how it's captured (see
  Novel Contribution 1 below).
- **Attention as a technique**: the base paper reports four evaluated configurations —
  RGB-with-attention, RGB-without, grayscale-with-attention, grayscale-without — i.e.
  attention is ablated as present/absent **within a single input pipeline**. We
  inherit "attention helps" as a starting hypothesis but apply it differently (see
  Novel Contribution 2).
- **Evaluation across compression tiers and across datasets**: the base paper
  evaluates on **FaceForensics++ (C23, C40)** and **Celeb-DF-v2 (C0, C40)**, including
  intra-compression, inter-compression, and cross-dataset settings. We inherit
  "report performance as a function of compression/quality degradation" as a required
  evaluation axis. Note this means cross-dataset evaluation itself is **not new** —
  see Novel Contribution 4, which had to be narrowed for this reason.
- **Classifier**: a "lightweight ResNet-inspired classifier" sits downstream of the
  hand-crafted features — **assumed, not verified**: exact layer count/width, since
  the full text wasn't accessible.
- **Design philosophy**: the base paper explicitly favors interpretability and
  computational efficiency over end-to-end deep learning (it's described as a
  hand-crafted-feature + lightweight-classifier hybrid, not an end-to-end trained
  network). This is a philosophical point where this work **diverges**, not inherits:
  we're proposing an end-to-end trainable multi-stream network. Worth stating clearly
  in the paper's related-work section as the key positioning difference, not just an
  implementation detail.

## Novel contributions of this work

### 1. Multi-scale feature learning (separate spatial / frequency / temporal streams)

- **Distinguishing factor**: the base paper combines a hand-crafted wavelet feature
  with a single joint Conv3D spatiotemporal descriptor, then classifies. This work
  instead uses **three independently learnable streams** (spatial CNN, frequency CNN,
  temporal Transformer/LSTM), each producing its own embedding before fusion — space
  and time are *not* fused inside one convolution, and the frequency feature is
  learned, not fixed. This is a genuine architectural split from the base paper, not
  just a bigger model.
- **Evidence that would demonstrate it worked**: ablation table — spatial-only vs.
  +frequency vs. +temporal vs. full three-stream — measured in accuracy/F1/AUC on a
  held-out set; per-stream embedding analysis (e.g. do frequency-stream activations
  separate real/fake even when spatial-stream activations don't, at high compression).
- **Status**: smoke-tested, verified end-to-end on real data (50-video DFD
  subset, 40/10 train/val split). Each of the three streams trains, produces
  finite loss, and its checkpoint saves/reloads correctly. This is pipeline
  correctness verification on a tiny, deliberately-overfit split — **not**
  production-trained or validated, and carries no claim about accuracy or
  generalization.

### 2. Attention-based fusion across streams

- **Distinguishing factor**: the base paper's attention is ablated on/off **within a
  single RGB or grayscale pipeline** (likely spatial or channel attention inside the
  classifier — assumed, not verified, since we don't have layer-level detail). This
  work's attention is **cross-stream**: a fusion module that learns how much to weight
  the spatial, frequency, and temporal embeddings *per sample*, instead of
  concatenating them with fixed weight. This is a different use of "attention" than
  the base paper's, not a bigger version of the same thing.
- **Evidence that would demonstrate it worked**: compare a fixed-concatenation fusion
  baseline against the learned attention fusion on the identical three-stream
  backbone (accuracy/F1/AUC delta, same data); visualize learned attention weights
  across compression levels to check the hypothesis that frequency-stream weight
  should increase as compression increases (this is the interpretability story that
  ties directly to the base paper's compression-aware framing).
- **Status**: smoke-tested, verified end-to-end on real data (50-video DFD
  subset, 40/10 train/val split) — both the softmax-gating attention fusion
  and a matched-capacity (~100K param) naive-concatenation baseline trained
  cleanly and their checkpoints save/reload. Not production-trained or
  validated. See result note immediately below.
- **Smoke-test result (2026-08-16)**: on this 50-video subset, attention and
  concat fusion both reached 100% train accuracy at the same epoch (2), but
  their train-loss trajectories diverged — concat converged faster through
  epochs 1–4 (e.g. epoch 3: concat 0.293 vs. attention 0.344), while attention
  ended lower by epoch 5 (0.110 vs. concat's 0.146). Val accuracy for both sat
  at 50–60%, statistically indistinguishable at only 10 val videos. **This is
  evidence the two fusion mechanisms behave differently from each other, not
  evidence attention is better** — it's one run, one seed, 40 training
  examples. Worth re-checking once there's a real dataset and multiple seeds
  behind the comparison; not a result to cite as-is.
- **First real-training result (250-clip set, see `docs/results.md`)**: on
  the held-out 50-video validation split, attention (84.0% accuracy, 0.8333
  F1, 0.8752 AUC) and concat (84.0% accuracy, 0.8333 F1, 0.8704 AUC) are
  again statistically indistinguishable — identical accuracy/F1, attention
  ahead by only 0.005 AUC. Same conclusion as the smoke test, now on real
  (if still small, n=50) data: **still no evidence attention fusion beats
  naive concatenation** — one run, one seed, 200 training videos. This
  needs multiple seeds and a larger validation set before it can support
  the paper's core claim either way.

### 3. Degradation-aware augmentation strategy

- **Distinguishing factor**: the base paper evaluates robustness by testing on
  **fixed, dataset-provided compression tiers** (C23/C40/C0 — discrete, pre-baked
  variants of FF++ and Celeb-DF-v2). This work instead applies **randomized,
  on-the-fly degradation during training** — JPEG re-compression, video re-encoding,
  Gaussian noise, downscaling, motion blur — so the model sees a continuum of
  degradation severity and *types the base paper's tiers don't cover at all* (noise,
  blur). This is a training-time methodology difference, not just a different test
  set.
- **Evidence that would demonstrate it worked**: a robustness curve — accuracy/F1/AUC
  plotted against degradation severity (compression CRF, noise σ, or measured
  PSNR/SSIM of the degraded input vs. clean) — comparing a model trained *with*
  degradation-aware augmentation against one trained without, evaluated at severities
  beyond what either model saw in fixed-tier training data.
- **Status**: not started. `src/features/` folder exists; no augmentation code.

### 4. Cross-dataset generalization testing — narrowed after checking the base paper

- **Honesty check**: the base paper already runs cross-dataset evaluation (FF++
  trained → Celeb-DF-v2 tested, and vice versa per the "inter-compression" and
  cross-dataset settings mentioned in its abstract). So "we test cross-dataset" is
  **not** a valid novelty claim on its own — I want to flag that plainly rather than
  let it stand as originally scoped.
- **What's actually different**: (a) this work trains on **DFDC**, a dataset the base
  paper doesn't use at all, then tests zero-shot on Celeb-DF-v2 — a different
  train/test pairing than the base paper's FF++↔Celeb-DF-v2 axis; (b) the
  generalization gap is measured for an **end-to-end learned multi-stream attention
  model**, vs. the base paper's hand-crafted-feature + lightweight-classifier
  approach — so the interesting question becomes whether learned fusion generalizes
  better or worse than hand-crafted features do, not just "does it generalize."
- **Suggestion**: consider also running FF++→Celeb-DF-v2 (matching the base paper's
  exact pairing) alongside DFDC→Celeb-DF-v2, once on GPU — that gives a
  directly comparable number against the base paper's reported cross-dataset result,
  which is a much stronger claim than a new pairing alone. Flagging this now so it's
  a decision, not a default.
- **Evidence that would demonstrate it worked**: accuracy/AUC drop from in-distribution
  (held-out DFDC) to out-of-distribution (Celeb-DF-v2) test, ideally set against the
  base paper's own reported cross-dataset drop (pending access to their actual
  numbers — currently unverified).
- **Status**: not started. No dataset beyond the local DFDC sample subset has been
  touched; Celeb-DF-v2 access process hasn't been started.

## Current implementation status (summary)

| Component | Status |
|---|---|
| Repo/folder scaffolding | done |
| `docs/architecture.md` | done (draft) |
| `docs/contribution-notes.md` | done (draft, this file) |
| Local dataset subset (50-video DFD, see `docs/dataset.md`) | done — note: plan originally scoped DFDC, actually used DFD after access friction, see `docs/dataset.md` |
| Face detection/cropping pipeline | done — smoke-tested, 97% detection success rate |
| Spatial stream | trained on 250-clip set (200/50 split, 10 epochs, CPU) — see `docs/results.md`. Directional result, not GPU-scale final training. |
| Frequency stream | trained on 250-clip set — weakest performer, converges slowly under augmentation, see caveats in `docs/results.md` |
| Temporal stream | trained on 250-clip set — strongest single-stream result so far, see `docs/results.md` |
| Attention fusion module | trained on 250-clip set (both attention and concat-baseline variants) — does not yet show attention beating concat by a meaningful margin, see `docs/results.md` |
| Degradation-aware augmentation | done — 5 degradation types, mild/medium/heavy tiers, verified via visual QA + PSNR/SSIM spread check (see `docs/degradation_qa` outputs), used in the 250-clip training run above |
| Cross-dataset (Celeb-DF-v2) eval | not started |

Prior rows said "smoke-tested" (pipeline-correctness only, on a 50-video
overfit-prone split, no accuracy claim). These now reflect the first real
training pass on the 250-video training-phase subset — still a small local
CPU run, not GPU-scale final training, but a genuine (if early and
directional) accuracy/F1/AUC result rather than a wiring check. See
`docs/results.md` for the full table and honest caveats about what these
numbers do and don't support yet. This table should keep being updated
honestly as each piece actually changes state — not marked "done" on intent.
