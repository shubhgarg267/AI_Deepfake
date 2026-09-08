"""Smoke test for degradation_augmentation.py:
  - 10 sample frames x (5 degradation types x 3 tiers + clean) = 160 variants
  - saves a labeled before/after grid per frame to results/figures/degradation_preview/
  - logs PSNR/SSIM for every variant to data/interim/degradation_log.csv
  - reports the PSNR/SSIM spread across tiers

Uses the midpoint of each tier's parameter range (not random sampling) so the
grid is deterministic and inspectable.
"""
import csv
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.degradation_augmentation import (
    TIER_RANGES,
    DEGRADATION_TYPES,
    apply_jpeg,
    apply_h264,
    apply_gaussian_noise,
    apply_downscale_upscale,
    apply_motion_blur,
    compute_psnr,
    compute_ssim,
)

MANIFEST = Path("data/interim/manifest_spatial_baseline.csv")
OUT_DIR = Path("results/figures/degradation_preview")
LOG_OUT = Path("data/interim/degradation_log.csv")
N_FRAMES = 10
CELL = 224
LABEL_H = 18


def midpoint(rng):
    return (rng[0] + rng[1]) / 2


def apply_single(img, dtype, tier):
    p = TIER_RANGES[tier]
    if dtype == "jpeg":
        return apply_jpeg(img, int(round(midpoint(p["jpeg_quality"]))))
    if dtype == "h264":
        return apply_h264(img, int(round(midpoint(p["h264_crf"]))))
    if dtype == "noise":
        return apply_gaussian_noise(img, midpoint(p["noise_sigma"]))
    if dtype == "downscale":
        return apply_downscale_upscale(img, midpoint(p["downscale_factor"]))
    if dtype == "blur":
        k = int(round(midpoint(p["blur_kernel"])))
        k = k if k % 2 == 1 else k + 1
        return apply_motion_blur(img, k)
    raise ValueError(dtype)


def label_cell(img, text):
    canvas = Image.new("RGB", (CELL, CELL + LABEL_H), "white")
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((2, CELL + 2), text, fill="black")
    return canvas


def main():
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(MANIFEST) as f:
        rows = list(csv.DictReader(f))
    seen_videos = []
    frame_rows = []
    for r in rows:
        if r["video_id"] not in seen_videos:
            seen_videos.append(r["video_id"])
            frame_rows.append(r)
        if len(frame_rows) == N_FRAMES:
            break

    log_rows = []
    tiers = ["mild", "medium", "heavy"]

    for fi, row in enumerate(frame_rows):
        clean_img = Image.open(row["frame_path"]).convert("RGB")
        clean_arr = np.asarray(clean_img)

        n_cols = len(DEGRADATION_TYPES) + 1  # +1 for clean reference column
        n_rows = len(tiers)
        grid = Image.new("RGB", (n_cols * CELL, n_rows * (CELL + LABEL_H)), "white")

        clean_cell = label_cell(clean_img, "clean")
        for r_idx in range(n_rows):
            grid.paste(clean_cell, (0, r_idx * (CELL + LABEL_H)))

        log_rows.append(
            [row["video_id"], row["frame_path"], "clean", "", float("inf"), 1.0]
        )

        for r_idx, tier in enumerate(tiers):
            for c_idx, dtype in enumerate(DEGRADATION_TYPES, start=1):
                degraded = apply_single(clean_img, dtype, tier)
                degraded_arr = np.asarray(degraded.resize((CELL, CELL)))
                psnr = compute_psnr(clean_arr, degraded_arr)
                ssim = compute_ssim(clean_arr, degraded_arr)

                cell = label_cell(degraded.resize((CELL, CELL)), f"{dtype}/{tier}")
                grid.paste(cell, (c_idx * CELL, r_idx * (CELL + LABEL_H)))

                log_rows.append(
                    [row["video_id"], row["frame_path"], tier, dtype, psnr, ssim]
                )

        out_path = OUT_DIR / f"grid_{fi:02d}_{row['video_id']}.png"
        grid.save(out_path)
        print(f"[{fi+1}/{N_FRAMES}] saved {out_path}")

    with open(LOG_OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "frame_path", "tier", "degradation_type", "psnr", "ssim"])
        w.writerows(log_rows)

    elapsed = time.time() - t_start
    print(f"\nprocessed {len(frame_rows)} frames x 16 variants in {elapsed:.1f}s")

    print("\nPSNR/SSIM spread by tier (mean across types x frames):")
    for tier in tiers:
        tier_rows = [r for r in log_rows if r[2] == tier]
        psnrs = [r[4] for r in tier_rows]
        ssims = [r[5] for r in tier_rows]
        print(
            f"  {tier}: PSNR={sum(psnrs)/len(psnrs):.2f} dB, "
            f"SSIM={sum(ssims)/len(ssims):.4f} (n={len(tier_rows)})"
        )


if __name__ == "__main__":
    main()
