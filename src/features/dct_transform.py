"""Block-wise 2D DCT (8x8, JPEG-style) frequency feature extraction.

Reads face crops from manifest_spatial_baseline.csv, writes log-magnitude
DCT coefficient maps (float16, compressed) to data/interim/frequency/, plus
a parallel manifest_frequency_baseline.csv.
"""
import csv
import time
from pathlib import Path

import numpy as np
from PIL import Image

BLOCK_SIZE = 8
MANIFEST_IN = Path("data/interim/manifest_spatial_baseline.csv")
OUT_DIR = Path("data/interim/frequency")
MANIFEST_OUT = Path("data/interim/manifest_frequency_baseline.csv")


def _dct_basis(n=BLOCK_SIZE):
    C = np.zeros((n, n), dtype=np.float32)
    for k in range(n):
        for i in range(n):
            if k == 0:
                C[k, i] = np.sqrt(1.0 / n)
            else:
                C[k, i] = np.sqrt(2.0 / n) * np.cos(np.pi / n * (i + 0.5) * k)
    return C


_C = _dct_basis()


def block_dct(gray_img):
    """gray_img: (H, W) float32, H/W divisible by BLOCK_SIZE.
    Returns log-magnitude block-DCT coefficient map, same (H, W) shape, float16."""
    h, w = gray_img.shape
    bh, bw = h // BLOCK_SIZE, w // BLOCK_SIZE
    blocks = (
        gray_img.reshape(bh, BLOCK_SIZE, bw, BLOCK_SIZE)
        .transpose(0, 2, 1, 3)
        .reshape(-1, BLOCK_SIZE, BLOCK_SIZE)
    )
    coeffs = _C[None] @ blocks @ _C.T[None]
    log_mag = np.log1p(np.abs(coeffs))
    out = (
        log_mag.reshape(bh, bw, BLOCK_SIZE, BLOCK_SIZE)
        .transpose(0, 2, 1, 3)
        .reshape(h, w)
    )
    return out.astype(np.float16)


def main():
    t_start = time.time()

    with open(MANIFEST_IN) as f:
        rows = list(csv.DictReader(f))

    out_rows = []
    for i, row in enumerate(rows):
        img = Image.open(row["frame_path"]).convert("L")
        gray = np.asarray(img, dtype=np.float32)
        dct_map = block_dct(gray)

        video_id = row["video_id"]
        frame_stem = Path(row["frame_path"]).stem
        out_dir = OUT_DIR / video_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{frame_stem}.npz"
        np.savez_compressed(out_path, dct=dct_map)

        out_rows.append([str(out_path).replace("\\", "/"), video_id, row["label"]])
        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{len(rows)}] processed")

    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frequency_path", "video_id", "label"])
        w.writerows(out_rows)

    elapsed = time.time() - t_start
    total_bytes = sum(p.stat().st_size for p in OUT_DIR.rglob("*.npz"))
    print(f"\nprocessed {len(out_rows)} frames in {elapsed:.1f}s")
    print(f"total disk size: {total_bytes/1e6:.1f} MB")


if __name__ == "__main__":
    main()
