"""Run a frozen frequency checkpoint over all DCT maps to produce per-video
sequences of 128-d embeddings (mirrors extract_spatial_embeddings.py).

Defaults (smoke-test subset): reads data/interim/manifest_frequency_baseline.csv, writes:
  - data/interim/frequency_embeddings/{video_id}.npy  (T, 128) float32
  - data/interim/manifest_frequency_embeddings.csv     (video_id, embedding_path, label, num_frames)

Pass --manifest-in/--checkpoint/--out-dir/--manifest-out for a different subset/checkpoint.
"""
import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.frequency.small_cnn import FrequencyCNN


def load_frozen_encoder(checkpoint):
    model = FrequencyCNN(num_classes=1)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.classifier = torch.nn.Identity()  # drop the classifier head, keep 128-d features
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-in", default="data/interim/manifest_frequency_baseline.csv")
    parser.add_argument(
        "--checkpoint",
        default="results/checkpoints/frequency_baseline_smoke_test/frequency_baseline_smoke_test.pt",
    )
    parser.add_argument("--out-dir", default="data/interim/frequency_embeddings")
    parser.add_argument("--manifest-out", default="data/interim/manifest_frequency_embeddings.csv")
    args = parser.parse_args()

    MANIFEST_IN = Path(args.manifest_in)
    OUT_DIR = Path(args.out_dir)
    MANIFEST_OUT = Path(args.manifest_out)

    t_start = time.time()

    with open(MANIFEST_IN) as f:
        rows = list(csv.DictReader(f))

    by_video = defaultdict(list)
    for r in rows:
        by_video[r["video_id"]].append(r)
    for vid in by_video:
        by_video[vid].sort(key=lambda r: r["frequency_path"])  # zero-padded frame idx -> correct temporal order

    model = load_frozen_encoder(args.checkpoint)

    out_rows = []
    with torch.no_grad():
        for i, (video_id, frame_rows) in enumerate(by_video.items()):
            maps = []
            for r in frame_rows:
                data = np.load(r["frequency_path"])
                maps.append(data["dct"].astype(np.float32))
            arr = np.stack(maps)  # (T, 224, 224)
            arr = (arr - arr.mean(axis=(1, 2), keepdims=True)) / (arr.std(axis=(1, 2), keepdims=True) + 1e-6)
            batch = torch.from_numpy(arr).unsqueeze(1)  # (T, 1, 224, 224)
            embeddings = model(batch).numpy().astype(np.float32)  # (T, 128)

            OUT_DIR.mkdir(parents=True, exist_ok=True)
            out_path = OUT_DIR / f"{video_id}.npy"
            np.save(out_path, embeddings)

            label = frame_rows[0]["label"]
            out_rows.append([video_id, str(out_path).replace("\\", "/"), label, len(frame_rows)])
            if (i + 1) % 20 == 0:
                print(f"[{i+1}/{len(by_video)}] processed")

    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "embedding_path", "label", "num_frames"])
        w.writerows(out_rows)

    elapsed = time.time() - t_start
    total_bytes = sum(p.stat().st_size for p in OUT_DIR.glob("*.npy"))
    print(f"\nprocessed {len(out_rows)} videos in {elapsed:.1f}s")
    print(f"total disk size: {total_bytes/1e6:.2f} MB")


if __name__ == "__main__":
    main()
