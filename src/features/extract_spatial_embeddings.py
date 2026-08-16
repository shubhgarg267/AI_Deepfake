"""Run the frozen spatial_baseline_smoke_test checkpoint over all face crops
to produce per-video sequences of 512-d embeddings for the temporal stream.

Reads data/interim/manifest_spatial_baseline.csv, writes:
  - data/interim/spatial_embeddings/{video_id}.npy   (T, 512) float32
  - data/interim/manifest_temporal_baseline.csv       (video_id, embedding_path, label, num_frames)

Depends on results/checkpoints/spatial_baseline_smoke_test/spatial_baseline_smoke_test.pt
already existing (see docs/dataset.md for the staleness note).
"""
import csv
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

MANIFEST_IN = Path("data/interim/manifest_spatial_baseline.csv")
CHECKPOINT = Path("results/checkpoints/spatial_baseline_smoke_test/spatial_baseline_smoke_test.pt")
OUT_DIR = Path("data/interim/spatial_embeddings")
MANIFEST_OUT = Path("data/interim/manifest_temporal_baseline.csv")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def load_frozen_encoder():
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    state = torch.load(CHECKPOINT, map_location="cpu")
    model.load_state_dict(state)
    model.fc = nn.Identity()  # drop the classifier head, keep 512-d pooled features
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def main():
    t_start = time.time()

    with open(MANIFEST_IN) as f:
        rows = list(csv.DictReader(f))

    by_video = defaultdict(list)
    for r in rows:
        by_video[r["video_id"]].append(r)
    for vid in by_video:
        by_video[vid].sort(key=lambda r: r["frame_path"])  # zero-padded frame idx -> correct temporal order

    model = load_frozen_encoder()

    out_rows = []
    with torch.no_grad():
        for i, (video_id, frame_rows) in enumerate(by_video.items()):
            imgs = []
            for r in frame_rows:
                img = Image.open(r["frame_path"]).convert("RGB")
                imgs.append(transform(img))
            batch = torch.stack(imgs)  # (T, 3, 224, 224)
            embeddings = model(batch).numpy().astype(np.float32)  # (T, 512)

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
