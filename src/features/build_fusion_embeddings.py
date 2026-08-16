"""Combine the three frozen streams into one video-level embedding triple each:
  - spatial:  mean-pool the (T,512) spatial_embeddings sequence -> (512,)
  - frequency: mean-pool the (T,128) frequency_embeddings sequence -> (128,)
  - temporal: frozen temporal_baseline_smoke_test.pt's clip embedding -> (128,)

Reads data/interim/manifest_temporal_baseline.csv (video_id/label list),
writes:
  - data/interim/fusion_embeddings/{video_id}.npz  (spatial, frequency, temporal)
  - data/interim/manifest_fusion_baseline.csv       (video_id, embedding_path, label)
"""
import csv
import time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.temporal.bilstm import TemporalBiLSTM

TEMPORAL_MANIFEST = Path("data/interim/manifest_temporal_baseline.csv")
SPATIAL_EMB_DIR = Path("data/interim/spatial_embeddings")
FREQ_EMB_DIR = Path("data/interim/frequency_embeddings")
TEMPORAL_CHECKPOINT = Path("results/checkpoints/temporal_baseline_smoke_test/temporal_baseline_smoke_test.pt")
OUT_DIR = Path("data/interim/fusion_embeddings")
MANIFEST_OUT = Path("data/interim/manifest_fusion_baseline.csv")


def load_frozen_temporal():
    model = TemporalBiLSTM(input_size=512, hidden_size=64, num_classes=1)
    state = torch.load(TEMPORAL_CHECKPOINT, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def main():
    t_start = time.time()

    with open(TEMPORAL_MANIFEST) as f:
        rows = list(csv.DictReader(f))

    temporal_model = load_frozen_temporal()

    out_rows = []
    with torch.no_grad():
        for i, row in enumerate(rows):
            video_id = row["video_id"]
            label = row["label"]

            spatial_seq = np.load(SPATIAL_EMB_DIR / f"{video_id}.npy").astype(np.float32)  # (T,512)
            freq_seq = np.load(FREQ_EMB_DIR / f"{video_id}.npy").astype(np.float32)  # (T,128)

            spatial_pooled = spatial_seq.mean(axis=0)  # (512,)
            freq_pooled = freq_seq.mean(axis=0)  # (128,)

            seq_tensor = torch.from_numpy(spatial_seq).unsqueeze(0)  # (1,T,512)
            length = torch.tensor([spatial_seq.shape[0]])
            temporal_emb = temporal_model.embed(seq_tensor, length).squeeze(0).numpy()  # (128,)

            OUT_DIR.mkdir(parents=True, exist_ok=True)
            out_path = OUT_DIR / f"{video_id}.npz"
            np.savez(out_path, spatial=spatial_pooled, frequency=freq_pooled, temporal=temporal_emb)

            out_rows.append([video_id, str(out_path).replace("\\", "/"), label])
            if (i + 1) % 20 == 0:
                print(f"[{i+1}/{len(rows)}] processed")

    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "embedding_path", "label"])
        w.writerows(out_rows)

    elapsed = time.time() - t_start
    total_bytes = sum(p.stat().st_size for p in OUT_DIR.glob("*.npz"))
    print(f"\nprocessed {len(out_rows)} videos in {elapsed:.1f}s")
    print(f"total disk size: {total_bytes/1e6:.2f} MB")


if __name__ == "__main__":
    main()
