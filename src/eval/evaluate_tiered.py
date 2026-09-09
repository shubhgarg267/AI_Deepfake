"""Per-compression-tier evaluation of the fusion-attention model on the
250-clip val split: applies a FIXED (deterministic, no random sampling)
degradation pass per tier (clean/mild/medium/heavy) to the raw val frames,
recomputes spatial/frequency/temporal embeddings from the already-trained
frozen checkpoints, and evaluates fusion-attention on each tier.

Also captures the per-sample (w_spatial, w_freq, w_temporal) gating weights
grouped by tier, for the attention-weight visualization (item 4).
"""
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.degradation_augmentation import degrade_fixed_tier
from src.features.dct_transform import block_dct
from src.models.frequency.small_cnn import FrequencyCNN
from src.models.temporal.bilstm import TemporalBiLSTM
from src.models.fusion.attention_fusion import AttentionFusion
from src.eval.metrics import accuracy, f1, auc, eer

MANIFEST = Path("data/interim/manifest_spatial_250.csv")
VAL_SPLIT = Path("data/splits/spatial_250_val.csv")
TIERS = ["clean", "mild", "medium", "heavy"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
spatial_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def load_spatial_encoder():
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    state = torch.load(
        "results/checkpoints/spatial_250_train/spatial_baseline_smoke_test.pt", map_location="cpu"
    )
    model.load_state_dict(state)
    model.fc = nn.Identity()
    model.eval()
    return model


def load_frequency_encoder():
    model = FrequencyCNN(num_classes=1)
    state = torch.load(
        "results/checkpoints/frequency_250_train/frequency_baseline_smoke_test.pt", map_location="cpu"
    )
    model.load_state_dict(state)
    model.classifier = nn.Identity()
    model.eval()
    return model


def load_temporal():
    model = TemporalBiLSTM(input_size=512, hidden_size=64, num_classes=1)
    state = torch.load(
        "results/checkpoints/temporal_250_train/temporal_baseline_smoke_test.pt", map_location="cpu"
    )
    model.load_state_dict(state)
    model.eval()
    return model


def load_fusion_attention():
    model = AttentionFusion()
    state = torch.load(
        "results/checkpoints/fusion_attention_250_train/fusion_attention_baseline_smoke_test.pt",
        map_location="cpu",
    )
    model.load_state_dict(state)
    model.eval()
    return model


def frequency_input(degraded_img):
    gray = np.asarray(degraded_img.convert("L"), dtype=np.float32)
    dct_map = block_dct(gray).astype(np.float32)
    dct_map = (dct_map - dct_map.mean()) / (dct_map.std() + 1e-6)
    return torch.from_numpy(dct_map).unsqueeze(0)  # (1, H, W)


def main():
    t_start = time.time()

    with open(VAL_SPLIT) as f:
        val_ids = {r["video_id"] for r in csv.DictReader(f)}
    with open(MANIFEST) as f:
        rows = [r for r in csv.DictReader(f) if r["video_id"] in val_ids]

    by_video = defaultdict(list)
    for r in rows:
        by_video[r["video_id"]].append(r)
    for vid in by_video:
        by_video[vid].sort(key=lambda r: r["frame_path"])

    spatial_enc = load_spatial_encoder()
    freq_enc = load_frequency_encoder()
    temporal_model = load_temporal()
    fusion_model = load_fusion_attention()

    tier_results = {}
    weight_log_rows = []

    for tier in TIERS:
        labels, probs = [], []
        with torch.no_grad():
            for video_id, frame_rows in by_video.items():
                label = 1.0 if frame_rows[0]["label"] == "FAKE" else 0.0

                degraded_imgs = []
                for r in frame_rows:
                    img = Image.open(r["frame_path"]).convert("RGB")
                    degraded_imgs.append(degrade_fixed_tier(img, tier))

                spatial_batch = torch.stack([spatial_transform(im) for im in degraded_imgs])
                spatial_seq = spatial_enc(spatial_batch)  # (T, 512)

                freq_batch = torch.stack([frequency_input(im) for im in degraded_imgs])  # (T,1,224,224)
                freq_seq = freq_enc(freq_batch)  # (T, 128)

                spatial_pooled = spatial_seq.mean(dim=0, keepdim=True)  # (1,512)
                freq_pooled = freq_seq.mean(dim=0, keepdim=True)  # (1,128)

                seq_len = torch.tensor([spatial_seq.shape[0]])
                temporal_emb = temporal_model.embed(spatial_seq.unsqueeze(0), seq_len)  # (1,128)

                logit, weights = fusion_model(spatial_pooled, freq_pooled, temporal_emb, return_weights=True)
                prob = torch.sigmoid(logit).item()
                w = weights.squeeze(0).tolist()  # [w_spatial, w_freq, w_temporal]

                labels.append(label)
                probs.append(prob)
                weight_log_rows.append([video_id, tier, label, w[0], w[1], w[2]])

        tier_results[tier] = {
            "accuracy": accuracy(labels, probs),
            "f1": f1(labels, probs),
            "auc": auc(labels, probs),
            "eer": eer(labels, probs),
            "n": len(labels),
        }
        print(f"[{tier}] n={len(labels)} acc={tier_results[tier]['accuracy']:.4f} "
              f"f1={tier_results[tier]['f1']:.4f} auc={tier_results[tier]['auc']:.4f} "
              f"eer={tier_results[tier]['eer']:.4f}")

    weight_log_path = Path("data/interim/fusion_attention_weights_by_tier.csv")
    with open(weight_log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "tier", "label", "w_spatial", "w_frequency", "w_temporal"])
        w.writerows(weight_log_rows)

    print(f"\nwrote per-video attention weights to {weight_log_path}")

    print("\nmean attention weights per tier:")
    for tier in TIERS:
        rows_t = [r for r in weight_log_rows if r[1] == tier]
        w_s = np.mean([r[3] for r in rows_t])
        w_f = np.mean([r[4] for r in rows_t])
        w_t = np.mean([r[5] for r in rows_t])
        print(f"  {tier:8s}: w_spatial={w_s:.4f}  w_frequency={w_f:.4f}  w_temporal={w_t:.4f}")

    elapsed = time.time() - t_start
    print(f"\ntotal elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
