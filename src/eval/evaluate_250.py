"""Compute accuracy/F1/AUC on the 250-clip set's held-out 50-video val split
for all trained components. Spatial/frequency are evaluated at frame-level
(their natural training granularity); temporal/fusion at video-level (their
natural granularity, one prediction per clip).
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.dataset import (
    SpatialFrameDataset,
    FrequencyFrameDataset,
    TemporalSequenceDataset,
    temporal_collate_fn,
    FusionEmbeddingDataset,
)
from src.models.frequency.small_cnn import FrequencyCNN
from src.models.temporal.bilstm import TemporalBiLSTM
from src.models.fusion.attention_fusion import AttentionFusion
from src.models.fusion.concat_fusion import ConcatFusion
from src.eval.metrics import compute_all

VAL_SPLIT = "data/splits/spatial_250_val.csv"
DEVICE = torch.device("cpu")


def eval_spatial():
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    state = torch.load(
        "results/checkpoints/spatial_250_train/spatial_baseline_smoke_test.pt", map_location="cpu"
    )
    model.load_state_dict(state)
    model.eval()

    ds = SpatialFrameDataset("data/interim/manifest_spatial_250.csv", VAL_SPLIT, 224)
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    labels, probs = [], []
    with torch.no_grad():
        for images, y in loader:
            logits = model(images).squeeze(1)
            probs.extend(torch.sigmoid(logits).tolist())
            labels.extend(y.tolist())
    return compute_all(labels, probs), len(labels)


def eval_frequency(checkpoint="results/checkpoints/frequency_250_train/frequency_baseline_smoke_test.pt"):
    model = FrequencyCNN(num_classes=1)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    ds = FrequencyFrameDataset("data/interim/manifest_frequency_250.csv", VAL_SPLIT)
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    labels, probs = [], []
    with torch.no_grad():
        for images, y in loader:
            logits = model(images).squeeze(1)
            probs.extend(torch.sigmoid(logits).tolist())
            labels.extend(y.tolist())
    return compute_all(labels, probs), len(labels)


def eval_temporal():
    model = TemporalBiLSTM(input_size=512, hidden_size=64, num_classes=1)
    state = torch.load(
        "results/checkpoints/temporal_250_train/temporal_baseline_smoke_test.pt", map_location="cpu"
    )
    model.load_state_dict(state)
    model.eval()

    ds = TemporalSequenceDataset("data/interim/manifest_temporal_250.csv", VAL_SPLIT)
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=temporal_collate_fn)

    labels, probs = [], []
    with torch.no_grad():
        for padded, lengths, y in loader:
            logits = model(padded, lengths).squeeze(1)
            probs.extend(torch.sigmoid(logits).tolist())
            labels.extend(y.tolist())
    return compute_all(labels, probs), len(labels)


def eval_fusion(fusion_type, checkpoint=None, manifest="data/interim/manifest_fusion_250.csv", return_raw=False):
    if fusion_type == "attention":
        model = AttentionFusion()
        ckpt = checkpoint or "results/checkpoints/fusion_attention_250_train/fusion_attention_baseline_smoke_test.pt"
    else:
        model = ConcatFusion()
        ckpt = checkpoint or "results/checkpoints/fusion_concat_250_train/fusion_concat_baseline_smoke_test.pt"
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    ds = FusionEmbeddingDataset(manifest, VAL_SPLIT)
    loader = DataLoader(ds, batch_size=16, shuffle=False)

    labels, probs = [], []
    with torch.no_grad():
        for spatial, freq, temporal, y in loader:
            logits = model(spatial, freq, temporal).squeeze(1)
            probs.extend(torch.sigmoid(logits).tolist())
            labels.extend(y.tolist())
    if return_raw:
        return labels, probs
    return compute_all(labels, probs), len(labels)


if __name__ == "__main__":
    results = {}
    results["spatial (frame-level)"] = eval_spatial()
    results["frequency (frame-level)"] = eval_frequency()
    results["temporal (video-level)"] = eval_temporal()
    results["fusion-attention (video-level)"] = eval_fusion("attention")
    results["fusion-concat (video-level)"] = eval_fusion("concat")

    print(f"{'component':32s} {'n':>6s} {'accuracy':>10s} {'f1':>10s} {'auc':>10s}")
    for name, (metrics, n) in results.items():
        print(f"{name:32s} {n:6d} {metrics['accuracy']:10.4f} {metrics['f1']:10.4f} {metrics['auc']:10.4f}")
