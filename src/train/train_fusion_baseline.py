import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.dataset import FusionEmbeddingDataset
from src.models.fusion.attention_fusion import AttentionFusion
from src.models.fusion.concat_fusion import ConcatFusion


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(cfg):
    m = cfg["model"]
    if m["fusion_type"] == "attention":
        return AttentionFusion(
            spatial_dim=m["spatial_dim"],
            freq_dim=m["freq_dim"],
            temporal_dim=m["temporal_dim"],
            proj_dim=m["proj_dim"],
            num_classes=m["num_classes"],
        )
    elif m["fusion_type"] == "concat":
        return ConcatFusion(
            spatial_dim=m["spatial_dim"],
            freq_dim=m["freq_dim"],
            temporal_dim=m["temporal_dim"],
            hidden_dim=m["hidden_dim"],
            num_classes=m["num_classes"],
        )
    raise ValueError(f"unknown fusion_type: {m['fusion_type']}")


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for spatial, freq, temporal, labels in loader:
            spatial, freq, temporal, labels = (
                spatial.to(device),
                freq.to(device),
                temporal.to(device),
                labels.to(device),
            )
            logits = model(spatial, freq, temporal).squeeze(1)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["train"]["seed"])
    device = torch.device(cfg["train"]["device"])

    train_set = FusionEmbeddingDataset(cfg["data"]["manifest"], cfg["data"]["train_split"])
    val_set = FusionEmbeddingDataset(cfg["data"]["manifest"], cfg["data"]["val_split"])
    train_loader = DataLoader(train_set, batch_size=cfg["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg["train"]["batch_size"], shuffle=False)
    print(f"[{cfg['model']['fusion_type']}] train videos: {len(train_set)}, val videos: {len(val_set)}")

    model = build_model(cfg).to(device)
    print(f"[{cfg['model']['fusion_type']}] params: {sum(p.numel() for p in model.parameters())}")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    checkpoint_dir = Path(cfg["output"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )
        dt = time.time() - t0
        print(
            f"[{cfg['model']['fusion_type']}] epoch {epoch}/{cfg['train']['epochs']} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} ({dt:.1f}s)"
        )

    ckpt_path = checkpoint_dir / f"fusion_{cfg['model']['fusion_type']}_baseline_smoke_test.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"[{cfg['model']['fusion_type']}] saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
