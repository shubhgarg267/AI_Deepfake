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
from src.data.dataset import FrequencyFrameDataset
from src.models.frequency.small_cnn import FrequencyCNN


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="experiments/configs/frequency_baseline_smoke_test.yaml"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["train"]["seed"])
    device = torch.device(cfg["train"]["device"])

    train_set = FrequencyFrameDataset(cfg["data"]["manifest"], cfg["data"]["train_split"])
    val_set = FrequencyFrameDataset(cfg["data"]["manifest"], cfg["data"]["val_split"])
    train_loader = DataLoader(
        train_set, batch_size=cfg["train"]["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg["train"]["batch_size"], shuffle=False
    )
    print(f"train frames: {len(train_set)}, val frames: {len(val_set)}")

    model = FrequencyCNN(num_classes=cfg["model"]["num_classes"]).to(device)
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
            f"epoch {epoch}/{cfg['train']['epochs']} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} ({dt:.1f}s)"
        )

    ckpt_path = checkpoint_dir / "frequency_baseline_smoke_test.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
