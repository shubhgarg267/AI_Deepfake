"""PyTorch Datasets for the spatial and frequency baselines: read frame-level
manifests, filtered to a video-level train/val split."""
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_video_ids(split_csv):
    with open(split_csv) as f:
        return {row["video_id"] for row in csv.DictReader(f)}


class SpatialFrameDataset(Dataset):
    def __init__(self, manifest_csv, split_csv, crop_size=224, augment=False):
        video_ids = load_video_ids(split_csv)
        with open(manifest_csv) as f:
            rows = list(csv.DictReader(f))
        self.rows = [r for r in rows if r["video_id"] in video_ids]
        self.augment = augment

        self.transform = transforms.Compose(
            [
                transforms.Resize((crop_size, crop_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = Image.open(row["frame_path"]).convert("RGB")
        if self.augment:
            from src.features.degradation_augmentation import degrade
            img, _ = degrade(img)
        img = self.transform(img)
        label = 1.0 if row["label"] == "FAKE" else 0.0
        return img, torch.tensor(label, dtype=torch.float32)


class FrequencyFrameDataset(Dataset):
    """augment=True recomputes the DCT map on the fly from a degraded raw
    crop (frames_dir required) instead of the precomputed clean .npz --
    mirrors the raw crop path from frequency_path by directory convention
    (frequency_dir/{video_id}/{stem}.npz -> frames_dir/{video_id}/{stem}.jpg)."""

    def __init__(self, manifest_csv, split_csv, augment=False, frames_dir=None):
        video_ids = load_video_ids(split_csv)
        with open(manifest_csv) as f:
            rows = list(csv.DictReader(f))
        self.rows = [r for r in rows if r["video_id"] in video_ids]
        self.augment = augment
        self.frames_dir = frames_dir
        if augment and frames_dir is None:
            raise ValueError("frames_dir is required when augment=True")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        if self.augment:
            from src.features.degradation_augmentation import degrade
            from src.features.dct_transform import block_dct

            freq_path = Path(row["frequency_path"])
            frame_path = Path(self.frames_dir) / freq_path.parent.name / (freq_path.stem + ".jpg")
            img = Image.open(frame_path).convert("RGB")
            img, _ = degrade(img)
            gray = np.asarray(img.convert("L"), dtype=np.float32)
            dct_map = block_dct(gray).astype(np.float32)
        else:
            data = np.load(row["frequency_path"])
            dct_map = data["dct"].astype(np.float32)

        dct_map = (dct_map - dct_map.mean()) / (dct_map.std() + 1e-6)
        tensor = torch.from_numpy(dct_map).unsqueeze(0)  # (1, H, W)
        label = 1.0 if row["label"] == "FAKE" else 0.0
        return tensor, torch.tensor(label, dtype=torch.float32)


class TemporalSequenceDataset(Dataset):
    def __init__(self, manifest_csv, split_csv):
        video_ids = load_video_ids(split_csv)
        with open(manifest_csv) as f:
            rows = list(csv.DictReader(f))
        self.rows = [r for r in rows if r["video_id"] in video_ids]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        embeddings = np.load(row["embedding_path"]).astype(np.float32)  # (T, 512)
        tensor = torch.from_numpy(embeddings)
        label = 1.0 if row["label"] == "FAKE" else 0.0
        return tensor, torch.tensor(label, dtype=torch.float32)


def temporal_collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True)  # (batch, T_max, 512)
    labels = torch.stack(labels)
    return padded, lengths, labels


class FusionEmbeddingDataset(Dataset):
    def __init__(self, manifest_csv, split_csv):
        video_ids = load_video_ids(split_csv)
        with open(manifest_csv) as f:
            rows = list(csv.DictReader(f))
        self.rows = [r for r in rows if r["video_id"] in video_ids]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        data = np.load(row["embedding_path"])
        spatial = torch.from_numpy(data["spatial"].astype(np.float32))
        freq = torch.from_numpy(data["frequency"].astype(np.float32))
        temporal = torch.from_numpy(data["temporal"].astype(np.float32))
        label = 1.0 if row["label"] == "FAKE" else 0.0
        return spatial, freq, temporal, torch.tensor(label, dtype=torch.float32)
