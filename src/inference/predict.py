"""End-to-end inference on a single raw video: face crop (MTCNN) -> spatial
stream -> frequency stream (DCT) -> temporal stream (Bi-LSTM) ->
fusion-attention -> real/fake prediction with confidence and gating weights.

Loads the 250-video-trained checkpoints (not the smoke-test ones). This is
a sanity check that the trained pipeline runs end-to-end on new input --
not a training or evaluation-metrics script.

Usage: python src/inference/predict.py <path/to/video.mp4>
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.evaluate_tiered import (
    load_spatial_encoder,
    load_frequency_encoder,
    load_temporal,
    load_fusion_attention,
    spatial_transform,
    frequency_input,
)

FRAMES_PER_VIDEO = 10

mtcnn = MTCNN(image_size=224, margin=40, post_process=False, device="cpu")


def sample_and_crop_faces(video_path):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError(f"could not read any frames from {video_path}")

    indices = sorted(set(int(x) for x in np.linspace(0, total - 1, FRAMES_PER_VIDEO)))
    crops = []
    skipped = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            skipped += 1
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        face_tensor = mtcnn(pil_img)  # (3,224,224) tensor, or None if no face found
        if face_tensor is None:
            skipped += 1
            continue
        # mtcnn(post_process=False) returns uint8-range float tensor (0-255), CHW
        arr = face_tensor.permute(1, 2, 0).byte().numpy()
        crops.append(Image.fromarray(arr))
    cap.release()

    if not crops:
        raise ValueError(f"no face detected in any sampled frame of {video_path}")
    return crops, skipped, len(indices)


def predict(video_path):
    crops, skipped, n_sampled = sample_and_crop_faces(video_path)

    spatial_enc = load_spatial_encoder()
    freq_enc = load_frequency_encoder()
    temporal_model = load_temporal()
    fusion_model = load_fusion_attention()

    with torch.no_grad():
        spatial_batch = torch.stack([spatial_transform(im) for im in crops])
        spatial_seq = spatial_enc(spatial_batch)  # (T, 512)

        freq_batch = torch.stack([frequency_input(im) for im in crops])  # (T,1,224,224)
        freq_seq = freq_enc(freq_batch)  # (T, 128)

        spatial_pooled = spatial_seq.mean(dim=0, keepdim=True)
        freq_pooled = freq_seq.mean(dim=0, keepdim=True)

        seq_len = torch.tensor([spatial_seq.shape[0]])
        temporal_emb = temporal_model.embed(spatial_seq.unsqueeze(0), seq_len)

        logit, weights = fusion_model(spatial_pooled, freq_pooled, temporal_emb, return_weights=True)
        prob_fake = torch.sigmoid(logit).item()
        w = weights.squeeze(0).tolist()

    label = "FAKE" if prob_fake > 0.5 else "REAL"
    confidence = prob_fake if label == "FAKE" else 1 - prob_fake

    return {
        "label": label,
        "prob_fake": prob_fake,
        "confidence": confidence,
        "w_spatial": w[0],
        "w_frequency": w[1],
        "w_temporal": w[2],
        "frames_used": len(crops),
        "frames_sampled": n_sampled,
        "frames_skipped": skipped,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    args = parser.parse_args()

    result = predict(args.video_path)
    print(f"video: {args.video_path}")
    print(f"frames: {result['frames_used']}/{result['frames_sampled']} used "
          f"({result['frames_skipped']} skipped, no face detected)")
    print(f"prediction: {result['label']}  (P(fake)={result['prob_fake']:.4f}, confidence={result['confidence']:.4f})")
    print(f"attention weights: w_spatial={result['w_spatial']:.4f}  "
          f"w_frequency={result['w_frequency']:.4f}  w_temporal={result['w_temporal']:.4f}")


if __name__ == "__main__":
    main()
