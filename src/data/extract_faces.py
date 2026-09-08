"""Spatial-baseline preprocessing: sample frames, crop faces via MTCNN, save 224x224 crops.

Defaults (smoke-test subset):
  Reads data/splits/dfd_subset_manifest.csv, writes:
  - data/interim/frames/{video_id}/{frame_idx:03d}.jpg
  - data/interim/manifest_spatial_baseline.csv  (frame_path, video_id, label)
  - data/interim/face_detection_log.csv         (video_id, label, frame_idx, status)

Pass --manifest-in/--raw-dir/--out-dir/--manifest-out/--log-out to point at a
different subset (e.g. the 250-clip training set) without touching the
smoke-test outputs above.
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN

FRAMES_PER_VIDEO = 10

mtcnn = MTCNN(image_size=224, margin=40, post_process=False, device="cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-in", default="data/splits/dfd_subset_manifest.csv")
    parser.add_argument("--raw-dir", default="data/raw/dfd_subset")
    parser.add_argument("--out-dir", default="data/interim/frames")
    parser.add_argument("--manifest-out", default="data/interim/manifest_spatial_baseline.csv")
    parser.add_argument("--log-out", default="data/interim/face_detection_log.csv")
    args = parser.parse_args()

    MANIFEST_IN = Path(args.manifest_in)
    RAW_DIR = Path(args.raw_dir)
    OUT_DIR = Path(args.out_dir)
    MANIFEST_OUT = Path(args.manifest_out)
    LOG_OUT = Path(args.log_out)

    with open(MANIFEST_IN) as f:
        videos = list(csv.DictReader(f))

    frame_rows = []
    log_rows = []

    for i, v in enumerate(videos):
        label = v["label"]
        src_path = RAW_DIR / Path(v["file_path"]).name
        video_id = src_path.stem

        cap = cv2.VideoCapture(str(src_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            log_rows.append([video_id, label, -1, "no_frames"])
            cap.release()
            continue

        indices = sorted(set(int(x) for x in np.linspace(0, total - 1, FRAMES_PER_VIDEO)))
        out_dir = OUT_DIR / video_id
        out_dir.mkdir(parents=True, exist_ok=True)

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                log_rows.append([video_id, label, idx, "read_fail"])
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            out_path = out_dir / f"{idx:03d}.jpg"

            try:
                face = mtcnn(pil_img, save_path=str(out_path))
            except Exception as e:
                log_rows.append([video_id, label, idx, f"error:{e}"])
                continue

            if face is None:
                log_rows.append([video_id, label, idx, "no_face_detected"])
                continue

            frame_rows.append([str(out_path).replace("\\", "/"), video_id, label])
            log_rows.append([video_id, label, idx, "ok"])

        cap.release()
        print(f"[{i+1}/{len(videos)}] {video_id} ({label}) done")

    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_path", "video_id", "label"])
        w.writerows(frame_rows)

    with open(LOG_OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "label", "frame_idx", "status"])
        w.writerows(log_rows)

    total_attempted = len(log_rows)
    ok = sum(1 for r in log_rows if r[3] == "ok")
    print(f"\nTotal frames attempted: {total_attempted}")
    print(f"Faces detected successfully: {ok}")
    print(f"Success rate: {ok/total_attempted*100:.1f}%")


if __name__ == "__main__":
    main()
