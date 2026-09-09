"""Degradation-aware augmentation: JPEG re-compression, H.264 re-encoding,
Gaussian noise, downscale/upscale, and motion blur, compounded 1-3 at a time
across mild/medium/heavy severity tiers (25% of samples stay clean).

PSNR/SSIM are hand-rolled in pure numpy rather than via scikit-image, which
requires numpy>=2.0 and conflicts with facenet-pytorch's numpy<2.0.0 pin
(see requirements.txt).
"""
import io
import os
import random
import subprocess
import tempfile

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image

TIER_RANGES = {
    "mild": {
        "jpeg_quality": (85, 95),
        "h264_crf": (18, 23),
        "noise_sigma": (2, 6),
        "downscale_factor": (0.6, 0.75),
        "blur_kernel": (3, 6),
    },
    "medium": {
        "jpeg_quality": (60, 84),
        "h264_crf": (24, 32),
        "noise_sigma": (7, 13),
        "downscale_factor": (0.4, 0.6),
        "blur_kernel": (7, 10),
    },
    "heavy": {
        "jpeg_quality": (30, 59),
        "h264_crf": (33, 40),
        "noise_sigma": (14, 20),
        "downscale_factor": (0.25, 0.4),
        "blur_kernel": (11, 15),
    },
}
DEGRADATION_TYPES = ["jpeg", "h264", "noise", "downscale", "blur"]
CLEAN_PROB = 0.25

_FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()


def apply_jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB").copy()


def apply_gaussian_noise(img: Image.Image, sigma: float) -> Image.Image:
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def apply_downscale_upscale(img: Image.Image, factor: float) -> Image.Image:
    w, h = img.size
    small = img.resize((max(1, int(w * factor)), max(1, int(h * factor))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def apply_motion_blur(img: Image.Image, kernel_size: int, angle: float = None) -> Image.Image:
    arr = np.asarray(img.convert("RGB"))
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[(kernel_size - 1) // 2, :] = 1.0
    if angle is None:
        angle = random.uniform(0, 360)
    center = (kernel_size / 2 - 0.5, kernel_size / 2 - 0.5)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
    total = kernel.sum()
    if total > 0:
        kernel /= total
    blurred = cv2.filter2D(arr, -1, kernel)
    return Image.fromarray(blurred)


def apply_h264(img: Image.Image, crf: int) -> Image.Image:
    """Approximates H.264 re-encoding via an ffmpeg round-trip: encode a
    single frame as a 1-frame mp4 at the target CRF, then decode it back.
    No inter-frame prediction (single frame), but captures the same
    intra-frame block-DCT quantization/deblocking H.264 CRF controls."""
    rgb = img.convert("RGB")
    w, h = rgb.size
    w2, h2 = w - (w % 2), h - (h % 2)  # libx264 requires even dimensions
    rgb = rgb.crop((0, 0, w2, h2))

    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, "in.png")
        mp4_path = os.path.join(tmpdir, "out.mp4")
        out_path = os.path.join(tmpdir, "out.png")
        rgb.save(in_path)

        subprocess.run(
            [_FFMPEG_EXE, "-y", "-loop", "1", "-i", in_path, "-frames:v", "1",
             "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", mp4_path],
            check=True, capture_output=True,
        )
        subprocess.run(
            [_FFMPEG_EXE, "-y", "-i", mp4_path, "-frames:v", "1", out_path],
            check=True, capture_output=True,
        )
        result = Image.open(out_path).convert("RGB").copy()

    if result.size != (w, h):
        result = result.resize((w, h), Image.BILINEAR)
    return result


_APPLY_FN = {
    "jpeg": lambda img, params: apply_jpeg(img, random.randint(*params["jpeg_quality"])),
    "h264": lambda img, params: apply_h264(img, random.randint(*params["h264_crf"])),
    "noise": lambda img, params: apply_gaussian_noise(img, random.uniform(*params["noise_sigma"])),
    "downscale": lambda img, params: apply_downscale_upscale(img, random.uniform(*params["downscale_factor"])),
    "blur": lambda img, params: apply_motion_blur(
        img, random.randrange(params["blur_kernel"][0] | 1, params["blur_kernel"][1] + 1, 2)
    ),
}


def degrade(img: Image.Image, rng: random.Random = None):
    """Randomly degrades a PIL image per the tier/compounding policy.
    Returns (degraded_img, metadata) where metadata records tier and the
    exact types/params applied, for later logging/analysis."""
    rng = rng or random
    if rng.random() < CLEAN_PROB:
        return img.copy(), {"tier": "clean", "types": []}

    tier = rng.choice(["mild", "medium", "heavy"])
    params = TIER_RANGES[tier]
    n_types = rng.randint(1, 3)
    types = rng.sample(DEGRADATION_TYPES, n_types)

    out = img
    applied = []
    for t in types:
        prev = out
        out = _APPLY_FN[t](out, params)
        applied.append({"type": t})
        del prev

    return out, {"tier": tier, "types": types}


def degrade_fixed_tier(img: Image.Image, tier: str) -> Image.Image:
    """Deterministic, single fixed pass for per-tier evaluation: applies all
    5 degradation types (fixed order) at the tier's midpoint parameters, no
    random sampling of type/severity/angle. tier='clean' returns img unchanged."""
    if tier == "clean":
        return img.copy()

    p = TIER_RANGES[tier]
    out = img
    out = apply_jpeg(out, int(round((p["jpeg_quality"][0] + p["jpeg_quality"][1]) / 2)))
    out = apply_h264(out, int(round((p["h264_crf"][0] + p["h264_crf"][1]) / 2)))
    out = apply_gaussian_noise(out, (p["noise_sigma"][0] + p["noise_sigma"][1]) / 2)
    out = apply_downscale_upscale(out, (p["downscale_factor"][0] + p["downscale_factor"][1]) / 2)
    k = int(round((p["blur_kernel"][0] + p["blur_kernel"][1]) / 2))
    k = k if k % 2 == 1 else k + 1
    out = apply_motion_blur(out, k, angle=45.0)
    return out


def compute_psnr(clean: np.ndarray, degraded: np.ndarray) -> float:
    mse = np.mean((clean.astype(np.float64) - degraded.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10((255.0 ** 2) / mse)


def _box_filter(a: np.ndarray, win: int) -> np.ndarray:
    pad = win // 2
    ap = np.pad(a, pad, mode="reflect")

    def cs_1d(arr, axis):
        c = np.cumsum(arr, axis=axis)
        c = np.insert(c, 0, 0, axis=axis)
        n_out = arr.shape[axis] - win + 1
        hi = np.take(c, range(win, win + n_out), axis=axis)
        lo = np.take(c, range(0, n_out), axis=axis)
        return (hi - lo) / win

    out = cs_1d(ap, axis=0)
    out = cs_1d(out, axis=1)
    return out


def compute_ssim(clean: np.ndarray, degraded: np.ndarray, win: int = 7) -> float:
    """Windowed SSIM (Wang et al. 2004) on grayscale, hand-rolled box filter."""
    def to_gray(x):
        return np.dot(x[..., :3].astype(np.float64), [0.299, 0.587, 0.114])

    x, y = to_gray(clean), to_gray(degraded)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    mu_x, mu_y = _box_filter(x, win), _box_filter(y, win)
    mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
    sigma_x2 = _box_filter(x * x, win) - mu_x2
    sigma_y2 = _box_filter(y * y, win) - mu_y2
    sigma_xy = _box_filter(x * y, win) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
        (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    )
    return float(ssim_map.mean())
