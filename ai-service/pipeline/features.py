"""
features.py – Per-frame-pair feature extraction.

For every consecutive frame pair in a video compute 7 raw features:
  0  hsv_hist_diff   – L1 distance between normalised HSV histograms
  1  ssim            – structural similarity (skimage, converted to distance: 1-SSIM)
  2  pixel_mad       – mean absolute difference of pixel values
  3  edge_diff       – L1 difference of Canny edge-magnitude images
  4  phash_dist      – DCT perceptual-hash Hamming distance (0-64)
  5  flow_mag        – mean Farneback optical-flow magnitude
  6  orb_ratio       – matched ORB keypoints / total detected (0 = poor match)

Then 7 context features (each raw value / rolling-median over k=5 pairs).
Total: 14 features per frame pair.

Cache: ai-service/cache/<video_stem>.npy  (shape: [n_pairs, 14])
       ai-service/cache/<video_stem>_meta.npy  saves fps and total_frames
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from skimage.metrics import structural_similarity as _ssim
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
AI_SERVICE = _HERE.parent
CACHE_DIR = AI_SERVICE / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_LONG_SIDE = 320          # resize so the long side is at most 320 px
TARGET_H = 180               # used for SSIM window sizing only
CONTEXT_K = 5                # half-width of rolling-median context window
FEATURE_NAMES = [
    "hsv_hist_diff", "ssim_dist", "pixel_mad", "edge_diff",
    "phash_dist", "flow_mag", "orb_ratio",
    # context (divided by rolling median)
    "ctx_hsv", "ctx_ssim", "ctx_mad", "ctx_edge",
    "ctx_phash", "ctx_flow", "ctx_orb",
]
N_RAW = 7
N_FEAT = len(FEATURE_NAMES)   # 14


# ---------------------------------------------------------------------------
# Utility: resize preserving aspect ratio, long side <= MAX_LONG_SIDE
# ---------------------------------------------------------------------------
def _resize_frame(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    long = max(h, w)
    if long <= MAX_LONG_SIDE:
        return frame
    scale = MAX_LONG_SIDE / long
    nw = int(round(w * scale / 2) * 2)
    nh = int(round(h * scale / 2) * 2)
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Individual feature functions (operate on small BGR uint8 frames)
# ---------------------------------------------------------------------------
def _hsv_hist_diff(a: np.ndarray, b: np.ndarray) -> float:
    def _hist(f):
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1, 2], None, [16, 8, 8],
                         [0, 180, 0, 256, 0, 256])
        h = h.astype(np.float32)
        s = h.sum()
        return h / s if s > 0 else h
    return float(np.sum(np.abs(_hist(a) - _hist(b))))


def _ssim_dist(a: np.ndarray, b: np.ndarray) -> float:
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    # win_size must be odd and <= min dim
    min_dim = min(ga.shape)
    win = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
    win = max(win, 3)
    val = _ssim(ga, gb, win_size=win, data_range=255)
    return float(1.0 - val)   # distance: 0=identical


def _pixel_mad(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def _edge_diff(a: np.ndarray, b: np.ndarray) -> float:
    ea = cv2.Canny(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), 50, 150).astype(np.float32)
    eb = cv2.Canny(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), 50, 150).astype(np.float32)
    return float(np.mean(np.abs(ea - eb)))


def _dct_phash(frame: np.ndarray, size: int = 32) -> int:
    """Return a 64-bit integer DCT perceptual hash."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(img)
    top = dct[:8, :8].flatten()
    mean = (top.sum() - top[0]) / 63.0
    bits = (top[1:] > mean)
    val = 0
    for bit in bits:
        val = (val << 1) | int(bit)
    return val


def _hamming64(a: int, b: int) -> int:
    x = a ^ b
    count = 0
    while x:
        count += x & 1
        x >>= 1
    return count


def _phash_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(_hamming64(_dct_phash(a), _dct_phash(b)))


def _flow_mag(a: np.ndarray, b: np.ndarray) -> float:
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        ga, gb, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.1, flags=0
    )
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return float(mag.mean())


_ORB = None
_BF  = None

def _orb_ratio(a: np.ndarray, b: np.ndarray) -> float:
    global _ORB, _BF
    if _ORB is None:
        _ORB = cv2.ORB_create(nfeatures=200)
        _BF  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    kpa, da = _ORB.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), None)
    kpb, db = _ORB.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None)
    if da is None or db is None or len(da) < 2 or len(db) < 2:
        return 0.0
    matches = _BF.match(da, db)
    total = max(len(kpa), len(kpb))
    return float(len(matches) / total) if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Rolling-median context
# ---------------------------------------------------------------------------
def _rolling_median_context(arr: np.ndarray, k: int = CONTEXT_K) -> np.ndarray:
    """
    For each position i compute median over [i-k, i+k] (excluding i itself).
    Returns same shape; division is guarded against zero.
    """
    n = len(arr)
    ctx = np.empty_like(arr)
    for i in range(n):
        lo = max(0, i - k)
        hi = min(n, i + k + 1)
        window = np.concatenate([arr[lo:i], arr[i + 1:hi]])
        if len(window) == 0:
            ctx[i] = arr[i]
        else:
            med = np.median(window)
            ctx[i] = arr[i] / (med + 1e-9)
    return ctx


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------
def extract_features(
    video_path: str | Path,
    force: bool = False,
    desc: Optional[str] = None,
) -> np.ndarray:
    """
    Extract features for every consecutive frame pair in *video_path*.

    Returns array of shape (n_pairs, N_FEAT=14).
    Result is cached in CACHE_DIR; pass force=True to recompute.
    """
    video_path = Path(video_path)
    stem = video_path.stem
    cache_file = CACHE_DIR / f"{stem}.npy"
    meta_file  = CACHE_DIR / f"{stem}_meta.npy"

    if cache_file.exists() and meta_file.exists() and not force:
        return np.load(cache_file)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0

    frames: list[np.ndarray] = []
    while True:
        ok, raw = cap.read()
        if not ok:
            break
        frames.append(_resize_frame(raw))
    cap.release()

    if len(frames) < 2:
        empty = np.zeros((0, N_FEAT), dtype=np.float32)
        np.save(cache_file, empty)
        np.save(meta_file, np.array([fps, len(frames)]))
        return empty

    n_pairs = len(frames) - 1
    label = desc or video_path.name
    raw_feats = np.zeros((n_pairs, N_RAW), dtype=np.float32)

    for i in tqdm(range(n_pairs), desc=label, leave=False):
        a, b = frames[i], frames[i + 1]
        raw_feats[i, 0] = _hsv_hist_diff(a, b)
        raw_feats[i, 1] = _ssim_dist(a, b)
        raw_feats[i, 2] = _pixel_mad(a, b)
        raw_feats[i, 3] = _edge_diff(a, b)
        raw_feats[i, 4] = _phash_dist(a, b)
        raw_feats[i, 5] = _flow_mag(a, b)
        raw_feats[i, 6] = _orb_ratio(a, b)

    del frames  # free memory

    # Context features
    ctx_feats = np.zeros_like(raw_feats)
    for col in range(N_RAW):
        ctx_feats[:, col] = _rolling_median_context(raw_feats[:, col])

    out = np.concatenate([raw_feats, ctx_feats], axis=1).astype(np.float32)
    np.save(cache_file, out)
    np.save(meta_file, np.array([fps, n_pairs]))  # fps, n_pairs
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.features <video_path>")
        sys.exit(1)
    arr = extract_features(sys.argv[1], force=True)
    print(f"Feature array shape: {arr.shape}")
    print(f"Feature names: {FEATURE_NAMES}")

