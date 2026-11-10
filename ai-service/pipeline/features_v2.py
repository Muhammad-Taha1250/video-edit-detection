"""
features_v2.py – Day 3 Expanded Feature Extraction.

Features per frame-pair (27 total):
  0..6   Raw features (Day 2):
           hsv_hist_diff, ssim_dist, pixel_mad, edge_diff, phash_dist, flow_mag, orb_ratio
  7..13  Context features (Day 2 rolling median ratio, k=5):
           ctx_hsv, ctx_ssim, ctx_mad, ctx_edge, ctx_phash, ctx_flow, ctx_orb
  14..20 Robust normalization (per-video median & MAD scaling):
           rob_hsv, rob_ssim, rob_mad, rob_edge, rob_phash, rob_flow, rob_orb
  21..23 Long-context step features (mean next 15 - mean prev 15 / local std):
           step_flow, step_mad, step_phash
  24..26 Self-similarity (pHash Hamming distance in [t-200, t-10]):
           self_min_dist, self_best_lag, self_lag_run

Cache: ai-service/cache_v2/<stem>.npy
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from skimage.metrics import structural_similarity as _ssim
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
AI_SERVICE = _HERE.parent
CACHE_V2_DIR = AI_SERVICE / "cache_v2"
CACHE_V2_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_NAMES_V2 = [
    # Day 2 raw (7)
    "hsv_hist_diff", "ssim_dist", "pixel_mad", "edge_diff",
    "phash_dist", "flow_mag", "orb_ratio",
    # Day 2 k=5 context (7)
    "ctx_hsv", "ctx_ssim", "ctx_mad", "ctx_edge",
    "ctx_phash", "ctx_flow", "ctx_orb",
    # Day 3 robust normalized (7)
    "rob_hsv", "rob_ssim", "rob_mad", "rob_edge",
    "rob_phash", "rob_flow", "rob_orb",
    # Day 3 step features (3)
    "step_flow", "step_mad", "step_phash",
    # Day 3 self-similarity (3)
    "self_min_dist", "self_best_lag", "self_lag_run",
]
N_FEAT_V2 = len(FEATURE_NAMES_V2)  # 27
MAX_LONG_SIDE = 320
CONTEXT_K = 5


def _resize_frame(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    long = max(h, w)
    if long <= MAX_LONG_SIDE:
        return frame
    scale = MAX_LONG_SIDE / long
    nw = int(round(w * scale / 2) * 2)
    nh = int(round(h * scale / 2) * 2)
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)


def _dct_phash_uint64(frame: np.ndarray, size: int = 32) -> int:
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


def _hsv_hist_diff(a: np.ndarray, b: np.ndarray) -> float:
    def _hist(f):
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1, 2], None, [16, 8, 8], [0, 180, 0, 256, 0, 256]).astype(np.float32)
        s = h.sum()
        return h / s if s > 0 else h
    return float(np.sum(np.abs(_hist(a) - _hist(b))))


def _ssim_dist(a: np.ndarray, b: np.ndarray) -> float:
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    min_dim = min(ga.shape)
    win = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
    win = max(win, 3)
    val = _ssim(ga, gb, win_size=win, data_range=255)
    return float(1.0 - val)


def _pixel_mad(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def _edge_diff(a: np.ndarray, b: np.ndarray) -> float:
    ea = cv2.Canny(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), 50, 150).astype(np.float32)
    eb = cv2.Canny(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), 50, 150).astype(np.float32)
    return float(np.mean(np.abs(ea - eb)))


def _flow_mag(a: np.ndarray, b: np.ndarray) -> float:
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        ga, gb, None, pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.1, flags=0
    )
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return float(mag.mean())


_ORB = None
_BF = None

def _orb_ratio(a: np.ndarray, b: np.ndarray) -> float:
    global _ORB, _BF
    if _ORB is None:
        _ORB = cv2.ORB_create(nfeatures=200)
        _BF = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    kpa, da = _ORB.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), None)
    kpb, db = _ORB.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None)
    if da is None or db is None or len(da) < 2 or len(db) < 2:
        return 0.0
    matches = _BF.match(da, db)
    total = max(len(kpa), len(kpb))
    return float(len(matches) / total) if total > 0 else 0.0


def _rolling_median_context(arr: np.ndarray, k: int = CONTEXT_K) -> np.ndarray:
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


def _robust_norm(series: np.ndarray) -> np.ndarray:
    """(series - median) / (MAD + 1e-6)"""
    med = np.median(series)
    mad = np.median(np.abs(series - med))
    return (series - med) / (mad + 1e-6)


def _long_step_feature(series: np.ndarray, w: int = 15) -> np.ndarray:
    """
    Step feature: mean(next w pairs) - mean(prev w pairs) / (local std + 1e-6).
    """
    n = len(series)
    step = np.zeros(n, dtype=np.float32)
    # Pad symmetrically for boundaries
    padded = np.pad(series, w, mode="edge")
    for i in range(n):
        idx = i + w
        prev_win = padded[idx - w : idx]
        next_win = padded[idx : idx + w]
        local_win = padded[idx - w : idx + w]
        m_next = np.mean(next_win)
        m_prev = np.mean(prev_win)
        l_std = np.std(local_win)
        step[i] = (m_next - m_prev) / (l_std + 1e-6)
    return step


def _self_similarity(frame_hashes: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each frame t in 1..N-1 (corresponding to pair t-1):
    Min pHash Hamming distance to frames in [t-200, t-10], best lag, and lag run length.
    """
    n_frames = len(frame_hashes)
    n_pairs = max(0, n_frames - 1)
    min_dists = np.full(n_pairs, 64.0, dtype=np.float32)
    best_lags = np.zeros(n_pairs, dtype=np.float32)
    lag_runs = np.zeros(n_pairs, dtype=np.float32)

    cur_run = 0
    prev_lag = -999

    for pair_idx in range(n_pairs):
        t = pair_idx + 1  # current frame index
        if t < 11:
            continue
        lo = max(0, t - 200)
        hi = t - 10
        cur_h = frame_hashes[t]
        dists = [int(cur_h ^ frame_hashes[s]).bit_count() for s in range(lo, hi)]
        if not dists:
            continue
        min_idx = int(np.argmin(dists))
        min_d = dists[min_idx]
        lag = t - (lo + min_idx)

        min_dists[pair_idx] = float(min_d)
        best_lags[pair_idx] = float(lag)

        # Constant lag run: duplicate frames have matching lag and low distance
        if abs(lag - prev_lag) <= 1 and min_d <= 12:
            cur_run += 1
        else:
            cur_run = 1 if min_d <= 12 else 0

        prev_lag = lag
        lag_runs[pair_idx] = float(cur_run)

    return min_dists, best_lags, lag_runs


def extract_features_v2(
    video_path: str | Path,
    force: bool = False,
    desc: Optional[str] = None,
) -> np.ndarray:
    """
    Extracts 27 features per consecutive frame pair.
    Saves and returns (n_pairs, 27).
    """
    video_path = Path(video_path)
    stem = video_path.stem
    cache_file = CACHE_V2_DIR / f"{stem}.npy"
    meta_file = CACHE_V2_DIR / f"{stem}_meta.npy"

    if cache_file.exists() and meta_file.exists() and not force:
        return np.load(cache_file)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    frames: list[np.ndarray] = []
    while True:
        ok, raw = cap.read()
        if not ok:
            break
        frames.append(_resize_frame(raw))
    cap.release()

    n_frames = len(frames)
    if n_frames < 2:
        empty = np.zeros((0, N_FEAT_V2), dtype=np.float32)
        np.save(cache_file, empty)
        np.save(meta_file, np.array([fps, 0]))
        return empty

    n_pairs = n_frames - 1

    # 1. Compute frame pHashes for all frames
    frame_hashes = [_dct_phash_uint64(f) for f in frames]

    # 2. Compute 7 raw pairwise features
    raw_feats = np.zeros((n_pairs, 7), dtype=np.float32)
    for i in range(n_pairs):
        a, b = frames[i], frames[i + 1]
        raw_feats[i, 0] = _hsv_hist_diff(a, b)
        raw_feats[i, 1] = _ssim_dist(a, b)
        raw_feats[i, 2] = _pixel_mad(a, b)
        raw_feats[i, 3] = _edge_diff(a, b)
        raw_feats[i, 4] = float(int(frame_hashes[i] ^ frame_hashes[i + 1]).bit_count())
        raw_feats[i, 5] = _flow_mag(a, b)
        raw_feats[i, 6] = _orb_ratio(a, b)

    del frames

    # 3. Compute 7 context features (k=5 rolling median)
    ctx_feats = np.zeros_like(raw_feats)
    for col in range(7):
        ctx_feats[:, col] = _rolling_median_context(raw_feats[:, col])

    # 4. Compute 7 robust normalized features (per-video median / MAD)
    rob_feats = np.zeros_like(raw_feats)
    for col in range(7):
        rob_feats[:, col] = _robust_norm(raw_feats[:, col])

    # 5. Compute 3 long-context step features (flow_mag, pixel_mad, phash_dist)
    step_flow = _long_step_feature(raw_feats[:, 5], w=15)
    step_mad = _long_step_feature(raw_feats[:, 2], w=15)
    step_phash = _long_step_feature(raw_feats[:, 4], w=15)
    step_feats = np.column_stack([step_flow, step_mad, step_phash]).astype(np.float32)

    # 6. Compute 3 self-similarity features
    min_dists, best_lags, lag_runs = _self_similarity(frame_hashes)
    self_feats = np.column_stack([min_dists, best_lags, lag_runs]).astype(np.float32)

    out = np.concatenate([raw_feats, ctx_feats, rob_feats, step_feats, self_feats], axis=1).astype(np.float32)
    np.save(cache_file, out)
    np.save(meta_file, np.array([fps, n_pairs]))
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.features_v2 <video_path>")
        sys.exit(1)
    feats = extract_features_v2(sys.argv[1], force=True)
    print(f"Features v2 shape: {feats.shape}")
    print(f"Feature count: {len(FEATURE_NAMES_V2)}")

