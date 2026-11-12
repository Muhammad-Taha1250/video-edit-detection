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
from concurrent.futures import ProcessPoolExecutor
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


# ─────────────────────────── Parallel Chunk Worker (Spawn-Safe) ──────────────────────

def _calc_hsv_hist_single(frame: np.ndarray) -> np.ndarray:
    """Computes normalized 3D HSV histogram once for a frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [16, 8, 8], [0, 180, 0, 256, 0, 256]).astype(np.float32)
    s = h.sum()
    return h / s if s > 0 else h


def _dct_phash_uint64_from_gray(gray: np.ndarray, size: int = 32) -> int:
    """Computes uint64 pHash directly from a precomputed grayscale image."""
    img = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(img)
    top = dct[:8, :8].flatten()
    mean = (top.sum() - top[0]) / 63.0
    bits = (top[1:] > mean)
    val = 0
    for bit in bits:
        val = (val << 1) | int(bit)
    return val


def _process_frame_chunk_worker(chunk_frames: list[np.ndarray]) -> tuple[np.ndarray, list[int]]:
    """
    Top-level spawn-safe worker for processing a contiguous chunk of frames.
    Precomputes grayscale, HSV histograms, Canny edges, pHash, and ORB descriptors once
    per frame and reuses them across consecutive pairs.
    """
    cv2.setNumThreads(1)
    m = len(chunk_frames)
    if m < 2:
        return np.zeros((0, 7), dtype=np.float32), []

    n_pairs = m - 1

    # 1. Precompute per-frame representations once (reused across pair transitions)
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in chunk_frames]
    frames_f32 = [f.astype(np.float32) for f in chunk_frames]
    hsv_hists = [_calc_hsv_hist_single(f) for f in chunk_frames]
    edges = [cv2.Canny(g, 50, 150).astype(np.float32) for g in grays]
    phashes = [_dct_phash_uint64_from_gray(g) for g in grays]

    orb = cv2.ORB_create(nfeatures=200)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    orb_data = [orb.detectAndCompute(g, None) for g in grays]

    min_dim = min(grays[0].shape)
    win = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
    win = max(win, 3)

    raw_feats = np.zeros((n_pairs, 7), dtype=np.float32)

    for i in range(n_pairs):
        # 0: HSV histogram diff
        raw_feats[i, 0] = float(np.sum(np.abs(hsv_hists[i] - hsv_hists[i + 1])))

        # 1: SSIM distance
        val = _ssim(grays[i], grays[i + 1], win_size=win, data_range=255)
        raw_feats[i, 1] = float(1.0 - val)

        # 2: Pixel MAD
        raw_feats[i, 2] = float(np.mean(np.abs(frames_f32[i] - frames_f32[i + 1])))

        # 3: Edge diff
        raw_feats[i, 3] = float(np.mean(np.abs(edges[i] - edges[i + 1])))

        # 4: pHash distance
        raw_feats[i, 4] = float(int(phashes[i] ^ phashes[i + 1]).bit_count())

        # 5: Optical flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(
            grays[i], grays[i + 1], None, pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.1, flags=0
        )
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        raw_feats[i, 5] = float(mag.mean())

        # 6: ORB ratio
        kpa, da = orb_data[i]
        kpb, db = orb_data[i + 1]
        if da is None or db is None or len(da) < 2 or len(db) < 2:
            raw_feats[i, 6] = 0.0
        else:
            matches = bf.match(da, db)
            total = max(len(kpa), len(kpb))
            raw_feats[i, 6] = float(len(matches) / total) if total > 0 else 0.0

    return raw_feats, phashes


def compute_features_from_frames(
    frames: list[np.ndarray],
    n_workers: Optional[int] = None,
) -> np.ndarray:
    """Computes the 27 temporal features from an in-memory list of frames (scaled to <= 320)."""
    n_frames = len(frames)
    if n_frames < 2:
        return np.zeros((0, N_FEAT_V2), dtype=np.float32)

    n_pairs = n_frames - 1
    if n_workers is None:
        cpu_cnt = os.cpu_count() or 4
        n_workers = min(6, cpu_cnt)

    if n_pairs <= 16 or n_workers <= 1:
        raw_feats, frame_hashes = _process_frame_chunk_worker(frames)
    else:
        n_chunks = min(n_workers, n_pairs)
        chunk_size_pairs = int(np.ceil(n_pairs / n_chunks))
        chunks = []
        for c in range(n_chunks):
            start_p = c * chunk_size_pairs
            end_p = min(n_pairs, (c + 1) * chunk_size_pairs)
            if start_p >= end_p:
                break
            chunks.append(frames[start_p : end_p + 1])

        with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
            chunk_results = list(executor.map(_process_frame_chunk_worker, chunks))

        raw_feats_list = []
        frame_hashes = []
        for c_idx, (r_feats, p_hashes) in enumerate(chunk_results):
            raw_feats_list.append(r_feats)
            if c_idx == 0:
                frame_hashes.extend(p_hashes)
            else:
                frame_hashes.extend(p_hashes[1:])

        raw_feats = np.vstack(raw_feats_list).astype(np.float32)

    ctx_feats = np.zeros_like(raw_feats)
    for col in range(7):
        ctx_feats[:, col] = _rolling_median_context(raw_feats[:, col])

    rob_feats = np.zeros_like(raw_feats)
    for col in range(7):
        rob_feats[:, col] = _robust_norm(raw_feats[:, col])

    step_flow = _long_step_feature(raw_feats[:, 5], w=15)
    step_mad = _long_step_feature(raw_feats[:, 2], w=15)
    step_phash = _long_step_feature(raw_feats[:, 4], w=15)
    step_feats = np.column_stack([step_flow, step_mad, step_phash]).astype(np.float32)

    min_dists, best_lags, lag_runs = _self_similarity(frame_hashes)
    self_feats = np.column_stack([min_dists, best_lags, lag_runs]).astype(np.float32)

    return np.concatenate([raw_feats, ctx_feats, rob_feats, step_feats, self_feats], axis=1).astype(np.float32)


def extract_features_v2(
    video_path: str | Path,
    force: bool = False,
    desc: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    n_workers: Optional[int] = None,
) -> np.ndarray:
    """
    Extracts 27 features per consecutive frame pair using parallel chunk processing.
    Saves and returns (n_pairs, 27).
    """
    video_path = Path(video_path)
    stem = video_path.stem
    active_cache = Path(cache_dir) if cache_dir is not None else CACHE_V2_DIR
    active_cache.mkdir(parents=True, exist_ok=True)
    cache_file = active_cache / f"{stem}.npy"
    meta_file = active_cache / f"{stem}_meta.npy"

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
    out = compute_features_from_frames(frames, n_workers=n_workers)
    del frames

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

