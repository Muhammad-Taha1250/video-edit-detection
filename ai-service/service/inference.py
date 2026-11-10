"""
inference.py – Production Inference Engine for Video Edit Detection.

Provides analyze_video(path, progress_cb=None, job_id=None) -> dict:
  1. Validates and inspects input video (fps, total_frames, duration).
  2. Extracts 27 temporal features per frame pair using features_v2.
  3. Builds temporal stacked window features (+/-15 pairs = 837 dimensions).
  4. Runs Stage-1 class-agnostic Cut Detector & Frame-Drop Detector.
  5. Runs Stage-2 rule-based classification cascade to produce edit events.
  6. Downsamples cut probability series (<= 500 points) for visualization.
  7. Extracts representative frames around junctions into ai-service/extracted_frames/<job_id>/.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Callable, Optional

import cv2
import joblib
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from pipeline.events_v3 import extract_events_v3, pick_peaks
from pipeline.features_v2 import extract_features_v2

AI_SERVICE = Path(__file__).resolve().parent.parent
MODELS_DIR = AI_SERVICE / "models"
FINAL_DIR = MODELS_DIR / "final_v3"
EXTRACTED_FRAMES_DIR = AI_SERVICE / "extracted_frames"
EXTRACTED_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

_MODEL_CACHE: dict = {}


def refine_cut_pair(pair_idx: int, feats_mat: np.ndarray, w: int = 2) -> int:
    """Snaps a detected pair index to argmax of raw pixel_mad * (phash_dist + 1) in +/-w pairs."""
    n = len(feats_mat)
    lo = max(0, pair_idx - w)
    hi = min(n, pair_idx + w + 1)
    scores = feats_mat[lo:hi, 2] * (feats_mat[lo:hi, 4] + 1.0)
    return lo + int(np.argmax(scores))


def refine_fd_pair(pair_idx: int, feats_mat: np.ndarray, w: int = 2) -> int:
    """Snaps frame_drop boundary to max absolute step feature in +/-w pairs."""
    n = len(feats_mat)
    lo = max(0, pair_idx - w)
    hi = min(n, pair_idx + w + 1)
    step_mag = np.abs(feats_mat[lo:hi, 21]) + np.abs(feats_mat[lo:hi, 22]) + np.abs(feats_mat[lo:hi, 23])
    return lo + int(np.argmax(step_mag))


def get_models():
    """Lazily loads and caches the final trained models and thresholds."""
    if "models" not in _MODEL_CACHE:
        cut_path = FINAL_DIR / "cut_detector.joblib"
        fd_path = FINAL_DIR / "framedrop_detector.joblib"
        thresh_path = FINAL_DIR / "thresholds.json"

        if not cut_path.exists() or not fd_path.exists():
            raise FileNotFoundError(
                f"Trained models not found in {FINAL_DIR}. Please run pipeline/train_final_v3.py first."
            )

        cut_detector = joblib.load(cut_path)
        fd_detector = joblib.load(fd_path)

        thresholds = {"thresh_cut": 0.35, "thresh_fd": 0.20}
        if thresh_path.exists():
            try:
                thresholds = json.loads(thresh_path.read_text())
            except Exception:
                pass

        _MODEL_CACHE["models"] = (cut_detector, fd_detector, thresholds)

    return _MODEL_CACHE["models"]


def build_stacked_features(mat: np.ndarray, w: int = 15) -> np.ndarray:
    """Builds +/-w sliding window stacked features ((2w+1)*d dimensions)."""
    n, d = mat.shape
    padded = np.pad(mat, ((w, w), (0, 0)), mode="edge")
    windows = sliding_window_view(padded, (2 * w + 1, d))[:, 0, :, :]
    stacked = windows.reshape(n, (2 * w + 1) * d)
    return stacked.astype(np.float32)


def downsample_probabilities(
    prob_cut: np.ndarray,
    fps: float,
    critical_indices: set[int],
    max_points: int = 500,
) -> list[dict]:
    """
    Downsamples the cut-probability curve to at most max_points,
    guaranteeing that all detected event boundaries / peak indices are preserved.
    """
    n = len(prob_cut)
    if n <= max_points:
        chosen_indices = sorted(range(n))
    else:
        # Uniform sample
        base_indices = set(np.linspace(0, n - 1, max_points, dtype=int).tolist())
        all_chosen = sorted(base_indices | critical_indices)
        if len(all_chosen) > max_points:
            # Keep all critical and sub-sample remainder
            non_crit = [i for i in all_chosen if i not in critical_indices]
            needed = max_points - len(critical_indices)
            if needed > 0 and len(non_crit) > needed:
                step = len(non_crit) / needed
                sub_non_crit = {non_crit[int(i * step)] for i in range(needed)}
                chosen_indices = sorted(set(critical_indices) | sub_non_crit)
            else:
                chosen_indices = sorted(critical_indices)
        else:
            chosen_indices = all_chosen

    return [
        {
            "frame": int(idx),
            "time": round(float(idx / fps), 3),
            "prob": round(float(prob_cut[idx]), 4),
        }
        for idx in chosen_indices
    ]


def extract_event_frames(
    video_path: Path,
    events: list[dict],
    job_id: str,
    total_frames: int,
) -> None:
    """
    Extracts representative frames around junctions into extracted_frames/<job_id>/:
      - Single-cut edits (deletion): before (s-1), junction (s), after (s+1)
      - Range edits (insertion, duplication, frame_drop):
          Start junction: before (s-1), start (s), after (s+1)
          End junction: before (e-1), end (e), after (e+1)
    """
    job_frames_dir = EXTRACTED_FRAMES_DIR / job_id
    job_frames_dir.mkdir(parents=True, exist_ok=True)

    needed_frames: dict[int, list[tuple[int, str]]] = {}
    for ev_idx, ev in enumerate(events):
        s = ev["start_frame"]
        e = ev["end_frame"]
        if s == e:
            needed_frames.setdefault(max(0, s - 1), []).append((ev_idx, "before"))
            needed_frames.setdefault(s, []).append((ev_idx, "junction"))
            needed_frames.setdefault(min(total_frames - 1, s + 1), []).append((ev_idx, "after"))
        else:
            needed_frames.setdefault(max(0, s - 1), []).append((ev_idx, "before_start"))
            needed_frames.setdefault(s, []).append((ev_idx, "start"))
            needed_frames.setdefault(min(total_frames - 1, s + 1), []).append((ev_idx, "after_start"))
            if ev.get("type") == "insertion":
                needed_frames.setdefault(e, []).append((ev_idx, "before_end"))
                needed_frames.setdefault(min(total_frames - 1, e + 1), []).append((ev_idx, "end"))
                needed_frames.setdefault(min(total_frames - 1, e + 2), []).append((ev_idx, "after_end"))
            else:
                needed_frames.setdefault(max(0, e - 1), []).append((ev_idx, "before_end"))
                needed_frames.setdefault(e, []).append((ev_idx, "end"))
                needed_frames.setdefault(min(total_frames - 1, e + 1), []).append((ev_idx, "after_end"))

    for ev in events:
        ev["extracted_frames"] = {}

    if not needed_frames:
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return

    for f_idx in sorted(needed_frames.keys()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        for ev_idx, role in needed_frames[f_idx]:
            ev_type = events[ev_idx]["type"]
            filename = f"event_{ev_idx}_{ev_type}_f{f_idx}_{role}.jpg"
            out_file = job_frames_dir / filename
            cv2.imwrite(str(out_file), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            rel_path = f"extracted_frames/{job_id}/{filename}"
            events[ev_idx]["extracted_frames"][role] = {
                "frame": int(f_idx),
                "role": role,
                "filename": filename,
                "relative_path": rel_path,
                "url": f"/frames/{job_id}/{filename}",
            }

    cap.release()


def analyze_video(
    video_path: str | Path,
    progress_cb: Optional[Callable[[int, str], None]] = None,
    job_id: Optional[str] = None,
) -> dict:
    """
    Analyzes a video for editing events.

    Returns:
      {
        "job_id": str,
        "video_path": str,
        "fps": float,
        "total_frames": int,
        "duration": float,
        "events": [
            {
               "type": "deletion" | "insertion" | "duplication" | "frame_drop",
               "edit_type": str,
               "start_frame": int,
               "end_frame": int,
               "start_time": float,
               "end_time": float,
               "confidence": float,
               "extracted_frames": dict
            }, ...
        ],
        "cut_probabilities": [{"frame": int, "time": float, "prob": float}, ...]
      }
    """
    video_path = Path(video_path).resolve()
    if not video_path.exists() or not video_path.is_file():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    if job_id is None:
        job_id = uuid.uuid4().hex[:12]

    def _progress(pct: int, msg: str):
        if progress_cb is not None:
            try:
                progress_cb(pct, msg)
            except Exception:
                pass

    _progress(5, "Inspecting video metadata...")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    if total_frames < 2:
        raise ValueError(f"Video has insufficient frames ({total_frames}) for pairwise analysis.")

    duration = round(float(total_frames / fps), 3)

    # 1. Feature extraction
    _progress(15, "Extracting temporal features (27 dimensions)...")
    feats_mat = extract_features_v2(video_path)
    n_pairs = len(feats_mat)
    if n_pairs == 0:
        raise ValueError(f"Extracted 0 frame pairs from video {video_path}")

    # 2. Build stacked window
    _progress(60, "Building temporal neighborhood features (837 dimensions)...")
    X = build_stacked_features(feats_mat, w=15)

    # 3. Model inference
    _progress(70, "Running cut detector & frame drop detector...")
    cut_detector, fd_detector, thresholds = get_models()
    thresh_cut = thresholds.get("thresh_cut", 0.35)
    thresh_fd = thresholds.get("thresh_fd", 0.20)

    prob_cut = cut_detector.predict_proba(X)[:, 1]
    prob_fd = fd_detector.predict_proba(X)[:, 1]

    # 4. Stage 2 rule-based classification with junction refinement
    _progress(80, "Classifying edit events and boundaries...")
    n_pairs = len(prob_cut)
    raw_cuts = pick_peaks(prob_cut, threshold=thresh_cut, min_distance=10)
    cuts = sorted(list({refine_cut_pair(c, feats_mat, w=2) for c in raw_cuts}))

    events: list[dict] = []
    used_cuts = set()

    # Rule 1: INSERTION
    for i in range(len(cuts)):
        if i in used_cuts:
            continue
        c1 = cuts[i]
        for j in range(i + 1, len(cuts)):
            if j in used_cuts:
                continue
            c2 = cuts[j]
            gap = c2 - c1
            if 10 <= gap <= 95:
                used_cuts.add(i)
                used_cuts.add(j)
                conf = float((prob_cut[c1] + prob_cut[c2]) / 2.0)
                start_f = int(c1 + 1)
                end_f = int(c2)
                events.append({
                    "type": "insertion",
                    "edit_type": "insertion",
                    "start_frame": start_f,
                    "end_frame": end_f,
                    "start_time": round(start_f / fps, 3),
                    "end_time": round(end_f / fps, 3),
                    "confidence": round(conf, 4),
                })
                break

    min_dists = feats_mat[:, 24] if feats_mat.shape[1] > 24 else np.full(n_pairs, 64.0)
    best_lags = feats_mat[:, 25] if feats_mat.shape[1] > 25 else np.zeros(n_pairs)
    lag_runs = feats_mat[:, 26] if feats_mat.shape[1] > 26 else np.zeros(n_pairs)

    # Rule 2: DUPLICATION
    for i in range(len(cuts)):
        if i in used_cuts:
            continue
        c = cuts[i]
        sub_end = min(n_pairs, c + 65)
        sub_runs = lag_runs[c : sub_end]
        sub_dists = min_dists[c : sub_end]
        sub_lags = best_lags[c : sub_end]

        is_dup = False
        est_l = 0

        if len(sub_runs) > 0 and np.max(sub_runs) >= 10:
            valid_lags = sub_lags[sub_runs >= 4]
            if len(valid_lags) > 0:
                est_l = int(round(float(np.median(valid_lags))))
                if 10 <= est_l <= 90:
                    is_dup = True
        elif len(sub_dists) >= 10:
            valid = sub_lags[sub_dists <= 12]
            if len(valid) >= 10:
                est_l = int(round(float(np.median(valid))))
                if 10 <= est_l <= 90:
                    is_dup = True

        if is_dup:
            used_cuts.add(i)
            start_f = int(c + 1)
            end_f = min(total_frames - 1, start_f + est_l - 1)
            events.append({
                "type": "duplication",
                "edit_type": "duplication",
                "start_frame": start_f,
                "end_frame": int(end_f),
                "start_time": round(start_f / fps, 3),
                "end_time": round(end_f / fps, 3),
                "confidence": round(float(prob_cut[c]), 4),
            })

    # Rule 3: DELETION
    for i in range(len(cuts)):
        if i in used_cuts:
            continue
        c = cuts[i]
        used_cuts.add(i)
        junc_f = int(c + 1)
        events.append({
            "type": "deletion",
            "edit_type": "deletion",
            "start_frame": junc_f,
            "end_frame": junc_f,
            "start_time": round(junc_f / fps, 3),
            "end_time": round(junc_f / fps, 3),
            "confidence": round(float(prob_cut[c]), 4),
        })

    # Separate Frame Drop Detector
    raw_fd = pick_peaks(prob_fd, threshold=thresh_fd, min_distance=10)
    fd_peaks = sorted(list({refine_fd_pair(p, feats_mat, w=2) for p in raw_fd}))
    used_fd = set()

    for i in range(len(fd_peaks)):
        if i in used_fd:
            continue
        p1 = fd_peaks[i]
        for j in range(i + 1, len(fd_peaks)):
            if j in used_fd:
                continue
            p2 = fd_peaks[j]
            gap = p2 - p1
            if 10 <= gap <= 95:
                used_fd.add(i)
                used_fd.add(j)
                conf = float((prob_fd[p1] + prob_fd[p2]) / 2.0)
                events.append({
                    "type": "frame_drop",
                    "edit_type": "frame_drop",
                    "start_frame": int(p1),
                    "end_frame": int(p2),
                    "start_time": round(p1 / fps, 3),
                    "end_time": round(p2 / fps, 3),
                    "confidence": round(conf, 4),
                })
                break

    events.sort(key=lambda e: (e["start_frame"], -e["confidence"]))

    critical_indices = set()
    for ev in events:
        critical_indices.add(ev["start_frame"])
        critical_indices.add(ev["end_frame"])

    # 5. Extract representative junction frames
    _progress(88, "Extracting junction frames...")
    extract_event_frames(
        video_path=video_path,
        events=events,
        job_id=job_id,
        total_frames=total_frames,
    )

    # 6. Downsample cut probabilities for UI charting
    _progress(95, "Generating visualization curves...")
    chart_series = downsample_probabilities(
        prob_cut=prob_cut,
        fps=fps,
        critical_indices=critical_indices,
        max_points=500,
    )

    _progress(100, "Analysis complete.")

    return {
        "job_id": job_id,
        "video_path": str(video_path),
        "fps": round(fps, 3),
        "total_frames": total_frames,
        "duration": duration,
        "events": events,
        "cut_probabilities": chart_series,
    }

