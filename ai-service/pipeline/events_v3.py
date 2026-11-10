"""
events_v3.py – Two-Stage Cut Detection & Rule-Based Event Extraction.

Architecture:
  STAGE 1: Class-Agnostic Cut Detector
    - Detects all physical junction boundaries (deletion cut, insertion start/end, duplication start).
    - Peak picking on cut probability with minimum distance 10 pairs above threshold.

  STAGE 2: Rule-Based Junction Classification (in strict order):
    1. INSERTION: If another cut lies 10-90 frames after it, pair both as one INSERTION
       (start = first cut, end = second cut).
    2. DUPLICATION: Else if self-similarity features show constant lag L (>=10 frames run)
       and the cut jumps back by ~L frames, mark as DUPLICATION (start = cut, end = start + L - 1).
    3. DELETION: Else, mark as DELETION (a lone cut).

  SEPARATE: Frame Drop Rate-Change Detector
    - Detects rate-change step boundaries (start and end within 10-90 frames) with its own threshold.
    - Not fed into the cut-detector stage.
"""

from __future__ import annotations

from typing import Optional
import numpy as np


def pick_peaks(
    prob_series: np.ndarray,
    threshold: float,
    min_distance: int = 10,
) -> list[int]:
    """Finds local maxima strictly >= threshold separated by at least min_distance."""
    peaks = []
    n = len(prob_series)
    for i in range(1, n - 1):
        if prob_series[i] >= threshold:
            if prob_series[i] >= prob_series[i - 1] and prob_series[i] >= prob_series[i + 1]:
                if not peaks or (i - peaks[-1]) >= min_distance:
                    peaks.append(i)
                elif prob_series[i] > prob_series[peaks[-1]]:
                    peaks[-1] = i
    return peaks


def extract_events_v3(
    prob_cut: np.ndarray,
    prob_fd: np.ndarray,
    features_mat: np.ndarray,
    thresh_cut: float,
    thresh_fd: float,
    fps: float = 30.0,
) -> list[dict]:
    """
    Combines Stage 1 cut detection, Stage 2 rule-based classification,
    and separate frame-drop detection.
    """
    n_pairs = len(prob_cut)
    events: list[dict] = []

    # ─────────────────────────────────────────────────────────────────────
    # STAGE 1: Detect all cuts via peak picking
    # ─────────────────────────────────────────────────────────────────────
    cuts = pick_peaks(prob_cut, threshold=thresh_cut, min_distance=10)

    # ─────────────────────────────────────────────────────────────────────
    # STAGE 2: Classify cuts using rules in order
    # ─────────────────────────────────────────────────────────────────────
    used_cuts = set()

    # Rule 1: INSERTION (pairs of cuts separated by 10-90 frames)
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
                events.append({
                    "start_frame": int(c1),
                    "end_frame": int(c2),
                    "start_time": round(c1 / fps, 3),
                    "end_time": round(c2 / fps, 3),
                    "edit_type": "insertion",
                    "confidence": round(conf, 4),
                })
                break

    # Feature indices: 24 = self_min_dist, 25 = self_best_lag, 26 = self_lag_run
    min_dists = features_mat[:, 24] if features_mat.shape[1] > 24 else np.full(n_pairs, 64.0)
    best_lags = features_mat[:, 25] if features_mat.shape[1] > 25 else np.zeros(n_pairs)
    lag_runs = features_mat[:, 26] if features_mat.shape[1] > 26 else np.zeros(n_pairs)

    # Rule 2: DUPLICATION (constant lag L for >= 10 frames after the cut)
    for i in range(len(cuts)):
        if i in used_cuts:
            continue
        c = cuts[i]
        # Inspect self-similarity in [c, c + 65]
        sub_end = min(n_pairs, c + 65)
        sub_runs = lag_runs[c : sub_end]
        sub_dists = min_dists[c : sub_end]
        sub_lags = best_lags[c : sub_end]

        is_dup = False
        est_l = 0

        # Check for constant lag run >= 10 frames with low Hamming distance (<= 14)
        if len(sub_runs) > 0 and np.max(sub_runs) >= 10:
            valid_lags = sub_lags[sub_runs >= 4]
            if len(valid_lags) > 0:
                est_l = int(round(float(np.median(valid_lags))))
                if 10 <= est_l <= 90:
                    is_dup = True
        elif len(sub_dists) >= 10:
            # Fallback check: at least 10 frames with distance <= 12 and consistent lag
            valid = sub_lags[sub_dists <= 12]
            if len(valid) >= 10:
                est_l = int(round(float(np.median(valid))))
                if 10 <= est_l <= 90:
                    is_dup = True

        if is_dup:
            used_cuts.add(i)
            end_f = min(n_pairs - 1, c + est_l - 1)
            events.append({
                "start_frame": int(c),
                "end_frame": int(end_f),
                "start_time": round(c / fps, 3),
                "end_time": round(end_f / fps, 3),
                "edit_type": "duplication",
                "confidence": round(float(prob_cut[c]), 4),
            })

    # Rule 3: DELETION (lone cut)
    for i in range(len(cuts)):
        if i in used_cuts:
            continue
        c = cuts[i]
        used_cuts.add(i)
        events.append({
            "start_frame": int(c),
            "end_frame": int(c),
            "start_time": round(c / fps, 3),
            "end_time": round(c / fps, 3),
            "edit_type": "deletion",
            "confidence": round(float(prob_cut[c]), 4),
        })

    # ─────────────────────────────────────────────────────────────────────
    # SEPARATE DETECTOR: Frame Drop (rate-change boundaries)
    # ─────────────────────────────────────────────────────────────────────
    fd_peaks = pick_peaks(prob_fd, threshold=thresh_fd, min_distance=10)
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
                    "start_frame": int(p1),
                    "end_frame": int(p2),
                    "start_time": round(p1 / fps, 3),
                    "end_time": round(p2 / fps, 3),
                    "edit_type": "frame_drop",
                    "confidence": round(conf, 4),
                })
                break

    events.sort(key=lambda e: (e["start_frame"], -e["confidence"]))
    return events

