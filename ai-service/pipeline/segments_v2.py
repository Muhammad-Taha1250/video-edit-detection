"""
segments_v2.py – Day 3 Event-Level Segment Post-Processor.

Post-processing logic:
  - Deletion: Single high-confidence cut peak.
  - Insertion: Two boundary junctions separated by 10 to 90 frames.
  - Duplication: Start loopback junction reinforced by self-similarity constant-lag run.
  - Frame Drop: Plateau of rate-change / paired transition step signatures.

Output segment format:
  start_frame, end_frame, start_time, end_time, edit_type, confidence
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
AI_SERVICE = _HERE.parent
sys.path.insert(0, str(AI_SERVICE))

from pipeline.features_v2 import FEATURE_NAMES_V2


def _find_peaks(prob_series: np.ndarray, threshold: float = 0.35, min_distance: int = 6) -> list[int]:
    """Finds local maxima above threshold."""
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


def postprocess_events(
    proba: np.ndarray,
    classes: list[str],
    features: np.ndarray,
    fps: float = 30.0,
    del_thresh: float = 0.30,
    ins_thresh: float = 0.25,
    dup_thresh: float = 0.25,
    fd_thresh: float = 0.25,
) -> list[dict]:
    """
    Given (N, 5) probabilities and (N, 27) features for a video,
    produces a list of predicted event segments.
    """
    n_pairs = len(proba)
    if n_pairs == 0:
        return []

    c2i = {c: i for i, c in enumerate(classes)}
    events: list[dict] = []

    # 1. Deletion (single cut peak)
    if "deletion" in c2i:
        del_prob = proba[:, c2i["deletion"]]
        del_peaks = _find_peaks(del_prob, threshold=del_thresh, min_distance=15)
        for p in del_peaks:
            conf = float(del_prob[p])
            events.append({
                "start_frame": int(p),
                "end_frame": int(p),
                "start_time": round(p / fps, 3),
                "end_time": round(p / fps, 3),
                "edit_type": "deletion",
                "confidence": round(conf, 4),
            })

    # 2. Insertion (two junctions 10-90 frames apart)
    if "insertion" in c2i:
        ins_prob = proba[:, c2i["insertion"]]
        ins_peaks = _find_peaks(ins_prob, threshold=ins_thresh, min_distance=8)
        used_peaks = set()
        for idx1 in range(len(ins_peaks)):
            if idx1 in used_peaks:
                continue
            p1 = ins_peaks[idx1]
            best_pair = None
            best_gap = 0
            for idx2 in range(idx1 + 1, len(ins_peaks)):
                if idx2 in used_peaks:
                    continue
                p2 = ins_peaks[idx2]
                gap = p2 - p1
                if 10 <= gap <= 95:
                    best_pair = idx2
                    best_gap = gap
                    break
            if best_pair is not None:
                p2 = ins_peaks[best_pair]
                used_peaks.add(idx1)
                used_peaks.add(best_pair)
                conf = float((ins_prob[p1] + ins_prob[p2]) / 2.0)
                events.append({
                    "start_frame": int(p1),
                    "end_frame": int(p2),
                    "start_time": round(p1 / fps, 3),
                    "end_time": round(p2 / fps, 3),
                    "edit_type": "insertion",
                    "confidence": round(conf, 4),
                })
            elif ins_prob[p1] >= 0.55:
                # Strong isolated insertion boundary: span to local anomaly end (~30f default)
                used_peaks.add(idx1)
                end_cand = min(n_pairs - 1, p1 + 35)
                events.append({
                    "start_frame": int(p1),
                    "end_frame": int(end_cand),
                    "start_time": round(p1 / fps, 3),
                    "end_time": round(end_cand / fps, 3),
                    "edit_type": "insertion",
                    "confidence": round(float(ins_prob[p1]), 4),
                })

    # 3. Duplication (start junction + constant-lag run)
    # self_best_lag is col 25, self_lag_run is col 26, self_min_dist is col 24
    if "duplication" in c2i:
        dup_prob = proba[:, c2i["duplication"]]
        lag_runs = features[:, 26] if features.shape[1] > 26 else np.zeros(n_pairs)
        best_lags = features[:, 25] if features.shape[1] > 25 else np.zeros(n_pairs)
        min_dists = features[:, 24] if features.shape[1] > 24 else np.zeros(n_pairs)

        dup_peaks = _find_peaks(dup_prob, threshold=dup_thresh, min_distance=20)
        # Also check where lag_runs spike >= 10 with low distance
        run_spikes = [i for i in range(n_pairs) if lag_runs[i] >= 12 and min_dists[i] <= 12]

        for p in dup_peaks:
            # Check if followed by low distance or constant lag
            sub_runs = lag_runs[p : min(n_pairs, p + 60)]
            sub_lags = best_lags[p : min(n_pairs, p + 60)]
            if len(sub_runs) > 0 and (np.max(sub_runs) >= 8 or np.min(min_dists[p : min(n_pairs, p + 60)]) <= 10):
                est_lag = int(np.median(sub_lags[sub_runs >= 4])) if any(sub_runs >= 4) else int(np.max(sub_runs))
                est_lag = max(10, min(90, est_lag))
                end_p = min(n_pairs - 1, p + est_lag)
                conf = float(dup_prob[p])
                events.append({
                    "start_frame": int(p),
                    "end_frame": int(end_p),
                    "start_time": round(p / fps, 3),
                    "end_time": round(end_p / fps, 3),
                    "edit_type": "duplication",
                    "confidence": round(conf, 4),
                })

    # 4. Frame Drop (rate-change plateau or paired transition step signatures)
    if "frame_drop" in c2i:
        fd_prob = proba[:, c2i["frame_drop"]]
        fd_peaks = _find_peaks(fd_prob, threshold=fd_thresh, min_distance=8)
        used_fd = set()
        for idx1 in range(len(fd_peaks)):
            if idx1 in used_fd:
                continue
            p1 = fd_peaks[idx1]
            for idx2 in range(idx1 + 1, len(fd_peaks)):
                if idx2 in used_fd:
                    continue
                p2 = fd_peaks[idx2]
                gap = p2 - p1
                if 12 <= gap <= 95:
                    used_fd.add(idx1)
                    used_fd.add(idx2)
                    conf = float((fd_prob[p1] + fd_prob[p2]) / 2.0)
                    events.append({
                        "start_frame": int(p1),
                        "end_frame": int(p2),
                        "start_time": round(p1 / fps, 3),
                        "end_time": round(p2 / fps, 3),
                        "edit_type": "frame_drop",
                        "confidence": round(conf, 4),
                    })
                    break

    # Sort events chronologically by start_frame
    events.sort(key=lambda e: (e["start_frame"], -e["confidence"]))
    return events

