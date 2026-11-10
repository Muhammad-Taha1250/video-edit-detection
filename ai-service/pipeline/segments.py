"""
segments.py – Convert per-frame-pair predictions into edit segments.

Algorithm:
  1. Identify runs of consecutive non-"none" predictions.
  2. Drop isolated blips shorter than MIN_RUN frames.
  3. Merge adjacent runs with a gap <= MERGE_GAP frames.
  4. For each surviving segment: dominant class, mean confidence, timestamps.

Exported function: predict_segments(video_path, rf, le, fps) -> list[dict]
Also usable as a module: segments.py <video_path>
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
AI_SERVICE = _HERE.parent
sys.path.insert(0, str(AI_SERVICE))

from pipeline.features import extract_features

MIN_RUN    = 2    # min consecutive flagged pairs to keep
MERGE_GAP  = 4    # merge runs separated by <= this many "none" frames


def _runs_of_positives(labels: np.ndarray) -> list[tuple[int, int]]:
    """Return list of (start, end) inclusive runs where labels != 'none'."""
    runs: list[tuple[int, int]] = []
    n = len(labels)
    i = 0
    while i < n:
        if labels[i] != "none":
            j = i
            while j < n and labels[j] != "none":
                j += 1
            runs.append((i, j - 1))
            i = j
        else:
            i += 1
    return runs


def _merge_runs(
    runs: list[tuple[int, int]], gap: int
) -> list[tuple[int, int]]:
    if not runs:
        return []
    merged = [runs[0]]
    for start, end in runs[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end - 1 <= gap:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def predict_segments(
    video_path: str | Path,
    rf,
    le,
    fps: float = 30.0,
    min_run: int = MIN_RUN,
    merge_gap: int = MERGE_GAP,
) -> list[dict]:
    """
    Returns a list of segment dicts:
      start_frame, end_frame, start_time, end_time, edit_type, confidence
    """
    video_path = Path(video_path)
    feats = extract_features(video_path)
    if len(feats) == 0:
        return []

    proba  = rf.predict_proba(feats)          # (n_pairs, n_classes)
    pred_i = np.argmax(proba, axis=1)
    conf   = np.max(proba, axis=1)
    pred_labels = le.inverse_transform(pred_i)

    runs = _runs_of_positives(pred_labels)
    runs = [(s, e) for s, e in runs if (e - s + 1) >= min_run]
    runs = _merge_runs(runs, merge_gap)

    segments: list[dict] = []
    for start, end in runs:
        seg_labels = pred_labels[start: end + 1]
        seg_conf   = conf[start: end + 1]

        # Dominant class (excluding "none")
        from collections import Counter
        cnt = Counter(l for l in seg_labels if l != "none")
        if not cnt:
            continue
        dom_class = cnt.most_common(1)[0][0]
        mean_conf = float(np.mean(seg_conf))

        segments.append({
            "start_frame": int(start),
            "end_frame":   int(end),
            "start_time":  round(start / fps, 3),
            "end_time":    round(end / fps, 3),
            "edit_type":   dom_class,
            "confidence":  round(mean_conf, 4),
        })

    return segments


if __name__ == "__main__":
    import joblib, json
    _HERE2 = Path(__file__).resolve().parent.parent
    rf_path = _HERE2 / "models" / "rf.joblib"
    if not rf_path.exists():
        print("No trained model found. Run train_rf.py first.")
        sys.exit(1)
    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.segments <video_path>")
        sys.exit(1)
    _rf, _le = joblib.load(rf_path)
    segs = predict_segments(sys.argv[1], _rf, _le)
    print(json.dumps(segs, indent=2))

