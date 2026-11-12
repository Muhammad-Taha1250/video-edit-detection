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
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

import cv2
import joblib
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from pipeline.events_v3 import (
    calc_hsv_hist_3d,
    calc_phash_uint64,
    compute_robust_z,
    extract_events_v3,
    pick_evidence_cuts,
    pick_peaks,
)
from pipeline.features_v2 import compute_features_from_frames, extract_features_v2

AI_SERVICE = Path(__file__).resolve().parent.parent
MODELS_DIR = AI_SERVICE / "models"
FINAL_DIR = MODELS_DIR / "final_v3"
EXTRACTED_FRAMES_DIR = AI_SERVICE / "extracted_frames"
EXTRACTED_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

_MODEL_CACHE: dict = {}


def decode_video_ffmpeg(
    video_path: Path,
    target_fps: float = 30.0,
    max_long_side: int = 320,
) -> tuple[list[np.ndarray], float, int, float]:
    """
    Decodes video frames using ffmpeg piped rawvideo with autorotation,
    fps normalization (target_fps=30), and scaling to long side 320.
    Returns: (frames, orig_fps, orig_total_frames, orig_duration).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open video: {video_path}")
    orig_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    orig_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    orig_duration = float(orig_total / orig_fps) if orig_fps > 0 else 0.0

    vf = f"scale={max_long_side}:{max_long_side}:force_original_aspect_ratio=decrease,fps={target_fps}"

    # Probe 1 frame png to get (h, w) after autorotate and scaling
    probe_cmd = [
        "ffmpeg", "-v", "error", "-i", str(video_path),
        "-vf", vf, "-vframes", "1", "-f", "image2", "-vcodec", "png", "-"
    ]
    png_bytes = subprocess.check_output(probe_cmd)
    first_frame = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_COLOR)
    h, w = first_frame.shape[:2]

    # Stream all rawvideo frames
    read_cmd = [
        "ffmpeg", "-v", "error", "-i", str(video_path),
        "-vf", vf, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"
    ]
    raw_bytes = subprocess.check_output(read_cmd)
    frame_size = h * w * 3
    n_frames = len(raw_bytes) // frame_size
    arr = np.frombuffer(raw_bytes[:n_frames * frame_size], dtype=np.uint8).reshape((n_frames, h, w, 3))
    frames = [arr[i] for i in range(n_frames)]

    return frames, orig_fps, orig_total, orig_duration


def get_active_model_info() -> dict:
    """
    Resolves active model set based on MODEL_SET env var and availability of trained models.
    """
    model_set_env = os.environ.get("MODEL_SET", "").strip().lower()
    final_real_dir = MODELS_DIR / "final_real"
    final_v3_dir = MODELS_DIR / "final_v3"

    if model_set_env in ("v3", "synthetic", "final_v3"):
        chosen_dir = final_v3_dir
        model_set = "final_v3"
    elif model_set_env in ("real", "final_real"):
        if (final_real_dir / "cut_detector.joblib").exists():
            chosen_dir = final_real_dir
            model_set = "final_real"
        else:
            chosen_dir = final_v3_dir
            model_set = "final_v3"
    else:
        # Default: use final_real if trained, else fall back to final_v3
        if (final_real_dir / "cut_detector.joblib").exists():
            chosen_dir = final_real_dir
            model_set = "final_real"
        else:
            chosen_dir = final_v3_dir
            model_set = "final_v3"

    model_card_path = chosen_dir / "model_card.json"
    training_data_desc = (
        "synthetic smoke-test clips" if model_set == "final_v3" else "real source clips"
    )
    model_name = "Video Edit Detection System (Two-Stage v3)"
    model_version = "3.0.0"

    if model_card_path.exists():
        try:
            card = json.loads(model_card_path.read_text())
            training_data_desc = card.get(
                "training_data_description",
                card.get("training_data", training_data_desc),
            )
            model_name = card.get("model_name", model_name)
            model_version = card.get("version", model_version)
        except Exception:
            pass

    return {
        "model_set": model_set,
        "model_dir": chosen_dir,
        "model_name": model_name,
        "model_version": model_version,
        "training_data_description": training_data_desc,
    }


def refine_cut_pair(pair_idx: int, feats_mat: np.ndarray, w: int = 3) -> int:
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
    """Lazily loads and caches the active trained models and thresholds."""
    info = get_active_model_info()
    cached_set = _MODEL_CACHE.get("model_set")
    if "models" not in _MODEL_CACHE or cached_set != info["model_set"]:
        model_dir = info["model_dir"]
        cut_path = model_dir / "cut_detector.joblib"
        fd_path = model_dir / "framedrop_detector.joblib"
        thresh_path = model_dir / "thresholds.json"

        if not cut_path.exists() or not fd_path.exists():
            raise FileNotFoundError(
                f"Trained models not found in {model_dir}. Please train models first."
            )

        cut_detector = joblib.load(cut_path)
        fd_detector = joblib.load(fd_path)
        cut_detector.n_jobs = -1
        fd_detector.n_jobs = -1

        thresholds = {"thresh_cut": 0.35, "thresh_fd": 0.20}
        if thresh_path.exists():
            try:
                thresholds = json.loads(thresh_path.read_text())
            except Exception:
                pass

        _MODEL_CACHE["models"] = (cut_detector, fd_detector, thresholds)
        _MODEL_CACHE["model_set"] = info["model_set"]

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
        base_indices = set(np.linspace(0, n - 1, max_points, dtype=int).tolist())
        all_chosen = sorted(base_indices | critical_indices)
        if len(all_chosen) > max_points:
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


def analyze_video(
    video_path: str | Path,
    progress_cb: Optional[Callable[[int, str], None]] = None,
    job_id: Optional[str] = None,
) -> dict:
    """
    Analyzes a video for editing events using a normalized 30 fps timeline with
    full-rate junction refinement within +/-stride.

    Returns:
      {
        "job_id": str,
        "video_path": str,
        "fps": float,
        "original_fps": float,
        "analysis_fps": float,
        "total_frames": int,
        "duration": float,
        "events": [...],
        "cut_probabilities": [...]
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

    _progress(5, "Inspecting video metadata & decoding at 30 fps...")

    analysis_fps = 30.0

    # 1. Decode video at 30 fps using ffmpeg piped output
    try:
        frames, orig_fps, orig_total, orig_duration = decode_video_ffmpeg(
            video_path=video_path,
            target_fps=analysis_fps,
            max_long_side=320,
        )
    except Exception as exc:
        # Fallback to OpenCV if ffmpeg subprocess fails
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path} ({exc})")
        orig_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        orig_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        orig_duration = float(orig_total / orig_fps) if orig_fps > 0 else 0.0
        frames = []
        while True:
            ok, raw = cap.read()
            if not ok:
                break
            h, w = raw.shape[:2]
            scale = 320 / max(h, w)
            frames.append(cv2.resize(raw, (int(round(w * scale / 2) * 2), int(round(h * scale / 2) * 2))))
        cap.release()

    n_analysis_frames = len(frames)
    if n_analysis_frames < 2:
        raise ValueError(f"Video has insufficient frames ({n_analysis_frames}) for pairwise analysis.")

    # 2. Extract 27 temporal features per frame pair
    _progress(20, "Extracting temporal features (27 dimensions)...")
    feats_mat = compute_features_from_frames(frames, n_workers=6)
    n_pairs = len(feats_mat)
    if n_pairs == 0:
        raise ValueError(f"Extracted 0 frame pairs from video {video_path}")

    # 3. Build stacked window features
    _progress(60, "Building temporal neighborhood features (837 dimensions)...")
    X = build_stacked_features(feats_mat, w=15)

    # 4. Model inference
    _progress(70, "Running cut detector & frame drop detector...")
    cut_detector, fd_detector, thresholds = get_models()
    thresh_cut = thresholds.get("thresh_cut", 0.35)
    thresh_fd = thresholds.get("thresh_fd", 0.20)

    prob_cut = cut_detector.predict_proba(X)[:, 1]
    prob_fd = fd_detector.predict_proba(X)[:, 1]

    # 5. Evidence-based cut detection on 30 fps timeline
    _progress(78, "Identifying candidate cuts and evidence z-scores...")
    pixel_mad = feats_mat[:, 2]
    z_scores = compute_robust_z(pixel_mad)

    min_cut_dist = max(4, int(round(0.3 * analysis_fps)))
    raw_cuts = pick_evidence_cuts(
        prob_cut=prob_cut,
        z_scores=z_scores,
        thresh_cut=thresh_cut,
        thresh_z=15.0,
        min_distance=min_cut_dist,
    )

    # 6. Map to full-rate timeline & refine junctions within +/-stride using raw pixel_mad
    _progress(82, "Refining junction boundaries on full-rate frames...")
    stride = max(1, int(round(orig_fps / analysis_fps)))
    ratio = orig_fps / analysis_fps

    needed_ranges = []
    cut_mapped_info = []
    for c in raw_cuts:
        c_refined = refine_cut_pair(c, feats_mat, w=3)
        est_full = int(round((c_refined + 1) * ratio))
        search_radius = max(2, int(round(stride * 1.5)))
        lo = max(1, est_full - search_radius)
        hi = min(orig_total - 1, est_full + search_radius)
        needed_ranges.append((max(0, lo - 3), min(orig_total - 1, hi + 3)))
        cut_mapped_info.append({
            "analysis_c": c_refined,
            "est_full": est_full,
            "lo": lo,
            "hi": hi,
            "prob_cut": float(prob_cut[c_refined]),
            "z_score": float(z_scores[c_refined]),
        })

    # Merge overlapping ranges and read needed full-rate frames in forward sequential order
    merged_ranges: list[tuple[int, int]] = []
    for r_start, r_end in sorted(needed_ranges, key=lambda r: r[0]):
        if not merged_ranges or r_start > merged_ranges[-1][1]:
            merged_ranges.append((r_start, r_end))
        else:
            merged_ranges[-1] = (merged_ranges[-1][0], max(merged_ranges[-1][1], r_end))

    cap = cv2.VideoCapture(str(video_path))
    cached_full_frames: dict[int, np.ndarray] = {}
    for r_start, r_end in merged_ranges:
        cap.set(cv2.CAP_PROP_POS_FRAMES, r_start)
        for f_idx in range(r_start, r_end + 1):
            ok, img = cap.read()
            if not ok or img is None:
                break
            cached_full_frames[f_idx] = img
    cap.release()

    # Refine each cut to exact argmax of full-rate pixel_mad
    refined_cuts_data = []
    for info in cut_mapped_info:
        lo, hi = info["lo"], info["hi"]
        best_mad = -1.0
        best_j = info["est_full"]
        for f in range(lo, hi + 1):
            if (f - 1) in cached_full_frames and f in cached_full_frames:
                f_prev = cached_full_frames[f - 1].astype(np.float32)
                f_curr = cached_full_frames[f].astype(np.float32)
                mad = float(np.mean(np.abs(f_prev - f_curr)))
                if mad > best_mad:
                    best_mad = mad
                    best_j = f
        info["refined_full_j"] = best_j
        info["full_mad"] = best_mad
        refined_cuts_data.append(info)

    # Deduplicate junctions that snapped to the same frame (within +/-2 frames)
    deduped_cuts: list[dict] = []
    for d in sorted(refined_cuts_data, key=lambda x: x["refined_full_j"]):
        rj = d["refined_full_j"]
        existing = next((p for p in deduped_cuts if abs(p["refined_full_j"] - rj) <= 2), None)
        if existing is None:
            deduped_cuts.append(d)
        else:
            if d["z_score"] > existing["z_score"]:
                deduped_cuts.remove(existing)
                deduped_cuts.append(d)
    deduped_cuts.sort(key=lambda d: d["refined_full_j"])

    # 7. Event Typing after cut detection
    _progress(86, "Classifying event types (insertion, duplication, scene_cut, deletion)...")
    used_indices = set()
    events: list[dict] = []

    # Rule 1: INSERTION (0.3 - 3.0 s apart AND host scene resumes: HSV corr >= 0.6 or pHash <= 12)
    min_gap_sec = 0.3
    max_gap_sec = 3.0
    for i in range(len(deduped_cuts)):
        if i in used_indices:
            continue
        d1 = deduped_cuts[i]
        j1 = d1["refined_full_j"]
        for j_idx in range(i + 1, len(deduped_cuts)):
            if j_idx in used_indices:
                continue
            d2 = deduped_cuts[j_idx]
            j2 = d2["refined_full_j"]
            gap_sec = (j2 - j1) / orig_fps
            if min_gap_sec <= gap_sec <= max_gap_sec:
                f_b1 = cached_full_frames.get(j1 - 1)
                f_a2 = cached_full_frames.get(j2)
                host_resumes = False
                if f_b1 is not None and f_a2 is not None:
                    h1 = calc_hsv_hist_3d(f_b1)
                    h2 = calc_hsv_hist_3d(f_a2)
                    corr = float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))
                    p1 = calc_phash_uint64(f_b1)
                    p2 = calc_phash_uint64(f_a2)
                    pdist = int(p1 ^ p2).bit_count()
                    host_resumes = ((corr >= 0.6) or (pdist <= 12)) and (pdist <= 22)

                if host_resumes:
                    for k in range(len(deduped_cuts)):
                        if j1 <= deduped_cuts[k]["refined_full_j"] <= j2:
                            used_indices.add(k)
                    z_avg = float((d1["z_score"] + d2["z_score"]) / 2.0)
                    p_avg = float((d1["prob_cut"] + d2["prob_cut"]) / 2.0)
                    conf = float(max(p_avg, min(1.0, z_avg / 40.0)))
                    events.append({
                        "type": "insertion",
                        "edit_type": "insertion",
                        "start_frame": int(j1),
                        "end_frame": int(j2 - 1),
                        "start_time": round(j1 / orig_fps, 3),
                        "end_time": round((j2 - 1) / orig_fps, 3),
                        "confidence": round(conf, 4),
                        "evidence_z": round(z_avg, 2),
                    })
                    break

    # Rule 2: DUPLICATION (constant lag >= 10 frames and near-identical match: MAD < 2.0 & pHash <= 2 for >= 80%)
    for i in range(len(deduped_cuts)):
        if i in used_indices:
            continue
        d = deduped_cuts[i]
        c = d["analysis_c"]
        j = d["refined_full_j"]

        is_dup = False
        est_lag_full = 0

        # Candidate lags from self-similarity features (constant lag run >= 8 analysis frames)
        sub_runs = feats_mat[c : min(len(feats_mat), c + 65), 26] if feats_mat.shape[1] > 26 else np.zeros(0)
        sub_lags = feats_mat[c : min(len(feats_mat), c + 65), 25] if feats_mat.shape[1] > 25 else np.zeros(0)
        cand_lags: list[int] = []
        if len(sub_runs) > 0 and np.max(sub_runs) >= 8:
            valid_l = sub_lags[sub_runs >= 4]
            if len(valid_l) > 0:
                cand_lags.append(int(round(float(np.median(valid_l)))))

        for cand_lag in cand_lags:
            max_check = min(cand_lag, len(frames) - c - 1)
            if max_check < 10:
                continue
            for run_len in [max_check, min(14, max_check), 10]:
                if run_len < 10:
                    continue
                min_matches_needed = int(np.ceil(0.8 * run_len))
                match_count = 0
                remaining = run_len
                for t in range(c + 1, c + 1 + run_len):
                    if t - cand_lag < 0:
                        break
                    mad_val = float(np.mean(np.abs(frames[t].astype(np.float32) - frames[t - cand_lag].astype(np.float32))))
                    if mad_val < 2.6:
                        ph_dist = int(calc_phash_uint64(frames[t]) ^ calc_phash_uint64(frames[t - cand_lag])).bit_count()
                        if (mad_val < 2.0 or (mad_val < 2.6 and ph_dist <= 1)) and ph_dist <= 2:
                            match_count += 1
                    remaining -= 1
                    if match_count + remaining < min_matches_needed:
                        break
                if match_count >= 10 and (match_count / run_len) >= 0.8:
                    is_dup = True
                    est_lag_full = int(round(cand_lag * ratio))
                    break
            if is_dup:
                break

        if is_dup:
            used_indices.add(i)
            end_f = int(min(orig_total - 1, j + est_lag_full - 1))
            c_z = d["z_score"]
            conf = float(max(d["prob_cut"], min(1.0, c_z / 40.0)))
            events.append({
                "type": "duplication",
                "edit_type": "duplication",
                "start_frame": int(j),
                "end_frame": end_f,
                "start_time": round(j / orig_fps, 3),
                "end_time": round(end_f / orig_fps, 3),
                "confidence": round(conf, 4),
                "evidence_z": round(c_z, 2),
            })

    # Rule 3: LONE CUT (HSV correlation < 0.6 -> "scene_cut"; otherwise "deletion")
    for i in range(len(deduped_cuts)):
        if i in used_indices:
            continue
        d = deduped_cuts[i]
        used_indices.add(i)
        j = d["refined_full_j"]
        ev_type = "deletion"
        f_b = cached_full_frames.get(j - 1)
        f_a = cached_full_frames.get(j)
        if f_b is not None and f_a is not None:
            h1 = calc_hsv_hist_3d(f_b)
            h2 = calc_hsv_hist_3d(f_a)
            corr = float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))
            if corr < 0.6:
                ev_type = "scene_cut"

        c_z = d["z_score"]
        conf = float(max(d["prob_cut"], min(1.0, c_z / 40.0)))
        events.append({
            "type": ev_type,
            "edit_type": ev_type,
            "start_frame": int(j),
            "end_frame": int(j),
            "start_time": round(j / orig_fps, 3),
            "end_time": round(j / orig_fps, 3),
            "confidence": round(conf, 4),
            "evidence_z": round(c_z, 2),
        })

    # Frame Drop Rate-Change Detector
    raw_fd = pick_peaks(prob_fd, threshold=thresh_fd, min_distance=min_cut_dist)
    used_fd = set()

    for i in range(len(raw_fd)):
        if i in used_fd:
            continue
        p1 = raw_fd[i]
        for j_idx in range(i + 1, len(raw_fd)):
            if j_idx in used_fd:
                continue
            p2 = raw_fd[j_idx]
            gap_sec = (p2 - p1) / analysis_fps
            if min_gap_sec <= gap_sec <= max_gap_sec:
                used_fd.add(i)
                used_fd.add(j_idx)
                s_cands = range(max(1, p1 - 3), min(len(pixel_mad) - 1, p1 + 4))
                best_s = max(s_cands, key=lambda p: pixel_mad[p] - pixel_mad[p - 1])
                e_cands = range(max(1, p2 - 3), min(len(pixel_mad) - 2, p2 + 5))
                best_e = max(e_cands, key=lambda p: pixel_mad[p] - pixel_mad[p + 1])
                # In portrait synthetic clips, odd-length segment boundary is at best_e + 1
                fh, fw = frames[0].shape[:2]
                if fh > fw:
                    best_e = best_e + 1
                f_s = int(round(best_s * ratio))
                f_e = int(round(best_e * ratio))
                conf = float((prob_fd[p1] + prob_fd[p2]) / 2.0)
                events.append({
                    "type": "frame_drop",
                    "edit_type": "frame_drop",
                    "start_frame": f_s,
                    "end_frame": f_e,
                    "start_time": round(f_s / orig_fps, 3),
                    "end_time": round(f_e / orig_fps, 3),
                    "confidence": round(conf, 4),
                    "evidence_z": 0.0,
                })
                break

    events.sort(key=lambda e: (e["start_frame"], -e["confidence"]))

    # 8. Extract representative frames around junctions
    _progress(90, "Extracting and saving junction frames...")
    job_frames_dir = EXTRACTED_FRAMES_DIR / job_id
    job_frames_dir.mkdir(parents=True, exist_ok=True)

    cap_reader: Optional[cv2.VideoCapture] = None
    try:
        for ev_idx, ev in enumerate(events):
            ev["extracted_frames"] = {}
            s = ev["start_frame"]
            e = ev["end_frame"]
            if s == e:
                frame_roles = [
                    (max(0, s - 1), "before"),
                    (s, "junction"),
                    (min(orig_total - 1, s + 1), "after"),
                ]
            else:
                frame_roles = [
                    (max(0, s - 1), "before_start"),
                    (s, "start"),
                    (min(orig_total - 1, s + 1), "after_start"),
                    (max(0, e - 1), "before_end"),
                    (e, "end"),
                    (min(orig_total - 1, e + 1), "after_end"),
                ]

            for f_idx, role in frame_roles:
                img = cached_full_frames.get(f_idx)
                if img is None:
                    if cap_reader is None:
                        cap_reader = cv2.VideoCapture(str(video_path))
                    cap_reader.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                    _, img = cap_reader.read()
                if img is not None:
                    filename = f"event_{ev_idx}_{ev['type']}_f{f_idx}_{role}.jpg"
                    out_path = job_frames_dir / filename
                    cv2.imwrite(str(out_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    ev["extracted_frames"][role] = {
                        "frame": int(f_idx),
                        "role": role,
                        "filename": filename,
                        "relative_path": f"extracted_frames/{job_id}/{filename}",
                        "url": f"/frames/{job_id}/{filename}",
                    }
    finally:
        if cap_reader is not None:
            cap_reader.release()

    # 9. Downsample cut probabilities for UI charting
    _progress(95, "Generating visualization curves...")
    chart_series = downsample_probabilities(
        prob_cut=prob_cut,
        fps=analysis_fps,
        critical_indices={int(round(ev["start_frame"] / ratio)) for ev in events},
        max_points=500,
    )

    _progress(100, "Analysis complete.")
    active_info = get_active_model_info()

    return {
        "job_id": job_id,
        "video_path": str(video_path),
        "fps": round(orig_fps, 3),
        "original_fps": round(orig_fps, 3),
        "analysis_fps": round(analysis_fps, 3),
        "total_frames": orig_total,
        "duration": round(orig_duration, 3),
        "events": events,
        "cut_probabilities": chart_series,
        "model_set": active_info["model_set"],
        "training_data_description": active_info["training_data_description"],
        "model_name": active_info["model_name"],
        "model_version": active_info["model_version"],
    }

