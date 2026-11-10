"""
eval_loso.py – Leave-One-Source-Out (LOSO) Cross-Validation & Threshold Optimization.

Pipeline:
  1. Leave-One-Source-Out CV over all 10 source_file groups for Random Forest v2
     (stacked +/-15 neighborhood of 27 features = 837 dimensions).
  2. Collect out-of-fold (OOF) per-pair probabilities for every video.
  3. Optimize per-class probability thresholds on OOF predictions to maximize event-level F1.
  4. Peak picking event extraction: local maxima, minimum distance 10 pairs, above chosen threshold.
  5. Report event-level (+/-5 frames) metrics pooled over all 10 folds:
     - Per-class precision, recall, F1
     - Macro F1
     - False positives per clean video
     - Video-level edited-vs-clean accuracy
     - Mean boundary error (frames)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

AI_SERVICE = Path(__file__).resolve().parent.parent
MODELS_DIR = AI_SERVICE / "models"
GENERATED_DIR = AI_SERVICE / "dataset" / "generated"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_TABLE_PATH = MODELS_DIR / "feature_table_v2.csv"
LABELS_CSV_PATH = GENERATED_DIR / "labels.csv" if (GENERATED_DIR / "labels.csv").exists() else AI_SERVICE / "dataset" / "labels.csv"
OOF_PROBS_PATH = MODELS_DIR / "loso_oof_probs.npy"
LOSO_RESULTS_JSON = MODELS_DIR / "loso_metrics.json"

STACK_W = 15
TOLERANCE_EVENT = 5
MIN_PEAK_DIST = 10
CLASSES = ["deletion", "duplication", "frame_drop", "insertion", "none"]
EDIT_CLASSES = ["deletion", "duplication", "frame_drop", "insertion"]


def load_data():
    table = pd.read_csv(FEATURE_TABLE_PATH)
    labels_df = pd.read_csv(LABELS_CSV_PATH)
    return table, labels_df


def build_stacked_features(table: pd.DataFrame, w: int = STACK_W) -> np.ndarray:
    # 27 feature columns
    feat_cols = [c for c in table.columns if c not in ("video", "source_file", "pair_idx", "label")]
    stacked_list = []
    d = len(feat_cols)

    for _, grp in table.groupby("video", sort=False):
        mat = grp[feat_cols].values.astype(np.float32)
        n = len(mat)
        padded = np.pad(mat, ((w, w), (0, 0)), mode="edge")
        windows = sliding_window_view(padded, (2 * w + 1, d))[:, 0, :, :]
        stacked = windows.reshape(n, (2 * w + 1) * d)
        stacked_list.append(stacked)

    return np.vstack(stacked_list).astype(np.float32)


# =========================================================================
# Peak Picking & Event Extraction
# =========================================================================
def pick_peaks(prob_series: np.ndarray, threshold: float, min_distance: int = MIN_PEAK_DIST) -> list[int]:
    """Finds local maxima strictly above threshold with min distance separation."""
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


def extract_events_for_class(
    c: str,
    prob_series: np.ndarray,
    threshold: float,
    features_mat: np.ndarray,
    fps: float = 30.0,
) -> list[dict]:
    n_pairs = len(prob_series)
    peaks = pick_peaks(prob_series, threshold=threshold, min_distance=MIN_PEAK_DIST)
    events = []

    if c == "deletion":
        # Each peak is a single cut junction
        for p in peaks:
            events.append({
                "start_frame": int(p),
                "end_frame": int(p),
                "edit_type": "deletion",
                "confidence": float(prob_series[p]),
            })

    elif c == "duplication":
        # Col 25 is self_best_lag, col 24 is self_min_dist
        best_lags = features_mat[:, 25] if features_mat.shape[1] > 25 else np.zeros(n_pairs)
        min_dists = features_mat[:, 24] if features_mat.shape[1] > 24 else np.zeros(n_pairs)

        for p in peaks:
            sub_lags = best_lags[p : min(n_pairs, p + 65)]
            sub_dists = min_dists[p : min(n_pairs, p + 65)]
            valid = sub_lags[sub_dists <= 14]
            if len(valid) >= 4:
                est_lag = int(round(float(np.median(valid))))
            else:
                est_lag = 35  # fallback
            est_lag = max(10, min(90, est_lag))
            events.append({
                "start_frame": int(p),
                "end_frame": int(min(n_pairs - 1, p + est_lag)),
                "edit_type": "duplication",
                "confidence": float(prob_series[p]),
            })

    elif c in ("insertion", "frame_drop"):
        # Look for pairs of peaks separated by 10 to 95 frames
        used = set()
        for i in range(len(peaks)):
            if i in used:
                continue
            p1 = peaks[i]
            paired = False
            for j in range(i + 1, len(peaks)):
                if j in used:
                    continue
                p2 = peaks[j]
                gap = p2 - p1
                if 10 <= gap <= 95:
                    used.add(i)
                    used.add(j)
                    events.append({
                        "start_frame": int(p1),
                        "end_frame": int(p2),
                        "edit_type": c,
                        "confidence": float((prob_series[p1] + prob_series[p2]) / 2.0),
                    })
                    paired = True
                    break
            if not paired and prob_series[p1] >= max(threshold * 1.3, 0.35):
                # High confidence single boundary: span 35 frames
                used.add(i)
                events.append({
                    "start_frame": int(p1),
                    "end_frame": int(min(n_pairs - 1, p1 + 35)),
                    "edit_type": c,
                    "confidence": float(prob_series[p1]),
                })

    return events


# =========================================================================
# Event Matching & Evaluation Logic
# =========================================================================
def evaluate_class_events(
    c: str,
    predicted_events_by_video: dict[str, list[dict]],
    labels_df: pd.DataFrame,
) -> tuple[float, float, float, int, int, int, list[float]]:
    tp = 0
    fp = 0
    fn = 0
    boundary_errors = []

    for _, row in labels_df.iterrows():
        vname = Path(row["output_file"]).name
        gt_type = str(row["edit_type"])
        gt_start = int(row["start_frame"])
        gt_end = int(row["end_frame"])

        preds = [e for e in predicted_events_by_video.get(vname, []) if e["edit_type"] == c]

        if gt_type != c:
            # Any prediction of class c in a video of different type is a False Positive
            fp += len(preds)
            continue

        # Video is of class c: exactly 1 ground truth event
        matched = False
        for p in preds:
            ps = p["start_frame"]
            pe = p["end_frame"]
            if c == "deletion":
                err = abs(ps - gt_start)
                if err <= TOLERANCE_EVENT and not matched:
                    matched = True
                    tp += 1
                    boundary_errors.append(err)
                else:
                    fp += 1
            else:
                err_s = abs(ps - gt_start)
                err_e = abs(pe - gt_end)
                if err_s <= TOLERANCE_EVENT and err_e <= TOLERANCE_EVENT and not matched:
                    matched = True
                    tp += 1
                    boundary_errors.append((err_s + err_e) / 2.0)
                else:
                    fp += 1

        if not matched:
            fn += 1

    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1, tp, fp, fn, boundary_errors


# =========================================================================
# Main LOSO Cross-Validation & Analysis
# =========================================================================
def run_loso(force: bool = False):
    table, labels_df = load_data()
    unique_sources = sorted(table["source_file"].unique().tolist())
    n_sources = len(unique_sources)
    print(f"\nLoaded {len(table):,} rows across {n_sources} source groups.")

    feat_cols = [c for c in table.columns if c not in ("video", "source_file", "pair_idx", "label")]

    # Compute or load OOF probabilities
    if OOF_PROBS_PATH.exists() and not force:
        print(f"Loading cached Out-Of-Fold probabilities from {OOF_PROBS_PATH} …")
        oof_probs = np.load(OOF_PROBS_PATH)
        le = LabelEncoder()
        le.fit(CLASSES)
    else:
        print(f"Building +/-{STACK_W} stacked temporal features (837 dimensions) …")
        t0 = time.time()
        X = build_stacked_features(table, w=STACK_W)
        print(f"✓ Stacked features ready ({X.shape}) in {time.time() - t0:.2f}s")

        le = LabelEncoder()
        le.fit(CLASSES)
        y = le.transform(table["label"].values)
        sources = table["source_file"].values

        oof_probs = np.zeros((len(table), len(CLASSES)), dtype=np.float32)

        print(f"\nRunning Leave-One-Source-Out (LOSO) Cross-Validation over {n_sources} folds:")
        t0_loso = time.time()

        for fold_idx, held_out in enumerate(unique_sources, 1):
            train_mask = (sources != held_out)
            test_mask = (sources == held_out)

            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=18,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            )
            t0_f = time.time()
            rf.fit(X[train_mask], y[train_mask])
            probs = rf.predict_proba(X[test_mask])

            # Align classes (c_idx are integers 0..4)
            for ci, c_idx in enumerate(rf.classes_):
                oof_probs[test_mask, int(c_idx)] = probs[:, ci]

            f_time = time.time() - t0_f
            print(f"  Fold {fold_idx:02d}/{n_sources:02d} — Held-out: {held_out:<28} | {f_time:.1f}s")

        print(f"✓ LOSO CV complete in {time.time() - t0_loso:.1f}s")
        np.save(OOF_PROBS_PATH, oof_probs)

    # Attach OOF probabilities to table groups per video
    video_oof_data = {}
    row_offset = 0
    for vname, grp in table.groupby("video", sort=False):
        n = len(grp)
        v_probs = oof_probs[row_offset : row_offset + n]
        v_feats = grp[feat_cols].values.astype(np.float32)
        video_oof_data[vname] = (v_probs, v_feats)
        row_offset += n

    # =====================================================================
    # 2. Threshold Optimization on OOF Predictions
    # =====================================================================
    candidate_thresholds = [0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35, 0.40]
    best_thresholds = {}
    best_class_stats = {}

    print("\n" + "=" * 70)
    print("STEP 2: OPTIMIZING PER-CLASS PROBABILITY THRESHOLDS (ON OOF PREDICTIONS)")
    print("=" * 70)

    for c in EDIT_CLASSES:
        ci = le.transform([c])[0]
        best_f1 = -1.0
        best_t = 0.20
        best_stats = None

        for thresh in candidate_thresholds:
            pred_events_by_vid = {}
            for vname, (v_probs, v_feats) in video_oof_data.items():
                evs = extract_events_for_class(c, v_probs[:, ci], threshold=thresh, features_mat=v_feats)
                pred_events_by_vid[vname] = evs

            p, r, f1, tp, fp, fn, errs = evaluate_class_events(c, pred_events_by_vid, labels_df)
            if f1 > best_f1 or (f1 == best_f1 and p > (best_stats[0] if best_stats else 0)):
                best_f1 = f1
                best_t = thresh
                best_stats = (p, r, f1, tp, fp, fn, errs)

        best_thresholds[c] = best_t
        best_class_stats[c] = best_stats
        p, r, f1, tp, fp, fn, errs = best_stats
        b_err = f"{np.mean(errs):.1f}f" if errs else "N/A"
        print(f"  {c:<15}: Optimal Threshold = {best_t:.2f}  |  P={p:.3f}  R={r:.3f}  F1={f1:.3f} (TP={tp}, FP={fp}, FN={fn})  |  Mean Err: {b_err}")

    # =====================================================================
    # 3. Overall Pooled Evaluation using Optimized Thresholds
    # =====================================================================
    pooled_events_by_video = {vname: [] for vname in video_oof_data}
    for c in EDIT_CLASSES:
        ci = le.transform([c])[0]
        thresh = best_thresholds[c]
        for vname, (v_probs, v_feats) in video_oof_data.items():
            evs = extract_events_for_class(c, v_probs[:, ci], threshold=thresh, features_mat=v_feats)
            pooled_events_by_video[vname].extend(evs)

    # Compute overall metrics across all 100 videos
    clean_vids = 0
    clean_fps = 0
    video_correct = 0
    total_videos = len(labels_df)

    for _, row in labels_df.iterrows():
        vname = Path(row["output_file"]).name
        gt_type = str(row["edit_type"])
        is_clean_gt = gt_type in ("none", "unedited")
        preds = pooled_events_by_video.get(vname, [])
        is_clean_pred = (len(preds) == 0)

        if is_clean_gt == is_clean_pred:
            video_correct += 1

        if is_clean_gt:
            clean_vids += 1
            clean_fps += len(preds)

    macro_f1 = float(np.mean([best_class_stats[c][2] for c in EDIT_CLASSES]))
    fp_per_clean = float(clean_fps / clean_vids) if clean_vids > 0 else 0.0
    video_accuracy = float(video_correct / total_videos)

    all_boundary_errors = []
    for c in EDIT_CLASSES:
        all_boundary_errors.extend(best_class_stats[c][6])
    mean_b_err = float(np.mean(all_boundary_errors)) if all_boundary_errors else None

    # Print Clean Table
    print("\n" + "=" * 78)
    print("LOSO CROSS-VALIDATION RESULTS (Random Forest v2, Pooled over all 10 Folds)")
    print("Headline Metric: Event-Level Detection (+/-5 frame tolerance)")
    print("=" * 78)
    print(f"  Macro Event-Level F1              : {macro_f1:.4f}")
    print(f"  Video-Level Accuracy (Edit/Clean) : {video_accuracy * 100:.2f}% ({video_correct}/{total_videos} videos)")
    print(f"  False Positives per Clean Video   : {fp_per_clean:.2f} ({clean_fps} FPs across {clean_vids} unedited videos)")
    print(f"  Overall Mean Boundary Error       : {mean_b_err:.2f} frames" if mean_b_err else "  Overall Mean Boundary Error: N/A")
    print("-" * 78)
    print(f"{'Edit Class':<15} | {'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Mean Error':<10}")
    print("-" * 78)
    for c in EDIT_CLASSES:
        p, r, f1, tp, fp, fn, errs = best_class_stats[c]
        err_str = f"{np.mean(errs):.1f}f" if errs else "N/A"
        print(f"{c:<15} | {best_thresholds[c]:<10.2f} | {p:<10.3f} | {r:<10.3f} | {f1:<10.3f} | {err_str:<10}")
    print("=" * 78)

    results = {
        "macro_f1": round(macro_f1, 4),
        "video_level_accuracy": round(video_accuracy, 4),
        "false_positives_per_clean_video": round(fp_per_clean, 2),
        "mean_boundary_error_frames": round(mean_b_err, 2) if mean_b_err else None,
        "per_class": {
            c: {
                "threshold": best_thresholds[c],
                "precision": round(best_class_stats[c][0], 4),
                "recall": round(best_class_stats[c][1], 4),
                "f1": round(best_class_stats[c][2], 4),
                "tp": best_class_stats[c][3],
                "fp": best_class_stats[c][4],
                "fn": best_class_stats[c][5],
                "mean_boundary_error_frames": round(float(np.mean(best_class_stats[c][6])), 2) if best_class_stats[c][6] else None,
            }
            for c in EDIT_CLASSES
        },
    }
    LOSO_RESULTS_JSON.write_text(json.dumps(results, indent=2))
    print(f"\n✓ Saved LOSO evaluation metrics → {LOSO_RESULTS_JSON}\n")
    return results


if __name__ == "__main__":
    import sys
    run_loso(force="--force" in sys.argv)
