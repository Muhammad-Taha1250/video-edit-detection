"""
eval_loso_v3.py – Two-Stage Cut Detection & Rule-Based LOSO Evaluation.

System Design:
  STAGE 1: Class-Agnostic Cut Detector
    - Target = 1 if the frame pair is any cut junction (deletion cut, insertion start/end,
      duplication start), else 0.
    - Stacked +/-15 temporal neighborhood (837 dimensions).
    - Leave-One-Source-Out (LOSO) cross-validation across all 10 sources.
  SEPARATE: Frame Drop Detector
    - Target = 1 if frame_drop rate-change boundary, else 0.
    - Not fed into the cut-detector stage.
  STAGE 2: Rule-Based Classification (in strict order):
    1. Insertion: Another cut lies 10-90 frames after it -> pair as INSERTION.
    2. Duplication: Self-similarity features show constant lag L (>=10 frames run) -> DUPLICATION.
    3. Deletion: Lone cut -> DELETION.
    4. Frame Drop: Paired rate-change boundaries (10-90 frames) -> FRAME_DROP.

Strict Nested Threshold Selection:
  For each held-out source, thresholds are selected from OOF predictions of the OTHER 9 sources only.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import RandomForestClassifier

AI_SERVICE = Path(__file__).resolve().parent.parent
MODELS_DIR = AI_SERVICE / "models"
GENERATED_DIR = AI_SERVICE / "dataset" / "generated"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_TABLE_PATH = MODELS_DIR / "feature_table_v2.csv"
LABELS_CSV_PATH = GENERATED_DIR / "labels.csv" if (GENERATED_DIR / "labels.csv").exists() else AI_SERVICE / "dataset" / "labels.csv"
LOSO_V2_METRICS_PATH = MODELS_DIR / "loso_metrics.json"
LOSO_V3_METRICS_PATH = MODELS_DIR / "loso_v3_metrics.json"

OOF_CUT_PATH = MODELS_DIR / "loso_v3_oof_cut.npy"
OOF_FD_PATH  = MODELS_DIR / "loso_v3_oof_fd.npy"

from pipeline.events_v3 import extract_events_v3

STACK_W = 15
TOLERANCE_EVENT = 5
EDIT_CLASSES = ["deletion", "duplication", "frame_drop", "insertion"]


def load_data():
    table = pd.read_csv(FEATURE_TABLE_PATH)
    labels_df = pd.read_csv(LABELS_CSV_PATH)
    return table, labels_df


def build_stacked_features(table: pd.DataFrame, w: int = STACK_W) -> np.ndarray:
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


def evaluate_events_batch(
    predicted_events_by_video: dict[str, list[dict]],
    labels_df: pd.DataFrame,
    target_videos: list[str],
) -> tuple[dict, float, float, float, float]:
    class_stats = {c: {"tp": 0, "fp": 0, "fn": 0, "boundary_errors": []} for c in EDIT_CLASSES}
    clean_fps = 0
    clean_vids = 0
    video_correct = 0

    for vname in target_videos:
        sub = labels_df[labels_df["output_file"].str.endswith(vname)]
        if sub.empty:
            continue
        row = sub.iloc[0]
        gt_type = str(row["edit_type"])
        gt_start = int(row["start_frame"])
        gt_end = int(row["end_frame"])
        is_clean_gt = gt_type in ("none", "unedited")

        preds = predicted_events_by_video.get(vname, [])
        is_clean_pred = (len(preds) == 0)

        if is_clean_gt == is_clean_pred:
            video_correct += 1

        if is_clean_gt:
            clean_vids += 1
            clean_fps += len(preds)
            for p in preds:
                pt = p["edit_type"]
                if pt in class_stats:
                    class_stats[pt]["fp"] += 1
            continue

        matched = False
        for p in preds:
            pt = p["edit_type"]
            ps = p["start_frame"]
            pe = p["end_frame"]

            if pt != gt_type:
                if pt in class_stats:
                    class_stats[pt]["fp"] += 1
                continue

            if pt == "deletion":
                err = abs(ps - gt_start)
                if err <= TOLERANCE_EVENT and not matched:
                    matched = True
                    class_stats[pt]["tp"] += 1
                    class_stats[pt]["boundary_errors"].append(err)
                else:
                    class_stats[pt]["fp"] += 1
            else:
                err_s = abs(ps - gt_start)
                err_e = abs(pe - gt_end)
                if err_s <= TOLERANCE_EVENT and err_e <= TOLERANCE_EVENT and not matched:
                    matched = True
                    class_stats[pt]["tp"] += 1
                    class_stats[pt]["boundary_errors"].append((err_s + err_e) / 2.0)
                else:
                    class_stats[pt]["fp"] += 1

        if not matched and gt_type in class_stats:
            class_stats[gt_type]["fn"] += 1

    per_class = {}
    f1s = []
    all_errs = []
    for c in EDIT_CLASSES:
        tp = class_stats[c]["tp"]
        fp = class_stats[c]["fp"]
        fn = class_stats[c]["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        b_err = float(np.mean(class_stats[c]["boundary_errors"])) if class_stats[c]["boundary_errors"] else None
        if b_err is not None:
            all_errs.extend(class_stats[c]["boundary_errors"])
        f1s.append(f1)
        per_class[c] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "mean_boundary_error_frames": round(b_err, 2) if b_err is not None else None,
        }

    macro_f1 = float(np.mean(f1s))
    fp_per_clean = float(clean_fps / clean_vids) if clean_vids > 0 else 0.0
    video_acc = float(video_correct / len(target_videos)) if target_videos else 0.0
    overall_err = float(np.mean(all_errs)) if all_errs else None

    return per_class, macro_f1, fp_per_clean, video_acc, overall_err


def optimize_thresholds_for_sources(
    source_list: list[str],
    video_oof_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    labels_df: pd.DataFrame,
    video_source_map: dict[str, str],
) -> tuple[float, float]:
    """
    Finds (thresh_cut, thresh_fd) that maximizes event macro F1 on source_list only.
    """
    eval_videos = [v for v, s in video_source_map.items() if s in source_list]

    candidate_cuts = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    candidate_fds  = [0.10, 0.15, 0.20, 0.25, 0.30]

    best_macro = -1.0
    best_t_cut = 0.20
    best_t_fd  = 0.20

    for tc in candidate_cuts:
        for tf in candidate_fds:
            preds_by_vid = {}
            for v in eval_videos:
                p_cut, p_fd, f_mat = video_oof_data[v]
                preds_by_vid[v] = extract_events_v3(p_cut, p_fd, f_mat, thresh_cut=tc, thresh_fd=tf)

            _, macro_f1, _, _, _ = evaluate_events_batch(preds_by_vid, labels_df, eval_videos)
            if macro_f1 > best_macro:
                best_macro = macro_f1
                best_t_cut = tc
                best_t_fd = tf

    return best_t_cut, best_t_fd


def run_eval_loso_v3(force: bool = False):
    table, labels_df = load_data()
    unique_sources = sorted(table["source_file"].unique().tolist())
    n_sources = len(unique_sources)

    print("\n" + "=" * 70)
    print("  TWO-STAGE CUT DETECTION & RULE-BASED LOSO EVALUATION (v3)")
    print("=" * 70)

    feat_cols = [c for c in table.columns if c not in ("video", "source_file", "pair_idx", "label")]

    # 1. Targets
    # Stage 1: Class-agnostic cut junction (deletion cut, insertion start/end, duplication start)
    # Target = 1 if label in ('deletion', 'insertion', 'duplication'), else 0
    y_cut = table["label"].isin(["deletion", "insertion", "duplication"]).astype(int).values

    # Separate Frame Drop Detector: Target = 1 if label == 'frame_drop', else 0
    y_fd = (table["label"] == "frame_drop").astype(int).values

    sources = table["source_file"].values

    print(f"Dataset: {len(table):,} samples across {n_sources} source groups.")
    print(f"  Stage 1 Cut Junction Positives : {np.sum(y_cut):,} (deletion, insertion, duplication)")
    print(f"  Separate Frame Drop Positives  : {np.sum(y_fd):,} (rate change transitions)")

    # 2. Compute OOF Predictions for both detectors
    if OOF_CUT_PATH.exists() and OOF_FD_PATH.exists() and not force:
        print(f"\nLoading cached OOF predictions from {OOF_CUT_PATH.name} and {OOF_FD_PATH.name} …")
        oof_cut = np.load(OOF_CUT_PATH)
        oof_fd  = np.load(OOF_FD_PATH)
    else:
        print(f"\nBuilding +/-{STACK_W} stacked temporal features (837 dimensions) …")
        t0_s = time.time()
        X = build_stacked_features(table, w=STACK_W)
        print(f"✓ Stacked features ready ({X.shape}) in {time.time() - t0_s:.2f}s")

        oof_cut = np.zeros(len(table), dtype=np.float32)
        oof_fd  = np.zeros(len(table), dtype=np.float32)

        print(f"\nRunning 10-Fold LOSO Training (Stage 1 Cut Detector + Separate Frame Drop Detector):")
        t0_loso = time.time()

        for fold_idx, held_out in enumerate(unique_sources, 1):
            train_mask = (sources != held_out)
            test_mask  = (sources == held_out)

            # Stage 1: Cut Detector RF
            rf_cut = RandomForestClassifier(
                n_estimators=100,
                max_depth=18,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            )
            t0_f = time.time()
            rf_cut.fit(X[train_mask], y_cut[train_mask])
            oof_cut[test_mask] = rf_cut.predict_proba(X[test_mask])[:, 1]

            # Separate: Frame Drop Detector RF
            rf_fd = RandomForestClassifier(
                n_estimators=100,
                max_depth=18,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            )
            rf_fd.fit(X[train_mask], y_fd[train_mask])
            oof_fd[test_mask] = rf_fd.predict_proba(X[test_mask])[:, 1]

            f_time = time.time() - t0_f
            print(f"  Fold {fold_idx:02d}/{n_sources:02d} — Held-out: {held_out:<28} | {f_time:.1f}s")

        print(f"✓ Dual-detector LOSO complete in {time.time() - t0_loso:.1f}s")
        np.save(OOF_CUT_PATH, oof_cut)
        np.save(OOF_FD_PATH, oof_fd)

    # Attach per video
    video_oof_data = {}
    video_source_map = {}
    row_offset = 0
    for vname, grp in table.groupby("video", sort=False):
        n = len(grp)
        v_cut = oof_cut[row_offset : row_offset + n]
        v_fd  = oof_fd[row_offset : row_offset + n]
        v_mat = grp[feat_cols].values.astype(np.float32)
        s_name = grp["source_file"].iloc[0]
        video_oof_data[vname] = (v_cut, v_fd, v_mat)
        video_source_map[vname] = s_name
        row_offset += n

    # 3. Strict Nested Evaluation:
    # For EACH held-out source, tune thresholds on the OTHER 9 sources, then score on held-out source
    print("\n" + "=" * 70)
    print("STEP 2: STRICT NESTED THRESHOLD SELECTION & PREDICTION PER FOLD")
    print("=" * 70)

    final_predicted_events_by_video = {}
    fold_thresholds_record = {}

    for fold_idx, held_out in enumerate(unique_sources, 1):
        other_sources = [s for s in unique_sources if s != held_out]
        # Optimize on the 9 training sources
        t_cut_opt, t_fd_opt = optimize_thresholds_for_sources(
            other_sources, video_oof_data, labels_df, video_source_map
        )
        fold_thresholds_record[held_out] = {"thresh_cut": t_cut_opt, "thresh_fd": t_fd_opt}

        # Apply strictly to the held-out source videos
        held_out_videos = [v for v, s in video_source_map.items() if s == held_out]
        for v in held_out_videos:
            p_cut, p_fd, f_mat = video_oof_data[v]
            events = extract_events_v3(p_cut, p_fd, f_mat, thresh_cut=t_cut_opt, thresh_fd=t_fd_opt)
            final_predicted_events_by_video[v] = events

        print(f"  Fold {fold_idx:02d} [{held_out:<28}]: Tuned thresholds on other 9 sources -> Cut={t_cut_opt:.2f}, FD={t_fd_opt:.2f}")

    # 4. Overall Event-Level Evaluation pooled over all 10 folds
    all_videos = list(video_oof_data.keys())
    v3_per_class, v3_macro_f1, v3_fp_clean, v3_video_acc, v3_overall_err = evaluate_events_batch(
        final_predicted_events_by_video, labels_df, all_videos
    )

    # 5. Load v2 metrics for side-by-side comparison
    v2_metrics = {}
    if LOSO_V2_METRICS_PATH.exists():
        with open(LOSO_V2_METRICS_PATH) as f:
            v2_metrics = json.load(f)

    # 6. Print Comparison Table
    print("\n" + "=" * 80)
    print("DAY 3 LOSO EVALUATION COMPARISON: v2 (Per-Class RF) vs. v3 (Two-Stage + Rules)")
    print("Headline Metric: Event-Level (+/-5 frame tolerance) — Synthetic Smoke-Test Data")
    print("=" * 80)

    header = f"{'Metric':<34} | {'v2 (Per-Class RF)':<20} | {'v3 (Two-Stage + Rules)':<22}"
    print(header)
    print("-" * 80)
    v2_macro = f"{v2_metrics.get('macro_f1', 0):.4f}" if v2_metrics else "N/A"
    print(f"{'Macro Event-Level F1':<34} | {v2_macro:<20} | {v3_macro_f1:<22.4f}")

    v2_vacc = f"{v2_metrics.get('video_level_accuracy', 0)*100:.1f}%" if v2_metrics else "N/A"
    print(f"{'Video-Level Accuracy (Edit/Clean)':<34} | {v2_vacc:<20} | {v3_video_acc*100:<21.1f}%")

    v2_fp = f"{v2_metrics.get('false_positives_per_clean_video', 0):.2f}" if v2_metrics else "N/A"
    print(f"{'FP per Clean Video':<34} | {v2_fp:<20} | {v3_fp_clean:<22.2f}")

    v2_b = f"{v2_metrics.get('mean_boundary_error_frames', 0):.2f}f" if v2_metrics.get('mean_boundary_error_frames') else "N/A"
    v3_b = f"{v3_overall_err:.2f}f" if v3_overall_err else "N/A"
    print(f"{'Mean Boundary Error':<34} | {v2_b:<20} | {v3_b:<22}")
    print("-" * 80)
    print("Per-Class Event-Level F1 Breakdown:")

    for c in EDIT_CLASSES:
        v2_c = v2_metrics.get("per_class", {}).get(c, {}) if v2_metrics else {}
        v2_str = f"F1={v2_c.get('f1', 0):.3f} (P={v2_c.get('precision', 0):.2f}, R={v2_c.get('recall', 0):.2f})" if v2_c else "N/A"
        v3_c = v3_per_class[c]
        v3_str = f"F1={v3_c['f1']:.3f} (P={v3_c['precision']:.2f}, R={v3_c['recall']:.2f})"
        print(f"  {c:<32} | {v2_str:<20} | {v3_str:<22}")

    print("=" * 80)

    # 7. Save JSON metrics
    output_data = {
        "note": "SYNTHETIC SMOKE-TEST DATA — numbers verify pipeline mechanics; swap real videos for final training.",
        "macro_f1": round(v3_macro_f1, 4),
        "video_level_accuracy": round(v3_video_acc, 4),
        "false_positives_per_clean_video": round(v3_fp_clean, 2),
        "mean_boundary_error_frames": round(v3_overall_err, 2) if v3_overall_err else None,
        "per_class": v3_per_class,
        "nested_fold_thresholds": fold_thresholds_record,
    }
    with open(LOSO_V3_METRICS_PATH, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Saved v3 LOSO metrics → {LOSO_V3_METRICS_PATH}\n")
    return output_data


if __name__ == "__main__":
    import sys
    run_eval_loso_v3(force="--force" in sys.argv)

