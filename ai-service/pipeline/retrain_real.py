"""
retrain_real.py – Retraining Pipeline for Real Video Clips.

Usage:
    python -m pipeline.retrain_real --sources <folder> --rounds 5 --seed 7

Order of operations:
  (1) Refuse to run if sources folder contains synthetic clips
      (filenames starting with clip_ or synthetic_) or fewer than 8 videos.
  (2) Copy clips into ai-service/dataset/sources_real/ and generate edited dataset
      into ai-service/dataset/generated_real/ with labels_real.csv using generate_dataset logic.
  (3) Extract features into NEW cache folder (cache_v2_real/) without touching synthetic cache,
      and assemble feature table.
  (4) Run Leave-One-Source-Out (LOSO) evaluation for Two-Stage v3 detector with nested threshold selection;
      print per-class event-level table (precision, recall, F1), macro F1, false positives per clean video,
      and video-level accuracy.
  (5) Train final models on all real sources and save to ai-service/models/final_real/
      with thresholds.json and model_card.json stating date, real source count, total generated videos,
      "real source clips", and LOSO metrics.
  (6) Never overwrite ai-service/models/final_v3/ (keep synthetic models intact).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import RandomForestClassifier

_HERE = Path(__file__).resolve().parent
AI_SERVICE = _HERE.parent
if str(AI_SERVICE) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE))

from dataset.generate_dataset import generate_dataset
from pipeline.build_table_v2 import build_table_v2
from pipeline.events_v3 import extract_events_v3

# ─────────────────────────── Paths & Constants ───────────────────────
DATASET_DIR          = AI_SERVICE / "dataset"
SOURCES_REAL_DIR     = DATASET_DIR / "sources_real"
GENERATED_REAL_DIR   = DATASET_DIR / "generated_real"
LABELS_REAL_CSV      = DATASET_DIR / "labels_real.csv"

CACHE_V2_REAL_DIR    = AI_SERVICE / "cache_v2_real"
MODELS_DIR           = AI_SERVICE / "models"
FINAL_REAL_DIR       = MODELS_DIR / "final_real"
FINAL_V3_DIR         = MODELS_DIR / "final_v3"

FEATURE_TABLE_REAL_PQ  = MODELS_DIR / "feature_table_real.parquet"
FEATURE_TABLE_REAL_CSV = MODELS_DIR / "feature_table_real.csv"

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
STACK_W = 15
TOLERANCE_EVENT = 5
EDIT_CLASSES = ["deletion", "duplication", "frame_drop", "insertion"]


# ─────────────────────────── Step 1: Validation ───────────────────────

def validate_sources(sources_dir: Path) -> list[Path]:
    """
    Validates the source clips folder:
      - Refuses if folder contains synthetic clips (clip_* or synthetic_*)
      - Refuses if folder contains fewer than 8 videos
    """
    if not sources_dir.exists() or not sources_dir.is_dir():
        print(f"\n❌ ERROR: Sources folder does not exist or is not a directory: '{sources_dir}'")
        sys.exit(1)

    all_files = [f for f in sources_dir.iterdir() if f.is_file() and not f.name.startswith(".")]

    # Check for synthetic clips
    synthetic_files = [
        f.name for f in all_files
        if f.name.lower().startswith(("clip_", "synthetic_"))
    ]
    if synthetic_files:
        sample_str = ", ".join(synthetic_files[:3])
        if len(synthetic_files) > 3:
            sample_str += f", ... ({len(synthetic_files)} total)"
        print(
            f"\n❌ ERROR: Refusing to run: sources folder contains synthetic clips: [{sample_str}]. "
            f"Only real source clips are permitted."
        )
        sys.exit(1)

    # Filter recognized video formats
    video_files = [
        f for f in all_files
        if f.suffix.lower() in VIDEO_EXTS
    ]

    if len(video_files) < 8:
        print(
            f"\n❌ ERROR: Refusing to run: sources folder contains fewer than 8 videos "
            f"(found {len(video_files)} video file(s) in '{sources_dir}'). "
            f"Minimum 8 real source videos are required for leave-one-source-out training."
        )
        sys.exit(1)

    return sorted(video_files)


# ─────────────────────────── Stacked Features Helper ──────────────────

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


# ─────────────────────────── Evaluation Helpers ───────────────────────

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
    """Finds (thresh_cut, thresh_fd) that maximizes event macro F1 on source_list only."""
    eval_videos = [v for v, s in video_source_map.items() if s in source_list]

    candidate_cuts = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    candidate_fds  = [0.10, 0.15, 0.20, 0.25, 0.30]

    best_macro = -1.0
    best_t_cut = 0.25
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


# ─────────────────────────── Full Pipeline Runner ─────────────────────

def run_retrain_real(sources_dir: Path, rounds: int = 5, seed: int = 7) -> None:
    print("\n" + "=" * 70)
    print("  RETRAINING PIPELINE ON REAL SOURCE CLIPS")
    print("=" * 70)
    print(f"Sources input folder : {sources_dir}")
    print(f"Rounds per source    : {rounds}")
    print(f"Random seed          : {seed}")

    # ── Step 1: Validate Sources ─────────────────────────────────
    print("\n[Step 1/5] Validating source clips …")
    valid_videos = validate_sources(sources_dir)
    print(f"✓ Validated {len(valid_videos)} real source videos (≥ 8 required, no synthetic names).")

    # ── Step 2: Copy clips & Generate Dataset ────────────────────
    print(f"\n[Step 2/5] Setting up real dataset in {SOURCES_REAL_DIR} & {GENERATED_REAL_DIR} …")
    SOURCES_REAL_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_REAL_DIR.mkdir(parents=True, exist_ok=True)

    for v in valid_videos:
        dest = SOURCES_REAL_DIR / v.name
        if not dest.exists():
            shutil.copy2(v, dest)

    print(f"Generating edited variants into {GENERATED_REAL_DIR} with {LABELS_REAL_CSV.name} …")
    labels = generate_dataset(
        src_dir=SOURCES_REAL_DIR,
        out_dir=GENERATED_REAL_DIR,
        labels_csv=LABELS_REAL_CSV,
        rounds=rounds,
        seed=seed,
    )
    df_labels = pd.DataFrame(labels)

    # ── Step 3: Extract Features into cache_v2_real/ ─────────────
    print(f"\n[Step 3/5] Extracting features into isolated cache {CACHE_V2_REAL_DIR} …")
    CACHE_V2_REAL_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    table = build_table_v2(
        force=False,
        generated_dir=GENERATED_REAL_DIR,
        labels_csv=LABELS_REAL_CSV,
        cache_dir=CACHE_V2_REAL_DIR,
        out_parquet=FEATURE_TABLE_REAL_PQ,
        out_csv=FEATURE_TABLE_REAL_CSV,
    )

    feat_cols = [c for c in table.columns if c not in ("video", "source_file", "pair_idx", "label")]
    unique_sources = sorted(table["source_file"].unique().tolist())
    n_sources = len(unique_sources)

    y_cut = table["label"].isin(["deletion", "insertion", "duplication"]).astype(int).values
    y_fd  = (table["label"] == "frame_drop").astype(int).values
    sources = table["source_file"].values

    print(f"\nReal dataset samples : {len(table):,} across {n_sources} source groups.")
    print(f"Cut junction positives : {np.sum(y_cut):,}")
    print(f"Frame drop positives   : {np.sum(y_fd):,}")

    # ── Step 4: Leave-One-Source-Out Evaluation ──────────────────
    print(f"\n[Step 4/5] Running Leave-One-Source-Out (LOSO) Cross-Validation across {n_sources} folds …")
    t0_s = time.time()
    X = build_stacked_features(table, w=STACK_W)
    print(f"✓ Stacked features ready ({X.shape}) in {time.time() - t0_s:.2f}s")

    oof_cut = np.zeros(len(table), dtype=np.float32)
    oof_fd  = np.zeros(len(table), dtype=np.float32)

    for fold_idx, held_out in enumerate(unique_sources, 1):
        t0_f = time.time()
        train_mask = (sources != held_out)
        test_mask  = (sources == held_out)

        # Stage 1: Cut Detector RF
        rf_cut = RandomForestClassifier(
            n_estimators=100,
            max_depth=18,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
        rf_cut.fit(X[train_mask], y_cut[train_mask])
        oof_cut[test_mask] = rf_cut.predict_proba(X[test_mask])[:, 1]

        # Frame Drop Detector RF
        rf_fd = RandomForestClassifier(
            n_estimators=100,
            max_depth=18,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
        rf_fd.fit(X[train_mask], y_fd[train_mask])
        oof_fd[test_mask] = rf_fd.predict_proba(X[test_mask])[:, 1]

        print(f"  Fold {fold_idx:02d}/{n_sources:02d} — Held-out: {held_out:<25} | {time.time() - t0_f:.1f}s")

    # Map predictions to video
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

    # Nested Threshold Selection & Prediction
    print("\nRunning strict nested threshold selection per fold …")
    final_predicted_events_by_video = {}
    fold_thresholds_record = {}

    for fold_idx, held_out in enumerate(unique_sources, 1):
        other_sources = [s for s in unique_sources if s != held_out]
        t_cut_opt, t_fd_opt = optimize_thresholds_for_sources(
            other_sources, video_oof_data, df_labels, video_source_map
        )
        fold_thresholds_record[held_out] = {"thresh_cut": t_cut_opt, "thresh_fd": t_fd_opt}

        held_out_videos = [v for v, s in video_source_map.items() if s == held_out]
        for v in held_out_videos:
            p_cut, p_fd, f_mat = video_oof_data[v]
            events = extract_events_v3(p_cut, p_fd, f_mat, thresh_cut=t_cut_opt, thresh_fd=t_fd_opt)
            final_predicted_events_by_video[v] = events

    all_videos = list(video_oof_data.keys())
    per_class, macro_f1, fp_clean, video_acc, overall_err = evaluate_events_batch(
        final_predicted_events_by_video, df_labels, all_videos
    )

    # Print Event-Level Table
    print("\n" + "=" * 78)
    print("LEAVE-ONE-SOURCE-OUT (LOSO) EVENT-LEVEL EVALUATION — REAL CLIPS")
    print(f"Tolerance: ±{TOLERANCE_EVENT} frames | Sources: {n_sources} | Total Videos: {len(all_videos)}")
    print("=" * 78)
    print(f"{'Edit Class':<18} | {'Precision':<10} | {'Recall':<10} | {'F1':<10} | {'TP':<5} | {'FP':<5} | {'FN':<5}")
    print("-" * 78)
    for c in EDIT_CLASSES:
        st = per_class[c]
        print(
            f"{c:<18} | {st['precision']:<10.4f} | {st['recall']:<10.4f} | {st['f1']:<10.4f} | "
            f"{st['tp']:<5} | {st['fp']:<5} | {st['fn']:<5}"
        )
    print("-" * 78)
    print(f"{'Macro Event-Level F1':<32} : {macro_f1:.4f}")
    print(f"{'Video-Level Accuracy (Edit/Clean)':<32} : {video_acc * 100:.1f}%")
    print(f"{'False Positives per Clean Video':<32} : {fp_clean:.2f}")
    if overall_err is not None:
        print(f"{'Mean Boundary Error':<32} : {overall_err:.2f} frames")
    print("=" * 78 + "\n")

    # ── Step 5: Final Model Training on ALL Real Sources ─────────
    print(f"[Step 5/5] Training final production models on all {n_sources} real sources …")
    FINAL_REAL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Train final cut detector
    t0_cut = time.time()
    final_rf_cut = RandomForestClassifier(
        n_estimators=100,
        max_depth=18,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )
    final_rf_cut.fit(X, y_cut)
    cut_path = FINAL_REAL_DIR / "cut_detector.joblib"
    joblib.dump(final_rf_cut, cut_path)
    print(f"✓ Saved final Cut Detector RF → {cut_path} ({time.time() - t0_cut:.1f}s)")

    # 2. Train final frame drop detector
    t0_fd = time.time()
    final_rf_fd = RandomForestClassifier(
        n_estimators=100,
        max_depth=18,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )
    final_rf_fd.fit(X, y_fd)
    fd_path = FINAL_REAL_DIR / "framedrop_detector.joblib"
    joblib.dump(final_rf_fd, fd_path)
    print(f"✓ Saved final Frame Drop Detector RF → {fd_path} ({time.time() - t0_fd:.1f}s)")

    # 3. Final operating thresholds
    # Choose optimal threshold on pooled OOF predictions
    best_tc, best_tf = optimize_thresholds_for_sources(
        unique_sources, video_oof_data, df_labels, video_source_map
    )
    thresholds = {
        "thresh_cut": round(float(best_tc), 2),
        "thresh_fd": round(float(best_tf), 2),
    }
    thresh_path = FINAL_REAL_DIR / "thresholds.json"
    thresh_path.write_text(json.dumps(thresholds, indent=2))
    print(f"✓ Saved thresholds {thresholds} → {thresh_path}")

    # 4. Model Card
    model_card = {
        "model_name": "Video Edit Detection System (Two-Stage v3 - Real Clips)",
        "version": "3.1.0",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "training_data_description": "real source clips",
        "num_real_source_clips": n_sources,
        "total_generated_videos": len(all_videos),
        "architecture": {
            "stage_1_cut_detector": {
                "type": "RandomForestClassifier",
                "n_estimators": 100,
                "max_depth": 18,
                "input_dimensions": 837,
                "stacked_window": "+/-15 frame pairs",
            },
            "stage_2_classifier": "Rule-based temporal decision cascade (Insertion -> Duplication -> Deletion)",
            "separate_framedrop_detector": {
                "type": "RandomForestClassifier",
                "n_estimators": 100,
                "max_depth": 18,
                "input_dimensions": 837,
            },
        },
        "operating_thresholds": thresholds,
        "leave_one_source_out_validation_metrics": {
            "training_data": "real source clips",
            "macro_f1": round(macro_f1, 4),
            "video_level_accuracy": round(video_acc, 4),
            "false_positives_per_clean_video": round(fp_clean, 2),
            "mean_boundary_error_frames": round(overall_err, 2) if overall_err else None,
            "per_class": per_class,
            "nested_fold_thresholds": fold_thresholds_record,
        },
    }
    card_path = FINAL_REAL_DIR / "model_card.json"
    card_path.write_text(json.dumps(model_card, indent=2))
    print(f"✓ Saved model card → {card_path}")

    # Step 6: Verify final_v3 fallback is intact
    if FINAL_V3_DIR.exists():
        print(f"✓ Verified fallback model directory intact: {FINAL_V3_DIR}")

    print("\n" + "=" * 70)
    print("  ✅ REAL CLIP RETRAINING COMPLETED SUCCESSFULLY")
    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrain Video Edit Detection models on real video clips.",
    )
    parser.add_argument(
        "--sources",
        type=Path,
        required=True,
        help="Folder containing real source video clips (min 8 videos, no synthetic names).",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="Number of randomized editing passes per source video (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for reproducibility (default: 7).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run only Step 1 validation without training.",
    )
    args = parser.parse_args()

    # Step 1 validation
    validate_sources(args.sources)

    if args.validate_only:
        print(f"✓ Sources in '{args.sources}' passed validation.")
        return

    # Run full retraining
    run_retrain_real(
        sources_dir=args.sources,
        rounds=args.rounds,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

