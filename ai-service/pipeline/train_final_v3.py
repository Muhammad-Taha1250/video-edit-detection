"""
train_final_v3.py – Train FINAL Production Models on All 10 Sources (No Held-Out).

Artifacts saved to ai-service/models/final_v3/:
  - cut_detector.joblib
  - framedrop_detector.joblib
  - thresholds.json
  - model_card.json
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import RandomForestClassifier

AI_SERVICE = Path(__file__).resolve().parent.parent
MODELS_DIR = AI_SERVICE / "models"
FINAL_DIR = MODELS_DIR / "final_v3"
FINAL_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_TABLE_PATH = MODELS_DIR / "feature_table_v2.csv"
LOSO_V3_METRICS_PATH = MODELS_DIR / "loso_v3_metrics.json"

STACK_W = 15


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


def main():
    print("\n" + "=" * 65)
    print("TRAINING FINAL MODELS ON ALL 10 SOURCES (final_v3)")
    print("=" * 65)

    table = pd.read_csv(FEATURE_TABLE_PATH)
    print(f"Loaded {len(table):,} total samples from {table['source_file'].nunique()} sources.")

    # 1. Targets
    y_cut = table["label"].isin(["deletion", "insertion", "duplication"]).astype(int).values
    y_fd  = (table["label"] == "frame_drop").astype(int).values

    print(f"  Stage 1 Cut Positives : {np.sum(y_cut):,}")
    print(f"  Frame Drop Positives  : {np.sum(y_fd):,}")

    print(f"\nBuilding stacked temporal features (+/-{STACK_W} window = 837 dimensions) …")
    t0 = time.time()
    X = build_stacked_features(table, w=STACK_W)
    print(f"✓ Stacked features shape: {X.shape} in {time.time() - t0:.2f}s")

    # 2. Train Stage 1 Cut Detector
    print("\nTraining Stage 1 Cut Detector RF (100 trees, balanced) …")
    t0_cut = time.time()
    rf_cut = RandomForestClassifier(
        n_estimators=100,
        max_depth=18,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    rf_cut.fit(X, y_cut)
    cut_time = time.time() - t0_cut
    cut_path = FINAL_DIR / "cut_detector.joblib"
    joblib.dump(rf_cut, cut_path)
    print(f"✓ Cut detector trained in {cut_time:.2f}s -> {cut_path.name}")

    # 3. Train Frame Drop Detector
    print("\nTraining Frame Drop Detector RF (100 trees, balanced) …")
    t0_fd = time.time()
    rf_fd = RandomForestClassifier(
        n_estimators=100,
        max_depth=18,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    rf_fd.fit(X, y_fd)
    fd_time = time.time() - t0_fd
    fd_path = FINAL_DIR / "framedrop_detector.joblib"
    joblib.dump(rf_fd, fd_path)
    print(f"✓ Frame drop detector trained in {fd_time:.2f}s -> {fd_path.name}")

    # 4. Save Thresholds
    thresholds = {
        "thresh_cut": 0.35,
        "thresh_fd": 0.20,
    }
    thresh_path = FINAL_DIR / "thresholds.json"
    thresh_path.write_text(json.dumps(thresholds, indent=2))
    print(f"✓ Saved thresholds: {thresholds} -> {thresh_path.name}")

    # 5. Save Model Card
    loso_metrics = {}
    if LOSO_V3_METRICS_PATH.exists():
        with open(LOSO_V3_METRICS_PATH) as f:
            loso_metrics = json.load(f)

    model_card = {
        "model_name": "Video Edit Detection System (Two-Stage v3)",
        "version": "3.0.0",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "training_data_description": "synthetic smoke-test clips (10 source clips, 100 generated videos, 36,854 frame pairs)",
        "architecture": {
            "stage_1_cut_detector": {
                "type": "RandomForestClassifier",
                "n_estimators": 100,
                "max_depth": 18,
                "input_dimensions": 837,
                "stacked_window": "+/-15 frame pairs",
                "training_time_sec": round(cut_time, 2),
            },
            "stage_2_classifier": "Rule-based temporal decision cascade (Insertion -> Duplication -> Deletion)",
            "separate_framedrop_detector": {
                "type": "RandomForestClassifier",
                "n_estimators": 100,
                "max_depth": 18,
                "input_dimensions": 837,
                "training_time_sec": round(fd_time, 2),
            },
        },
        "operating_thresholds": thresholds,
        "leave_one_source_out_validation_metrics": loso_metrics,
    }
    card_path = FINAL_DIR / "model_card.json"
    card_path.write_text(json.dumps(model_card, indent=2))
    print(f"✓ Saved model card -> {card_path.name}")
    print("\nFINAL MODELS READY IN " + str(FINAL_DIR))


if __name__ == "__main__":
    main()

