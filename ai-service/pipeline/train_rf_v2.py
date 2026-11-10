"""
train_rf_v2.py – Day 3 Random Forest with +/-15 Stacked Temporal Neighborhood.

Features:
  Each sample is a +/-15 pair neighborhood of 27 features = 837 stacked features.
Split:
  Identical GroupShuffleSplit 80/20 by source_file and GroupKFold (k=5).
Output:
  ai-service/models/rf_v2.joblib, ai-service/models/rf_v2_cv.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_validate
from sklearn.preprocessing import LabelEncoder

_HERE = Path(__file__).resolve().parent
AI_SERVICE = _HERE.parent
sys.path.insert(0, str(AI_SERVICE))

from pipeline.features_v2 import FEATURE_NAMES_V2

MODELS_DIR = AI_SERVICE / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RF_V2_PATH = MODELS_DIR / "rf_v2.joblib"
CV_V2_JSON = MODELS_DIR / "rf_v2_cv.json"
STACK_W = 15  # +/- 15 frame-pair neighborhood


def load_table_v2() -> pd.DataFrame:
    parquet = MODELS_DIR / "feature_table_v2.parquet"
    csv_ = MODELS_DIR / "feature_table_v2.csv"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv_.exists():
        return pd.read_csv(csv_)
    raise FileNotFoundError("feature_table_v2 not found. Run build_table_v2.py first.")


def build_stacked_features(table: pd.DataFrame, w: int = STACK_W) -> np.ndarray:
    """
    Per-video sliding window stacking of w past and w future pairs.
    Returns (len(table), (2*w + 1) * 27).
    """
    stacked_list = []
    d = len(FEATURE_NAMES_V2)

    for _, grp in table.groupby("video", sort=False):
        mat = grp[FEATURE_NAMES_V2].values.astype(np.float32)
        n = len(mat)
        padded = np.pad(mat, ((w, w), (0, 0)), mode="edge")
        windows = sliding_window_view(padded, (2 * w + 1, d))[:, 0, :, :]
        stacked = windows.reshape(n, (2 * w + 1) * d)
        stacked_list.append(stacked)

    return np.vstack(stacked_list).astype(np.float32)


def train_rf_v2(force: bool = False) -> tuple[RandomForestClassifier, LabelEncoder, dict]:
    if RF_V2_PATH.exists() and not force:
        print(f"Loading existing RF v2 model from {RF_V2_PATH}")
        rf, le = joblib.load(RF_V2_PATH)
        cv_res = json.loads(CV_V2_JSON.read_text()) if CV_V2_JSON.exists() else {}
        return rf, le, cv_res

    table = load_table_v2()
    print(f"Building +/-{STACK_W} stacked neighborhood features (837 dimensions) …")
    t0_feat = time.time()
    X = build_stacked_features(table, w=STACK_W)
    print(f"✓ Stacked features shape: {X.shape} in {time.time() - t0_feat:.2f}s")

    y_raw = table["label"].values
    groups = table["source_file"].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    print(f"\nDataset: {len(X):,} samples, {len(le.classes_)} classes, {len(np.unique(groups))} source groups")
    print(f"Classes: {list(le.classes_)}")

    # Identical GroupShuffleSplit 80/20 by source_file
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    groups_train = groups[train_idx]

    print(f"Train samples: {len(X_train):,}  |  Test samples: {len(X_test):,}")
    print(f"Test sources : {np.unique(groups[test_idx]).tolist()}")

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    # GroupKFold (k=5) on TRAIN split only
    print("\nRunning GroupKFold (k=5) CV on train split …")
    t0_cv = time.time()
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups_train))))
    cv_scores = cross_validate(
        rf, X_train, y_train,
        groups=groups_train,
        cv=gkf,
        scoring=["accuracy", "f1_macro", "f1_weighted"],
        return_train_score=False,
        n_jobs=-1,
    )
    cv_time = time.time() - t0_cv
    cv_res = {
        metric: {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
        }
        for metric, scores in cv_scores.items()
        if metric.startswith("test_")
    }
    print(f"CV complete in {cv_time:.2f}s:")
    for metric, stats in cv_res.items():
        print(f"  {metric:<20} {stats['mean']:.4f} ± {stats['std']:.4f}")

    print("\nFitting final RF v2 on full train split …")
    t0_fit = time.time()
    rf.fit(X_train, y_train)
    fit_time = time.time() - t0_fit
    print(f"✓ Model fit in {fit_time:.2f}s")

    cv_res["_test_idx"] = test_idx.tolist()
    cv_res["_train_idx"] = train_idx.tolist()
    cv_res["_classes"] = le.classes_.tolist()
    cv_res["training_time_sec"] = round(fit_time, 2)

    joblib.dump((rf, le), RF_V2_PATH)
    CV_V2_JSON.write_text(json.dumps(cv_res, indent=2))
    print(f"✓ Model saved → {RF_V2_PATH}")
    return rf, le, cv_res


if __name__ == "__main__":
    train_rf_v2(force="--force" in sys.argv)

