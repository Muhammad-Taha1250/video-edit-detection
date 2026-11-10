"""
train_rf.py – Train a Random Forest classifier on the frame-pair feature table.

Split strategy: GroupShuffleSplit 80/20 keyed on source_file (so no source
leaks into test). Also runs GroupKFold (k=5) CV and reports mean ± std.

Saves: ai-service/models/rf.joblib
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_validate
from sklearn.preprocessing import LabelEncoder

_HERE = Path(__file__).resolve().parent
AI_SERVICE = _HERE.parent
sys.path.insert(0, str(AI_SERVICE))

from pipeline.features import FEATURE_NAMES

MODELS_DIR = AI_SERVICE / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RF_PATH    = MODELS_DIR / "rf.joblib"
CV_JSON    = MODELS_DIR / "rf_cv.json"


def load_table() -> pd.DataFrame:
    parquet = MODELS_DIR / "feature_table.parquet"
    csv_    = MODELS_DIR / "feature_table.csv"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv_.exists():
        return pd.read_csv(csv_)
    raise FileNotFoundError("Run build_table first (pipeline/build_table.py).")


def train(force: bool = False) -> tuple[RandomForestClassifier, LabelEncoder, dict]:
    if RF_PATH.exists() and not force:
        print(f"Loading existing model from {RF_PATH}")
        rf, le = joblib.load(RF_PATH)
        cv_res = json.loads(CV_JSON.read_text()) if CV_JSON.exists() else {}
        return rf, le, cv_res

    table = load_table()

    X = table[FEATURE_NAMES].values.astype(np.float32)
    y_raw = table["label"].values
    groups = table["source_file"].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    print(f"\nDataset: {len(X):,} samples, {len(le.classes_)} classes, "
          f"{len(np.unique(groups))} source groups")
    print(f"Classes: {list(le.classes_)}")

    # -- GroupShuffleSplit 80/20 for final train/test split --
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, y_train = X[train_idx], y[train_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]
    groups_train = groups[train_idx]

    print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    # -- GroupKFold CV (k=5) on TRAIN set only --
    print("\nRunning GroupKFold (k=5) cross-validation on train split …")
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups_train))))
    cv_scores = cross_validate(
        rf, X_train, y_train,
        groups=groups_train,
        cv=gkf,
        scoring=["accuracy", "f1_macro", "f1_weighted"],
        return_train_score=False,
        n_jobs=-1,
    )
    cv_res = {
        metric: {
            "mean": float(np.mean(scores)),
            "std":  float(np.std(scores)),
        }
        for metric, scores in cv_scores.items()
        if metric.startswith("test_")
    }
    print("CV results (train split):")
    for metric, stats in cv_res.items():
        print(f"  {metric:<20} {stats['mean']:.4f} ± {stats['std']:.4f}")

    # -- Final fit on full train split --
    print("\nFitting final model on full train split …")
    rf.fit(X_train, y_train)

    # Store test indices and test predictions in CV results for evaluate.py
    cv_res["_test_idx"]  = test_idx.tolist()
    cv_res["_train_idx"] = train_idx.tolist()
    cv_res["_classes"]   = le.classes_.tolist()

    joblib.dump((rf, le), RF_PATH)
    CV_JSON.write_text(json.dumps(cv_res, indent=2))
    print(f"\n✓ Model saved → {RF_PATH}")
    return rf, le, cv_res


if __name__ == "__main__":
    train(force="--force" in sys.argv)

