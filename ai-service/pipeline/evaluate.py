"""
evaluate.py – Full evaluation on the held-out test split.

Outputs:
  ai-service/models/rf_metrics.json         – per-class + macro metrics
  ai-service/models/plots/confusion.png     – confusion matrix
  ai-service/models/plots/feat_importance.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

_HERE = Path(__file__).resolve().parent
AI_SERVICE = _HERE.parent
sys.path.insert(0, str(AI_SERVICE))

from pipeline.features import FEATURE_NAMES

MODELS_DIR = AI_SERVICE / "models"
PLOTS_DIR  = MODELS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RF_PATH    = MODELS_DIR / "rf.joblib"
CV_JSON    = MODELS_DIR / "rf_cv.json"
METRICS_JSON = MODELS_DIR / "rf_metrics.json"


def _load_table() -> pd.DataFrame:
    parquet = MODELS_DIR / "feature_table.parquet"
    csv_    = MODELS_DIR / "feature_table.csv"
    if parquet.exists():
        return pd.read_parquet(parquet)
    return pd.read_csv(csv_)


def _mean_abs_boundary_error(
    table: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
) -> dict[str, float]:
    """
    For each edit class, find the first pair index predicted as that class
    and compare it to the true first boundary in y_true. Mean absolute error.
    Very approximate – meaningful only after merging into segments.
    """
    errors: dict[str, list[float]] = {c: [] for c in classes if c != "none"}
    n = len(y_true)
    for cls in errors:
        # Find ground-truth boundary starts
        gt_positions = np.where(y_true == cls)[0]
        if len(gt_positions) == 0:
            continue
        # Boundaries = first index of each contiguous run
        gaps = np.where(np.diff(gt_positions) > 1)[0] + 1
        starts = np.concatenate([[0], gaps])
        gt_starts = gt_positions[starts]

        pred_positions = np.where(y_pred == cls)[0]
        if len(pred_positions) == 0:
            continue
        gaps2 = np.where(np.diff(pred_positions) > 1)[0] + 1
        starts2 = np.concatenate([[0], gaps2])
        pred_starts = pred_positions[starts2]

        # Match each GT start to nearest predicted start
        for gs in gt_starts:
            nearest = np.argmin(np.abs(pred_starts - gs))
            errors[cls].append(abs(int(pred_starts[nearest]) - int(gs)))

    return {cls: float(np.mean(errs)) if errs else float("nan")
            for cls, errs in errors.items()}


def _plot_confusion(cm: np.ndarray, classes: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(5, len(classes)), max(4, len(classes) - 1)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted",
        ylabel="True",
        title="Confusion matrix (test split)",
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    thresh = cm.max() / 2.0
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Confusion matrix → {out_path.name}")


def _plot_importance(rf, out_path: Path) -> None:
    imp   = rf.feature_importances_
    idx   = np.argsort(imp)[::-1]
    names = [FEATURE_NAMES[i] for i in idx]
    vals  = imp[idx]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(vals)), vals, color="#2176ae")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_title("Random Forest feature importances")
    ax.set_ylabel("Mean decrease in impurity")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Feature importance → {out_path.name}")


def evaluate(force: bool = False) -> dict:
    if METRICS_JSON.exists() and not force:
        print(f"Loading cached metrics from {METRICS_JSON}")
        return json.loads(METRICS_JSON.read_text())

    if not RF_PATH.exists():
        raise FileNotFoundError("No trained model found. Run train_rf.py first.")

    rf, le = joblib.load(RF_PATH)
    cv_res = json.loads(CV_JSON.read_text()) if CV_JSON.exists() else {}

    table    = _load_table()
    test_idx = cv_res.get("_test_idx")
    if test_idx is None:
        raise RuntimeError("No test indices in rf_cv.json. Re-run train_rf.py.")

    test_idx   = np.array(test_idx)
    X_test     = table[FEATURE_NAMES].values[test_idx].astype(np.float32)
    y_true_enc = le.transform(table["label"].values[test_idx])
    y_pred_enc = rf.predict(X_test)
    y_true     = le.inverse_transform(y_true_enc)
    y_pred     = le.inverse_transform(y_pred_enc)

    classes = le.classes_.tolist()
    acc     = float(accuracy_score(y_true_enc, y_pred_enc))

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    prec_macro = float(precision_score(y_true_enc, y_pred_enc, average="macro", zero_division=0))
    rec_macro  = float(recall_score(y_true_enc, y_pred_enc, average="macro", zero_division=0))
    f1_macro   = float(f1_score(y_true_enc, y_pred_enc, average="macro", zero_division=0))

    mabe = _mean_abs_boundary_error(table.iloc[test_idx], y_true, y_pred, classes)

    cm = confusion_matrix(y_true, y_pred, labels=classes)

    _plot_confusion(cm, classes, PLOTS_DIR / "confusion.png")
    _plot_importance(rf, PLOTS_DIR / "feat_importance.png")

    cv_clean = {k: v for k, v in cv_res.items() if not k.startswith("_")}

    metrics = {
        "note": "SYNTHETIC SMOKE-TEST DATA — numbers are not representative of real-world performance.",
        "accuracy": round(acc, 4),
        "macro_precision": round(prec_macro, 4),
        "macro_recall":    round(rec_macro, 4),
        "macro_f1":        round(f1_macro, 4),
        "per_class":       {
            cls: {k: round(v, 4) for k, v in report[cls].items()}
            for cls in classes if cls in report
        },
        "mean_abs_boundary_error_frames": {k: round(v, 2) if not np.isnan(v) else None
                                           for k, v in mabe.items()},
        "cv_on_train_split": cv_clean,
    }

    METRICS_JSON.write_text(json.dumps(metrics, indent=2))
    print(f"\n✓ Metrics saved → {METRICS_JSON}")
    return metrics


def print_metrics(metrics: dict) -> None:
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print(f"⚠  {metrics['note']}")
    print("=" * 60)
    print(f"  Accuracy        : {metrics['accuracy']:.4f}")
    print(f"  Macro Precision : {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall    : {metrics['macro_recall']:.4f}")
    print(f"  Macro F1        : {metrics['macro_f1']:.4f}")
    print("\n  Per-class metrics:")
    pc = metrics.get("per_class", {})
    classes_edit = [c for c in pc if c != "none"]
    weak_classes = []
    for cls in sorted(pc.keys()):
        row = pc[cls]
        f1  = row.get("f1-score", 0)
        flag = " ⚠ WEAK" if cls != "none" and f1 < 0.5 else ""
        if flag:
            weak_classes.append(cls)
        print(f"    {cls:<20}  P={row.get('precision',0):.3f}  "
              f"R={row.get('recall',0):.3f}  F1={f1:.3f}{flag}")
    print("\n  Mean abs boundary error (frames):")
    for cls, err in metrics.get("mean_abs_boundary_error_frames", {}).items():
        val = f"{err:.1f}" if err is not None else "N/A"
        print(f"    {cls:<20} {val}")
    cv = metrics.get("cv_on_train_split", {})
    if cv:
        print("\n  GroupKFold-5 CV (train split only):")
        for metric, stats in cv.items():
            print(f"    {metric:<25} {stats['mean']:.4f} ± {stats['std']:.4f}")
    print("=" * 60)
    if weak_classes:
        print(f"\n⚠  WEAK CLASSES (F1 < 0.50): {', '.join(weak_classes)}")
        print("   Expected on synthetic data — retrain with real source videos.")
    else:
        print("\n✓ All classes F1 ≥ 0.50 on synthetic data.")
    print()


if __name__ == "__main__":
    m = evaluate(force="--force" in sys.argv)
    print_metrics(m)

