"""
evaluate_day3.py – Day 3 Comprehensive Evaluation: Exact-Junction AUC + Event-Level Metrics.

Evaluates:
  1. Exact-Junction ROC-AUC (Part G) for every edit type against 'none':
     - Deletion cut junction
     - Insertion start junction
     - Insertion end junction
     - Duplication start junction
     - Frame drop start of rate change
     - Frame drop end of rate change
     Flags any junction with AUC < 0.90.
  2. Event-Level Evaluation (headline metric) with +/-5 frame tolerance:
     - Per-class precision, recall, F1, macro F1
     - Mean boundary error (frames)
     - False positives per clean video
     - Video-level edited-vs-clean classification accuracy
  3. Side-by-side comparison table: Random Forest v2 vs. Temporal 1D-CNN+BiLSTM.

Outputs:
  ai-service/models/day3_metrics.json
  ai-service/models/plots_v2/event_metrics.png
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
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
AI_SERVICE = _HERE.parent
sys.path.insert(0, str(AI_SERVICE))

from pipeline.features_v2 import FEATURE_NAMES_V2, extract_features_v2
from pipeline.segments_v2 import postprocess_events
from pipeline.train_temporal import TemporalCNN, _slice_windows, WINDOW_T, STRIDE_T
from pipeline.train_rf_v2 import build_stacked_features, STACK_W

MODELS_DIR = AI_SERVICE / "models"
GENERATED_DIR = AI_SERVICE / "dataset" / "generated"
PLOTS_V2_DIR = MODELS_DIR / "plots_v2"
PLOTS_V2_DIR.mkdir(parents=True, exist_ok=True)

TOLERANCE_EVENT = 5  # +/- 5 frames event-level matching tolerance


def _find_labels_csv() -> Path:
    for p in [GENERATED_DIR / "labels.csv", AI_SERVICE / "dataset" / "labels.csv"]:
        if p.exists():
            return p
    return GENERATED_DIR / "labels.csv"


# =========================================================================
# Part G: Exact-Junction ROC-AUC Audit
# =========================================================================
def audit_exact_junctions() -> dict[str, dict]:
    table_csv = MODELS_DIR / "feature_table_v2.csv"
    table_parquet = MODELS_DIR / "feature_table_v2.parquet"
    if table_parquet.exists():
        table = pd.read_parquet(table_parquet)
    else:
        table = pd.read_csv(table_csv)

    labels_df = pd.read_csv(_find_labels_csv())
    none_data = table[table["label"] == "none"]

    junction_specs = [
        ("deletion_cut", "deletion", lambda r: [int(r["start_frame"]) - 1]),
        ("insertion_start", "insertion", lambda r: [int(r["start_frame"]) - 1]),
        ("insertion_end", "insertion", lambda r: [int(r["end_frame"])]),
        ("duplication_start", "duplication", lambda r: [int(r["start_frame"]) - 1]),
        ("framedrop_start", "frame_drop", lambda r: [int(r["start_frame"])]),
        ("framedrop_end", "frame_drop", lambda r: [int(r["end_frame"])]),
    ]

    audit_results = {}
    print("\n" + "=" * 65)
    print("PART G: EXACT-JUNCTION ROC-AUC AUDIT (vs. 'none')")
    print("=" * 65)

    for jname, etype, get_pairs in junction_specs:
        v_sub = labels_df[labels_df["edit_type"] == etype]
        j_rows = []
        for _, row in v_sub.iterrows():
            vname = Path(row["output_file"]).name
            target_pairs = get_pairs(row)
            v_tab = table[table["video"] == vname]
            for p in target_pairs:
                match = v_tab[v_tab["pair_idx"] == p]
                if not match.empty:
                    j_rows.append(match)

        if not j_rows:
            continue

        df_j = pd.concat(j_rows, ignore_index=True)
        combined = pd.concat([df_j.assign(is_junction=1), none_data.assign(is_junction=0)], ignore_index=True)

        feature_aucs = {}
        for f in FEATURE_NAMES_V2:
            try:
                auc = float(roc_auc_score(combined["is_junction"], combined[f]))
                feature_aucs[f] = max(auc, 1.0 - auc)
            except Exception:
                feature_aucs[f] = 0.50

        # Best feature
        best_f = max(feature_aucs, key=feature_aucs.get)
        best_auc = feature_aucs[best_f]
        flag = " [FAIL: AUC < 0.90]" if best_auc < 0.90 else " [PASS: AUC >= 0.90]"

        audit_results[jname] = {
            "best_feature": best_f,
            "best_auc": round(best_auc, 4),
            "junction_samples": len(df_j),
            "flag": flag.strip(),
            "all_aucs": {k: round(v, 4) for k, v in feature_aucs.items()},
        }

        print(f"  {jname:<20}: Best = {best_f:<14} (AUC = {best_auc:.4f}){flag}")

    print("=" * 65)
    return audit_results


# =========================================================================
# Event-Level Matching & Evaluation
# =========================================================================
def _evaluate_events(
    pred_events_by_video: dict[str, list[dict]],
    labels_df: pd.DataFrame,
    test_videos: list[str],
) -> dict:
    edit_classes = ["deletion", "duplication", "frame_drop", "insertion"]
    class_stats = {c: {"tp": 0, "fp": 0, "fn": 0, "boundary_errors": []} for c in edit_classes}
    clean_fps = 0
    clean_vids = 0
    video_correct = 0

    for vname in test_videos:
        lsub = labels_df[labels_df["output_file"].str.endswith(vname)]
        if lsub.empty:
            continue
        row = lsub.iloc[0]
        gt_type = str(row["edit_type"])
        gt_start = int(row["start_frame"])
        gt_end = int(row["end_frame"])
        is_clean_gt = gt_type in ("none", "unedited")

        preds = pred_events_by_video.get(vname, [])
        is_clean_pred = len(preds) == 0

        # Video-level edited vs clean
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

        # Video with 1 ground-truth edit event
        matched = False
        for p in preds:
            pt = p["edit_type"]
            ps = p["start_frame"]
            pe = p["end_frame"]

            if pt != gt_type:
                if pt in class_stats:
                    class_stats[pt]["fp"] += 1
                continue

            # Matching tolerance check
            if pt == "deletion":
                start_err = abs(ps - gt_start)
                if start_err <= TOLERANCE_EVENT and not matched:
                    matched = True
                    class_stats[pt]["tp"] += 1
                    class_stats[pt]["boundary_errors"].append(start_err)
                else:
                    class_stats[pt]["fp"] += 1
            else:
                start_err = abs(ps - gt_start)
                end_err = abs(pe - gt_end)
                if start_err <= TOLERANCE_EVENT and end_err <= TOLERANCE_EVENT and not matched:
                    matched = True
                    class_stats[pt]["tp"] += 1
                    class_stats[pt]["boundary_errors"].append((start_err + end_err) / 2.0)
                else:
                    class_stats[pt]["fp"] += 1

        if not matched and gt_type in class_stats:
            class_stats[gt_type]["fn"] += 1

    per_class_metrics = {}
    f1_list = []
    all_b_errs = []

    for c in edit_classes:
        tp = class_stats[c]["tp"]
        fp = class_stats[c]["fp"]
        fn = class_stats[c]["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        b_err = float(np.mean(class_stats[c]["boundary_errors"])) if class_stats[c]["boundary_errors"] else None
        if b_err is not None:
            all_b_errs.extend(class_stats[c]["boundary_errors"])
        f1_list.append(f1)
        per_class_metrics[c] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "mean_boundary_error_frames": round(b_err, 2) if b_err is not None else None,
        }

    macro_f1 = float(np.mean(f1_list))
    fp_per_clean = float(clean_fps / clean_vids) if clean_vids > 0 else 0.0
    video_acc = float(video_correct / len(test_videos)) if test_videos else 0.0
    overall_b_err = float(np.mean(all_b_errs)) if all_b_errs else None

    return {
        "macro_f1": round(macro_f1, 4),
        "video_level_accuracy": round(video_acc, 4),
        "false_positives_per_clean_video": round(fp_per_clean, 2),
        "overall_mean_boundary_error_frames": round(overall_b_err, 2) if overall_b_err else None,
        "per_class": per_class_metrics,
    }


def evaluate_both_models() -> dict:
    table_csv = MODELS_DIR / "feature_table_v2.csv"
    table_parquet = MODELS_DIR / "feature_table_v2.parquet"
    if table_parquet.exists():
        table = pd.read_parquet(table_parquet)
    else:
        table = pd.read_csv(table_csv)

    labels_df = pd.read_csv(_find_labels_csv())

    with open(MODELS_DIR / "rf_v2_cv.json") as f:
        rf_meta = json.load(f)
    test_idx = np.array(rf_meta["_test_idx"])
    test_df = table.iloc[test_idx]
    test_videos = sorted(test_df["video"].unique().tolist())

    print(f"\nEvaluating Event-Level Performance on Held-Out Test Set ({len(test_videos)} videos):")

    # 1. Random Forest Predictions
    rf, le = joblib.load(MODELS_DIR / "rf_v2.joblib")
    rf_preds_by_vid = {}

    print("  Predicting with Random Forest v2 (+/-15 neighborhood) …")
    for vname in test_videos:
        v_tab = table[table["video"] == vname]
        v_feats = v_tab[FEATURE_NAMES_V2].values.astype(np.float32)
        n = len(v_feats)
        padded = np.pad(v_feats, ((STACK_W, STACK_W), (0, 0)), mode="edge")
        from numpy.lib.stride_tricks import sliding_window_view
        d = len(FEATURE_NAMES_V2)
        X_stack = sliding_window_view(padded, (2 * STACK_W + 1, d))[:, 0, :, :].reshape(n, (2 * STACK_W + 1) * d)

        proba = rf.predict_proba(X_stack)
        events = postprocess_events(proba, le.classes_.tolist(), v_feats)
        rf_preds_by_vid[vname] = events

    rf_event_metrics = _evaluate_events(rf_preds_by_vid, labels_df, test_videos)

    # 2. Temporal PyTorch Model Predictions
    temporal_model = TemporalCNN(in_channels=len(FEATURE_NAMES_V2), num_classes=len(le.classes_))
    temporal_model.load_state_dict(torch.load(MODELS_DIR / "temporal_v2.pt"))
    temporal_model.eval()
    temporal_preds_by_vid = {}

    print("  Predicting with PyTorch Temporal 1D-CNN + BiLSTM …")
    with torch.no_grad():
        for vname in test_videos:
            v_tab = table[table["video"] == vname]
            v_feats = v_tab[FEATURE_NAMES_V2].values.T.astype(np.float32)  # (D, N)
            n = v_feats.shape[1]

            if n < WINDOW_T:
                pad_w = WINDOW_T - n
                inp = np.pad(v_feats, ((0, 0), (0, pad_w)), mode="edge")
                t_inp = torch.tensor(inp, dtype=torch.float32).unsqueeze(0)
                logits = temporal_model(t_inp)[0, :, :n]  # (5, n)
            else:
                # Accumulate overlapping window logits
                accum_logits = np.zeros((len(le.classes_), n), dtype=np.float32)
                counts = np.zeros(n, dtype=np.float32)
                for start in range(0, n - WINDOW_T + 1, STRIDE_T):
                    end = start + WINDOW_T
                    sub_inp = torch.tensor(v_feats[:, start:end], dtype=torch.float32).unsqueeze(0)
                    sub_out = temporal_model(sub_inp)[0].numpy()
                    accum_logits[:, start:end] += sub_out
                    counts[start:end] += 1.0
                if (n - WINDOW_T) % STRIDE_T != 0:
                    sub_inp = torch.tensor(v_feats[:, n - WINDOW_T : n], dtype=torch.float32).unsqueeze(0)
                    sub_out = temporal_model(sub_inp)[0].numpy()
                    accum_logits[:, n - WINDOW_T : n] += sub_out
                    counts[n - WINDOW_T : n] += 1.0
                logits = torch.tensor(accum_logits / np.maximum(counts, 1.0))

            probs = torch.softmax(logits, dim=0).T.numpy()  # (N, 5)
            events = postprocess_events(probs, le.classes_.tolist(), v_feats.T)
            temporal_preds_by_vid[vname] = events

    temp_event_metrics = _evaluate_events(temporal_preds_by_vid, labels_df, test_videos)

    # 3. Compile full comparison results
    junction_audit = audit_exact_junctions()

    with open(MODELS_DIR / "temporal_v2_meta.json") as f:
        t_meta = json.load(f)

    full_results = {
        "note": "SYNTHETIC SMOKE-TEST DATA — numbers verify pipeline mechanics; swap real videos for final training.",
        "junction_audit": junction_audit,
        "rf_v2_event_metrics": rf_event_metrics,
        "temporal_event_metrics": temp_event_metrics,
        "model_specs": {
            "random_forest": {
                "n_estimators": 200,
                "stacked_window": "+/-15 frame pairs (837 dimensions)",
                "training_time_sec": rf_meta.get("training_time_sec", 0),
            },
            "temporal_cnn_bilstm": {
                "param_count": t_meta["param_count"],
                "window_t": WINDOW_T,
                "stride_t": STRIDE_T,
                "training_time_sec": t_meta["training_time_sec"],
            },
        },
    }

    out_json = MODELS_DIR / "day3_metrics.json"
    out_json.write_text(json.dumps(full_results, indent=2))
    print(f"\n✓ Metrics saved → {out_json}")

    _plot_day3_comparison(rf_event_metrics, temp_event_metrics, PLOTS_V2_DIR / "event_metrics.png")
    return full_results


def _plot_day3_comparison(rf_m: dict, temp_m: dict, out_path: Path):
    classes = ["deletion", "duplication", "frame_drop", "insertion"]
    rf_f1s = [rf_m["per_class"][c]["f1"] for c in classes]
    temp_f1s = [temp_m["per_class"][c]["f1"] for c in classes]

    x = np.arange(len(classes))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w / 2, rf_f1s, w, label=f"Random Forest v2 (Macro={rf_m['macro_f1']:.3f})", color="#2176ae")
    ax.bar(x + w / 2, temp_f1s, w, label=f"Temporal CNN+BiLSTM (Macro={temp_m['macro_f1']:.3f})", color="#f26419")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Event-Level F1 (+/-5f tolerance)", fontsize=9)
    ax.set_title("Day 3 Model Comparison: Event-Level F1 on Held-Out Test Set", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved comparison plot → {out_path.name}")


if __name__ == "__main__":
    evaluate_both_models()

