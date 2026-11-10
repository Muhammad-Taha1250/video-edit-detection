"""
run_day3.py – One-command Day 3 Entry Point.

Steps:
  1. Extract Day 3 Features v2 (27 features) & build Table v2 (boundary labeling).
  2. Train Random Forest v2 with +/-15 stacked temporal neighborhood (837 dimensions).
  3. Train PyTorch Temporal Model (Dilated 1D-CNN + BiLSTM sequence tagger).
  4. Run Exact-Junction ROC-AUC Audit (Part G) & Event-Level Evaluation (headline metric).
  5. Print side-by-side comparison table (RF vs. Temporal Model) and diagnostic notes.

Usage:
  python -m pipeline.run_day3
  python -m pipeline.run_day3 --force
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
AI_SERVICE = _HERE.parent


def main():
    force = "--force" in sys.argv

    print("\n" + "=" * 70)
    print("  DAY 3 PIPELINE — Temporal Sequence Models & Event-Level Detection")
    print("=" * 70)

    # 1. Feature Extraction & Table Building
    print("\n[Step 1/4] Features v2 & Boundary-Centric Table v2 …")
    from pipeline.build_table_v2 import build_table_v2
    t0 = time.time()
    table = build_table_v2(force=force)
    print(f"✓ Table v2 ready ({len(table):,} rows) in {time.time() - t0:.1f}s")

    # 2. Train Random Forest v2
    print("\n[Step 2/4] Train Random Forest v2 (+/-15 Stacked Neighborhood) …")
    from pipeline.train_rf_v2 import train_rf_v2
    t0 = time.time()
    rf, le, rf_cv = train_rf_v2(force=force)
    print(f"✓ Random Forest v2 complete in {time.time() - t0:.1f}s")

    # 3. Train PyTorch Temporal Model
    print("\n[Step 3/4] Train PyTorch Temporal 1D-CNN + BiLSTM …")
    from pipeline.train_temporal import train_temporal
    t0 = time.time()
    temp_model, _, temp_meta = train_temporal(force=force)
    print(f"✓ PyTorch Temporal Model complete in {time.time() - t0:.1f}s")

    # 4. Evaluation & Comparison
    print("\n[Step 4/4] Evaluation (Exact-Junction AUC + Event-Level Metrics) …")
    from pipeline.evaluate_day3 import evaluate_both_models
    results = evaluate_both_models()

    # 5. Display Clean Terminal Summary Table
    rf_m = results["rf_v2_event_metrics"]
    temp_m = results["temporal_event_metrics"]
    classes = ["deletion", "duplication", "frame_drop", "insertion"]

    print("\n" + "=" * 78)
    print("DAY 3 MODEL COMPARISON — EVENT-LEVEL EVALUATION (+/-5 frame tolerance)")
    print(f"⚠  {results['note']}")
    print("=" * 78)
    header = f"{'Metric':<32} | {'Random Forest v2':<18} | {'Temporal 1D-CNN+LSTM':<20}"
    print(header)
    print("-" * 78)
    print(f"{'Macro Event-Level F1':<32} | {rf_m['macro_f1']:<18.4f} | {temp_m['macro_f1']:<20.4f}")
    print(f"{'Video-Level Accuracy (Edit/Clean)':<32} | {rf_m['video_level_accuracy']:<18.4f} | {temp_m['video_level_accuracy']:<20.4f}")
    print(f"{'FP per Clean Video':<32} | {rf_m['false_positives_per_clean_video']:<18.2f} | {temp_m['false_positives_per_clean_video']:<20.2f}")
    rf_b = f"{rf_m['overall_mean_boundary_error_frames']:.1f}f" if rf_m['overall_mean_boundary_error_frames'] else "N/A"
    temp_b = f"{temp_m['overall_mean_boundary_error_frames']:.1f}f" if temp_m['overall_mean_boundary_error_frames'] else "N/A"
    print(f"{'Mean Boundary Error':<32} | {rf_b:<18} | {temp_b:<20}")
    print("-" * 78)
    print("Per-Class Event-Level F1:")
    for c in classes:
        rf_c = rf_m["per_class"][c]
        temp_c = temp_m["per_class"][c]
        line = f"  {c:<30} | F1={rf_c['f1']:.3f} (P={rf_c['precision']:.2f}, R={rf_c['recall']:.2f}) | F1={temp_c['f1']:.3f} (P={temp_c['precision']:.2f}, R={temp_c['recall']:.2f})"
        print(line)
    print("-" * 78)
    print("Model Specifications:")
    print(f"  Random Forest v2 : 200 trees | 837 features | Fit time: {rf_cv.get('training_time_sec', 0):.1f}s")
    print(f"  Temporal Model   : {temp_meta['param_count']:,} params | 15 epochs CPU | Fit time: {temp_meta['training_time_sec']:.1f}s")
    print("=" * 78)
    print("\nGenerated Artifacts:")
    print(f"  Day 3 Metrics JSON : {AI_SERVICE / 'models' / 'day3_metrics.json'}")
    print(f"  Comparison Plot    : {AI_SERVICE / 'models' / 'plots_v2' / 'event_metrics.png'}")
    print(f"  RF v2 Model        : {AI_SERVICE / 'models' / 'rf_v2.joblib'}")
    print(f"  Temporal Model     : {AI_SERVICE / 'models' / 'temporal_v2.pt'}\n")


if __name__ == "__main__":
    main()

