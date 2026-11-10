"""
run_day2.py – One-command Day 2 entry point.

Steps (in order):
  0. If ai-service/dataset/generated/ is empty, auto-run:
       a. make_synthetic_sources.py  → sources/
       b. generate_dataset.py        → generated/  (seed=42, rounds=2)
  1. pipeline/features.py    – extract & cache per-video features
  2. pipeline/build_table.py – assemble labeled feature table
  3. pipeline/plot_features.py – feature-over-time plots per class
  4. pipeline/train_rf.py    – train Random Forest
  5. pipeline/evaluate.py    – compute & print metrics

Usage (from ai-service/):
  python -m pipeline.run_day2
  python -m pipeline.run_day2 --force   # recompute everything
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
AI_SERVICE = _HERE.parent

GENERATED_DIR = AI_SERVICE / "dataset" / "generated"
SOURCES_DIR   = AI_SERVICE / "dataset" / "sources"
LABELS_CSV    = GENERATED_DIR / "labels.csv"


def _has_videos(directory: Path) -> bool:
    return any(directory.glob("*.mp4"))


def _run_py(script: Path, *args: str) -> None:
    cmd = [sys.executable, str(script)] + list(args)
    print(f"\n▶  {' '.join(cmd)}\n{'─' * 60}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(AI_SERVICE))
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n✗  Script failed (exit {result.returncode}). Stopping.")
        sys.exit(result.returncode)
    print(f"{'─' * 60}\n✓  Done in {elapsed:.1f}s")


def main() -> None:
    force = "--force" in sys.argv
    force_flag = ["--force"] if force else []

    print("\n" + "=" * 60)
    print("  DAY 2 PIPELINE  –  Video Edit Detection")
    print("=" * 60)

    # ── Step 0: ensure dataset exists ──────────────────────────────
    if not LABELS_CSV.exists() or not _has_videos(GENERATED_DIR):
        print("\n⚡ No generated dataset found. Creating synthetic data first …")

        if not _has_videos(SOURCES_DIR):
            print("  → Generating synthetic source clips …")
            _run_py(AI_SERVICE / "dataset" / "make_synthetic_sources.py")
        else:
            print(f"  → Found existing sources in {SOURCES_DIR}")

        GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        print("  → Running dataset generator (seed=42, rounds=2) …")
        _run_py(
            AI_SERVICE / "dataset" / "generate_dataset.py",
            "--sources", str(SOURCES_DIR),
            "--output",  str(GENERATED_DIR),
            "--seed",    "42",
            "--rounds",  "2",
        )
    else:
        print(f"\n✓ Dataset found ({sum(1 for _ in GENERATED_DIR.glob('*.mp4'))} videos)")

    if not LABELS_CSV.exists():
        print("✗  labels.csv still missing after generation. Aborting.")
        sys.exit(1)

    # ── Step 1-6: pipeline ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1 / 5 – Feature extraction")
    print("=" * 60)
    from pipeline.build_table import build_table
    table = build_table(force=force)

    print("\n" + "=" * 60)
    print("STEP 2 / 5 – Feature plots")
    print("=" * 60)
    from pipeline.plot_features import plot_features
    plot_features(table)

    print("\n" + "=" * 60)
    print("STEP 3 / 5 – Train Random Forest")
    print("=" * 60)
    from pipeline.train_rf import train
    rf, le, cv_res = train(force=force)

    print("\n" + "=" * 60)
    print("STEP 4 / 5 – Evaluate")
    print("=" * 60)
    from pipeline.evaluate import evaluate, print_metrics
    metrics = evaluate(force=force)

    print("\n" + "=" * 60)
    print("STEP 5 / 5 – Results")
    print("=" * 60)
    print_metrics(metrics)

    models_dir = AI_SERVICE / "models"
    plots_dir  = models_dir / "plots"
    print("\nOutputs:")
    print(f"  Model         → {models_dir / 'rf.joblib'}")
    print(f"  Metrics JSON  → {models_dir / 'rf_metrics.json'}")
    print(f"  Feature plots → {plots_dir}/features_*.png")
    print(f"  Confusion     → {plots_dir}/confusion.png")
    print(f"  Importances   → {plots_dir}/feat_importance.png")
    print()


if __name__ == "__main__":
    main()

