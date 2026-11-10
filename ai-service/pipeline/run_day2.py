"""
run_day2.py – One-command Day 2 entry point.

Steps (in order):
  0. If ai-service/dataset/generated/ is empty or labels.csv missing, auto-run:
       a. make_synthetic_sources.py  → sources/
       b. generate_dataset.py        → generated/  (seed=42, rounds=2)
  1. pipeline/build_table.py   – extracts & caches features, assembles labeled table
  2. pipeline/plot_features.py – feature-over-time plots per class with edit lines
  3. pipeline/train_rf.py      – train Random Forest (GroupShuffleSplit 80/20 + GroupKFold-5)
  4. pipeline/evaluate.py      – compute & print metrics, confusion & importance PNGs

Usage (from ai-service/):
  python -m pipeline.run_day2
  python -m pipeline.run_day2 --force   # recompute everything
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
AI_SERVICE = _HERE.parent

GENERATED_DIR = AI_SERVICE / "dataset" / "generated"
SOURCES_DIR   = AI_SERVICE / "dataset" / "sources"


def _find_labels_csv() -> Path | None:
    candidates = [
        GENERATED_DIR / "labels.csv",
        AI_SERVICE / "dataset" / "labels.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


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

    print("\n" + "=" * 60)
    print("  DAY 2 PIPELINE  –  Video Edit Detection")
    print("=" * 60)

    # ── Step 0: ensure dataset exists ──────────────────────────────
    labels_csv = _find_labels_csv()
    if labels_csv is None or not _has_videos(GENERATED_DIR):
        print("\n⚡ Dataset not found. Generating synthetic test dataset …")

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
        labels_csv = _find_labels_csv()

    if labels_csv is None:
        print("✗  labels.csv missing. Aborting.")
        sys.exit(1)

    # Ensure labels.csv exists in both dataset/ and dataset/generated/ for consistency
    target_in_gen = GENERATED_DIR / "labels.csv"
    if not target_in_gen.exists() and labels_csv != target_in_gen:
        shutil.copy2(labels_csv, target_in_gen)

    n_vids = sum(1 for _ in GENERATED_DIR.glob("*.mp4"))
    print(f"\n✓ Dataset ready: {n_vids} videos in {GENERATED_DIR}")
    print(f"✓ Labels: {labels_csv}")

    # ── Step 1: build feature table ────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1 / 4 – Feature extraction & Table building")
    print("=" * 60)
    from pipeline.build_table import build_table
    table = build_table(force=force)

    # ── Step 2: feature trajectories plot ──────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2 / 4 – Feature plots per class")
    print("=" * 60)
    from pipeline.plot_features import plot_features
    plot_features(table)

    # ── Step 3: train Random Forest ────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3 / 4 – Train Random Forest (GroupShuffleSplit 80/20 + GroupKFold-5)")
    print("=" * 60)
    from pipeline.train_rf import train
    rf, le, cv_res = train(force=force)

    # ── Step 4: evaluate ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4 / 4 – Evaluation & Metrics")
    print("=" * 60)
    from pipeline.evaluate import evaluate, print_metrics
    metrics = evaluate(force=force)
    print_metrics(metrics)

    models_dir = AI_SERVICE / "models"
    plots_dir  = models_dir / "plots"
    print("\nGenerated Artifacts:")
    print(f"  Model         → {models_dir / 'rf.joblib'}")
    print(f"  Metrics JSON  → {models_dir / 'rf_metrics.json'}")
    print(f"  Feature plots → {plots_dir}/features_*.png")
    print(f"  Confusion     → {plots_dir}/confusion.png")
    print(f"  Importances   → {plots_dir}/feat_importance.png")
    print()


if __name__ == "__main__":
    main()
