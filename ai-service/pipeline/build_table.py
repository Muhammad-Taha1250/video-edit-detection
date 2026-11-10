"""
build_table.py – Assemble one feature-row per frame pair with ground-truth labels.

Label assignment:
  - Every frame pair i→i+1 within ±TOLERANCE frames of a labeled edit boundary
    is assigned the edit_type from labels.csv.
  - All other pairs are labeled "none".
  - For classes with a range (deletion, insertion, duplication, frame_drop) the
    *entire* range [start_frame, end_frame] is labeled, not just the boundary.

Output: ai-service/models/feature_table.parquet  (or .csv fallback)
Columns: video, source_file, pair_idx, label, <14 feature columns>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
AI_SERVICE = _HERE.parent
sys.path.insert(0, str(AI_SERVICE))

from pipeline.features import extract_features, FEATURE_NAMES, CACHE_DIR

GENERATED_DIR = AI_SERVICE / "dataset" / "generated"
LABELS_CSV    = GENERATED_DIR / "labels.csv"
MODELS_DIR    = AI_SERVICE / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TOLERANCE = 3   # ±frames around edit boundary that also get labeled as edit


def _label_pairs(n_pairs: int, row: pd.Series) -> np.ndarray:
    """
    Return a boolean mask of length n_pairs indicating which frame pairs
    belong to the edit segment, with ±TOLERANCE expansion.
    """
    mask = np.zeros(n_pairs, dtype=bool)
    start = int(row["start_frame"])
    end   = int(row["end_frame"])
    # treat unedited as "none" (no mask)
    if row["edit_type"] == "unedited":
        return mask
    lo = max(0, start - TOLERANCE)
    hi = min(n_pairs - 1, end + TOLERANCE)
    if lo <= hi:
        mask[lo: hi + 1] = True
    return mask


def build_table(force: bool = False) -> pd.DataFrame:
    out_parquet = MODELS_DIR / "feature_table.parquet"
    out_csv     = MODELS_DIR / "feature_table.csv"

    if out_parquet.exists() and not force:
        print(f"Loading cached table from {out_parquet}")
        return pd.read_parquet(out_parquet)
    if out_csv.exists() and not force:
        print(f"Loading cached table from {out_csv}")
        return pd.read_csv(out_csv)

    if not LABELS_CSV.exists():
        raise FileNotFoundError(
            f"labels.csv not found at {LABELS_CSV}\n"
            "Run the dataset generator first:\n"
            "  python ai-service/dataset/generate_dataset.py --sources ai-service/dataset/sources/"
        )

    df_labels = pd.read_csv(LABELS_CSV)
    rows: list[dict] = []

    for video_file, grp in tqdm(df_labels.groupby("output_file"), desc="Building table"):
        video_path = GENERATED_DIR / Path(video_file).name
        if not video_path.exists():
            print(f"  ⚠  Missing video: {video_path.name} – skipping")
            continue

        try:
            feats = extract_features(video_path, desc=video_path.stem)
        except Exception as exc:
            print(f"  ⚠  Feature extraction failed for {video_path.name}: {exc}")
            continue

        n_pairs = len(feats)
        if n_pairs == 0:
            continue

        # Build per-pair label array (default "none")
        pair_labels = np.full(n_pairs, "none", dtype=object)

        for _, label_row in grp.iterrows():
            mask = _label_pairs(n_pairs, label_row)
            pair_labels[mask] = label_row["edit_type"]

        # Source file (same for all rows in this group)
        source_file = grp["source_file"].iloc[0]

        for pair_idx in range(n_pairs):
            rec = {
                "video":       video_path.name,
                "source_file": str(source_file),
                "pair_idx":    pair_idx,
                "label":       pair_labels[pair_idx],
            }
            for fi, fname in enumerate(FEATURE_NAMES):
                rec[fname] = float(feats[pair_idx, fi])
            rows.append(rec)

    if not rows:
        raise RuntimeError("No rows were generated – check that generated/ contains videos.")

    table = pd.DataFrame(rows)
    # Save
    try:
        table.to_parquet(out_parquet, index=False)
        print(f"✓ Feature table saved → {out_parquet}  ({len(table):,} rows)")
    except Exception:
        table.to_csv(out_csv, index=False)
        print(f"✓ Feature table saved → {out_csv}  ({len(table):,} rows)")

    # Class distribution
    dist = table["label"].value_counts()
    print("\nLabel distribution:")
    for cls, cnt in dist.items():
        print(f"  {cls:<20} {cnt:>6,}")

    return table


if __name__ == "__main__":
    build_table(force="--force" in sys.argv)

