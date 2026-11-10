"""
build_table_v2.py – Day 3 Boundary-Centric Labeling & Table Generation.

Labeling v2 Rules:
  - deletion   : cut point (start_frame) +/- 2 frames.
  - insertion  : start junction (start_frame +/- 2 frames) and
                 end junction (end_frame +/- 2 frames).
  - duplication: start loopback junction (start_frame +/- 2 frames).
  - frame_drop : start rate-change (start_frame +/- 2 frames) and
                 end rate-change (end_frame +/- 2 frames).
  - all interior and clean frames : "none".

Output: ai-service/models/feature_table_v2.parquet (and .csv)
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
AI_SERVICE = _HERE.parent
sys.path.insert(0, str(AI_SERVICE))

from pipeline.features_v2 import extract_features_v2, FEATURE_NAMES_V2, CACHE_V2_DIR

GENERATED_DIR = AI_SERVICE / "dataset" / "generated"
MODELS_DIR    = AI_SERVICE / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TOLERANCE_V2 = 2  # +/- 2 frames around edit junctions


def _find_labels_csv() -> Path:
    candidates = [
        GENERATED_DIR / "labels.csv",
        AI_SERVICE / "dataset" / "labels.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return GENERATED_DIR / "labels.csv"


def _label_pairs_v2(n_pairs: int, row: pd.Series) -> np.ndarray:
    """
    Returns boolean mask of length n_pairs indicating which frame pairs
    are true boundary junction frames (+/- 2 frames).
    """
    mask = np.zeros(n_pairs, dtype=bool)
    etype = str(row["edit_type"])
    start = int(row["start_frame"])
    end   = int(row["end_frame"])

    if etype in ("none", "unedited") or start < 0:
        return mask

    def mark_junction(idx: int):
        lo = max(0, idx - TOLERANCE_V2)
        hi = min(n_pairs - 1, idx + TOLERANCE_V2)
        if lo <= hi:
            mask[lo : hi + 1] = True

    if etype == "deletion":
        mark_junction(start)
    elif etype == "duplication":
        mark_junction(start)
    elif etype in ("insertion", "frame_drop"):
        mark_junction(start)
        mark_junction(end)

    return mask


def _extract_worker_v2(path_str: str) -> tuple[str, bool, str]:
    try:
        extract_features_v2(Path(path_str))
        return (path_str, True, "")
    except Exception as exc:
        return (path_str, False, str(exc))


def build_table_v2(force: bool = False) -> pd.DataFrame:
    out_parquet = MODELS_DIR / "feature_table_v2.parquet"
    out_csv     = MODELS_DIR / "feature_table_v2.csv"

    if out_parquet.exists() and not force:
        print(f"Loading cached v2 table from {out_parquet}")
        return pd.read_parquet(out_parquet)
    if out_csv.exists() and not force:
        print(f"Loading cached v2 table from {out_csv}")
        return pd.read_csv(out_csv)

    labels_csv = _find_labels_csv()
    if not labels_csv.exists():
        raise FileNotFoundError(f"labels.csv not found at {labels_csv}")

    df_labels = pd.read_csv(labels_csv)

    unique_videos = [
        GENERATED_DIR / Path(f).name
        for f in df_labels["output_file"].unique()
        if (GENERATED_DIR / Path(f).name).exists()
    ]
    print(f"Found {len(unique_videos)} videos to process in {GENERATED_DIR}")

    missing = [
        v for v in unique_videos
        if force or not (CACHE_V2_DIR / f"{v.stem}.npy").exists()
    ]

    if missing:
        n_workers = max(1, min(8, (os.cpu_count() or 4) - 2))
        print(f"Extracting Day 3 features v2 for {len(missing)} videos ({n_workers} workers) …")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_extract_worker_v2, str(v)): v
                for v in missing
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Features v2"):
                path_str, ok, err = fut.result()
                if not ok:
                    print(f"  ⚠ Failed {Path(path_str).name}: {err}")
    else:
        print(f"All {len(unique_videos)} videos already have cached v2 features.")

    # Assemble table
    rows: list[dict] = []
    for video_file, grp in tqdm(df_labels.groupby("output_file"), desc="Assembling table v2"):
        video_path = GENERATED_DIR / Path(video_file).name
        if not video_path.exists():
            continue

        try:
            feats = extract_features_v2(video_path)
        except Exception as exc:
            print(f"  ⚠ Cannot load features for {video_path.name}: {exc}")
            continue

        n_pairs = len(feats)
        if n_pairs == 0:
            continue

        pair_labels = np.full(n_pairs, "none", dtype=object)

        for _, label_row in grp.iterrows():
            mask = _label_pairs_v2(n_pairs, label_row)
            pair_labels[mask] = label_row["edit_type"]

        source_file = grp["source_file"].iloc[0]

        for pair_idx in range(n_pairs):
            rec = {
                "video":       video_path.name,
                "source_file": str(source_file),
                "pair_idx":    pair_idx,
                "label":       pair_labels[pair_idx],
            }
            for fi, fname in enumerate(FEATURE_NAMES_V2):
                rec[fname] = float(feats[pair_idx, fi])
            rows.append(rec)

    table = pd.DataFrame(rows)

    try:
        table.to_parquet(out_parquet, index=False)
        print(f"\n✓ Feature table v2 saved → {out_parquet} ({len(table):,} rows)")
    except Exception:
        table.to_csv(out_csv, index=False)
        print(f"\n✓ Feature table v2 saved → {out_csv} ({len(table):,} rows)")

    dist = table["label"].value_counts()
    print("\nLabel distribution (v2 boundary labeling):")
    for cls, cnt in dist.items():
        print(f"  {cls:<20} {cnt:>6,}")

    return table


if __name__ == "__main__":
    build_table_v2(force="--force" in sys.argv)

