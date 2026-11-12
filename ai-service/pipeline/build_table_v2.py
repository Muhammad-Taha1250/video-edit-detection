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


def _extract_worker_v2(args_tuple: tuple[str, Optional[str]]) -> tuple[str, bool, str]:
    path_str, cache_dir_str = args_tuple
    try:
        extract_features_v2(
            Path(path_str),
            cache_dir=Path(cache_dir_str) if cache_dir_str else None,
        )
        return (path_str, True, "")
    except Exception as exc:
        return (path_str, False, str(exc))


def build_table_v2(
    force: bool = False,
    generated_dir: Optional[Path] = None,
    labels_csv: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    out_parquet: Optional[Path] = None,
    out_csv: Optional[Path] = None,
) -> pd.DataFrame:
    gen_dir = Path(generated_dir) if generated_dir is not None else GENERATED_DIR
    lab_csv = Path(labels_csv) if labels_csv is not None else _find_labels_csv()
    active_cache = Path(cache_dir) if cache_dir is not None else CACHE_V2_DIR
    out_pq = Path(out_parquet) if out_parquet is not None else (MODELS_DIR / "feature_table_v2.parquet")
    out_c  = Path(out_csv) if out_csv is not None else (MODELS_DIR / "feature_table_v2.csv")

    if out_pq.exists() and not force:
        print(f"Loading cached v2 table from {out_pq}")
        return pd.read_parquet(out_pq)
    if out_c.exists() and not force:
        print(f"Loading cached v2 table from {out_c}")
        return pd.read_csv(out_c)

    if not lab_csv.exists():
        raise FileNotFoundError(f"labels.csv not found at {lab_csv}")

    df_labels = pd.read_csv(lab_csv)

    unique_videos = [
        gen_dir / Path(f).name
        for f in df_labels["output_file"].unique()
        if (gen_dir / Path(f).name).exists()
    ]
    print(f"Found {len(unique_videos)} videos to process in {gen_dir}")

    missing = [
        v for v in unique_videos
        if force or not (active_cache / f"{v.stem}.npy").exists()
    ]

    if missing:
        n_workers = max(1, min(8, (os.cpu_count() or 4) - 2))
        print(f"Extracting Day 3 features v2 for {len(missing)} videos ({n_workers} workers) …")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_extract_worker_v2, (str(v), str(active_cache))): v
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
        video_path = gen_dir / Path(video_file).name
        if not video_path.exists():
            continue

        try:
            feats = extract_features_v2(video_path, cache_dir=active_cache)
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
        table.to_parquet(out_pq, index=False)
        print(f"\n✓ Feature table saved → {out_pq} ({len(table):,} rows)")
    except Exception:
        table.to_csv(out_c, index=False)
        print(f"\n✓ Feature table saved → {out_c} ({len(table):,} rows)")

    dist = table["label"].value_counts()
    print("\nLabel distribution (v2 boundary labeling):")
    for cls, cnt in dist.items():
        print(f"  {cls:<20} {cnt:>6,}")

    return table


if __name__ == "__main__":
    build_table_v2(force="--force" in sys.argv)

