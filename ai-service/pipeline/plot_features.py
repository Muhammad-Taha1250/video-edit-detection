"""
plot_features.py – Visualise features over time for one sample video per class.

For each edit class, picks the first video of that class from the feature table,
plots the 7 raw features over time, and draws vertical red lines at the labeled
edit boundaries.

Output: ai-service/models/plots/features_<class>.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
AI_SERVICE = _HERE.parent
sys.path.insert(0, str(AI_SERVICE))

from pipeline.features import FEATURE_NAMES, extract_features

GENERATED_DIR = AI_SERVICE / "dataset" / "generated"
LABELS_CSV    = GENERATED_DIR / "labels.csv"
MODELS_DIR    = AI_SERVICE / "models"
PLOTS_DIR     = MODELS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RAW_FEAT_NAMES = FEATURE_NAMES[:7]   # first 7 are raw (un-normalised)


def _edit_boundaries(labels_df: pd.DataFrame, video_name: str) -> list[tuple[int, int, str]]:
    """Return [(start_frame, end_frame, edit_type)] for a given output video."""
    sub = labels_df[labels_df["output_file"].str.endswith(video_name)]
    out = []
    for _, row in sub.iterrows():
        if row["edit_type"] != "unedited":
            out.append((int(row["start_frame"]), int(row["end_frame"]), row["edit_type"]))
    return out


def plot_features(table: pd.DataFrame | None = None) -> None:
    if table is None:
        parquet = MODELS_DIR / "feature_table.parquet"
        csv_    = MODELS_DIR / "feature_table.csv"
        if parquet.exists():
            table = pd.read_parquet(parquet)
        elif csv_.exists():
            table = pd.read_csv(csv_)
        else:
            raise FileNotFoundError("Run build_table first.")

    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"labels.csv not found at {LABELS_CSV}")

    labels_df = pd.read_csv(LABELS_CSV)

    classes = [c for c in table["label"].unique() if c != "none"]
    if not classes:
        print("No edit classes found – nothing to plot.")
        return

    for cls in tqdm(classes, desc="Plotting"):
        sub = table[table["label"] == cls]
        if sub.empty:
            continue

        # Pick first video that has this class
        video_name = sub["video"].iloc[0]
        video_path = GENERATED_DIR / video_name

        # Get full feature sequence for this video
        try:
            feats = extract_features(video_path, desc=video_name)
        except Exception as exc:
            print(f"  ⚠  Cannot load features for {video_name}: {exc}")
            continue

        boundaries = _edit_boundaries(labels_df, video_name)

        n_pairs = len(feats)
        pair_idx = np.arange(n_pairs)

        fig, axes = plt.subplots(len(RAW_FEAT_NAMES), 1,
                                 figsize=(14, 2.2 * len(RAW_FEAT_NAMES)),
                                 sharex=True)
        fig.suptitle(f"Features – class: {cls}\nVideo: {video_name}", fontsize=11)

        for ax, fname in zip(axes, RAW_FEAT_NAMES):
            col_idx = FEATURE_NAMES.index(fname)
            ax.plot(pair_idx, feats[:, col_idx], linewidth=0.8, color="#2176ae")
            ax.set_ylabel(fname, fontsize=7, rotation=0, labelpad=80, va="center")
            ax.yaxis.set_label_position("left")
            ax.tick_params(labelsize=7)

            for start, end, etype in boundaries:
                ax.axvspan(max(start - 3, 0), min(end + 3, n_pairs - 1),
                           alpha=0.18, color="red", label=etype if fname == RAW_FEAT_NAMES[0] else "")
                ax.axvline(start, color="red",   linestyle="--", linewidth=0.8, alpha=0.7)
                ax.axvline(end,   color="orange", linestyle="--", linewidth=0.8, alpha=0.7)

        axes[0].legend(fontsize=7, loc="upper right")
        axes[-1].set_xlabel("Frame pair index")
        plt.tight_layout()

        out_path = PLOTS_DIR / f"features_{cls}.png"
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Saved {out_path.name}")

    print(f"\nAll feature plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    plot_features()

