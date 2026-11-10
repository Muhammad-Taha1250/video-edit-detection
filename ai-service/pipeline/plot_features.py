"""
plot_features.py – Visualise features over time for one sample video per class.

For each class (including 'none' for clean unedited, and each edit type),
picks a sample video from the feature table, plots the 7 raw features over time,
and draws vertical lines/spans at the labeled edit boundaries.

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
MODELS_DIR    = AI_SERVICE / "models"
PLOTS_DIR     = MODELS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RAW_FEAT_NAMES = FEATURE_NAMES[:7]   # first 7 are raw (un-normalised)


def _find_labels_csv() -> Path:
    candidates = [
        GENERATED_DIR / "labels.csv",
        AI_SERVICE / "dataset" / "labels.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return GENERATED_DIR / "labels.csv"


def _edit_boundaries(labels_df: pd.DataFrame, video_name: str) -> list[tuple[int, int, str]]:
    """Return [(start_frame, end_frame, edit_type)] for a given output video."""
    sub = labels_df[labels_df["output_file"].str.endswith(video_name)]
    out = []
    for _, row in sub.iterrows():
        etype = str(row["edit_type"])
        start = int(row["start_frame"])
        end   = int(row["end_frame"])
        if etype not in ("unedited", "none") and start >= 0 and end >= 0:
            out.append((start, end, etype))
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

    labels_csv = _find_labels_csv()
    if not labels_csv.exists():
        raise FileNotFoundError(f"labels.csv not found at {labels_csv}")

    labels_df = pd.read_csv(labels_csv)

    classes = sorted(list(table["label"].unique()))
    if not classes:
        print("No classes found – nothing to plot.")
        return

    print(f"\nGenerating feature plots for classes: {classes}")

    for cls in tqdm(classes, desc="Plotting features"):
        if cls == "none":
            # Pick a purely unedited video if possible
            unedited_videos = labels_df[labels_df["edit_type"].isin(["none", "unedited"])]["output_file"].tolist()
            cand = [v for v in unedited_videos if (GENERATED_DIR / Path(v).name).exists()]
            video_name = Path(cand[0]).name if cand else table[table["label"] == cls]["video"].iloc[0]
        else:
            sub = table[table["label"] == cls]
            if sub.empty:
                continue
            video_name = sub["video"].iloc[0]

        video_path = GENERATED_DIR / video_name
        if not video_path.exists():
            continue

        try:
            feats = extract_features(video_path, desc=None)
        except Exception as exc:
            print(f"  ⚠ Cannot load features for {video_name}: {exc}")
            continue

        boundaries = _edit_boundaries(labels_df, video_name)
        n_pairs = len(feats)
        pair_idx = np.arange(n_pairs)

        fig, axes = plt.subplots(
            len(RAW_FEAT_NAMES), 1,
            figsize=(14, 2.0 * len(RAW_FEAT_NAMES)),
            sharex=True
        )
        title_extra = f" (Boundaries: {len(boundaries)})" if boundaries else " (Clean / No edits)"
        fig.suptitle(f"Feature Trajectories — Class: {cls} | Video: {video_name}{title_extra}", fontsize=11, y=0.99)

        for ax, fname in zip(axes, RAW_FEAT_NAMES):
            col_idx = FEATURE_NAMES.index(fname)
            ax.plot(pair_idx, feats[:, col_idx], linewidth=0.9, color="#2176ae")
            ax.set_ylabel(fname, fontsize=7, rotation=0, labelpad=75, va="center")
            ax.yaxis.set_label_position("left")
            ax.tick_params(labelsize=7)
            ax.grid(True, linestyle=":", alpha=0.5)

            for start, end, etype in boundaries:
                ax.axvspan(
                    max(start - 3, 0), min(end + 3, n_pairs - 1),
                    alpha=0.18, color="red",
                    label=etype if fname == RAW_FEAT_NAMES[0] else ""
                )
                ax.axvline(start, color="red", linestyle="--", linewidth=1.0, alpha=0.8)
                ax.axvline(end, color="darkorange", linestyle=":", linewidth=1.0, alpha=0.8)

        if boundaries and axes[0].get_legend_handles_labels()[0]:
            axes[0].legend(fontsize=7, loc="upper right")
        axes[-1].set_xlabel("Frame Pair Index", fontsize=9)
        plt.tight_layout()

        out_path = PLOTS_DIR / f"features_{cls}.png"
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Saved {out_path.name}")

    print(f"\nAll feature plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    plot_features()
