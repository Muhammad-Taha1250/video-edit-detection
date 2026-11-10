#!/usr/bin/env python3
"""
Verification Script for Generated Video Edit Dataset
====================================================

Validates that:
1. Every video in labels.csv exists on disk and is readable.
2. Actual frame count matches total_frames recorded in labels.csv.
3. Edit boundaries [start_frame, end_frame] are valid indices in the output video.
4. Extracts sample frames around the labeled edit boundary as JPGs for visual inspection.

Usage:
    python verify_dataset.py
    python verify_dataset.py --samples-per-class 3
    python verify_dataset.py --extract-all
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT           = Path(__file__).resolve().parent
LABELS_CSV     = ROOT / "labels.csv"
GENERATED_DIR  = ROOT / "generated"
INSPECT_DIR    = ROOT / "inspection_frames"


def extract_frame_at(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
    """Seek to a 0-indexed frame and return the BGR image."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    return frame if ok else None


def annotate_frame(frame: np.ndarray, text: str, subtext: str = "") -> np.ndarray:
    """Draw a semi-transparent HUD overlay on the frame for easy eyeballing."""
    vis = frame.copy()
    h, w = vis.shape[:2]
    banner_h = 55 if subtext else 35

    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, vis, 0.35, 0, vis)

    cv2.putText(vis, text, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    if subtext:
        cv2.putText(vis, subtext, (15, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    return vis


def verify_dataset(
    labels_path: Path,
    videos_dir: Path,
    inspect_dir: Path,
    samples_per_class: int,
    extract_all: bool,
) -> bool:
    if not labels_path.exists():
        sys.exit(f"ERROR: {labels_path} not found. Run generate_dataset.py first.")

    with open(labels_path, newline="") as fh:
        rows = list(csv.DictReader(fh))

    if not rows:
        sys.exit(f"ERROR: {labels_path} is empty.")

    print(f"\n{'=' * 65}")
    print(f"  Dataset Verification & Frame Inspection")
    print(f"{'=' * 65}")
    print(f"📂  Labels file     : {labels_path}")
    print(f"📂  Videos folder   : {videos_dir}")
    print(f"🖼️   Inspection dir  : {inspect_dir}")
    print(f"Total videos to verify: {len(rows)}\n")

    inspect_dir.mkdir(parents=True, exist_ok=True)

    class_extracted_count: dict[str, int] = defaultdict(int)
    passed_count = 0
    failed_count = 0
    extracted_frames_total = 0

    print(f"{'Video File':<32} {'Edit Type':<13} {'Frames':<12} {'Edit Range':<15} {'Status'}")
    print(f"{'─' * 80}")

    for r in rows:
        vid_name   = r["output_file"]
        edit_type  = r["edit_type"]
        sf         = int(r["start_frame"])
        ef         = int(r["end_frame"])
        expected_n = int(r.get("total_frames", -1))
        vid_path   = videos_dir / vid_name

        if not vid_path.exists():
            print(f"{vid_name:<32} {edit_type:<13} {'MISSING':<12} {'-':<15} ❌ FILE MISSING")
            failed_count += 1
            continue

        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            print(f"{vid_name:<32} {edit_type:<13} {'UNREADABLE':<12} {'-':<15} ❌ CORRUPT")
            failed_count += 1
            continue

        actual_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if actual_n <= 0:
            # Fallback frame counting by reading
            actual_n = 0
            while True:
                ok, _ = cap.read()
                if not ok:
                    break
                actual_n += 1

        # Check frame count
        count_ok = (expected_n <= 0 or actual_n == expected_n)

        # Check bounds
        if edit_type == "none":
            bounds_ok = (sf == -1 and ef == -1)
            range_str = "N/A"
        else:
            bounds_ok = (0 <= sf <= ef < actual_n)
            range_str = f"[{sf}, {ef}]"

        frames_str = f"{actual_n}" if count_ok else f"{actual_n} (exp {expected_n})"

        if count_ok and bounds_ok:
            status_str = "✅ PASS"
            passed_count += 1
        else:
            status_str = "❌ FAIL"
            failed_count += 1

        print(f"{vid_name:<32} {edit_type:<13} {frames_str:<12} {range_str:<15} {status_str}")

        # Determine whether to extract inspection frames
        should_extract = (
            extract_all or
            (class_extracted_count[edit_type] < samples_per_class)
        )

        if should_extract and edit_type != "none":
            sample_dir = inspect_dir / Path(vid_name).stem
            sample_dir.mkdir(parents=True, exist_ok=True)

            # Pick target frame indices around edit boundary
            target_indices: list[tuple[int, str]] = []
            if edit_type == "deletion":
                # Junction where cut happened
                if sf - 1 >= 0:
                    target_indices.append((sf - 1, "before_cut"))
                target_indices.append((sf, "at_cut_boundary"))
                if sf + 1 < actual_n:
                    target_indices.append((sf + 1, "after_cut"))
            elif edit_type in ("insertion", "duplication"):
                if sf - 1 >= 0:
                    target_indices.append((sf - 1, "pre_edit"))
                target_indices.append((sf, "first_edited_frame"))
                if ef != sf:
                    target_indices.append((ef, "last_edited_frame"))
                if ef + 1 < actual_n:
                    target_indices.append((ef + 1, "post_edit"))
            elif edit_type == "frame_drop":
                if sf - 1 >= 0:
                    target_indices.append((sf - 1, "normal_speed"))
                target_indices.append((sf, "drop_start"))
                if ef < actual_n:
                    target_indices.append((ef, "drop_end"))
                if ef + 1 < actual_n:
                    target_indices.append((ef + 1, "resumed_speed"))

            for idx, tag in target_indices:
                frame = extract_frame_at(cap, idx)
                if frame is not None:
                    hdr = f"{vid_name} | Frame {idx:04d} | {edit_type.upper()}"
                    sub = f"Tag: {tag} (labeled range: [{sf}, {ef}])"
                    annotated = annotate_frame(frame, hdr, sub)
                    out_jpg = sample_dir / f"frame_{idx:04d}_{tag}.jpg"
                    cv2.imwrite(str(out_jpg), annotated)
                    extracted_frames_total += 1

            class_extracted_count[edit_type] += 1

        cap.release()

    print(f"{'─' * 80}")
    print(f"\nVerification Results:")
    print(f"  Total videos checked : {len(rows)}")
    print(f"  Passed validations   : {passed_count}")
    print(f"  Failed validations   : {failed_count}")
    print(f"  Inspection JPGs saved: {extracted_frames_total} frames in {inspect_dir}/")
    print(f"{'=' * 65}\n")

    return failed_count == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify generated video edit dataset.")
    parser.add_argument("--labels", type=Path, default=LABELS_CSV, help="Path to labels.csv")
    parser.add_argument("--videos", type=Path, default=GENERATED_DIR, help="Path to generated/ folder")
    parser.add_argument("--inspect-dir", type=Path, default=INSPECT_DIR, help="Folder for inspection JPGs")
    parser.add_argument("--samples-per-class", type=int, default=2, help="Samples per edit class to extract (default: 2)")
    parser.add_argument("--extract-all", action="store_true", help="Extract inspection frames for all videos")
    args = parser.parse_args()

    ok = verify_dataset(
        labels_path=args.labels,
        videos_dir=args.videos,
        inspect_dir=args.inspect_dir,
        samples_per_class=args.samples_per_class,
        extract_all=args.extract_all,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

