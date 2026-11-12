#!/usr/bin/env python3
"""
ai-service/benchmarks/index_check.py

Verifies frame-accurate boundary detection against ground-truth labels.csv
on 20 generated videos (4 per edit type: none, deletion, insertion, duplication, frame_drop).
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

# Add ai-service to sys.path
_HERE = Path(__file__).resolve().parent
AI_SERVICE = _HERE.parent
if str(AI_SERVICE) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE))

from service.inference import analyze_video

LABELS_CSV = AI_SERVICE / "dataset" / "labels.csv"
GENERATED_DIR = AI_SERVICE / "dataset" / "generated"


def run_index_check():
    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"labels.csv not found at {LABELS_CSV}")

    df = pd.read_csv(LABELS_CSV)
    # Pick 4 videos per edit type in order of appearance
    sample_df = df.groupby("edit_type", sort=False).head(4).copy()
    if len(sample_df) < 20:
        raise ValueError(f"Expected at least 20 test videos in labels.csv, found {len(sample_df)}")

    print(f"\nRunning frame-accuracy check on {len(sample_df)} generated videos (4 per edit type)...\n")

    results = []
    all_passed = True

    for _, row in sample_df.iterrows():
        video_filename = str(row["output_file"])
        exp_type = str(row["edit_type"])
        exp_s = int(row["start_frame"])
        exp_e = int(row["end_frame"])

        video_path = GENERATED_DIR / video_filename
        if not video_path.exists():
            raise FileNotFoundError(f"Video file missing: {video_path}")

        res = analyze_video(video_path)
        events = res.get("events", [])

        if exp_type == "none":
            if not events:
                det_type = "none"
                det_s, det_e = -1, -1
                err_s, err_e = 0, 0
                status = "PASS"
            else:
                ev = events[0]
                det_type = ev["type"]
                det_s, det_e = ev["start_frame"], ev["end_frame"]
                err_s = abs(det_s - exp_s) if exp_s != -1 else det_s
                err_e = abs(det_e - exp_e) if exp_e != -1 else det_e
                status = "FAIL"
                all_passed = False
        else:
            # Find primary event matching expected type (or first detected event)
            matched_ev = next((ev for ev in events if ev["type"] == exp_type), None)
            if matched_ev is None and events:
                matched_ev = events[0]

            if matched_ev is not None:
                det_type = matched_ev["type"]
                det_s = matched_ev["start_frame"]
                det_e = matched_ev["end_frame"]
                err_s = abs(det_s - exp_s)
                err_e = abs(det_e - exp_e)
                if det_type == exp_type and err_s == 0 and err_e == 0:
                    status = "PASS"
                else:
                    status = "FAIL"
                    all_passed = False
            else:
                det_type = "none"
                det_s, det_e = -1, -1
                err_s = abs(det_s - exp_s)
                err_e = abs(det_e - exp_e)
                status = "FAIL"
                all_passed = False

        results.append({
            "video": video_filename,
            "exp_type": exp_type,
            "det_type": det_type,
            "exp_start": exp_s,
            "exp_end": exp_e,
            "det_start": det_s,
            "det_end": det_e,
            "err_start": err_s,
            "err_end": err_e,
            "status": status,
        })

    # Print error table
    header_fmt = "{:<42} {:<12} {:<12} {:>9} {:>9} {:>9} {:>9} {:>8} {:>8} {:>8}"
    row_fmt = "{:<42} {:<12} {:<12} {:>9} {:>9} {:>9} {:>9} {:>8} {:>8} {:>8}"

    header = header_fmt.format(
        "Video", "Exp Type", "Det Type", "Exp Start", "Exp End", "Det Start", "Det End", "Err Start", "Err End", "Status"
    )
    separator = "-" * len(header)

    print(separator)
    print(header)
    print(separator)

    total_err_s = 0
    total_err_e = 0

    for r in results:
        print(row_fmt.format(
            r["video"],
            r["exp_type"],
            r["det_type"],
            r["exp_start"],
            r["exp_end"],
            r["det_start"],
            r["det_end"],
            r["err_start"],
            r["err_end"],
            r["status"],
        ))
        total_err_s += r["err_start"]
        total_err_e += r["err_end"]

    print(separator)
    passed_count = sum(1 for r in results if r["status"] == "PASS")
    print(f"\nSummary: {passed_count}/{len(results)} videos passed exact 0-error match.")
    print(f"Total start error: {total_err_s} frames | Total end error: {total_err_e} frames\n")

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    run_index_check()

