#!/usr/bin/env python3
"""
real_check.py

Runs analyze_video on a video file and compares ALL detected event boundaries
against ground-truth JSON with tolerance +/- 2 frames at original fps.
Prints detected events, matches, misses, and extras.

Usage:
    python ai-service/benchmarks/real_check.py [video_path] [ground_truth_json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add ai-service to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
AI_SERVICE = REPO_ROOT / "ai-service"
sys.path.insert(0, str(AI_SERVICE))

from service.inference import analyze_video


def run_check(video_path: Path, gt_path: Path, tolerance: int = 2) -> bool:
    print("=" * 70)
    print("REAL VIDEO GROUND TRUTH EVALUATION")
    print("=" * 70)
    print(f"Video:        {video_path}")
    print(f"Ground Truth: {gt_path}")
    print(f"Tolerance:    +/- {tolerance} frames")
    print("-" * 70)

    if not video_path.exists():
        print(f"ERROR: Video file not found: {video_path}")
        return False
    if not gt_path.exists():
        print(f"ERROR: Ground truth file not found: {gt_path}")
        return False

    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    gt_cuts = gt_data.get("cuts", [])
    gt_fps = gt_data.get("fps")

    print(f"Ground-truth cuts ({len(gt_cuts)}): {gt_cuts}")
    if gt_fps:
        print(f"Ground-truth FPS: {gt_fps}")
    print("-" * 70)
    print("Running analyze_video...")

    t0 = time.perf_counter()
    result = analyze_video(str(video_path))
    elapsed = time.perf_counter() - t0

    orig_fps = result.get("original_fps", result.get("fps", 30.0))
    analysis_fps = result.get("analysis_fps", 30.0)
    total_frames = result.get("total_frames", 0)
    duration = result.get("duration", 0.0)
    events = result.get("events", [])

    print(f"Analysis completed in {elapsed:.3f} s ({total_frames / elapsed:.1f} fps)")
    print(f"Reported FPS: {orig_fps:.3f} (analyzed at {analysis_fps:.1f} fps), Duration: {duration:.2f} s, Frames: {total_frames}")
    print("-" * 70)

    # 1. Print Detected Events
    print(f"\nDETECTED EVENTS ({len(events)}):")
    header = f"{'#':<3} | {'Type':<12} | {'Start':<7} | {'End':<7} | {'Start Time':<10} | {'End Time':<10} | {'Conf':<7} | {'Z-Score':<8}"
    print(header)
    print("-" * len(header))
    for idx, ev in enumerate(events):
        ev_type = ev.get("type", ev.get("edit_type", "unknown"))
        sf = ev.get("start_frame", 0)
        ef = ev.get("end_frame", 0)
        st = f"{ev.get('start_time', 0.0):.3f}s"
        et = f"{ev.get('end_time', 0.0):.3f}s"
        conf = f"{ev.get('confidence', 0.0):.3f}"
        z = f"{ev.get('evidence_z', 0.0):.1f}"
        print(f"{idx+1:<3} | {ev_type:<12} | {sf:<7} | {ef:<7} | {st:<10} | {et:<10} | {conf:<7} | {z:<8}")

    # 2. Compare with ground truth
    # Collect detected boundaries
    # Each event has start_frame and end_frame. If start_frame == end_frame, it's 1 boundary; otherwise 2 boundaries.
    detected_boundaries: list[dict] = []
    for ev_idx, ev in enumerate(events):
        sf = ev.get("start_frame", 0)
        ef = ev.get("end_frame", 0)
        ev_type = ev.get("type", ev.get("edit_type", "unknown"))
        detected_boundaries.append({
            "event_idx": ev_idx,
            "type": ev_type,
            "role": "start" if sf != ef else "junction",
            "frame": sf,
        })
        if sf != ef:
            detected_boundaries.append({
                "event_idx": ev_idx,
                "type": ev_type,
                "role": "end",
                "frame": ef,
            })

    matched_gt: set[int] = set()
    matched_det: set[int] = set()
    matches: list[dict] = []

    for gt_c in gt_cuts:
        best_match = None
        best_diff = 999999
        for d_idx, d_b in enumerate(detected_boundaries):
            diff = abs(d_b["frame"] - gt_c)
            if diff <= tolerance and diff < best_diff and d_idx not in matched_det:
                best_diff = diff
                best_match = (d_idx, d_b)

        if best_match is not None:
            d_idx, d_b = best_match
            matched_gt.add(gt_c)
            matched_det.add(d_idx)
            matches.append({
                "gt_cut": gt_c,
                "det_frame": d_b["frame"],
                "det_type": d_b["type"],
                "role": d_b["role"],
                "diff": d_b["frame"] - gt_c,
            })

    misses = [gt_c for gt_c in gt_cuts if gt_c not in matched_gt]
    extras = [d_b for d_idx, d_b in enumerate(detected_boundaries) if d_idx not in matched_det]

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS:")
    print("=" * 70)

    print(f"\nMATCHES ({len(matches)}/{len(gt_cuts)}):")
    if matches:
        for m in matches:
            print(f"  ✓ Ground Truth {m['gt_cut']:<5} -> Detected {m['det_type']} {m['role']} at {m['det_frame']:<5} (diff: {m['diff']:+d})")
    else:
        print("  None")

    print(f"\nMISSES ({len(misses)}):")
    if misses:
        for mis in misses:
            print(f"  ✗ Ground Truth {mis} not detected within +/-{tolerance} frames")
    else:
        print("  None (all ground truth cuts detected)")

    print(f"\nEXTRAS / FALSE POSITIVES ({len(extras)}):")
    if extras:
        for ex in extras:
            print(f"  ! Detected {ex['type']} {ex['role']} at frame {ex['frame']} (not in ground truth)")
    else:
        print("  None (clean detection without false positives)")

    print("-" * 70)
    success = (len(misses) == 0 and len(extras) == 0)
    if success:
        print("STATUS: PASS (100% precision and recall within tolerance)")
    else:
        print(f"STATUS: {'FAIL' if len(misses) > 0 else 'WARNING'} ({len(matches)}/{len(gt_cuts)} matches, {len(extras)} extras)")
    print("=" * 70)
    return success


def main():
    parser = argparse.ArgumentParser(description="Check real video ground truth boundaries")
    parser.add_argument(
        "video",
        nargs="?",
        default="server/uploads/IMG_0754.MOV",
        help="Path to video file (default: server/uploads/IMG_0754.MOV)",
    )
    parser.add_argument(
        "gt",
        nargs="?",
        default="ai-service/benchmarks/ground_truth/IMG_0754.json",
        help="Path to ground truth JSON (default: ai-service/benchmarks/ground_truth/IMG_0754.json)",
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=2,
        help="Tolerance in frames at original fps (default: 2)",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    gt_path = Path(args.gt)

    ok = run_check(video_path, gt_path, tolerance=args.tolerance)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

