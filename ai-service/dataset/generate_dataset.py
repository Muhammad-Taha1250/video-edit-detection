#!/usr/bin/env python3
"""
Dataset Generator for Video Edit Detection & Frame Extraction System
=====================================================================

Creates a labelled dataset of edited and unedited videos from source clips.

Place ≥ 2 short (10–60 s) videos in  sources/  then run:
    python generate_dataset.py

For each source video it creates 6 variants:
  1. unedited copy          (label: none)
  2. re-encoded at low br   (label: none — hard negative)
  3. segment deletion       (label: deletion)
  4. segment insertion      (label: insertion)
  5. frame duplication      (label: duplication)
  6. frame dropping         (label: frame_drop)

Ground truth is written to  labels.csv.
"""

from __future__ import annotations

import argparse
import csv
import gc
import random
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    # lightweight fallback so the script still runs without tqdm
    def tqdm(iterable, **_kw):  # type: ignore[misc]
        return iterable


# ─────────────────────────── Paths & Constants ───────────────────────
ROOT        = Path(__file__).resolve().parent
SOURCES_DIR = ROOT / "sources"
OUTPUT_DIR  = ROOT / "generated"
LABELS_CSV  = ROOT / "labels.csv"

VIDEO_EXTS  = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MIN_FRAMES  = 90          # skip sources shorter than this (~3 s at 30 fps)
MARGIN_PCT  = 0.10        # keep 10 % buffer at head / tail so edits don't
                          # land right at the first or last frame

# Memory optimization: cap long side at 640, preserve aspect ratio with even dimensions
MAX_LONG_SIDE = 640
MAX_FPS       = 30.0
MAX_DURATION  = 30.0  # seconds


# ─────────────────────────── Utility Helpers ─────────────────────────

def check_ffmpeg() -> None:
    """Exit immediately with a clear error if ffmpeg is not on PATH."""
    try:
        res = subprocess.run(["ffmpeg", "-version"], capture_output=True)
        if res.returncode != 0:
            raise FileNotFoundError
    except (FileNotFoundError, subprocess.SubprocessError):
        sys.exit(
            "\n❌ ERROR: 'ffmpeg' not found on PATH.\n"
            "This pipeline requires FFmpeg with libx264 for identical encode/decode parity.\n"
            "Install it via Homebrew:\n\n"
            "    brew install ffmpeg\n"
        )


def get_video_rotation(path: str) -> int:
    """Extract display orientation rotation metadata (0, 90, 180, 270 degrees)."""
    # 1. Try ffprobe stream tags and side data
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream_tags=rotate:stream_side_data=rotation",
            "-of", "default=noprint_wrappers=1",
            path,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        for line in res.stdout.splitlines():
            if "rotate=" in line:
                val = int(float(line.split("rotate=")[1].strip()))
                return (val % 360 + 360) % 360
            if "rotation=" in line:
                val = int(float(line.split("rotation=")[1].strip()))
                return (val % 360 + 360) % 360
    except Exception:
        pass

    # 2. Try OpenCV CAP_PROP_ORIENTATION_META
    try:
        cap = cv2.VideoCapture(path)
        if hasattr(cv2, "CAP_PROP_ORIENTATION_META"):
            val = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
            cap.release()
            if val in (90, 180, 270):
                return val
        cap.release()
    except Exception:
        pass

    return 0


def compute_target_size(w: int, h: int, max_long_side: int = MAX_LONG_SIDE) -> tuple[int, int]:
    """Compute (target_w, target_h) preserving aspect ratio with long side = 640 and even dimensions."""
    if w >= h:
        tw = max_long_side
        th = int(round(max_long_side * h / w))
        th = max(2, (th // 2) * 2)
    else:
        th = max_long_side
        tw = int(round(max_long_side * w / h))
        tw = max(2, (tw // 2) * 2)
    return (tw, th)


def crop_and_resize(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Center-crop frame to target aspect ratio, then resize to exact (target_w, target_h). No letterboxing."""
    fh, fw = frame.shape[:2]
    target_aspect = target_w / target_h
    current_aspect = fw / fh

    if current_aspect > target_aspect + 1e-4:
        # Frame is wider than target -> crop left & right
        crop_w = int(round(fh * target_aspect))
        x1 = max(0, (fw - crop_w) // 2)
        x2 = min(fw, x1 + crop_w)
        cropped = frame[:, x1:x2]
    elif current_aspect < target_aspect - 1e-4:
        # Frame is taller than target -> crop top & bottom
        crop_h = int(round(fw / target_aspect))
        y1 = max(0, (fh - crop_h) // 2)
        y2 = min(fh, y1 + crop_h)
        cropped = frame[y1:y2, :]
    else:
        cropped = frame

    if cropped.shape[1] != target_w or cropped.shape[0] != target_h:
        cropped = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return cropped


def video_props(path: str) -> dict:
    """Return fps, frame count, display width/height, and target (w, h) preserving aspect ratio."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    raw_n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    raw_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if raw_fps <= 0 or raw_n <= 0 or raw_w <= 0 or raw_h <= 0:
        raise RuntimeError(f"Invalid video metadata for {path}")

    rot = get_video_rotation(path)
    # If rotated 90 or 270 degrees, display dimensions are swapped
    if rot in (90, 270):
        disp_w, disp_h = raw_h, raw_w
    else:
        disp_w, disp_h = raw_w, raw_h

    # Preserved aspect ratio with long side = 640 and even dimensions
    target_w, target_h = compute_target_size(disp_w, disp_h, MAX_LONG_SIDE)

    # Effective FPS capped at 30.0 fps
    fps = min(MAX_FPS, raw_fps) if raw_fps > 0 else MAX_FPS

    return {
        "raw_fps":  raw_fps,
        "fps":      fps,
        "n":        raw_n,
        "raw_w":    raw_w,
        "raw_h":    raw_h,
        "disp_w":   disp_w,
        "disp_h":   disp_h,
        "target_w": target_w,
        "target_h": target_h,
        "rotation": rot,
    }


def read_frames(
    path: str,
    target_w: int,
    target_h: int,
    rot: int = 0,
    raw_fps: float = MAX_FPS,
    max_duration_sec: float = MAX_DURATION,
    max_fps: float = MAX_FPS,
) -> list[np.ndarray]:
    """Read frames from video into memory:
    - Respects rotation metadata for vertical/rotated videos
    - Downscales immediately to (target_w, target_h) preserving aspect ratio
    - Caps frame rate at max_fps (drops intermediate frames if raw_fps > max_fps)
    - Caps total duration at max_duration_sec (trims excess frames, max 30s)
    """
    cap = cv2.VideoCapture(path)
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)

    effective_fps = min(max_fps, raw_fps) if raw_fps > 0 else max_fps
    max_frames = int(round(max_duration_sec * effective_fps))
    fps_stride = (raw_fps / max_fps) if (raw_fps > max_fps + 0.1) else 1.0

    frames: list[np.ndarray] = []
    raw_idx = 0
    next_sample_idx = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if raw_idx >= int(round(next_sample_idx)):
            # Respect rotation if OpenCV hasn't already auto-rotated
            if rot in (90, 270) and frame.shape[1] > frame.shape[0]:
                frame = cv2.rotate(
                    frame,
                    cv2.ROTATE_90_CLOCKWISE if rot == 90 else cv2.ROTATE_90_COUNTERCLOCKWISE,
                )
            elif rot == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            # Resize to target (target_w, target_h)
            if frame.shape[1] != target_w or frame.shape[0] != target_h:
                frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

            frames.append(frame)
            next_sample_idx += fps_stride

            # Cap total frames at 30 seconds
            if len(frames) >= max_frames:
                break

        raw_idx += 1

    cap.release()
    return frames


def encode_video(
    frames: list[np.ndarray],
    dst: Path,
    fps: float,
    w: int,
    h: int,
    crf: int = 23,
) -> None:
    """Encode frames via FFmpeg libx264 with identical settings across all classes.

    Ensures that encoding artifacts CANNOT reveal the class:
    - Every class (including unedited) is encoded through this exact rawvideo pipe.
    - libx264 with the chosen CRF (18-32), yuv420p pixel format.
    """
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-loglevel", "error",
        str(dst),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for f in frames:
            if f.shape[1] != w or f.shape[0] != h:
                f = cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR)
            proc.stdin.write(f.tobytes())
        proc.stdin.close()
        proc.wait()
    except Exception:
        proc.kill()
        raise
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg encoding failed with exit code {proc.returncode}")


def safe_randint(lo: int, hi: int) -> int:
    """``random.randint`` that silently clamps when *hi < lo*."""
    if hi < lo:
        hi = lo
    return random.randint(lo, hi)


def margin(n: int) -> int:
    """How many frames to keep untouched at the head / tail."""
    return max(1, int(n * MARGIN_PCT))


# ─────────────────────────── Edit Operations ─────────────────────────
#
# Every ``op_*`` function returns ``(edit_type, out_frames, start_frame, end_frame)``
# where ``start_frame`` and ``end_frame`` strictly refer to 0-indexed frame positions
# in the OUTPUT video.
#
#   * "none"        → (-1, -1)
#   * "deletion"    → (cut_point, cut_point)     — frame index in output where cut occurs
#   * "insertion"   → (first_foreign, last_foreign)
#   * "duplication" → (first_dup_copy, last_dup_copy) — range of the duplicated copy
#   * "frame_drop"  → (first_affected, last_affected)


def op_unedited(
    frames: list[np.ndarray],
    **_kw: object,
) -> tuple[str, list[np.ndarray], int, int]:
    """Unedited copy passed through identical decode->encode pipeline (positive control)."""
    return ("none", list(frames), -1, -1)


def op_deletion(
    frames: list[np.ndarray],
    fps: float,
    **_kw: object,
) -> tuple[str, list[np.ndarray], int, int]:
    """Remove a random contiguous segment (0.5 – 3 s).

    The cut discontinuity in the output video occurs at index ``start``:
    output[start - 1] was original[start - 1], output[start] was original[start + del_len].
    """
    n = len(frames)
    m = margin(n)
    min_del = max(1, int(fps * 0.5))
    max_del = min(int(fps * 3), n // 3)
    del_len = safe_randint(min_del, max_del)
    start   = safe_randint(m, n - del_len - m)

    out_frames = frames[:start] + frames[start + del_len:]
    return ("deletion", out_frames, start, start)


def op_insertion(
    frames: list[np.ndarray],
    donor_frames: list[np.ndarray],
    fps: float,
    w: int,
    h: int,
    donor_fps: float = 30.0,
    **_kw: object,
) -> tuple[str, list[np.ndarray], int, int]:
    """Splice in a segment from a donor video (0.5 – 2 s).

    Donor frames are center-cropped to the host's aspect ratio, then resized
    to the host's exact (w, h) with no letterboxing, and resampled to the host's fps.
    In the output video, the foreign donor frames occupy [point, point + ins_len - 1].
    """
    n  = len(frames)
    dn = len(donor_frames)
    m  = margin(n)

    # Random duration between 0.5s and 2.0s
    dur_sec = random.uniform(0.5, 2.0)
    target_len = max(1, min(int(round(dur_sec * fps)), n // 2))
    donor_len  = max(1, min(int(round(dur_sec * donor_fps)), dn // 2))

    donor_start = safe_randint(0, dn - donor_len)
    raw_donor   = donor_frames[donor_start : donor_start + donor_len]

    # Temporally resample donor frames to match target fps
    sample_indices = np.linspace(0, len(raw_donor) - 1, target_len).astype(int)
    resampled_donor = [raw_donor[i] for i in sample_indices]

    # Center-crop donor frames to host aspect ratio, then resize to exact (w, h) - no letterboxing
    segment = [crop_and_resize(f, w, h) for f in resampled_donor]

    point = safe_randint(m, n - m)
    out_frames = frames[:point] + segment + frames[point:]
    return ("insertion", out_frames, point, point + len(segment) - 1)


def op_duplication(
    frames: list[np.ndarray],
    fps: float,
    **_kw: object,
) -> tuple[str, list[np.ndarray], int, int]:
    """Duplicate a contiguous segment immediately after its original (0.3 – 2 s).

    In the output video, the duplicated copy occupies [end, end + dup_len - 1].
    """
    n = len(frames)
    m = margin(n)
    min_dup = max(1, int(fps * 0.3))
    max_dup = min(int(fps * 2), n // 4)
    dup_len = safe_randint(min_dup, max_dup)
    start   = safe_randint(m, n - dup_len - m)
    end     = start + dup_len

    segment    = frames[start:end]
    out_frames = frames[:end] + segment + frames[end:]
    return ("duplication", out_frames, end, end + dup_len - 1)


def op_frame_drop(
    frames: list[np.ndarray],
    fps: float,
    **_kw: object,
) -> tuple[str, list[np.ndarray], int, int]:
    """Drop every other frame in a segment (0.5 – 3 s).

    In the output video, the stuttered region occupies [start, start + len(kept) - 1].
    """
    n = len(frames)
    m = margin(n)
    min_seg = max(4, int(fps * 0.5))
    max_seg = min(int(fps * 3), n // 3)
    seg_len = safe_randint(min_seg, max_seg)
    start   = safe_randint(m, n - seg_len - m)
    end     = start + seg_len

    kept = [frames[i] for i in range(start, end) if (i - start) % 2 == 0]
    out_frames = frames[:start] + kept + frames[end:]
    return ("frame_drop", out_frames, start, start + len(kept) - 1)


# ─────────────────────────── Main Entry Point ────────────────────────

def generate_dataset(
    src_dir: Path | str = SOURCES_DIR,
    out_dir: Path | str = OUTPUT_DIR,
    labels_csv: Path | str = LABELS_CSV,
    rounds: int = 1,
    seed: int = 42,
) -> list[dict]:
    # Enforce FFmpeg installation upfront
    check_ffmpeg()

    src_dir = Path(src_dir)
    out_dir = Path(out_dir)
    labels_csv = Path(labels_csv)
    rounds  = max(1, rounds)

    # ── Discover source videos ───────────────────────────────────
    if not src_dir.exists():
        src_dir.mkdir(parents=True, exist_ok=True)
        sys.exit(
            f"Created {src_dir}/\n"
            f"Place 2+ short video clips (10–60 s, .mp4/.avi/.mov) there, "
            f"then re-run."
        )

    vids = sorted(
        f for f in src_dir.iterdir() if f.suffix.lower() in VIDEO_EXTS
    )
    if not vids:
        sys.exit(
            f"No videos found in {src_dir}/\n"
            f"Add .mp4/.avi/.mov files (10–60 s each) and re-run."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    labels_csv.parent.mkdir(parents=True, exist_ok=True)

    # ── Read metadata ONLY (do NOT keep frames in memory) ─────────
    print(f"\n{'=' * 55}")
    print(f"  Video Edit Detection — Dataset Generator")
    print(f"{'=' * 55}")
    print(f"\n📂  Source folder : {src_dir}/")
    print(f"📂  Output folder : {out_dir}/")
    print(f"🎲  Base seed     : {seed}")
    print(f"🔁  Rounds/source : {rounds}")
    print(f"📐  Resolution    : per-source, aspect-preserved, long side = {MAX_LONG_SIDE}")
    print(f"\nScanning {len(vids)} source video metadata …\n")

    source_props: dict[Path, dict] = {}
    for v in vids:
        try:
            props = video_props(str(v))
        except RuntimeError as e:
            print(f"  ⚠  Skipping {v.name}: {e}")
            continue

        if props["n"] < MIN_FRAMES:
            print(
                f"  ⚠  Skipping {v.name}: only {props['n']} frames "
                f"(need ≥ {MIN_FRAMES})"
            )
            continue

        source_props[v] = props
        secs = props["n"] / props["fps"]
        print(
            f"  ✓  {v.name}: {props['n']} frames, "
            f"{props['fps']:.1f} fps, {secs:.1f} s"
        )

    if not source_props:
        sys.exit("No usable source videos (all too short or unreadable).")

    all_sources = list(source_props.keys())

    # ── Generate dataset ─────────────────────────────────────────
    # 5 variants: unedited, deletion, insertion, duplication, frame_drop
    total_expected = len(all_sources) * 5 * rounds
    print(f"\nGenerating 5 variants × {rounds} round(s) per source ({total_expected} total videos) …\n")
    labels: list[dict] = []
    extra_gen_count = 0

    for round_idx in range(1, rounds + 1):
        # Deterministic, distinct seed derived per round
        round_seed = seed + round_idx
        random.seed(round_seed)
        np.random.seed(round_seed)

        if rounds > 1:
            print(f"\n--- Round {round_idx}/{rounds} (seed: {round_seed}) ---")

        for src_path in tqdm(all_sources, desc=f"Generating R{round_idx}"):
            p = source_props[src_path]
            stem = src_path.stem

            # Load frames ONLY for this single source, downscaled to aspect-preserving target size
            # Capped at 30s duration and 30 fps
            fr = read_frames(
                path=str(src_path),
                target_w=p["target_w"],
                target_h=p["target_h"],
                rot=p["rotation"],
                raw_fps=p["raw_fps"],
            )
            if len(fr) < MIN_FRAMES:
                del fr
                gc.collect()
                continue

            round_suffix = f"_r{round_idx}" if rounds > 1 else ""

            # 5 classes (reencode removed; multi-generation applied probabilistically)
            operations: list[tuple[str, str, object]] = [
                ("unedited",    f"unedited_{stem}{round_suffix}.mp4",     op_unedited),
                ("deletion",    f"deletion_{stem}{round_suffix}.mp4",     op_deletion),
                ("insertion",   f"insertion_{stem}{round_suffix}.mp4",    op_insertion),
                ("duplication", f"duplication_{stem}{round_suffix}.mp4",  op_duplication),
                ("frame_drop",  f"framedrop_{stem}{round_suffix}.mp4",    op_frame_drop),
            ]

            for op_tag, out_name, fn in operations:
                # Set donor_fr = None at the start of every operation loop iteration
                donor_fr: list[np.ndarray] | None = None
                out_path = out_dir / out_name

                try:
                    if op_tag == "insertion":
                        # Load donor frames on-demand at donor's target size, capped at 30s/30fps
                        donors = [s for s in all_sources if s != src_path]
                        donor_path = random.choice(donors) if donors else src_path
                        donor_sd = source_props[donor_path]
                        donor_fr = read_frames(
                            path=str(donor_path),
                            target_w=donor_sd["target_w"],
                            target_h=donor_sd["target_h"],
                            rot=donor_sd["rotation"],
                            raw_fps=donor_sd["raw_fps"],
                        )
                        donor_fps = donor_sd["fps"]
                    else:
                        donor_fps = p["fps"]

                    edit_type, out_frames, sf, ef = fn(
                        frames=fr,
                        donor_frames=donor_fr if donor_fr is not None else [],
                        fps=p["fps"],
                        w=p["target_w"],
                        h=p["target_h"],
                        donor_fps=donor_fps,
                    )

                    # Free donor frames immediately after use
                    if donor_fr is not None:
                        del donor_fr
                        donor_fr = None
                        gc.collect()

                    # Sample random CRF (18-32) identically for all classes
                    crf1 = random.randint(18, 32)
                    extra_gen = 1 if random.random() < 0.5 else 0

                    if extra_gen == 1:
                        # Generation 1 to temp file
                        temp_path = out_dir / f"__temp_{out_name}"
                        encode_video(out_frames, temp_path, p["fps"], p["target_w"], p["target_h"], crf=crf1)
                        # Generation 2: decode from compressed stream & re-encode at new random CRF
                        re_frames = read_frames(
                            path=str(temp_path),
                            target_w=p["target_w"],
                            target_h=p["target_h"],
                            rot=0,
                            raw_fps=p["fps"],
                        )
                        temp_path.unlink(missing_ok=True)
                        crf2 = random.randint(18, 32)
                        encode_video(re_frames, out_path, p["fps"], p["target_w"], p["target_h"], crf=crf2)
                        del re_frames
                        final_crf = crf2
                        extra_gen_count += 1
                    else:
                        encode_video(out_frames, out_path, p["fps"], p["target_w"], p["target_h"], crf=crf1)
                        final_crf = crf1

                    labels.append({
                        "output_file":       out_name,
                        "source_file":       src_path.name,
                        "edit_type":         edit_type,
                        "start_frame":       sf,
                        "end_frame":         ef,
                        "total_frames":      len(out_frames),
                        "fps":               p["fps"],
                        "quality_crf":       final_crf,
                        "extra_generations": extra_gen,
                    })

                    del out_frames
                    gc.collect()
                    gen_tag = " [2-gen]" if extra_gen else ""
                    print(f"    ✓ {out_name} (CRF {final_crf}, {p['target_w']}x{p['target_h']}){gen_tag}")

                except Exception as exc:
                    print(f"    ✗ {out_name}: {exc}")
                finally:
                    if donor_fr is not None:
                        del donor_fr
                        donor_fr = None
                        gc.collect()

            # Free source frames before loading next source
            del fr
            gc.collect()

    # ── Write labels.csv ─────────────────────────────────────────
    fieldnames = [
        "output_file",
        "source_file",
        "edit_type",
        "start_frame",
        "end_frame",
        "total_frames",
        "fps",
        "quality_crf",
        "extra_generations",
    ]
    with open(labels_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(labels)

    # ── Summary ──────────────────────────────────────────────────
    op_counts: Counter = Counter()
    for row in labels:
        op_counts[row["edit_type"]] += 1

    total_edited   = sum(c for k, c in op_counts.items() if k != "none")
    total_unedited = op_counts.get("none", 0)

    print(f"\n{'=' * 55}")
    print(f"  ✅  Dataset generation complete!")
    print(f"{'=' * 55}")
    print(f"  Source videos used       : {len(all_sources)}")
    print(f"  Videos generated         : {len(labels)}")
    print(f"  2-generation re-encoded  : {extra_gen_count} ({extra_gen_count/max(1, len(labels))*100:.1f}%)")
    print(f"  Labels written to        : {labels_csv}")
    print(f"\n  Class breakdown:")
    print(f"  {'─' * 50}")
    for cls in sorted(op_counts):
        print(f"    {cls:40s}  {op_counts[cls]}")
    print(f"  {'─' * 50}")
    print(f"    {'TOTAL unedited (clean)':40s}  {total_unedited}")
    print(f"    {'TOTAL edited':40s}  {total_edited}")
    print(f"    {'TOTAL':40s}  {len(labels)}")
    print(f"{'=' * 55}\n")

    return labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a labelled dataset for video-edit detection.",
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=SOURCES_DIR,
        help="Folder containing source videos (default: sources/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR,
        help="Where to write generated videos (default: generated/)",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=LABELS_CSV,
        help="Where to write labels CSV (default: labels.csv)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Number of randomized passes over each source video (default: 1)",
    )
    args = parser.parse_args()

    generate_dataset(
        src_dir=args.sources,
        out_dir=args.output,
        labels_csv=args.labels,
        rounds=args.rounds,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

