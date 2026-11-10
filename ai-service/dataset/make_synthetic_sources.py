#!/usr/bin/env python3
"""
Generate 10 varied synthetic source clips (10-15 s, 30 fps, mixed orientations)
using pure OpenCV. No FFmpeg or external downloads required.

Each clip has a fixed per-clip seed, varied backgrounds, moving shapes,
gradients, noise, and different motion speeds.
"""
from pathlib import Path
import cv2
import numpy as np

SOURCES_DIR = Path(__file__).resolve().parent / "sources"
SOURCES_DIR.mkdir(parents=True, exist_ok=True)


def _gradient_bg(h: int, w: int, c0: tuple, c1: tuple, axis: int = 0) -> np.ndarray:
    """Return an HxWx3 gradient frame blending c0 → c1 along the given axis."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    steps = h if axis == 0 else w
    for s in range(steps):
        t = s / max(steps - 1, 1)
        col = tuple(int(c0[k] * (1 - t) + c1[k] * t) for k in range(3))
        if axis == 0:
            frame[s, :] = col
        else:
            frame[:, s] = col
    return frame


CLIP_SPECS = [cd ~/Downloads/"frontend 4"/ai-service && source venv/bin/activate && cd dataset
    # (filename, seed, duration_sec, w, h, description, speed_factor)
    ("clip_01_landscape_slow.mp4",   1, 12, 640, 480, "Clip 01 Slow",   0.5),
    ("clip_02_landscape_fast.mp4",   2, 10, 640, 480, "Clip 02 Fast",   2.5),
    ("clip_03_portrait.mp4",         3, 15, 360, 640, "Clip 03 Portrait",1.0),
    ("clip_04_landscape_noise.mp4",  4, 11, 640, 360, "Clip 04 Noisy",  1.2),
    ("clip_05_widescreen.mp4",       5, 13, 640, 272, "Clip 05 Wide",   1.8),
    ("clip_06_landscape_grad.mp4",   6, 10, 640, 480, "Clip 06 Grad",   0.8),
    ("clip_07_portrait_fast.mp4",    7, 12, 360, 640, "Clip 07 P-Fast", 3.0),
    ("clip_08_square_like.mp4",      8, 14, 480, 480, "Clip 08 Square", 1.0),
    ("clip_09_landscape_multi.mp4",  9, 15, 640, 480, "Clip 09 Multi",  1.5),
    ("clip_10_landscape_dark.mp4",  10, 11, 640, 360, "Clip 10 Dark",   0.7),
]


def create_clip(spec: tuple):
    filename, seed, duration_sec, w, h, label, speed = spec
    rng = np.random.default_rng(seed)
    total_frames = duration_sec * 30

    # Choose background style deterministically per seed
    bg_style = seed % 4  # 0=solid, 1=h-grad, 2=v-grad, 3=diagonal-grad
    c0 = tuple(int(x) for x in rng.integers(10, 200, 3))
    c1 = tuple(int(x) for x in rng.integers(10, 200, 3))

    # Random shapes to draw each frame: 2-4 moving objects
    n_shapes = int(rng.integers(2, 5))
    shape_types  = rng.integers(0, 3, n_shapes)   # 0=circle, 1=rect, 2=line
    shape_colors = [tuple(int(x) for x in rng.integers(50, 255, 3)) for _ in range(n_shapes)]
    shape_sizes  = rng.integers(15, 55, n_shapes)
    shape_speeds = rng.uniform(0.5, 3.0, n_shapes) * speed
    shape_phases = rng.uniform(0, 2 * np.pi, n_shapes)
    noise_sigma  = float(rng.uniform(0, 20))   # per-clip fixed noise level

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    path   = SOURCES_DIR / filename
    out    = cv2.VideoWriter(str(path), fourcc, 30.0, (w, h))

    # Pre-build gradient bg once if static
    static_bg = None
    if bg_style == 1:
        static_bg = _gradient_bg(h, w, c0, c1, axis=1)
    elif bg_style == 2:
        static_bg = _gradient_bg(h, w, c0, c1, axis=0)
    elif bg_style == 3:
        # diagonal: blend both axes
        hg = _gradient_bg(h, w, c0, c1, axis=0).astype(np.float32)
        vg = _gradient_bg(h, w, c1, c0, axis=1).astype(np.float32)
        static_bg = np.clip((hg + vg) / 2, 0, 255).astype(np.uint8)

    for i in range(total_frames):
        t = i / 30.0  # time in seconds

        if bg_style == 0:
            frame = np.full((h, w, 3), c0, dtype=np.uint8)
        else:
            frame = static_bg.copy()

        # Add per-frame noise
        if noise_sigma > 1:
            noise = rng.normal(0, noise_sigma, (h, w, 3)).astype(np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Draw moving shapes
        for s in range(n_shapes):
            sp  = shape_speeds[s]
            ph  = shape_phases[s]
            sz  = int(shape_sizes[s])
            col = shape_colors[s]
            st  = shape_types[s]

            cx = int((w * 0.1) + (w * 0.8) * (0.5 + 0.5 * np.sin(sp * t + ph)))
            cy = int((h * 0.1) + (h * 0.8) * (0.5 + 0.5 * np.cos(sp * t * 0.7 + ph)))

            if st == 0:
                cv2.circle(frame, (cx, cy), sz, col, -1)
            elif st == 1:
                x1, y1 = max(0, cx - sz), max(0, cy - sz)
                x2, y2 = min(w - 1, cx + sz), min(h - 1, cy + sz)
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, -1)
            else:
                ex = int(cx + sz * 2 * np.cos(sp * t))
                ey = int(cy + sz * 2 * np.sin(sp * t))
                ex = np.clip(ex, 0, w - 1)
                ey = np.clip(ey, 0, h - 1)
                cv2.line(frame, (cx, cy), (ex, ey), col, 3)

        # HUD overlay
        cv2.putText(frame, label, (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"f{i:04d}/{total_frames}",
                    (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        out.write(frame)

    out.release()
    print(f"  ✓ {filename}  ({duration_sec}s · {total_frames}f · {w}×{h} · "
          f"speed×{speed} · noise={noise_sigma:.1f})")


if __name__ == "__main__":
    print(f"\nGenerating {len(CLIP_SPECS)} synthetic source clips → {SOURCES_DIR}/\n")
    for spec in CLIP_SPECS:
        create_clip(spec)
    print("\nDone! Populate sources/ with your own real videos or use these for a quick smoke-test.\n")
