# Dataset Generator — Video Edit Detection

Generates a labelled dataset of edited and unedited videos for training the edit-detection models.

## Prerequisites

| Tool | Install command |
|------|----------------|
| **Python 3.11+** | Already on your Mac (check: `python3 --version`) |
| **FFmpeg** | `brew install ffmpeg` |
| **pip packages** | Installed via `requirements.txt` (see below) |

## Setup (one-time)

```bash
# 1. Navigate to ai-service and create a virtual environment
cd ai-service
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

## What to Put in `sources/`

Place **2 or more** short video clips (10–60 seconds each) into:

```
ai-service/dataset/sources/
```

### Good sources of clips

| Option | How |
|--------|-----|
| **Record your own** | Use QuickTime → File → New Screen Recording for 15–30 s |
| **Phone camera** | Record a short clip and AirDrop it to your Mac |
| **Pexels (free)** | Download Creative Commons clips from [pexels.com/videos](https://www.pexels.com/videos/) |
| **FFmpeg test pattern** | `ffmpeg -f lavfi -i testsrc=duration=20:size=640x480:rate=30 -c:v libx264 sources/test1.mp4` |

**Requirements:**
- At least 2 different video files (so insertion edits can splice from a different source)
- Each clip should be **at least 3 seconds** long (≥ 90 frames at 30 fps)
- Formats: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`
- Keep clips **under 60 seconds** — the generator reads all frames into memory

### Quick test: generate synthetic clips (no external downloads needed)

You can run the included OpenCV script to generate 2 synthetic test clips automatically:

```bash
cd dataset
python make_synthetic_sources.py
```

Or generate them using FFmpeg test patterns:

```bash
# Create two 15-second test videos
ffmpeg -f lavfi -i testsrc=duration=15:size=640x480:rate=30 \
       -c:v libx264 -pix_fmt yuv420p dataset/sources/test_pattern1.mp4

ffmpeg -f lavfi -i "color=c=blue:size=640x480:rate=30:duration=15,drawtext=text='Test 2':fontsize=40:fontcolor=white:x=250:y=220" \
       -c:v libx264 -pix_fmt yuv420p dataset/sources/test_pattern2.mp4
```

## Run the Generator

```bash
# Make sure the venv is active
source venv/bin/activate

# Run it
cd dataset
python generate_dataset.py
```

### Options

```bash
python generate_dataset.py --sources /path/to/vids --output /path/to/out --seed 42 --rounds 2
```

| Flag | Default | Description |
|------|---------|-------------|
| `--sources` | `sources/` | Folder of input videos |
| `--output` | `generated/` | Where generated videos are written |
| `--seed` | `42` | Random seed for reproducibility |
| `--rounds` | `1` | Number of randomized passes over each source video |

## Output

After running, you'll have:

```
ai-service/dataset/
├── sources/              ← your original clips (unchanged)
├── generated/            ← all generated videos (6 × rounds per source)
│   ├── unedited_clip1.mp4
│   ├── reencode_clip1.mp4
│   ├── deletion_clip1.mp4
│   ├── insertion_clip1.mp4
│   ├── duplication_clip1.mp4
│   ├── framedrop_clip1.mp4
│   └── ...
└── labels.csv            ← ground truth annotations
```

### `labels.csv` columns

| Column | Description |
|--------|-------------|
| `output_file` | Filename of generated video |
| `source_file` | Original source video it was derived from (used for train/test split) |
| `edit_type` | `none`, `deletion`, `insertion`, `duplication`, or `frame_drop` |
| `start_frame` | 0-indexed frame where edit starts in the **output** video (−1 for none) |
| `end_frame` | 0-indexed frame where edit ends in the **output** video (−1 for none) |
| `total_frames` | Total frames in the generated output video |
| `fps` | Frames per second of the video |
| `quality_crf` | H.264 CRF quality level (18–32) applied independently of edit class |
| `extra_generations` | `1` if an extra decode->re-encode generation was applied (50% prob), else `0` |

### What each variant does

All variants pass through the **exact same decode -> encode pipeline** at 640x360 resolution with randomized CRF levels (18–32). With 50% probability, an extra re-encoding generation is applied across all classes to simulate social media / multi-generation re-compression.

| File prefix | Label | What happens | Output Frame Range |
|-------------|-------|--------------|-------------------|
| `unedited_*` | `none` | Decoded & re-encoded (50% extra gen) — positive control | `[-1, -1]` |
| `deletion_*` | `deletion` | A random 0.5–3 s segment is cut out | `[cut_idx, cut_idx]` |
| `insertion_*` | `insertion` | A 0.5–2 s segment from another source is spliced in | `[first_donor, last_donor]` |
| `duplication_*` | `duplication` | A 0.3–2 s segment is duplicated right after itself | `[first_dup, last_dup]` |
| `framedrop_*` | `frame_drop` | Every other frame in a 0.5–3 s segment is dropped | `[first_drop, last_drop]` |

## Step 2 — Verify the Dataset & Eyeball Edits

Run the verification script to confirm frame counts and extract annotated sample JPGs around every edit:

```bash
# Verify frame counts and extract 2 sample videos per edit class
python verify_dataset.py

# Or extract inspection frames for all generated videos
python verify_dataset.py --extract-all
```

Inspection JPGs with visual HUD overlays will be saved to:
```
ai-service/dataset/inspection_frames/<video_stem>/frame_<index>_<tag>.jpg
```
You can open this folder in macOS Finder (`open inspection_frames`) to visually confirm each cut point, splice, and duplication!

## Exact Commands (copy-paste)

```bash
# Full setup from scratch
cd "/Users/tahasmacbook/Downloads/frontend 4/ai-service"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install ffmpeg if you haven't (for libx264 encoding)
brew install ffmpeg

# Populate sources/ (either drop your own .mp4 files or generate synthetic clips)
cd dataset
python make_synthetic_sources.py

# Generate dataset (e.g. 1 round or 2 rounds)
python generate_dataset.py --rounds 1 --seed 42

# Verify frame counts and extract inspection JPGs
python verify_dataset.py

# Eyeball the extracted frames in Finder
open inspection_frames
```

