# Day 2 Diagnosis: Video Edit Detection Pipeline

**Dataset Context:** Synthetic smoke-test dataset (10 source clips, 100 generated videos, 36,854 frame pairs).
**Model Evaluated:** Balanced Random Forest (300 estimators) with GroupShuffleSplit (80/20 train/test by `source_file`) and GroupKFold ($k=5$).

---

## 1. Label Audit

### 1.1 Class Counts by Dataset Partition

The table below shows the distribution of labeled frame pairs across the full table, the 8-source training split, and the 2-source held-out test split.

| Class | Full Table Rows | Train Split (8 sources) | Test Split (2 sources) | Test Share (%) |
|---|---|---|---|---|
| `none` (unedited) | 34,433 (93.43%) | 27,386 (93.27%) | 7,047 (94.07%) | 20.47% |
| `duplication` | 812 (2.20%) | 704 (2.40%) | 108 (1.44%) | 13.30% |
| `insertion` | 812 (2.20%) | 632 (2.15%) | 180 (2.40%) | 22.17% |
| `frame_drop` | 657 (1.78%) | 529 (1.80%) | 128 (1.71%) | 19.48% |
| `deletion` | 140 (0.38%) | 112 (0.38%) | 28 (0.37%) | 20.00% |
| **Total** | **36,854** | **29,363** | **7,491** | **20.33%** |

> **Key Observation on Class Balance:** `deletion` has only 140 rows total (28 in test), because each deletion is a single cut point padded by $\pm 3$ frames ($7$ rows per video $\times 20$ videos). In contrast, `insertion` and `duplication` span 30–60 frames each, generating $5\times$ to $6\times$ more labeled rows per video despite having the same number of videos (20 each).

---

### 1.2 Video-Level Label vs. Real Discontinuity Audit (3 Samples per Edit Type)

| Edit Type | Output Video | `labels.csv` Range | `feature_table` Labeled Pairs | Total Pairs Labeled | Physical Discontinuity in Output Video | Ground Truth Mismatch |
|---|---|---|---|---|---|---|
| **Deletion** | `deletion_clip_01_landscape_slow_r1.mp4` | [272, 272] | [269, 275] | 7 | Exact cut at frame 272 (pair 271 $\to$ 272) | **Single-frame cut vs. 7-pair window**: Only pair 271 $\to$ 272 contains the cut discontinuity. The remaining 6 pairs are normal, consecutive pre/post frames forced to be labeled as "deletion". |
| | `deletion_clip_02_landscape_fast_r1.mp4` | [71, 71] | [68, 74] | 7 | Exact cut at frame 71 (pair 70 $\to$ 71) | Same: 6 of 7 labeled rows have zero cut signal. |
| | `deletion_clip_03_portrait_r1.mp4` | [76, 76] | [73, 79] | 7 | Exact cut at frame 76 (pair 75 $\to$ 76) | Same: 6 of 7 labeled rows have zero cut signal. |
| **Duplication** | `duplication_clip_01_landscape_slow_r1.mp4` | [302, 349] | [299, 352] | 54 | Loopback jump at frame 302 (jump back 48 frames); frame 349 $\to$ 350 is smooth continuation. | **Loopback cut vs. 54-pair span**: Only pair 301 $\to$ 302 jumps in content. Inside [302, 349], consecutive pairs are smooth, natural donor frames with zero difference. |
| | `duplication_clip_02_landscape_fast_r1.mp4` | [190, 201] | [187, 204] | 18 | Loopback jump at frame 190 (jump back 12 frames). | 1 boundary junction vs. 18 labeled pairs. |
| | `duplication_clip_03_portrait_r1.mp4` | [322, 374] | [319, 377] | 59 | Loopback jump at frame 322 (jump back 53 frames). | 1 boundary junction vs. 59 labeled pairs. |
| **Frame Drop** | `framedrop_clip_01_landscape_slow_r1.mp4` | [136, 170] | [133, 173] | 41 | Motion speed $2\times$ throughout [136, 170]. No scene cut. | **Distributed rate change**: No spatial discontinuity exists at any single frame pair; the motion is simply accelerated by $2\times$. |
| | `framedrop_clip_02_landscape_fast_r1.mp4` | [105, 122] | [102, 125] | 24 | Motion speed $2\times$ throughout [105, 122]. No scene cut. | Same: Distributed rate change. |
| | `framedrop_clip_03_portrait_r1.mp4` | [70, 97] | [67, 100] | 34 | Motion speed $2\times$ throughout [70, 97]. No scene cut. | Same: Distributed rate change. |
| **Insertion** | `insertion_clip_01_landscape_slow_r1.mp4` | [45, 79] | [42, 82] | 41 | Junction 1 at frame 45 (host $\to$ donor); Junction 2 at frame 80 (donor $\to$ host). | **Two boundary cuts vs. 41-pair span**: Only pairs 44 $\to$ 45 and 79 $\to$ 80 exhibit spliced boundary jumps. The 33 intermediate pairs are regular, consecutive frames from the donor clip. |
| | `insertion_clip_02_landscape_fast_r1.mp4` | [99, 148] | [96, 151] | 56 | Junction 1 at frame 99; Junction 2 at frame 149. | 2 boundary cuts vs. 56 labeled pairs (93% normal frames). |
| | `insertion_clip_03_portrait_r1.mp4` | [214, 263] | [211, 266] | 56 | Junction 1 at frame 214; Junction 2 at frame 264. | 2 boundary cuts vs. 56 labeled pairs (93% normal frames). |

---

## 2. Signal Check

### 2.1 ROC-AUC: $\pm 3$ Frame Window vs. `none`

ROC-AUC evaluated over all videos comparing frame pairs labeled as edit ($\pm 3$ frame window) against the 34,433 `none` frame pairs:

| Feature Name | Deletion ($\pm 3$f) | Duplication ($\pm 3$f) | Frame Drop ($\pm 3$f) | Insertion ($\pm 3$f) |
|---|---|---|---|---|
| `hsv_hist_diff` | 0.5714 | 0.5889 | 0.4938 | 0.6108 |
| `ssim_dist` | 0.5534 | 0.5173 | 0.4281 | 0.5186 |
| `pixel_mad` | 0.5831 | 0.5499 | 0.5264 | 0.5904 |
| `edge_diff` | 0.4975 | 0.5046 | 0.4934 | 0.4513 |
| `phash_dist` | 0.5678 | 0.5432 | 0.6410 | 0.6637 |
| `flow_mag` | 0.5409 | 0.5282 | 0.5391 | 0.5315 |
| `orb_ratio` | 0.4950 | 0.5012 | 0.4598 | 0.3849 |
| `ctx_hsv` | 0.5348 | 0.5180 | 0.5138 | 0.5347 |
| `ctx_ssim` | 0.5400 | 0.4850 | 0.4996 | 0.4896 |
| `ctx_mad` | 0.5493 | 0.5196 | 0.5038 | 0.4899 |
| `ctx_edge` | 0.5183 | 0.5686 | 0.4991 | 0.4912 |
| `ctx_phash` | 0.5518 | 0.5106 | 0.5037 | 0.5194 |
| `ctx_flow` | 0.5094 | 0.5270 | 0.5235 | 0.4921 |
| `ctx_orb` | 0.5320 | 0.4841 | 0.4978 | 0.4690 |

### 2.2 Isolating the Exact Junction Cut Point

When the evaluation isolates the **exact single cut pair** (e.g. for `deletion`, the pair connecting frame $\text{cut}-1 \to \text{cut}$) rather than the diluted $\pm 3$ frame window:

| Feature | Exact Junction ROC-AUC | Exact Junction Mean | `none` Baseline Mean |
|---|---|---|---|
| `ctx_mad` (context MAD ratio) | **0.9966** | **2.65** | **1.00** |
| `phash_dist` (pHash Hamming distance) | **0.9957** | **28.80** | **6.75** |
| `pixel_mad` (mean absolute difference) | **0.8887** | **8.55** | **3.92** |
| `ctx_hsv` (context HSV diff ratio) | **0.8817** | **2.63** | **1.08** |
| `ctx_ssim` (context SSIM ratio) | **0.8481** | **1.40** | **1.01** |
| `ctx_flow` (context optical flow ratio) | **0.8308** | **2.15** | **0.99** |
| `hsv_hist_diff` (raw HSV diff) | **0.8135** | **0.10** | **0.04** |
| `flow_mag` (raw optical flow) | **0.7561** | **1.32** | **0.83** |

> **Conclusion on Features:** The visual features are **strongly reactive** to actual edit cuts (ROC-AUC $> 0.99$). The low ROC-AUC in Section 2.1 is caused entirely by diluting the single cut point with 6 adjacent normal frames ($\pm 3$), and in insertions/duplications, by labeling 30–50 consecutive normal frames as positive.

---

### 2.3 Feature Statistics: `duplication` & `frame_drop` vs. `none`

| Feature | `duplication` Mean $\pm$ Std | `duplication` Max | `frame_drop` Mean $\pm$ Std | `frame_drop` Max | `none` Mean $\pm$ Std | `none` Max |
|---|---|---|---|---|---|---|
| `flow_mag` | $0.781 \pm 0.529$ | 3.273 | $0.899 \pm 0.547$ | 2.646 | $0.834 \pm 0.552$ | 3.187 |
| `ctx_flow` | $1.024 \pm 0.392$ | 7.495 | $1.025 \pm 0.215$ | 1.983 | $0.993 \pm 0.129$ | 2.290 |
| `pixel_mad` | $3.658 \pm 2.441$ | 12.396 | $4.066 \pm 2.444$ | 10.027 | $3.917 \pm 2.438$ | 12.161 |
| `ctx_mad` | $1.050 \pm 0.434$ | 6.123 | $1.012 \pm 0.167$ | 1.949 | $1.003 \pm 0.129$ | 5.189 |
| `ssim_dist` | $0.151 \pm 0.148$ | 0.566 | $0.119 \pm 0.103$ | 0.434 | $0.172 \pm 0.155$ | 0.652 |
| `ctx_ssim` | $1.011 \pm 0.166$ | 2.743 | $1.025 \pm 0.256$ | 4.794 | $1.005 \pm 0.119$ | 4.228 |
| `phash_dist` | $6.855 \pm 6.538$ | 37.000 | $10.935 \pm 8.579$ | 41.000 | $6.750 \pm 5.495$ | 33.000 |

#### Insights on Duplication and Frame Drop:
1. **Duplication:** The mean features inside the duplication range ($0.781$ flow, $3.658$ MAD) are virtually identical to `none` ($0.834$ flow, $3.917$ MAD). The only signal is the solitary loopback junction at the start (reflected by `ctx_flow` max = 7.495 and `ctx_mad` max = 6.123).
2. **Frame Drop:** `flow_mag` ($0.899$) and `pixel_mad` ($4.066$) are only slightly higher than `none` ($0.834$ and $3.917$). In a static frame-pair classifier, natural motion variation across clips (speed factor $0.5\times$ to $3.0\times$) completely swamps the $2\times$ acceleration from dropped frames.

---

## 3. Test Split Audit

### 3.1 Partitioning by Source Video (`GroupShuffleSplit` 80/20)

- **Training Sources (8 clips):** `clip_01_landscape_slow.mp4`, `clip_03_portrait.mp4`, `clip_04_landscape_noise.mp4`, `clip_05_widescreen.mp4`, `clip_06_landscape_grad.mp4`, `clip_07_portrait_fast.mp4`, `clip_08_square_like.mp4`, `clip_10_landscape_dark.mp4`.
- **Held-Out Test Sources (2 clips):** `clip_02_landscape_fast.mp4`, `clip_09_landscape_multi.mp4`.

### 3.2 Test Rows by Source File and Class

| Test Source Video | `none` | `insertion` | `frame_drop` | `duplication` | `deletion` | Total Rows |
|---|---|---|---|---|---|---|
| `clip_02_landscape_fast.mp4` | 2,751 | 89 | 68 | 51 | 14 | 2,973 |
| `clip_09_landscape_multi.mp4` | 4,296 | 91 | 60 | 57 | 14 | 4,518 |
| **Total Test Split** | **7,047** | **180** | **128** | **108** | **28** | **7,491** |

> **Sample Count Warning for Test Split:** `deletion` has only **28 positive rows** across the entire test set (14 rows from each of 2 test clips, representing just 2 actual cut events per clip across rounds). A miss on even 2 cut windows drops recall significantly.

---

## 4. Root Cause Analysis: What Caused Low Recall?

Based on the quantitative evidence, the primary and secondary causes are:

### Primary Cause: (a) Label Design (Entire Span Labeled Instead of Junctions)
- **The Mechanism:** 
  - In `build_table.py`, every frame pair in $[start - 3, end + 3]$ was labeled with the edit class. 
  - For `insertion` (30–50 frames) and `duplication` (15–55 frames), **only 1 or 2 frame pairs are actual boundary cuts**. The remaining 90–95% of pairs are identical to normal video frames.
  - The model was punished during training for predicting `none` on ordinary donor frames inside an insertion, and during evaluation, recall was computed against a denominator inflated with normal frames that have zero physical edit signature.
- **The Evidence:** 
  - When the exact junction cut is isolated, features have ROC-AUC of **0.9966** (`ctx_mad`) and **0.9957** (`phash_dist`).
  - At the single junction point, Random Forest predicted `deletion` with **100% precision** and segment extraction localized the cut with exact boundary tolerance (frame 269–275 around cut 272).
  - The per-frame recall is low because 90%+ of the labeled "positive" frames in the denominator are indistinguishable from clean video by definition.

### Secondary Cause: (d) Synthetic Artifacts & Static Frame-Pair Limitation for `frame_drop`
- **The Mechanism:** 
  - `frame_drop` produces no boundary cut. It produces a $2\times$ motion increase over consecutive frames.
  - However, synthetic sources have randomized speed multipliers ranging from $0.5\times$ to $3.0\times$. 
  - The test split contains `clip_02_landscape_fast` (speed factor $2.5\times$) and `clip_09_landscape_multi` (speed factor $1.5\times$). A clean unedited frame in `clip_02` has higher optical flow and MAD than a `frame_drop` frame in `clip_01_landscape_slow`.
- **The Evidence:** 
  - `frame_drop` feature distributions overlap almost completely with `none` (mean flow $0.899$ vs. $0.834$). 
  - Without a temporal sequence model (1D-CNN or BiLSTM) to observe the *local change in cadence relative to preceding seconds of the same video*, a static per-frame Random Forest cannot distinguish frame drop from natural clip velocity.

### Minor / Excluded Factors:
- **(b) Features Not Reacting:** **Refuted.** Features react emphatically to true cuts ($\text{AUC} > 0.99$ for context MAD and pHash distance).
- **(c) Tiny Test Split:** While 28 deletion rows is small, the low recall ($0.071$) is fundamentally due to the 7-frame window penalty around a 1-frame cut, not sample variance alone.

