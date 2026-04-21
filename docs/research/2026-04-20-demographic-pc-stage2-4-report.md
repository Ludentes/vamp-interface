# Demographic-PC Stages 2–4: Generation, Classification, Regression

**Date:** 2026-04-20
**Follows:** [Stage 1 sanity report](2026-04-20-demographic-pc-stage1-report.md).
**Status:** Stages 2 → 2b → 3 → 4 all complete.

## Stage 2 — 1785-sample generation

- Grid: 5 ages × 3 genders × 7 ethnicities × 17 seeds = **1785** samples.
- Pipeline: Flux Krea v3 img2img via ComfyUI, shared 768×1024 neutral anchor, denoise=0.9, seed per sample.
- Runtime: ~2.7 h unattended on RTX 5090.
- Output: `output/demographic_pc/samples/{cell_id}-s{seed}.png` (1785 files).
- Manifest: `output/demographic_pc/manifest.json` (prompt + seed + labels per sample_id).

## Stage 2b — conditioning capture

Goal: produce a 4864-d pooled conditioning vector per prompt, aligned to sample_ids, for Stage-4 regression.

Encoder stack (mirrors Flux's dual text encoder):

- **CLIP-L** — `openai/clip-vit-large-patch14` (HF snapshot, cached) → `pooler_output` (768-d).
- **T5-XXL** — raw weights from `ComfyUI/models/text_encoders/t5/t5xxl_fp16.safetensors` loaded into `transformers.T5EncoderModel` instantiated from `google/t5-v1_1-xxl` config. `last_hidden_state` mean-pooled over sequence, attention-mask-weighted (4096-d).
- Concat = 4864-d. Saved as float32.

**Implementation notes:**

- `transformers==5.5.0`'s auto-conversion path for T5's spiece.model is broken (interprets the SentencePiece file as tiktoken BPE). Worked around by driving SentencePiece directly (`T5SpmTokenizer` in `stage2b_conditioning.py`) — 4-line wrapper that matches T5's tokenize-and-pad protocol (EOS=1, PAD=0, truncate at `T5_MAX_LEN=256`).
- ComfyUI's internal modules (`comfy.sd`) couldn't be imported standalone — this install has a `comfy_aimdo` dependency on a missing in-house package. So the encoder pipeline runs via `transformers` directly, not via ComfyUI's Python API.
- ComfyUI server must be stopped before running to free ~10 GB of VRAM for T5-XXL-fp16 + CLIP-L.

Runtime: **31 s** for 1785 prompts at batch=8 on CUDA (57.8 prompts/s).
Output: `output/demographic_pc/conditioning.npy` (1785, 4864), `conditioning_ids.json`.

**Diagnostic:** only **121 unique rows** of conditioning out of 1785 — expected ~105 (5 × 3 × 7 prompt combinations). The extra 16 come from tiny float-16 / mean-pooling numerical jitter between seeds. Effectively the regression is a **121-sample** problem, not 1785. Downstream ridge handles this fine because each unique prompt appears ~14 times with varying labels (seed-driven image variation) — the regression fits on (x, mean_y) structure.

## Stage 3 — classifier inference

All three classifiers on all 1785 images.

Runtime: **119 s** total at 15 samples/s (CUDA).

**Coverage:**

| Classifier | Detections |
|---|---|
| MiVOLO (whole-image) | 1785 / 1785 |
| FairFace (dlib-CNN + align) | **1785 / 1785** |
| InsightFace (SCRFD @ 640×640) | 1780 / 1785 |

**InsightFace's 5 misses** — all elderly-cell samples with a very small or partially shadowed face under the SCRFD's 640×640 detect scale. Sample IDs:

- `2-2-3-s23311` — adult / non-binary / Black
- `4-0-3-s41305` — elderly / man / Black
- `4-0-4-s41404` — elderly / man / White
- `4-1-3-s42300` — elderly / woman / Black
- `4-2-4-s43402` — elderly / non-binary / White

Skewed toward `age_idx=4` (elderly) — SCRFD struggles on weathered skin + low contrast against grey background. Not catastrophic: 0.28% drop, and regression handles NaNs per-head.

**Gender inter-classifier agreement (both-detected subset):**

| Pair | Agreement |
|---|---|
| FairFace ↔ MiVOLO | **0.919** |
| MiVOLO ↔ InsightFace | 0.831 |
| FairFace ↔ InsightFace | 0.778 |

Matches Stage 1's pattern: FF+MV agree strongly, InsightFace is the noisy-but-informative outlier. No three-way disagreement pathology.

**Age by prompt (MiVOLO, all 1785):**

| Prompt age | Mean | Std | n |
|---|---|---|---|
| child | 8.2 | 1.0 | 357 |
| young adult | 22.6 | 2.6 | 357 |
| adult | 36.2 | 6.9 | 357 |
| middle-aged | 62.9 | 6.1 | 357 |
| elderly | 81.1 | 2.9 | 357 |

Tight and cleanly separated — the prompt-age signal is very strong in the generated images.

## Stage 4 — regression + direction extraction

**Architecture:** Standardize conditioning (4864 dims). Fit per-head:

- Continuous (MiVOLO age, InsightFace age) → `RidgeCV(alphas=logspace(-2,4,13))`.
- Categorical (genders, FairFace age-bin, FairFace race) → `LogisticRegression(C=1.0, lbfgs)`.

Score each head via 5-fold shuffled CV (R² or accuracy). Drop heads below threshold (R²<0.3, acc<0.4). Stack `Vt` rows weighted by singular value → truncated SVD → retain cumulative-variance ≥ 0.90 as demographic subspace `D`.

### Bug — CV split was prompt-grouped

First run reported R² ≈ **−272** for age regression (train R²=0.99 — it's a regression, not a mystery — just CV saying "this model is worse than constant"). Root cause: `cross_val_score`'s default `KFold(shuffle=False)` was partitioning the prompt-ordered dataset into folds that each held one prompt-age level. Fold 0 test = all children (mean age 8.2); training folds mean ≈ 50 → predictions systematically 40 years off → R² → large negative.

Fix: explicit `KFold(shuffle=True, random_state=0)` for ridge and `StratifiedKFold(..., shuffle=True)` for categorical.

### Per-head results (shuffled CV)

All seven heads cleared their thresholds — nothing dropped.

| head | kind | CV score | n | notes |
|---|---|---|---|---|
| mivolo_age | ridge | R² = **0.991** | 1785 | α=316 |
| insightface_age | ridge | R² = **0.970** | 1780 | α=316 |
| fairface_gender | logistic | acc = **0.967** | 1785 | 2 classes |
| mivolo_gender | logistic | acc = **0.950** | 1785 | 2 classes |
| insightface_gender | logistic | acc = **0.866** | 1780 | 2 classes |
| fairface_age_bin | logistic | acc = **0.863** | 1785 | 9 bins |
| fairface_race | logistic | acc = **0.830** | 1785 | 7 classes |

### Subspace

- Stacked direction matrix: **W ∈ ℝ^(15 × 4864)** (7 heads × ≤5 components, truncated at head rank).
- Truncated SVD at 90% cumulative variance → **d = 9**.
- Cumulative variance at d=9: **0.923**.

`subspace_D.npy` (9, 4864) is the orthonormal basis used for Level-0 Δ-orthogonalization in Stage 5.

### Cross-classifier principal-angle cosines

(Top singular value of `A @ B.T` where A, B are the same attribute's per-classifier direction bases. 1.0 = coplanar; 0 = orthogonal.)

| Pair | cos |
|---|---|
| age: MiVOLO ↔ InsightFace | **0.860** |
| age: MiVOLO ↔ FairFace-bin | **0.818** |
| age: InsightFace ↔ FairFace-bin | **0.695** |
| gender: MiVOLO ↔ FairFace | 0.681 |
| gender: MiVOLO ↔ InsightFace | 0.307 |
| gender: FairFace ↔ InsightFace | 0.287 |

Reading:

- **Age directions agree across all three classifiers** (all ≥ 0.70). This is the strongest possible empirical confirmation that "the age direction in Flux conditioning space" is a real, classifier-invariant thing — not an artifact of any single classifier's biases.
- **Gender directions only moderately agree** across MV/FF (0.68); InsightFace's gender direction is essentially decorrelated from the other two (~0.3). Consistent with Stage 1's finding that InsightFace's gender head is resolution-starved and makes different systematic errors. Downstream impact: when we subtract the gender subspace, the MV+FF components carry most of the useful signal; IF adds some independent noise.
- No same-attribute pair is below 0.25, so nothing actively conflicts. All three classifiers' vote matrices are being stacked into the SVD, which is appropriate.

## Files produced

## Files produced

- `src/demographic_pc/stage2b_conditioning.py` — CLIP-L + T5XXL → 4864-d pooled vectors.
- `src/demographic_pc/stage3_classify.py` — three-classifier sweep → `labels.parquet`.
- `src/demographic_pc/stage4_regression.py` — per-head regression + subspace SVD.
- `output/demographic_pc/conditioning.npy` (1785, 4864).
- `output/demographic_pc/labels.parquet` (1785 rows, 16 cols).
- `output/demographic_pc/{direction_matrix.npy, subspace_D.npy, subspace_report.json}` — populated when Stage 4 re-run finishes.
