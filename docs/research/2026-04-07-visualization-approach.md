# Vamp Interface: Visualization Approach

**Date:** 2026-04-07  
**Status:** Complete — all versions generated and measured. See [2026-04-07-final-findings.md](2026-04-07-final-findings.md) for full metric results.

---

## What We Are Visualizing

Each job posting in the telejobs corpus is rendered as a **photorealistic portrait**. The face is not decorative — it is a direct encoding of two independent signals:

1. **Identity** (who the poster is): derived from the job text's embedding geometry
2. **Affect** (how suspicious the posting is): derived from `sus_level`, a 0–100 fraud score

The hypothesis is that the uncanny valley handles the second channel for free: viewers will feel that a high-fraud face is *wrong* without being able to articulate why. The first channel gives each face a stable identity so that jobs from similar semantic clusters look like similar people.

---

## Data Source

**Database:** telejobs PostgreSQL, `jobs` table  
**Corpus size:** 27,212 job postings, all with embeddings (as of 2026-04-07)  
**Test dataset:** 543 jobs, post-2026-02-20, 12 balanced cohorts  
**Embedding model:** qwen3-embedding:0.6b via Ollama (1024-d vectors), stored in pgvector

### Cohorts in test dataset

| Cohort | Type | N | Sus range |
|--------|------|---|-----------|
| warehouse_legit | Legitimate | 50 | 0–30 |
| construction_legit | Legitimate | 50 | 0–30 |
| cleaning_legit | Legitimate | 50 | 0–30 |
| office_legit | Legitimate | 50 | 0–30 |
| courier_scam | Fraud | 50 | 70–100 |
| remote_scam | Fraud | 50 | 60–100 |
| office_scam | Fraud | 40 | 60–100 |
| warehouse_scam | Fraud | 40 | 60–100 |
| easy_money_scam | Fraud | 45 | 80–100 |
| pay_mismatch_scam | Fraud | 41 | 80–100 |
| medium_sus | Ambiguous | 50 | 35–65 |
| high_medium_sus | Ambiguous | 27 | 65–89 |

---

## Two-Channel Encoding

### Channel 1: Identity (embedding → face demographics)

The 1024-d qwen3 embedding is reduced to PCA components (whitened, StandardScaler normalized). Each PC maps to a face attribute:

| PC | Semantic axis | Face attribute |
|----|--------------|----------------|
| PC1 | Physical/knowledge work polarity | Age + build (muscular 45-55y → slender 22-32y) |
| PC2 | Formal/informal register | Gender presentation (feminine → masculine/worn) |
| PC3 | Style/vocabulary sophistication | Hair style (styled → buzzcut) |
| PC4 (v2 only) | Complexion axis | Skin tone (fair → dark/tanned) |
| PC5 (v2 only) | Texture/experience axis | Skin texture (smooth → weathered) |

**v1 mapping** (divisor=2.0): clips PC values at ±1 after dividing by 2. Since whitened PCA scores are N(0,1), most values fall in the central two bands — low visual spread.

**v2 mapping** (divisor=1.0): full N(0,1) spread. ±1σ now spans all 5 bands, pushing jobs toward the extremes of the descriptor vocabulary.

The descriptor is a natural-language string fed as the positive prompt to the diffusion model:

```
"muscular build, 45-55 years old, weathered skin, calloused hands, 
masculine features, clean-shaven, practical clothing, medium-length hair, unstyled,
natural relaxed expression, open gaze, genuine affect,
soft studio lighting, plain background, sharp focus, 4k"
```

### Channel 2: Affect (sus_level → denoising strength)

`sus_level` 0–100 maps linearly to img2img denoising strength:

| sus_level | denoise | Affect prompt |
|-----------|---------|---------------|
| 0 | 0.35 | Natural relaxed expression, open gaze, genuine affect |
| ~30 | 0.44 | Slightly guarded expression, neutral-to-tense affect |
| ~55 | 0.52 | Uncomfortable expression, avoidant gaze, slight asymmetry |
| ≥80 | 0.59+ | Performative smile, slightly too-wide eyes, uncanny valley effect, wrong facial proportions, forced cheerfulness, hollow affect |
| 100 | 0.65 | (same as ≥80, maximum denoising) |

The anchor face (fixed seed=42, neutral portrait generated once) is used as the img2img base for every job. Same anchor → every face is a perturbation of the same starting point, making differences meaningful and comparable.

**Seed stability:** Each job gets a deterministic seed from MD5 of its first 12 embedding dimensions. Same job always → same face.

---

## Generation Pipeline

### Infrastructure

- **ComfyUI** at `localhost:8188` (RTX 5090)
- **img2img workflow**: anchor face → KSampler with denoising strength as fraud dial
- **Output:** `output/dataset_faces/<cohort>/<job_id>.png` (256×256, SDXL resolution 832×1216 downscaled)

### Checkpoints tested

| Version | Checkpoint | Workflow |
|---------|-----------|---------|
| v1 | SDXL Lightning / JuggernautXL | CheckpointLoaderSimple + KSampler |
| v2 | SDXL Lightning / JuggernautXL | Same, 5 PCA axes |
| Flux | flux1-krea-dev fp8 | UNETLoader + DualCLIPLoader (clip_l + t5xxl_fp8) + VAELoader (ae.safetensors) |

Note: Flux fp8 checkpoints do not bundle VAE or text encoders. Separate loaders required.

### Generation times (543 jobs, RTX 5090)

| Version | Time |
|---------|------|
| v1 (SDXL) | ~90 min |
| v2 (SDXL, 5-axis) | 22 min |
| Flux | ~5 min (after workflow fix) |

---

## Visual Distinctness Metric

**Method:** CLIP (openai/clip-vit-base-patch32) embeds each PNG into 512-d space (L2-normalized). Separation score measures how well cluster boundaries are preserved visually.

```
separation = (avg_inter_cluster_distance - avg_intra_cluster_distance) / avg_inter_cluster_distance
```

- 0.0 = clusters visually indistinguishable  
- 1.0 = perfect visual separation (intra_dist → 0)  
- Negative = clusters look more similar than random pairs (encoding failure)

**Implementation note:** `CLIPModel.get_image_features()` in this transformers version returns `BaseModelOutputWithPooling`, not a Tensor. Use `.pooler_output` directly (already 512-d projected).

---

## Results

### CLIP Separation Scores

| Version | Separation | Avg intra sim | Avg inter sim | Notes |
|---------|-----------|--------------|--------------|-------|
| v1 (SDXL, 3-axis) | **0.515** | 0.920 | 0.835 | Narrow band spread |
| v2 (SDXL, 5-axis) | **0.516** | 0.896 | 0.786 | Wider spread, more diverse faces |
| v3 (SDXL, work_type) | **0.516** | — | — | Explicit archetypes |
| Flux fp8 | **0.449** | 0.949 | 0.906 | CLIP-biased; ArcFace r=+0.848 |

### Key observations from v1/v2

**courier_scam is the most visually isolated cluster** across both versions. Inter-sim with legitimate cohorts: 0.66–0.72 (v2), down from 0.70–0.72 (v1). The high denoising + uncanny affect prompt is the primary driver.

**cleaning_legit ↔ construction_legit similarity = 0.953 (v2)** — still the weakest pair. Both map to the same PCA region: physical work, masculine, ~45yo. The identity channel produces near-identical faces because the embedding clusters genuinely overlap.

**Cross-work-type scams cluster together** (easy_money / pay_mismatch / remote_scam inter-sim 0.88–0.90 in v2). These share embedding geometry — the model encodes them similarly regardless of stated work type.

**v2 intra-cluster diversity is meaningfully higher.** warehouse_scam std went from 0.134 (v1) to 0.153 (v2). cleaning_legit std went from 0.012 to 0.062. The wider PCA spread is working.

**Separation score is nearly identical (0.515 vs 0.516)** despite v2's lower absolute similarities. Both inter and intra distances increased proportionally — the ratio held. This is expected: widening the descriptor vocabulary increases both within-cluster and between-cluster variance.

### The cohort collapse problem

Legitimate physical-work cohorts (warehouse, construction, cleaning) have very high intra-cluster similarity. The cause: these jobs occupy a narrow, overlapping region of embedding space. PC1 is their primary axis (physical work) but they share similar PC2/PC3 values. Better differentiation would require sub-cluster identity (e.g., nationality/age distribution within the cohort) or additional PCA dimensions that separate warehouse from construction semantically.

---

## What the Viewer Sees

A grid of portraits. Jobs from the same work type look like the same kind of person. Fraud jobs in that same category look like that same person but something is wrong with their face — too-wide eyes, hollow affect, forced smile, slight asymmetry. The viewer doesn't need to read a sus score; the uncanny valley delivers the verdict.

The embedding geometry means scam postings that mimic legitimate work (courier_scam is the clearest case in this corpus) are visually distinct: they produce a young professional with business attire and a performative smile — visually distinct from the weathered warehouse worker, but distinctly off.

---

## Files

| File | Purpose |
|------|---------|
| `src/generate_dataset.py` | Full generation pipeline (`--face-version 1/2`, `--flux`) |
| `src/face_distinctness.py` | CLIP separation metric |
| `src/build_test_dataset.py` | Builds the 543-job test dataset from DB |
| `data/test_dataset.json` | 543 jobs with embeddings inline |
| `output/dataset_faces/` | v1 SDXL faces (543 PNGs) |
| `output/dataset_faces_v2/` | v2 SDXL faces, 5-axis mapping (543 PNGs) |
| `output/dataset_faces_flux/` | Flux faces (543 PNGs, pending) |
