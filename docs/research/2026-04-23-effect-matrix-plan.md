---
status: live
topic: metrics-and-direction-quality
---

# Effect-matrix pipeline — plan (2026-04-23)

## Why

The controllable-edit program needs a procedure, not a set of anecdotes. Given an edit axis and a scale, we want to know everywhere it moves us across a measurement vocabulary: atoms, age, gender, race, identity, total drift. That table — the **effect matrix** — is what turns "add smile makes younger" into `δ_smile − k·δ_age`, tuned by a solver.

This doc is the plan for finishing the scaffolding (measurements in the index) and then building the effect-matrix report.

## Stack, status before today

- Backbone: FluxSpace pair-averaged attention edits (live, used for all edits we have).
- Step control: `scale` at fixed `mix_b=0.5`; `max_env` + `T_ratio≈1.275` as collapse predictor (glasses + smile validated).
- Vocabulary: 11 sparse-NMF atoms over 52 MediaPipe blendshape channels; ridge atom→δ directions CV R² 0.82–0.97.
- Edits available as δ: smile, beard (add/remove), glasses, anger, surprise, pucker.
- **Not** yet as δ: age, gender, race — only available as measurement axes today.

## What landed today

| Step | Status |
|---|---|
| Overnight renders (smile rungs + beard add/remove + rebalance reseeds, 1,440 imgs) | done pre-today |
| Beard-rebalance renders (3 bearded-base male cells × 4 scales × 8 seeds = 96) | done pre-today |
| MediaPipe blendshape + NMF atom projection for all 5,038 rows | done |
| SigLIP-2 So400m/16 probe margins (12 probes) over 1,536 overnight rows | **done** |
| SigLIP parquet merged into `sample_index.parquet` (30.5% coverage, NaN on older rows) | **done** |
| MiVOLO + FairFace + InsightFace(+ArcFace r50) over all 5,038 rows | **done** |
| Classifier sidecar merged into `sample_index.parquet` | **done** |

Detection coverage post-merge: MiVOLO 100%, FairFace 100%, InsightFace 96.2% (193 collapsed-scale rows where SCRFD couldn't find a face — those will be NaN on ArcFace embedding and downstream identity cosines).

## What's left before the effect matrix

### Task #3 — derive-drift cosines *(next)*

For each row, compute:

- `identity_cos_to_base`: cosine(this render's ArcFace 512-d embedding, the seed-matched scale=0 render of the same base's embedding). Hard identity constraint for the solver.
- `siglip_img_cos_to_base`: analogous but with SigLIP-2 image features. Catch-all for drift outside the blendshape+probe vocabulary.

Prerequisites: SigLIP image features cached per row (not just text-pair margins — we need the image-side 1152-d feature). Re-use the SigLIP-2 backend we already loaded.

Storage: two scalar columns. Column width cost ~80 KB total.

### Task #4 — max_env stamp

For each row with `has_attn=True`, open the corresponding attention pkl, compute `max|attn_base + s·δ_mix|`, cache as `max_env`. Enables per-row collapse flagging without re-opening pkls downstream.

Cost: ~2,300 pkls × ~50 MB avg = 110 GB reads but sequential; a few minutes end to end.

### Task #5 — prompt provenance columns

Add seven columns:
- `prompt_base` — full base prompt string
- `prompt_edit_pos` — positive edit prompt (NaN on base-only rows)
- `prompt_edit_neg` — negative edit prompt (NaN on base-only rows)
- `prompt_intent_age`, `prompt_intent_ethnicity`, `prompt_intent_gender` — raw words extracted from the prompt, preserve nuance the coded bins collapse (e.g., "elderly" vs "senior" vs "old")
- `prompt_extras` — non-demographic modifiers on the base (e.g., "_bearded" variants)

Source: reconstruct by keying `(source, subtag, base)` against the render-script constants. Write a single `prompt_provenance.parquet` sidecar, merge.

Once this lands: any row is a complete experiment record — reproducible from the parquet without touching code.

### Task #6 — effect-matrix v0 report

Output: `docs/research/2026-04-23-effect-matrix-v0.md` + `output/demographic_pc/effect_matrix_v0.parquet`.

Per (axis, base) cell, fit slopes of each measurement vs scale:

- Atom slopes (21 atoms × 6 axes × N bases)
- SigLIP probe slopes (12 probes × same)
- Age drift: `mv_age`, `ff_age_bin` midpoint, `ins_age` — three slopes, divergence is itself a signal
- Gender drift: majority-flip rate, MiVOLO confidence drop
- Race drift: FairFace 7-way distribution shift (KL vs base)
- Identity drift: `identity_cos_to_base` slope
- Total drift: `siglip_img_cos_to_base` slope
- Safe-window edge: `max_env` at which T_ratio threshold is crossed per base

The report surfaces: which edits have clean primary-axis signal, which carry big confounds, where the atom decomposition agrees with the SigLIP probe signal vs where they diverge, and which collapse edges generalize across bases.

## After the matrix

Two branches, independent:

1. **Composition solver (math).** Given the effect matrix, linear solve for counter-edits: "add smile without aging" → find the atom/axis mixture that maximizes target-axis slope while minimizing age/identity drift. First iteration is closed-form linear least squares on the matrix; later iterations are local / scale-dependent.

2. **Demographic-edit δs (render batch).** To edit age, gender, race we need FluxSpace attention pairs on prompt pairs like "young woman" / "elderly woman", "man" / "woman", cross-race pairs. ~400–800 new renders, half a day. This moves age/gender/race from *measurement-only* to *editable* members of the dictionary.

The solver can start on the six edits we already have; adding demographic δs is pure expansion, not a blocker.

## Open questions

- Does `max_env` / `T_ratio=1.275` hold for the new beard-rebalance bases and the rebalance_reseed axes?
- Do ridge atom→δ directions (trained pre-overnight) visually reproduce the atom slopes we'll measure post-overnight — or did we overfit to the pre-overnight corpus skew?
- Is 96.2% InsightFace detection acceptable, or do we need to add a second detector (dlib, MTCNN) as fallback for the collapsed-scale frames?
- For solver linearity: do pairs of edits compose near-linearly at `scale ≤ 0.5`, or does saturation kick in earlier than we think?
