---
status: live
topic: arkit-bridge
---

# Anchor Test Plan — PersonaLive Bridge Robustness

**Date:** 2026-05-06
**Context:** GPU offline; calibration v3 deferred to a separate thread. This doc plans the **anchor pool** we'll use once GPU returns to stress-test PersonaLive's portrait coverage with our ARKit-driven bridge.

## Goal

Move from "the bridge works on `asian_m__06_neutral`" to "the bridge works (or gracefully degrades) across a categorized portrait pool." Each category answers a different failure-mode question:

| Category        | What it stresses                                                   |
|-----------------|--------------------------------------------------------------------|
| Race            | Skin-tone / facial-structure prior of motion encoder               |
| Gender          | Eyebrow / jawline / hair-occlusion robustness                      |
| Age             | Wrinkle texture, eye-bag detail, upper-eyelid hooding              |
| Anime / 2D      | Out-of-distribution stylistic prior (PersonaLive trained on photo) |
| Old photo / B&W | Tonal range, film grain, period clothing, monochrome               |
| Painting        | Hand-rendered stylization (oil / pencil / illustration)            |
| Glasses         | Anchor-side occluder — separate thread, known degenerate           |

Per-anchor scorecard (computed on a fixed synthetic ARKit drive sequence so anchors are directly comparable):

- Mediapipe yaw/pitch/roll Pearson + RMSE vs input
- Identity drift: ArcFace cosine(anchor_frame, render_frame_mean)
- Visual sanity: 3-still strip (yaw extreme, pitch extreme, smile extreme)

## Anchor pool inventory

### Photoreal pools — two parallel 42-cell tiers

We now have **two parallel photoreal pools** at the same factorial taxonomy (7 races × 3 ages × 2 genders = 42 cells). Each tier stresses different priors:

| Pool | Source | Style | Path |
|---|---|---|---|
| Flux grid | Solver-A squint grid renders | Photoreal Flux output, neutral by construction (scale=0) | `data/anchors/photoreal_grid/` |
| FFHQ real | Real-world photographs (FFHQ) | Real photos, FairFace-classified, MediaPipe-blendshape-filtered for neutrality | `data/anchors/photoreal_ffhq/` |

Per-cell PNG + manifest in each. Manifests use `format_version: 1` and the same `(race, age, gender)` join keys.

#### `data/anchors/photoreal_grid/` (Flux-rendered, 13 MB)

Lifted directly from the Solver-A squint grid corpus (`output/demographic_pc/solver_a_squint_grid/` on the Seagate archive drive) via `scripts/select_photoreal_anchors.py`. Full factorial: **7 races × 3 ages × 2 genders = 42 cells**, one PNG per cell, picked by neutrality score:

| Axis    | Levels                                                                        |
|---------|-------------------------------------------------------------------------------|
| race    | black, east_asian, latino, middle_eastern, south_asian, southeast_asian, white |
| age     | young, adult, elderly                                                         |
| gender  | m, f                                                                          |

Selection rule (`scripts/select_photoreal_anchors.py`):

1. Require `anchor_face_detected = True`.
2. Drop seeds where any of `{glasses, eyes_closed, smiling, open_mouth, surprised, puckered_lips, angry}` SigLIP probe ≥ 0.05. **Note:** `wrinkled` deliberately excluded — it correlates with the elderly demographic by construction; filtering on it would bias elderly cells toward atypically smooth faces.
3. Among survivors, minimise `anchor_squint + anchor_smile_bs + |anchor_brow|`.
4. Tie-break on lowest seed (deterministic).
5. Manifest at `data/anchors/photoreal_grid/manifest.parquet` (`format_version: 1`); `fallback_used` column flags any cell where the strict filter wiped all 16 seeds and we relaxed to face-detected-only. **Current run: 0/42 fallback** — every cell has at least one clean seed.

This **subsumes** the earlier hand-curated `data/llf-phase2/*` demographics (asian_m, black_f, european_m) and `phase3_full_replay/{young_european_f, elderly_latin_m}` — the grid covers all those cells and 35 more, with consistent prompt template + neutral expression by construction.

#### `data/anchors/photoreal_ffhq/` (real photos, 55 MB)

`scripts/select_ffhq_anchors.py` lifts one best-neutral real photo per cell from the 70k-row FFHQ corpus. Two-stage:

1. **`output/reverse_index/ffhq_sha_lookup.parquet` (cached, 70k rows).** Walks `/media/newub/Seagate Hub/arc_distill/metrics/train-*-of-*.pt` and emits `(image_sha256, shard_path, row_idx)`. Per-shard metrics .pt position k corresponds to row k of the matching parquet shard — verified at runtime by re-hashing row 0 of shard 0 and comparing to the cached sha (`verify_lookup_alignment` gate, per project standing rule on training-data verification).

2. **Selection.** From the unified reverse_index, filter `source='ffhq'`, face_detected, classifier-detected. Map FairFace → squint-grid taxonomy:

   | FairFace | Squint-grid |
   |---|---|
   | White, Black, East Asian, Southeast Asian, Indian, Middle Eastern, Latino_Hispanic | white, black, east_asian, southeast_asian, **south_asian**, middle_eastern, latino |
   | age_bin 0-29 / 30-49 / 50+ | young / adult / elderly |
   | M / F | m / f |

   Then drop occluder probes (same set as the grid selector minus `wrinkled`), minimise `max(squint_L,R) + max(smile_L,R) + |brow|`, tie-break on classifier confidence then sha. PNG bytes extracted from the FFHQ shard parquet via the cached lookup; saved as `<race>__<age>__<gender>__<sha8>.png`.

3. **Coverage achieved (2026-05-06):** 42/42 cells, 0 fallback, 69811 → 69039 rows after occluder filter. All 42 winners have full classifier metadata + neutrality components in the manifest.

### Earlier (legacy) per-cell baselines — kept for backward compatibility

`data/llf-phase2/*__06_neutral.midframe.png` is still the default anchor in `calibrate_euler_signs_v3.py:--anchor` (path: `data/llf-phase2/asian_m__06_neutral.midframe.png`). Don't break that pinning until calibration is finished. Once anchor-pool sweep is the active workload, switch to the grid manifest.

### Stylistic anchors (need to source)

#### Anime / 2D (target: 1 — best style we like)

PersonaLive is photoreal-trained; anime is the most aggressive OOD stress. Single representative pick.

**Selected:** `data/anchors/anime/anime__01_oksmith.png` — 1570×2400, CC0 (Public Domain Dedication), oksmith via OpenClipart → Wikimedia Commons. Modern celshade frontal portrait of an anime girl, transparent background, single character. License-clean for any use.

Source page: [commons.wikimedia.org/wiki/File:Anime_girl_publicdomainq.png](https://commons.wikimedia.org/wiki/File:Anime_girl_publicdomainq.png)

If we want to upgrade later: [Frontviewofananimeface.png](https://commons.wikimedia.org/wiki/File:Frontviewofananimeface.png) is higher-res (3248×4000) but the uploader requests permission before use; would need to email.

#### Old photo — Вячеслав Тихонов (target: 1, B&W older)

**Selected:** `data/anchors/oldphoto_tikhonov/tikhonov__1948.jpg` — 600×451 JPEG, B&W, frontal formal portrait, 1948, public domain via Wikimedia Commons. Earliest portrait in the WMC Tikhonov category — the youngest, most formally-posed, lowest-noise image, which gives mediapipe the cleanest landmark surface. Higher recognizability targets (Stierlitz frame stills from 1973) are not freely licensed; defer those to a v2 slate if internal-only use clears.

Source: [commons.wikimedia.org/wiki/File:Vyacheslav_Tikhonov_1948.JPG](https://commons.wikimedia.org/wiki/File:Vyacheslav_Tikhonov_1948.JPG)

Caveat: 600×451 is small. PersonaLive's anchor pre-process will likely resize to its target (typically 512×512 with face crop), so resolution is not the bottleneck — but if the face crop ends up under ~256², we'll need a larger source. Verify after first render.

#### Painting — Александр Пушкин (target: 1, best quality)

**Selected:** `data/anchors/painting_pushkin/pushkin__01_kiprensky.jpg` — Kiprensky 1827 oil portrait via Google Art Project, **3455×4000** (10.4 MB), public domain. Highest-resolution scan of the canonical museum-grade reproduction (Tretyakov Gallery). ¾ frontal view, soft Romantic-era oil shading.

Source: [commons.wikimedia.org/wiki/Category:Portrait_of_Alexander_Pushkin_(Orest_Kiprensky,_1827)](https://commons.wikimedia.org/wiki/Category:Portrait_of_Alexander_Pushkin_(Orest_Kiprensky,_1827))

Why this matters as a stress case: Pushkin exists *only* as paintings/drawings — there is no photographic ground truth that could ever have existed. Strongest possible stylization-OOD test on a globally-recognizable identity (familiar by design — see open question 4).

## Test protocol (when GPU returns)

For every anchor in the pool:

1. **Single fixed driver clip** — `data/llf-clips-auto/20260505_MySlate_5_yaw/MySlate_5_iPhone.csv` (the canonical 600-frame yaw stress, post-calibration).
2. Render via `scripts/apply_bridge_to_personalive.py --mode bridge --anchor <png>`.
3. Run `scripts/build_render_metrics_parquet.py` (mediapipe extraction + parquet append). `clip_id` column lets us aggregate per anchor.
4. Compute scorecard:
   - `pearson(yaw_in, yaw_out)`, RMSE; same for pitch/roll
   - `arcface_cos(anchor_png, mean_render_frame)` and per-frame min
   - Save 3-still strip to `exp_output/arkit_bridge/anchor_pool/<anchor>/strip.png`
5. Aggregate to one CSV: `exp_output/arkit_bridge/anchor_pool/scorecard.csv`, columns `[anchor, category, yaw_r, pitch_r, roll_r, arcface_mean, arcface_min, notes]`.

**Pass / fail thresholds (provisional):**

| Tier               | yaw r   | arcface_min | Notes                                           |
|--------------------|---------|-------------|-------------------------------------------------|
| Photoreal in-dist  | ≥ 0.95  | ≥ 0.55      | Hard requirement                                |
| Photoreal OOD age  | ≥ 0.90  | ≥ 0.45      | Some identity drift on elderly is acceptable    |
| Anime              | ≥ 0.80  | n/a         | ArcFace meaningless on 2D; visual-only          |
| Old photo / B&W    | ≥ 0.90  | ≥ 0.40      | ArcFace less reliable on grayscale              |
| Painting           | ≥ 0.75  | n/a         | Visual-only                                     |

## Pre-flight gating

Before running the full pool we need:
1. ✅ Preprocess take pipeline → done
2. ⏳ Calibration v3 result merged into `closed_form_pose.py:EULER_SIGNS` (deferred to other-thread)
3. ⏳ `clip_id` parquet schema in production → done in builder, not yet exercised
4. ⏳ ArcFace evaluator helper script (separate small task; we have insightface buffalo_l installed already per the `project_vamp_measured_baseline` memory)

## Slate (resolved 2026-05-06)

Final pool: **84 photoreal anchors** (42 Flux-grid + 42 FFHQ-real, parallel demographic taxonomies) + **3 stylistic anchors**.

| Category | Count | File(s) | License |
|---|---|---|---|
| Photoreal Flux | 42 | `data/anchors/photoreal_grid/<race>__<age>__<gender>__seed*.png` | Internal |
| Photoreal real (FFHQ) | 42 | `data/anchors/photoreal_ffhq/<race>__<age>__<gender>__<sha8>.png` | FFHQ research license |
| Anime | 1 | `data/anchors/anime/anime__01_oksmith.png` (1570×2400) | CC0 |
| Old photo / B&W | 1 | `data/anchors/oldphoto_tikhonov/tikhonov__1948.jpg` (600×451) | Public domain |
| Painting | 1 | `data/anchors/painting_pushkin/pushkin__01_kiprensky.jpg` (3455×4000) | Public domain |

**Total: 87 anchors.** Internal-use only confirmed. No public release of derived renders without a separate review.
