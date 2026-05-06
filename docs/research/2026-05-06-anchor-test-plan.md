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

### What we already have (local, photoreal Flux baselines)

`data/llf-phase2/*.midframe.png` — 3 demographics × 6 expressions; the `*__06_neutral.midframe.png` of each is the canonical anchor:

- `asian_m__06_neutral.midframe.png` — current default anchor in `calibrate_euler_signs_v3.py`
- `black_f__06_neutral.midframe.png`
- `european_m__06_neutral.midframe.png`

`output/demographic_pc/phase3_full_replay/` — additional Flux baselines at `s+0.00`:

- `young_european_f_s777_x+0.00.png` (young / female)
- `elderly_latin_m_s777_x+0.00.png` (elderly / latin / male)

**Coverage today:** races {asian, black, european, latin} × genders {m, f} × ages {young (implicit), elderly (1)} — 5 identities. Skewed: only one explicit elderly, no child, no south-asian/indian, no mixed.

### Gaps to fill from existing Flux infrastructure

If a quick render budget appears, generate (FluxSpace node, neutral prompt, no edit, seed=2026, scale=0):

- `indian_f__neutral` — fills south-asian female
- `latin_f__neutral` — fills latin female (companion to `elderly_latin_m`)
- `child__neutral` (race-balanced) — fills age-young extreme
- `elderly_european_f__neutral` — fills age × race × gender corner currently empty

Recipe is already in `output/demographic_pc/phase3_full_replay/` — same prompt template, swap the demographic slot. Defer until GPU returns.

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

Per user decision in this session, slate is **3 stylistic anchors + the existing photoreal Flux pool**. One image per stylistic category, picking the highest-quality / best-fit option:

| Category | File | Source | License |
|---|---|---|---|
| Anime | `data/anchors/anime/anime__01_oksmith.png` (1570×2400) | Wikimedia Commons (oksmith via OpenClipart) | CC0 |
| Old photo / B&W | `data/anchors/oldphoto_tikhonov/tikhonov__1948.jpg` (600×451) | Wikimedia Commons (Vyacheslav_Tikhonov_1948.JPG) | Public domain |
| Painting | `data/anchors/painting_pushkin/pushkin__01_kiprensky.jpg` (3455×4000) | Wikimedia Commons (Google Art Project) | Public domain |

Internal-use only confirmed. No public release of derived renders without a separate review.
