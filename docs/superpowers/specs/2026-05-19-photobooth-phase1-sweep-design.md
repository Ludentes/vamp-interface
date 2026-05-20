---
status: live
topic: arkit-controlnet
---

# Photobooth Phase 1 sweep — adaptive pipeline characterization

## Purpose

Characterize the parameter response surface for the matryoshka photobooth
pipeline (Z-Image Turbo + Z-Image-Turbo-Fun-CN-Union + HyperSwap-1c) across the
range of inputs a deployed booth will see. The output is three artifacts:

- **Robust default** — the config with the highest 5th-percentile id_cos at
  100% SCRFD detection across the source-photo stratum.
- **Adaptive lookup table** — best config per source-feature bucket
  (skin tone × gender × glasses × hair-density).
- **Failure-mode map** — which source-photo classes break which lever.

Phase 1 is a Latin-hypercube **pilot**: 4 stratified source photos × 40 configs
(160 cells). Its job is to *prune* axes for Phase 2 (20 photos × ~30 reduced
configs), not to ship the final defaults.

## Success criteria

Phase 1 is done when:

- All 160 cells either complete or are explicitly logged as `ok=False` with
  reason.
- `scores.parquet` contains the 7 metric columns for every successful cell.
- A short follow-up doc (`docs/research/2026-05-XX-photobooth-phase1-findings.md`)
  ranks the 8 axes by main-effect magnitude on id_cos and lists axes safe to
  pin in Phase 2.

Out of scope for Phase 1: shipping a production config, evaluating PuLID /
InfiniteYou / ACE++ injection paths, tuning the swap stage. Swap stage is
isolated and held constant.

## Constraints and isolation rule

- 8 h on the RTX 3090 (`shard`) — Phase 1 compute is ~30 min, leaves slack
  for Phase 2 in the same window if findings allow.
- **Swap stage = constant.** `scripts/swap_core.py` `swap_identity` with
  HyperSwap-1c via `DEFAULT_SWAPPER`, no parameters varied. CPU swap stays
  off-GPU and does not contend with ComfyUI.
- Calibration corpus = `data/importer/identities/` (20 photos with
  `manifest.csv` race / gender / age tags).

## Approach selection

Three designs considered:

**A. Full Cartesian on a reduced subset.** Pick top-3 axes, full grid.
Simple and balanced. Falsifies *nothing* about the other 5 axes — we'd be
committing in advance to which axes matter and silencing the rest.

**B. Latin hypercube over all 8 live axes.** ~40 mixed-type LHS samples,
evaluated per photo. Maximises joint-space coverage in fewest renders;
supports per-axis bin-mean fits + random-forest feature-importance on the
resulting (axes, metrics) parquet.

**C. Plackett-Burman screening, then focused grid.** Classical DOE.
Maximally efficient for linear main effects but our likely-non-monotone axes
(Canny thresholds, refine denoise) violate PB's linearity assumption; also
needs 2 orchestration phases for Phase 1 alone.

**Choice: B.** With 8 axes and exploratory intent (we don't yet know which
matter), LHS gives the broadest learning per render. The downstream Phase 2
is the "focused grid" that approach C would have started with — running it
*after* LHS pruning is strictly better.

## Live axes (8)

Picked after consolidating with the prior brainstorm; locked axes (already
swept or settled) are listed in the next section.

| # | Axis | Type | Levels | Notes |
|---|---|---|---|---|
| 1 | `color_match` | categorical (3) | off / Lab / Reinhard | Post-swap skin-tone transfer in mediapipe face mask |
| 2 | `face_pixel_budget` | categorical (3) | `tight_1024` / `med_1024` / `tight_768` | Composite of render-res × framing-prompt × source-crop tightness |
| 3 | `cn_condition` | categorical (3) | `canny` / `depth` / `canny+depth` | What `Z-Image-Turbo-Fun-CN-Union` is fed |
| 4 | `cn_strength` | continuous | [0.80, 1.00] | Tightened around the cn_grid winner (0.90) |
| 5 | `cn_start_percent` | categorical (3) | 0.00 / 0.10 / 0.20 | Late-start frees identity early in denoise |
| 6 | `canny_preset` | categorical (3) | `soft(50/150, blur=1.5)` / `default(100/200, blur=0)` / `aggressive(150/250, blur=0)` | Only meaningful when `cn_condition` includes `canny`; otherwise NA |
| 7 | `refine_denoise` | categorical (3) | off / 0.30 / 0.50 | Face-mask masked img2img on swap output |
| 8 | `demo_inject` | categorical (2) | off / on | When on, prepend `"a {age_bin}-year-old {race} {gender} face, "` to the existing CN-grid prompt; `age_bin`/`race`/`gender` come from `manifest.csv` |

## Locked / pinned axes

- **Sampler / scheduler** — `euler / simple` (bake-off verdict).
- **Steps** — 6 (cn_grid verdict; 8 within noise, costs 20% more).
- **CN strength upper bound** — 1.00 (anything higher introduces over-control
  artifacts per the cn_grid follow-up).
- **HyperSwap variant** — 1c (`DEFAULT_SWAPPER`).
- **Source upscale before swap** — 512 (existing `crop_and_upscale`).
- **GFPGAN / CodeFormer restore** — off (lowers id_cos per `swap_core` audit).
- **Seed** — per-identity fixed: `91_000_000 + id_idx` (matches cn_grid for
  cross-sweep comparison).

## Source-photo stratification

Hand-picked from `manifest.csv` to span the skin-tone × gender axes most
likely to break HyperSwap's tone-wash:

| photo_id | race | gender | age | Why |
|---|---|---|---|---|
| `id_00` | Black | F | 20-29 | Dark skin — HyperSwap's worst-case wash class |
| `id_11` | East Asian | F | 30-39 | Distinct facial morphology, light-medium skin |
| `id_16` | Middle Eastern | M | 40-49 | Beard + older skin (Canny clutter stressor) |
| `id_01` | White | F | 20-29 | Control / "default" identity |

Per-photo features computed once at start and stored in `sources/features.json`
(skin_tone Lab L* mean over mediapipe skin mask, hair_density proxy as
non-face skin pixels / face pixels, glasses heuristic via SCRFD landmark
geometry, plus copied race/gender/age from manifest.csv).

## Pipeline architecture

New code lives under `scripts/photobooth_sweep/` — one file per responsibility,
all stateless except the driver:

```
scripts/photobooth_sweep/
  __init__.py
  axes.py              # axis catalog + LHS sampling + manifest writer
  color_match.py       # Lab and Reinhard skin-region transfer
  refine.py            # build masked-img2img workflow for refine pass
  scorer.py            # id_cos, det_mode, lab_delta, clip_style, face_frac
  features.py          # per-source-photo feature extraction
  driver.py            # main loop, resumable, talks to remote ComfyUI
  README.md
```

Two ComfyUI workflows, both parametric (`$$`-substituted):

```
comfyui/workflows/
  photobooth_zimage_cn.api.json       # Z-Image Turbo + Fun-CN-Union (cond depends on cn_condition)
  photobooth_zimage_refine.api.json   # masked img2img refine (one face-region pass)
```

The CN workflow accepts a single composite control input — `cn_condition`
selects Canny / Depth / blended, prepared on the Linux box and staged via the
ComfyUI `/upload/image` API (the cn_grid pattern). Depth uses
`comfyui_controlnet_aux`' DepthAnything-V2 preprocessor on the box.

## Data flow

For each (photo_id, cfg_id) cell:

1. Look up cell in `scores.parquet` — skip if `ok=True`.
2. Build the per-photo control image (Canny / Depth / blend, sized to
   `face_pixel_budget`'s render res, with source-crop tightness applied).
3. Optionally inject demographic tokens into the prompt template.
4. Submit Z-Image+Fun-CN workflow to remote ComfyUI; fetch the doll.
5. Run `swap_core.swap_identity` (CPU, HyperSwap-1c).
6. Optionally apply `color_match` post-swap.
7. Optionally run the refine workflow if `refine_denoise != off`.
8. Score: id_cos, det_mode, lab_delta, clip_style, face_frac.
9. Append a row to `scores.parquet` (parquet append via duckdb / pyarrow);
   write artifacts atomically.

## Output schema

```
exp_output/photobooth_phase1/
  manifest.parquet        # 160 rows — one per cell, axis values
  scores.parquet          # 160 rows — one per cell, metrics
  sources/
    id_00.png  id_11.png  id_16.png  id_01.png   # symlinks
    features.json                                # per-photo features
  renders/{photo_id}/{cfg_id}/
    control.png   # the CN input fed to the workflow
    doll.png      # post-Z-Image, pre-swap
    swap.png      # post-HyperSwap
    refine.png    # post-refine (only if refine_denoise != off)
    final.png     # the cell's nominal output (refine if on, else swap)
  logs/
    run.log
    failures.jsonl
```

**`manifest.parquet` columns:**

`photo_id` (str), `cfg_id` (str, "cfg_XXX"), `seed` (int),
`color_match` (str), `face_pixel_budget` (str), `cn_condition` (str),
`cn_strength` (float), `cn_start_percent` (float), `canny_preset` (str | null),
`refine_denoise` (str), `demo_inject` (bool).

**`scores.parquet` columns:**

`photo_id` (str), `cfg_id` (str), `ok` (bool),
`id_cos` (float | null), `det_mode` (str — `default` | `forced` | `failed`),
`det_score` (float | null), `lab_delta_ab` (float | null),
`clip_style` (float | null), `face_frac` (float | null),
`render_s` (float | null), `swap_s` (float | null), `refine_s` (float | null),
`fail_reason` (str | null).

Joining `manifest.parquet` and `scores.parquet` on (`photo_id`, `cfg_id`) is
the analysis interface; Phase 1's follow-up doc is generated from this join.

## Scoring details

- **`id_cos`** — `swap_core.crop_and_upscale` → insightface buffalo_l, cosine
  vs `source_face.normed_embedding`. NaN tagged as null. Matches cn_grid.
- **`det_mode`** — `default` if SCRFD detects on the upscaled crop;
  `forced` if MediaPipe-Face-mesh keypoints pushed through `rec.get()`
  directly; `failed` if neither.
- **`lab_delta_ab`** — `sqrt(Δa² + Δb²)` on mean Lab over the *skin mask*
  (defined below) between source-photo skin and `final.png` skin. Quantifies
  HyperSwap's documented tone wash directly.

The **skin mask** is shared across `color_match` and `lab_delta_ab`: the
mediapipe FaceLandmarker oval polygon dilated by 5 px and intersected with a
Lab-space gate that excludes eyebrows, lips, and eyes (`L*` in [25, 95],
`a*` > 5, `b*` > 8). Defined once in `scripts/photobooth_sweep/features.py`
and imported by both consumers.
- **`clip_style`** — `open_clip` ViT-B/32 image-image cosine between `doll.png`
  (pre-swap) and one curated anchor image
  (`exp_output/cn_grid/renders/id_03_str90_st6.png` — a known clean doll).
  Doll-likeness proxy; insulated from swap-side artifacts.
- **`face_frac`** — SCRFD bbox area / image area on `final.png`. Diagnostic
  for face-pixel budget effects.

## Resumability and error handling

- Cell completion is the unit of resume. On startup, `driver.py` reads
  `scores.parquet`, builds the set of `(photo_id, cfg_id, ok=True)` pairs,
  and runs the complement.
- ComfyUI submit / poll errors → row with `ok=False`, `fail_reason=...`,
  cell stays incomplete (will be retried on next run unless explicitly
  marked done).
- Swap detection failure → row written with `ok=True`, `det_mode=failed`,
  `id_cos=null`. The cell *did* complete; the result is informative.
- Refine workflow failure → keep swap as `final.png`, set `refine_s=null`,
  `ok=True`. Logged in `failures.jsonl`.
- `clip_style` model unavailable → all rows get `clip_style=null`. Run
  continues; logged once at start.

## Components in detail

### `axes.py`

Single file owning the axis catalog. Exports `AXES` (list of axis specs
with name, type, levels, encoder) and `lhs_sample(n_configs, seed) →
list[dict]` using `scipy.stats.qmc.LatinHypercube` with `maximin` optimization
to maximise minimum pairwise distance. Maps the [0,1]^8 cube to mixed-type
levels via per-axis encoders. Writes `manifest.parquet`.

When `cn_condition == "depth"`, `canny_preset` is set to `null` in the
manifest (no Canny preprocessor is invoked). The driver enforces this.

### `color_match.py`

Two pure functions, both operating on (`final_bgr`, `source_bgr`,
`face_mask`):

- `lab_transfer(swap, source, mask)` — match swap's mean+std of Lab a/b
  channels to source's, *only inside `mask`*. Preserves luminance.
- `reinhard_transfer(swap, source, mask)` — classical full-Lab Reinhard,
  mean+std matched inside mask.

`face_mask` is the mediapipe FaceLandmarker oval polygon, dilated 5 px, then
intersected with a Lab-based skin-tone gate to suppress eyebrows / lips.

### `refine.py`

`refine(final_bgr, denoise: float, face_mask) → refined_bgr`. Builds a
masked img2img workflow on the fly:

- Encodes `final_bgr` and `face_mask` to PNG.
- Stages both to remote ComfyUI via `/upload/image`.
- Substitutes into `photobooth_zimage_refine.api.json`
  (`$$IMAGE`, `$$MASK`, `$$DENOISE`, `$$STEPS=6`, `$$SEED=seed`).
- Submits, polls, fetches the result.

### `scorer.py`

Pure functions, no I/O beyond input paths. One entry point
`score(final_path, source_face, source_skin_lab, anchor_clip) → dict`.

### `features.py`

`extract_source_features(photo_path) → dict` computes:

- `skin_lab_L`, `skin_lab_a`, `skin_lab_b` — Lab mean inside skin mask
- `hair_density` — `(face_bbox_area - skin_pixels_in_bbox) / face_bbox_area`
- `has_glasses` — SCRFD landmark separation heuristic (eye-nose
  geometric ratios; binary flag, error-on-the-side-of-False)
- copied: `race`, `gender`, `age_bin` from `manifest.csv`
- `source_face_resolution` — bbox h×w in pixels

Run once per source photo at sweep start; stored in `sources/features.json`.

### `driver.py`

`main()` orchestrates the loop. Stateless except for `scores.parquet`. One
HTTP session reused across cells. The cn_grid pattern is the precedent for
the request/poll loop; lift it.

## Compute budget

Per cell, 3090 warm:
- Z-Image Turbo 6-step @ 1024² ≈ 7 s, @ 768² ≈ 4 s
- HyperSwap CPU ≈ 1.5 s (parallel with next ComfyUI submit if pipelined)
- Refine workflow (when on) ≈ 3 s
- Scoring (id_cos + lab + CLIP) ≈ 0.5 s

Mean cell ≈ 10 s; 160 cells ≈ 27 min compute. Add ~5 min model-load + 5 min
feature-extraction overhead = **~40 min wall-clock Phase 1**.

## Non-goals (explicit)

- *No* PuLID / InfiniteYou / ACE++ identity-injection paths. Those are an
  architectural switch, separately scoped.
- *No* sweep over the swap stage; HyperSwap-1c held constant.
- *No* LoRA sweep. Style LoRAs are an *orthogonal* axis ("which stylization")
  not part of Phase 1's parameter surface.
- *No* GFPGAN / CodeFormer in the loop.
- *No* prompt engineering on the style descriptor (lacquer / matte / paint).
  The prompt comes from the cn_grid sweep's verbatim string except for the
  optional `demo_inject` block.

## Open questions for the implementation plan

- `cn_condition == "canny+depth"` — does `Z-Image-Turbo-Fun-CN-Union` accept
  a 2-channel composite or must we run two preprocessors and the workflow
  handles dual conditioning? Verify on the shard before locking the manifest.
- Anchor image for `clip_style` — `id_03_str90_st6.png` is a placeholder; if
  a curated reference doll exists in `exp_output/matryoshka_bakeoff/`, prefer
  that.
- `has_glasses` heuristic — currently landmark-geometry-based; might need a
  small classifier if the FP rate is high on the calibration set. Defer
  until features.json's first run shows the rate.
