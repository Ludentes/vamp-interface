---
status: live
topic: photobooth-sweep
---

# Photobooth pipeline — full architecture reference

**Date:** 2026-05-20
**Purpose:** Single canonical reference for every stage, parameter, and toggle
in the photobooth pipeline. Used as the design basis for future experiments
(Phase 4+). Pair with [`_topics/photobooth-sweep.md`](_topics/photobooth-sweep.md)
for current beliefs and findings.

## Goal in one sentence

Turn a real photograph of a person into a photo of a vibrant matryoshka doll
whose painted face is recognisably *them*, by rendering a generic doll with the
photo's facial structure as a control signal, then swapping the photo's
ArcFace identity onto the doll's painted face.

## High-level shape

```
source photo
   │
   ▼  preprocess         (CPU, scripts/photobooth_sweep/preprocess.py)
framed canvas + control image (canny or depth)
   │
   ▼  CN render          (remote ComfyUI on Windows 3090 shard)
generic doll with photo's facial geometry
   │
   ▼  identity swap      (CPU, scripts/swap_core.py)
doll with photo's face painted on
   │
   ▼  refine             (remote ComfyUI, low-denoise pass)
seam-cleaned final
   │
   ▼  score              (CPU, scripts/photobooth_sweep/scorer.py)
id_cos, det_score, face_frac, clip_style → scores.parquet
```

Five stages; each one is independently parametrised.

## Hardware and lifecycle

- **Render + refine stages**: remote ComfyUI on Windows RTX 3090 shard
  (`192.168.87.25`, alias `shard`, runbook
  `docs/runbooks/comfyui-windows-shard.md`).
- **All other stages** (preprocess, swap, score): local CPU on the driver host.
  Locked to CPU after `feedback_sweep_avoid_local_gpu.md` (silent OOM-kill
  when CLIP shared the 5090 with another job).
- **No local-GPU usage in the driver.** This is load-bearing.

## Locked Phase-2 defaults (as of 2026-05-20)

Established by Phase 2 (`docs/research/2026-05-20-photobooth-phase2-findings.md`).
These are the values held constant in Phase 3 and forward.

| Parameter             | Locked value     | Source                          |
|-----------------------|------------------|---------------------------------|
| `face_pixel_budget`   | `natural_1024`   | Phase 2 grid                    |
| `cn_condition`        | `canny`          | Phase 2 grid                    |
| `canny_preset`        | `soft`           | Phase 2 grid                    |
| `cn_strength`         | `0.85`           | Phase 1 LHS pilot               |
| `refine_denoise`      | `0.00`           | Phase 1 LHS pilot               |
| `demo_inject`         | `on`             | Phase 1 LHS pilot               |
| `swap_weight`         | `0.5` (no-op)    | HyperSwap default until Phase 3 |
| `mask_erode_px`       | `0` (no erosion) | Phase 4 introduces              |
| `mask_feather_px`     | `0` (no feather) | Phase 4 introduces              |

Everything else listed below is fixed at the workflow / code level and not
currently a sweep axis.

---

## Stage 1: Preprocess

**File:** `scripts/photobooth_sweep/preprocess.py`
**Function:** `build_control(app, comfy_url, src_bgr, cn_condition,
canny_preset, face_pixel_budget) → (control_bgr, (render_h, render_w))`

Source photo → render-sized framed canvas → ControlNet conditioning image.

### Substep 1a — face detection

- `detect_face_rect(app, bgr)` — insightface buffalo_l on the source photo,
  highest-`det_score` face wins.
- Returns `(x0, y0, x1, y1)`. Fails the cell if no face is detected.

### Substep 1b — framing (`frame_face`)

Builds a square `render_h × render_w` canvas of source pixels — no padding,
no reflection. **The face must already fill the canvas; nothing is composited
in.** Two modes selected via `face_pixel_budget`:

| Budget          | Render H×W | Mode      | Behaviour                                                       |
|-----------------|------------|-----------|-----------------------------------------------------------------|
| `natural_1024`  | 1024²      | `natural` | Scale source so `min(H,W) = 1024`, center-crop on face. **Locked default.** |
| `tight_1024`    | 1024²      | `tight`   | Crop `TIGHT_ZOOM × max(face_w, face_h)` square around face, scale to 1024. |
| `natural_768`   | 768²       | `natural` | Same as `natural_1024` at 768 — pruned in Phase 1. |

**Hard-coded knob:** `TIGHT_ZOOM = 1.4` (preprocess.py:30). Open axis if we
re-explore tight framing.

### Substep 1c — control image

Routed on `cn_condition`:

| `cn_condition` | What                                | Knobs                                 |
|----------------|-------------------------------------|---------------------------------------|
| `canny`        | OpenCV `cv2.Canny` on framed canvas | `canny_preset`                        |
| `depth`        | `DepthAnythingV2Preprocessor` node  | None exposed (resolution=1024 hard)   |

#### Canny presets (`CANNY_PRESETS` in preprocess.py:33)

| Preset       | (t_lo, t_hi, blur_sigma) | Notes                                          |
|--------------|---------------------------|------------------------------------------------|
| `soft`       | (50, 150, 1.5)            | **Locked default.** Gaussian-blurred edges, fewer fine lines, leaves Z-Image more freedom. |
| `default`    | (100, 200, 0.0)           | OpenCV defaults, no blur.                      |
| `aggressive` | (150, 250, 0.0)           | Sharp, sparse edges only.                      |

**Closed-over-but-unstudied knobs:** Canny input resolution (currently framed
canvas native = 1024²); whether to canny on grayscale or per-channel.

### Substep 1d — upload

`upload_control` posts the PNG to remote ComfyUI's `/upload/image`
endpoint, returns the server-side path used by the next stage.

---

## Stage 2: CN render

**File:** `scripts/photobooth_sweep/driver.py` (`cn_workflow`, `comfy_submit`).
**Workflow JSON:** `comfyui/workflows/photobooth_zimage_cn.api.json`.

ComfyUI text-to-image with a ControlNet. Produces the generic matryoshka doll
whose facial geometry follows the source photo's edges/depth.

### Workflow node map

| Node | Class                         | Role / fixed value                                                       |
|------|-------------------------------|---------------------------------------------------------------------------|
| 1    | UNETLoader                    | `z_image_turbo_bf16.safetensors` — locked model                          |
| 2    | CLIPLoader                    | `qwen_3_4b.safetensors`, type `lumina2`                                   |
| 3    | VAELoader                     | `z_image_ae.safetensors`                                                  |
| 4    | CLIPTextEncode (positive)     | prompt                                                                    |
| 5    | ConditioningZeroOut (negative)| zero-vector negative — Z-Image Turbo has no negative-prompt training      |
| 6    | EmptySD3LatentImage           | `width × height` = render H×W from Stage 1                                |
| 7    | ModelSamplingAuraFlow         | `shift = 3.0` (fixed)                                                     |
| 20   | ModelPatchLoader              | `Z-Image-Turbo-Fun-Controlnet-Union.safetensors`                         |
| 21   | LoadImage                     | the uploaded control image                                                |
| 22   | ZImageFunControlnet           | `strength = cn_strength`                                                  |
| 8    | KSampler                      | `seed`, `steps=6`, `cfg=1.0`, sampler `euler`, scheduler `simple`, `denoise=1.0` |
| 9    | VAEDecode                     |                                                                           |
| 10   | SaveImage                     | filename prefix `phb_cn_<cell_id>`                                        |

### Parameters

| Param           | Where             | Type / range          | Current locked value | Notes                                       |
|-----------------|-------------------|------------------------|----------------------|---------------------------------------------|
| `cn_strength`   | sweep axis        | float [0.80, 1.00]     | **0.85**             | Phase 1 winner. Lower = more prompt freedom.|
| `seed`          | sweep axis (s0/s1/s2) | int                | `100000 + cfg_idx + 1e6*seed_iter` | Deterministic per (cfg, iter). |
| `prompt`        | derived           | string                 | `BASE_PROMPT + demo_inject`                        | See below. |
| `steps`         | workflow hard     | int = 6                | locked               | Z-Image Turbo native step count.            |
| `cfg`           | workflow hard     | float = 1.0            | locked               | Distilled, no CFG.                          |
| `sampler`       | workflow hard     | `euler`                | locked               | From matryoshka-bakeoff verdict.            |
| `scheduler`     | workflow hard     | `simple`               | locked               | "                                            |
| `denoise`       | workflow hard     | 1.0                    | locked               | Pure text-to-image with CN guidance.        |
| `shift`         | workflow hard     | 3.0 (ModelSamplingAuraFlow) | locked          | Z-Image default.                            |
| `render H×W`    | from preprocess   | (int, int)             | 1024×1024            | From `face_pixel_budget`.                   |

### Prompt construction (`make_prompt`, driver.py:60)

```
BASE_PROMPT = (
  "a vibrant traditional Russian matryoshka nesting doll, glossy red and "
  "gold lacquer, ornate floral painting, with a realistic photographic "
  "human face, soft three-dimensional shading, correct facial proportions, "
  "defined nose and lips, natural-sized eyes, centered frontal face, "
  "wooden doll, plain background")
```

If `demo_inject == "on"`: prepended with `"a {age_bin}-year-old {race} {man's|woman's|person's} face, "` from `data/importer/identities/manifest.csv`.

**Open axes (not yet swept):**
- alternate prompt templates ("oil painting", "lacquerware close-up",…)
- token-weighting on `matryoshka`/`wooden`/`lacquer`
- adding a matryoshka-style LoRA on the Z-Image-Turbo UNet
- `shift` (ModelSamplingAuraFlow) — controls noise-schedule shape
- `steps` — 6 is Z-Image Turbo's native count, but maybe 8 with the right scheduler unlocks more style

### Output

A 1024×1024 BGR PNG saved to `<cell_dir>/render.png`. The painted doll-face
occupies a sub-region of this image, typically ~25% of canvas area.

---

## Stage 3: Identity swap

**File:** `scripts/swap_core.py`
**Function:** `swap_identity(app, swapper, doll_bgr, source_face, collapse=True,
restore=False, swap_weight=0.5) → (result_bgr, det_mode, det_score)`

Swap the source photo's ArcFace identity onto the doll's painted face.
CPU-only, never touches the GPU.

### Substep 3a — locate the doll's face

1. `mediapipe_kps_bbox(doll_bgr)` — MediaPipe FaceLandmarker (Tasks API,
   model file shipped next to swap_core.py). Returns 5-point arcface kps +
   bbox, or `(None, None)` on failure. **MediaPipe is used because SCRFD
   alone can't see flat painted doll faces** — flagged in Phase 2 as a
   structural failure mode for some photos.
2. If no face: return doll unchanged, `det_mode="failed"`.
3. `crop_and_upscale(doll_bgr, bbox)` — Lanczos-upscale the face crop to
   512 px so SCRFD has enough pixels.
4. Re-run MediaPipe on the upscaled crop.

### Substep 3b — eye collapse (toggle: `collapse=True`)

`collapse_eyes(up, kps_up)` shrinks oversized painted doll-eyes to small
folk-art dots **before** the swap. inswapper is identity-only and would
otherwise carry the giant target eyes through.

Parameters (constants at top of swap_core.py:196):
- `_EYE_WIN_FRAC = 0.40` — disk radius around eye centre (×inter-eye dist)
- `_EYE_DOT_FRAC = 0.06` — replacement folk-art dot radius
- `_EYE_DARK_T = 110` — grayscale threshold below which a pixel is "eye paint"
- `_EYE_OVERSIZE_FRAC = 0.18` — collapse only if dark fills >18% of disk

Only fires when the doll's eyes are oversized; small-eye dolls are skipped.
**Open axes:** disable collapse entirely (`collapse=False`), tighten/loosen
thresholds, or replace with a learned mask.

### Substep 3c — SCRFD-on-crop or MediaPipe fallback

`app.get(work)` on the (possibly collapsed) 512-up-crop. If SCRFD finds a
face → `det_mode="default"`. Else if MediaPipe kps available →
`det_mode="forced"` (synthetic Face with kps + bbox + det_score=1.0). Else
`det_mode="failed"` and the cell aborts.

### Substep 3d — HyperSwap

`swapper.get(work.copy(), target, source_face, paste_back=True,
weight=swap_weight)`

**Model:** HyperSwap 1c 256px (FaceFusion port), CPU, ONNX. Chosen over
inswapper_128 in 2026-05-18 bake-off (`docs/research/2026-05-18-face-swapper-landscape.md`).

#### Internals (swap_core.py:71)

1. Warp doll's face crop to 256×256 via 5-pt `_ARCFACE_128` template.
2. Normalise to `[-1, 1]`.
3. Take L2-normalised source ArcFace embedding (from buffalo_l).
4. **`face_swapper_weight` blend** (Phase 3 axis):
   `α = interp(w, [0,1], [+0.35, -0.35])`
   `src ← (1-α)·src + α·target_emb`, re-normalise.
   - `w=0.5` is the historical no-op default.
   - `w=0.0` pulls the embedding 35% toward the doll's own painted-face
     embedding (matryoshka-deferred).
   - `w=1.0` extrapolates 35% past pure source (more photoreal).
   - Requires `target.normed_embedding` (filled by buffalo_l in 3c). If
     missing (e.g. MediaPipe-forced target), the blend is silently no-op.
5. Run ONNX → `(out_image, face_mask)`. Both 256² in crop space.
6. Inv-affine warp `out` + `face_mask` back into 512-up-crop space.
7. **HyperSwap-internal paste-back:**
   `blended = warped * mask + img * (1 - mask)` (swap_core.py:130-132).
   This `mask` is the per-pixel swap-region mask the model emits — **this is
   the mask Phase 4 erodes / feathers.**

#### Substep 3e — face restoration (toggle: `restore=False`)

`restore_face(swapped)` runs GFPGAN over the swapped 512-crop. **Off by
default** — the 2026-05-18 A/B measured GFPGAN *lowering* median id_cos
0.773 → 0.525. It regularises toward a generic restoration prior.

### Substep 3f — paste back to full doll

The restored 512-crop is Lanczos-resized back to its source rectangle
`(rx0, ry0, rx1, ry1)` in the full doll image, then alpha-blended via
`_feathered_mask(h, w, feather_frac=0.12)`. This is an outer rectangle
Gaussian-feathered mask — purely to hide the crop seam, not to control
how much swap leaks into the doll.

**`feather_frac = 0.12`** is hard-coded (swap_core.py:220). Not currently a
sweep axis but available.

### Parameters / toggles

| Param            | Type     | Default | What it does                                          |
|------------------|----------|---------|-------------------------------------------------------|
| `swap_weight`    | float    | 0.5     | **Phase 3 axis.** ArcFace-embedding mix toward target. |
| `collapse`       | bool     | True    | Shrink oversized doll-eyes pre-swap.                  |
| `restore`        | bool     | False   | GFPGAN over swap output. Off (hurts identity).        |
| `mask_erode_px`  | int (256²)| 0      | **Phase 4 axis.** Pixels to erode HyperSwap's `face_mask` before paste. |
| `mask_feather_px`| int (256²)| 0      | **Phase 4 axis.** Gaussian-blur the mask edge.       |
| `feather_frac` (outer) | float | 0.12  | Crop-rectangle feather. Not currently swept.         |
| `_EYE_*` constants | float  | see above | Eye-collapse thresholds. Not currently swept.      |
| `_ARCFACE_128`   | array    | fixed   | 5-pt warp template. Not a knob.                       |
| crop size        | int      | 512     | `crop_and_upscale` target. Not currently swept.       |

### Output

A full-resolution doll PNG where the painted face has been replaced with the
swapped result, saved as `<cell_dir>/swap.png`.

---

## Stage 4: Refine

**File:** `scripts/photobooth_sweep/refine.py`
**Workflow:** `comfyui/workflows/photobooth_zimage_refine.api.json`

Whole-image low-denoise pass through Z-Image Turbo. Hides paste-back seams
and any residual artefacts. **Short-circuits if `denoise == 0`** — returns
the swap image untouched, no network round-trip.

### Workflow

| Node | Class             | Notes                                                       |
|------|-------------------|-------------------------------------------------------------|
| 1-3  | UNET/CLIP/VAE     | Same Z-Image Turbo + qwen + ae as Stage 2.                  |
| 4    | CLIPTextEncode    | same prompt as Stage 2                                      |
| 5    | ConditioningZeroOut | zero negative                                             |
| 6    | LoadImage         | swap.png as init                                            |
| 7    | VAEEncode         | encode to latent                                            |
| 8    | ModelSamplingAuraFlow | shift=3.0                                              |
| 9    | KSampler          | seed, **steps=6 (hard)**, cfg=1.0, euler/simple, `denoise=$$DENOISE` |
| 10   | VAEDecode         |                                                             |
| 11   | SaveImage         |                                                             |

### Parameters

| Param            | Type   | Locked / range   | Notes                                              |
|------------------|--------|------------------|----------------------------------------------------|
| `refine_denoise` | float  | **0.00** (locked, Phase 1) | Phase 1 found higher values destroy identity. |
| `seed`           | int    | same as Stage 2 seed       | Deterministic.                              |
| `prompt`         | string | reuses Stage 2's prompt    | Same `BASE_PROMPT + demo_inject`.           |
| `steps`          | int    | 6 (workflow hard)          | Locked.                                     |

### Output

`<cell_dir>/refined.png` — the final pipeline output, scored in Stage 5.
With `refine_denoise=0` (current lock), this is bit-identical to `swap.png`.

---

## Stage 5: Score

**File:** `scripts/photobooth_sweep/scorer.py`
**Function:** `score_cell(app, refined_bgr, src_emb, anchor_emb, det_mode, wall_clock)`

All metrics on the *refined* image (final output). Detector is `app` =
buffalo_l SCRFD.

| Metric        | Range       | Definition                                                                 |
|---------------|-------------|----------------------------------------------------------------------------|
| `id_cos`      | [-1, 1] / NaN | Cosine(refined ArcFace embedding, source ArcFace embedding). NaN if no face found. |
| `det_score`   | [0, 1] / NaN | SCRFD detection confidence on the refined image's largest face.            |
| `det_mode`    | enum string | `default` / `forced` / `failed` / `exception` — provenance flag from swap stage. |
| `face_frac`   | [0, 1]      | Largest face bbox area / image area. Catches "the swap ate the whole image" pathologies. |
| `clip_style`  | [-1, 1]    | Cosine(CLIP image embedding, anchor doll's CLIP embedding). Anchor = `exp_output/matryoshka_bakeoff/renders/zimage_turbo_st06_euler_simple_seed74029470.png`. |
| `wall_clock`  | seconds     | End-to-end per-cell time.                                                  |

**CLIP model:** `open_clip` ViT-B-32 OpenAI pretrained, **CPU only**
(scorer.py:45-46 — load-bearing per `feedback_sweep_avoid_local_gpu.md`).

### What we look for

- **`id_cos`** — Phase-2 baseline lock at 0.6–0.8 range. Drops monotonically
  with lower `swap_weight` (Phase 3: 0.5→0.0 drops 0.705→0.586).
- **`clip_style`** — Phase 3 finding: **flat across swap_weight** (0.666–0.700).
  The doll's ArcFace target is OOD; pulling toward it doesn't add style.
- **`face_frac`** — sanity check. `<0.005` means depth-collapse or similar
  catastrophe.
- **NaN `id_cos` rate** — Phase 3: 19% of cells. Mostly photo-dependent
  (bimodal: 4/20 photos are structurally hard — beard, glasses, hat).

---

## Driver loop

**File:** `scripts/photobooth_sweep/driver.py`
**Entry:** `main()` → resumes from `scores.parquet`, iterates photos × configs.

### Modes

| `--mode`   | Configs source                                  | Notes                                          |
|------------|--------------------------------------------------|------------------------------------------------|
| `phase1`   | `lhs_sample(n_cells, seed)` — Latin Hypercube   | Initial 6-axis LHS pilot.                      |
| `phase2`   | `phase2_configs()` — hard-coded grid             | Post-pruning, 6 cfgs × 20 photos × 3 seed iters. |
| `phase3`   | `phase3_configs()` — swap_weight sweep           | 6 cfgs × 20 photos × 1 seed.                  |
| `phase4`   | `phase4_configs()` — mask × swap_weight (planned)| 12 cfgs × 20 photos × 1 seed.                 |

### Per-cell loop (`run_cell`, driver.py:129)

```
ctrl, render_hw = pp.build_control(...)        # Stage 1
ctrl_name        = pp.upload_control(...)
prompt           = make_prompt(pid, demo, inject)
wf               = cn_workflow(...)
render           = comfy_submit(wf)             # Stage 2
swap, det_mode, det_pre = swap_core.swap_identity(
                     app, swapper, render, src_face,
                     collapse=..., restore=..., swap_weight=...)  # Stage 3
refined          = rf.refine(swap, denoise=..., seed=...)  # Stage 4
row              = sc.score_cell(refined, src_emb, anchor_emb, ...)  # Stage 5
append_row(scores_path, row)
```

### Resumability

`existing_cells(scores_path)` reads `cell_id`s already in the parquet.
Cells in the set are skipped on re-run. **Append-only parquet**
(`append_row` reads → concats → writes whole file each time — fine at our
scale but O(n²) writes; acceptable up to ~1k cells).

### Output layout

```
exp_output/photobooth_phase<N>/
├── manifest.json                — the configs list
├── scores.parquet               — one row per cell, append-only
└── cells/
    └── <photo_id>__cfg<NNN>[_s<i>]/
        ├── ctrl.png             — Stage 1 control image
        ├── render.png           — Stage 2 generic doll
        ├── swap.png             — Stage 3 swap output
        └── refined.png          — Stage 4 final (== swap if denoise=0)
```

### Seed scheme

`seed = 100000 + cfg_idx + 1000000 * seed_iter`. Deterministic per (cfg, iter).
**Important:** different `cfg_idx` produces a different render even if all
generation parameters are identical — the seed is keyed to `cfg_idx`. To
reuse a render across cfgs, copy the file rather than re-render.

---

## What an "experiment phase" is

Each Phase iterates this loop:

1. Identify a hypothesis or open axis.
2. Define a new `phaseN_configs()` in `axes.py` (or extend `lhs_sample`).
3. Add new parameter columns to the parquet schema (driver pulls them off
   the cfg dict; `append_row` accepts new columns via `pa.concat_tables`
   schema promotion).
4. Run the sweep on the shard. Resumable.
5. Render the contact sheet via
   `python -m scripts.photobooth_sweep.contact_sheet --root <phase_dir>
     --view grid --stages render,refined` (or `swap`).
6. Score-table analysis: per-cfg means, per-photo bimodality check, named
   failure modes (see Phase 2 findings doc for the taxonomy).
7. Update `_topics/photobooth-sweep.md` with the new findings doc link and
   any defaults that get re-locked.

## Phase history

| Phase | Cells | Hypothesis tested                            | Verdict                                            | Findings doc                                       |
|-------|-------|----------------------------------------------|----------------------------------------------------|----------------------------------------------------|
| 0     | —     | matryoshka bake-off (model selection)        | Z-Image Turbo @6 steps wins                        | `2026-05-18-matryoshka-bakeoff-results.md`         |
| 1     | 40 LHS| 6-axis response surface                      | Lock `cn_strength≈0.85, refine=0, demo_inject=on`; prune `natural_768` and refine-as-axis | `2026-05-20-photobooth-phase1-findings.md` |
| 2     | 360   | cn_condition × face_pixel_budget × canny_preset | Robust default identified; photo-bimodal failure pattern; 5 named failure modes | `2026-05-20-photobooth-phase2-findings.md` |
| 3     | 120   | `swap_weight` ∈ {0.5, 0.4, 0.3, 0.2, 0.1, 0.0} as style dial | **Falsified**: id_cos drops monotone, but clip_style flat. Doll ArcFace target is OOD (matches research). | (Phase-3 wrap-up forthcoming.) |
| 4     | 240*  | mask erode × feather × `swap_weight ∈ {0.3, 0.1}` (planned) | TBD — tests "shrink swap surface to preserve style" hypothesis | TBD |

*Phase 4 reuses Phase 3 renders, no ComfyUI render calls.

## Open axes (unexplored)

Roughly ordered by "most likely to move the needle":

**Render side (Stage 2):**
- Matryoshka-style LoRA on the Z-Image-Turbo UNet
- Prompt token-weighting on `matryoshka`/`wooden`/`lacquer`
- Alternate prompt templates ("lacquerware close-up", "painted oil portrait", …)
- `ModelSamplingAuraFlow.shift` (currently 3.0)
- KSampler `steps` (currently 6)
- Per-token negative prompt (would need a non-zero negative chain)

**Swap side (Stage 3):**
- Mask erosion / feather (Phase 4 — testing now)
- `collapse=False` toggle — does removing eye-collapse change anything?
- Tighter `_EYE_OVERSIZE_FRAC` to fire on more dolls
- Source-embedding doping (average over N source photos, or
  `mean(doll_arcface) − mean(photo_arcface)` stylization direction — see
  `2026-05-20-arcface-topology.md`)
- Replace HyperSwap with a swap conditioned on style features (open research)

**Refine side (Stage 4):**
- Non-zero `refine_denoise` paired with stylization tokens to ADD style
  post-swap (Phase 1 found bare denoise destroys identity, but didn't
  combine with prompt push)
- Refine as a LoRA pass with matryoshka-style LoRA

**Score side (Stage 5):**
- Alternate `clip_style` anchor or anchor-average
- Per-region scoring (eye/lip/cheek-area separately)

## Cross-references

- `_topics/photobooth-sweep.md` — live state, findings index
- `2026-05-20-arcface-topology.md` — why `swap_weight` alone doesn't stylize
- `2026-05-20-hyperswap-parameters.md` — HyperSwap knob inventory
- `2026-05-20-photobooth-phase2-findings.md` — failure-mode taxonomy
- `2026-05-20-photobooth-phase1-findings.md` — LHS pilot results
- `2026-05-18-face-swapper-landscape.md` — why HyperSwap over inswapper
- `2026-05-18-matryoshka-bakeoff-results.md` — why Z-Image Turbo
- `runbooks/comfyui-windows-shard.md` — remote-render host
