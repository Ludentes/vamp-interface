---
status: live
topic: photobooth-sweep
---

# Photobooth Phases 3–4 findings + human judging verdicts

Experiments ran 2026-05-20 (immediately after Phase 2); the writeup and the
human judging verdicts landed 2026-07-16 during thread cleanup. Data:
`exp_output/photobooth_phase3/`, `exp_output/photobooth_phase4/`,
`exp_output/photobooth_judging/` (all local-only; only `scores.parquet` +
`manifest.json` + driver logs are committed — cells and sheet PNGs are too
heavy for git).

## What was swept

**Phase 3 — identity axis (`swap_weight`).** 120 cells: 20 identities ×
6 configs × 1 seed. Base config frozen at the Phase 2 robust default
(`natural_1024`, canny/soft, `cn_strength=0.85`, `refine_denoise=0.00`,
`demo_inject=on`). Swept HyperSwap's `face_swapper_weight`
∈ {0.5 (historical no-op), 0.4, 0.3, 0.2, 0.1, 0.0} — implemented in
`scripts/swap_core.py` as an embedding blend
α = interp(weight, [0,1], [+0.35, −0.35]); the source ArcFace embedding is
lerped toward the target's (the painted doll face) and renormalised. Lower
weight = more matryoshka, less photoreal. See
`2026-05-20-hyperswap-parameters.md` for the knob's provenance.

**Phase 4 — mask geometry axis.** 240 cells: 20 identities ×
{`mask_mode` erode|feather} × {`mask_radius` 16, 32, 64} ×
{`swap_weight` 0.1, 0.3}. Same frozen base. `mask_erode_px` shrinks the
ONNX-emitted swap mask (more painted doll survives);
`mask_feather_px` Gaussian-softens the seam. Both operate in 256-crop
space, applied before the inverse-affine paste-back.

## Quantitative results

### Phase 3 — swap_weight

| swap_weight | id_cos (valid) | clip_style | nan / 20 |
|---|---|---|---|
| 0.5 (baseline) | 0.705 | 0.666 | 2 |
| **0.4 (light mix)** | **0.743** | **0.700** | **2** |
| 0.3 | 0.668 | 0.700 | 5 |
| 0.2 | 0.696 | 0.680 | 3 |
| 0.1 | 0.619 | 0.679 | 7 |
| 0.0 | 0.586 | 0.684 | 4 |

Surprise: `swap_weight=0.4` beats the 0.5 baseline on *both* identity and
style — the light pull toward the doll embedding is not a pure trade-off,
it's a free improvement at this step size. Below 0.3 the identity floor
falls out (per-photo bimodality again: e.g. id_18 goes 0.76 → 0.20 across
the sweep; id_19 is broken at every weight).

### Phase 4 — mask geometry

| mode | radius | sw=0.1 id_cos | sw=0.3 id_cos | clip_style |
|---|---|---|---|---|
| erode | 16 | 0.537 | 0.611 | 0.67 |
| erode | 32 | 0.417 | 0.475 | 0.69 |
| erode | 64 | 0.151 | 0.170 | 0.72 |
| feather | 16 | 0.577 | 0.673 | 0.67 |
| feather | 32 | 0.572 | 0.667 | 0.67 |
| feather | 64 | 0.554 | 0.649 | 0.67 |

Erosion monotonically destroys identity while buying almost no style
(+0.05 clip_style at radius 64 for a −0.44 id_cos collapse) — a bad dial.
Feathering is nearly radius-invariant: it neither hurts identity nor moves
style, i.e. it's a **seam-quality knob, not a style knob**. Verdict: keep
feather available for seam artifacts, drop erode from the axis catalog.

## Human judging (2026-07-16)

Sheets built by `scripts/photobooth_sweep/judging_sheet.py` ("collage of
collages", pick a column not an image):

- **`sheet_p3.png`** — the 6 Phase-3 swap_weight columns over all 20
  identities. **Verdict: "Light mix" (swap_weight=0.40) is best.**
  Agrees with the metrics (best id_cos *and* best clip_style).
- **`sheet_p4.png`** — all-phases survey (note: despite the name this is
  NOT the Phase-4 mask sweep; it's a 15-column cross-phase survey — 6
  Phase-2 render-axis configs + 3 Phase-1 corners + the 6 Phase-3 columns,
  subsampled to the 4 Phase-1 identities). **Verdict: "P2 — tight + aggr
  canny" (`tight_1024` + canny/aggressive) is best.**

The tight+aggressive-canny pick is a conscious high-risk/high-reward call:
in Phase 2 `tight_1024` had the best valid-cell identity (cfg004 mean
id_cos 0.785) but a ~40 % nan rate (24/60 cells produced no detectable
face). Choosing it as the product look means the nan failure mode must be
handled by retry/fallback, not ignored — see "Open items" below.

## Production recipe (as of 2026-07-16)

```
face_pixel_budget = tight_1024
cn_condition     = canny, preset=aggressive, cn_strength=0.85
refine_denoise   = 0.00
demo_inject      = on
swap_weight      = 0.40          # HyperSwap embedding light mix
mask             = none          # feather optional for seam repair only
```

Note the two winning picks were never run *together* (Phase 3 ran on
natural_1024/soft). The combined cell is untested.

## Open items

1. **Run the combined recipe** (tight+aggressive × sw=0.4) on the full
   20-identity set to confirm the picks compose.
2. **Handle the tight_1024 nan mode** — fallback to natural_1024 when the
   swap finds no face, or a detect-retry loop. Without this the product
   recipe fails ~1/3 of photos outright.
3. **Adaptive per-photo lookup** (from Phase 2) is still the open axis for
   the 4/20 structurally hard photos (beard, glasses, hat, depth-collapse).
4. Wire the winning recipe into `src/group_photobooth/face_renderer.py`,
   which currently wraps the Phase-3 *heavy* mix default from the
   group-photobooth spec — revisit that choice against this verdict.
