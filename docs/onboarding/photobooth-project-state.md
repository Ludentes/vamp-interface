# Photobooth — current state (2026-07-16)

The product: a photo goes in, a **matryoshka-doll portrait of that
person** comes out — recognizably them, unmistakably a painted doll.
The group variant composites several people onto a chosen background as
a group portrait. This guide is the orientation snapshot; the living
source of truth is `docs/research/_topics/photobooth-sweep.md` — read
it first whenever you resume work.

## How the pipeline works

Single-face render, in order:

1. **Preprocess** — detect the face, crop at a chosen pixel budget
   (`natural_1024` = generous context, `tight_1024` = face-dominant).
2. **Stylize** — Z-Image Turbo + Fun-CN-Union ControlNet (canny edges
   from the photo constrain structure; `cn_strength` sets how hard) +
   demographic tag injection. This is the img2img/denoise machinery
   from guide 1, with edges as the leash.
3. **Identity swap-back** — HyperSwap-1c pastes the person's identity
   into the doll render. Our fork (`scripts/swap_core.py`) adds three
   knobs: `swap_weight` (ArcFace embedding blend between person and
   doll; 0.5 = pure person), `mask_erode_px`, `mask_feather_px`.
4. **Score** — `id_cos` (ArcFace similarity to the source person) and
   `clip_style` (how doll-like); per-cell results in
   `exp_output/photobooth_phase*/scores.parquet`.

Everything runs on the Windows 3090 shard via
`scripts/photobooth_sweep/driver.py`; workflows are
`comfyui/workflows/photobooth_zimage_cn.api.json` and
`photobooth_zimage_refine.api.json`.

## What has been swept and judged

Four sweep phases (May 2026) + human judging (July 2026):

- **Phase 1** (LHS pilot) locked `refine_denoise=0.00`,
  `demo_inject=on`, `cn_strength≈0.85`; pruned `natural_768`.
- **Phase 2** (360 cells) mapped the render axes. `tight_1024` +
  aggressive canny gave the best identity (id_cos 0.785) but ~40% of
  cells found no face (nan). Found the per-photo bimodality: 4/20
  photos always work, 4/20 (beard, glasses, hat, depth-collapse)
  are structurally hard.
- **Phase 3** (swap_weight sweep) — surprise result:
  **`swap_weight=0.40` beats the 0.5 baseline on both identity
  (0.743 vs 0.705) and style (0.700 vs 0.666)**. A light pull toward
  the doll embedding is a free improvement, not a trade-off.
- **Phase 4** (mask geometry) — erode is a **dead dial** (monotonically
  destroys identity, buys almost no style); feather is radius-invariant
  and only useful as seam repair.
- **Human judging** confirmed the metrics: light mix (0.40) won the
  swap_weight sheet; `tight_1024` + aggressive canny won the
  cross-phase survey.

## The production recipe

```
face_pixel_budget = tight_1024
cn_condition     = canny, preset=aggressive, cn_strength=0.85
refine_denoise   = 0.00
demo_inject      = on
swap_weight      = 0.40          # HyperSwap light mix
mask             = none          # feather only for seam repair
```

Full evidence: `docs/research/2026-07-16-photobooth-phase3-4-findings.md`.

## Group photobooth

Feature-complete per plan as of commit `7f6bc97`: person detection
(YOLO + buffalo_l, SAM2 masks via ComfyUI), per-person render + cutout,
painter's-order composite with lab_match + drop_shadow, background
library, silhouette handling, e2e tests. Code:
`src/group_photobooth/`. Runbook:
`docs/research/2026-05-21-group-photobooth-architecture.md`.

## Open items — likely your first tasks

1. **Run the combined recipe.** The two judged winners (`tight_1024`+
   aggressive canny, and `swap_weight=0.40`) were **never run
   together** — Phase 3 swept on `natural_1024`/soft. Verify they
   compose on the full 20-identity set.
2. **Handle the tight_1024 nan mode.** ~40% of photos yield no
   detectable face under the tight crop. Needs a fallback to
   `natural_1024` (or a detect-retry loop) or the product fails ~1/3 of
   inputs outright.
3. **Reconcile `src/group_photobooth/face_renderer.py`** — it still
   hardcodes the old heavy mix (`swap_weight=0.10`) from the
   group-photobooth spec, which conflicts with the judged verdict.
4. **Adaptive per-photo config** — the open axis for the 4/20
   structurally hard photos.

## Reading list, in order

1. `docs/research/_topics/photobooth-sweep.md` — current beliefs +
   pointers (always start here)
2. `docs/research/2026-07-16-photobooth-phase3-4-findings.md` — latest
   evidence + verdicts
3. `docs/research/2026-05-20-photobooth-phase2-findings.md` — failure
   modes catalog
4. `docs/research/2026-05-21-group-photobooth-architecture.md` — group
   pipeline runbook
5. `docs/superpowers/specs/2026-05-19-photobooth-phase1-sweep-design.md`
   — original sweep design rationale
