---
status: superseded
topic: lam-chibi-recipe
superseded_by: 2026-05-17-chibi-geometry-redesign-design.md
---

# Differentiable Chibi-Geometry Recipe — Design

**Date:** 2026-05-15
**Topic:** `lam-chibi-recipe`

## Problem

The current chibi deformation (`scripts/chibi_make_assets.py`) is a y-stratified affine field with **hand-tuned constant knots** (`T_KNOTS`/`SY_KNOTS`/`SR_KNOTS`). It bends the head into roughly-chibi proportions but: (a) the knots were eyeballed, not fitted to any rule; (b) it has **no per-feature targeting** — every vertex at a given height scales identically, so it cannot enlarge the eyes, collapse the nose to a button, or compress the mouth to a strip; (c) re-tuning for a new anchor is manual.

The research docs `2026-05-15-chibi-painter-proportion-rules.md` (rules + the chibi quarter-grid) gave us an explicit, measurable target. This design turns that target into a **differentiable geometry fit**: the painter rules become an optimization loss, the deformation becomes a small set of learnable parameters, and the fit runs the same way on any anchor.

## Scope

**In scope (this design — Stage 1, geometry):** vertical quarter-grid remap, per-height radial skull shaping, per-feature region transforms (eyes, nose, mouth), fit by a differentiable proportion loss; emit the existing 4-asset bundle.

**Out of scope:** the render-space iris-through-lid leak fix — that is **Stage 2**, already spec'd and planned (`2026-05-15-chibi-differentiable-leak-fix-design.md` / `-plan.md`). Stage 2 consumes whatever chibi assets exist, so it runs *after* Stage 1 unchanged. Also out: flat-skin (SH-DC appearance edit, rule 8b — separate appearance pass) and hair (rule 9 — not in FLAME topology).

## Target — the chibi quarter-grid

From the research doc, expressed on the vertical head axis `u ∈ [0,1]` (`u=0` crown, `u=1` chin):

| Landmark | Realistic `u` | Chibi target `u` |
|---|---|---|
| Brow | ~0.33 | ~0.42 |
| Eye centre | ~0.42 | **0.50** (2/4 line) |
| Nose tip | ~0.67 | **0.625** |
| Mouth (lip line) | ~0.80 | **0.75** (3/4 line) |
| Chin | 1.00 | 1.00 |

Plus per-feature size rules, all expressed as **multipliers of the realistic (undeformed) extent** so every target is reachable on the same scale: eye-lid aperture ~2× (enlarge), nose z-depth → 0.45 (collapse to a button), mouth height → 0.55 (compress to a strip). The painter "eyes ≈ ¼ head" rule refers to the whole eye *graphic*, not the lid aperture FLAME landmarks trace — so it is recast here as an aperture multiplier rather than a head-fraction. These are the differentiable rules 1–7 + 8a; they are all expressible as **vertex-position targets**, so the fit needs neither a face detector nor the renderer.

## Approaches Considered

**A — Direct closed-form fit (no optimization).** Solve knot positions and region scales algebraically from the quarter-grid. *Rejected:* cannot jointly satisfy the mesh-smoothness regularizer (which couples all vertices at region boundaries) or weigh competing soft targets; brittle; and the blendshape-basis rescaling needs the field's exact local Jacobian, which is clean to get from autograd and painful by hand.

**B — Structured differentiable field fit (recommended).** Parameterize the deformation as a small (~13-param) structured field: a learnable monotone vertical remap + learnable per-height radial scale + learnable per-feature region transforms. Optimize the parameters with Adam against a proportion loss computed on FLAME canonical landmark vertices, plus a Laplacian smoothness term and a minimal-deformation regularizer. No renderer, no detector. *Chosen:* the structure encodes the painter rules so any optimum is a valid chibi (cannot tear or go feral); ~13 params fit in seconds; autograd gives the exact field Jacobian for basis rescaling for free; and "fit a new anchor" becomes "run Adam" — it generalizes as a procedure.

**C — Free-form per-vertex displacement, render-in-the-loop.** Optimize a 20018×3 displacement through LAM's differentiable renderer against a landmark loss on the render. *Rejected:* the targets are already exact in vertex space, so rendering buys nothing; 60k params overfit and destroy identity; orders of magnitude slower. (LAM's differentiable renderer is the right tool for Stage 2's occlusion loss, which genuinely is not expressible in vertex space — not for this.)

## Architecture — Approach B

Four units, each independently testable.

### `src/chibi/field.py` — the differentiable deformation field

A `torch.nn.Module`, `ChibiField`, holding the learnable parameters and applying the deformation. Operates on the normalized head fraction `u ∈ [0,1]` of each vertex (`u=0` crown … `u=1` chin), reusing the existing anchor/extent convention but flipped to the painter axis.

Parameters (all `nn.Parameter`, ~13 total):
- **Vertical remap** — chibi `u` of the 4 interior knots {brow, eye, nose, mouth}; crown/chin pinned to 0/1. Stored as 5 positive softplus increments normalized to sum to 1, so the remap is **monotone by construction** (no tearing, no constraint needed). Realistic knot `u` positions are fixed constants (from the table above).
- **Radial skull scale** — per-height radial (xz) scale at the 5 knots (`g(u)`), 5 params. Identity = all 1.0. This is the existing `SR` profile made learnable; it carries the rounded-skull / temple shaping (rule 1).
- **Region transforms** — applied after the global remap, one similarity transform per FLAME feature region about that region's centroid: `s_eye` (uniform enlarge, 1 param), `s_nose_xy` + `s_nose_z` (anisotropic — width shrink + depth/bridge collapse, 2 params), `s_mouth_y` (vertical compress to a strip, 1 param).

`forward(verts, u) → deformed_verts`:
1. Global vertical remap: `u_chibi = piecewise_linear(u; realistic_knots → chibi_knots)`, new `y = anchor + u_chibi · head_height`.
2. Global radial: `xz *= interp(g, u)` about the head's z-centroid.
3. Region transforms: for each region, blend the region's similarity transform in with a **fixed smooth falloff** (Euclidean distance to the mask, normalized) so the transform fades to identity outside the mask — no boundary tear.

The module also exposes `local_jacobian(verts, region_weights) → (N,3,3)`: the per-vertex Jacobian of the **complete** field — vertical remap, radial scale, *and* the per-feature region transforms — obtained via autograd (`torch.func.jacrev` vmapped). This is what rescales the blendshape basis: `bs_chibi[v] = J[v] · bs_real[v]`, replacing the hand-derived `diag(s_r, s_y, s_r)`.

Including the region transforms is **load-bearing, not optional**. The defining chibi move enlarges the eye ~2×; if the blink/squint basis rows are not pushed through the eye region Jacobian, the lid travels the original (smaller) aperture height and cannot close the enlarged eye — the iris-through-lid artifact, reconstructed by geometry. (Stage 2's splat-scale fix corrects render density, not lid travel — orthogonal; this must be right independently.) The pushforward is the principled operation — the exact differential of the deformation applied uniformly to all 52 rows — and is distinct from the falsified ad-hoc per-row eye-basis rescaling (retired-hypotheses list, topic index): that was hand-tuned per-blendshape guessing, this is closed-form. Region centroids are held fixed (detached) when forming the Jacobian, so each vertex's `J` is its local linear map; the residual centroid coupling is O(1/M) per region and negligible.

Region masks: load `eye_region ∪ left_eyeball ∪ right_eyeball`, `nose`, `lips` from `FLAME_masks.pkl`. Falloff widths are fixed constants (not learned — YAGNI).

### `src/chibi/landmarks.py` — differentiable landmark positions + targets

Loads `landmark_embedding_with_eyes.npy` (`full_lmk_faces_idx` (1,70), `full_lmk_bary_coords` (1,70,3)). A landmark position is the barycentric combination of its triangle's 3 deformed template vertices — a differentiable function of the deformed verts. Standard 68-point indexing within the 70: chin = 8, nose tip = 30, eye verts 36–47, brow verts 17–26, mouth verts 48–67.

Exposes:
- `landmark_lines(verts) → dict` — the deformed `u` of {brow, eye, nose, mouth, chin} as mean landmark `u` per group.
- `feature_extents(verts) → dict` — eye bbox height, nose z-depth, mouth height, each measured on the relevant region/landmark verts.
- `QUARTER_GRID_TARGETS` — the constant target dict from the table above + size targets (eye height, nose-depth fraction, mouth height).

### `src/chibi/fit.py` — the optimization loop

`fit_chibi_field(template_verts, faces, masks, landmarks, *, n_steps, lr) → ChibiField`.

Loss (`total = L_landmark + λ_smooth·L_smooth + λ_reg·L_reg`):
- **`L_landmark`** — squared error between deformed landmark lines / feature extents and `QUARTER_GRID_TARGETS`.
- **`L_smooth`** — mesh-Laplacian of the displacement field `(v_chibi − v_real)`; keeps region transforms blending smoothly into the global field.
- **`L_reg`** — minimal-deformation guardrail: L2 of region scales toward 1.0 and remap knots toward identity. This is the identity guard — with a structured 13-param field, "do the least deformation that hits the targets" is sufficient; no perceptual identity metric is needed.

Adam, ~300 steps, CPU (the field is tiny; no GPU, no renderer). Defaults: `lr=0.05`, `λ_smooth=1.0`, `λ_reg=0.005`. `λ_reg` is low because every feature line and size now carries a reachable target — reg only picks the minimal-norm solution among the underdetermined DOFs (the 5th remap increment, the untargeted radial knots), it is not trading off against the targets. Saves fitted parameters to `chibi_field_params.json` and a `loss_curve.png`.

### `scripts/chibi_make_assets.py` — extended to consume fitted params

Add a `--field_params PATH` argument. When given, the script builds a `ChibiField` from the fitted JSON and uses it for both vertex deformation and basis rescaling (via `local_jacobian`), instead of the hardcoded `T/SY/SR_KNOTS` + `diag()` path. The legacy hand-knob path stays as the default when `--field_params` is absent (escape hatch / regression fixture #55). Output is the **unchanged 4-asset bundle** — deformed 5023 template, deformed 20018 baked obj, rescaled `(52,5023,3)` arkit_bs, `chibi_scale_ratio.npy` — so the LAM env-var hooks and Stage 2 need no changes.

## Data Flow

```
template.obj + flame_arkit_bs.npy + FLAME_masks.pkl + landmark_embedding.npy
        │
        ▼  chibi/fit.py  (Adam over ChibiField params; landmark+smooth+reg loss)
chibi_field_params.json
        │
        ▼  chibi_make_assets.py --field_params …  (applies ChibiField + Jacobian)
deformed template.obj │ rescaled arkit_bs.npy │ deformed baked.obj │ chibi_scale_ratio.npy
        │
        ▼  LAM env-var hooks  →  render
        │
        ▼  Stage 2: diff-leak spike (render-space iris-occlusion fix) — already planned
```

## Error Handling

- Monotone remap is structural (softplus increments) — no runtime monotonicity failure possible.
- Region masks / landmark embedding: fail fast with a clear error if a file is missing or a region is empty.
- Degenerate feature extent (zero bbox) → guard the division, raise rather than emit NaN.
- Landmark embedding is on the **5023** template; the baked mesh is **20018**. The fit runs entirely in 5023 space; `chibi_make_assets.py` already lifts the field to 20018 by shared `u`-fraction semantics — keep that, do not re-fit in 20018.

## Testing (TDD)

- `ChibiField` at identity params is the identity map (verts unchanged, Jacobian = I).
- The vertical remap is monotone for arbitrary random params.
- `local_jacobian` matches a finite-difference Jacobian within tolerance.
- `landmark_lines` on the undeformed template reproduces the known realistic `u` values (≈0.33/0.42/0.67/0.80/1.0) within tolerance.
- `fit_chibi_field` on the real template drives `L_landmark` below a threshold and lands eye/mouth `u` within ±0.02 of the quarter-grid.
- A region transform with `s_eye>1` enlarges the eye bbox and leaves non-eye verts within the falloff tolerance.
- `chibi_make_assets.py --field_params` emits all 4 assets with correct shapes; the legacy path is byte-stable (regression fixture #55).

## Open Questions

- **Block vs sphere skull.** The radial scale `g(u)` shapes skull *fullness* but not silhouette *squareness*. The artist wants slightly more block-like; the research doc flags this as a tunable. This design gives skull fullness via `g(u)` and defers an explicit silhouette-squareness target — revisit only if the fitted result reads too round against the reference.
- **Does better geometry shrink the Stage-2 leak?** The iris-through-lid leak is partly caused by the eyeball being pushed anterior of the lid in the current hand-tuned field. A fitted field with an explicit eyeball/lid relationship may reduce the leak before Stage 2 even runs. Not assumed here; observe after the first fitted render.
- **Centroid-detach approximation at high region scale.** `local_jacobian` holds region centroids fixed, dropping an O(1/M) coupling term. For the eye region (M≈1800 verts) this is well under 0.1%. If a future definition change pushes region scales much higher (the open *anime-style oversized eyes* knob), re-check that the fixed-centroid Jacobian still matches a finite-difference of the full `forward`.
