---
status: live
topic: arkit-controlnet
---

# ARKit depth-ControlNet stacked spike — verdict

**Date:** 2026-05-18
**Verdict:** FALSIFIED. Approach C — a stock FLUX Depth ControlNet stacked
alongside identity-only InfuseNet, driven by a landmark-rasterized depth map,
zero training — does not give an expression channel. This was the last cheap
zero-train fallback. Expression must be trained (CFM).

## What was tested

The falsified landmark-control verdict (`2026-05-18-arkit-landmark-control-spike-verdict.md`)
left one untried shortcut: instead of feeding mesh geometry into InfuseNet's
own 5-keypoint-trained spatial slot, give InfuseNet *only* identity (black
control image, ArcFace path) and route expression through a **separate stock
FLUX Depth ControlNet** — a module actually trained on depth conditioning.
The depth map is a flat-shaded, z-sorted painter's-algorithm render of the
MediaPipe 478-vertex face tessellation of a high-coefficient FFHQ exemplar.

- Model: InstantX `FLUX.1-dev-Controlnet-Union`, `depth` mode via
  `SetUnionControlNetType`.
- Graph: `comfyui/workflows/arkit_depth_spike.json` — depth CN feeds
  conditioning into `InfuseNetApply` (identity, black image, strength 0.6).
- Sweep: 3 identities × {neutral, smile, pucker, surprise} × depth strength
  {0.5, 0.8}, seed 2026. 24/24 rendered.
- Runner: `src/arkit_controlnet/run_depth_spike.py`. Metrics: ArcFace cosine
  (identity), MediaPipe-blendshape cosine vs exemplar (expression).

## Result — both strength regimes fail, no usable band between

**str 0.8 — ControlNet dominates, paints the mesh artifact.** 11/12 outputs
have no detectable face at all (ArcFace *and* MediaPipe both fail). The output
is the literal low-poly faceted tessellation rendered as a 3D plastic mask —
brown skin-toned facets, triangle edges visible, the surprise mesh's mouth
hole rendered as an actual hole. The depth ControlNet was trained on smooth
MiDaS-style continuous depth of real scenes; a faceted triangle raster is
out-of-distribution, so at high strength it imprints the facets verbatim.

**str 0.5 — ControlNet too weak, expression does not transfer.** Outputs are
photoreal and identity is preserved (ArcFace 0.39–0.66), but the face is
**neutral regardless of the depth-map expression**. The surprise and neutral
outputs of the same identity are indistinguishable — no raised brows, no open
mouth, no widened eyes. Baseline-relative expression delta (expr_cos for the
axis minus expr_cos for the neutral control, same identity/strength):

| axis     | mean Δ  | min     | max    |
|----------|---------|---------|--------|
| smile    | −0.009  | −0.392  | +0.545 |
| pucker   | −0.183  | −0.467  | +0.018 |
| surprise | −0.328  | −0.598  | +0.033 |

If the depth map transferred its expression, the axis outputs would score
*higher* toward the exemplar than the neutral baseline → positive Δ. Instead
every axis mean is negative or zero. The lone positive outlier (one identity,
smile, +0.545) is contradicted by the collage and is measurement noise.

There is no strength between 0.5 and 0.8 that buys expression transfer while
keeping a photoreal face — the failure is not a tuning miss, it is the OOD
gap between a faceted landmark raster and real depth.

Artifacts: `exp_output/arkit_depth_spike/` — `metrics.parquet`, `collage.png`
(rows = identity; cols = id ref / smile 0.5 / smile 0.8 / surprise 0.5 /
surprise 0.8).

## Why it fails — same OOD root cause as the landmark spike

Both zero-train routes fail for one reason: a MediaPipe-derived control image
is out of distribution for any module trained on real conditioning signals.
InfuseNet's slot was trained on 5 sparse keypoints; the depth ControlNet was
trained on smooth scene depth. A dense faceted face raster matches neither. A
better render modality (surface normals, a smoother depth, sparse contours)
would only narrow the OOD gap, not close it — the conditioning module still
has no learned mapping from "synthetic face raster" to "this expression."

## Verdict

All three zero-train expression routes are now falsified: FluxSpace attention
compose, InfuseNet mesh slot (Path 1), and stacked depth ControlNet
(Approach C). The expression channel must be **trained** — conditional flow
matching on real `(photo, FLAME-render)` pairs, per the topic-index design.
The identity channel (InfuseNet / ArcFace) is solid throughout and is not in
question. No more cheese; proceed directly to the CFM run.
