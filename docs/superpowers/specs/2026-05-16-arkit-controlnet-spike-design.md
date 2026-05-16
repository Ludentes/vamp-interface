# ARKit-ControlNet zero-train feasibility spike — design

**Date:** 2026-05-16
**Thread:** `_topics/arkit-controlnet.md`
**Background:** `docs/research/2026-05-16-arkit-controlnet-infiniteyou.md`

## Goal

Prove, with **zero model training**, the core function:

> `f(identity_image, ARKit_blendshape_vector) → image`
> — the output preserves the identity and wears the requested expression.

If the spike succeeds, the full ARKit-InfuseNet train (Path 2) is justified.
If it fails, we learn why before committing GPU-months.

## Success criteria

A grid of 5 identities (spanning the `reverse_index` FairFace demographic
range) × the expression set below, rendered through the pipeline, evaluated
against `reverse_index` baselines:

- **Identity holds.** `cos(ArcFace(output), ArcFace(identity_image))` stays
  above the within-corpus identity floor — target ≥ 0.65 (InfiniteYou's own
  ID Loss 0.209 ≈ cos 0.79; ≥ 0.65 is a permissive spike gate).
- **Expression follows.** Re-measured blendshapes on the output correlate with
  the intended vector — target Spearman ρ ≥ 0.6 across the swept channels,
  *and* a human eyeball pass: jawOpen / eyeBlink / smile are visibly present.
- **Continuity (soft).** Monotone expressions (e.g. jawOpen 0 → 1 in steps)
  produce monotone output — no discrete jumps.

A spike passes on a clear yes; a partial result (identity holds, expression
weak) still informs Path 2.

## Approach

InfU (frozen, official weights) carries identity from a reference photo. A
**stock FLUX ControlNet** carries a FLAME-mesh render of the ARKit expression.
The two run stacked — InfiniteYou Fig 6 b/c demonstrates exactly this with
off-the-shelf Depth/Pose ControlNets, no retraining.

**Rejected alternatives** (recorded for the audit trail):
- Feeding the render into InfU's *native* 5-keypoint control slot — InfU's
  control encoder was trained on sparse keypoints; a dense render is OOD.
  Kept only as a negative-control condition.
- A face-mesh / dense-landmark ControlNet — reads expression natively, but
  FLUX-native face-mesh ControlNets are scarce. Fallback if depth/normal is
  too coarse to resolve expression.

**Render modality:** surface normals primary (resolves mouth crease, lid fold
better than depth), depth as a cheap second condition to A/B. Use
`Shakker-Labs/FLUX.1-dev-ControlNet-Union` if it exposes normal+depth modes;
else a dedicated FLUX depth ControlNet. Confirming the exact stock ControlNet
is the spike's first task.

## Components

Four units, each independently testable.

**FLAME-ARKit renderer** — `render(bs_vector, pose) → control_image`.
Applies the `flame_arkit_bs.npy` (52, 5023, 3) basis to the FLAME template to
get per-vertex offsets, adds them, rasterises with nvdiffrast (already in the
repo from the chibi texture-bake work). Outputs a surface-normal (and/or depth)
image at FLUX resolution, framed to FFHQ-like head crop. Depends on: FLAME
assets, nvdiffrast. Standalone — testable by rendering known expressions and
eyeballing.

**ComfyUI workflow** — `workflow(identity_image, control_image, prompt) → output`.
A JSON graph: FLUX.1-dev + InfU (identity reference) + stock ControlNet
(control image). Saved under `comfyui/workflows/`. Depends on: ComfyUI, InfU
custom nodes, FLUX ControlNet weights. Testable by hand in the ComfyUI UI.

**Eval harness** — `eval(output, identity_image, intended_bs) → {arcface_cos, bs_rho, bs_l1}`.
Reuses the `reverse_index` extractor stack (ArcFace encoder, MediaPipe
blendshape reader). Standalone — testable on known image pairs.

**Spike runner** — selects `(identity_image, target_bs)` cases from
`reverse_index`, drives renderer → workflow → eval, writes a result grid PNG
and a metrics table. Resumable (skip-if-exists, per project rule). Depends on
the three units above.

## Data flow

```
reverse_index.parquet
   │  select N identities (FFHQ photos) × M ARKit expressions
   ▼
FLAME-ARKit renderer ──► control_image (normal / depth)
   │
   ├──────────────► ComfyUI workflow ◄────── identity_image (FFHQ photo)
   │                      │
   │                      ▼
   │                  output_image
   ▼                      │
intended_bs ──────► Eval harness ◄──────────┘
                          │
                          ▼
              {arcface_cos, bs_rho, bs_l1} + result grid
```

Expression sources: a small fixed set — neutral, jawOpen,
eyeBlink, smile (mouthSmile), browUp — plus one monotone ramp (jawOpen 0→1 in
4 steps) for the continuity check. Coefficients taken from canonical ARKit
unit poses, not from `reverse_index` measurements, so the intended vector is
exact.

## Risks

- **Domain gap.** A bare FLAME-head normal render may not look enough like the
  normal maps the stock ControlNet trained on (full scenes, real photos).
  Mitigation: try both normal and depth; if both fail, fall to the face-mesh
  ControlNet fallback before declaring the spike failed.
- **VRAM.** FLUX.1-dev + InfuseNet + a ControlNet branch on a 32 GB 5090 is
  tight. Run fp8 with sequential offload; accept slow inference — this is a
  spike, not a product.
- **Pose/framing mismatch.** The FLAME render crop must match how InfU frames
  the identity. Mitigation: render at a fixed frontal pose, FFHQ-style crop;
  defer arbitrary head pose to Path 2.
- **Identity-vs-control fight.** The ControlNet may pull the face off the
  ArcFace identity. Mitigation: sweep ControlNet conditioning scale; record
  the identity/expression trade-off curve rather than a single point.

## Out of scope (YAGNI)

- Any training or fine-tuning — this is the zero-train spike by definition.
- Arbitrary head pose, lighting control, temporal/video.
- The auxiliary blendshape critic loss — that belongs to Path 2 only.
- Wiring into the vamp-interface product or the animation-teacher pipeline.

## Verdict

The spike ends with a short dated doc in `docs/research/` recording the
metrics grid, the result collage, the chosen render modality + ControlNet, and
a go / no-go on Path 2. Update `_topics/arkit-controlnet.md` in the same commit.
