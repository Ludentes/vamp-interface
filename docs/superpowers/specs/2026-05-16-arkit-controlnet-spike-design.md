# ARKit-ControlNet zero-train feasibility spike — design

**Date:** 2026-05-16
**Thread:** `_topics/arkit-controlnet.md`
**Background:** `docs/research/2026-05-16-arkit-controlnet-infiniteyou.md`

## Goal

Prove, with **zero model training**, the core function:

> `f(identity_image, expression) → image`
> — the output preserves the identity and wears the requested expression.

For the spike, `expression` is a FluxSpace edit along one of a small set of
named axes (smile, pucker, surprise, …) at a chosen edit scale — not yet an
arbitrary 52-d ARKit vector. Proving the mechanism on these axes justifies the
full ARKit-InfuseNet train (Path 2); a clean failure tells us why before any
GPU-months are committed.

## Success criteria

A grid of 5 identities (spanning the `reverse_index` FairFace demographic
range) × the expression set below, rendered through the pipeline, evaluated
against `reverse_index` baselines:

- **Identity holds.** `cos(ArcFace(output), ArcFace(identity_image))` stays
  above the within-corpus identity floor — target ≥ 0.65 (InfiniteYou's own
  ID Loss 0.209 ≈ cos 0.79; ≥ 0.65 is a permissive spike gate).
- **Expression follows.** The ARKit channels the axis should move (re-measured
  by the `mediapipe_distill` blendshape reader) shift in the correct direction
  as edit scale rises — `bs_delta` monotone in scale — *and* a human eyeball
  pass: the smile / pucker / surprise is visibly present.
- **Continuity (soft).** A monotone edit-scale ramp produces monotone output —
  no discrete jumps or scale-collapse inside the swept band.

A spike passes on a clear yes; a partial result (identity holds, expression
weak) still informs Path 2.

## Approach

Both halves of the function are driven by tools the project already has, with
**zero training and no new renderer**:

- **Identity** — InfU (InfiniteYou, official weights + ComfyUI nodes), frozen.
  Residual injection of an ArcFace embedding from a reference photo.
- **Expression** — **FluxSpace** edits, training-free, via the project's
  existing `demographic_pc_fluxspace` custom nodes (`FluxSpaceEdit`,
  `FluxSpaceEditPair`) at `~/w/ComfyUI/custom_nodes/`. FluxSpace edits Flux's
  joint-attention internally along a prompt-pair direction.

FluxSpace is the right relative: prior experiments
(`project_fluxspace_smile_axis`, `project_fluxspace_crossdemo`) showed the
pair-averaged FluxSpace **smile** axis moves expression cleanly and generalises
across six demographics — exactly where the slider LoRA semi-failed. Smile is
the one expression axis with verified prompt pairs and per-base scale windows;
other expression axes (pucker, surprise) are constructed by analogy on the same
prompt template but are uncharacterised.

The real question the spike answers: **do residual-injection identity (InfU)
and attention-editing expression (FluxSpace) compose, or fight?** InfU adds
InfuseNet residuals; FluxSpace edits the base DiT's joint attention. Both act
on the same frozen FLUX.1-dev. Whether they coexist — expression moves, identity
holds — is unknown and cheap to test.

**Rejected alternative** (audit trail): a FLAME-mesh render fed to a stock FLUX
ControlNet. Sound, but pulls in LAM-lineage rendering machinery (FLAME assets,
nvdiffrast) that is a weaker relative to this thread than FluxSpace, and means
building a renderer before testing anything. Deferred — reconsider for Path 2
if FluxSpace's per-axis coverage proves too narrow.

**Coverage note:** FluxSpace gives a handful of named expression axes, not an
arbitrary 52-d ARKit vector. For a feasibility spike that is sufficient — prove
the *mechanism* on smile / jawOpen-like axes. Full ARKit-52 coverage is a
Path-2 concern.

## Components

Three units, each independently testable.

**FluxSpace expression axis set** — the prompt pairs and pair-averaging config
for the spike's expression axes: **smile** as the verified primary, **pucker**
and **surprise** by analogy on the same template. A small config file under
`src/`, plus the per-axis edit-scale band. Depends on: the characterised smile
direction from the `project_fluxspace_*` experiments. Testable by running each
axis on a plain FLUX render (no InfU) and eyeballing.

**ComfyUI workflow** — `workflow(identity_image, axis, scale, prompt) → output`.
A JSON graph: FLUX.1-dev + InfU identity node + `FluxSpaceEditPair` node.
Saved under `comfyui/workflows/`. Depends on: ComfyUI, InfU custom nodes, the
`demographic_pc_fluxspace` nodes. Testable by hand in the ComfyUI UI.

**Eval harness** — `eval(output, identity_image, axis) → {arcface_cos, bs_delta}`.
Reuses the `reverse_index` extractor stack — ArcFace encoder for identity drift,
the `mediapipe_distill` blendshape reader for expression. `bs_delta` is the
change in the ARKit channels the axis is expected to move (e.g. smile axis →
`mouthSmileLeft/Right`). Standalone — testable on known image pairs.

**Spike runner** — selects 5 identity photos from `reverse_index`, sweeps each
axis × edit scale, drives workflow → eval, writes a result grid PNG and a
metrics table. Resumable (skip-if-exists, per project rule). Depends on the
units above.

## Data flow

```
reverse_index.parquet
   │  select 5 identity photos (FFHQ, demographic spread)
   ▼
identity_image ──► ComfyUI workflow ◄── FluxSpace axis + edit scale
                         │              (smile / pucker / surprise / …)
                         ▼
                    output_image
                         │
   ┌─────────────────────┼─────────────────────┐
   ▼                     ▼                     ▼
ArcFace(output)    mediapipe bs(output)   result grid PNG
 vs identity        vs axis's target
   │                  channels
   ▼                     ▼
arcface_cos          bs_delta  ──►  metrics table
```

Expression cases: smile (verified axis) as primary, plus one monotone edit-
scale ramp on smile (4 steps) for the continuity check; pucker and surprise as
an extended pass if smile succeeds.

## Risks

- **Composition failure.** InfU's identity residuals and FluxSpace's attention
  edit may interfere — the edit washes out, or identity collapses. This is the
  *point* of the spike; a clean failure here is a valid, informative result.
- **FluxSpace scale collapse.** FluxSpace edits break down past an axis-
  dependent scale (`project_fluxspace_collapse_prediction`). Mitigation: sweep
  edit scale; stay in the validated 0.5–1.0 band; record the trade-off curve.
- **VRAM.** FLUX.1-dev + InfuseNet + FluxSpace attention caching on a 32 GB
  5090 is tight. Run fp8 with sequential offload; accept slow inference — this
  is a spike, not a product.
- **Axis ≠ ARKit channel.** A FluxSpace "smile" axis is a prompt-pair
  direction, not a calibrated `mouthSmile` coefficient. `bs_delta` measures
  *direction of change*, not absolute fidelity. Accept for the spike.

## Out of scope (YAGNI)

- Any training or fine-tuning — this is the zero-train spike by definition.
- A FLAME renderer / ControlNet path — deferred to a Path-2 decision.
- Full arbitrary 52-d ARKit control — FluxSpace covers a handful of axes.
- Arbitrary head pose, lighting, temporal/video.
- The auxiliary blendshape critic loss — Path 2 only.
- Wiring into the vamp-interface product or the animation-teacher pipeline.

## Verdict

The spike ends with a short dated doc in `docs/research/` recording the metrics
grid, the result collage, the InfU+FluxSpace composition behaviour, and a
go / no-go on Path 2. Update `_topics/arkit-controlnet.md` in the same commit.
