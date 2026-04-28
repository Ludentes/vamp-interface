# arc-distill — ArcFace identity in Flux latent space

**Status:** Layer 1+2 validation passed; Layer 3 (slider A/B) specced, not implemented.

The first of three "consume teacher T as a loss without rendering" heads
(ArcFace, MediaPipe, SigLIP).

## Current belief (2026-04-30)

A frozen-IResNet50 backbone with a small trainable stem (and IResNet50
layer-1) maps Flux VAE latents `(16, 64, 64)` to 512-d ArcFace-compatible
identity embeddings at val cos = 0.881 / median 0.939 / 87.6% above 0.9.
That is **plausibly sufficient for use as a slider/LoRA training loss**
(Use A) but **not validated for inference-time classifier guidance** (Use B,
where 0.881 fidelity is more concerning under operating conditions).

The shipped checkpoint is `latent_a2_full_native_shallow` at
[models/arc_distill/](../../../models/arc_distill/). Full validation report
in `validation_report.json` next to the weights.

## Load-bearing dated docs

- [2026-04-30-arcface-frozen-adapter.md](../2026-04-30-arcface-frozen-adapter.md) —
  primary writeup. Method, results table across 6+ stem variants, run-log,
  validation results, possible-uses fidelity-demand discussion.
- [2026-04-30-arc-distill-lessons.md](../2026-04-30-arc-distill-lessons.md) —
  what to carry into the next "distill teacher T" head. Pre-compute hygiene,
  geometry beats capacity, hard negatives, k-fold ridge.
- [2026-04-30-layer3-slider-ab-spec.md](../2026-04-30-layer3-slider-ab-spec.md) —
  the real Use-A test: smile-axis slider trained with vs without the student
  as identity-preservation loss; ~half-day to implement and run.

## Adjacent threads

- [2026-04-30-mediapipe-distill-plan.md](../2026-04-30-mediapipe-distill-plan.md) —
  next head, expression-preservation loss via 52-d ARKit blendshapes.
  Inherits arc_distill's lessons. Plan only, not started.
- [2026-04-27-arc-latent-distillation-plan.md](../2026-04-27-arc-latent-distillation-plan.md) —
  the canonical plan that spawned this thread.

## Open questions

- Use B (inference-time guidance) — fidelity bar is open. 0.881 may be too low
  under operating conditions; needs measurement before being trusted.
- The long left tail (5% of val rows ≪ 0): localization failures on faces
  with hats / sunglasses / dense beards. Ignored for now (averaging masks
  it in batched losses) but a real failure mode for guidance use.
- Partially-noised latents at `t > 0` of the diffusion schedule: behaviour
  untested; the student was trained on clean VAE-encoded latents only.
