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

## Operational handbook (READ FIRST on resume)

- **[2026-05-03-slider-operational-handbook.md](../2026-05-03-slider-operational-handbook.md)** —
  Complete recipe: solver taxonomy, dataset construction, critic training,
  loss specification, phased slider procedure with gates, iteration loop,
  falsified approaches. Synthesises everything from 2026-04-23 framework
  procedure through 2026-04-30 v1k working recipe. Pair with
  [2026-05-03-slider-thread-recipe.md](../2026-05-03-slider-thread-recipe.md)
  (TL;DR parking note).

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
- [2026-04-29-siglip-distill-design.md](../2026-04-29-siglip-distill-design.md) —
  third head: 1152-d SigLIP-2 SO400M image embedding. v1 design captured
  (v2c trunk, linear 1152-d head, 0.5·MSE + 0.5·(1−cos), iteration plan if
  poor). Pair file `output/siglip_distill/compact_siglip.pt` ready (26,108
  rows, 100% coverage). Reverse-index now carries
  `siglip_img_emb_fp16` (79,116 rows). Training not yet started.
- [2026-04-27-arc-latent-distillation-plan.md](../2026-04-27-arc-latent-distillation-plan.md) —
  the canonical plan that spawned this thread.
- [2026-04-30-noise-conditional-distill-design.md](../2026-04-30-noise-conditional-distill-design.md) —
  cross-head upgrade plan to make all three students (`arc_distill`,
  `mediapipe_distill`, `siglip_distill`) usable as inference-time
  classifier guidance (Use B). Two paths: cheap retrain on noisy latents
  (same arch, ~20 % more wall-time) vs noise-conditional `(z_t, t)` input
  via FiLM/AdaLN. Triggered 2026-04-29: siglip sg_c (Path 2) trained to
  epoch 10 — schedule curve nearly flat to t=0.75. arc Path 1 saturates at
  t=0.5 ≈ 0.525 (frozen-backbone ceiling); arc Path 2 (5-stage FiLM)
  running with modest gains.
- [2026-04-29-arcnet-informed-scheduler-design.md](../2026-04-29-arcnet-informed-scheduler-design.md) —
  Use B application. Once a noise-conditional student is reliable across
  the schedule, use it to *measure* where on the trajectory identity
  commits, then reallocate Flux step density (analogue of NVIDIA's
  Align Your Steps but with an ID-specific metric). Three designs:
  step redistribution, early termination, adaptive CFG. Gating experiment
  is the per-step ID emergence curve; gated on noise-conditional student
  reliability across t.

## Open questions

- Use B (inference-time guidance) — fidelity bar is open. 0.881 may be too low
  under operating conditions; needs measurement before being trusted.
- The long left tail (5% of val rows ≪ 0): localization failures on faces
  with hats / sunglasses / dense beards. Ignored for now (averaging masks
  it in batched losses) but a real failure mode for guidance use.
- Partially-noised latents at `t > 0` of the diffusion schedule: behaviour
  untested; the student was trained on clean VAE-encoded latents only.
