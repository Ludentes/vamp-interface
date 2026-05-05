---
status: live
topic: personalive-acceleration
---

# What PersonaLive Stage-1 fine-tuning unlocks, and what still needs Stage 2

Companion to the [Moore Stage-1 feasibility probe](2026-05-04-moore-stage1-feasibility-probe.md)
and [reproduce/handoff](2026-05-05-moore-stage1-probe-handoff.md). Now
that Stage 1 fits on a single 5090 with PersonaLive's actual weights
loaded, this is the strategic readout: what we can build today versus
what is gated on the still-unprobed Stage 2.

## What Stage 1 actually trains

Per the PersonaLive paper, Stage 1 is the **image-level appearance**
path:

- **ReferenceNet** (UNet2D, ~860M) — extracts identity / style features
  from a single reference image
- **Denoising UNet3D, motion modules off** (~860M) — denoises target
  frames conditioned on pose + reference features
- **PoseGuider** (~0.7M) — encodes a pose-style image (DWPose-rendered
  by default) for cross-attention conditioning

Stage 1 does *not* exercise: `temporal_module.pth`, MotEncoder,
MotionExtractor, sliding training, motion-interp init (MII),
historical-keyframe mechanism (HKM), or the few-step distillation.

## What this opens up for us

Six concrete directions, in rough leverage order. All are static-image
operations — none require temporal, motion, or video work.

### 1. Pose-conditioning swap (highest-leverage)

PoseGuider currently consumes DWPose 2D keypoints. It is a small (~0.7M
param) conditioning encoder that we can retrain to consume *anything
with the same image shape*. Concrete swap targets:

- **ARKit blendshapes** rendered as a face image (we already have the
  rendering for the FluxSpace / NMF work)
- **Our NMF atom basis** — k=8 atoms applied to a neutral face,
  rasterized
- **Slider axes** — the smile / squint / age / etc. axes from the
  slider operational handbook, projected onto a neutral mesh

Why this matters: it hooks PersonaLive into the entire
blendshape-NMF / slider authoring stack you already have. Single-image
conditioning, no temporal needed, ~few hours of training to converge
because PoseGuider is tiny.

### 2. Identity locking

Fine-tune ReferenceNet on a small portrait set (say, 20–50 images of a
single person). Output: PersonaLive that preserves *that* identity
better in static portraits. Static-only — no driving video required.
Useful for personalized portrait generation where identity drift is
the failure mode.

### 3. Demographic / style rebalance

PersonaLive's pretraining has the same skews ours always do — narrow
slice of demographics, narrow lighting, narrow camera framing.
Fine-tune Stage 1 on a curated, balanced face corpus to flatten that.
Connects directly to the au_library Phase-3 corpus-balance work
(which already produces the kind of balanced corpus you'd want).

### 4. Concept-Sliders on PersonaLive

Once Stage-1 fine-tuning is real, the natural next thing is to train
Concept-Sliders-style LoRAs on PersonaLive's UNet for axes we already
have: smile, age, glasses, ethnicity. PersonaLive is a face-specialized
model — a slider trained against it should be more stable than the
same axis trained against generic SDXL or Flux. This is a way to
forward-port the slider thread to a model that already does faces
well.

### 5. Static-portrait quality bump for vamp-interface

For the static portrait grid in this repo, Stage 1 alone produces
images (just don't ask for animation). Could replace the current
SDXL portrait path if the quality delta justifies the inference
slowdown. Quick A/B against the existing portrait set is a
weekend's work.

### 6. Custom PoseGuider for the sus → face mapping

End-to-end vamp-interface use: `sus_factor` → blendshape decomposition
→ pose-image (via our renderer) → fine-tuned PoseGuider →
PersonaLive denoising → photoreal portrait. Closes the loop between
the fraud-signal embedding and the rendered face. Requires the work
in (1) plus a thin glue script.

## What is gated on Stage 2

Anything that moves. Stage 2 adds, in the paper's order: temporal
attention (`temporal_module.pth`, ~46M params), the Sliding Training
Strategy, Motion-Interpolated Initialization, the Historical Keyframe
Mechanism (motion bank, history bank, motion threshold τ=17), and
the **few-step distillation** that takes the model from many-step
diffusion down to 4-step DDIM via adversarial loss.

Items that strictly need Stage 2:

- **Animation / video output** — frame-to-frame temporal coherence is
  the temporal_module's whole job
- **Face reenactment** (driving a still from a driving video) — needs
  `MotionExtractor` (LivePortrait 3D keypoints) + `MotEncoder` (FAN
  face crop) plumbed in alongside the temporal module
- **Real-time webcam pipeline** — depends on the distilled 4-step
  student produced in Stage 2
- **Long-video generation** without flicker / identity drift across
  chunks — needs HKM + sliding training to stitch chunks together
- **Live VTuber-style portrait** (the strategic-default thread per
  `project_personalive_default_path` memory) — fundamentally Stage 2

## What we still don't know about Stage 2

The Stage 1 probe doesn't say anything about these, and they are the
gates on whether a Stage 2 reconstruction is feasible on a single
5090:

- **Stage 2 memory budget.** Temporal attention's activation memory
  scales with `temporal_window_size`. Even at batch=1 with 8-bit
  Adam, this could push past 32 GB. Needs its own probe.
- **MotEncoder / MotionExtractor cost during training.** Modest extra
  parameters but each adds an upstream forward/backward over driving
  frames per training step. Could push us from "fits" to "doesn't
  fit." Untested.
- **IPS at meaningful dataset size.** Our 10-step result was 1.05 s/it
  on a single synthetic 30-frame video read from RAM cache. Real
  training reads ~10k clips off disk, runs DWPose / MediaPipe, etc.
  Steady-state could be 2–5× slower per step.
- **Distillation memory profile.** The Stage 2 adversarial-distillation
  phase typically pairs teacher + student forward passes. Memory
  blowup is plausible.

## Pragmatic order if we push further

1. **Cheap, in-domain win** — fine-tune Stage 1 PoseGuider on our
   blendshape → pose-image pipeline (item 1 above). Highest leverage
   because it links PersonaLive into work that's already done.
   Maybe 1–2 days end-to-end.
2. **Stage 2 memory probe.** Same shape as the Stage 1 probe but on
   `train_stage_2.py` with `temporal_module.pth` loaded from
   PersonaLive. Tells us whether a real Stage-2 reconstruction is on
   the table or needs offload (bnb 8-bit + DeepSpeed Zero-2 CPU
   offload). Independent of (1), can run in a parallel session.
3. **MotEncoder / MotionExtractor port.** PersonaLive ships their
   forward code under `src/`. Porting the modules into Moore's tree
   is the bridge from "AnimateAnyone clone" to "real PersonaLive
   Stage-1 reconstruction." Required before we can faithfully
   fine-tune on driving-video signal.

Items (1) and (2) are independent and don't block each other. Item (3)
is a prerequisite for any meaningful PersonaLive training that uses
the maintainers' own motion path; without it we are training a
portrait diffuser that happens to share PersonaLive's appearance
weights.

## Risks / things to watch

- **Vendor patches drift.** Our Moore patches are pinned via the
  vendor-patch.diff in this repo. If Moore pushes upstream changes to
  the trainer or the model files, the diff will likely conflict.
  Re-apply manually; the five semantic changes are documented in the
  handoff.
- **PersonaLive issue #17.** If their training code drops, we should
  switch to it for any Stage-2 work. Until then, Moore is the closest
  faithful proxy.
- **Diffusers 0.24 pin.** Our env is locked to diffusers 0.24 because
  newer drift cascades through Moore's source. If we want to integrate
  with anything that needs diffusers ≥ 0.30 (e.g., recent
  text-to-image LoRAs, newer ControlNet implementations), we'll need
  a separate venv or a heavier source-side compat patch.
