---
status: live
topic: arkit-controlnet
---

# CFM expression-ControlNet training run — design

**Date:** 2026-05-19
**Goal:** Train InfiniteYou's InfuseNet control channel to follow a FLAME
expression render, via conditional flow matching against real
`(photo, FLAME-render)` pairs — base FLUX DiT and the ArcFace identity path
frozen, only a LoRA on InfuseNet (plus its control-image input stem) trains.

## Context

All three zero-train expression routes are falsified
(`_topics/arkit-controlnet.md`): FluxSpace compose, the MediaPipe-mesh
InfuseNet slot, and a stacked depth ControlNet. Expression must be trained.
The render-cache that supplies the FLAME side of the training pairs is built
and verified (`2026-05-18-cfm-render-cache-design.md`): `flame_render.render()`
plus `output/flame_pose_cache/pose_cache.parquet` (69,928 detected rows) and
`output/ffhq_index/ffhq_sha_index.parquet`. This spec covers the training run
itself — the model wiring, the dataloader, the loss, and evaluation.

The design analysis (`2026-05-16-arkit-controlnet-infiniteyou.md`) is the
load-bearing prior: InfuseNet is a generalization of ControlNet with a frozen
identity path and an *already-existing* spatial-control slot; the recommended
loss is CFM alone, with an auxiliary blendshape critic only as a fallback.

## What trains, what is frozen

- **Frozen:** the FLUX.1-dev base DiT; the ArcFace face encoder and the
  IP-Adapter-style identity projection (`image_proj_model.bin`, 8 identity
  tokens); the VAE; the text encoders.
- **Trains:** a LoRA on InfuseNet's DiT-copy blocks, **and** InfuseNet's
  control-image input stem fully (the conv that ingests the control image — it
  has only ever seen 5-keypoint images and must learn the FLAME-render
  modality from scratch, which a low-rank delta cannot supply).

Rationale: a single 32 GB RTX 5090. A full InfuseNet fine-tune's optimizer
state strains 32 GB and resumes poorly; LoRA + a small fully-trained stem fits,
is resumable, and matches the repo's existing slider / distill patterns.

## Framework risk and the gating spike

The InfuseNet weights on disk (`infusenet_sim_fp8e4m3fn.safetensors`) are a
ComfyUI-converted fp8 inference weight, and the ComfyUI `InfuseNet` class
subclasses comfy's `ControlNet` — inference-only, no gradients. Training needs
diffusers FLUX plus a **bf16** InfuseNet. ByteDance publishes the bf16
InfiniteYou weights and inference model classes on HuggingFace but no InfuseNet
*training* code; the CFM loop is reimplemented here (standard FLUX
flow-matching — the repo already has it in `train_flux_image_slider.py`).

**Task 0 is a gating feasibility spike.** Download the bf16 InfiniteYou
weights; instantiate diffusers FLUX + InfuseNet; attach the LoRA; run one
forward + backward CFM step on a single real `(photo, render)` pair; assert the
loss is finite and the LoRA / stem gradients are non-zero. If Task 0 fails the
plan is replanned before any data work — it de-risks the whole run for the cost
of one training step.

## Architecture

A `src/arkit_controlnet/cfm/` package, four focused units plus a one-time
precompute.

### `cfm/model.py`

Loads frozen diffusers FLUX + bf16 InfuseNet, attaches the LoRA to InfuseNet's
DiT-copy blocks, marks the control-image stem trainable, freezes everything
else. Exposes `build_model() -> CfmModel` and a `CfmModel.velocity(z_t, t,
text_embeds, id_tokens, control_latent) -> v_pred` forward that runs base FLUX
with InfuseNet residuals added and returns the predicted velocity. Gradient
checkpointing on InfuseNet blocks (VRAM headroom).

### `cfm/precompute.py`

One-time, resumable, atomic writes (the slider-trainer cache pattern). For each
of the 69,928 detected FFHQ rows: VAE-encode the photo → latent; run the frozen
InfiniteYou projection → 8 identity tokens. Encode the single fixed training
prompt (T5 + CLIP) once. Cache to `output/cfm_precompute/` keyed by
`image_sha256`. The FLAME control render is **not** precomputed — it is cheap
and the render modality is still being A/B'd (per the render-cache design).

### `cfm/dataset.py`

`CfmPairDataset`: reads the held-in split of `reverse_index.parquet`, joins to
`pose_cache.parquet` and `ffhq_sha_index.parquet` on `image_sha256`, drops
`pose_detected == False`. Per item: load the cached photo latent + id tokens;
build the 52-d ARKit vector from the `bs_*` columns; `deform()` + `render()`
the FLAME control image live at the row's cached pose; VAE-encode the control
image to `control_latent`. Returns `(photo_latent, control_latent, id_tokens,
text_embeds)`. Render `modality` is a constructor argument.

### `cfm/train.py`

The CFM loop. Per step: sample a logit-normal timestep, apply the FLUX sigma
shift (3.0), form `z_t = (1-σ)·z_0 + σ·ε`, predict velocity, loss
`L_CFM = ‖v_pred − (ε − z_0)‖²`. AdamW 8-bit (bitsandbytes), bf16, constant LR
with short warmup, gradient accumulation to a useful effective batch. Resumable
`latest.pt` (atomic write), CSV step log, periodic held-out eval + a sample
collage to `exp_output/cfm_train/`.

### `cfm/eval.py`

Held-out evaluation (~2k FFHQ rows split off by `image_sha256` hash, fixed):

- **Identity retention** — ArcFace cos(generated, reference); target ≥ the
  zero-train InfuseNet baseline (0.60–0.83 measured in the spike).
- **Control fidelity** — re-run MediaPipe on the generated face, cos(tracked
  blendshapes, intended blendshapes); target a clear margin over a
  neutral-control baseline (the spikes used ≥0.05; the bar here is *strong*
  following, not bare detectability).
- **Continuous modulation** — interpolate one blendshape across its range,
  confirm the expression changes monotonically (the failure that killed
  FluxSpace: s0.5 ≈ s1.0).
- Eyeball collage of generated faces overlaid against held-out photos.

## Two early gates before the long run

- **Render-fit calibration.** The render-cache verification flagged the FLAME
  mesh sitting slightly high / oversized — `render()` fits all 5023 verts
  (whole skull) into a face-only MediaPipe bbox. Fix the bbox→render fit (a
  face-vertex-subset extent, or a calibrated scale/offset) before training; a
  systematically misaligned control image is bad supervision.
- **Render modality A/B.** Normals vs depth vs flat-shaded is an open question.
  Run short training probes (a few hundred steps each) on the three modalities,
  pick the one whose CFM loss drops fastest and whose collage reads cleanest,
  then commit it for the full run.

## Data flow

```
reverse_index (held-in) ─sha─┬─▶ pose_cache  ──▶ rotation + bbox ┐
                             ├─▶ ffhq_sha_index ▶ photo bytes    │
                             └─▶ bs_* ▶ arkit52 ─deform─▶ verts ─┼▶ render() ▶ control img ▶ VAE ▶ control_latent
cfm_precompute ──▶ photo_latent (z_0), id_tokens, text_embeds ────┘
                                                                   ▼
                              CFM loss  ‖v_pred − (ε − z_0)‖²  ◀── CfmModel.velocity()
```

## Error handling

- `pose_detected == False` rows excluded at dataset construction.
- Non-finite ARKit vector or degenerate render `bbox` → `render()`/`deform()`
  already raise `ValueError`; the dataset skips and logs the row rather than
  aborting the epoch.
- CUDA OOM → gradient checkpointing is on by default; the knobs are batch size
  and accumulation, documented in the plan.
- Training is resumable from `latest.pt`; precompute is resumable per row.
- Task 0 failure → escalate and replan (do not proceed to data work).

## Testing

- `model.py` — `build_model()` returns with exactly the LoRA + control stem
  parameters `requires_grad=True` and all else frozen; `velocity()` output
  shape matches `z_t`.
- `precompute.py` — round-trip a known row, assert latent shape and that a
  resumed run skips completed rows.
- `dataset.py` — one item has the right tuple shapes; a `pose_detected==False`
  sha is absent; the `bs_*`→arkit52 column order matches `flame_render`'s
  `BASIS_CHANNEL_NAMES`.
- `train.py` — one step on a 2-row fixture drives the loss finite and downward
  over a handful of steps; a resumed run continues from the saved step.
- The Task 0 spike is itself the end-to-end smoke test.

## Out of scope

The auxiliary blendshape-critic loss (the documented fallback if `eval.py`
shows weak control-following — its own follow-up, with the slider thread's
`t_max≈0.5` gating machinery); synthetic or pseudo-SPMS data (SPSS-only first
by decision — revisit only if eval demands it); distillation into a fast
student; live animation. This spec ends at: a trained InfuseNet-LoRA that
follows a FLAME expression render, with held-out identity and control-fidelity
numbers.
