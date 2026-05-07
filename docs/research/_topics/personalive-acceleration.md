## PersonaLive real-time acceleration

**Status:** live, active. PersonaLive (CVPR 2026 portrait-animation
diffusion stack) ships at ~10 FPS on RTX 5090 (Blackwell sm_120) with
plain SDPA attention. Target: ≥25 FPS for real-time webcam pipeline.

Gate (per `2026-05-04-personalive-tensorrt-plan.md`): each probe ≥22
FPS ships, 15-22 means tune, <15 advances to next probe.

### Current beliefs

- **xformers / flash-attn source build for sm_120**: falsified
  (`2026-05-03-xformers-sm120-falsified.md`). Both projects' Blackwell
  PRs landed but not in a release torch could resolve against without
  ABI breakage; manual builds repeatedly OOM-killed nvcc or hit
  template instantiation errors. Stock SDPA is the floor.
- **Probe A — torch.compile**: works, +21% (12.10 FPS). `dynamic=True`
  is a trap on Blackwell — torch.compile + sympy spends ~10 minutes
  per diffusion step trying to specialize symbolic shapes. Drop the
  flag, specialize per-shape, finish in normal time.
- **Probe B — torch_tensorrt 2.11**: works, +77% (17.72 FPS). Three
  off-the-shelf failure modes (memory format, GuardOnDataDependentSymNode,
  pipeline._execution_device) — each a 3-line fix. Lands in 15-22 tune
  zone, not yet ship target. Engine cache enabled; rebuilds skipped on
  subsequent runs.
- **Probe C — bundled torch2trt.py + ONNX → polygraphy → TRT engine**:
  works, **+88% (18.87 FPS)**. Single 3.7 GB engine fuses denoising_unet
  + vae.decode + scheduler.step. Only +6.5% over Probe B — the
  "no PyTorch fallback" advantage netted less than expected; most
  gain was already captured by per-module ttrt compile. Build cost:
  ~10 min (ONNX export + polygraphy autotune over 16 dynamic-shape
  inputs). Detours: upstream's `auto_cast=True` flag on `export_onnx`
  forces fp32 BN inputs against fp16 weights → falsified, set to False.
  Lands in 15-22 tune zone, still below ≥22 ship gate.

### Verdict (2026-05-04)

Working estimate: this stack tops out at **20-21 FPS** on RTX 5090,
short of the ≥22 ship gate. Per-module ttrt already captured the
fallback-boundary win Probe C was supposed to add, so further gains
have to come from the model side (fewer steps, smaller VAE) — not
the runtime side. **No further runtime tuning is planned;** if we
need to ship ≥22, the path is architectural (see options below).

### Options to push further (not actively pursued)

Cheap (hours, runtime-only — pre-test ceiling ~20-21 FPS):

- TAESD swap for `vae.decode` (PersonaLive ships `vae_tiny_path`
  in offline config); ~10× faster decoder, slight quality loss.
- Probe B engine config tuning: `optimization_level=5`, narrower
  dynamic-shape range, `enable_autocast`, `decompose_attention=True`.
  Each lever ~2-5%, stacked ~10-15%, ~30s/iter with cache warm.
- TRT engine inspector to find slowest subgraph (likely temporal
  attention or VAE upsamples), apply layer-specific overrides.
- CUDA graphs around the per-frame call to drop launch overhead
  for the surrounding torch ops (~5-15%, low risk).

Medium (1-2 days, model-level — could clear 22 FPS):

- Drop denoising steps from 4 to 2 with proper schedule re-tune;
  near-doubling of throughput if quality holds. Risk: flicker /
  identity drift.
- Smaller temporal window (`temporal_window_size`,
  `temporal_adaptive_step`); trades quality for FPS.

Expensive (week+, architectural):

- Re-distill PersonaLive at 2-step target (Hyper-SD style on top
  of its existing temporal distillation).
- Re-check xformers / flash-attn / SageAttention 2 source build
  for sm_120 in a few weeks (currently falsified).
- Pivot to a warp-based path (LivePortrait / X-Nemo) — already
  an active thread, see `_topics/neural-deformation-control.md`.

### Open

- Visual artifact comparison across modes (videos exist, side-by-side
  pending).

### Training feasibility — GREEN at batch=1 with 8-bit Adam (2026-05-05)

PersonaLive training code is deferred indefinitely (issue #17, no
ETA; maintainer redirects to Moore-AnimateAnyone). Ran the Moore
Stage-1 feasibility probe on RTX 5090 (32 GB):

- **PersonaLive's actual `reference_unet.pth` + `denoising_unet.pth`
  + `pose_guider.pth` load 1:1** into Moore's stage-1 model classes
  (one trivial `conv_out_modify`→`conv_out` rename for pose_guider).
- **10/10 steps** completed at ~1.05 s/it: batch=1, 512², grad-accum=4,
  bf16, gradient checkpointing on, **8-bit Adam (bitsandbytes)**, no
  xformers (Blackwell falsified per the rest of this thread).
- 32-bit Adam OOMs at 28.99 GiB on the same config — the doubled
  optimizer state on ~1.7B trainable params is the killer. 8-bit Adam
  buys back ~10 GB.

A PersonaLive Stage-1 reconstruction *fits* on a single 5090. Tight
but feasible — leaves headroom for MotEncoder + MotionExtractor.
Stage 2 (temporal) and the motion modules remain unprobed.

Pointers:

- [`2026-05-04-moore-stage1-feasibility-probe.md`](../2026-05-04-moore-stage1-feasibility-probe.md) — spec + verdict
- [`2026-05-05-moore-stage1-probe-handoff.md`](../2026-05-05-moore-stage1-probe-handoff.md) — reproduce + vendor-patch capture
- [`2026-05-05-moore-stage1-vendor-patch.diff`](../2026-05-05-moore-stage1-vendor-patch.diff) — exact patches against Moore vendor `main`
- [`2026-05-05-personalive-stage1-unlocks.md`](../2026-05-05-personalive-stage1-unlocks.md) — what Stage-1 ft enables for us, what still needs Stage 2

### Strategic framing (2026-05-06)

[`2026-05-06-vtuber-pipeline-priorities.md`](../2026-05-06-vtuber-pipeline-priorities.md) splits the
work into two regimes (realtime puppeteering vs static-portrait authoring) and four products. Confirms
PersonaLive RGB-OBS is product #1, ARKit-OBS parallel #2, LivePortrait OBS gated on sanity test as #3;
disqualifies FlashPortrait/DiT for Regime A on speed; clarifies LoRA/FluxSpace work is Regime B and
not in competition with LivePortrait.

### 8-step monkey-patch unlocks LoRA budget; off-the-shelf ceiling at Ghibli α=3 (2026-05-06)

[`2026-05-06-personalive-8step-monkey-patch-lora-sweep.md`](../2026-05-06-personalive-8step-monkey-patch-lora-sweep.md)
— PersonaLive's 4-step distilled inference is hardcoded in 6+ places (timesteps `[999,666,333,0]` + `set_step_length(333)` magic-number-coupled). Monkey-patched at runtime to vanilla 8-step DDIM (`alphas_cumprod` cast to fp16 to fix dtype leak in `scheduler.step()`); engages on `--num_inference_steps != 4`. With Ghibli LoRA both α=3 + decoupled anime CLIP, output visibly flips toward Ghibli ("kinda there, minor artifacts"). Demon Slayer at same config is 2.3× weaker by Δ AND more artifacty — magnitude does not track quality across LoRA trainings. **Off-the-shelf SD1.5 LoRA ceiling at 8-step monkey-patch ≈ Ghibli α=3 quality**; production-clean stylization needs custom LoRA training per the canonical answer below. Sibling: `2026-05-06-decoupled-clip-channel-falsified-cleanup-effect.md` characterizes the 4-step compound as a *photoreal cleanup knob* useful for vamp-interface's sus=0 anchor.

### Stylized refs require LoRA on PersonaLive (2026-05-06)

[`2026-05-06-stylized-vtuber-requires-personalive-lora.md`](../2026-05-06-stylized-vtuber-requires-personalive-lora.md)
— after running both backbones over photoreal/painting/stylized-humanoid/non-human refs against the
yaw-stress driver: photoreal frontal both work (PersonaLive marginally cleaner), atypical photoreal
(Tikhonov) goes to LivePortrait (PersonaLive collapses to FFHQ prior), painting (Pushkin) to
PersonaLive, stylized humanoid both crop-confounded, **non-human (cartoon ducks) fails on both** —
no vtuber path for very-non-human refs today. PersonaLive has a LoRA surface (RefNet/UNet attention,
SD1.5 lineage) so its stylized failures are tunable; LivePortrait has no analogous surface
(warp-based, no prior, no attention to inject δ into) — fixing it requires retraining the implicit-
keypoint regressor on stylized data. **Vanilla LivePortrait benched at ~8.4 FPS end-to-end (~10 FPS
steady-state) on RTX 5090** — not realtime; FasterLivePortrait+TRT is the realtime path
(~30 FPS on 3090, expect 40+ on 5090). Pre-conditions for LoRA training: lock standardized
cropper + frame-0 baseline-subtraction control first.

### Read-first (strategic)

- [`docs/research/2026-05-06-vtuber-pipeline-priorities.md`](../2026-05-06-vtuber-pipeline-priorities.md)
  — Regime A (realtime puppeteering) vs Regime B (static authoring),
  four products, the LivePortrait-descendants survey ending in
  *FasterLivePortrait+TRT is the realtime path*.
- [`docs/research/2026-05-06-rendering-stack-replacement-options.md`](../2026-05-06-rendering-stack-replacement-options.md)
  — speed/quality candidate survey: Hyper-SD, DMD2, FlashPortrait,
  FasterLivePortrait, FLUX, Z-Image, discriminator-swap path for
  stylized refs.

### LLF → OBS pipeline (in progress)

- Spec: [`docs/superpowers/specs/2026-05-06-llf-to-obs-pipeline-design.md`](../../superpowers/specs/2026-05-06-llf-to-obs-pipeline-design.md)
- Plan: [`docs/superpowers/plans/2026-05-06-llf-obs-streaming.md`](../../superpowers/plans/2026-05-06-llf-obs-streaming.md)
- Streaming research notes: [`2026-05-06-rain-streaming-research.md`](../2026-05-06-rain-streaming-research.md)

### Reading order

- [`docs/research/2026-05-04-personalive-tensorrt-plan.md`](../2026-05-04-personalive-tensorrt-plan.md) — the 3-probe plan with gates
- [`docs/research/personalive-trt-logs/`](../personalive-trt-logs/) — raw bench logs (01-sdpa, 02-compile, 03c-torch_trt)
- [`docs/research/2026-05-04-personalive-trt-probe-c.md`](../2026-05-04-personalive-trt-probe-c.md) — Probe C (bundled torch2trt) result + detours
- [`docs/blog/2026-05-04-personalive-trt-probe-b.md`](../../blog/2026-05-04-personalive-trt-probe-b.md) — journal of the Probe B/C journey

### Environment notes

- torch 2.11.0+cu128, RTX 5090, sm_120
- torch-tensorrt 2.11.0 PyPI wheel needs CUDA 13 runtime libs
  (libcudart.so.13). Install `nvidia-cuda-runtime` + `nvidia-cuda-nvrtc`
  metas (cu13) and run with `LD_LIBRARY_PATH` pointing at
  `nvidia/cu13/lib`. cu128 torch and cu13 TRT runtime co-exist via
  separate dlopen paths.
