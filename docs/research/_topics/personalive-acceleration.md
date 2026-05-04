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
