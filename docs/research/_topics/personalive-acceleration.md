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

### Open

- Probe B tuning: try larger `min_block_size`, `enable_autocast`,
  `optimization_level=5`, `decompose_attention=True`. Each iteration
  is ~5 min once cache is warm.
- Probe B and Probe C both in tune zone; cheapest next step is to
  tune one of them (autocast, larger min_block_size, optimization_level=5,
  decompose_attention=True). Probe C has the smaller surface area for
  tuning since only the engine config matters.
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
