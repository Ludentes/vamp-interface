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
  not yet attempted. Higher prior of acceleration since it bypasses
  TRT↔PyTorch fallback boundaries, but ~1-2 h setup (onnx, polygraphy,
  pycuda, custom UNet wrapper).

### Open

- Probe B tuning: try larger `min_block_size`, `enable_autocast`,
  `optimization_level=5`, `decompose_attention=True`. Each iteration
  is ~5 min once cache is warm.
- Probe C run-through.
- Visual artifact comparison across modes (videos exist, side-by-side
  pending).

### Reading order

- [`docs/research/2026-05-04-personalive-tensorrt-plan.md`](../2026-05-04-personalive-tensorrt-plan.md) — the 3-probe plan with gates
- [`docs/research/personalive-trt-logs/`](../personalive-trt-logs/) — raw bench logs (01-sdpa, 02-compile, 03c-torch_trt)
- [`docs/blog/2026-05-04-personalive-trt-probe-b.md`](../../blog/2026-05-04-personalive-trt-probe-b.md) — journal of the Probe B journey

### Environment notes

- torch 2.11.0+cu128, RTX 5090, sm_120
- torch-tensorrt 2.11.0 PyPI wheel needs CUDA 13 runtime libs
  (libcudart.so.13). Install `nvidia-cuda-runtime` + `nvidia-cuda-nvrtc`
  metas (cu13) and run with `LD_LIBRARY_PATH` pointing at
  `nvidia/cu13/lib`. cu128 torch and cu13 TRT runtime co-exist via
  separate dlopen paths.
