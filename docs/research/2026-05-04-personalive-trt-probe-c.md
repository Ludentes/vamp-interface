---
status: live
topic: personalive-acceleration
summary: Probe C result — bundled torch2trt.py + polygraphy → 18.87 FPS (+88% over SDPA, +6.5% over Probe B). Lands in tune zone, below ship gate. Single 3.7 GB engine fuses denoising_unet + vae.decode + scheduler.step into one TRT graph.
---

# Probe C — bundled torch2trt.py result

## Headline

```
mode=trt_bundled L=100 warmup=16 time=5.301s fps=18.87
```

vs SDPA 10.03 / torch.compile 12.10 / torch_tensorrt 17.72.

**+88% over the baseline. +6.5% over Probe B.** Lands in the 15-22 tune
band, still below the ≥22 ship gate.

## What's different from Probe B

Probe B (`torch_tensorrt`) compiled three modules independently —
denoising_unet, reference_unet, vae.decode — and replaced the
`nn.Module`s with TRT-wrapped runtimes. The PyTorch pipeline still
drove the loop: scheduler.step ran in eager torch, the
diffusers `Pose2VideoPipeline_Stream` orchestrated everything,
and there were per-call boundary copies between TRT and torch tensors.

Probe C builds a single TRT engine for `unet_work` (the upstream's
fused composite) that takes 23 inputs and produces 6 outputs. The
engine internally chains: `pose_guider → motion_encoder → unet_3d →
scheduler.step → vae.decode`. The wrapper (`src.wrapper_trt.PersonaLive`)
prefills the 16 reference-cache tensors once per reference image and
binds output→input tensors so the latent / pose_cond_fea /
motion_hidden_states feedback loops never copy. Per-frame call: 4
inputs in (pose, motion, new_noise, encoder_hidden_states), 6 outputs
out, no torch ops in between.

## What we expected vs what we got

The hypothesis going in: bypassing PyTorch fallback boundaries should
deliver the largest acceleration. We had Probe C marked as the path
with the highest ceiling.

The actual delta: **+6.5% over Probe B**. Most of the win
torch_tensorrt couldn't deliver was tiny.

What this tells us: the per-module ttrt approach captured almost all
of the achievable speedup in the kernel-launch and convolution phase.
The remaining headroom was in scheduler arithmetic + the cross-graph
boundary tax — both small fractions of the 5.3-second run.

## Build cost

- ONNX export: ~30 seconds.
- ONNX resave with external data: ~10 seconds.
- Polygraphy + TRT autotune over 16 dynamic-shape inputs: ~9 minutes.
- Total wall time: ~10 minutes.
- Disk: 3.6 GB ONNX external data + 3.7 GB engine.

## Detours

**`auto_cast=True` falsifies the trace.** Upstream's `torch2trt.py`
passes `auto_cast=True` into `export_onnx`. Inside, it wraps the
trace in `torch.autocast("cuda")`, which forces BatchNorm inputs
through fp32 even though the wrapper's weights are fp16. The
motion_encoder's FAN feature extractor crashes during ONNX trace with:

```
RuntimeError: Expected tensor for argument #1 'input' to have the same
type as tensor for argument #2 'weight'; but type torch.cuda.FloatTensor
does not equal torch.cuda.HalfTensor
```

Set `auto_cast=False`. The wrapper holds fp16 weights and feeds fp16
inputs already; autocast is just adding a fight. Trace passes cleanly.

**TRT version mismatch is benign.** Upstream pins `tensorrt==10.1.0` in
`requirements_trt.txt`. We have 10.15.1.29 (cu13) from the Probe B
install. polygraphy 0.49.26 builds the engine without complaint.
sm_120 is supported in 10.15 (`libnvinfer_builder_resource_sm120.so`
ships in the wheel).

**Bench harness needs a custom adapter.** `webcam.vid2vid_trt.Pipeline`
is multiprocess Queue/Event-based — designed for streaming webcam
input, not offline timing. Calling it would measure queue plumbing.
We bypass it: `scripts/bench_trt_bundled.py` uses
`src.wrapper_trt.PersonaLive` directly, drives `process_input(...)` 4
frames at a time over the same driving frames `bench_offline.prepare_inputs`
gives the SDPA / compile / torch_trt modes, and times around the
loop with `cuda.synchronize()` at both ends.

## What's load-bearing for Probe C, in order

The contiguous-clone fix from Probe B is no-op here — we don't replace
nn.Modules with TRT wrappers, so diffusers' pipeline doesn't probe
their parameters and the `_execution_device` walk doesn't trip.

The CUDA 13 / 12.8 hybrid runtime trick from Probe B is *still*
load-bearing. Without `LD_LIBRARY_PATH=$VENV/.../nvidia/cu13/lib`,
tensorrt 10.15.1.29 fails to import `libnvinfer.so` because the wheel
links against cu13 runtime symbols.

The `auto_cast=False` flip is the only Probe-C-specific fix. Without
it the ONNX trace dies before producing anything useful.

## Where this leaves us

Two things in the tune zone (Probe B at 17.72, Probe C at 18.87),
neither at the ≥22 ship gate. The cheap remaining levers all live
inside the engine config:

- `optimization_level=5` (max autotune effort, longer build for ~5-10%)
- `BuilderFlag.STRICT_TYPES` off → let TRT pick fp16 vs fp32 per-layer
- Engine inspector to find the slowest subgraph; common candidates
  are the temporal attention layers and the vae upsamples
- Reduce engine dynamic-shape range (we ship 1×–4× the base, but
  only ever run 1× at L=100) — narrower profile means more
  specialized kernels

Probe B remains the cheaper iteration loop because each retry is ~30s
(cached engines), whereas Probe C rebuilds take ~10 min. We tune
Probe B first; if that doesn't reach 22 FPS, fall back to Probe C
config space.

## Files

- Build script: `PersonaLive/scripts/build_trt_engine.py`
- Bench adapter: `PersonaLive/scripts/bench_trt_bundled.py`
- Bench config: `PersonaLive/configs/prompts/personalive_trt_bench.yaml`
- Engine: `/tmp/personalive_trt/bundled/unet_work.engine` (3.7 GB)
- Bench log: `/tmp/personalive_trt/04-bench.log`
- Build log: `/tmp/personalive_trt/04-build.log`
