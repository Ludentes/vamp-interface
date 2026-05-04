# PersonaLive Inference Acceleration (sm_120) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Push PersonaLive inference from 8.0 FPS (SDPA baseline) toward the ≥25 FPS real-time target on RTX 5090 (sm_120) by trying three acceleration paths in increasing cost order, stopping at the first that clears the gate.

**Architecture:** Three ordered probes. Probe A: `torch.compile` (Inductor + Triton, zero deps). Probe B: `torch_tensorrt.compile` per submodule (stays in PyTorch, TRT kernels with PyTorch fallback for unsupported ops). Probe C: upstream `torch2trt.py` bundled-engine path (full ONNX export + polygraphy build of fused `unet_work`). Each probe is gated by the same offline benchmark (`inference_offline.py -L 100`) and a single decision rule: ≥22 FPS ship, 15–22 continue tuning, <15 advance to next probe.

**Tech Stack:** torch 2.11.0+cu128, Triton 3.6.0 (already installed), torch_tensorrt ≥2.5 (Blackwell-capable), TensorRT ≥10.4 (sm_120), polygraphy, onnx 1.17, pycuda 2024.1.2, uv, RTX 5090 sm_120.

---

## Pre-flight context

**Repo facts (verified 2026-05-04):**
- PersonaLive `.venv` at `/home/newub/w/PersonaLive/.venv/` — Python 3.10, uv-managed. All `pip` commands use `uv pip` from inside the venv.
- Current SDPA baseline: 13.7 s / 100 frames = 7.3 FPS (xformers attempts closed yesterday with no gain — see `2026-05-03-xformers-flashattn-saga.md`).
- Custom attention processors active: `set_attn_processor(AttnProcessor())` on both unets and `set_default_attn_processor()` on VAE. These are vanilla `AttnProcessor`, not `AttnProcessor2_0`, which matters for graph-fusion friendliness.
- Pipeline structure: `Pipeline` (`webcam/vid2vid.py`) holds `reference_unet` (2D), `denoising_unet` (3D w/ motion module), `pose_guider`, `motion_encoder`, `vae`, `scheduler`. Per-frame call drives the denoising_unet across `num_inference_steps=4` × `temporal_window_size=4`.
- Memory hazard: any heavy compile/build wraps in `systemd-run --user --scope -p MemoryMax=40G`.
- Decision gate (same for all probes): **≥22 FPS = ship & stop. 15–22 = tune within probe (warmup, dynamic shapes, mode flags). <15 = advance to next probe.**

**Out of scope:** quality regression study (numeric drift acceptable if visually fine), Windows port, ComfyUI integration, INT8 quantization.

---

## File / artifact map

**Create:**
- `/home/newub/w/PersonaLive/scripts/bench_offline.py` — single benchmark entrypoint, takes `--mode {sdpa,compile,torch_trt,trt_bundled}`, runs warmup + N timed frames, prints FPS. Reused by all probes.
- `/home/newub/w/vamp-interface/docs/research/personalive-trt-logs/` — log directory (per-probe).
- `/home/newub/w/vamp-interface/docs/research/2026-05-04-personalive-acceleration-result.md` — final result doc.

**Modify (per-probe, may revert):**
- `/home/newub/w/PersonaLive/webcam/vid2vid.py` — wrap modules in `torch.compile` / `torch_tensorrt.compile` (Probes A and B).
- `/home/newub/w/PersonaLive/requirements_trt.txt` — bump TRT pin (Probe C only).

**Touch but don't commit:**
- `/home/newub/w/PersonaLive/pretrained_weights/onnx/...`, `pretrained_weights/tensorrt/unet_work.engine` (Probe C only; back up the H100 ship-engine first).

---

## Task 1: Baseline rematch + shared benchmark harness

**Files:**
- Create: `/home/newub/w/PersonaLive/scripts/bench_offline.py`
- Backup: `/home/newub/w/PersonaLive/pretrained_weights/tensorrt/unet_work.engine.h100.bak`

- [ ] **Step 1: Confirm hardware + torch state**

```bash
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
cd /home/newub/w/PersonaLive && source .venv/bin/activate
python -c "import torch, triton; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cap', torch.cuda.get_device_capability(0), 'triton', triton.__version__)"
```

Expected: RTX 5090, compute_cap 12.0, torch 2.11.0+cu128, capability `(12, 0)`, triton 3.6.0.

- [ ] **Step 2: Read the streaming Pipeline to find the per-frame entry**

```bash
grep -n "def \(__init__\|run\|step\|forward\|__call__\)" /home/newub/w/PersonaLive/webcam/vid2vid.py
sed -n '1,40p' /home/newub/w/PersonaLive/webcam/vid2vid.py
```

Note the constructor signature, the per-frame method name (likely `step()` or `__call__`), and which attributes are model modules (`self.denoising_unet`, `self.reference_unet`, `self.vae`, `self.pose_guider`, `self.motion_encoder`). Record these — Probes A and B need to wrap them.

- [ ] **Step 3: Write `bench_offline.py`**

```python
# /home/newub/w/PersonaLive/scripts/bench_offline.py
"""Single benchmark harness for SDPA / compile / torch_tensorrt / TRT-bundled."""
import argparse, time, gc
import torch
from omegaconf import OmegaConf

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["sdpa", "compile", "torch_trt", "trt_bundled"], required=True)
ap.add_argument("-L", type=int, default=100)
ap.add_argument("--warmup", type=int, default=10)
ap.add_argument("--reference_image", default="./assets/ref_1.png")
ap.add_argument("--driving_video", default="./assets/drive_1.mp4")
args = ap.parse_args()

cfg = OmegaConf.load("./configs/prompts/personalive_online.yaml")
device = torch.device("cuda:0")

if args.mode == "trt_bundled":
    from webcam.vid2vid_trt import Pipeline
else:
    from webcam.vid2vid import Pipeline
pipe = Pipeline(cfg, device)

if args.mode == "compile":
    # Wrap the heavy modules — names confirmed from Step 2 above.
    pipe.denoising_unet = torch.compile(pipe.denoising_unet, mode="reduce-overhead", dynamic=True)
    pipe.reference_unet = torch.compile(pipe.reference_unet, mode="reduce-overhead", dynamic=True)
    pipe.vae.decode = torch.compile(pipe.vae.decode, mode="reduce-overhead", dynamic=True)
elif args.mode == "torch_trt":
    import torch_tensorrt as ttrt
    # Per-submodule compile — input specs filled in Task 3 once shapes are known.
    raise NotImplementedError("filled in by Task 3 step 4")

# Resolve per-frame call. ADAPT after reading vid2vid.py in Step 2:
def one_frame():
    return pipe.step()  # replace with actual method discovered in Step 2

# Drive a real reference + video so the call is representative
pipe.set_reference(args.reference_image) if hasattr(pipe, "set_reference") else None
pipe.load_driving(args.driving_video) if hasattr(pipe, "load_driving") else None

for _ in range(args.warmup):
    one_frame()
torch.cuda.synchronize()

t0 = time.time()
for _ in range(args.L):
    one_frame()
torch.cuda.synchronize()
dt = time.time() - t0
print(f"mode={args.mode} L={args.L} warmup={args.warmup} time={dt:.3f}s fps={args.L/dt:.2f}")
```

**Important:** the `pipe.step()` / `pipe.set_reference()` / `pipe.load_driving()` calls are placeholders. Adapt them after Step 2 to match the actual `Pipeline` API. Do not leave placeholders in — that produces a fake benchmark.

- [ ] **Step 4: Run SDPA baseline through the harness**

```bash
mkdir -p /tmp/personalive_trt
systemd-run --user --scope -p MemoryMax=40G \
  python scripts/bench_offline.py --mode sdpa -L 100 2>&1 | tee /tmp/personalive_trt/01-sdpa.log
```

Expected: `mode=sdpa L=100 fps=7.X` (matches the prior baseline within ±10%). If it doesn't, the harness is wrong (probably because the per-frame call doesn't match `inference_offline.py`'s timed region) — fix before proceeding.

- [ ] **Step 5: Back up the shipped H100 engine (used in Probe C only, but back up now)**

```bash
cp /home/newub/w/PersonaLive/pretrained_weights/tensorrt/unet_work.engine \
   /home/newub/w/PersonaLive/pretrained_weights/tensorrt/unet_work.engine.h100.bak
ls -la /home/newub/w/PersonaLive/pretrained_weights/tensorrt/
```

- [ ] **Step 6: Commit baseline**

```bash
cd /home/newub/w/vamp-interface
mkdir -p docs/research/personalive-trt-logs
cp /tmp/personalive_trt/01-sdpa.log docs/research/personalive-trt-logs/
git add docs/research/personalive-trt-logs/01-sdpa.log docs/research/2026-05-04-personalive-tensorrt-plan.md
git commit -m "docs(personalive-accel): SDPA harness baseline + plan"
```

---

## Task 2: Probe A — `torch.compile`

**Files:** `/home/newub/w/PersonaLive/scripts/bench_offline.py` (already supports `--mode compile`)

- [ ] **Step 1: First compile run (cold)**

```bash
cd /home/newub/w/PersonaLive && source .venv/bin/activate
systemd-run --user --scope -p MemoryMax=40G \
  python scripts/bench_offline.py --mode compile -L 100 --warmup 20 2>&1 | tee /tmp/personalive_trt/02-compile.log
```

Expected: long pause on first warmup frame (Inductor compilation, can take 30–120 s per module), then normal frames. If warmup itself OOMs, drop `mode="reduce-overhead"` to `mode="default"`.

- [ ] **Step 2: Diagnose recompilations**

If FPS is bad and the log shows "TorchDynamo: recompile" lines, shapes are changing per-frame. Set `dynamic=True` (already in harness) and increase warmup. If still recompiling:

```bash
TORCH_LOGS="recompiles" python scripts/bench_offline.py --mode compile -L 30 --warmup 30 2>&1 | grep -i "recompil\|guard" | head -40
```

Common culprit: the temporal_window slicing produces different last-batch shapes. Fix: pad to fixed window or skip the trailing partial window in benchmark.

- [ ] **Step 3: Re-run after fixes, save final number**

```bash
systemd-run --user --scope -p MemoryMax=40G \
  python scripts/bench_offline.py --mode compile -L 100 --warmup 20 2>&1 | tee /tmp/personalive_trt/02-compile-final.log
```

- [ ] **Step 4: Decision**

Read FPS from log. Apply gate:
- ≥22: write result doc (Task 5), stop.
- 15–22: try `mode="max-autotune"` once (longer warmup), keep best. Then advance to Probe B (could combine).
- <15: advance to Probe B.

- [ ] **Step 5: Commit**

```bash
cd /home/newub/w/vamp-interface
cp /tmp/personalive_trt/02-compile*.log docs/research/personalive-trt-logs/
git add docs/research/personalive-trt-logs/02-compile*.log
git commit -m "docs(personalive-accel): probe A (torch.compile) result"
```

---

## Task 3: Probe B — `torch_tensorrt.compile`

**Files:**
- Modify: `/home/newub/w/PersonaLive/scripts/bench_offline.py` (replace the `NotImplementedError` from Task 1 Step 3)

- [ ] **Step 1: Install torch_tensorrt**

```bash
cd /home/newub/w/PersonaLive && source .venv/bin/activate
uv pip install torch-tensorrt --index-url https://download.pytorch.org/whl/cu128 2>&1 | tee /tmp/personalive_trt/03-ttrt-install.log
python -c "import torch_tensorrt as ttrt; print('torch_tensorrt', ttrt.__version__); import tensorrt as trt; print('tensorrt', trt.__version__)"
```

Expected: torch_tensorrt ≥ 2.5, TRT ≥ 10.4 (it pulls TRT as a dep). If TRT < 10.4 lands, force `uv pip install "tensorrt>=10.4,<11"` after.

If install fails on pycuda (transitive in some torch_tensorrt builds), fall to:
```bash
uv pip install "torch-tensorrt[no-pycuda]" --index-url https://download.pytorch.org/whl/cu128
```

- [ ] **Step 2: Verify sm_120 is in the TRT kernel set**

```bash
python - <<'PY'
import tensorrt as trt
print("trt", trt.__version__)
import pycuda.autoinit, pycuda.driver as drv
dev = drv.Device(0)
print("device", dev.name(), "cap", dev.compute_capability())
b = trt.Builder(trt.Logger(trt.Logger.WARNING))
print("fp16 fast:", b.platform_has_fast_fp16)
PY
```

Expected: cap `(12, 0)`, no errors. If TRT logs "unsupported compute capability", bump: `uv pip install --reinstall "tensorrt>=10.7,<11"`.

- [ ] **Step 3: Probe shapes for the heavy modules**

We need `min_shape`, `opt_shape`, `max_shape` for each module's inputs to feed `torch_tensorrt.compile`. Add a one-shot probe inside `Pipeline.__init__` or via monkeypatch:

```python
# /home/newub/w/PersonaLive/scripts/probe_shapes.py
import torch
from omegaconf import OmegaConf
from webcam.vid2vid import Pipeline

cfg = OmegaConf.load("./configs/prompts/personalive_online.yaml")
pipe = Pipeline(cfg, torch.device("cuda:0"))

def hook(name):
    def f(mod, inp, out):
        shapes = [tuple(t.shape) if hasattr(t, "shape") else type(t).__name__ for t in inp]
        dtypes = [t.dtype if hasattr(t, "dtype") else None for t in inp]
        print(f"[{name}] inp shapes={shapes} dtypes={dtypes}")
    return f

pipe.denoising_unet.register_forward_hook(hook("denoising_unet"))
pipe.reference_unet.register_forward_hook(hook("reference_unet"))
pipe.vae.decoder.register_forward_hook(hook("vae.decoder"))

# drive a few frames
pipe.set_reference("./assets/ref_1.png")
pipe.load_driving("./assets/drive_1.mp4")
for _ in range(3):
    pipe.step()
```

Run:
```bash
python scripts/probe_shapes.py 2>&1 | tee /tmp/personalive_trt/03-shapes.log
```

Record the shapes seen. They are the `opt_shape`. `min_shape` = same with the dynamic dim → 1, `max_shape` = same with dynamic dim → max needed (read from cfg `temporal_window_size`).

- [ ] **Step 4: Fill `torch_trt` branch in `bench_offline.py`**

Replace the `NotImplementedError` in the harness with the per-submodule compile, using shapes from Step 3. Sketch:

```python
elif args.mode == "torch_trt":
    import torch_tensorrt as ttrt
    # Shapes from probe — REPLACE with values from Step 3 log
    pipe.denoising_unet = ttrt.compile(
        pipe.denoising_unet,
        ir="dynamo",
        inputs=[
            ttrt.Input(min_shape=(1, 4, 1, 64, 64), opt_shape=(1, 4, 4, 64, 64),
                       max_shape=(1, 4, 4, 64, 64), dtype=torch.float16),
            # add remaining inputs (timestep, encoder_hidden_states, ref features...)
        ],
        enabled_precisions={torch.float16},
        truncate_double=True,
    )
    pipe.reference_unet = ttrt.compile(pipe.reference_unet, ir="dynamo",
        inputs=[ttrt.Input(opt_shape=(1, 4, 64, 64), dtype=torch.float16)],
        enabled_precisions={torch.float16})
    pipe.vae.decoder = ttrt.compile(pipe.vae.decoder, ir="dynamo",
        inputs=[ttrt.Input(opt_shape=(1, 4, 64, 64), dtype=torch.float16)],
        enabled_precisions={torch.float16})
```

The exact `Input(...)` lists must match what Step 3 logged.

- [ ] **Step 5: Run, expect long compile on first warmup**

```bash
systemd-run --user --scope -p MemoryMax=40G \
  python scripts/bench_offline.py --mode torch_trt -L 100 --warmup 20 2>&1 | tee /tmp/personalive_trt/03-torch_trt.log
```

Expected: 1–5 minutes of compile printed by TRT logger on first frame, then steady-state FPS. Look for `[W] [TRT] ... falling back to PyTorch` lines — count them. Many fallbacks = poor coverage = limited speedup.

- [ ] **Step 6: If fallbacks dominate, try a larger graph**

If denoising_unet alone has lots of fallbacks (likely for the motion module's custom ops), try:
```python
pipe.denoising_unet = ttrt.compile(
    pipe.denoising_unet,
    ir="dynamo",
    require_full_compilation=False,  # explicit
    min_block_size=5,                # smaller blocks → more TRT coverage
    inputs=[...],
    enabled_precisions={torch.float16},
)
```

If that still leaves the motion module on PyTorch, that's expected — the motion module is what makes Probe C (full bundled engine) potentially worth it.

- [ ] **Step 7: Decision**

- ≥22: stop, write result doc.
- 15–22: try combining with `torch.compile` on remaining PyTorch fallback subgraphs. Then either ship or advance.
- <15: advance to Probe C.

- [ ] **Step 8: Commit**

```bash
cd /home/newub/w/vamp-interface
cp /tmp/personalive_trt/03*.log docs/research/personalive-trt-logs/
git add docs/research/personalive-trt-logs/03*.log
git commit -m "docs(personalive-accel): probe B (torch_tensorrt) result"
```

---

## Task 4: Probe C — upstream bundled `torch2trt.py`

Only run if Probes A+B both miss the 15 FPS gate. The hypothesis driving this probe: the win comes from bundling pose_guider + motion_encoder + denoising_unet + vae into a single ONNX → single engine, eliminating per-frame Python orchestration overhead between submodules.

- [ ] **Step 1: Bump TRT pin and install full toolchain**

```bash
cd /home/newub/w/PersonaLive
sed -i 's/^tensorrt==.*/tensorrt>=10.4,<11/' requirements_trt.txt
source .venv/bin/activate
uv pip install -r requirements_trt.txt 2>&1 | tee /tmp/personalive_trt/04-trt-deps.log
```

If pycuda fails to build a wheel, fall back to manual build:
```bash
CUDA_ROOT=/usr/local/cuda-12.8 uv pip install --no-binary :all: pycuda==2024.1.2
```

If that also fails, conda-install pycuda and symlink into the venv (see prior plan version for the symlink recipe). Document the path taken in the result doc.

- [ ] **Step 2: Pre-flight disk + memory**

```bash
df -h /home/newub/w/PersonaLive
free -h
```

Expected: ≥30 GB free disk, ≥32 GB RAM headroom.

- [ ] **Step 3: Run `torch2trt.py` (ONNX export + engine build)**

```bash
mkdir -p /tmp/personalive_trt
systemd-run --user --scope -p MemoryMax=40G \
  python -u torch2trt.py 2>&1 | tee /tmp/personalive_trt/04-build.log
```

Stages in log: model load → `export_onnx` → "finished" → "Optimizing Onnx Model..." → polygraphy verbose → engine save.

If ONNX export dies on an unsupported op, read the op name. Common fixes: bump opset 17 → 18/19 in `torch2trt.py` line `onnx_opset = 17`. If onnx-simplifier hangs > 30 min, edit `src/modeling/onnx_export.py:optimize_onnx` to copy file unchanged. If polygraphy fails on sm_120, bump TRT to 10.7+.

The script's own resume gate (`if not os.path.exists(onnx_path):`) means re-running after a crash skips re-export.

- [ ] **Step 4: Verify engine deserializes for sm_120**

```bash
python - <<'PY'
import tensorrt as trt
runtime = trt.Runtime(trt.Logger(trt.Logger.INFO))
with open("./pretrained_weights/tensorrt/unet_work.engine", "rb") as f:
    eng = runtime.deserialize_cuda_engine(f.read())
print("ok, num_io_tensors:", eng.num_io_tensors)
for i in range(eng.num_io_tensors):
    n = eng.get_tensor_name(i)
    print(f"  {n}: {eng.get_tensor_shape(n)} {eng.get_tensor_dtype(n)}")
PY
```

- [ ] **Step 5: Benchmark via the harness**

```bash
systemd-run --user --scope -p MemoryMax=40G \
  python scripts/bench_offline.py --mode trt_bundled -L 100 --warmup 10 2>&1 | tee /tmp/personalive_trt/04-trt-bundled.log
```

The `trt_bundled` mode imports `webcam.vid2vid_trt.Pipeline`. If its per-frame call uses a different method name than `webcam.vid2vid.Pipeline`, the harness's `one_frame()` needs to dispatch on `args.mode`. Adjust now.

- [ ] **Step 6: Commit**

```bash
cd /home/newub/w/vamp-interface
cp /tmp/personalive_trt/04*.log docs/research/personalive-trt-logs/
git add docs/research/personalive-trt-logs/04*.log
git commit -m "docs(personalive-accel): probe C (bundled TRT engine) result"
```

---

## Task 5: Result doc + topic index

- [ ] **Step 1: Write `2026-05-04-personalive-acceleration-result.md`**

Save to `/home/newub/w/vamp-interface/docs/research/`. Template:

```markdown
---
status: live
topic: neural-deformation-control
---

# PersonaLive inference acceleration on RTX 5090 — result

**Date:** 2026-05-04
**Hardware:** RTX 5090 (sm_120), torch 2.11.0+cu128, triton 3.6.0

## Numbers
| Probe | mode | warmup | L | time (s) | FPS | × baseline |
| baseline | sdpa | 10 | 100 | … | … | 1.00 |
| A | compile | 20 | 100 | … | … | … |
| B | torch_trt | 20 | 100 | … | … | … |
| C | trt_bundled | 10 | 100 | … | … | … |

## Probe-by-probe
### Probe A (torch.compile)
What worked, fallbacks/recompiles seen, mode used (default / reduce-overhead / max-autotune).

### Probe B (torch_tensorrt)
torch_tensorrt version, TRT version, install path (wheel / pycuda fallback), per-module fallback count, dominant unsupported ops.

### Probe C (bundled torch2trt.py)
Only filled if reached. ONNX export issues, simplifier behavior, engine build wall time, kernel-arch issues.

## Decision
Picked: <probe>
Why: <one paragraph — speedup, complexity, portability tradeoffs>

## Next levers (if didn't hit 25 FPS)
- temporal_window cuts (fewer denoising steps per call)
- batch the temporal window better
- INT8 quantization (TRT-side, separate calibration)
- CUDA Graphs on the per-frame loop

## Cross-references
- Plan: 2026-05-04-personalive-tensorrt-plan.md
- Predecessor: 2026-05-03-xformers-flashattn-saga.md
```

- [ ] **Step 2: Update topic index**

Edit `/home/newub/w/vamp-interface/docs/research/_topics/neural-deformation-control.md`. Add one line under "Key dated docs" pointing at the result doc with a one-sentence hook describing the chosen probe and FPS.

- [ ] **Step 3: Update memory iff outcome is decisive**

If a probe ships at ≥22 FPS, write a `feedback_personalive_acceleration.md` memory: which probe, why, and the trap to avoid (e.g., "skip upstream torch2trt.py — torch_tensorrt is enough"). If outcome is "tried, none cleared 15", a single MEMORY.md pointer is enough.

- [ ] **Step 4: Final commit**

```bash
cd /home/newub/w/vamp-interface
git add docs/research/2026-05-04-personalive-acceleration-result.md docs/research/_topics/neural-deformation-control.md
git commit -m "docs(personalive-accel): result + topic index"
```

---

## Notes on what NOT to do

- **Don't skip Probe A.** It's free and may be enough.
- **Don't run Probes B+C in the same venv state without rolling back monkey-patches.** The harness loads a fresh `Pipeline` each invocation, but `torch._dynamo` global state can carry over — restart the Python process between probes (already true since each is a separate `python` invocation).
- **Don't time without `torch.cuda.synchronize()`.** Async kernel launches will report fake speedups, especially for `torch.compile` which queues aggressively.
- **Don't trust a 3×+ speedup without an output-pixel check.** Compare one frame's output to the SDPA baseline (mean-abs diff < 0.01 in [0,1] range is fine). Silent kernel mismatches happen, especially with `torch_tensorrt` fallbacks.
- **Don't try to combine probes without a reason.** If A gets 16 FPS, ship it. Don't stack `torch.compile` on top of `torch_tensorrt.compile` outputs unless A+B alone don't clear the gate and there's a clear PyTorch fallback subgraph that A can speed up.
- **Don't replace the H100 ship engine without backing it up** (Task 1 Step 5).
