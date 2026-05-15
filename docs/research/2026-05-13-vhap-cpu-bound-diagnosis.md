---
status: live
topic: arkit-bridge
---

# VHAP is CPU-bound on single-view iPhone takes — diagnosis

**Date:** 2026-05-13
**Context:** Running `tools/flame_tracking_video.py` (our multi-frame extension of
LAM's `tools/flame_tracking_single_image.py`) across 7 LLF takes for the LAM
bake-off, full-take duration. RTX 5090, torch 2.11+cu128, batch=1.

## Observation

- GPU utilization: **19%**. VRAM: 3.3 GB / 32 GB.
- Throughput on sequential-tracking stage: **~55 frames/min**, i.e. **~1.1 s/frame**.
- VHAP's per-frame inner loop runs **~100 optimizer steps** → **~10 ms per step**.

## Why it's CPU-bound, not GPU-bound

Three load-bearing pieces of `vhap/model/tracker.py`:

1. `DataLoader(..., num_workers=0)` at line 1407 / 1430 — single-threaded
   sample prep. But the per-frame sample is re-used across the 100-step inner
   loop, so PNG decode is amortized; not the hot path.

2. `optimize_stage()` body (line ~1448): `for step_i in range(num_steps): self.optimize_iter(sample, optimizer, stage)` —
   **batch-size 1**, FLAME forward + landmark loss + backward + Adam step,
   100 iterations per frame.

3. VHAP was authored for the **NeRSemble multi-view tracker** which runs ~16
   cameras per sample. That implicit batching keeps the GPU saturated. Our
   single-view iPhone .mov forces batch=1 and the optimizer is starved.

The 10 ms / step number is **kernel-launch overhead-limited**: FLAME forward
at batch=1 is microseconds of actual compute, but ~10 small CUDA kernels per
step × ~0.5–2 ms launch overhead each + Python overhead + Adam param-group
iteration + occasional `.item()` syncs = ~10 ms of CPU waiting on GPU sync.
Classic small-batch shape.

## Fixes (ranked by payoff per hour of dev)

| Fix | Effort | Wall-cut | Risk |
|---|---|---|---|
| **CUDA Graphs** capturing the 100-step inner loop, replay per frame | ~1 day | 5–10× | Brittle to control flow; needs warm-up |
| **`torch.compile(optimize_iter)`** | 1–2 h | 2–3× | Compile-time slows first frame; needs test |
| **Process-level parallelism** (3–4 VHAP procs on one GPU) | ~1 h | 3–4× | None — 3 × 3.3 GB fits in 32 GB VRAM |
| **Multi-frame temporal window** in sequential (K=4 frames/step) | ~1 day | 4× | Changes optimizer dynamics; re-tune required |
| **Drop num_steps 100→50** in `configs/vhap_tracking/base_tracking_config.yaml` | 5 min | 2× | Worse FLAME fit; eyeball-gate must hold |
| `num_workers=2` on the DataLoader | 5 min | <1.2× | Negligible — sample amortized |

## Implications for the LAM-driver pipeline

Drives the path-dependent answer on whether the **ARKit→FLAME retargeter spike**
is the right next step. Quick math:

- Today: VHAP wall on a 60 s LLF take ≈ 45 min. **Not live-streamable.**
- With CUDA-Graphs (10×): ≈ 4.5 min. Still not live.
- With process-parallel + CUDA-Graphs: ≈ 1.5 min for 60 s of footage. Faster
  than real-time but still batch-mode, not streaming.

VHAP is structurally an offline tracker. **Sequential init from previous
frame plus 100-step per-frame optimization is incompatible with live operation
on a 30 fps stream**, no matter what we accelerate. This is the strongest
single argument for the ARKit→FLAME retargeter path:

- Retargeter is **stateless per frame** — single forward through a ridge / MLP / small
  transformer, microseconds at batch=1.
- VHAP is **stateful and iterative** — by construction needs ~1 s of compute
  per 33 ms of input video.

For the bake-off this week, VHAP-offline + LAM-render is the right cohort
producer. For the live demo we have to go ARKit→FLAME.

## Action

- Sweep continues as-is. Wall ~7.5 h, completes overnight.
- Next session: pick **CUDA Graphs** or **process parallelism** depending on
  whether the goal is a single-take iteration loop (Graphs) or a corpus
  generator (parallelism). For the bake-off cohort, parallelism is the
  cheaper win.
- ARKit→FLAME retargeter spike (the `project_arkit_bridge_next_experiments`
  pointer) is reinforced as the right next step independent of any VHAP
  acceleration.

## Related

- [[project_lam_bakeoff_x6_verdict]] — LAM as renderer-of-record verdict
- `rorschach/docs/shaping/slices/X6-lam-bakeoff-limitations-and-next.md` —
  driver-contract limitation is the long pole
- `LAM/tools/flame_tracking_video.py` — our multi-frame VHAP wrapper
- `scripts/lam_take_sweep.sh` — the in-flight sweep driver
