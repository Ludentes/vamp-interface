---
status: live
topic: neural-deformation-control
---

# The xformers / flash-attn saga on RTX 5090 (Blackwell sm_120)

**Date:** 2026-05-03
**Context:** Bringing PersonaLive up on the local RTX 5090 box. Companion + supersedes parts of [2026-05-03-xformers-blackwell-pivot.md](2026-05-03-xformers-blackwell-pivot.md). This doc captures the full chain of dead ends, why each one died, and what the working setup looks like — so we don't repeat any of it on the office Windows 5090, on ComfyUI, or on any future Blackwell venv.

## TL;DR

1. **xformers ≥ 0.0.35 no longer ships CUDA fmha kernels** in its source tree. Source-building it for sm_120 is a no-op (gives you working sparse24 ops and nothing else). Use the published wheel.
2. **The published xformers wheel (`xformers==0.0.35`, cu128 channel) has Python attention dispatchers but no compiled-in fmha backends for Blackwell.** It still imports cleanly. At enable time it probes for backends; on Blackwell every probe fails (`fa3F`/`cutlassF` blocked by capability cap; `fa2F` blocked by fp32 probe dtype). PersonaLive's try/except swallows the error and falls through to PyTorch SDPA — which is the right behavior, you just don't get the speedup.
3. **Installing `flash-attn` separately gives xformers a backend to dispatch to.** On Blackwell this means a source build (no Blackwell wheels on PyPI). Build is **RAM-hungry** — eats >40 GB peak with default `MAX_JOBS`. Always cap with `systemd-run --user --scope -p MemoryMax=40G` and `MAX_JOBS=2 NVCC_THREADS=1`. Build takes ~2.5 h because it compiles for sm_80/90/100/120 (four archs).
4. **Windows + Ampere is trivial** — kingbri1/flash-attention ships prebuilt cu128torch2.7 cp310 wheels that work on a cu126 torch (CUDA forward-compat). 2 minutes flat.
5. **Trap: source-building xformers from a clean checkout overwrites the working wheel** with a broken local build (`xformers==0.0.35+<git-hash>.d<date>`). That broken build's `xformers.ops` lacks `memory_efficient_attention` entirely (since the kernels aren't there and the dispatcher table isn't populated). Symptom: `module 'xformers.ops' has no attribute 'memory_efficient_attention'`. Recovery: `uv pip install --reinstall xformers --index-url https://download.pytorch.org/whl/cu128`.

## Timeline of attempts

### Attempt 1 — install published wheel only

```
VIRTUAL_ENV=.venv uv pip install xformers --index-url https://download.pytorch.org/whl/cu128
# → xformers 0.0.35 installed
```

PersonaLive smoke test: enable_xformers fails with:

```
Failed to enable xformers: No operator found for `memory_efficient_attention_forward` ...
  fa3F@0.0.0 not supported because: requires capability <= (9, 0) but GPU has (12, 0)
  fa2F@2.5.7-pt not supported because: dtype=torch.float32 (only fp16/bf16)
  cutlassF-pt not supported because: requires capability <= (9, 0) but GPU has (12, 0)
```

PersonaLive falls through to SDPA. **7.3 FPS** for 100 frames.

This is the "working but unaccelerated" state. Everything imports and runs, just no flash kernels.

### Attempt 2 — source-build xformers for sm_120 (FALSIFIED)

Followed the late-2025 community recipe:

```
TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=4 FORCE_CUDA=1 \
  uv pip install -v --no-build-isolation \
  "xformers @ git+https://github.com/facebookresearch/xformers.git@main"
```

Build completes suspiciously fast (~90 s, 13 ninja steps). Resulting package has only sparse24 + indexing ops. **No fmha kernels.**

Cause: between 0.0.34 and 0.0.35 (Feb 2025), xformers migrated to PyTorch stable ABI and **structurally removed bundled CUDA fmha kernels** from `xformers/csrc/`. Only 11 `.cu` files remain in the source tree. The flash/cutlass modules became Python dispatchers that delegate to upstream packages.

**Side effect:** the source-build replaces the working wheel with a broken `xformers==0.0.35+<git-hash>.d<date>` install where `xformers.ops` no longer exposes `memory_efficient_attention` at all. This breaks PersonaLive enable in a different way than Attempt 1.

Full writeup: [2026-05-03-xformers-blackwell-pivot.md](2026-05-03-xformers-blackwell-pivot.md).

### Attempt 3 — `pip install flash-attn` straight (RAM blowout)

```
VIRTUAL_ENV=.venv uv pip install flash-attn --no-build-isolation
```

`--no-build-isolation` does **not** mean "wheel-only" — it just skips the build sandbox. When PyPI has no wheel matching the install target (no cu128 torch 2.11 cp310 Linux wheel for flash-attn 2.8.3), uv silently falls through to source build. Default `MAX_JOBS` on a 24-core box spawns ~12 nvcc workers, each peaking at 8–12 GB.

OOM kill after ~15 min. The kernel killed VS Code (highest oom_score) — flash-attn build itself wasn't directly killed but the global OOM stalled progress. Same trap had bitten 12 h earlier on a different attempt.

Journal evidence:

```
May 03 18:45:48 kernel: ... invoked oom-killer: ... order=0
May 03 18:45:48 kernel: Out of memory: Killed process 122839 (code) total-vm:1463376288kB
```

### Attempt 4 — memory-capped flash-attn source build (WORKS, 2.5 h)

```
systemd-run --user --scope -p MemoryMax=40G \
  env FLASH_ATTENTION_FORCE_BUILD=TRUE TORCH_CUDA_ARCH_LIST=12.0 \
      MAX_JOBS=2 NVCC_THREADS=1 \
      CUDA_HOME=/usr/local/cuda-12.8 \
      PATH=/usr/local/cuda-12.8/bin:/home/newub/.local/bin:/usr/bin:/bin \
      VIRTUAL_ENV=/home/newub/w/PersonaLive/.venv \
  /home/newub/.local/bin/uv pip install flash-attn --no-build-isolation
```

Notes:

- `systemd-run --user --scope -p MemoryMax=40G` is the protective wrapper. If RAM exceeds the cap, oomd kills *the build cgroup*, not whoever's biggest in the system (which last time was VS Code).
- `MAX_JOBS=2` and `NVCC_THREADS=1` cap concurrency. Empirically RAM peaks at ~37 GB during the busiest phases — under cap with margin.
- `env` resets PATH, so the absolute uv path matters. First attempt failed with `/usr/bin/env: 'uv': No such file or directory` because PATH only had cuda + system bins.
- `TORCH_CUDA_ARCH_LIST=12.0` is honored *only for `compute_120` codegen* — flash-attn's `setup.py` unconditionally adds sm_80/90/100, so you get a 4-arch build. This is the dominant cost driver: 72 object files instead of ~18.
- Total elapsed: ~2 h 30 min wall clock. 35–37 GB peak RAM. Final wheel: ~2 GB.

Outcome: `flash_attn 2.8.3` import works, `flash_attn_func` produces correct output.

But we're not done — see Attempt 5 below.

### Attempt 5 — re-run PersonaLive smoke after flash-attn install (DISCOVERS SECONDARY BREAKAGE)

```
MEDIAPIPE_DISABLE_GPU=1 .venv/bin/python inference_offline.py \
  --reference_image demo/ref_image.png --driving_video demo/driving_video.mp4 \
  --seed 42 -L 100 --name personalive_flashattn
```

Result: same 7–8 FPS as SDPA baseline, **xformers still failing to enable** — but with a different error:

```
Failed to enable xformers: module 'xformers.ops' has no attribute 'memory_efficient_attention'
```

Cause: the broken local source-build from Attempt 2 (`xformers==0.0.35+<git-hash>.d<date>`) was still installed. The published wheel's `xformers.ops.memory_efficient_attention` Python dispatcher would have found and called `flash_attn_func` automatically — but the broken local build had replaced that dispatcher with nothing.

### Attempt 6 — reinstall published xformers wheel over the broken build (FIX)

```
VIRTUAL_ENV=.venv uv pip install --reinstall xformers --index-url https://download.pytorch.org/whl/cu128
# replaces 0.0.35+<git-hash>.d<date> with 0.0.35
```

Verify:

```
python -c "import xformers.ops; print(hasattr(xformers.ops, 'memory_efficient_attention'))"
# True
```

This is the missing piece. After this, the published xformers dispatcher is back, and it should pick up the locally-installed flash-attn at call time.

## Final working setup (Linux RTX 5090)

```
torch==2.11.0+cu128
xformers==0.0.35              # published wheel from cu128 channel — NEVER source-build
flash-attn==2.8.3             # source-built locally with MAX_JOBS=2 + MemoryMax=40G
mediapipe==0.10.20            # 0.10.11 hits glibc tpp.c assertion; 0.10.35 drops solutions namespace
```

Required env at runtime:

```
MEDIAPIPE_DISABLE_GPU=1
```

## Final working setup (Windows RTX 3090)

```
torch==2.7.1+cu126            # whatever was already there
flash-attn==2.8.3+cu128torch2.7.0cxx11abiFALSE   # prebuilt wheel from kingbri1/flash-attention
```

Install command:

```
pip install https://github.com/kingbri1/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu128torch2.7.0cxx11abiFALSE-cp310-cp310-win_amd64.whl
```

cu128 wheel works on cu126 torch (CUDA runtime forward-compat). 2 minutes total.

## Lessons / rules

- **Never source-build xformers ≥ 0.0.35 expecting kernels.** The kernels aren't in the source tree. The build is structurally a no-op for fmha.
- **`uv/pip install --no-build-isolation` is not "wheel-only".** If no wheel matches, it source-builds with no concurrency limits. Always pre-check wheel availability with `uv pip download flash-attn --no-deps -d /tmp/check` if in doubt.
- **Wrap every Blackwell native-extension build in `systemd-run --user --scope -p MemoryMax=40G`.** This applies generally, not just to flash-attn — anything with `nvcc` × parallelism × Blackwell can OOM the box. (Already-existing rule from `feedback_eval_memory_cap.md` extended to compilation.)
- **`env` clears PATH.** When wrapping with `env VAR=...` always also pass `PATH=...` and use absolute paths to launchers (uv, python, nvcc).
- **If you ever did source-build xformers, the broken local build will sit over the wheel.** Recovery: `uv pip install --reinstall xformers --index-url https://download.pytorch.org/whl/cu128`. Then verify `hasattr(xformers.ops, 'memory_efficient_attention')`.
- **For Windows + Ampere/Ada, always reach for a kingbri1 prebuilt wheel before building anything.** cu128 wheels work on cu126 torch.
- **For Blackwell, accept the 2.5 h source build** — there's no faster legal path right now. The cost amortizes across PersonaLive, ComfyUI, and any other diffusion stack on this box.

## ComfyUI implication

ComfyUI on the same box should now also pick up flash-attn through xformers (provided ComfyUI's venv shares this xformers + flash-attn install, or ComfyUI is pointed at PersonaLive's venv). Alternative: launch ComfyUI with `--use-pytorch-cross-attention` to bypass xformers entirely and use SDPA directly. The pytorch SDPA path on Blackwell already dispatches to FlashAttention-2 internally for fp16/bf16 — so the bypass might be the cleanest answer for ComfyUI specifically.

## Final benchmark — xformers buys ~10% on this workload, not worth the maintenance cost

After Attempt 6 cleared the install, we hit a **third trap**: diffusers' `set_use_memory_efficient_attention_xformers` runs an enable-time probe at `attention_processor.py:269`:

```python
_ = xformers.ops.memory_efficient_attention(
    torch.randn((1, 2, 40), device="cuda"),  # default fp32!
    torch.randn((1, 2, 40), device="cuda"),
    torch.randn((1, 2, 40), device="cuda"),
)
```

The probe inputs are fp32 by default. fa2 (the only available backend on Blackwell now that cutlass-blackwell is unavailable in the wheel) accepts only fp16/bf16. So the probe always fails, the wrapper raises, and PersonaLive's try/except falls through to SDPA — even though fa2 *would* be selected at fp16 inference time.

Patch: edit the probe to pass `dtype=torch.float16`. After that:

| Run | xformers enable | pipe() time (100 frames) | FPS |
|-----|-----------------|--------------------------|-----|
| Baseline (SDPA, broken xformers) | failed | 13.7 s | 7.30 |
| flash-attn installed, broken xformers | failed (`no attribute mea`) | 12.23 s | 8.18 |
| flash-attn installed, wheel reinstalled, probe patched | enabled on 2D unet; failed on 3D unet (different probe shape) | 12.28 s | 8.14 |

**~10% over baseline at best — within run-to-run noise.** Conclusion: SDPA on torch 2.11+cu128 / Blackwell already dispatches to FA-2 internally for fp16. xformers + flash-attn buys nothing meaningful on top of that. Not worth maintaining the patched-diffusers + custom-built flash-attn install for ≤10%.

The 3D denoising_unet's xformers enable still fails after the probe patch — the temporal-window probe shape isn't fa2-compatible. Even fixing it would only buy us another fraction of a percent, since the 2D ref unet is the lighter of the two paths.

## What this means for the next optimization step

The TensorRT path (`pretrained_weights/tensorrt/` ships with PersonaLive) is now the only meaningful lever. PersonaLive's repo includes `requirements_trt.txt`, prebuilt TRT engines under `pretrained_weights/tensorrt/`, and a TRT-aware inference script. The paper's claimed 15–20 FPS on RTX 4090 was almost certainly with TRT engaged — our Blackwell box should match or beat that.

Action item: stop touching xformers/flash-attn. Move to TRT. Keep the flash-attn install (useful for any other workload that uses raw `flash_attn_func`) but don't fight the diffusers wrapper any further.

## Sources

- [xformers PR #1254 — capability 120](https://github.com/facebookresearch/xformers/pull/1254)
- [xformers PR #1262 — Blackwell support](https://github.com/facebookresearch/xformers/pull/1262)
- [xformers PR #1285 — precompiled wheels](https://github.com/facebookresearch/xformers/pull/1285)
- [xformers issue #1356 — capability (12,0) too new](https://github.com/facebookresearch/xformers/issues/1356)
- [PyTorch issue #164342 — sm_120 in stable](https://github.com/pytorch/pytorch/issues/164342)
- [flash-attention issue #1987 — Blackwell support](https://github.com/Dao-AILab/flash-attention/issues/1987)
- [kingbri1/flash-attention releases](https://github.com/kingbri1/flash-attention/releases) — Windows wheels for cu128torch2.7
- xformers `setup.py` line 225 — unconditional `TORCH_STABLE_ONLY`
