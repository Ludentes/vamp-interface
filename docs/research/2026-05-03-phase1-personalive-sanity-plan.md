---
status: live
topic: neural-deformation-control
---

# Phase 1 plan: PersonaLive sanity + motion-API discovery

**Date:** 2026-05-03
**Box:** local Linux (RTX 4090). Windows machine reserved for the Phase 3 corpus extraction it unblocks.
**Parent:** `2026-05-03-iphone-pipeline-unified-plan.md` §Phase 1.

## Goal

Two deliverables in one pass:

1. **Sanity:** PersonaLive runs end-to-end on a Flux Krea portrait at usable speed and acceptable identity preservation against the bundled driver clip.
2. **API surface capture:** dump the shape/dtype/layout of the two candidate bridge targets (`MotionExtractor` output, `MotEncoder` output) to a json, so the Windows extraction script can be written without further local probing.

The second deliverable is what makes this a one-shot rather than a "run, then re-instrument later" task.

## Context — what we found in `inference_offline.py`

Two candidate motion representations. Bridge MLP target is one of these:

- **`MotionExtractor(num_kp=21)`** from `src.liveportrait.motion_extractor` — the LivePortrait-derived 21-implicit-3D-kp + rotation + scale + expression-delta + translation descriptor. Geometrically meaningful, ARKit-shaped.
- **`MotEncoder()`** from `src.models.motion_encoder.encoder` — encodes raw motion further into the actual conditioning latent fed to the denoising UNet. Opaque, but it's what the model actually consumes.

Pick lives at the end of Phase 1 — see *Decision* below.

## Inputs

- **Demo pair (smoke test):** `/home/newub/w/PersonaLive/demo/ref_image.png` + `demo/driving_video.mp4`. Confirms install runs.
- **Flux portrait pair (real test):** one Flux Krea portrait from our anchor set (pick one of the three vamp-interface anchors used in the slider corpus). Same demo driver video.
- **Driver video:** keep `demo/driving_video.mp4` for both runs to isolate the source-photo variable.

## Steps

### Step 1 — smoke test (demo pair)

Goal: confirm install + weights are wired correctly before swapping the source photo.

```bash
cd /home/newub/w/PersonaLive
.venv/bin/python inference_offline.py \
  --reference_image demo/ref_image.png \
  --driving_video demo/driving_video.mp4 \
  --seed 42 -L 100
```

Wrap in `systemd-run --user --scope -p MemoryMax=45G` per `feedback_eval_memory_cap`.

Log: wall time, peak VRAM (`nvidia-smi --loop=2 > /tmp/smi.log &`), output path under `results/`.

Pass: video file produced, no NaN, demo face moves coherently with driver. Eyeball.

### Step 2 — Flux portrait test (real test)

Same CLI, swap `--reference_image` for a Flux Krea anchor. Pick one with neutral expression, frontal pose, no glasses (drivability easiest). Suggested: whichever anchor `compose_iterative` currently uses as the smile-axis base.

Log: same as Step 1, plus visual inspection of identity preservation across frames (eyeball ArcFace-equivalent — does it still look like the same person at frame 50, 100?).

Pass gates (from unified plan §Phase 1):
- Runs without OOM on 4090 (24 GB).
- ≥ ~10 FPS sustained (the paper claims 15–20 FPS on this hardware without CFG; allow slop).
- Identity stays recognisable across the 100-frame clip — ArcFace cosine to source ≥ 0.6 at frame 100. (This is a soft eyeball pass for now; Phase 2 makes it quantitative.)

Fail → log the failure mode (OOM / slow / identity drift / artefact) and stop. The decision tree (unified plan) routes to AdvancedLivePortrait fallback.

### Step 3 — motion-API probe

Independent of pass/fail of Steps 1–2, instrument a one-shot probe to capture the bridge-target API.

Write a small standalone script `tools/probe_motion_api.py` (in the PersonaLive checkout, not in vamp-interface — stays close to the imports). Pseudocode:

```python
import torch, json
from src.liveportrait.motion_extractor import MotionExtractor
from src.models.motion_encoder.encoder import MotEncoder
from PIL import Image
import numpy as np

device = "cuda"
dummy_frame = torch.randn(1, 3, 256, 256, device=device, dtype=torch.float16)

me = MotionExtractor(num_kp=21).to(device, torch.float16).eval()
with torch.no_grad():
    me_out = me(dummy_frame)  # likely a dict of tensors

mot_enc = MotEncoder().to(device, torch.float16).eval()
# MotEncoder input shape unknown — probe by trying the typical wiring used in inference_offline.py;
# fall back to printing the forward signature via inspect.signature if a guess fails.

def describe(x, prefix=""):
    if isinstance(x, dict):
        return {k: describe(v, prefix + k + ".") for k, v in x.items()}
    if isinstance(x, torch.Tensor):
        return {"shape": list(x.shape), "dtype": str(x.dtype)}
    if isinstance(x, (list, tuple)):
        return [describe(v, prefix) for v in x]
    return str(type(x))

api = {
    "MotionExtractor": describe(me_out),
    "MotEncoder": describe(mot_enc_out),  # filled in after wiring
    "weights_loaded_from": ...,            # paths from config
}
json.dump(api, open("/home/newub/w/vamp-interface/docs/research/_data/personalive_motion_api.json", "w"), indent=2)
```

Save the json under `docs/research/_data/` so it's checked in and the Windows session can grep it.

Pass: json file written; both targets have concrete shape/dtype recorded; `MotEncoder` input wiring is documented in a comment.

### Step 4 — bridge-target decision

Given the two candidate targets, decide which the Phase 3 bridge MLP regresses to. Decision rubric:

- **Target = `MotionExtractor` output** if: it's a low-d structured tuple (≤ a few hundred dims), it's the same descriptor format LivePortrait uses (so we can also drive AdvancedLivePortrait from the same bridge), and it's downstream-stable (PersonaLive consumes it deterministically).
- **Target = `MotEncoder` output** if: `MotionExtractor` output requires nontrivial post-processing before conditioning, OR if the conditioning latent is much lower-d / smoother and easier for an MLP to learn.

Default lean: **`MotionExtractor` output**. Rationale: it's the documented LivePortrait-style descriptor, it's portable to ALP/non-PersonaLive backends (preserves optionality if PersonaLive disappoints in Phase 2), and it's geometrically interpretable (which makes per-channel weighting in the MLP loss tractable). Override only if the API probe reveals it's a giant blob or carries identity entanglement we don't want to inherit.

Document the decision + reasoning in this file's *Outcome* section once Phase 1 is done.

## Pre-conditions checklist

- [ ] `/home/newub/w/PersonaLive/.venv` exists and pip-installs cleanly (verified prior session).
- [ ] Model weights present under `pretrained_weights/` (don't enumerate; the script will yell on first missing file).
- [ ] One Flux Krea anchor PNG copied or symlinked into the PersonaLive checkout for Step 2.
- [ ] `docs/research/_data/` directory exists (mkdir if not).
- [ ] `nvidia-smi` baseline VRAM noted before launch.

## Outputs

- `results/<timestamp>_demo/` — demo smoke video (PersonaLive's default).
- `results/<timestamp>_flux/` — Flux anchor video.
- `docs/research/_data/personalive_motion_api.json` — bridge API surface.
- This doc, with *Outcome* section filled in.

## Outcome (Step 1 done 2026-05-03)

### Demo smoke pass

- **pipe() time: 13.7 s for 100 frames** (steady-state 2.82 it/s, 28 chunks of frames given `temporal_window_size=4`)
- Effective rate: **~7.3 frames/s output**
- Total wall time including model load: 31.8 s
- Peak VRAM: 8.5 GB
- Output: `docs/research/_data/personalive_phase1/smoke_demo_{split,concat}.mp4` (archived from `/home/newub/w/PersonaLive/results/`)
- Full run log: `docs/research/_data/personalive_phase1/smoke_run.log`
- Pinned env state: `docs/research/_data/personalive_phase1/env_versions.json`

### Perf framing

Paper claim: 15-20 FPS on RTX 4090. We get 7.3 FPS on RTX 5090 with the unoptimized SDPA fp32 path → roughly **40% of paper, 0.3× real-time** (output target = 25 FPS).

For Phase 3 corpus extraction (10K frames) this rate gives ~24 min batch time — fine for a one-time job. **For the live iPhone use case (Phase 6) we need ≥25 FPS, i.e. a ≥3.4× speedup.** Optimization path queued separately — see *Optimization roadmap* below.

### Env setup gotchas (reproducibility)

The PersonaLive venv as shipped does not run on Blackwell (sm_120). Three fixes required, in order:

1. **Upgrade torch + torchvision + xformers to cu128 wheels.** Pinned `torch==2.1.0+cu121` has no sm_120 kernels; bumped via `VIRTUAL_ENV=...venv uv pip install --upgrade torch torchvision xformers --index-url https://download.pytorch.org/whl/cu128`. Resulting versions: torch 2.11.0+cu128, torchvision 0.26.0+cu128, xformers 0.0.35, CUDA 12.8.
2. **Upgrade scikit-image** for numpy 2.x ABI: `uv pip install --upgrade scikit-image` (gets 0.25.2). Required because torch upgrade pulled numpy 2.2.6 transiently (later pinned back to 1.26.4 by mediapipe — see step 3).
3. **Downgrade mediapipe to 0.10.20.** The shipped 0.10.11 hits a glibc 2.41+ pthread tpp assertion (`__pthread_tpp_change_priority`) that aborts the process at FaceMesh constructor. The latest 0.10.35 fixes the assertion but **drops the `solutions` namespace entirely** which PersonaLive uses. 0.10.20 is the sweet spot: has both `solutions` and the glibc fix. Side effect: pins numpy back to 1.26.4 (good, fewer ABI worries).
4. **Set `MEDIAPIPE_DISABLE_GPU=1`** before launch. Even on 0.10.20 the GPU-delegate path can deadlock on Blackwell (44 threads in `futex_do_wait`); CPU path runs at 150+ FPS for 256² FaceMesh, more than fast enough.
5. **xformers 0.0.35 fails to enable** on capability (12,0) — the wheel was built with `TORCH_CUDA_ARCH_LIST` capping at 9.0, so cutlass/fa3 backends silently reject sm_120. PersonaLive's try/except catches this and falls through to PyTorch SDPA. PersonaLive's config also uses fp32, which means even a Blackwell-built xformers would only activate the cutlass backend (not fa2/fa3). Both addressed in *Optimization roadmap*.

### Optimization roadmap (for the live use case)

Goal: 7.3 FPS → ≥25 FPS = **3.4× speedup**. Stack, in priority order:

1. ~~**Switch PersonaLive config to fp16.**~~ **Already done — `configs/prompts/personalive_offline.yaml` ships with `weight_dtype: 'fp16'`.** The 7.3 FPS baseline is fp16 SDPA, not fp32. The `(1, 2, 1, 40) torch.float32` tensor in the xformers error trace was misread initially — it's an internal probe at `.enable_xformers_memory_efficient_attention()` time, not an inference tensor. No action needed here.
2. ~~**Source-build xformers 0.0.35.**~~ **Falsified 2026-05-03.** xformers main has structurally removed its CUDA fmha kernels — only sparse24 remains in `csrc/`. A from-source build produces an xformers package with no `memory_efficient_attention_*` backends at all. See [`2026-05-03-xformers-blackwell-pivot.md`](2026-05-03-xformers-blackwell-pivot.md) for full analysis. **Replacement: install `flash-attn` separately** (xformers' Python dispatcher will pick it up at call time), or rely on PyTorch SDPA's own FlashAttention-2 dispatch on torch 2.11+cu128. Both bypass the xformers kernel question.
3. **TensorRT path.** PersonaLive ships `pretrained_weights/tensorrt/`, `requirements_trt.txt`, `torch2trt.py`. The paper's 15-20 FPS number was almost certainly TRT, not eager PyTorch. NVIDIA-native, best Blackwell performance achievable. 1-2 h plumbing for engine builds. This is the path that gets us cleanly past the paper claim into ≥25 FPS territory.

Test order: 2 → measure → 3 → measure. If 2 clears 25 FPS, 3 becomes nice-to-have rather than required.

### Bridge target choice

Deferred to Step 3 (motion-API probe). Step 1 confirms the runtime; Step 3 will dump `MotionExtractor` and `MotEncoder` shapes so we can decide the bridge MLP target.

### Notes / surprises

- The "200× slower than realtime" extrapolation we made mid-debug was wrong — that was the FaceMesh deadlock holding the GPU idle, not actual denoising throughput. Real denoising is 2.82 it/s, well within striking distance of paper.
- Five rounds of bash-debugging required (smoke1 → smoke8) to clear all the env issues. Worth documenting them above so the office Windows 5090 setup hits each fix in order.
- xformers 0.0.35 shows `dtype=fp32` in its rejection trace — confirms PersonaLive's default precision and explains why fa2/fa3 wouldn't help even if they were Blackwell-compatible. The fp16 switch in step 1 of the roadmap is therefore load-bearing for steps 2-3 to pay off.

## What this unblocks

- **Phase 2** (recipe grid) — one CLI verified, multiply by 3 anchors × 6 drivers.
- **Phase 3 corpus extraction (Windows)** — once the bridge target shape/dtype is known, the extraction script is a tight loop: `for frame in driver_video: m = motion_extractor(frame); b = mediapipe(frame); pkl.append((b, m))`. This is the script we hand to the Windows machine.

## Falsification

- **OOM on 4090:** PersonaLive's claim that it fits in 12 GB is wrong for our config; reduce `-L`, drop xformers, or fall back to AdvancedLivePortrait-only path.
- **Identity collapse on Flux anchor:** PersonaLive doesn't generalise to the Flux distribution out of the box. Phase 2 grid would have caught this anyway; Phase 1 catches it cheaper.
- **Both motion targets are giant opaque blobs:** the bridge MLP problem becomes harder than expected; reconsider whether to regress to a downstream layer or use a flow-matching head instead of MLP.
