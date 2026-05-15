# Chibi iris-leak: differentiable-render optimization spike — design

**Date:** 2026-05-15
**Status:** design, pre-implementation
**Related:** `docs/research/2026-05-15-gaussian-splat-editing-best-practices.md`, memory `project_chibi_iris_lid_mask_v3_falsification`

## Problem

The chibi deform leaks the eyeball through the closed lid. Every fix so far — v2 isotropic scale-ratio, v3 anisotropic J·SVD, the α×s lid-boost sweep — is a **hand-set asset**: a human picks a number, renders, eyeballs the result, repeats. The research note established (a) the leak is a splat-coverage/occlusion problem, not a covariance problem, and (b) LAM's output is an explicit splat set rendered by a *differentiable* CUDA rasterizer (`diff_gaussian_rasterization`, `gs_renderer.py:562`, `screenspace_points.retain_grad()`).

So we can stop hand-searching. Put the leak in a loss function and let gradient descent find the asset. This is **not** training the LAM network — the network stays frozen; we optimize a small per-vertex parameter applied to its explicit output, exactly the regime our existing `LAM_CHIBI_SCALE_RATIO` / `LAM_OPACITY_MUL_NPY` hooks already occupy. The optimizer produces the same kind of `.npy` asset; it just finds the numbers instead of a human.

## Approaches considered

**A — optimize opacity only.** Released LAM has `fix_opacity=True`; the `LAM_OPACITY_MUL_NPY` hook clamps multipliers >1 (memory). So we cannot make lids *more* opaque — only make the eyeball *less* opaque. Exp C confirmed dropping eyeball α reduces the leak, so an eyeball-opacity multiplier <1 is a valid, real lever, but on its own it makes the eyeball semi-transparent when the eye is open. Useful only as a *secondary* free parameter under a multi-frame loss that also protects the open eye.

**B — optimize the lid scale-ratio.** Same quantity v2/v3/the α-sweep set by hand, on the same ~1662-vertex lid mask. Gradients flow `ratio → _gm.scaling → animate(LBS) → rasterize → loss`. Reuses the `LAM_CHIBI_SCALE_RATIO` hook for the verdict render. Cheap, ~1662 params, zero new render-side code. Directly automates what the α-sweep brute-forces.

**C — optimize an aux-splat set (densification).** The literature-aligned fix (GSDeformer splits Gaussians; GaussianAvatars enforces ≥1 splat/triangle in over-occlusion regions). Stamp new splats on the lid, optimize their position/opacity/scale/SH. Most powerful, most code, and aux splats get *free* opacity (not `fix_opacity`-bound). This is the real fix — but it is a follow-on, not the spike.

**Decision.** Spike = **B + A together**: optimize a per-vertex lid scale-ratio plus a per-vertex eyeball-opacity multiplier (<1), under a multi-frame loss. Both feed existing hooks, so the only new code is one standalone optimization script plus a tiny in-process injection point in the hook block. The spike's job is to answer *"does the differentiable loop close the leak, and does it beat the hand α-sweep?"* If B+A converges clean, scope C as the follow-on plan. If B+A plateaus with residual leak, that is itself the evidence that densification (C) is mandatory — either outcome is decisive.

## Mechanism

**Frames.** From the ARKit take (`MySlate_2`), pick ~4 driving frames: 2 fully-closed (max `eyeBlink`) and 2 fully-open. The leak is judged on closed frames; the open frames are the guardrail.

**Render.** Eyeballs painted a fixed diagnostic magenta (the existing `_eyemagenta.obj`) so the leak is a colour the loss can isolate — magenta colour is a constant, gradients flow through geometry/opacity only.

**Loss.**
- *Leak* (closed frames): `mean(magenta-ness)` over the eye bounding box, where `magenta-ness = relu(min(R,B) − G)`. Differentiable; →0 when no eyeball shows.
- *Identity* (open frames): `L1(render, baseline_render)` over the eye bbox, baseline = the un-optimized chibi render. Stops the optimizer from solving the leak by shrinking lids shut or by making the eyeball invisible when open.
- *Regularization*: `L2(scale_ratio − v2_ratio)` + `L2(opacity_mul − 1)` — stay near the v2 asset; the GaussianAvatars scaling-loss analogue, also kills the large-`s` smear by penalizing runaway scale.
- Total: `leak + λ_id·identity + λ_reg·reg`, λ tuned on the first run.

**Optimizer.** Adam, a few hundred steps. One LAM forward (cached `_gm`); per step re-animate 4 frames + rasterize. Fast (renderer sustains ~310 fps). Minutes total. Wrap in `systemd-run --user --scope -p MemoryMax=45G` per the eval-memory-cap rule.

**Output.** `chibi_scale_ratio_opt.npy` + `opacity_mul_opt.npy`. The verdict full-take render consumes them through the unchanged `LAM_CHIBI_SCALE_RATIO` / `LAM_OPACITY_MUL_NPY` env hooks — no new render path.

## Code surface

- **New:** `scripts/chibi_diff_leak_opt.py` — builds LAM, runs `infer_single_view` once to obtain `_gm`, applies the chibi xyz edit, captures base scaling/opacity, then the Adam loop over the 4 frames; saves the two `.npy` assets and a loss-curve PNG.
- **Modify (small):** the hook block in `~/w/LAM/lam/models/modeling_lam.py` — add an in-process injection: if a module-level tensor is registered, use it (grad-carrying) instead of reading the env-var `.npy`. Keeps the env-var path intact for verdict renders.
- **Risk / flagged unknown:** factoring `infer_single_view` so the loop can run *forward → inject params → animate(frame) → render* in-process and keep the graph connected. `infer_single_view` is a torch model method so it is differentiable end-to-end; the unknown is purely plumbing. Spike task 1 resolves it before any optimization code is written.

## Acceptance

Spike is complete when we can state: whether the differentiable loop converges, the final leak-loss value vs. the best hand-tuned α render, whether the open-eye guardrail held, and — if a residual leak remains — quantitative evidence that aux-splat densification (C) is required. Verdict artifact: optimized vs. best-α full-take, side by side, magenta + normal.

## Self-review

- Placeholders: none — λ values are explicitly "tuned on first run", not left as TODO.
- Consistency: B and A both terminate in existing env hooks; output filenames fixed (`chibi_scale_ratio_opt.npy`, `opacity_mul_opt.npy`) and used identically in script and verdict render.
- Scope: single spike, single script + one small hook edit. C is explicitly deferred.
- Ambiguity: "magenta-ness" given a concrete differentiable formula; frame selection given a concrete count and criterion.
