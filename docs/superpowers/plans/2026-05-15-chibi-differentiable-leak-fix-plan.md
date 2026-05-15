# Chibi Differentiable-Render Leak-Fix Spike — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hand-tuned chibi lid-boost with a differentiable-render optimization that finds the lid scale-ratio and eyeball opacity-multiplier which close the iris-through-lid leak.

**Architecture:** LAM's `infer_single_view` is differentiable end-to-end (CUDA rasterizer retains screen-space grad). We add an in-process injection point to LAM's hook block so an optimizer can apply grad-carrying per-vertex parameters to the canonical splat set, render ~4 ARKit frames (2 closed-eye, 2 open-eye), compute a leak + identity + regularization loss, and Adam-optimize. The result is two `.npy` assets consumed unchanged by the existing `LAM_CHIBI_SCALE_RATIO` / `LAM_OPACITY_MUL_NPY` env hooks for the verdict render.

**Tech Stack:** Python 3.12, PyTorch, LAM (`conda env lam`), `diff_gaussian_rasterization`, numpy, pytest.

**Key paths (absolute):**
- LAM repo: `/home/newub/w/LAM` — conda env bin `/home/newub/miniconda3/envs/lam/bin`
- vamp repo: `/home/newub/w/vamp-interface` (`$VAMP` below)
- anchor: `$VAMP/exp_output/lam_chibi/user_anchor/me.png`
- chibi assets (s=1.5): `$VAMP/exp_output/lam_chibi/me_s1.5/` — `chibi_textured_mesh.obj`, `chibi_arkit_bs.npy`, `chibi_scale_ratio.npy`
- lid mask: `$VAMP/exp_output/lam_chibi/me_s2.0/lid_mask_20018.npy` (int64 vertex indices, 1662 entries, deformation-invariant — reused as-is)
- magenta eyeball obj: `$VAMP/exp_output/lam_chibi/me/me_textured_mesh_display_eyemagenta.obj`
- motion: `$VAMP/exp_output/lam_bakeoff/take_runs_arkit_600/MySlate_2_arkit/export/MySlate_2/` — `flame_param/NNNNN.npz` ×600, each with `expr (1,52)`
- checkpoint: `$LAM/model_zoo/lam_models/releases/lam/lam-20k/step_045500/`
- config: `$LAM/configs/inference/lam-20k-8gpu.yaml`

**Run environment for every script:** `PATH` prepended with the lam conda env, `PYTHONPATH=$LAM`, `XFORMERS_DISABLED=1`, `LAM_USE_ARKIT=1`. Wrap the optimization run in `systemd-run --user --scope -p MemoryMax=45G` (eval-memory-cap rule).

---

## File Structure

- `scripts/chibi_leak_loss.py` — pure, framework-only loss functions (`magenta_score`, `leak_loss`, `identity_loss`, `ratio_reg`). No LAM imports. Unit-tested.
- `tests/test_chibi_leak_loss.py` — pytest unit tests for the above.
- `scripts/chibi_select_blink_frames.py` — reads the 600 `flame_param` npz files, picks 2 max-blink + 2 min-blink frame indices, writes `frames.json`.
- `/home/newub/w/LAM/lam/models/modeling_lam.py` — MODIFY: add an in-process injection point to the hook block.
- `scripts/chibi_diff_leak_opt.py` — optimization driver: builds LAM, runs the Adam loop, saves `chibi_scale_ratio_opt.npy` + `opacity_mul_opt.npy` + `loss_curve.png`.
- `scripts/chibi_diff_leak_verdict.sh` — full-take verdict render with the optimized assets, magenta + normal, vs the best-α baseline.

---

## Task 1: In-process injection hook in LAM + differentiability smoke test

This is the design's flagged unknown — resolve it first. Goal: prove the optimizer can apply a grad-carrying tensor to the canonical splats and that `infer_single_view` returns a rendered image with a live gradient back to that tensor.

**Files:**
- Modify: `/home/newub/w/LAM/lam/models/modeling_lam.py` (hook block, ~line 545–595, the `LAM_CHIBI_SCALE_RATIO` region)
- Create: `scripts/chibi_inproc_smoke.py`

- [ ] **Step 1: Add the in-process injection point.** In `modeling_lam.py`, immediately before the `LAM_CHIBI_SCALE_RATIO` env-var block, insert:

```python
        # --- vamp-interface in-process injection (differentiable optimization) ---
        # An optimizer process sets module-global tensors on this module before
        # each forward; if present they are applied with grad intact, taking
        # precedence over the env-var .npy path below. Verdict renders leave
        # these unset and use the env vars.
        import lam.models.modeling_lam as _self_mod
        _inproc_ratio = getattr(_self_mod, "_VAMP_INPROC_SCALE_RATIO", None)
        _inproc_opa = getattr(_self_mod, "_VAMP_INPROC_OPACITY_MUL", None)
        if _inproc_ratio is not None:
            for _gm in gs_model_list:
                _r = _inproc_ratio.to(_gm.scaling.device)
                if _r.shape[0] != _gm.scaling.shape[0]:
                    raise ValueError(
                        f"_VAMP_INPROC_SCALE_RATIO len {_r.shape[0]} != "
                        f"splat count {_gm.scaling.shape[0]}")
                _gm.scaling = _gm.scaling * _r.unsqueeze(-1)
        if _inproc_opa is not None:
            for _gm in gs_model_list:
                _o = _inproc_opa.to(_gm.opacity.device)
                _gm.opacity = (_gm.opacity * _o.reshape(_gm.opacity.shape)).clamp(0.0, 1.0)
        # --- end in-process injection ---
```

  Place it so it runs *after* `LAM_EDIT_XYZ_OBJ` (chibi mesh already applied) and *before* the env-var `LAM_CHIBI_SCALE_RATIO` block. Do not remove the env-var block — verdict renders depend on it.

- [ ] **Step 2: Write the smoke script.** Create `scripts/chibi_inproc_smoke.py` that builds the LAM inferrer (reuse `lam/runners/infer/lam.py:LAMInferrer` construction — load config `lam-20k-8gpu.yaml`, checkpoint `step_045500`), sets `lam.models.modeling_lam._VAMP_INPROC_SCALE_RATIO` to a leaf tensor `torch.ones(20018, requires_grad=True)`, runs `infer_single_view` for a single frame, takes `out["comp_rgb"].sum()`, calls `.backward()`, and asserts the leaf tensor's `.grad` is not None and non-zero. Print `grad.abs().mean()`.

- [ ] **Step 3: Run the smoke test.**

Run: `cd $LAM && systemd-run --user --scope -p MemoryMax=45G env PATH=/home/newub/miniconda3/envs/lam/bin:$PATH PYTHONPATH=$LAM XFORMERS_DISABLED=1 LAM_USE_ARKIT=1 python $VAMP/scripts/chibi_inproc_smoke.py`
Expected: prints a non-zero `grad.abs().mean()`; exits 0. If `.grad` is None, the graph is broken — escalate (the design's flagged unknown failed; the spike cannot proceed without it).

- [ ] **Step 4: Record the per-step wall time** printed by the smoke script (one forward+backward). This sets the optimization budget — if >3 s/step, note it; Task 4 caps step count accordingly.

- [ ] **Step 5: Commit.**

```bash
cd $VAMP && git add scripts/chibi_inproc_smoke.py && git -C /home/newub/w/LAM add lam/models/modeling_lam.py
cd $VAMP && git commit -m "feat(chibi): in-process injection hook + differentiability smoke"
```
(Note: the LAM repo is separate — commit it in its own repo: `git -C /home/newub/w/LAM commit -m "feat: vamp in-process splat injection point"`.)

---

## Task 2: Loss functions module

Pure tensor functions, no LAM dependency, fully unit-testable.

**Files:**
- Create: `scripts/chibi_leak_loss.py`
- Test: `tests/test_chibi_leak_loss.py`

- [ ] **Step 1: Write the failing test.** Create `tests/test_chibi_leak_loss.py`:

```python
import torch
from scripts.chibi_leak_loss import magenta_score, leak_loss, identity_loss, ratio_reg


def test_magenta_score_zero_on_gray():
    img = torch.full((1, 3, 8, 8), 0.5)          # neutral gray
    assert magenta_score(img).item() < 1e-6


def test_magenta_score_high_on_magenta():
    img = torch.zeros((1, 3, 8, 8))
    img[:, 0], img[:, 2] = 1.0, 1.0               # R+B high, G zero
    assert magenta_score(img).item() > 0.9


def test_leak_loss_uses_bbox_only():
    img = torch.zeros((1, 3, 10, 10))
    img[:, 0], img[:, 2] = 1.0, 1.0               # whole image magenta
    bbox = (0, 5, 0, 5)                           # top-left quadrant
    full = leak_loss(img, (0, 10, 0, 10))
    quad = leak_loss(img, bbox)
    assert torch.allclose(full, quad)             # uniform → bbox-invariant magnitude
    assert quad.item() > 0.9


def test_identity_loss_zero_when_equal():
    a = torch.rand(1, 3, 8, 8)
    assert identity_loss(a, a, (0, 8, 0, 8)).item() < 1e-6


def test_ratio_reg_zero_at_baseline():
    base = torch.rand(100)
    assert ratio_reg(base, base).item() < 1e-6
```

- [ ] **Step 2: Run it, verify it fails.**

Run: `cd $VAMP && python -m pytest tests/test_chibi_leak_loss.py -v`
Expected: FAIL — `ModuleNotFoundError: scripts.chibi_leak_loss`.

- [ ] **Step 3: Implement the module.** Create `scripts/chibi_leak_loss.py`:

```python
"""Differentiable loss terms for the chibi iris-leak optimization.

All functions take image tensors shaped [B, 3, H, W] in [0,1] and bounding
boxes as (y0, y1, x0, x1). No LAM dependency — pure torch.
"""
import torch


def magenta_score(img: torch.Tensor) -> torch.Tensor:
    """Mean magenta-ness over the whole tensor: relu(min(R,B) - G), in [0,1]."""
    r, g, b = img[:, 0], img[:, 1], img[:, 2]
    return torch.relu(torch.minimum(r, b) - g).mean()


def leak_loss(img: torch.Tensor, bbox) -> torch.Tensor:
    """Magenta-ness inside the eye bounding box (the visible-eyeball leak)."""
    y0, y1, x0, x1 = bbox
    return magenta_score(img[:, :, y0:y1, x0:x1])


def identity_loss(img: torch.Tensor, baseline: torch.Tensor, bbox) -> torch.Tensor:
    """L1 between render and un-optimized baseline inside the eye bbox.

    Guardrail: stops the optimizer from 'solving' the leak by sealing the lid
    or erasing the eyeball when the eye is open.
    """
    y0, y1, x0, x1 = bbox
    return (img[:, :, y0:y1, x0:x1] - baseline[:, :, y0:y1, x0:x1]).abs().mean()


def ratio_reg(ratio: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
    """L2 pull toward the v2 hand-set asset (GaussianAvatars scaling-loss analogue)."""
    return ((ratio - baseline) ** 2).mean()
```

- [ ] **Step 4: Run the tests, verify pass.**

Run: `cd $VAMP && python -m pytest tests/test_chibi_leak_loss.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit.**

```bash
cd $VAMP && git add scripts/chibi_leak_loss.py tests/test_chibi_leak_loss.py
git commit -m "feat(chibi): differentiable leak/identity/reg loss functions"
```

---

## Task 3: Blink-frame selection

Pick the driving frames the optimization renders: 2 fully-closed, 2 fully-open.

**Files:**
- Create: `scripts/chibi_select_blink_frames.py`

- [ ] **Step 1: Write the selection script.** Create `scripts/chibi_select_blink_frames.py`:

```python
"""Pick 2 max-blink (closed) and 2 min-blink (open) frames from an ARKit take.

ARKit-52 standard order: index 0 = eyeBlinkLeft, 7 = eyeBlinkRight.
Writes frames.json: {"closed": [i, j], "open": [k, l]}.
Run after selecting: verify the 'closed' frames actually show shut eyes by
eyeballing the verdict render — see Task 5.
"""
import json
import os
import sys
import numpy as np

EYE_BLINK_L, EYE_BLINK_R = 0, 7


def main(motion_dir: str, out_path: str):
    fp_dir = os.path.join(motion_dir, "flame_param")
    files = sorted(f for f in os.listdir(fp_dir) if f.endswith(".npz"))
    blink = np.zeros(len(files), dtype=np.float64)
    for i, f in enumerate(files):
        expr = np.load(os.path.join(fp_dir, f))["expr"].reshape(-1)
        blink[i] = expr[EYE_BLINK_L] + expr[EYE_BLINK_R]
    closed = sorted(np.argsort(blink)[-2:].tolist())
    open_ = sorted(np.argsort(blink)[:2].tolist())
    out = {"closed": closed, "open": open_,
           "blink_closed": blink[closed].tolist(),
           "blink_open": blink[open_].tolist()}
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"closed frames {closed} blink={blink[closed]} | "
          f"open frames {open_} blink={blink[open_]}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
```

- [ ] **Step 2: Run it.**

Run: `cd $VAMP && python scripts/chibi_select_blink_frames.py exp_output/lam_bakeoff/take_runs_arkit_600/MySlate_2_arkit/export/MySlate_2 exp_output/lam_chibi/diff_leak/frames.json`
Expected: prints 4 frame indices; `blink_closed` values clearly larger than `blink_open` (closed should be near the take's max eyeBlink, open near 0). If closed and open blink values are similar, the take has no real blink — escalate (wrong take).

- [ ] **Step 3: Commit.**

```bash
cd $VAMP && git add scripts/chibi_select_blink_frames.py exp_output/lam_chibi/diff_leak/frames.json
git commit -m "feat(chibi): blink-frame selector for diff-leak optimization"
```

---

## Task 4: Optimization driver

The Adam loop. Builds LAM once, optimizes `scale_ratio` (lid mask) + `opacity_mul` (eyeball mask), saves assets.

**Files:**
- Create: `scripts/chibi_diff_leak_opt.py`

- [ ] **Step 1: Write the driver skeleton — model build + base capture.** Create `scripts/chibi_diff_leak_opt.py`. It must:
  - Build the LAM inferrer exactly as `lam/runners/infer/lam.py` does (config `lam-20k-8gpu.yaml`, checkpoint `step_045500/`), reusing that file's input-preprocessing helpers (`prepare_motion_seqs`, the anchor-image pipeline) so the motion/camera tensors match the render path bit-for-bit.
  - Set env so the chibi mesh is applied: `LAM_EDIT_XYZ_OBJ=me_s1.5/chibi_textured_mesh.obj`, `LAM_CHIBI_ARKIT_BS=me_s1.5/chibi_arkit_bs.npy`, `LAM_USE_ARKIT=1`. Do **not** set `LAM_CHIBI_SCALE_RATIO` (the optimizer supplies scaling in-process).
  - Restrict the motion to the 4 indices in `frames.json`.
  - Run `infer_single_view` once with magenta eyeballs (`LAM_EDIT_VERTEX_COLORS_OBJ=me/me_textured_mesh_display_eyemagenta.obj`) and **no** in-process params → this render is the per-frame `baseline` for `identity_loss`. Detach and keep it.

- [ ] **Step 2: Define the optimizable parameters.** Append to the driver:

```python
import numpy as np, torch
# lid mask: vertex indices whose scaling the optimizer may change
lid = torch.as_tensor(np.load(f"{VAMP}/exp_output/lam_chibi/me_s2.0/lid_mask_20018.npy")).long()
# eyeball mask: vertices painted magenta in the eyemagenta obj (parsed once, by colour)
# eye_mask is computed in Step 3.
N = 20018
v2_ratio = torch.as_tensor(np.load(f"{VAMP}/exp_output/lam_chibi/me_s1.5/chibi_scale_ratio.npy")).float()

# free params: log-ratio on lid verts (so ratio>0), opacity logit-delta on eye verts
log_ratio_lid = torch.zeros(lid.shape[0], requires_grad=True)   # exp(0)=1 → starts at v2
opa_delta_eye = torch.zeros(0, requires_grad=True)              # sized in Step 3
opt = torch.optim.Adam([log_ratio_lid, opa_delta_eye], lr=0.05)
```

  The full per-vertex scale-ratio passed in-process is `v2_ratio` with `v2_ratio[lid] *= exp(log_ratio_lid)`. The opacity multiplier is `ones(N)` with `mul[eye] = sigmoid(opa_delta_eye)` clamped to `(0, 1]` — eyeball opacity can only be reduced (`fix_opacity`), which is the valid lever per Exp C.

- [ ] **Step 3: Build the eyeball mask.** Parse `me/me_textured_mesh_display_eyemagenta.obj` with the existing `_parse_vamp_obj` helper (in `modeling_lam.py`, or copy its few lines); the eyeball vertices are those whose vertex colour is the magenta sentinel `(1, 0, 1)` within tolerance. Save indices to `exp_output/lam_chibi/diff_leak/eye_mask_20018.npy` and size `opa_delta_eye` to `len(eye_mask)`.

- [ ] **Step 4: Write the optimization loop.** Append:

```python
from scripts.chibi_leak_loss import leak_loss, identity_loss, ratio_reg
import lam.models.modeling_lam as mlam

LAMBDA_ID, LAMBDA_REG = 1.0, 0.1     # tune after first run; see Step 6
BBOX = (eye_y0, eye_y1, eye_x0, eye_x1)   # eye-region bbox in render pixels, set in Step 5
losses = []
for step in range(N_STEPS):
    opt.zero_grad()
    ratio = v2_ratio.clone()
    ratio[lid] = ratio[lid] * torch.exp(log_ratio_lid)
    opa = torch.ones(N)
    opa[eye_mask] = torch.sigmoid(opa_delta_eye).clamp(1e-3, 1.0)
    mlam._VAMP_INPROC_SCALE_RATIO = ratio.cuda()
    mlam._VAMP_INPROC_OPACITY_MUL = opa.cuda()
    out = inferrer.model.infer_single_view(*infer_args)   # 4 frames
    rgb = out["comp_rgb"].permute(0, 3, 1, 2)             # [4, 3, H, W]
    closed = rgb[closed_local_idx]
    open_ = rgb[open_local_idx]
    L_leak = leak_loss(closed, BBOX)
    L_id = identity_loss(open_, baseline[open_local_idx], BBOX)
    L_reg = ratio_reg(torch.exp(log_ratio_lid), torch.ones_like(log_ratio_lid))
    loss = L_leak + LAMBDA_ID * L_id + LAMBDA_REG * L_reg
    loss.backward()
    opt.step()
    losses.append((L_leak.item(), L_id.item(), L_reg.item()))
    print(f"[{step:03d}] leak={L_leak:.4f} id={L_id:.4f} reg={L_reg:.4f}")
mlam._VAMP_INPROC_SCALE_RATIO = None      # clear so later renders use env vars
mlam._VAMP_INPROC_OPACITY_MUL = None
```

  `N_STEPS` defaults to 300; cap lower if Task 1 Step 4 measured >3 s/step (keep total run ≤30 min).

- [ ] **Step 5: Set the eye bbox.** Before the loop, render frame 0 once, save it to `exp_output/lam_chibi/diff_leak/bbox_probe.png`, open it, and read off a generous bounding box around both eyes in pixel coords. Hard-code `BBOX` from that. (One-time manual step — the eyes don't move far in screen space across this take.)

- [ ] **Step 6: First run — tune lambdas.**

Run: `cd $LAM && systemd-run --user --scope -p MemoryMax=45G env PATH=/home/newub/miniconda3/envs/lam/bin:$PATH PYTHONPATH=$LAM XFORMERS_DISABLED=1 LAM_USE_ARKIT=1 python $VAMP/scripts/chibi_diff_leak_opt.py`
Expected: `leak` decreases monotonically-ish; `id` stays small (<~0.02). If `id` climbs (open eye degrading) raise `LAMBDA_ID`; if `leak` plateaus high while `reg` is near zero, lower `LAMBDA_REG`. Re-run until `leak` drops at least 5× from step 0 or plateaus.

- [ ] **Step 7: Save the assets.** After the loop, write the full-length assets the verdict render consumes:

```python
final_ratio = v2_ratio.clone()
final_ratio[lid] = final_ratio[lid] * torch.exp(log_ratio_lid).detach()
np.save(f"{VAMP}/exp_output/lam_chibi/diff_leak/chibi_scale_ratio_opt.npy",
        final_ratio.cpu().numpy().astype(np.float32))
final_opa = np.ones(N, dtype=np.float32)
final_opa[eye_mask.numpy()] = torch.sigmoid(opa_delta_eye).detach().clamp(1e-3, 1.0).cpu().numpy()
np.save(f"{VAMP}/exp_output/lam_chibi/diff_leak/opacity_mul_opt.npy", final_opa)
# loss curve
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
arr = np.array(losses)
for j, name in enumerate(["leak", "id", "reg"]):
    plt.plot(arr[:, j], label=name)
plt.legend(); plt.savefig(f"{VAMP}/exp_output/lam_chibi/diff_leak/loss_curve.png")
```

- [ ] **Step 8: Commit.**

```bash
cd $VAMP && git add scripts/chibi_diff_leak_opt.py
git commit -m "feat(chibi): differentiable leak-fix optimization driver"
```

---

## Task 5: Verdict render + spike write-up

Render the full take with the optimized assets through the unchanged env-var path, compare against the best hand-tuned α.

**Files:**
- Create: `scripts/chibi_diff_leak_verdict.sh`

- [ ] **Step 1: Write the verdict render script.** Create `scripts/chibi_diff_leak_verdict.sh` modeled on `_lid_boost_video.sh`: it renders the full `MySlate_2` take twice with the optimized assets — once magenta, once normal — by setting `LAM_CHIBI_SCALE_RATIO=$VAMP/exp_output/lam_chibi/diff_leak/chibi_scale_ratio_opt.npy` and `LAM_OPACITY_MUL_NPY=$VAMP/exp_output/lam_chibi/diff_leak/opacity_mul_opt.npy` (env-var path, no in-process injection), and copies the outputs to `exp_output/lam_chibi/renders/diff_leak/opt_{magenta,normal}.mp4`.

- [ ] **Step 2: Run the verdict render.**

Run: `cd $LAM && bash $VAMP/scripts/chibi_diff_leak_verdict.sh`
Expected: two mp4s written.

- [ ] **Step 3: Build the side-by-side.** `hstack` the optimized normal render against the current best hand-tuned α render (`exp_output/lam_chibi/renders/alpha_sweep/s1.5_a0.50.mp4`) into `exp_output/lam_chibi/renders/diff_leak/opt_vs_alpha.mp4` with ffmpeg, labels baked in.

- [ ] **Step 4: Verify the closed frames truly closed.** Extract the `closed` frame indices (from `frames.json`) from `opt_magenta.mp4` and confirm no magenta is visible — this also retroactively confirms Task 3's ARKit blink-index assumption. If the "closed" frames are not actually closed, the selector picked wrong channels — fix `EYE_BLINK_L/R` indices in Task 3 and re-run from Task 4.

- [ ] **Step 5: Write the spike verdict.** Create `docs/research/2026-05-15-chibi-diff-leak-spike.md` stating: did the loop converge (final vs initial `leak`), did the open-eye guardrail hold (`id` curve), how the optimized render compares to best-α visually, and the decision — either "differentiable optimization closes the leak, retire the α-sweep" or "residual leak remains → quantitative evidence that aux-splat densification (approach C) is required." Add frontmatter (`status: live`, `topic: lam-chibi-recipe`). Update `docs/research/_topics/lam-chibi-recipe.md` in the same commit.

- [ ] **Step 6: Commit.**

```bash
cd $VAMP && git add scripts/chibi_diff_leak_verdict.sh docs/research/2026-05-15-chibi-diff-leak-spike.md docs/research/_topics/lam-chibi-recipe.md
git commit -m "feat(chibi): diff-leak verdict render + spike write-up"
```

---

## Automated Verification

When all tasks complete:
- `cd $VAMP && python -m pytest tests/test_chibi_leak_loss.py -v` → 5 passed.
- `chibi_inproc_smoke.py` exits 0 with non-zero gradient (Task 1).
- `chibi_diff_leak_opt.py` produces `chibi_scale_ratio_opt.npy`, `opacity_mul_opt.npy`, `loss_curve.png` (Task 4).
- Verdict mp4s exist and the closed-frame check passes (Task 5).

There is no full-suite regression here — the spike adds only new scripts. The decisive verification is the spike verdict doc (Task 5 Step 5): a converged loss curve + a clean (or quantified-residual) verdict render.

---

## Notes for the implementer

- The two repos are separate: `modeling_lam.py` lives in `/home/newub/w/LAM` and is committed there; everything else is in `/home/newub/w/vamp-interface`.
- Do not delete or alter the existing env-var hook blocks — verdict renders and every prior chibi script depend on them. The in-process injection is strictly additive and takes precedence only when its module globals are set.
- After Task 4, `_VAMP_INPROC_SCALE_RATIO` / `_VAMP_INPROC_OPACITY_MUL` must be set back to `None` (loop end) so the verdict render in Task 5 uses the env-var path cleanly.
- Per the run-code-review rule: before declaring the spike done, the new scripts go through `superpowers:code-reviewer`.
- Approach C (aux-splat densification) is deliberately out of scope — it is the follow-on if Task 5 finds a residual leak.
