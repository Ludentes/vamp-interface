# CFM Expression-ControlNet — De-Risk Phase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** De-risk the CFM expression-ControlNet training run — prove a trainable bf16 InfuseNet runs one CFM step on diffusers FLUX, and fix the FLAME control-render alignment — before the full training run is planned.

**Architecture:** Two independent tasks. Task 0 is a *gating feasibility spike*: it discovers and records the concrete diffusers-FLUX + bf16-InfuseNet training API and proves one forward+backward CFM step. Task 1 fixes the render-fit calibration flagged by the render-cache verification (the FLAME mesh is fit by its whole-skull extent into a face-only bbox) by fitting only the FLAME `face`-region vertices.

**Tech Stack:** Python 3.12, diffusers FLUX.1-dev, peft (LoRA), bitsandbytes, PyTorch, NumPy, OpenCV. Spike runs under the project `uv` env (`uv run python ...`, same env as `src/demographic_pc/train_flux_image_slider.py`). Geometry code runs under `/home/newub/miniconda3/bin/python` with `PYTHONPATH=src`.

**Reference docs:**
- Spec: `docs/superpowers/specs/2026-05-19-cfm-expression-controlnet-design.md`
- Design analysis: `docs/research/2026-05-16-arkit-controlnet-infiniteyou.md`
- Render-cache design + verification: `docs/superpowers/specs/2026-05-18-cfm-render-cache-design.md`, `docs/research/2026-05-18-cfm-render-cache-verification.md`
- Existing FLUX flow-matching trainer (pattern to mirror): `src/demographic_pc/train_flux_image_slider.py`

**Outcome handoff:** When both tasks pass, the full training run (`cfm/model.py`, `precompute.py`, `dataset.py`, `train.py`, `eval.py`, render-modality A/B) gets a *second* plan, written against the concrete API recorded by Task 0. Do not extend this plan with training tasks.

---

### Task 0: Trainable-InfuseNet feasibility spike (GATING)

This is a **research spike**, not a TDD task. Its deliverable is a verdict: can a bf16 InfuseNet be loaded trainably on diffusers FLUX and complete one CFM step on the RTX 5090? If NO-GO, stop and escalate — do not proceed to Task 1's dependents or write the training-run plan.

**Files:**
- Create: `src/arkit_controlnet/cfm/__init__.py` (empty package marker)
- Create: `src/arkit_controlnet/cfm/spike_trainable_infusenet.py`
- Create: `docs/research/2026-05-19-trainable-infusenet-spike-verdict.md`

- [ ] **Step 1: Create the package marker**

```bash
mkdir -p src/arkit_controlnet/cfm
touch src/arkit_controlnet/cfm/__init__.py
```

- [ ] **Step 2: Confirm the uv training env has the needed deps**

The slider trainer documents its env in `src/demographic_pc/train_flux_image_slider.py` (`uv add diffusers peft bitsandbytes accelerate`). Verify:

Run: `uv run python -c "import diffusers, transformers, peft, bitsandbytes, torch; print(diffusers.__version__, transformers.__version__, peft.__version__, torch.__version__)"`
Expected: four version strings, no `ModuleNotFoundError`. If `peft` or `bitsandbytes` is missing, run `uv add peft bitsandbytes` and retry.

- [ ] **Step 3: Locate the bf16 InfiniteYou weights and the InfuseNet model class**

Research sub-step — record findings in the verdict doc as you go:

1. The weights on disk (`~/w/ComfyUI/models/infinite_you/sim_stage1/`, symlinked to `data/infiniteyou_dl/infu_flux_v1.0/sim_stage1/`) are a ComfyUI-converted **fp8** inference weight — not trainable. Locate the original **bf16** InfiniteYou release on HuggingFace (repo `ByteDance/InfiniteYou`, path `infu_flux_v1.0/sim_stage1/`). It contains a diffusers-format `InfuseNetModel/` directory plus `image_proj_model.bin`.
2. Download the bf16 `sim_stage1` weights with `huggingface-cli download ByteDance/InfiniteYou --include "infu_flux_v1.0/sim_stage1/*"` into `data/infiniteyou_dl_bf16/` (note: `data/` is gitignored — these are large weights, never commit them).
3. Find the InfuseNet model class. ByteDance publishes inference model code (the `InfiniteYou` GitHub repo, `pipelines/pipeline_infu_flux.py` and the `InfuseNetModel` definition). Determine: how `InfuseNetModel` is constructed / loaded (`from_pretrained`?), and how its forward injects residuals into the FLUX transformer during a generation step.

Record in the verdict doc, under a heading **"Concrete API"**: the exact load calls, the InfuseNet forward signature, and which arguments carry (a) the noised latent, (b) timestep, (c) text embeds, (d) the 8 identity tokens, (e) the control image. This section is the input to the training-run plan.

- [ ] **Step 4: Write the spike script**

Create `src/arkit_controlnet/cfm/spike_trainable_infusenet.py`. The CFM-step math below is fixed and correct (standard FLUX flow matching, mirrors `train_flux_image_slider.py`); the model-loading block is filled from Step 3's findings — that is the spike's discovery, not a placeholder.

```python
"""GATING feasibility spike: prove a trainable bf16 InfuseNet runs one CFM step.

Run: uv run python -m arkit_controlnet.cfm.spike_trainable_infusenet

Loads diffusers FLUX.1-dev (frozen) + bf16 InfuseNet, attaches a LoRA to
InfuseNet, marks its control-image input stem trainable, and runs one
conditional-flow-matching forward+backward step on a single synthetic pair.
Asserts the loss is finite and the trainable gradients are non-zero, then
prints peak VRAM. Exit code 0 = GO, non-zero = NO-GO.
"""
import sys
import torch

DEVICE = "cuda"
DTYPE = torch.bfloat16


def build_trainable_infusenet():
    """Load frozen FLUX + bf16 InfuseNet; attach a LoRA + trainable control
    stem; freeze everything else. Returns (flux, infusenet, trainable_params).

    Filled in from Task 0 Step 3 findings — the exact diffusers / InfiniteYou
    load calls. Contract this function must satisfy:
      - FLUX transformer + VAE + text encoders loaded, all requires_grad=False.
      - InfuseNet loaded in bf16; a peft LoRA attached to its DiT-copy blocks;
        its control-image input stem set requires_grad=True.
      - `trainable_params` is the non-empty list of params with requires_grad.
    """
    raise NotImplementedError("fill from Task 0 Step 3 findings")


def one_cfm_step(flux, infusenet, trainable_params):
    """One conditional-flow-matching forward+backward on a synthetic pair."""
    # Synthetic stand-ins sized like real FLUX-latent / conditioning tensors.
    # Replace the shapes with the real ones once Step 3 fixes them; keep the
    # flow-matching math exactly as below.
    z0 = torch.randn(1, 16, 64, 64, device=DEVICE, dtype=DTYPE)      # photo latent
    eps = torch.randn_like(z0)                                       # noise
    t = torch.sigmoid(torch.randn(1, device=DEVICE))                 # logit-normal
    shift = 3.0                                                      # FLUX sigma shift
    sigma = (shift * t) / (1 + (shift - 1) * t)
    s = sigma.view(-1, 1, 1, 1).to(DTYPE)
    z_t = (1 - s) * z0 + s * eps
    target = eps - z0                                                # CFM velocity

    opt = torch.optim.AdamW(trainable_params, lr=1e-4)
    opt.zero_grad()
    v_pred = forward_velocity(flux, infusenet, z_t, sigma)           # see below
    loss = torch.nn.functional.mse_loss(v_pred.float(), target.float())
    loss.backward()
    grad_norm = torch.sqrt(sum((p.grad.float() ** 2).sum()
                               for p in trainable_params if p.grad is not None))
    opt.step()
    return loss.item(), grad_norm.item()


def forward_velocity(flux, infusenet, z_t, sigma):
    """Run frozen FLUX with InfuseNet residuals added; return predicted velocity.

    Filled in from Task 0 Step 3 findings (the residual-injection forward).
    """
    raise NotImplementedError("fill from Task 0 Step 3 findings")


def main() -> int:
    torch.cuda.reset_peak_memory_stats()
    flux, infusenet, trainable_params = build_trainable_infusenet()
    assert len(trainable_params) > 0, "no trainable parameters"
    loss, grad_norm = one_cfm_step(flux, infusenet, trainable_params)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"loss={loss:.5f}  grad_norm={grad_norm:.3e}  peak_vram={peak_gb:.1f}GB")

    import math
    if not math.isfinite(loss):
        print("NO-GO: loss is not finite"); return 1
    if grad_norm == 0.0 or not math.isfinite(grad_norm):
        print("NO-GO: trainable gradients are zero / non-finite"); return 1
    print("GO: one CFM step completed, finite loss, non-zero gradients")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run the spike**

Run: `uv run python -m arkit_controlnet.cfm.spike_trainable_infusenet`
Expected (GO): final line `GO: one CFM step completed, finite loss, non-zero gradients`, exit code 0, and `peak_vram` comfortably under 32 GB.
If NO-GO or it cannot be made to run: stop. Record the blocker in the verdict doc and escalate — the training-run plan cannot be written.

- [ ] **Step 6: Write the verdict doc**

Create `docs/research/2026-05-19-trainable-infusenet-spike-verdict.md` with frontmatter:

```markdown
---
status: live
topic: arkit-controlnet
---
```

Body (use descriptive headings, no section numbers — house style): a one-line GO/NO-GO verdict; the **"Concrete API"** section from Step 3 (load calls, InfuseNet forward signature, residual-injection call); measured `loss` / `grad_norm` / `peak_vram`; the bf16 weights path under `data/`; and any caveats for the training-run plan (e.g. whether gradient checkpointing was needed to fit 32 GB).

- [ ] **Step 7: Update the topic index**

Add a dated section to `docs/research/_topics/arkit-controlnet.md` recording the spike verdict and linking the verdict doc, mirroring the existing entries' style.

- [ ] **Step 8: Commit**

```bash
git add src/arkit_controlnet/cfm/ docs/research/2026-05-19-trainable-infusenet-spike-verdict.md docs/research/_topics/arkit-controlnet.md
git commit -m "feat(arkit-controlnet): trainable-InfuseNet feasibility spike

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 1: Render-fit calibration via FLAME face-mask

The render-cache verification flagged the FLAME mesh sitting high / oversized: `render()` fits the xy-extent of **all 5023 vertices** (whole skull) into `bbox`, but `bbox` is the MediaPipe 478-landmark **face-front** extent — so the head is shrunk to cram the skull into a face box. Fix: fit only the FLAME **`face`-region** vertices (1787 of them, from `FLAME_masks.pkl`) into `bbox`; still rasterize every face. This aligns the FLAME face region with the photo's face box.

**Files:**
- Modify: `src/arkit_controlnet/prep_flame_assets.py`
- Modify: `src/arkit_controlnet/flame_render.py`
- Test: `tests/arkit_controlnet/test_flame_render.py` (existing file — add tests)

- [ ] **Step 1: Write the failing tests**

Add to `tests/arkit_controlnet/test_flame_render.py`:

```python
def test_face_region_idx_loaded():
    """Assets carry the FLAME face-region vertex index list."""
    from arkit_controlnet.flame_render import load_flame_assets
    a = load_flame_assets()
    assert a.face_region_idx.shape == (1787,)
    assert a.face_region_idx.dtype == np.int32
    assert a.face_region_idx.min() >= 0
    assert a.face_region_idx.max() < 5023


def test_projection_fits_face_region_to_bbox():
    """The face-region vertices — not the whole skull — fill the bbox."""
    from arkit_controlnet.flame_render import (
        load_flame_assets, deform, _project)
    a = load_flame_assets()
    verts = deform(np.zeros(52))
    bbox = (0.5, 0.5, 0.4, 0.5)          # cx, cy, w, h normalized
    H = W = 512
    px = _project(verts, np.eye(3), bbox, H, W)
    face_px = px[a.face_region_idx]
    lo, hi = face_px.min(axis=0), face_px.max(axis=0)
    # face region spans the bbox to within 2% of its pixel size
    assert abs((hi[0] - lo[0]) - bbox[2] * W) < 0.02 * bbox[2] * W
    assert abs((hi[1] - lo[1]) - bbox[3] * H) < 0.02 * bbox[3] * H
    # the full skull extends ABOVE the face box (no longer squashed in)
    assert px[:, 1].min() < lo[1]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /home/newub/w/vamp-interface && PYTHONPATH=src /home/newub/miniconda3/bin/python -m pytest tests/arkit_controlnet/test_flame_render.py::test_face_region_idx_loaded tests/arkit_controlnet/test_flame_render.py::test_projection_fits_face_region_to_bbox -v`
Expected: both FAIL — `test_face_region_idx_loaded` with `AttributeError: 'FlameAssets' object has no attribute 'face_region_idx'`; `test_projection_fits_face_region_to_bbox` with `ImportError` / `cannot import name '_project'`.

- [ ] **Step 3: Extract the FLAME face-region indices in prep_flame_assets.py**

In `src/arkit_controlnet/prep_flame_assets.py`, add the masks path next to the other LAM paths (after the `BASIS_SRC` line, line 17):

```python
MASKS_SRC = LAM / "flame_assets" / "flame" / "FLAME_masks.pkl"
```

In `build()`, replace the `np.savez(...)` call (line 89) with:

```python
    with open(MASKS_SRC, "rb") as mf:
        masks = pickle.load(mf, encoding="latin1")
    face_region_idx = np.asarray(masks["face"], dtype=np.int32)
    assert face_region_idx.ndim == 1 and face_region_idx.max() < 5023, \
        face_region_idx.shape

    np.savez(OUT_DIR / "flame_base.npz", v_template=v_template, faces=faces,
             face_region_idx=face_region_idx)
```

And extend the final print (line 91-92) to mention the mask:

```python
    print(f"wrote {OUT_DIR / 'flame_base.npz'} — {len(v_template)} verts, "
          f"{len(faces)} faces, {len(face_region_idx)} face-region verts; "
          f"copied basis")
```

- [ ] **Step 4: Re-run the asset prep**

Run: `cd /home/newub/w/vamp-interface && PYTHONPATH=src /home/newub/miniconda3/bin/python -m arkit_controlnet.prep_flame_assets`
Expected: `wrote output/flame_assets/flame_base.npz — 5023 verts, 9976 faces, 1787 face-region verts; copied basis`

- [ ] **Step 5: Add face_region_idx to FlameAssets and the loader**

In `src/arkit_controlnet/flame_render.py`, add the field to `FlameAssets` (after the `arkit_basis` line):

```python
    face_region_idx: np.ndarray  # (1787,) int32 — FLAME `face` mask vertex idx
```

In `load_flame_assets()`, pass it through when constructing `FlameAssets`:

```python
        _assets = FlameAssets(v_template=d["v_template"], faces=d["faces"],
                              arkit_basis=np.load(basis),
                              face_region_idx=d["face_region_idx"])
```

- [ ] **Step 6: Extract `_project` and fit over the face region**

In `src/arkit_controlnet/flame_render.py`, add a `_project` helper before `render()`:

```python
def _project(verts: np.ndarray, rotation: np.ndarray,
             bbox: tuple[float, float, float, float],
             H: int, W: int) -> np.ndarray:
    """Rotate, orthographically project, and isotropically fit FLAME verts to
    the face bbox. The fit uses only the FLAME `face`-region vertices, so the
    photo's face box maps to the mesh face — not to the whole skull. Returns
    (N, 2) float pixel coordinates for every input vertex.
    """
    cx, cy, bw, bh = bbox
    verts = np.asarray(verts, dtype=np.float64)
    vr = verts @ np.asarray(rotation, dtype=np.float64).T
    xy = vr[:, :2].copy()
    xy[:, 1] *= -1.0                       # FLAME +Y up -> image +Y down

    a = load_flame_assets()
    face_xy = xy[a.face_region_idx]        # fit the face region, not the skull
    lo, hi = face_xy.min(axis=0), face_xy.max(axis=0)
    extent = np.maximum(hi - lo, 1e-9)
    scale = min(bw * W / extent[0], bh * H / extent[1])
    px = (xy - (lo + hi) / 2) * scale
    px[:, 0] += cx * W
    px[:, 1] += cy * H
    return px
```

Then rewrite the projection block inside `render()` — replace lines 113-123 (`# orthographic projection` through `pts = px.astype(np.int32)`) with:

```python
    a = load_flame_assets()
    faces = a.faces
    vr = verts @ np.asarray(rotation, dtype=np.float64).T   # rotate into camera
    pts = _project(verts, rotation, bbox, H, W).astype(np.int32)
```

(`vr` is still needed below for `_face_normals` and the painter's-order z-sort. The earlier `a = load_flame_assets(); faces = a.faces; vr = ...` lines at the top of the function — lines 110-112 — are now redundant with these; delete the original three so `a`, `faces`, `vr` are assigned exactly once.)

- [ ] **Step 7: Run the tests to verify they pass**

Run: `cd /home/newub/w/vamp-interface && PYTHONPATH=src /home/newub/miniconda3/bin/python -m pytest tests/arkit_controlnet/test_flame_render.py -v`
Expected: all tests PASS — the 2 new ones plus the existing render/deform/mapping tests (the existing `test_render_neutral_fills_bbox` still passes; the face region fills the bbox just as the whole mesh used to, only correctly scaled).

- [ ] **Step 8: Re-run the verification collage**

Run: `cd /home/newub/w/vamp-interface && PYTHONPATH=src /home/newub/miniconda3/bin/python -m arkit_controlnet.verify_flame_render`
Expected: writes `exp_output/flame_render_check/collage.png`. Eyeball it: the FLAME face should now sit on the photo face — no longer high / oversized.

- [ ] **Step 9: Update the verification doc**

In `docs/research/2026-05-18-cfm-render-cache-verification.md`, update the root-cause subsection that recorded the oversized-mesh imperfection: note it is now fixed by fitting the FLAME `face`-region mask (1787 verts) instead of the whole-skull extent, and that the re-run collage confirms alignment.

- [ ] **Step 10: Commit**

```bash
git add src/arkit_controlnet/prep_flame_assets.py src/arkit_controlnet/flame_render.py tests/arkit_controlnet/test_flame_render.py docs/research/2026-05-18-cfm-render-cache-verification.md
git commit -m "fix(arkit-controlnet): fit FLAME render by face-region mask, not whole skull

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:** The spec's "framework risk and the gating spike" → Task 0. The spec's "two early gates before the long run / render-fit calibration" → Task 1. The render-modality A/B and the five `cfm/*` training units are explicitly deferred to the post-Task-0 training-run plan (stated in the header) — that is the agreed two-plan split, not a gap.

**Placeholder scan:** Task 0 is a declared research spike; its two `raise NotImplementedError` stubs are the spike's discovery surface, with the exact contract each must satisfy written above them and the discovery procedure in Step 3 — not hand-waved implementation. All Task 1 steps carry complete code.

**Type consistency:** `face_region_idx` is (1787,) int32 throughout — added to the npz (Task 1 Step 3), to `FlameAssets` (Step 5), used in `_project` (Step 6) and tested (Step 1). `_project(verts, rotation, bbox, H, W) -> (N,2)` has one signature, used identically in `render()` and the test.
