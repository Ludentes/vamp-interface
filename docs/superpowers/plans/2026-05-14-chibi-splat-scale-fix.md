# Chibi Splat-Scale Fix — Implementation Plan

**Goal:** Make Gaussian splat sizes follow chibi mesh stretch so (a) the eyelid sheet tiles densely enough to occlude the iris during blink, and (b) the eyeball surface tiles densely enough that the iris reads at full proportional size — not a small dot inside a stretched FLAME socket.

**Architecture:** Two-stage hook, mirroring the existing chibi pipeline (`LAM_CHIBI_ARKIT_BS` for basis swap at FLAME init, `LAM_EDIT_XYZ_OBJ` for canonical-xyz swap before animation):

- **v1** — `LAM_CHIBI_SCALE_BOOST` env var, uniform log-space additive on `_gm.scaling`. Sanity test: confirms splat density is the dominant cause vs LBS joint mismatch. ~10 lines of code, no asset rebake.
- **v2** — `LAM_CHIBI_SCALE_RATIO` env var pointing at a per-vertex `(20018,) float32` array. Computed in `chibi_make_assets.py` as the ratio of mean 1-ring edge length (chibi / original) on the 20018-vert subdivided template. Applied at the same hook point as v1; broadcast across all 3 scale axes (isotropic).
- **v3 (deferred)** — anisotropic tangent-plane 2×2 covariance ratio, only if v2 leaves residual artifacts.

**Tech stack:** Python 3, NumPy, PyTorch, pytorch3d (already pulled in by chibi_make_assets.py via SubdivideMeshes). All changes live in `scripts/chibi_make_assets.py` (vamp-interface) and `LAM/lam/models/modeling_lam.py` (sibling LAM repo).

**Hypothesis being tested:** The s=0→1→2 strength sweep at frame 599 shows continuous degradation of blink closure with chibi strength. The Gaussian scales are predicted by a learned GSLayer head at canonical FLAME density, NOT derived from the mesh we override in `_gm.xyz`. Therefore chibi-stretched verts carry original-size splats, leaving sparsity-driven holes proportional to stretch magnitude. v1 falsifies/confirms this in one render; v2 is the proper fix.

---

## Task 1: v1 — uniform `LAM_CHIBI_SCALE_BOOST` hook

**Files:**
- Modify: `/home/newub/w/LAM/lam/models/modeling_lam.py` around line 537 (the existing `LAM_EDIT_XYZ_OBJ` block)

- [ ] **Step 1.1: Read the existing override block to lock in exact indentation & variable name**

Run: `sed -n '480,560p' /home/newub/w/LAM/lam/models/modeling_lam.py`

Expected: see the `LAM_EDIT_XYZ_OBJ` env-var read, the OBJ-load helper, and the `_gm.xyz = ...` assignment. Note the surrounding indent (likely 8 or 12 spaces).

- [ ] **Step 1.2: Add `LAM_CHIBI_SCALE_BOOST` block right after the xyz override**

Insert immediately after `_gm.xyz = chibi_xyz` (or whatever the assignment variable is):

```python
# v1 splat-scale boost — uniform log-space additive.
# `_gm.scaling` is stored in log-space (linear scale = exp(_gm.scaling)),
# so multiplying linear scale by `boost` == adding log(boost) here.
chibi_scale_boost = float(os.environ.get("LAM_CHIBI_SCALE_BOOST", "1.0"))
if chibi_scale_boost != 1.0:
    _gm.scaling = _gm.scaling + math.log(chibi_scale_boost)
```

- [ ] **Step 1.3: Verify `math` is in scope at this file**

Run: `grep -n '^import math\|^from math' /home/newub/w/LAM/lam/models/modeling_lam.py`

If missing, add `import math` at the top of the file in the existing stdlib import block.

- [ ] **Step 1.4: Sanity render at s=2.0 frame 599 with BOOST=1.5**

```bash
cd /home/newub/w/vamp-interface
LAM_CHIBI_SCALE_BOOST=1.5 bash scripts/chibi_eye_diag.sh
```

Then rename the produced `eye_s2.0_b1.0.png` (or current single-variant output) to `eye_s2.0_boost1.5.png` so it doesn't get overwritten.

Expected: image-left eye more closed at frame 599 than the baseline `eye_s2.0_nopin.png`. If yes → v1 confirms hypothesis, proceed to v2. If no → splat density is NOT the dominant cause; revisit LBS-joint-mismatch hypothesis before continuing.

- [ ] **Step 1.5: Boost sweep**

Re-render at BOOST ∈ {1.25, 1.5, 2.0}. Build a strip:

```bash
montage eye_s2.0_nopin.png eye_s2.0_boost1.25.png eye_s2.0_boost1.5.png eye_s2.0_boost2.0.png \
  -tile 4x1 -geometry +4+4 -label '%t' eye_boost_sweep.png
```

Expected: a monotone increase in lid closure with boost. Note the boost value where it visually closes — this is the rough magnitude v2 needs to deliver locally on lid verts.

- [ ] **Step 1.6: Commit v1**

```bash
cd /home/newub/w/LAM
git add lam/models/modeling_lam.py
git commit -m "feat(chibi): LAM_CHIBI_SCALE_BOOST env var for uniform splat-scale compensation"
```

(LAM is a sibling repo; commit there. The eye-diag PNGs live under vamp-interface and don't need committing.)

---

## Task 2: v2 — per-vertex edge-length ratio asset

**Files:**
- Modify: `/home/newub/w/vamp-interface/scripts/chibi_make_assets.py` — add `per_vert_edge_mean()` + emit `chibi_scale_ratio.npy`

- [ ] **Step 2.1: Locate the chibi xyz construction**

Run: `grep -n 'chibi_xyz\|subdivide\|20018\|edges' /home/newub/w/vamp-interface/scripts/chibi_make_assets.py`

Locate where the 20018-vert chibi mesh is finalized AND where `edges20018` is already available (we computed it during `lift_mask_to_subdivided`). Verify whether the 20018 ORIGINAL template (pre-chibi) is also still in scope at that point. If not, reload it from the template OBJ + SubdivideMeshes pass that produced the 20018 layout.

- [ ] **Step 2.2: Add `per_vert_edge_mean` helper**

Add near the other geometry helpers (after `lift_mask_to_subdivided`):

```python
def per_vert_edge_mean(verts: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Mean 1-ring edge length per vertex.

    verts: (N, 3) float
    edges: (E, 2) int64
    returns: (N,) float64 — mean over incident edges; 0 for orphans.
    """
    e_len = np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1)
    sums = np.zeros(verts.shape[0], dtype=np.float64)
    counts = np.zeros(verts.shape[0], dtype=np.float64)
    np.add.at(sums, edges[:, 0], e_len)
    np.add.at(counts, edges[:, 0], 1)
    np.add.at(sums, edges[:, 1], e_len)
    np.add.at(counts, edges[:, 1], 1)
    return sums / np.maximum(counts, 1.0)
```

- [ ] **Step 2.3: Emit `chibi_scale_ratio.npy` alongside the existing chibi outputs**

Right after the chibi 20018 verts are finalized (and before the OBJ is written), add:

```python
el_orig  = per_vert_edge_mean(tpl20018_orig,  edges20018)
el_chibi = per_vert_edge_mean(tpl20018_chibi, edges20018)
scale_ratio = (el_chibi / np.maximum(el_orig, 1e-8)).astype(np.float32)
np.save(outdir / "chibi_scale_ratio.npy", scale_ratio)
print(f"[chibi] scale_ratio: min={scale_ratio.min():.3f} median={np.median(scale_ratio):.3f} max={scale_ratio.max():.3f}")
```

Variable names (`tpl20018_orig`, `tpl20018_chibi`, `edges20018`) must match the actual names in the existing code — Step 2.1 confirms these.

- [ ] **Step 2.4: Rebake assets for `me` at s=1.0 and s=2.0**

```bash
cd /home/newub/w/vamp-interface
rm -f exp_output/lam_chibi/me/chibi_scale_ratio.npy
rm -f exp_output/lam_chibi/me_s1.0/chibi_scale_ratio.npy
# (rebuild only the npy + obj; do NOT delete the directory, the baked OBJ has the original mesh we still want)
bash scripts/chibi_make_assets.py \
    --template /home/newub/w/LAM/model_zoo/human_parametric_models/flame_assets/flame/head_template_mesh.obj \
    --arkit_bs /home/newub/w/LAM/model_zoo/human_parametric_models/flame_assets/flame_arkit_bs.npy \
    --baked    exp_output/lam_chibi/me/me_textured_mesh_display.obj \
    --outdir   exp_output/lam_chibi/me \
    --y_anchor_frac_baked 0.20 \
    --chibi_strength 2.0
# repeat for me_s1.0/ with --chibi_strength 1.0
```

Verify: `ls -la exp_output/lam_chibi/me*/chibi_scale_ratio.npy` shows two new files; printed median is roughly in `[1.0, 1.5]` for s=2.0 and `[1.0, 1.2]` for s=1.0.

- [ ] **Step 2.5: Commit v2 asset emitter**

```bash
cd /home/newub/w/vamp-interface
git add scripts/chibi_make_assets.py
git commit -m "feat(chibi): emit per-vertex chibi_scale_ratio.npy (edge-length ratio over 20018)"
```

---

## Task 3: v2 — `LAM_CHIBI_SCALE_RATIO` hook in LAM

**Files:**
- Modify: `/home/newub/w/LAM/lam/models/modeling_lam.py` — extend the v1 block

- [ ] **Step 3.1: Replace the v1 block with v1+v2 combined**

Replace the v1 block from Task 1.2 with:

```python
# v1 splat-scale boost — uniform log-space additive (sanity / fallback knob)
chibi_scale_boost = float(os.environ.get("LAM_CHIBI_SCALE_BOOST", "1.0"))
if chibi_scale_boost != 1.0:
    _gm.scaling = _gm.scaling + math.log(chibi_scale_boost)

# v2 per-vertex isotropic splat-scale ratio (matches chibi mesh stretch locally)
ratio_path = os.environ.get("LAM_CHIBI_SCALE_RATIO", "")
if ratio_path and os.path.isfile(ratio_path):
    ratio_np = np.load(ratio_path).astype(np.float32)          # (N,)
    assert ratio_np.shape[0] == _gm.scaling.shape[0], \
        f"scale_ratio length {ratio_np.shape[0]} != splat count {_gm.scaling.shape[0]}"
    ratio_t = torch.from_numpy(np.log(np.maximum(ratio_np, 1e-8))).to(_gm.scaling.device, dtype=_gm.scaling.dtype)
    _gm.scaling = _gm.scaling + ratio_t.unsqueeze(-1)          # broadcast over 3 axes
```

- [ ] **Step 3.2: Verify `np` and `torch` are in scope; add if missing**

Run: `grep -n '^import numpy\|^import torch' /home/newub/w/LAM/lam/models/modeling_lam.py`

Both should already be imported (they're used elsewhere in this file). If not, add to existing import block.

- [ ] **Step 3.3: Wire the env var in `chibi_eye_diag.sh`**

Edit `/home/newub/w/vamp-interface/scripts/chibi_eye_diag.sh` — after the existing `export LAM_EDIT_XYZ_OBJ=...` line, add:

```bash
export LAM_CHIBI_SCALE_RATIO=${ASSET_DIR}/chibi_scale_ratio.npy
```

Also add a guard so an older asset dir (no ratio file) silently disables v2 instead of failing:

```bash
if [[ ! -f "${ASSET_DIR}/chibi_scale_ratio.npy" ]]; then
  unset LAM_CHIBI_SCALE_RATIO
fi
```

- [ ] **Step 3.4: Render frame 599 with v2 only (no v1 boost)**

```bash
cd /home/newub/w/vamp-interface
unset LAM_CHIBI_SCALE_BOOST
bash scripts/chibi_eye_diag.sh
mv exp_output/lam_chibi/renders/frames_me/eye_diag/eye_s2.0_b1.0.png \
   exp_output/lam_chibi/renders/frames_me/eye_diag/eye_s2.0_v2ratio.png
```

Expected: image-left eye fully closed at frame 599; iris no longer poking through. Iris on the open right eye should look proportionally larger (no longer "tiny dot in big socket") than baseline `eye_s2.0_nopin.png`.

- [ ] **Step 3.5: s=1 sanity render**

```bash
LAM_CHIBI_ARKIT_BS=exp_output/lam_chibi/me_s1.0/chibi_arkit_bs.npy \
LAM_EDIT_XYZ_OBJ=exp_output/lam_chibi/me_s1.0/chibi_textured_mesh.obj \
LAM_CHIBI_SCALE_RATIO=exp_output/lam_chibi/me_s1.0/chibi_scale_ratio.npy \
bash scripts/chibi_render_with_assets.sh \
  exp_output/lam_chibi/user_anchor/me.png \
  exp_output/lam_chibi/me_s1.0 \
  s1.0_v2ratio
```

Expected: at s=1 chibi the iris also reads bigger (addresses the "eyes very small but eyelids large" feedback). Compare to the earlier `eye_strength_sweep.png` s=1 panel.

- [ ] **Step 3.6: Build verdict strip**

```bash
cd /home/newub/w/vamp-interface/exp_output/lam_chibi/renders/frames_me/eye_diag
montage eye_s0.0.png eye_s2.0_nopin.png eye_s2.0_boost1.5.png eye_s2.0_v2ratio.png \
  -tile 4x1 -geometry +4+4 \
  -title 's=0 baseline | s=2 no fix | s=2 v1 boost=1.5 | s=2 v2 per-vert ratio' \
  eye_fix_verdict.png
```

- [ ] **Step 3.7: Commit v2 hook**

```bash
cd /home/newub/w/LAM
git add lam/models/modeling_lam.py
git commit -m "feat(chibi): LAM_CHIBI_SCALE_RATIO for per-vertex splat-scale compensation"
cd /home/newub/w/vamp-interface
git add scripts/chibi_eye_diag.sh
git commit -m "feat(chibi): wire LAM_CHIBI_SCALE_RATIO into eye-diag harness"
```

---

## Task 4: Full-take render with v2 fix on `me` at s=2.0

- [ ] **Step 4.1: Run full 600-frame take 2 with all three hooks**

```bash
cd /home/newub/w/vamp-interface
LAM_CHIBI_SCALE_RATIO=exp_output/lam_chibi/me/chibi_scale_ratio.npy \
bash scripts/chibi_render_with_assets.sh \
  exp_output/lam_chibi/user_anchor/me.png \
  exp_output/lam_chibi/me \
  s2.0_v2ratio
```

- [ ] **Step 4.2: Compare to pre-fix full take side-by-side**

The pre-fix full take should already exist under `exp_output/lam_chibi/renders/`. Build a side-by-side mp4 (the existing `chibi_anchor_render.sh` already does this pattern at lines 113-122 — copy the ffmpeg drawtext+hstack invocation).

- [ ] **Step 4.3: Eye gate — visual review**

Open the side-by-side. Check three moments:
- Half-blink at frame 599 (down-gaze + eyeBlinkRight=0.86): lid should close cleanly.
- Full blink (any frame with both eyes shut): both eyes fully occluded.
- Wide-eye / eyeWide frames: iris should fill the socket, not look pinned.

If all three pass → v2 ships. If frame-599 still leaks → fall back to v1+v2 stacked (`LAM_CHIBI_SCALE_BOOST=1.15` on top of v2) and re-render before considering v3 anisotropic.

---

## Task 5: Documentation

- [ ] **Step 5.1: Add research note**

Create `docs/research/2026-05-14-chibi-splat-scale-fix.md` with frontmatter (status: live, topic: lam-chibi-recipe — create the topic file if not yet present), one-screen summary: the hypothesis (splat density frozen at canonical FLAME density), the falsification ladder we already ran (basis rescale, z-identity, row-identity all dead), the v1 confirmation render, the v2 fix, and a link to the verdict strip PNG.

Invoke the `frontmatter-tagger` agent on this doc as per CLAUDE.md rule.

- [ ] **Step 5.2: Update memory index**

Add one line to `~/.claude/projects/-home-newub-w-vamp-interface/memory/MEMORY.md` under a new `project_lam_chibi_splat_scale_fix.md` memory pointer once the verdict strip confirms it works.

- [ ] **Step 5.3: Update Task #37**

Mark task #37 (`Chibi-eye occlusion still broken on no-pin path`) completed once Step 4.3 passes. Unblocks Task #36 (write human-to-chibi-3dgs playbook), which should now mention the three env vars: `LAM_CHIBI_ARKIT_BS`, `LAM_EDIT_XYZ_OBJ`, `LAM_CHIBI_SCALE_RATIO`.

---

## Fallbacks

- **v1 boost fails to close the eye (Step 1.4):** splat density is not dominant. Don't proceed to v2. Pivot to LBS-joint-frame-mismatch hypothesis: the eyes_pose joint transforms in `flame_arkit.py:790-801` are computed from the canonical FLAME template, not the chibi mesh. Test by zeroing eye-pose joint contribution in a probe render.
- **v2 closes the eye but causes puffy cheeks / fat nose:** soft-taper `scale_ratio` toward 1.0 outside `eye_region ∪ left_eyeball ∪ right_eyeball` (lifted to 20018). Smooth taper, not a hard mask, to avoid seam splats.
- **v2 closes the eye but iris still feels small:** add `eye_boost` multiplier on `scale_ratio` for eyeball verts only — anime-style enlarged eyes. Knob lives in `chibi_make_assets.py`, no LAM changes.
- **v2 leaves residual streaks along stretched directions (anisotropy):** v3 — replace scalar ratio with 2×2 tangent-plane covariance ratio, applied to splat scale in the local frame. Defer until v2 ships and we see what's left.
