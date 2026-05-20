---
status: live
topic: arkit-bridge
supersedes: 2026-05-13-lam-capabilities-inventory.md
---

# LAM Python live-ARKit spike — resolved positive

**Date:** 2026-05-13
**Spike question:** Can the released LAM-20K Python checkpoint be driven live by iPhone Live Link Face ARKit-52 weights, or does it require retraining as a previous note claimed?
**Answer:** **Yes, no retraining needed.** Two-line source change.

## Retraction

The earlier `2026-05-13-lam-capabilities-inventory.md` (and the MEMORY.md
🚨 REFRAMING entry) claimed Python and Web LAM are "different architectures"
and that Python live-ARKit requires retraining. **This was wrong on three
counts**, all of which I verified by reading the released code:

1. **`flame_arkit.py` exists in the released repo** (since commit `5c204d4`,
   the first LAM V1 commit, March 2025) and was never modified. It is **not**
   a separate architecture — it's a drop-in FLAME forward with the same
   `(shape, expr, rotation, neck, jaw, eyes, translation)` signature, where
   `expr` is 52-dim ARKit instead of 100-dim FLAME PCA. The class is dead-code
   in the released inference path, not because it doesn't work, but because
   the demo's `flame_tracking_video.py` is VHAP-based which emits FLAME PCA.

2. **The Gaussian net is identity-only.** Reading
   `lam/models/rendering/gs_renderer.py`'s `forward_gs_attr`+`animate_gs_model`
   pair: the trained network produces a per-vertex Gaussian cloud
   conditioned on (image, canonical-pose vertex positions). Canonical-pose
   vertices depend on shape β only — not expression or pose. At animation
   time, *only* `xyz` of the per-vertex Gaussians is moved (line 617-622:
   `xyz=mean_3d[i]`; opacity/rotation/scaling/SHs are reused unchanged from
   identity prediction). Expression enters only through the deterministic
   FLAME blendshape evaluator, which has no learned weights. **So the
   trained checkpoint cannot care whether expression came from PCA or ARKit
   — it never sees expression.**

3. **The S1 subspace-mismatch finding (ARKit-51 vs FLAME-PCA-100 overlap
   ~13%) doesn't gate anything.** It would matter if we were trying to
   project ARKit→PCA and feed the released `expr` slot (which I had
   considered as the implementation path). The simpler path is to swap the
   FLAME class itself, which makes the projection irrelevant. Both bases
   produce valid FLAME meshes; the trained Gaussian binding is mesh-relative
   and indifferent to which basis moved the mesh.

The "different architectures" framing came from an earlier comprehensive
inventory agent which conflated *driver format* (PCA vs ARKit) with
*model architecture* (same checkpoint, same Gaussian net). The packaging is
different (Web bakes ARKit deltas into glTF morph targets; Python computes
them at runtime via `flame_arkit_bs.npy`), but the runtime math is identical.

## Spike evidence

### Architecture read

- `gs_renderer.py:31`: imports `FlameHeadSubdivided` from `.flame` (PCA
  variant). Released inference path uses PCA only because of this one-line
  import.
- `gs_renderer.py:617-622`: per-frame Gaussian list construction only
  varies `xyz`; opacity/rotation/scaling/SHs/offset all identity-shared.
  Confirms Gaussian net is identity-only.
- `flame_arkit.py:566`: `forward()` signature identical to
  `flame.py:FlameHead.forward()`. Drop-in.
- `flame_arkit.py:108`: `assert expr_params != 52` — typo bug; intent was
  `==`. Bypassable by passing `expr_params=53` (the value is purely a label;
  `flame_arkit_bs.npy` is always 52-dim and `n_expr_params` doesn't gate
  the forward).

### Git history

- `flame_arkit.py` shipped in `5c204d4` (first LAM V1 commit, March 2025)
  as 1815 lines. **Never modified since.**
- `c2b3728` (April 30, 2025, "Release export feature for openavatarchat")
  added the Blender-side ARKit bake tools (`generateARKITGLBWithBlender.py`,
  `convertFBX2GLB.py`, `generateVertexIndices.py`) but did NOT touch
  `flame_arkit.py`. The .npy basis is consumed by *both* paths — Python
  runtime (if you switch the import) and Blender-side bake (in the released
  web export tool).
- No GitHub issue (of 104 open/closed) mentions Python-side ARKit
  inference. Nobody has tried this upstream; it's not "known broken,"
  it's "unwired by default."

### S1: basis-span test (irrelevant in hindsight, but documented)

ARKit-51 (one zero blendshape — `tongueOut`) vs FLAME-PCA-100 across the
5023-vertex displacement space:

- median per-PCA-dim reconstruction error from ARKit span: **0.87**
- max:                                                       0.96
- PCA dims well-explained by ARKit span (rel err < 0.1):     **0/100**

So the two bases span almost-disjoint 51-d and 100-d subspaces of the
15069-d vertex-displacement space. Crucially, **this only matters if you
try to project ARKit→PCA**. We don't have to. Drop-in the class.

### S2: mesh sanity (the load-bearing test)

Instantiated `FlameHead` (ARKit variant) with `expr_params=53` (bypass
typo'd assert), drove with synthetic ARKit-52 vectors, exported OBJ:

| ARKit drive            | L2 vs neutral | Lower-jaw mean y | Interpretation                |
|------------------------|--------------:|-----------------:|-------------------------------|
| neutral (zeros)        |         0.000 |          -0.0995 | baseline                      |
| `jawOpen=1.0`          |         0.619 |          -0.1092 | jaw drops ~10mm ✅            |
| FLAME `jaw_pose=0.3`   |         0.500 |          -0.1093 | reference PCA path same drop  |
| `smile=0.7` (L+R)      |         0.138 |          -0.0999 | corner pull, no jaw drop ✅   |
| `eyeBlink=1.0` (L+R)   |         0.086 |          -0.0995 | local eye motion ✅           |

Each blendshape produces physically correct, isolated displacement.
`jawOpen=1.0` even moves the jaw farther than FLAME's hand-coded jaw joint
at moderate drive levels — the ARKit basis is calibrated for full
expressivity.

## Implementation: two-line patch

```python
# lam/models/rendering/gs_renderer.py:31
# from lam.models.rendering.flame_model.flame import FlameHeadSubdivided
from lam.models.rendering.flame_model.flame_arkit import FlameHeadSubdivided

# lam/models/rendering/flame_model/flame_arkit.py:108
# assert expr_params != 52, "The dimension of the ARKIT expression must be equal to 52."
assert expr_params == 52, "The dimension of the ARKIT expression must be equal to 52."
```

Then the live driver feeds `flame_data["expr"]` as a 52-dim ARKit vector
per frame. Head pose: ARKit's `HeadYaw/Pitch/Roll` → FLAME's `neck` joint
via the closed-form Euler signs we already established for PersonaLive
([[project-arkit-bridge-shipped]]). Anchor identity β: comes from one-shot
VHAP single-image bake (same as today's offline pipeline).

## Implications

- **Live LAM avatar from iPhone is a ~1 day plumbing job**, not weeks of
  retraining.
- Pipeline becomes: LLF UDP → 52-float decode + permute to alphabetical →
  `flame_data["expr"]` + head-Euler → `neck` → existing LAM forward →
  diff-gs-rast → v4l2loopback → OBS. ~5 ms/frame on RTX 5090 once we strip
  the Gradio scaffolding.
- The "structurally offline VHAP" point ([[vhap-cpu-bound-diagnosis]])
  becomes moot for live operation — VHAP only runs once per face, at bake
  time, not per frame.
- The earlier "Python and Web are different architectures" framing in
  `MEMORY.md` should be retracted. They're the same architecture with
  different driver-format adapters.

## What we still don't know

- **Visual quality** of ARKit-driven LAM vs PCA-driven LAM on the same
  anchor. The geometric meshes are correct (verified), but whether the
  Gaussian splats look as good on ARKit-driven poses as on the
  VHAP-PCA-driven poses they trained on is empirically open. The trained
  Gaussian binding is identity-only, but the splats' final pixel quality
  depends on whether ARKit poses fall inside the *training distribution*
  of FLAME meshes (which they should, since VHAP-derived FLAME PCA frames
  cover much of human expression space and ARKit alphabet covers a similar
  subset).
- **Teeth.** Hardcoded `add_teeth=False` in the released config. Toggle and
  see what happens — the Gaussian binding may not have learned teeth even
  if `add_teeth=True` exposes the mesh slots. This is a separate question
  from the ARKit-vs-PCA question and has the same answer regardless of
  driver choice.

These are GPU-render-time follow-ups, blocked behind the current sweep.

## Related

- [[arkit-flame-mapping-extracted]] — earlier doc; mostly correct but
  framed within the "needs retraining" misunderstanding. Math is sound.
- `LAM/lam/models/rendering/gs_renderer.py:570-625` — `animate_gs_model`,
  the key code showing Gaussian net is identity-only.
- `LAM/lam/models/rendering/flame_model/flame_arkit.py` — the dead-code
  ARKit FLAME class, ready to wire up.
