---
status: live
topic: arkit-bridge
---

# ARKit → FLAME mapping — extracted

**Date:** 2026-05-13
**Predecessor:** `docs/research/2026-05-13-realtime-flame-existing-solutions.md`

## Headline

The mapping we were going to build, derive, or extract from JavaScript is **a single .npy file** in LAM's standard model zoo:

```
model_zoo/human_parametric_models/flame_assets/flame_arkit_bs.npy
shape:  (52, 5023, 3)   float64
size:   6 MB
range:  [-0.044, +0.030]  (FLAME world units; ~7 mm peak per blendshape)
```

Axis 0 is the 52 ARKit blendshapes (alphabetical, per the `targetNames` list in
LAM_WebRender's `skin.glb`). Axis 1 is FLAME base-mesh vertices (5023). Axis 2 is
XYZ displacement. **Per ARKit blendshape, a per-vertex 3D offset.**

The basis is **anchor-independent** — universal across all LAM avatars. Per-avatar
personalization happens only in the FLAME identity shape `β` and the
LAM-encoded identity Gaussians. There is **no** per-avatar 52-LAM-forward
materialization step. This was my wrong guess initially; the actual architecture
is much simpler.

## How it's used in the LAM code

`lam/models/rendering/flame_model/flame_arkit.py` line 124–130:

```python
assert os.path.exists(flame_arkit_bs_path)
flame_arkit_bs = np.load(flame_arkit_bs_path).astype(np.float32)
flame_arkit_bs = torch.from_numpy(flame_arkit_bs).float().permute(1, 2, 0)
# now shape is (5023, 3, 52) — same layout FLAME expects for shape_dirs

self.shapedirs = nn.Parameter(torch.cat(
    [shapedirs[:, :, :shape_params], flame_arkit_bs], dim=-1), …)
```

The ARKit basis is **concatenated onto FLAME's shape_dirs**, replacing the PCA
expression basis. From that point on, LAM's downstream code treats ARKit weights
the same way it would treat FLAME PCA expression coefficients: a learned linear
basis on vertex displacement. The standard FLAME forward (linear blend skinning
+ blend_shapes) does the rest.

This is the "manual customization to align ARKit blendshapes with FLAME's
facial topology" that the LAM team's README mentions — the customization was
*producing this .npy*, not a per-avatar runtime step. Once produced, the live
runtime is a single 52×5023×3 matmul per frame.

## Per-frame math

```python
# Pre-compute once
identity_gaussians = LAM.encode(anchor_image)        # ~10–20s, persisted to disk
shape_template_verts = FLAME.forward(beta=β)          # static, ms

# Per frame
arkit52 = recv_llf_udp()                              # 52 floats from iPhone Live Link Face
expr_delta = np.einsum("k,kvd->vd", arkit52,
                       flame_arkit_bs)                # (5023, 3) — sub-ms
head_pose = quat_to_R(head_rotation)                  # from LLF rotation field
final_verts = (shape_template_verts + expr_delta) @ head_pose.T + t
splat(identity_gaussians, final_verts)                # diff-gs-rast, ~3–5 ms on 5090
```

**Total per-frame budget: ~5 ms on RTX 5090.** 33 ms (30 fps) leaves a 6× margin.

## How the WebRender uses the same mapping

LAM_WebRender doesn't load the .npy at runtime; instead the LAM team has *pre-
baked* the 52 ARKit deltas into a glTF morph-target structure inside an FBX
template (`tools/generateARKITGLBWithBlender.py`), then injects per-avatar
FLAME-identity vertex positions into that template via Blender. The resulting
`skin.glb` has 51 morph targets (52 minus `tongueOut`, which is omitted on the
web side) named after ARKit blendshapes.

This is glTF-standard morph-target animation. The web renderer is just a glTF
+ Gaussian-splatting renderer; the ARKit→Gaussian deltas are baked into the
asset, not computed at runtime. Same math as the Python side, in a
different container.

## Sanity checks before plumbing

1. **Ordering**: confirm our Live Link Face stream's 52-float order matches the
   alphabetical ARKit names (per `LAM_WebRender/asset/test_expression_1s.json`).
   Per `reference_llf_udp_wire_format`, Unity's FaceBlendShape enum **is**
   alphabetical, so this should match — but our wire format unpacker reads in
   `wire order` which is *not* alphabetical. Need a permutation lookup table.
   1 hour of work.
2. **Subdivision**: LAM upsamples FLAME base mesh (5023 verts) to ~20K
   subdivided mesh for Gaussian binding. The .npy here is on the base mesh;
   the subdivision is performed on the *final* vertices, so applying expr_delta
   pre-subdivision is correct. Verify via a single-frame round-trip.
3. **Head rotation**: LLF emits 3 Euler floats (HeadYaw/Pitch/Roll). Need to
   convert to FLAME's `neck` joint rotation. Should be the same closed-form
   `EULER_SIGNS=(+1,-1,+1) F=I` we shipped in [[project-arkit-bridge-shipped]]
   for PersonaLive, since the FLAME neck and PersonaLive's pose share the
   underlying iPhone Euler convention.

None of these are research questions; all are 1-day plumbing.

## Product implications

| Step | Wall | Frequency |
|---|---|---|
| Avatar bake (FLAME track + LAM identity encode) | ~12–15 s | once per face, persisted |
| Per-frame ARKit→expr_delta matmul | <1 ms | every frame |
| Per-frame Gaussian rasterize on 5090 | ~3–5 ms | every frame |
| Total per-frame budget | ~5 ms | 6× margin under 30 fps |
| Storage per avatar | ~10 MB | LAM weights + FLAME β |

**This means:**
- Onboarding UX is *putting on a filter*, not setting up a session.
- Serve cost is **zero ML inference per frame.** A single GPU can serve
  thousands of concurrent live avatars (limited only by rasterizer throughput).
- The avatar bundle is portable: bake on a server, ship to phone/browser.
- The live ARKit→OBS pipeline is **simpler than the shipped LP daemon** —
  no PersonaLive, no bridge student, no closed-form pose calibration. Just the
  matmul above.

What stays out of scope:
- The "live-drive a million faces" scenario at vamp-interface's data-viz
  product scale — 10 MB × 10⁶ = 10 TB. But that's not what live LAM is for;
  Flux+slider stays right for the dataset-scale view.

## Next concrete experiments

1. **Round-trip sanity**: load `flame_arkit_bs.npy`, multiply by a one-hot
   ARKit vector (e.g., `jawOpen=1.0`), add to FLAME zero-pose vertices, render
   from the side; verify mouth opens correctly. ~30 min.
2. **Live demo skeleton**: a `scripts/streaming_bridge_lam.py` mirroring
   `streaming_bridge_lp.py`'s structure. Inputs: LLF UDP. Outputs: v4l2loopback.
   Internal: anchor bake (one-shot), per-frame expr_delta + diff-gs-rast.
   ~1 day of plumbing.
3. **Verify ordering**: write a parity test that drives `jawOpen` via LLF only,
   captures the resulting expr_delta indexing, and asserts the diff visualizes
   on the jaw region. ~1 hour. Falsifier: wrong axis lights up, → fix
   permutation table.

## Falsified architecture guesses (from earlier today)

- "Per-avatar 51-LAM-forward bake step takes 5–10 s." False — the basis is
  universal, no per-blendshape LAM forward needed.
- "We need to extract the mapping from WebGL shader code." False — the mapping
  is a stand-alone .npy in the standard LAM model zoo. The WebRender path
  bakes it into glTF morph targets, but that's just packaging.
- "Avatar materialization wall is 20–30 s." False — it's just FLAME track +
  LAM identity encode, ~12–15 s.
- "We need to learn a 52→FLAME map ourselves." False — LAM team already did
  the work and shipped the basis with their model zoo.

## Related

- [[project-lam-bakeoff-x6-verdict]] — LAM as renderer-of-record
- [[reference-realtime-flame-solutions]] — survey that pointed us here
- `LAM/lam/models/rendering/flame_model/flame_arkit.py` — implementation
- `LAM/tools/generateARKITGLBWithBlender.py` — avatar GLB build pipeline
- `LAM/model_zoo/human_parametric_models/flame_assets/flame_arkit_bs.npy` — the asset
- `LAM_WebRender/asset/test_expression_1s.json` — example 52-name ARKit stream
