# Chibi Mesh Pivot — Design

**Date:** 2026-05-16
**Status:** spec
**Supersedes:** the splat-path chibi recipe (`_topics/lam-chibi-recipe.md`, concluded dead 2026-05-15)
**Research:** `docs/research/2026-05-15-splats-to-mesh-conversion.md`

## Problem

Chibi-on-splats is dead: chibi-magnitude deformation rescatters a baked Gaussian-splat
cloud (appearance glued to finite blobs at fixed density → blur / smear / iris-leak).
Pivot to a **mesh** representation, where appearance is carried by triangle interpolation
and is therefore deformation-invariant. The `ChibiField` deformation module and the
secant/neck fixes shipped on `chibi-diff-leak-fix` are kept; only the rendering target
changes — splats out, textured mesh in.

## Spike findings (the three unknowns, resolved against `/home/newub/w/LAM`)

These three answers reshape the whole design — read them first.

**1. Color is per-vertex RGB, not SH.** `configs/inference/lam-20k-8gpu.yaml` sets
`gs_use_rgb: True`. In `use_rgb` mode `GSLayer` emits `gs.shs` as a `(N,1,3)` tensor of
**direct sigmoid RGB** (`gs_renderer.py:252`, `colors_precomp = gs.shs.squeeze(1)` at
`:555`). It is view-independent. There is no SH-DC conversion, no `C0` factor, and no
"is there meaningful view-dependent gloss" question — there is none. The bake is: read
`_gm.shs.squeeze(1)`.

**2. There are no off-surface splats.** LAM places exactly one Gaussian per upsampled
mesh vertex; the GSLayer xyz output is a *bounded* offset (`restrict_offset=True`,
`v=(sigmoid(v)-0.5)*max_step`, `gs_renderer.py:266`). Every Gaussian sits on the FLAME
surface. **Hair is not a separate splat cloud** — it is scalp/skull vertices coloured
like hair. On a mesh that is rendered as hair-coloured scalp triangles automatically.

**3. The mesh already has teeth, and the 20018 mesh has no UV.** `add_teeth=True` is the
default everywhere (`modeling_lam.py:301`, `gs_renderer.py:431`); `FlameHead.add_teeth()`
builds upper/lower teeth geometry, LBS-weights them to neck/jaw, and appends teeth UVs
(`flame_arkit.py:211,355-361`). So teeth geometry already exists and is rigged.
Subdivision (`FlameHeadSubdivided`, `flame_arkit.py:677+`) upsamples vertex-indexed
buffers (v_template, lbs, posedirs) but **not** `verts_uvs`/`textures_idx` — the
per-corner UV layout lives only on the 5023+teeth base. `pytorch3d.SubdivideMeshes`
*can* carry features (`upsample_mesh_cpu` already calls `subdivider(mesh, feats)`), so
subdivided UVs are derivable, but are not needed for v1 (see below).

## Consequence: v1 needs no UV texture at all

Per-vertex RGB on a mesh is **already deformation-safe** — a stretched triangle still
interpolates its three corner colours smoothly across its larger area; there is no
finite-blob density to rescatter, no holes. The splat failure does not have a mesh
analogue. pytorch3d renders this directly with `TexturesVertex(verts_features=rgb)`,
which LAM's own `vis_utils.py:312` already uses. 20018 vertex colours at 256² output is
~3 px/vertex — ample for a flat chibi look.

A baked UV texture only buys resolution beyond vertex density and a Blender hand-off
path. Both are real but neither is needed to prove the pivot. **UV baking is deferred to
v2.** This is the YAGNI cut: v1 ships the smallest thing that falsifies-or-confirms the
pivot.

## Approaches considered

**A — Per-vertex-RGB mesh, pytorch3d render (recommended, v1).** Take LAM's baked avatar
(20018 verts, `face_upsampled` connectivity, `_gm.shs` RGB, opacity), deform verts with
`ChibiField`, render with `TexturesVertex` + pytorch3d `MeshRenderer`. Stays in the LAM
conda env, in Python, keeps the ARKit-52 drive path. No UV bake, no Blender, no new
asset format. Reuses `ChibiField` unchanged and LAM's existing rasteriser machinery.

**B — UV-texture-baked mesh, Blender render (v2 quality path).** Bake `_gm.shs` into the
5023+teeth FLAME UV (rasterise mesh in UV space, barycentric-interpolate corner colours,
1K texture), export a textured OBJ via LAM's `mesh_utils.save_obj(texture_type='surface')`,
deform + render in Blender. Higher output fidelity, real lighting/AA, but a handoff step
and loses the live ARKit path. Good later; overkill for the falsification test.

**C — SuGaR-style mesh + bound Gaussians.** Keep Gaussians, bind them to triangles so
deformation propagates. Rejected: it re-imports the exact density-coupling problem the
pivot exists to escape.

**Decision: A for v1, B kept as the documented v2 upgrade.** A is the minimum that
answers "does the mesh pivot resolve blur/smear/leak"; A's flat per-vertex look is
*expected* to be too low-fi to ship — v1 is built as the **foundation for v2**, not the
final renderer.

**v2-readiness constraint (load-bearing on v1).** v1 must not paint itself into a
corner. Two concrete obligations:

- The `ChibiMesh` dataclass is the **stable interface** between units. v2's UV bake
  consumes it; v2's Blender export consumes it. v1 keeps `mesh_deform` and the dataclass
  renderer-agnostic — no pytorch3d type leaks into them.
- The v2 UV bake reads the `verts_uvs`/`textures_idx` layout from the **static FLAME
  template** (`head_template_mesh.obj`, 5118 vt), not from a LAM run — so it is not
  run-dependent and need not be carried on `ChibiMesh`. What *is* run-dependent is the
  baked per-vertex RGB, and that is already persisted as `<stem>_textured_mesh.obj`.
  v1 must therefore preserve that OBJ as the canonical colour source for v2.

Spike correction: `extract_lam_mesh` is **not** a LAM-inference wrapper. LAM inference
already runs upstream (`chibi_anchor_render.sh` stage 1) and writes
`exps/cano_gs/<stem>_textured_mesh.obj` — an OBJ carrying 20018 `v x y z r g b` lines
(per-vertex RGB = `_gm.shs.squeeze(1)`, written by `lam/runners/infer/lam.py:408-411`)
and 39904 `f` lines. `extract_lam_mesh` is a pure OBJ parser: no LAM env, no checkpoint,
fully unit-testable in the vamp uv env. Opacity is not in the OBJ and not needed (a mesh
is opaque) — it is dropped from v1's `ChibiMesh`.

## Architecture (v1)

Four units, each independently testable.

**`src/chibi/mesh_extract.py` — `extract_lam_mesh(anchor) -> ChibiMesh`.**
Runs LAM inference on an anchor PNG, pulls the canonical (rest-pose) avatar, returns a
`ChibiMesh` dataclass: `verts (V,3)`, `faces (F,3)` = `face_upsampled`, `rgb (V,3)` =
`_gm.shs.squeeze(1)`, `opacity (V,)`. V = `vertex_num_upsampled` (20018 at
subdivide_num=1; whatever LAM is configured to). Rest pose only — no ARKit, no chibi.
Depends on: LAM checkpoint, LAM conda env.

**`src/chibi/mesh_deform.py` — `apply_chibi(mesh, field) -> ChibiMesh`.**
Applies an existing `ChibiField` (+ region weights from `chibi.landmarks`) to
`mesh.verts`, returns a new `ChibiMesh` with deformed verts, same faces/rgb/opacity.
Pure function over the dataclass; no LAM dependency → unit-testable in the vamp uv env.
Reuses `ChibiField.forward` verbatim.

**`src/chibi/mesh_render.py` — `render(mesh, cameras) -> images`.**
pytorch3d `MeshRenderer` (`MeshRasterizer` + soft/hard shader) with
`TexturesVertex(verts_features=mesh.rgb)`. Background composited per LAM's existing
convention. Single image and batched-frames entry points. Depends on: pytorch3d.

**`scripts/chibi_mesh_render.sh` — end-to-end driver.**
anchor PNG → `extract_lam_mesh` → `apply_chibi` → `render` → a **camera-turntable** mp4
(facial animation is v1.1; the v1 motion is camera orbit, which exercises the rasteriser
from all angles without the ARKit path), plus a baseline (field=identity) render and a
side-by-side against the splat-path verdict videos (`chibi_asian_m_secant_neck.mp4`,
`chibi_me_secant_neck.mp4`).

**Milestone 0 — the gate (built and run first).** Before any chibi code, render an
existing ARKit-driven take as an animated *mesh* and compare it to the splat render of
the same take, to confirm splats→mesh did not catastrophically lose quality. LAM already
dumps per-frame `.ply` files (`exps/images/lam/lam_20k/<stem>/NNNN.ply`) carrying the
animated per-vertex `xyz` and per-vertex RGB; Milestone 0 reads those, attaches the
constant faces, and renders the sequence. No chibi, no rig re-implementation — LAM is
the rig. If the gate fails, the pivot is reworked before chibi.

ARKit animation *of the chibi mesh*: `apply_chibi` operates on rest-pose verts; ARKit-52
+ FLAME LBS are applied *after* chibi, by the same `canonical + Σαₖbₖ` path LAM uses, on
the deformed verts. The secant-rescaled basis from `chibi_make_assets.py` is the correct
per-frame basis and is reused. Driven *chibi* animation is a later task; the chibi v1
render is a static turntable verdict.

## Data flow

```
anchor.png ──LAM infer──> canonical avatar (_gm)
                            │
              extract_lam_mesh: verts, faces, rgb, opacity
                            │
              apply_chibi (ChibiField + region weights)
                            │
              [v1.1: + ARKit-52 basis + FLAME LBS per frame]
                            │
              render (pytorch3d TexturesVertex)
                            │
                          frames ──> mp4 + side-by-side
```

## Error handling

- `extract_lam_mesh` asserts `verts.shape[0] == faces.max()+1` and that `rgb` is in
  `[0,1]` (sigmoid output) — a fail here means the use_rgb assumption broke.
- `apply_chibi` asserts vert count is unchanged (deformation must not re-topologise).
- `render` asserts no NaN in verts (chibi over-stretch / degenerate triangle guard).
- Anchor stress case: the `me` anchor's deeper-z canonical frame broke the splat path;
  v1 must render `me` *and* `asian_m` and the verdict explicitly compares both.

## Testing

- `mesh_deform`: unit tests in the vamp uv env — identity field leaves verts unchanged;
  a known chibi field moves the crown/chin as expected; vert count invariant. Reuses the
  `tests/test_chibi_field.py` fixtures.
- `mesh_extract`: smoke test in the LAM env — shapes/dtype/ranges of the returned
  dataclass on one anchor; gated by checkpoint presence (skipif).
- `mesh_render`: a 2-triangle textured-mesh fixture renders to a non-empty image of the
  expected colour; no LAM dependency.
- Integration: `chibi_mesh_render.sh` produces the side-by-side; the verdict is an
  eyeball gate (does blur/smear/leak survive on a mesh — expected: no).

## Out of scope (v2+)

UV-texture bake and Blender path (Approach B); helmet-hair volume shell beyond the FLAME
skull (scalp colour covers v1); hair cards (Strands2Cards/CGHair); subdivided-UV
derivation; relighting / albedo de-shading. Each is a separate spec when reached.
