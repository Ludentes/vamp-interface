# nvdiffrast UV-Texture Bake — Color Path v3 Design

**Date:** 2026-05-16
**Status:** spec
**Parent:** `docs/superpowers/specs/2026-05-16-splat-render-and-bake-design.md` (v2; this is its pre-authorized escalation)
**Research:** `docs/research/2026-05-16-splat-appearance-baking.md`

## Problem

The v2 render-and-bake works: clean no-chibi bakes of `status`, `asian_m`,
`me_512` produce coherent textured FLAME heads — the blotchy raw-meat per-splat
color is gone. But v2 writes one RGB triple **per vertex** into `ChibiMesh.rgb`,
and the FLAME base mesh has only 5023 vertices. Color interpolates linearly
across each face, so the bake carries no detail finer than a face: faces are
*soft* — no skin pores, no eyebrow strands, no stubble. The user pre-authorized
escalating to a UV-texture bake the moment v2 showed "any trace of problems";
the softness is that trace.

The root cause is precisely the per-vertex sampling rate. Nothing else about v2
is wrong — the splat renders are photoreal, the projection and occlusion math
were validated on three anchors. v3 keeps that math and changes only the
sampling: bake into a UV **texture atlas** instead of vertices.

## Constraints

- **Bake on the canonical, undeformed mesh.** Same as v2 — splats exist only in
  LAM canonical proportions. A UV atlas is deformation-invariant: chibi deform
  moves vertices, never touches UVs or the texture, so the bake stays strictly
  upstream of deform.
- **Splat-render cameras and mesh-projection share one camera model** (the v2
  `camera_rig` / `splat_render` pair — reused unchanged).
- Runs in the `lam` conda env: `nvdiffrast` 0.3.3, `diff_gaussian_rasterization`,
  pytorch3d all live there. nvdiffrast needs a CUDA raster context.
- **FLAME's stock UV layout transfers with zero remeshing.**
  `model_zoo/human_parametric_models/flame_assets/flame/head_template_mesh.obj`
  is 5023 v / 5118 vt / 9976 f, and its face vertex order is identical to
  LAM's `shaped_mesh.obj` (template face `4/1 2/2 1/3` ↔ shaped face `4 2 1`).
  The `vt` block and the `/vt` face indices splice directly onto any
  `shaped_mesh.obj`.

## Decisions

### Bake method: per-texel direct projection (not optimization)

Three candidates:

- **A — optimization bake (SuGaR reference).** Gradient-descend a texture atlas
  so nvdiffrast-rendering the textured mesh from each view matches the splat
  render. Robust multi-view blending, antialiased seams — but adds an iterative
  loop, a learning rate, antialias-gradient plumbing, and convergence behavior
  to debug.
- **B — per-texel direct projection (chosen).** Rasterize the mesh *in UV space*
  once to get, per texel, its canonical 3D surface position and normal. Then run
  the *exact v2 projection+occlusion+normal-weighted-average math* on those
  texel points instead of on vertices. Non-iterative. nvdiffrast is used for one
  thing only: the single UV-space rasterize.
- **C — forward scatter.** Rasterize each view, scatter splat-render pixels into
  the atlas at their interpolated UVs. Non-iterative but has write-conflict and
  coverage-ordering hazards; B's gather is the clean dual.

**Choose B.** The softness has one fully-understood cause — vertex sampling rate
— and B removes exactly that cause, raising the sampling rate from 5023 verts to
`tex_size²` texels while keeping the projection math already validated on three
anchors. Optimization's genuine advantage is reconciling *mutually inconsistent*
input views; our splat renders are mutually consistent by construction (one set
of canonical splats), so that advantage does not apply. B is the minimal,
lowest-risk escalation.

### Texture resolution: 2048², splat renders at 1024²

Per-texel color is ultimately capped by the splat-render resolution it samples.
v2 rendered splats at 512². v3 renders splats at **1024²** and bakes a **2048²**
atlas — the atlas oversamples the input so UV-triangle interiors are not the
bottleneck; the splat render is. Both are CLI knobs (`--splat_size`,
`--tex_size`).

### Hole filling: texture-space dilation

A texel is a *hole* if it lies outside every UV triangle, or its 3D point is
never visible (back-facing in all views). Holes left raw cause background bleed
under bilinear sampling / mipmapping. Fill by iterative dilation: repeatedly
copy each hole texel from its nearest filled 4-neighbor until the atlas is
solid. ~16 passes covers the FLAME UV gutters.

### Output: textured OBJ + MTL + PNG

Write `<stem>_textured.obj` (shaped geometry + spliced FLAME UVs +
`mtllib`/`usemtl`), `<stem>_textured.mtl` (`map_Kd <stem>_texture.png`), and the
baked `<stem>_texture.png`. This is a standard textured mesh any tool loads.

## Architecture

Data flow:

```
shaped_mesh.obj (5023v, no UV)
head_template_mesh.obj (FLAME UV) ──> uv_template.load_flame_uv
                                 │      → uv (5118,2), uv_faces (9976,3),
                                 │        vt2v (5118,) map
cano.ply (canonical splats) ─────┼──> splat_render.render_splats (1024²)
                                 │        → images (T,H,W,3), depth (T,H,W)
                                 ▼
        texture_bake.rasterize_uv_attrs ──> per-texel canonical position map
        (nvdiffrast, UV-space raster)        + normal map + coverage mask
                                 ▼
        bake.bake_points (v2 math, per-texel) ──> per-texel RGB
                                 ▼
        texture_bake.dilate ──> solid 2048² atlas
                                 ▼
        write OBJ+MTL+PNG ; mesh_render.render_textured ──> verdict renders
```

### Components

**`src/chibi/uv_template.py`** — `load_flame_uv(template_obj) -> FlameUV`.
`FlameUV` dataclass: `uv (Nvt,2) float32` in [0,1], `uv_faces (F,3) int64`,
`vt2v (Nvt,) int64` (UV-vertex → mesh-vertex map, built from the `v/vt` pairs in
the template faces; asserts each `vt` resolves to a single `v`). Pure parser, no
LAM/torch-CUDA dependency. Depends on: the template OBJ path.

**`src/chibi/bake.py`** — refactor. Extract the core into
`bake_points(points (N,3), normals (N,3), images, depth, views, *, fallback_rgb)
-> rgb (N,3)`: the existing project / occlusion-test / normal-weighted-average
loop, operating on an arbitrary point cloud. `bake_vertex_colors` becomes a thin
wrapper: compute vertex normals, call `bake_points`, wrap in `ChibiMesh`. v2
behavior and `test_bake.py` are unchanged (regression guard).

**`src/chibi/texture_bake.py`** — the nvdiffrast unit.
- `rasterize_uv_attrs(verts, faces, flame_uv, tex_size) -> (pos_map, nrm_map,
  mask)`: build per-UV-vertex attribute arrays — `attr_pos[i] = verts[vt2v[i]]`,
  `attr_nrm[i] = vertex_normal[vt2v[i]]`; rasterize UV space (clip pos =
  `(uv*2-1, 0, 1)`, triangles = `uv_faces`) with a `RasterizeCudaContext`;
  `interpolate` the two attrs. Returns `pos_map (tex,tex,3)`,
  `nrm_map (tex,tex,3)`, `mask (tex,tex) bool` (rast triangle-id > 0).
- `bake_texture(verts, faces, flame_uv, images, depth, views, tex_size) ->
  (texture (tex,tex,3) float32, mask)`: rasterize attrs, flatten masked texels,
  call `bake_points`, scatter results back into a `(tex,tex,3)` atlas.
- `dilate_texture(texture, mask, iters=16) -> texture`: iterative nearest-filled
  4-neighbor fill of `~mask` texels.

Depends on: `nvdiffrast`, `bake.bake_points`, `uv_template.FlameUV`.

**`src/chibi/mesh.py`** — add `TexturedMesh` dataclass: `verts (V,3)`,
`faces (F,3)`, `uv (Nvt,2)`, `uv_faces (F,3)`, `texture (H,W,3) float32`.
`__post_init__` validates shapes and that `uv_faces` and `faces` share length.
`ChibiMesh` is untouched.

**`src/chibi/mesh_render.py`** — add `render_textured(tmesh, azims, *,
image_size, dist, elev, device) -> frames`: pytorch3d `TexturesUV` +
`AmbientLights`, same camera/turntable conventions as the existing vertex-color
`render`. The existing `render` stays for v2 comparison renders.

**`scripts/bake_uv_texture.py`** + **`.sh`** — driver. Loads `shaped_mesh.obj` +
`cano.ply` (translation-only recenter, as v2), loads FLAME UV, renders splats at
`--splat_size`, bakes + dilates the atlas, writes OBJ/MTL/PNG, renders verdict
turntable frames at `--render_size` plus a side-by-side vs the v2 vertex bake.

## Error handling

- `load_flame_uv`: assert template is 5023 v / 9976 f and face count matches the
  target `shaped_mesh.obj`; assert each `vt` index maps to exactly one `v`
  (raise on a multi-mapped `vt` — would mean topology mismatch).
- `rasterize_uv_attrs`: assert `RasterizeCudaContext` constructs (CUDA present);
  assert UVs lie in [0,1].
- `bake_texture`: if `mask` is empty (no texel covered) raise — never write a
  blank atlas silently.
- `dilate_texture`: cap iterations; if holes remain after the cap, fill residual
  with the mean texel color and warn (don't leave NaNs/zeros).

## Testing

- `test_uv_template.py`: `load_flame_uv` on the real FLAME template returns
  `uv (5118,2)` in [0,1], `uv_faces (9976,3)`, `vt2v (5118,)`; spot-check that a
  template face's `vt2v` lookup recovers the shaped-mesh face vertex indices.
- `test_bake.py`: unchanged — regression guard that the `bake_points` refactor
  preserves v2 behavior (solid-color and occlusion tests still pass).
- `test_texture_bake.py`: synthetic unit-quad mesh with a known UV square, one
  azim-0 view, a solid-red splat image, infinite depth → the covered atlas
  region bakes red; `dilate_texture` on an atlas with a punched hole fills the
  hole from neighbors; `rasterize_uv_attrs` mask coverage > 0.
- **Verdict (manual gate):** bake `asian_m`, `me_512`, `status`; render the
  textured mesh side-by-side against the v2 vertex bake. Pass = fine detail
  (eyebrow/skin texture) visibly present, no seam bleed, no background halo.
  Per the standing instruction, any trace of a seam/coverage problem in B is
  itself the signal to fall back to A (optimization bake) — no patching.

## Out of scope (unchanged from v2)

Hair and clothing remain flat blobs — FLAME has no geometry there; a texture
atlas cannot add a hair shell. Separate, deferred problem. Chibi deform
(`apply_chibi`, tasks #63/#65) stays gated downstream of this working color path.
