# Splat Render-and-Bake — Color Path Design

**Date:** 2026-05-16
**Status:** spec
**Parent:** `docs/superpowers/specs/2026-05-16-chibi-mesh-pivot-design.md` (this is its color path — supersedes that spec's "v1 needs no UV texture" claim)
**Research:** `docs/research/2026-05-16-splat-appearance-baking.md`

## Problem

The Milestone-0 gate proved the split: splats→mesh **geometry** is clean (LAM's
`shaped_mesh.obj` = `xyz − offset` renders as a coherent FLAME head), but
**per-vertex splat color** (`f_dc`/`shs`) is blotchy raw-meat noise. Splat color
is per-primitive alpha-blend color — it only resolves into a face when many
overlapping Gaussians are composited at render time. Read as hard per-vertex
mesh color it is high-frequency garbage.

The fix is the accepted **render-and-bake**: render the canonical LAM Gaussian
splats from many controlled camera views — that render *is* photoreal — then
bake those rendered pixels onto the FLAME mesh. Bake the splat *output*, not the
splat *parameters*.

## Constraints

- **Bake on the canonical, undeformed mesh.** The splats exist only in LAM's
  canonical proportions; there is no chibi splat render. Chibi deform is strictly
  downstream of the bake (see parent spec / the bake-before-deform decision).
- **Splat-render cameras and mesh-projection must share one camera model**, or a
  mesh vertex projects to the wrong pixel.
- **The canonical gs and `shaped_mesh.obj` share one world frame.** The bake
  never index-aligns the two — it projects mesh-vertex *positions* into the
  splat *images* and samples pixels. (LAM's canonical gs is 20018 points;
  `shaped_mesh.obj` is the 5023-vert base after trimesh drops unreferenced
  verts. Counts differ; only the shared frame matters.)
- Runs in the `lam` conda env: `diff_gaussian_rasterization` (LAM's own splat
  rasteriser) and pytorch3d both live there.
- Reuse `ChibiMesh` and `mesh_render.render` unchanged — the bake fills
  `ChibiMesh.rgb`; nothing downstream changes.

## Decisions (the three open questions)

**1. Bake method: direct projection, not nvdiffrast optimisation.** For each
mesh vertex, project into every splat-render view, depth-test for visibility,
sample the rendered pixel, average across visible views weighted by
normal·view-direction. No differentiable rendering, no nvdiffrast dependency, no
optimisation loop. nvdiffrast UV-texture optimisation is the documented quality
upgrade (next phase) if seams or undersampling show; it is not needed to prove
the color path.

**2. Representation: per-vertex color, not UV texture.** The bake writes
`ChibiMesh.rgb` directly. The blotchiness was never the per-vertex
*representation* — it was the *source* (splat params). Per-vertex color sampled
from clean multi-view renders is an average of correct pixels and will be clean.
5023 base-FLAME verts at the 256² stylized-chibi product resolution is adequate.
This reuses `ChibiMesh` + `mesh_render` with zero new representation. A UV
texture (higher resolution, Blender hand-off) is the next phase, deferred.

**3. Splat render source: a dumped canonical `.ply` + a standalone renderer.**
Add a one-line canonical-gs dump to LAM inference (`cano_gs_lst[0].save_ply(...,
offset2xyz=False)` → `<stem>_cano.ply`, a complete standard 3DGS .ply: xyz,
`f_dc`, opacity, scale, rotation). Render it with `diff_gaussian_rasterization`
directly — same rasteriser LAM uses, so color/opacity/scale conventions match
exactly. Rejected: in-process LAM render (heavy full-model load); reusing the
driven per-frame plys (driven pose, not canonical — would not match
`shaped_mesh.obj`).

## Approaches considered

**A — direct projection bake, per-vertex color (recommended).** As decided
above. Smallest pipeline that falsifies-or-confirms the color path; reuses every
existing unit; no new dependency beyond the splat rasteriser already in the env.

**B — nvdiffrast UV-texture optimisation bake.** Learnable UV texture, photometric
loss against splat renders, nvdiffrast backprop. SuGaR-grade quality, handles
parallax and seams, still seconds. Deferred: needs the FLAME UV layout wired in
and an optimisation loop; overkill before A proves the premise.

**C — keep per-splat color, denoise it.** Smooth the per-vertex `f_dc` over the
mesh (Laplacian / bilateral). Rejected: per-splat color is not low-frequency
noise on top of albedo — it is structurally not albedo. Smoothing blurs garbage.

**Decision: A now, B as the documented next phase.**

## Architecture

Three new units in `src/chibi/`, plus a driver and a one-line LAM change.

**`src/chibi/camera_rig.py` — `turntable_cameras(n_azim, elevs, dist) -> Cameras`.**
Generates the shared camera set: `n_azim` azimuths × each elevation in `elevs`,
all looking at the origin. Default 12 azimuths × {−20°, 0°, +20°} = 36 views —
enough to cover a head including under-chin and crown. Returns pytorch3d
cameras (and the raw R/T so the splat renderer can consume them). One camera
model, used by both the splat render and the projection bake.

**`src/chibi/splat_render.py` — `render_splats(ply_path, cameras, image_size) -> (T,H,W,3)`.**
Loads a standard 3DGS `.ply` (xyz, `f_dc`, opacity=`sigmoid(stored)`,
scale=`exp(stored)`, rotation=quaternion) and rasterises it from each camera with
`diff_gaussian_rasterization`. `gs_use_rgb` → pass `f_dc` as `colors_precomp`.
Returns uint8 RGB images. Depends on: `diff_gaussian_rasterization`, plyfile.

**`src/chibi/bake.py` — `bake_vertex_colors(mesh, images, cameras) -> ChibiMesh`.**
For each mesh vertex: project into every view; rasterise the mesh from that view
(pytorch3d `MeshRasterizer`, reuse for depth) and depth-test the vertex against
the `zbuf` to reject occluded views; weight surviving views by
`max(normal·viewdir, 0)`; take the weighted-average sampled pixel. Vertices
visible in zero views (inner mouth, eyeball backs) get the nearest-visible
vertex's color. Returns a new `ChibiMesh` with baked `rgb`, faces/verts
unchanged. Pure function over `ChibiMesh` + arrays — no LAM dependency.

**`scripts/bake_anchor_texture.py` + `.sh` — driver.**
`<stem>_cano.ply` + `<stem>_shaped_mesh.obj` → render splats → bake → render the
textured mesh with `mesh_render.render` → side-by-side against a splat render
frame. Eyeball verdict: does the textured mesh match the splat appearance.

**LAM change.** In `lam/runners/infer/lam.py`, add one line in the cano-dump
block: `res['cano_gs_lst'][0].save_ply(<dump_dir>/<stem>_cano.ply,
offset2xyz=False)`. Re-run clean inference once per anchor to emit it.

## Data flow

```
anchor.png ──LAM infer──> <stem>_cano.ply        (canonical 3DGS)
                          <stem>_shaped_mesh.obj (canonical FLAME mesh)
                            │
        turntable_cameras ──┤── shared camera set ──┐
                            │                       │
        render_splats(cano.ply, cameras) ──> splat images (T,H,W,3)
                            │                       │
        bake_vertex_colors(mesh, images, cameras) ──> ChibiMesh (baked rgb)
                            │
        [downstream, unchanged: apply_chibi, ARKit/LBS, mesh_render.render]
```

## Error handling

- `splat_render`: assert the `.ply` has `scale_*`/`rot_*`/`opacity` (a 3DGS
  ply, not the `_gs_offset.ply`); assert rendered images are non-empty.
- `bake`: assert `len(images) == len(cameras)`; count zero-visibility vertices
  and warn if the fraction is large (camera rig does not cover the head).
- Color space: LAM `gs_use_rgb` color is display-RGB; the splat render is read
  and the vertex color is written in the same space — no linear/sRGB conversion.
  Verify on the first bake by eye against a splat-render frame (the washed-out
  failure mode from `sam-3d-objects` #75).

## Testing

- `camera_rig`: 36 cameras for the default rig; all look at the origin;
  deterministic.
- `splat_render`: a synthetic 1-Gaussian `.ply` renders to a non-empty image of
  the expected color; no LAM dependency.
- `bake`: synthetic — a flat quad facing +Z + one solid-color +Z view → every
  vertex gets that color. Occlusion — a two-layer mesh, the back layer does not
  take the front layer's color (depth test works).
- Integration: `bake_anchor_texture.sh` on `me` and the LAM reference; the
  textured-mesh render side-by-side against a splat-render frame; eyeball — the
  blotchy raw-meat look is gone, the mesh reads as the same face.

## Out of scope (next phases)

nvdiffrast UV-texture optimisation bake (Approach B); the FLAME UV layout and
Blender hand-off; relighting / albedo de-shading; chibi-stylised texture
(painterly flat skin — a separate appearance axis); hair shell. Each is its own
spec when reached.
