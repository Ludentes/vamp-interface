# nvdiffrast UV-Texture Bake Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bake the canonical LAM splat appearance into a FLAME UV-texture atlas so the textured mesh carries fine skin detail the v2 per-vertex bake cannot.

**Architecture:** Splice FLAME's stock UV layout onto LAM's `shaped_mesh.obj` (identical topology). Rasterize the mesh *in UV space* once with nvdiffrast to get a per-texel canonical 3D position + normal map. Run the v2 projection/occlusion/normal-weighted-average bake on those texel points instead of on vertices. Dilate UV-gutter holes, write a textured OBJ+MTL+PNG, render a verdict.

**Tech Stack:** Python 3.12, PyTorch, nvdiffrast 0.3.3, `diff_gaussian_rasterization`, pytorch3d — all in the `lam` conda env.

**Spec:** `docs/superpowers/specs/2026-05-16-nvdiffrast-uv-bake-design.md`

**Environment for every test / run command below:**
```bash
export PATH=/home/newub/miniconda3/envs/lam/bin:$PATH
export PYTHONPATH=/home/newub/w/LAM
export XFORMERS_DISABLED=1
cd /home/newub/w/vamp-interface
```

---

## File Structure

- `src/chibi/uv_template.py` — **create.** Parse FLAME's `head_template_mesh.obj`; expose its UV layout (`FlameUV`: `uv`, `uv_faces`, `vt2v`).
- `src/chibi/bake.py` — **modify.** Extract the projection loop into a reusable `bake_points`; `bake_vertex_colors` becomes a thin wrapper. v2 behavior unchanged.
- `src/chibi/texture_bake.py` — **create.** The nvdiffrast unit: `rasterize_uv_attrs`, `bake_texture`, `dilate_texture`.
- `src/chibi/mesh.py` — **modify.** Add the `TexturedMesh` dataclass. `ChibiMesh` untouched.
- `src/chibi/mesh_render.py` — **modify.** Add `render_textured` (pytorch3d `TexturesUV`). `render` untouched.
- `scripts/bake_uv_texture.py` + `scripts/bake_uv_texture.sh` — **create.** Driver: bake an anchor, write OBJ/MTL/PNG, render verdict.
- `tests/test_uv_template.py`, `tests/test_texture_bake.py` — **create.**
- `tests/test_bake.py` — **unchanged** (regression guard for the `bake.py` refactor).

---

### Task 1: FLAME UV template loader

**Files:**
- Create: `src/chibi/uv_template.py`
- Test: `tests/test_uv_template.py`

The FLAME stock template
`/home/newub/w/LAM/model_zoo/human_parametric_models/flame_assets/flame/head_template_mesh.obj`
is 5023 v / 5118 vt / 9976 f; its face vertex order is identical to LAM's
`shaped_mesh.obj` (template `f 4/1 2/2 1/3` ↔ shaped `f 4 2 1`), so its UVs
splice straight on. A UV seam *splits* a 3D vertex into several UV vertices but
never *merges* two, so each `vt` index maps to exactly one `v` index — that map
is `vt2v`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_uv_template.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import torch
from chibi.uv_template import load_flame_uv

TEMPLATE = ("/home/newub/w/LAM/model_zoo/human_parametric_models/"
            "flame_assets/flame/head_template_mesh.obj")


def test_load_flame_uv_shapes():
    fuv = load_flame_uv(TEMPLATE)
    assert fuv.uv.shape == (5118, 2)
    assert fuv.uv_faces.shape == (9976, 3)
    assert fuv.vt2v.shape == (5118,)
    assert fuv.uv.min() >= 0.0 and fuv.uv.max() <= 1.0001
    # every uv-vertex resolves to a real mesh vertex
    assert int(fuv.vt2v.min()) >= 0 and int(fuv.vt2v.max()) < 5023


def test_uv_faces_recover_position_faces():
    """uv_faces routed through vt2v must reproduce the mesh's position faces.
    Template face 0 is `f 4/1 2/2 1/3` -> position verts (3, 1, 0) 0-based."""
    fuv = load_flame_uv(TEMPLATE)
    pos_face0 = fuv.vt2v[fuv.uv_faces[0]]
    assert pos_face0.tolist() == [3, 1, 0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_uv_template.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chibi.uv_template'`

- [ ] **Step 3: Write the implementation**

```python
# src/chibi/uv_template.py
"""Load FLAME's stock UV layout from head_template_mesh.obj.

FLAME's template OBJ and LAM's `shaped_mesh.obj` share topology (same 9976
faces, same vertex order), so the template's UVs transfer with no remeshing.
This module exposes that UV layout; the bake splices it onto any shaped mesh.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import torch


@dataclass
class FlameUV:
    """uv: (Nvt,2) float32 UV coords in [0,1]. uv_faces: (F,3) int64, 0-based
    indices into uv. vt2v: (Nvt,) int64 map from uv-vertex -> mesh-vertex."""
    uv: torch.Tensor
    uv_faces: torch.Tensor
    vt2v: torch.Tensor


def load_flame_uv(template_obj: str, *, n_faces: int = 9976) -> FlameUV:
    """Parse the FLAME template OBJ. Faces are `f v/vt v/vt v/vt` (1-based).
    Builds vt2v from the v/vt pairings; raises if a vt maps to two verts."""
    uvs: list[list[float]] = []
    faces_v: list[list[int]] = []
    faces_vt: list[list[int]] = []
    for line in Path(template_obj).read_text().splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == "vt":
            uvs.append([float(p[1]), float(p[2])])
        elif p[0] == "f":
            vs, vts = [], []
            for tok in p[1:4]:
                a = tok.split("/")
                vs.append(int(a[0]) - 1)
                vts.append(int(a[1]) - 1)
            faces_v.append(vs)
            faces_vt.append(vts)

    uv = torch.tensor(uvs, dtype=torch.float32)
    uv_faces = torch.tensor(faces_vt, dtype=torch.int64)
    assert uv_faces.shape[0] == n_faces, \
        f"expected {n_faces} faces, got {uv_faces.shape[0]}"
    assert uv.min() >= 0.0 and uv.max() <= 1.0001, \
        f"UVs outside [0,1]: [{float(uv.min())}, {float(uv.max())}]"

    vt2v = torch.full((uv.shape[0],), -1, dtype=torch.int64)
    for fv, fvt in zip(faces_v, faces_vt):
        for v, vt in zip(fv, fvt):
            if vt2v[vt] >= 0:
                assert int(vt2v[vt]) == v, \
                    f"uv-vertex {vt} maps to verts {int(vt2v[vt])} and {v} " \
                    "— topology mismatch"
            else:
                vt2v[vt] = v
    assert (vt2v >= 0).all(), "some uv-vertex never referenced by a face"
    return FlameUV(uv=uv, uv_faces=uv_faces, vt2v=vt2v)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_uv_template.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/chibi/uv_template.py tests/test_uv_template.py
git commit -m "feat(chibi): uv_template — load FLAME stock UV layout

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 2: Refactor bake.py — extract `bake_points`

**Files:**
- Modify: `src/chibi/bake.py`
- Test: `tests/test_bake.py` (unchanged — regression guard)

The v2 `bake_vertex_colors` hard-codes vertices as the bake targets. The texture
bake needs the same projection/occlusion/average math on *texel* points. Extract
the loop into `bake_points(points, normals, ...)` returning per-point rgb **and**
a `seen` mask. `bake_vertex_colors` keeps its O(N²) nearest-seen fallback (5023
verts — fine); the texture path will instead route unseen texels to dilation.

- [ ] **Step 1: Confirm the regression test exists and passes today**

Run: `pytest tests/test_bake.py -v`
Expected: PASS (2 passed) — `test_bake_paints_solid_color_from_one_view`,
`test_bake_occlusion_rejects_view_behind_surface`.

- [ ] **Step 2: Rewrite `bake.py`**

Replace the whole file with:

```python
# src/chibi/bake.py
"""Bake splat-render appearance onto a set of points.

`bake_points` is the core: project each point with each View's w2c +
perspective matrix, sample the splat-render image, occlusion-test against the
splat-render depth, write a normal-weighted average. `bake_vertex_colors` wraps
it for a ChibiMesh (computes vertex normals, fills unseen verts from the nearest
seen vertex). The texture bake calls `bake_points` directly on texel points.
"""
from __future__ import annotations
from collections.abc import Sequence
import math
import torch

from chibi.mesh import ChibiMesh
from chibi.camera_rig import View

_DEPTH_TOL = 0.05      # point counted occluded if this far behind the surface


def _vertex_normals(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Area-weighted vertex normals, (V,3) float32."""
    v = verts.to(torch.float32)
    f = faces.to(torch.int64)
    fn = torch.linalg.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    vn = torch.zeros_like(v)
    for k in range(3):
        vn.index_add_(0, f[:, k], fn)
    return vn / vn.norm(dim=1, keepdim=True).clamp_min(1e-9)


def bake_points(points: torch.Tensor, normals: torch.Tensor,
                images: torch.Tensor, depth: torch.Tensor,
                views: Sequence[View], *,
                fallback_rgb: tuple[float, float, float] = (0.5, 0.5, 0.5)
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """points: (N,3). normals: (N,3) (need not be unit — used only for sign).
    images: (T,H,W,3) uint8. depth: (T,H,W) float32 camera-space Z. views: T.
    Returns (rgb (N,3) float32 in [0,1], seen (N,) bool). Unseen points carry
    `fallback_rgb`; the caller decides how to fill them."""
    assert len(images) == len(views) == len(depth), \
        "images, depth and views must have equal length"
    N = points.shape[0]
    pts = points.to(torch.float32)
    nrm = normals.to(torch.float32)
    accum = torch.zeros(N, 3)
    weight = torch.zeros(N)

    for img, dep, view in zip(images, depth, views):
        size = view.image_size
        w2c = view.w2c                                    # (4,4)
        ph = torch.cat([pts, torch.ones(N, 1)], dim=1)     # (N,4)
        cam = ph @ w2c.T                                   # (N,4) camera space
        cam_z = cam[:, 2]
        tan = math.tan(view.fov_rad * 0.5)
        in_front = cam_z > 1e-4
        x_ndc = cam[:, 0] / (cam_z.clamp_min(1e-4) * tan)
        y_ndc = cam[:, 1] / (cam_z.clamp_min(1e-4) * tan)
        px = ((x_ndc * 0.5 + 0.5) * size).long()
        py = ((y_ndc * 0.5 + 0.5) * size).long()
        on_screen = (px >= 0) & (px < size) & (py >= 0) & (py < size)
        ok = in_front & on_screen
        pxc = px.clamp(0, size - 1)
        pyc = py.clamp(0, size - 1)
        surf = dep[pyc, pxc]
        visible = ok & (cam_z <= surf + _DEPTH_TOL)
        view_dir = w2c[2, :3]                              # world-space fwd
        facing = (-(nrm @ view_dir)).clamp_min(0.0)        # normal toward cam
        w = torch.where(visible, facing, torch.zeros_like(facing))
        sampled = img[pyc, pxc].to(torch.float32) / 255.0  # (N,3)
        accum += sampled * w.unsqueeze(1)
        weight += w

    rgb = torch.empty(N, 3)
    seen = weight > 1e-6
    rgb[seen] = accum[seen] / weight[seen].unsqueeze(1)
    rgb[~seen] = torch.tensor(fallback_rgb, dtype=torch.float32)
    return rgb.clamp(0.0, 1.0).to(torch.float32), seen


def bake_vertex_colors(mesh: ChibiMesh, images: torch.Tensor,
                       depth: torch.Tensor, views: Sequence[View], *,
                       fallback_rgb: tuple[float, float, float] = (0.5, 0.5, 0.5)
                       ) -> ChibiMesh:
    """v2 per-vertex bake. Unseen verts take the nearest seen vertex's colour."""
    normals = _vertex_normals(mesh.verts, mesh.faces)
    rgb, seen = bake_points(mesh.verts, normals, images, depth, views,
                            fallback_rgb=fallback_rgb)
    if (~seen).any() and seen.any():
        verts = mesh.verts.to(torch.float32)
        seen_idx = seen.nonzero(as_tuple=True)[0]
        for i in (~seen).nonzero(as_tuple=True)[0]:
            d = (verts[seen_idx] - verts[i]).norm(dim=1)
            rgb[i] = rgb[seen_idx[int(d.argmin())]]
    return ChibiMesh(verts=mesh.verts, faces=mesh.faces,
                     rgb=rgb.to(torch.float32))
```

- [ ] **Step 3: Run the regression test**

Run: `pytest tests/test_bake.py -v`
Expected: PASS (2 passed) — v2 behavior preserved.

- [ ] **Step 4: Commit**

```bash
git add src/chibi/bake.py
git commit -m "refactor(chibi): extract bake_points from bake_vertex_colors

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 3: nvdiffrast UV-space bake — `texture_bake.py`

**Files:**
- Create: `src/chibi/texture_bake.py`
- Test: `tests/test_texture_bake.py`

`rasterize_uv_attrs` rasterizes the mesh in UV space (clip pos = `uv*2-1`,
triangles = `uv_faces`) and interpolates the per-uv-vertex 3D position and
normal, yielding per-texel maps. nvdiffrast's raster origin is bottom-left; the
maps are flipped to top-origin so the texture array, the written PNG, and
pytorch3d's `TexturesUV` all agree. `bake_texture` runs `bake_points` on the
covered texels. `dilate_texture` fills UV-gutter holes.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_texture_bake.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import torch
from chibi.uv_template import FlameUV
from chibi.camera_rig import turntable_views
from chibi.texture_bake import rasterize_uv_attrs, bake_texture, dilate_texture


def _quad():
    """Unit quad in the z=0 plane facing +Z, UV mapped into [0.25,0.75]^2."""
    verts = torch.tensor([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0],
                          [-0.5, 0.5, 0.0], [0.5, 0.5, 0.0]],
                         dtype=torch.float64)
    faces = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.int64)
    uv = torch.tensor([[0.25, 0.25], [0.75, 0.25],
                       [0.25, 0.75], [0.75, 0.75]], dtype=torch.float32)
    fuv = FlameUV(uv=uv, uv_faces=faces.clone(),
                  vt2v=torch.arange(4, dtype=torch.int64))
    return verts, faces, fuv


def test_rasterize_uv_attrs_covers_uv_region():
    verts, faces, fuv = _quad()
    pos_map, nrm_map, mask = rasterize_uv_attrs(verts, faces, fuv, 64)
    assert pos_map.shape == (64, 64, 3)
    assert mask.shape == (64, 64)
    # UV square [0.25,0.75]^2 -> roughly the central quarter of the atlas
    assert mask.float().mean() > 0.15
    assert not mask[0, 0]                      # corner texel is a gutter hole


def test_bake_texture_paints_red():
    """Azim-0 view, solid-red splat image, nothing occludes -> covered texels
    bake red."""
    verts, faces, fuv = _quad()
    views = turntable_views(n_azim=1, elevs=(0.0,), dist=2.7, image_size=64)
    images = torch.zeros(1, 64, 64, 3, dtype=torch.uint8)
    images[..., 0] = 255
    depth = torch.full((1, 64, 64), 1e3)
    texture, filled = bake_texture(verts, faces, fuv, images, depth, views, 64)
    assert texture.shape == (64, 64, 3)
    red = texture[filled]
    assert red[:, 0].min() > 0.9 and red[:, 1].max() < 0.1


def test_dilate_texture_fills_hole():
    texture = torch.zeros(8, 8, 3)
    texture[..., 1] = 1.0                      # solid green
    mask = torch.ones(8, 8, dtype=torch.bool)
    mask[3:5, 3:5] = False                     # punch a 2x2 hole
    texture[~mask] = 0.0
    out = dilate_texture(texture, mask, iters=4)
    assert out[3:5, 3:5, 1].min() > 0.9        # hole filled with green
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_texture_bake.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chibi.texture_bake'`

- [ ] **Step 3: Write the implementation**

```python
# src/chibi/texture_bake.py
"""nvdiffrast UV-space bake: splat appearance -> FLAME UV texture atlas.

Rasterise the mesh in UV space once to get a per-texel canonical 3D position +
normal, run the v2 projection bake on those texel points, dilate UV-gutter
holes. nvdiffrast's raster origin is bottom-left; maps are flipped to top-origin
so the atlas, the PNG, and pytorch3d TexturesUV agree.
"""
from __future__ import annotations
from collections.abc import Sequence
import torch

from chibi.bake import bake_points, _vertex_normals
from chibi.camera_rig import View
from chibi.uv_template import FlameUV


def rasterize_uv_attrs(verts: torch.Tensor, faces: torch.Tensor,
                       flame_uv: FlameUV, tex_size: int
                       ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rasterise the mesh in UV space. Returns top-origin maps:
    pos_map (tex,tex,3), nrm_map (tex,tex,3), mask (tex,tex) bool. tex_size
    must be a multiple of 8 (nvdiffrast requirement)."""
    import nvdiffrast.torch as dr
    assert tex_size % 8 == 0, "tex_size must be a multiple of 8"
    dev = torch.device("cuda")
    uv = flame_uv.uv.to(torch.float32).to(dev)                 # (Nvt,2)
    uv_faces = flame_uv.uv_faces.to(torch.int32).contiguous().to(dev)
    vt2v = flame_uv.vt2v
    vn = _vertex_normals(verts, faces)                         # (V,3)
    attr_pos = verts.to(torch.float32)[vt2v].to(dev)           # (Nvt,3)
    attr_nrm = vn[vt2v].to(dev)                                # (Nvt,3)
    nvt = uv.shape[0]
    clip = torch.cat([uv * 2.0 - 1.0,
                      torch.zeros(nvt, 1, device=dev),
                      torch.ones(nvt, 1, device=dev)], dim=1)
    clip = clip[None].contiguous()                             # (1,Nvt,4)

    glctx = dr.RasterizeCudaContext()
    rast, _ = dr.rasterize(glctx, clip, uv_faces,
                           resolution=[tex_size, tex_size])
    pos_map, _ = dr.interpolate(attr_pos[None].contiguous(), rast, uv_faces)
    nrm_map, _ = dr.interpolate(attr_nrm[None].contiguous(), rast, uv_faces)
    mask = rast[..., 3] > 0                                    # (1,H,W)
    # nvdiffrast raster origin is bottom-left -> flip rows to top-origin.
    pos_map = torch.flip(pos_map[0], dims=[0]).cpu()
    nrm_map = torch.flip(nrm_map[0], dims=[0]).cpu()
    mask = torch.flip(mask[0], dims=[0]).cpu()
    return pos_map, nrm_map, mask


def bake_texture(verts: torch.Tensor, faces: torch.Tensor, flame_uv: FlameUV,
                 images: torch.Tensor, depth: torch.Tensor,
                 views: Sequence[View], tex_size: int
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """Bake a (tex,tex,3) float32 texture atlas. Returns (texture, filled),
    where `filled` (tex,tex) bool marks texels a view actually saw — the
    complement is what dilate_texture must fill."""
    pos_map, nrm_map, mask = rasterize_uv_attrs(verts, faces, flame_uv, tex_size)
    flat_pos = pos_map.reshape(-1, 3)
    flat_nrm = nrm_map.reshape(-1, 3)
    idx = mask.reshape(-1).nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        raise ValueError("no texel covered by the UV layout")
    nrm = flat_nrm[idx]
    nrm = nrm / nrm.norm(dim=1, keepdim=True).clamp_min(1e-9)
    rgb, seen = bake_points(flat_pos[idx], nrm, images, depth, views)

    texture = torch.full((tex_size * tex_size, 3), 0.5, dtype=torch.float32)
    texture[idx] = rgb
    filled = torch.zeros(tex_size * tex_size, dtype=torch.bool)
    filled[idx] = seen
    return (texture.reshape(tex_size, tex_size, 3),
            filled.reshape(tex_size, tex_size))


def dilate_texture(texture: torch.Tensor, mask: torch.Tensor,
                   iters: int = 16) -> torch.Tensor:
    """Fill `~mask` texels from filled 4-neighbours, iteratively. Any texel
    still unfilled after `iters` passes takes the mean of all filled texels."""
    tex = texture.clone()
    m = mask.clone()
    for _ in range(iters):
        if bool(m.all()):
            break
        acc = torch.zeros_like(tex)
        cnt = torch.zeros(m.shape, dtype=torch.float32)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            sh_t = torch.roll(tex, shifts=(dy, dx), dims=(0, 1))
            sh_m = torch.roll(m, shifts=(dy, dx), dims=(0, 1))
            acc += sh_t * sh_m.unsqueeze(-1).float()
            cnt += sh_m.float()
        newly = (~m) & (cnt > 0)
        tex[newly] = acc[newly] / cnt[newly].unsqueeze(-1)
        m = m | newly
    if not bool(m.all()) and bool(m.any()):
        tex[~m] = tex[m].mean(0)
    return tex
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_texture_bake.py -v`
Expected: PASS (3 passed). Requires CUDA + nvdiffrast (present in the `lam`
env).

- [ ] **Step 5: Commit**

```bash
git add src/chibi/texture_bake.py tests/test_texture_bake.py
git commit -m "feat(chibi): texture_bake — nvdiffrast UV-space splat bake

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 4: `TexturedMesh` + `render_textured`

**Files:**
- Modify: `src/chibi/mesh.py`
- Modify: `src/chibi/mesh_render.py`
- Test: `tests/test_textured_mesh.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_textured_mesh.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import torch
from chibi.mesh import TexturedMesh
from chibi.mesh_render import render_textured


def _quad_textured():
    verts = torch.tensor([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0],
                          [-0.5, 0.5, 0.0], [0.5, 0.5, 0.0]],
                         dtype=torch.float64)
    faces = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.int64)
    uv = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                      dtype=torch.float32)
    texture = torch.zeros(32, 32, 3, dtype=torch.float32)
    texture[..., 0] = 1.0                                   # solid red atlas
    return TexturedMesh(verts=verts, faces=faces, uv=uv,
                        uv_faces=faces.clone(), texture=texture)


def test_textured_mesh_validates_shapes():
    tm = _quad_textured()
    assert tm.uv_faces.shape == tm.faces.shape


def test_render_textured_shows_texture():
    tm = _quad_textured()
    frames = render_textured(tm, [0.0], image_size=64)
    assert frames.shape == (1, 64, 64, 3)
    # the quad faces the azim-0 camera; its centre pixel is the red atlas
    centre = frames[0, 32, 32]
    assert int(centre[0]) > 200 and int(centre[1]) < 60
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_textured_mesh.py -v`
Expected: FAIL — `ImportError: cannot import name 'TexturedMesh'`

- [ ] **Step 3: Add `TexturedMesh` to `mesh.py`**

Append to `src/chibi/mesh.py`:

```python
@dataclass
class TexturedMesh:
    """A UV-textured triangle mesh. verts: (V,3). faces: (F,3) int64 position
    triangles. uv: (Nvt,2) float32 in [0,1]. uv_faces: (F,3) int64 triangles
    into uv. texture: (H,W,3) float32 atlas in [0,1], top-origin. Chibi deform
    moves `verts` only — uv/uv_faces/texture are deformation-invariant."""
    verts: torch.Tensor
    faces: torch.Tensor
    uv: torch.Tensor
    uv_faces: torch.Tensor
    texture: torch.Tensor

    def __post_init__(self) -> None:
        assert self.verts.ndim == 2 and self.verts.shape[1] == 3, \
            f"verts must be (V,3), got {tuple(self.verts.shape)}"
        assert self.faces.ndim == 2 and self.faces.shape[1] == 3, \
            f"faces must be (F,3), got {tuple(self.faces.shape)}"
        assert self.uv.ndim == 2 and self.uv.shape[1] == 2, \
            f"uv must be (Nvt,2), got {tuple(self.uv.shape)}"
        assert self.uv_faces.shape == self.faces.shape, \
            "uv_faces must have the same shape as faces"
        assert self.texture.ndim == 3 and self.texture.shape[2] == 3, \
            f"texture must be (H,W,3), got {tuple(self.texture.shape)}"
        assert int(self.faces.min()) >= 0 and \
            int(self.faces.max()) < self.verts.shape[0], \
            "face index out of range [0, V)"
        assert int(self.uv_faces.min()) >= 0 and \
            int(self.uv_faces.max()) < self.uv.shape[0], \
            "uv_faces index out of range [0, Nvt)"
```

- [ ] **Step 4: Add `render_textured` to `mesh_render.py`**

Append to `src/chibi/mesh_render.py`:

```python
def render_textured(tmesh, azims: Sequence[float], *,
                     image_size: int = 512, dist: float = 2.7,
                     elev: float = 0.0, device: str = "cuda") -> torch.Tensor:
    """Render a TexturedMesh turntable. Unlit (AmbientLights) so the output is
    the UV-sampled atlas — same recentre/unit-scale and camera conventions as
    `render`. Returns (len(azims), image_size, image_size, 3) uint8 on CPU."""
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        TexturesUV, FoVPerspectiveCameras, RasterizationSettings,
        MeshRenderer, MeshRasterizer, SoftPhongShader, AmbientLights,
        look_at_view_transform,
    )
    dev = torch.device(device)
    verts = tmesh.verts.to(torch.float32).to(dev)
    centre = verts.mean(0, keepdim=True)
    scale = (verts - centre).abs().max().clamp_min(1e-6)
    verts = (verts - centre) / scale
    faces = tmesh.faces.to(torch.int64).to(dev)
    textures = TexturesUV(
        maps=[tmesh.texture.to(torch.float32).to(dev)],
        faces_uvs=[tmesh.uv_faces.to(torch.int64).to(dev)],
        verts_uvs=[tmesh.uv.to(torch.float32).to(dev)],
    )
    raster = RasterizationSettings(image_size=image_size, blur_radius=0.0,
                                   faces_per_pixel=1)
    lights = AmbientLights(device=dev)
    frames = []
    for azim in azims:
        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim,
                                      device=dev)
        cameras = FoVPerspectiveCameras(R=R, T=T, device=dev)
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster),
            shader=SoftPhongShader(device=dev, cameras=cameras, lights=lights),
        )
        p3d = Meshes(verts=[verts], faces=[faces], textures=textures)
        img = renderer(p3d)[0, ..., :3]
        frames.append((img.clamp(0.0, 1.0) * 255.0).round()
                       .to(torch.uint8).cpu())
    return torch.stack(frames)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_textured_mesh.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add src/chibi/mesh.py src/chibi/mesh_render.py tests/test_textured_mesh.py
git commit -m "feat(chibi): TexturedMesh + render_textured (pytorch3d TexturesUV)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 5: Driver — `bake_uv_texture.py` + verdict bakes

**Files:**
- Create: `scripts/bake_uv_texture.py`
- Create: `scripts/bake_uv_texture.sh`

The driver mirrors `bake_anchor_texture.py`: load `shaped_mesh.obj` + `cano.ply`,
recentre both by the **same translation only** (rescaling positions without
rescaling Gaussian scales turns the splats into a dot grid), render splats, bake.
It additionally splices FLAME UVs, bakes the atlas, writes OBJ/MTL/PNG, and
renders a verdict.

- [ ] **Step 1: Write the driver**

```python
# scripts/bake_uv_texture.py
"""nvdiffrast UV-texture bake driver: shaped_mesh.obj + cano.ply -> textured OBJ.

  python scripts/bake_uv_texture.py \
      --ply  /home/newub/w/LAM/exps/cano_gs/me_512_cano.ply \
      --mesh /home/newub/w/LAM/exps/cano_gs/me_512_shaped_mesh.obj \
      --out  exp_output/lam_chibi/renders/bake_v3/me_512 --stem me_512

Mesh and splats are recentred onto the origin by TRANSLATION ONLY. Writes
<stem>_textured.obj/.mtl/_texture.png plus verdict turntable renders.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
import imageio.v2 as imageio                                # noqa: E402

from chibi.mesh import TexturedMesh                         # noqa: E402
from chibi.camera_rig import turntable_views                # noqa: E402
from chibi.splat_render import load_splat_ply, render_splats  # noqa: E402
from chibi.uv_template import load_flame_uv                 # noqa: E402
from chibi.texture_bake import bake_texture, dilate_texture  # noqa: E402
from chibi.mesh_render import render_textured               # noqa: E402

FLAME_TEMPLATE = ("/home/newub/w/LAM/model_zoo/human_parametric_models/"
                  "flame_assets/flame/head_template_mesh.obj")


def _load_plain_obj(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Parse a plain `v x y z` + `f` OBJ (trimesh export — no per-vertex rgb)."""
    verts, faces = [], []
    for line in Path(path).read_text().splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == "v":
            verts.append([float(x) for x in p[1:4]])
        elif p[0] == "f":
            faces.append([int(x.split("/")[0]) - 1 for x in p[1:4]])
    return (torch.tensor(verts, dtype=torch.float64),
            torch.tensor(faces, dtype=torch.int64))


def _write_textured_obj(out: Path, stem: str, verts: torch.Tensor,
                        faces: torch.Tensor, uv: torch.Tensor,
                        uv_faces: torch.Tensor) -> None:
    """Write <stem>_textured.obj + .mtl referencing <stem>_texture.png."""
    obj = [f"mtllib {stem}_textured.mtl", f"usemtl {stem}_mat"]
    for v in verts.tolist():
        obj.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
    for t in uv.tolist():
        obj.append(f"vt {t[0]:.6f} {t[1]:.6f}")
    for fp, ft in zip((faces + 1).tolist(), (uv_faces + 1).tolist()):
        obj.append(f"f {fp[0]}/{ft[0]} {fp[1]}/{ft[1]} {fp[2]}/{ft[2]}")
    (out / f"{stem}_textured.obj").write_text("\n".join(obj) + "\n")
    (out / f"{stem}_textured.mtl").write_text(
        f"newmtl {stem}_mat\nmap_Kd {stem}_texture.png\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True, help="<stem>_cano.ply")
    ap.add_argument("--mesh", required=True, help="<stem>_shaped_mesh.obj")
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--stem", required=True, help="anchor name for filenames")
    ap.add_argument("--n_azim", type=int, default=12)
    ap.add_argument("--splat_size", type=int, default=1024)
    ap.add_argument("--tex_size", type=int, default=2048)
    ap.add_argument("--render_size", type=int, default=512)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    verts, faces = _load_plain_obj(args.mesh)
    splats = load_splat_ply(args.ply)
    centre = verts.mean(0, keepdim=True)
    verts = verts - centre
    splats.xyz = (splats.xyz.double() - centre).to(torch.float32)
    extent = float((verts.max(0).values - verts.min(0).values).max())
    dist = extent * 2.2

    fuv = load_flame_uv(FLAME_TEMPLATE)
    assert fuv.uv_faces.shape[0] == faces.shape[0], \
        "FLAME template / shaped_mesh face count mismatch"

    views = turntable_views(n_azim=args.n_azim, elevs=(-20.0, 0.0, 20.0),
                            dist=dist, image_size=args.splat_size)
    images, depth = render_splats(splats, views)
    imageio.imwrite(out / "splat_ref.png", images[args.n_azim // 2].numpy())

    texture, filled = bake_texture(verts, faces, fuv, images, depth, views,
                                   args.tex_size)
    texture = dilate_texture(texture, filled, iters=16)

    tex_png = (texture.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).numpy()
    imageio.imwrite(out / f"{args.stem}_texture.png", tex_png)
    _write_textured_obj(out, args.stem, verts, faces, fuv.uv, fuv.uv_faces)

    tmesh = TexturedMesh(verts=verts, faces=faces, uv=fuv.uv,
                         uv_faces=fuv.uv_faces, texture=texture)
    frames = render_textured(tmesh, [0.0, 30.0, 60.0, 90.0],
                             image_size=args.render_size)
    for i, az in enumerate((0, 30, 60, 90)):
        imageio.imwrite(out / f"textured_mesh_{az:03d}.png", frames[i].numpy())

    # best-effort side-by-side vs the v2 vertex bake, if it exists
    v2 = Path(f"exp_output/lam_chibi/renders/bake_v1/{args.stem}/"
              "baked_mesh_000.png")
    if v2.exists():
        a = imageio.imread(v2)
        b = frames[0].numpy()
        if a.shape[:2] != b.shape[:2]:
            import PIL.Image
            a = np.asarray(PIL.Image.fromarray(a).resize(
                (b.shape[1], b.shape[0])))
        imageio.imwrite(out / "sidebyside_v2_v3.png",
                        np.concatenate([a[..., :3], b], axis=1))
    print(f"[uv-bake] {args.stem}: {int(filled.sum())}/{filled.numel()} texels "
          f"seen, dist={dist:.3f} -> {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the shell wrapper**

```bash
# scripts/bake_uv_texture.sh
#!/usr/bin/env bash
# nvdiffrast UV-texture bake driver. Activates the lam conda env.
#   bash scripts/bake_uv_texture.sh STEM
set -eu

STEM=${1:?usage: $0 STEM   (e.g. me_512, asian_m or status)}
VAMP=/home/newub/w/vamp-interface
LAM=/home/newub/w/LAM

source /home/newub/miniconda3/etc/profile.d/conda.sh
conda activate lam
export PATH=/home/newub/miniconda3/envs/lam/bin:$PATH
export PYTHONPATH=${LAM}
export XFORMERS_DISABLED=1

python "${VAMP}/scripts/bake_uv_texture.py" \
    --ply  "${LAM}/exps/cano_gs/${STEM}_cano.ply" \
    --mesh "${LAM}/exps/cano_gs/${STEM}_shaped_mesh.obj" \
    --out  "${VAMP}/exp_output/lam_chibi/renders/bake_v3/${STEM}" \
    --stem "${STEM}"
```

- [ ] **Step 3: Run the verdict bakes**

Run:
```bash
bash scripts/bake_uv_texture.sh status
bash scripts/bake_uv_texture.sh asian_m
bash scripts/bake_uv_texture.sh me_512
```
Expected: each prints `[uv-bake] <stem>: N/4194304 texels seen, dist=... -> ...`
and writes `<stem>_textured.obj/.mtl/_texture.png`, `textured_mesh_{000,030,060,090}.png`,
`splat_ref.png`, `sidebyside_v2_v3.png` under
`exp_output/lam_chibi/renders/bake_v3/<stem>/`.

If any anchor lacks `<stem>_cano.ply` / `<stem>_shaped_mesh.obj`, re-run clean
no-chibi LAM inference for it first (the v2 procedure).

- [ ] **Step 4: Commit**

```bash
git add scripts/bake_uv_texture.py scripts/bake_uv_texture.sh
git commit -m "feat(chibi): bake_uv_texture — nvdiffrast UV-bake driver

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

- [ ] **Step 5: Verdict gate (manual)**

Open `sidebyside_v2_v3.png` for `status`, `asian_m`, `me_512`. Pass when v3
(right) shows fine detail (eyebrow/skin texture) that v2 (left) lacks, with no
UV-seam bleed and no background halo. Per the standing instruction, any trace of
a seam/coverage problem here is the signal to fall back to the optimization bake
(spec Approach A) — no patching. Report the verdict before declaring done.

---

## Self-Review Notes

- **Spec coverage:** uv_template (Task 1) ↔ "FLAME UV transfers"; bake_points
  refactor (Task 2) ↔ "reuse v2 projection math"; texture_bake (Task 3) ↔ the B
  decision + dilation + 2048²/1024²; TexturedMesh + render_textured (Task 4) ↔
  output + verdict render; driver (Task 5) ↔ OBJ/MTL/PNG output + verdict gate.
- **Code review:** after Task 5, before declaring done, dispatch
  `superpowers:code-reviewer` over the diff (standing rule).
- **Out of scope:** hair/clothing flat blobs; chibi deform (#63/#65) stays gated
  downstream.
