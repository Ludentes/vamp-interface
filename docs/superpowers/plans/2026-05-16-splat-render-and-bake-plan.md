# Splat Render-and-Bake Color Path — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bake the LAM Gaussian-splat avatar's appearance onto the FLAME mesh by rendering the canonical splats from many controlled views and projecting those rendered pixels onto the mesh vertices — replacing the blotchy per-splat color the Milestone-0 gate rejected.

**Architecture:** Three small units in `src/chibi/` behind the existing `ChibiMesh` interface. `camera_rig` produces a turntable view set; `splat_render` rasterises a canonical 3DGS `.ply` from those views with LAM's own `diff_gaussian_rasterization`; `bake` projects each mesh vertex into every view, occlusion-tests against the splat-render depth, and writes a normal-weighted average color into `ChibiMesh.rgb`. One 3DGS camera convention throughout — no pytorch3d cameras in the bake. A one-line LAM change emits the canonical `.ply`.

**Tech Stack:** Python 3.12, PyTorch, `diff_gaussian_rasterization` (LAM's splat rasteriser), plyfile, numpy, imageio. All code in `src/chibi/`. All tests and runs in the `lam` conda env.

---

## Test environment

Run every `pytest` and script command as:

```bash
export PATH=/home/newub/miniconda3/envs/lam/bin:$PATH
cd /home/newub/w/vamp-interface
python -m pytest <args>
```

Scripts that touch `diff_gaussian_rasterization` also need `PYTHONPATH=/home/newub/w/LAM` (to import LAM's `gs_renderer` camera helpers) and `XFORMERS_DISABLED=1`.

## Background the engineer needs

- **Milestone 0 result** — splats→mesh geometry is clean; per-vertex splat color (`f_dc`) is not. See `docs/research/2026-05-16-splat-appearance-baking.md` and the spec `docs/superpowers/specs/2026-05-16-splat-render-and-bake-design.md`.
- **`ChibiMesh`** (`src/chibi/mesh.py`) — dataclass `verts (V,3) float64`, `faces (F,3) int64`, `rgb (V,3) float32` in `[0,1]`. `__post_init__` validates shapes and face-index range.
- **`mesh_render.render`** (`src/chibi/mesh_render.py`) — already renders a `ChibiMesh` turntable with pytorch3d; the bake fills `rgb`, then this renders the verdict.
- **`load_chibi_mesh`** (`src/chibi/mesh_extract.py`) — OBJ parser; used here to load `shaped_mesh.obj` (a plain `v x y z` OBJ — note: NOT 7-token, see Task 3 step 1).
- **LAM canonical mesh** — `exps/cano_gs/<stem>_shaped_mesh.obj` = `xyz − offset`, the clean 5023-vert / 9976-face base FLAME mesh. Plain `v x y z` lines (trimesh export).
- **LAM 3DGS `.ply` format** (`lam/models/rendering/gaussian_model.py:58` `save_ply`) — PLY `vertex` element: `x y z`, `nx ny nz`, `f_dc_0..2`, `opacity`, `scale_0..2`, `rot_0..3`. Stored encodings: `opacity` is `inverse_sigmoid`-ed (decode with `sigmoid`), `scale_*` is `log`-ged (decode with `exp`), `rot_*` is a raw quaternion, `f_dc_*` is direct RGB (`gs_use_rgb` mode — no SH).
- **LAM camera helpers** (`lam/models/rendering/gs_renderer.py`) — `getWorld2View2(R, t)`, `getProjectionMatrix(znear, zfar, fovX, fovY)`, `class Camera(w2c, intrinsic, FoVx, FoVy, height, width)`. `Camera` builds `world_view_transform`, `full_proj_transform`, `camera_center`. The rasteriser is `GaussianRasterizationSettings` / `GaussianRasterizer` from `diff_gaussian_rasterization`; LAM's call site is `gs_renderer.py:518-575` — copy its settings exactly.
- **LAM canonical-gs dump** — `lam/runners/infer/lam.py` ~line 380 has a commented-out `cano_gs_lst[0].save_ply(...)`. Task 4 adds an uncommented renderable dump.
- **Anchors** — `exp_output/lam_chibi/user_anchor/me_512.png` (the `me` anchor), `assets/sample_input/status.png` (LAM reference, in the LAM repo).

## File structure

| File | Responsibility |
|------|----------------|
| `src/chibi/camera_rig.py` | `View` dataclass + `turntable_views(...)` — the shared 3DGS camera set. |
| `src/chibi/splat_render.py` | `load_splat_ply(path)` + `render_splats(splats, views)` — rasterise a 3DGS ply from each view. |
| `src/chibi/bake.py` | `bake_vertex_colors(mesh, images, depth, views)` — project + occlusion-test + average → `ChibiMesh`. |
| `scripts/bake_anchor_texture.py` + `.sh` | Driver: canonical ply + mesh → bake → textured render + side-by-side. |
| `tests/test_camera_rig.py` | Tests for `turntable_views`. |
| `tests/test_splat_render.py` | Test for `render_splats` (synthetic 1-Gaussian ply). |
| `tests/test_bake.py` | Tests for `bake_vertex_colors` (synthetic flat quad + occlusion). |

## Task order

Task 1 (camera rig) and Task 2 (splat render) build the inputs; they have a joint visual sanity check at the end of Task 2 because camera-convention correctness cannot be unit-tested without rendering. Task 3 (bake) has synthetic unit tests. Task 4 wires the LAM dump and the end-to-end driver.

---

### Task 1: `camera_rig` — the shared turntable view set

**Files:**
- Create: `src/chibi/camera_rig.py`
- Test: `tests/test_camera_rig.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_camera_rig.py`:

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import math
import torch
from chibi.camera_rig import turntable_views, View


def test_turntable_view_count():
    views = turntable_views(n_azim=12, elevs=(-20.0, 0.0, 20.0), dist=2.7)
    assert len(views) == 36
    assert all(isinstance(v, View) for v in views)


def test_view_has_w2c_and_fov():
    v = turntable_views(n_azim=4, elevs=(0.0,), dist=2.7)[0]
    assert v.w2c.shape == (4, 4)
    assert v.w2c.dtype == torch.float32
    assert 0.0 < v.fov_rad < math.pi
    assert v.image_size == 512


def test_camera_centers_on_sphere():
    """Every camera sits at radius `dist` from the origin it looks at."""
    dist = 2.7
    for v in turntable_views(n_azim=8, elevs=(0.0, 30.0), dist=dist):
        c2w = torch.inverse(v.w2c)
        centre = c2w[:3, 3]
        assert abs(float(centre.norm()) - dist) < 1e-3, float(centre.norm())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_camera_rig.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chibi.camera_rig'`

- [ ] **Step 3: Write minimal implementation**

Create `src/chibi/camera_rig.py`:

```python
"""The shared camera set for splat render-and-bake.

A `View` is a 3DGS-convention world-to-camera matrix plus a field of view. The
same View list drives both splat_render (rasterises the Gaussians) and bake
(projects mesh vertices) — one convention, so a vertex projects to the pixel it
was rendered at. Cameras orbit the origin on a sphere; the canonical avatar is
recentred onto the origin by the bake driver before rendering.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch


@dataclass
class View:
    """w2c: (4,4) float32 world->camera matrix (3DGS/COLMAP convention,
    row-vector: p_cam_h = p_world_h @ w2c). fov_rad: symmetric vertical+
    horizontal FoV in radians (square image). image_size: pixels per side."""
    w2c: torch.Tensor
    fov_rad: float
    image_size: int


def _look_at_w2c(azim_deg: float, elev_deg: float, dist: float) -> torch.Tensor:
    """World->camera matrix for a camera at (azim, elev) on a sphere of radius
    `dist`, looking at the origin, with world +Y up."""
    az = math.radians(azim_deg)
    el = math.radians(elev_deg)
    # camera position in world space
    eye = torch.tensor([
        dist * math.cos(el) * math.sin(az),
        dist * math.sin(el),
        dist * math.cos(el) * math.cos(az),
    ], dtype=torch.float32)
    fwd = (-eye) / eye.norm()                       # look direction (toward origin)
    up0 = torch.tensor([0.0, 1.0, 0.0])
    right = torch.linalg.cross(fwd, up0)
    right = right / right.norm()
    up = torch.linalg.cross(right, fwd)
    # camera axes as rows of the rotation (world -> camera): x=right, y=-up, z=fwd
    R = torch.stack([right, -up, fwd], dim=0)       # (3,3)
    t = -R @ eye                                    # (3,)
    w2c = torch.eye(4, dtype=torch.float32)
    w2c[:3, :3] = R
    w2c[:3, 3] = t
    return w2c


def turntable_views(n_azim: int = 12, elevs=(-20.0, 0.0, 20.0),
                     dist: float = 2.7, fov_deg: float = 40.0,
                     image_size: int = 512) -> list[View]:
    """`n_azim` azimuths evenly around the circle, crossed with each elevation
    in `elevs`. Default 12x3 = 36 views covering a head incl. crown/under-chin."""
    fov = math.radians(fov_deg)
    views: list[View] = []
    for elev in elevs:
        for i in range(n_azim):
            azim = 360.0 * i / n_azim
            views.append(View(w2c=_look_at_w2c(azim, elev, dist),
                              fov_rad=fov, image_size=image_size))
    return views
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_camera_rig.py -v`
Expected: PASS — 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/chibi/camera_rig.py tests/test_camera_rig.py
git commit -m "feat(chibi): camera_rig — shared turntable view set"
```

---

### Task 2: `splat_render` — rasterise a 3DGS ply from the views

**Files:**
- Create: `src/chibi/splat_render.py`
- Test: `tests/test_splat_render.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_splat_render.py`:

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import pytest
import numpy as np
import torch

pytest.importorskip("diff_gaussian_rasterization",
                    reason="splat rasteriser only in the lam conda env")
from chibi.camera_rig import turntable_views
from chibi.splat_render import load_splat_ply, render_splats


def _one_gaussian_ply(tmp_path):
    """A single fat bright-green Gaussian at the origin."""
    from plyfile import PlyData, PlyElement
    cols = ("x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
            "opacity", "scale_0", "scale_1", "scale_2",
            "rot_0", "rot_1", "rot_2", "rot_3")
    arr = np.zeros(1, dtype=[(c, "f4") for c in cols])
    arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = 0.0, 1.0, 0.0   # green RGB
    arr["opacity"] = 8.0          # inverse_sigmoid-space -> ~1.0 after sigmoid
    arr["scale_0"] = arr["scale_1"] = arr["scale_2"] = -1.0  # log-space -> exp
    arr["rot_0"] = 1.0            # identity quaternion
    ply = tmp_path / "one.ply"
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(ply))
    return ply


def test_load_splat_ply_decodes_fields(tmp_path):
    s = load_splat_ply(str(_one_gaussian_ply(tmp_path)))
    assert s.xyz.shape == (1, 3)
    assert s.rgb.shape == (1, 3)
    assert torch.allclose(s.rgb[0], torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)
    assert 0.99 < float(s.opacity[0]) <= 1.0          # sigmoid(8) ~ 0.9997
    assert torch.allclose(s.scaling[0],
                          torch.full((3,), float(np.exp(-1.0))), atol=1e-5)


def test_render_splats_shape_and_color(tmp_path):
    s = load_splat_ply(str(_one_gaussian_ply(tmp_path)))
    views = turntable_views(n_azim=2, elevs=(0.0,), dist=2.7, image_size=64)
    rgb, depth = render_splats(s, views)
    assert rgb.shape == (2, 64, 64, 3) and rgb.dtype == torch.uint8
    assert depth.shape == (2, 64, 64)
    centre = rgb[0, 32, 32]
    assert centre[1] > 150 and centre[0] < 90 and centre[2] < 90, centre.tolist()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/home/newub/w/LAM python -m pytest tests/test_splat_render.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chibi.splat_render'`

- [ ] **Step 3: Write minimal implementation**

Create `src/chibi/splat_render.py`:

```python
"""Render a canonical LAM Gaussian-splat .ply from a set of Views.

Uses LAM's own diff_gaussian_rasterization + Camera helpers so colour, opacity
and scale conventions match LAM's renders exactly. LAM runs gs_use_rgb, so the
.ply's f_dc_* is direct RGB and is passed as colors_precomp (no SH).
"""
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Sequence
import math
import numpy as np
import torch

from chibi.camera_rig import View


@dataclass
class Splats:
    """Decoded 3DGS state. xyz (N,3), rgb (N,3) in [0,1], opacity (N,1),
    scaling (N,3), rotation (N,4) quaternion — all float32, all activated
    (sigmoid/exp already applied), ready for the rasteriser."""
    xyz: torch.Tensor
    rgb: torch.Tensor
    opacity: torch.Tensor
    scaling: torch.Tensor
    rotation: torch.Tensor


def load_splat_ply(ply_path: str) -> Splats:
    """Parse a LAM 3DGS .ply and decode the stored encodings: opacity is
    inverse-sigmoid'd, scale is log'd, rotation is a raw quaternion, f_dc is
    direct RGB."""
    from plyfile import PlyData
    el = PlyData.read(ply_path)["vertex"]
    names = set(el.data.dtype.names)
    assert {"scale_0", "rot_0", "opacity"} <= names, (
        f"{ply_path} is not a renderable 3DGS ply (missing scale/rot/opacity) "
        f"— did you pass _gs_offset.ply by mistake?")

    def col(*cs):
        return np.stack([el[c] for c in cs], axis=1).astype(np.float32)

    xyz = torch.from_numpy(col("x", "y", "z"))
    rgb = torch.from_numpy(col("f_dc_0", "f_dc_1", "f_dc_2")).clamp(0.0, 1.0)
    opacity = torch.sigmoid(torch.from_numpy(col("opacity")))          # (N,1)
    scaling = torch.exp(torch.from_numpy(col("scale_0", "scale_1", "scale_2")))
    rotation = torch.from_numpy(col("rot_0", "rot_1", "rot_2", "rot_3"))
    return Splats(xyz=xyz, rgb=rgb, opacity=opacity,
                  scaling=scaling, rotation=rotation)


def render_splats(splats: Splats, views: Sequence[View], *,
                  device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """Rasterise `splats` from each View. Returns (rgb (T,H,W,3) uint8 CPU,
    depth (T,H,W) float32 CPU) — depth is camera-space Z from the rasteriser."""
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings, GaussianRasterizer)
    from lam.models.rendering.gs_renderer import Camera

    dev = torch.device(device)
    xyz = splats.xyz.to(dev)
    rgb = splats.rgb.to(dev)
    opacity = splats.opacity.to(dev)
    scaling = splats.scaling.to(dev)
    rotation = splats.rotation.to(dev)
    means2d = torch.zeros_like(xyz, requires_grad=False)

    rgb_out, depth_out = [], []
    for v in views:
        size = v.image_size
        w2c = v.w2c.to(dev)
        # FoVx == FoVy (square image, symmetric); intrinsic only feeds Camera's
        # unused-here fields, so a focal consistent with fov is enough.
        focal = 0.5 * size / math.tan(v.fov_rad * 0.5)
        intrinsic = torch.tensor([[focal, 0, size / 2],
                                  [0, focal, size / 2],
                                  [0, 0, 1]], dtype=torch.float32, device=dev)
        cam = Camera(w2c=w2c, intrinsic=intrinsic, FoVx=v.fov_rad,
                     FoVy=v.fov_rad, height=size, width=size)
        raster = GaussianRasterizationSettings(
            image_height=size, image_width=size,
            tanfovx=math.tan(v.fov_rad * 0.5),
            tanfovy=math.tan(v.fov_rad * 0.5),
            bg=torch.zeros(3, device=dev), scale_modifier=1.0,
            viewmatrix=cam.world_view_transform,
            projmatrix=cam.full_proj_transform.float(),
            sh_degree=0, campos=cam.camera_center,
            prefiltered=False, debug=False)
        rasterizer = GaussianRasterizer(raster_settings=raster)
        with torch.autocast(device_type=dev.type, dtype=torch.float32):
            image, _radii, depth, _alpha = rasterizer(
                means3D=xyz, means2D=means2d, shs=None,
                colors_precomp=rgb, opacities=opacity,
                scales=scaling, rotations=rotation, cov3D_precomp=None)
        img = image.permute(1, 2, 0).clamp(0.0, 1.0)         # (H,W,3)
        rgb_out.append((img * 255.0).round().to(torch.uint8).cpu())
        depth_out.append(depth.squeeze(0).float().cpu())     # (H,W)
    return torch.stack(rgb_out), torch.stack(depth_out)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/home/newub/w/LAM python -m pytest tests/test_splat_render.py -v`
Expected: PASS — 2 passed

- [ ] **Step 5: Visual sanity check — camera convention**

This is the one thing unit tests cannot catch: a mirrored or upside-down camera convention. Render the real canonical splats (produced by Task 4 step 1 — if not yet present, do this step after Task 4 step 1) and eyeball the front view:

```bash
export PATH=/home/newub/miniconda3/envs/lam/bin:$PATH
cd /home/newub/w/vamp-interface
PYTHONPATH=/home/newub/w/LAM XFORMERS_DISABLED=1 python -c "
import sys; sys.path.insert(0, 'src')
import imageio.v2 as imageio
from chibi.camera_rig import turntable_views
from chibi.splat_render import load_splat_ply, render_splats
s = load_splat_ply('/home/newub/w/LAM/exps/cano_gs/status_cano.ply')
views = turntable_views(n_azim=4, elevs=(0.0,), dist=2.7, image_size=512)
rgb, _ = render_splats(s, views)
for i in range(4):
    imageio.imwrite(f'/tmp/splat_view_{i}.png', rgb[i].numpy())
print('wrote /tmp/splat_view_0..3.png')
"
```
Expected: `splat_view_0.png` is an upright, front-facing head (not mirrored, not upside-down); views 1–3 orbit it. If mirrored/inverted, fix the axis signs in `camera_rig._look_at_w2c` (the `R` rows / `-up`) and re-run — do not proceed to Task 3 until the front view is correct.

- [ ] **Step 6: Commit**

```bash
git add src/chibi/splat_render.py tests/test_splat_render.py
git commit -m "feat(chibi): splat_render — rasterise a 3DGS ply from views"
```

---

### Task 3: `bake` — project mesh verts, occlusion-test, average color

**Files:**
- Create: `src/chibi/bake.py`
- Test: `tests/test_bake.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_bake.py`:

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import torch
from chibi.mesh import ChibiMesh
from chibi.camera_rig import turntable_views
from chibi.bake import bake_vertex_colors


def _front_quad():
    """A unit quad in the z=0 plane, facing +Z (toward an azim-0 camera)."""
    verts = torch.tensor([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0],
                          [-0.5, 0.5, 0.0], [0.5, 0.5, 0.0]],
                         dtype=torch.float64)
    faces = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.int64)
    rgb = torch.zeros(4, 3, dtype=torch.float32)
    return ChibiMesh(verts=verts, faces=faces, rgb=rgb)


def test_bake_paints_solid_color_from_one_view():
    """One azim-0 view, a solid-red image, infinite depth (nothing occludes):
    every front-facing vertex takes the red."""
    mesh = _front_quad()
    views = turntable_views(n_azim=1, elevs=(0.0,), dist=2.7, image_size=64)
    images = torch.zeros(1, 64, 64, 3, dtype=torch.uint8)
    images[..., 0] = 255                                   # solid red
    depth = torch.full((1, 64, 64), 1e3)                   # nothing in front
    out = bake_vertex_colors(mesh, images, depth, views)
    assert torch.equal(out.faces, mesh.faces)
    assert out.rgb[:, 0].min() > 0.9        # all verts red
    assert out.rgb[:, 1].max() < 0.1

def test_bake_occlusion_rejects_view_behind_surface():
    """If the splat-depth map says the surface is much closer than the vertex,
    the vertex is occluded and does not take that view's color."""
    mesh = _front_quad()
    views = turntable_views(n_azim=1, elevs=(0.0,), dist=2.7, image_size=64)
    images = torch.zeros(1, 64, 64, 3, dtype=torch.uint8)
    images[..., 0] = 255
    depth = torch.full((1, 64, 64), 0.01)   # surface right at the lens
    out = bake_vertex_colors(mesh, images, depth, views,
                             fallback_rgb=(0.0, 0.0, 1.0))
    # every view rejected -> all verts fall back to blue
    assert out.rgb[:, 2].min() > 0.9 and out.rgb[:, 0].max() < 0.1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_bake.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chibi.bake'`

- [ ] **Step 3: Write minimal implementation**

Create `src/chibi/bake.py`:

```python
"""Bake splat-render appearance onto a mesh's vertices.

Single 3DGS camera convention — projects each vertex with the View's w2c +
perspective matrix, samples the splat-render image, occlusion-tests against the
splat-render depth, and writes a normal-weighted average into ChibiMesh.rgb.
Pure function over ChibiMesh + tensors; no LAM, no pytorch3d.
"""
from __future__ import annotations
from collections.abc import Sequence
import math
import torch

from chibi.mesh import ChibiMesh
from chibi.camera_rig import View

_DEPTH_TOL = 0.05      # vertex counted occluded if this far behind the surface


def _vertex_normals(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Area-weighted vertex normals, (V,3) float32."""
    v = verts.to(torch.float32)
    f = faces.to(torch.int64)
    fn = torch.linalg.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    vn = torch.zeros_like(v)
    for k in range(3):
        vn.index_add_(0, f[:, k], fn)
    return vn / vn.norm(dim=1, keepdim=True).clamp_min(1e-9)


def bake_vertex_colors(mesh: ChibiMesh, images: torch.Tensor,
                       depth: torch.Tensor, views: Sequence[View], *,
                       fallback_rgb: tuple[float, float, float] = (0.5, 0.5, 0.5)
                       ) -> ChibiMesh:
    """images: (T,H,W,3) uint8. depth: (T,H,W) float32 camera-space Z.
    views: T Views. Returns a new ChibiMesh with baked per-vertex rgb."""
    assert len(images) == len(views) == len(depth), \
        "images, depth and views must have equal length"
    V = mesh.verts.shape[0]
    verts = mesh.verts.to(torch.float32)
    normals = _vertex_normals(mesh.verts, mesh.faces)
    accum = torch.zeros(V, 3)
    weight = torch.zeros(V)

    for img, dep, view in zip(images, depth, views):
        size = view.image_size
        w2c = view.w2c                                   # (4,4)
        vh = torch.cat([verts, torch.ones(V, 1)], dim=1)  # (V,4)
        cam = vh @ w2c                                    # (V,4) camera space
        cam_z = cam[:, 2]
        # perspective project: x_ndc = x/(z*tan), in [-1,1]
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
        # occlusion: splat surface at this pixel vs the vertex depth
        surf = dep[pyc, pxc]
        visible = ok & (cam_z <= surf + _DEPTH_TOL)
        # facing weight: vertex normal vs view direction (camera looks +Z in
        # camera space, so the world view dir is w2c row 2's transpose)
        view_dir = w2c[:3, 2]                            # world-space fwd
        facing = (-(normals @ view_dir)).clamp_min(0.0)  # normal toward camera
        w = torch.where(visible, facing, torch.zeros_like(facing))
        sampled = img[pyc, pxc].to(torch.float32) / 255.0  # (V,3)
        accum += sampled * w.unsqueeze(1)
        weight += w

    baked = torch.empty(V, 3)
    seen = weight > 1e-6
    baked[seen] = accum[seen] / weight[seen].unsqueeze(1)
    if (~seen).any():
        # fill zero-visibility verts with the nearest seen vertex's colour
        fb = torch.tensor(fallback_rgb, dtype=torch.float32)
        if seen.any():
            seen_idx = seen.nonzero(as_tuple=True)[0]
            for i in (~seen).nonzero(as_tuple=True)[0]:
                d = (verts[seen_idx] - verts[i]).norm(dim=1)
                baked[i] = baked[seen_idx[int(d.argmin())]]
        else:
            baked[:] = fb
    return ChibiMesh(verts=mesh.verts, faces=mesh.faces,
                     rgb=baked.clamp(0.0, 1.0).to(torch.float32))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_bake.py -v`
Expected: PASS — 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/chibi/bake.py tests/test_bake.py
git commit -m "feat(chibi): bake — project + occlusion-test + average vertex colour"
```

---

### Task 4: LAM canonical-ply dump + end-to-end bake driver

**Files:**
- Modify: `/home/newub/w/LAM/lam/runners/infer/lam.py` (~line 382, the cano-dump block)
- Create: `scripts/bake_anchor_texture.py`
- Create: `scripts/bake_anchor_texture.sh`

- [ ] **Step 1: Add the canonical renderable-ply dump to LAM**

In `/home/newub/w/LAM/lam/runners/infer/lam.py`, find the cano-dump block (the line `res['cano_gs_lst'][0].save_ply(cano_ply_pth, rgb2sh=False, offset2xyz=True)` writing `_gs_offset.ply`). Immediately after it, add:

```python
            # renderable canonical 3DGS (positions, not offsets) for the bake
            cano_render_pth = os.path.join(
                dump_cano_dir, os.path.basename(dump_image_dir) + "_cano.ply")
            res['cano_gs_lst'][0].save_ply(cano_render_pth,
                                           rgb2sh=False, offset2xyz=False)
```

- [ ] **Step 2: Emit the canonical ply for both anchors**

Run a clean (no-chibi) LAM inference for each anchor; it now also writes `<stem>_cano.ply`:

```bash
cd /home/newub/w/LAM
source /home/newub/miniconda3/etc/profile.d/conda.sh && conda activate lam
export PATH=/home/newub/miniconda3/envs/lam/bin:$PATH
export PYTHONPATH=/home/newub/w/LAM XFORMERS_DISABLED=1
unset LAM_CHIBI_ARKIT_BS LAM_EDIT_XYZ_OBJ LAM_EDIT_VERTEX_COLORS_OBJ \
      LAM_AUX_SPLATS_PLY LAM_CHIBI_SCALE_RATIO LAM_CHIBI_SCALE_BOOST LAM_USE_ARKIT
bash scripts/inference.sh   # LAM reference: status.png
bash scripts/inference.sh configs/inference/lam-20k-8gpu.yaml \
  model_zoo/lam_models/releases/lam/lam-20k/step_045500/ \
  /home/newub/w/vamp-interface/exp_output/lam_chibi/user_anchor/me_512.png \
  assets/sample_motion/export/Look_In_My_Eyes/
```
Expected: `exps/cano_gs/status_cano.ply` and `exps/cano_gs/me_512_cano.ply` exist, each a 3DGS ply with `scale_*`/`rot_*`/`opacity`.

- [ ] **Step 3: Do Task 2 step 5 now** — run the camera-convention visual sanity check against `status_cano.ply`. Do not continue until the front view is upright and unmirrored.

- [ ] **Step 4: Write the bake driver**

Create `scripts/bake_anchor_texture.py`:

```python
"""Render-and-bake driver: canonical splat ply + FLAME mesh -> textured mesh.

  python scripts/bake_anchor_texture.py \
      --ply  /home/newub/w/LAM/exps/cano_gs/me_512_cano.ply \
      --mesh /home/newub/w/LAM/exps/cano_gs/me_512_shaped_mesh.obj \
      --out  exp_output/lam_chibi/renders/bake_v1/me
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch                                              # noqa: E402
import imageio.v2 as imageio                              # noqa: E402

from chibi.mesh import ChibiMesh                          # noqa: E402
from chibi.camera_rig import turntable_views              # noqa: E402
from chibi.splat_render import load_splat_ply, render_splats  # noqa: E402
from chibi.bake import bake_vertex_colors                 # noqa: E402
from chibi.mesh_render import render                      # noqa: E402


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True, help="<stem>_cano.ply")
    ap.add_argument("--mesh", required=True, help="<stem>_shaped_mesh.obj")
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--n_azim", type=int, default=12)
    ap.add_argument("--render_size", type=int, default=512)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    verts, faces = _load_plain_obj(args.mesh)
    # recentre the mesh + splats onto the origin the cameras orbit
    centre = verts.mean(0, keepdim=True)
    scale = (verts - centre).abs().max()
    splats = load_splat_ply(args.ply)
    splats.xyz = ((splats.xyz.double() - centre) / scale).to(torch.float32)
    nverts = (verts - centre) / scale
    mesh = ChibiMesh(verts=nverts, faces=faces,
                     rgb=torch.full((verts.shape[0], 3), 0.5,
                                    dtype=torch.float32))

    views = turntable_views(n_azim=args.n_azim, elevs=(-20.0, 0.0, 20.0),
                            dist=2.7, image_size=args.render_size)
    images, depth = render_splats(splats, views)
    imageio.imwrite(out / "splat_ref.png", images[args.n_azim // 2].numpy())

    baked = bake_vertex_colors(mesh, images, depth, views)
    frames = render([baked], [0.0, 30.0, 60.0, 90.0], image_size=256)
    for i, az in enumerate((0, 30, 60, 90)):
        imageio.imwrite(out / f"baked_mesh_{az:03d}.png", frames[i].numpy())
    print(f"[bake] {baked.verts.shape[0]} verts -> {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Write the env wrapper**

Create `scripts/bake_anchor_texture.sh`:

```bash
#!/usr/bin/env bash
# Render-and-bake driver. Activates the lam conda env (diff_gaussian_
# rasterization + pytorch3d live there) and runs bake_anchor_texture.py.
#   bash scripts/bake_anchor_texture.sh STEM
set -eu

STEM=${1:?usage: $0 STEM   (e.g. me_512 or status)}
VAMP=/home/newub/w/vamp-interface
LAM=/home/newub/w/LAM

source /home/newub/miniconda3/etc/profile.d/conda.sh
conda activate lam
export PATH=/home/newub/miniconda3/envs/lam/bin:$PATH
export PYTHONPATH=${LAM}
export XFORMERS_DISABLED=1

python "${VAMP}/scripts/bake_anchor_texture.py" \
    --ply  "${LAM}/exps/cano_gs/${STEM}_cano.ply" \
    --mesh "${LAM}/exps/cano_gs/${STEM}_shaped_mesh.obj" \
    --out  "${VAMP}/exp_output/lam_chibi/renders/bake_v1/${STEM}"
```

- [ ] **Step 6: Run the bake for both anchors**

```bash
chmod +x scripts/bake_anchor_texture.sh
bash scripts/bake_anchor_texture.sh status
bash scripts/bake_anchor_texture.sh me_512
```
Expected: `exp_output/lam_chibi/renders/bake_v1/<stem>/` holds `splat_ref.png` and `baked_mesh_000..090.png` for each anchor.

- [ ] **Step 7: Build the side-by-side and record the verdict**

```bash
cd /home/newub/w/vamp-interface
for s in status me_512; do
  ffmpeg -y -loglevel error \
    -i exp_output/lam_chibi/renders/bake_v1/$s/baked_mesh_000.png \
    -i exp_output/lam_chibi/renders/bake_v1/$s/splat_ref.png \
    -filter_complex "[0:v]scale=512:512[a];[1:v]scale=512:512[b];[a][b]hstack" \
    exp_output/lam_chibi/renders/bake_v1/$s/sidebyside.png
done
```
Eyeball each `sidebyside.png` (baked textured mesh left, splat render right). Verdict question: is the blotchy raw-meat look gone — does the baked mesh read as the same face as the splat render? Expected: yes, flatter (no view-dependent shading) but coherent. Record the answer in a `TaskUpdate` comment on task #32. If still blotchy, the camera convention or occlusion test is wrong — escalate with the side-by-side.

- [ ] **Step 8: Commit**

```bash
git add scripts/bake_anchor_texture.py scripts/bake_anchor_texture.sh
git commit -m "feat(chibi): bake_anchor_texture — render-and-bake driver"
```

---

## Self-review

**Spec coverage:**
- Direct projection bake, per-vertex color → Task 3. ✓
- Canonical-ply dump + standalone splat render → Task 4 step 1, Task 2. ✓
- Shared single-convention camera set → Task 1, used by Tasks 2 + 3. ✓
- Occlusion via splat-render depth (no pytorch3d cameras in bake) → Task 3. ✓
- Bake on canonical undeformed mesh → Task 4 driver uses `shaped_mesh.obj`. ✓
- Zero-visibility fallback (inner mouth / eyeball backs) → Task 3 nearest-seen fill. ✓
- Color-space check (no linear/sRGB convert) → Task 4 step 7 eyeball verdict. ✓
- `me` + LAM reference both baked → Task 4 step 6. ✓
- Reuse `ChibiMesh` + `mesh_render.render` unchanged → Tasks 3, 4. ✓
- Out of scope (nvdiffrast, UV texture, Blender, chibi deform) → not in any task. ✓

**Placeholder scan:** No TBD/TODO; every code step has complete code; every run step has a command and expected output. The Task 2 step 5 / Task 4 step 3 ordering note is an explicit, bounded instruction (the canonical ply must exist first), not a placeholder. ✓

**Type consistency:** `View(w2c, fov_rad, image_size)` constructed in Task 1, consumed identically in Tasks 2–3. `Splats(xyz, rgb, opacity, scaling, rotation)` defined and returned in Task 2, its `.xyz` reassigned in the Task 4 driver. `render_splats → (rgb, depth)` and `bake_vertex_colors(mesh, images, depth, views)` signatures match between definition and every call site. `ChibiMesh(verts, faces, rgb)` used as defined in `mesh.py`. ✓

**Note:** `Splats` is a mutable dataclass — the Task 4 driver reassigns `splats.xyz` to recentre. Intentional and local to the driver.
