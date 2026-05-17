# Chibi Mesh Pivot v1 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the splats→mesh pivot does not lose quality — first by rendering an existing ARKit-driven take as an animated mesh and comparing it to the splat render (the **gate**), then by adding chibi deformation on top.

**Architecture:** Small units behind a stable `ChibiMesh` dataclass. `mesh_extract` reads LAM output (canonical OBJ + per-frame `.ply`); `mesh_render` rasterises with pytorch3d; `mesh_deform` applies the existing `ChibiField`. Milestone 0 (driven-mesh quality check) is built and run **before** any chibi work — if it fails, the pivot is dead and Tasks 5–6 are moot. See spec `docs/superpowers/specs/2026-05-16-chibi-mesh-pivot-design.md`.

**Tech Stack:** Python 3.12, PyTorch, pytorch3d (mesh rasteriser), plyfile, numpy, imageio. Library code in `src/chibi/`. pytorch3d + plyfile live only in the `lam` conda env — **all tests and runs in this plan use the lam conda env**.

---

## Test environment

Run every `pytest` and script command in this plan as:

```bash
export PATH=/home/newub/miniconda3/envs/lam/bin:$PATH
cd /home/newub/w/vamp-interface
python -m pytest <args>
```

`src/` is added to `sys.path` by each test file's header, so no install step is needed.

## Background the engineer needs

- **`ChibiField`** (`src/chibi/field.py`) — `nn.Module` that bends a FLAME head into chibi proportions. `ChibiField(y_crown, y_chin, z_center)`; default params = identity. `field(verts, region_weights=rw)` → deformed `(N,3)` verts. Already shipped and tested.
- **`load_field_params(path)`** / **`save_field_params(field, path)`** (`src/chibi/fit.py`) — ChibiField JSON IO. **`_load_obj_verts(path)`** parses `v x y z` lines → `(N,3)`.
- **`region_falloff_weights(verts, masks_path)`** (`src/chibi/landmarks.py`) — region masks; needs `FLAME_masks.pkl`. **`landmark_positions(verts)`** — FLAME landmarks; index 8 is the chin. **`_template_faces()`** — the 5023-mesh face array. **`FLAME_TEMPLATE`** — path to `head_template_mesh.obj`.
- **LAM canonical OBJ** — `exps/cano_gs/<stem>_textured_mesh.obj`: 20018 `v x y z r g b` lines + 39904 1-based `f i j k` lines. Already on disk for `asian_m`, `me`.
- **LAM per-frame `.ply`** — when LAM inference runs a take with `save_ply`, it writes `exps/images/lam/lam_20k/<stem>/NNNN.ply` per frame (`lam/runners/infer/lam.py:374`). Each `.ply` is the *animated* Gaussian state for that frame: standard PLY `vertex` element with `x,y,z` (animated per-vertex position, 20018 rows) and `f_dc_0,f_dc_1,f_dc_2` (per-vertex RGB — LAM runs `gs_use_rgb`, so `f_dc` is sigmoid RGB, not SH). Topology is constant across frames; faces come from the canonical OBJ.
- **First 5023 verts** of the 20018 mesh are the original FLAME verts in order — what `landmark_positions` and the FLAME masks index into.
- **Splat-render comparison baselines** (already on disk): `exp_output/lam_blender_handoff/renders/take2__asian_m__arkit600.mp4` and `take2__young_european_f__arkit600.mp4`.
- **ARKit take motion dir**: `exp_output/lam_bakeoff/take_runs_arkit_600/MySlate_2_arkit/export/MySlate_2` (used by `scripts/chibi_anchor_render.sh`).

## File structure

| File | Responsibility |
|------|----------------|
| `src/chibi/mesh.py` | `ChibiMesh` dataclass — stable, renderer-agnostic interface. |
| `src/chibi/mesh_extract.py` | `load_chibi_mesh(obj)` — parse LAM textured-mesh OBJ. `load_gaussian_ply(ply)` — parse a LAM per-frame `.ply` → verts + rgb. `load_textured_mesh(obj)` — parse a v3 UV-textured OBJ (+ `.mtl` + atlas PNG) → `TexturedMesh`. |
| `src/chibi/mesh_render.py` | `render(meshes, azims, ...)` — pytorch3d render of a mesh sequence (animation) or one mesh from many angles (turntable). `render_textured(tmesh, azims, ...)` — same for a `TexturedMesh`. |
| `src/chibi/mesh_deform.py` | `apply_chibi(mesh, field_params, masks)` — deform verts with `ChibiField`. Accepts a `ChibiMesh` or a `TexturedMesh`, returns the same type. |
| `scripts/mesh_quality_check.py` | Milestone 0: per-frame `.ply` sequence → animated mesh video. |
| `scripts/chibi_mesh_render.py` + `.sh` | Chibi turntable render + side-by-side on the v3 textured mesh (Task 6). |
| `tests/test_chibi_mesh.py` | Tests for `ChibiMesh`, `load_chibi_mesh`, `load_gaussian_ply`, `load_textured_mesh`, `apply_chibi`. |
| `tests/test_chibi_mesh_render.py` | Test for `render`. |

## Task order and the gate

Tasks 1–3 build the primitives. **Task 4 (Milestone 0) is the gate** — a driven-mesh quality check, run before any chibi code. Tasks 5–6 (chibi) are **gated**: do not start them until Milestone 0's verdict is recorded and the quality is acceptable.

---

### Task 1: `ChibiMesh` dataclass

**Files:**
- Create: `src/chibi/mesh.py`
- Test: `tests/test_chibi_mesh.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_chibi_mesh.py`:

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import pytest
import torch
from chibi.mesh import ChibiMesh


def _trivial_mesh():
    verts = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                         dtype=torch.float64)
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    rgb = torch.zeros(3, 3, dtype=torch.float32)
    return verts, faces, rgb


def test_chibi_mesh_holds_verts_faces_rgb():
    verts, faces, rgb = _trivial_mesh()
    m = ChibiMesh(verts=verts, faces=faces, rgb=rgb)
    assert m.verts.shape == (3, 3)
    assert m.faces.shape == (1, 3)
    assert m.rgb.shape == (3, 3)


def test_chibi_mesh_rejects_out_of_range_face_index():
    verts, _, rgb = _trivial_mesh()
    bad_faces = torch.tensor([[0, 1, 9]], dtype=torch.int64)
    with pytest.raises(AssertionError):
        ChibiMesh(verts=verts, faces=bad_faces, rgb=rgb)


def test_chibi_mesh_rejects_rgb_count_mismatch():
    verts, faces, _ = _trivial_mesh()
    bad_rgb = torch.zeros(2, 3, dtype=torch.float32)
    with pytest.raises(AssertionError):
        ChibiMesh(verts=verts, faces=faces, rgb=bad_rgb)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_chibi_mesh.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chibi.mesh'`

- [ ] **Step 3: Write minimal implementation**

Create `src/chibi/mesh.py`:

```python
"""ChibiMesh — the renderer-agnostic interface for the chibi mesh pivot.

A textured triangle mesh: vertices, faces, per-vertex colour. Produced by
mesh_extract, transformed by mesh_deform, consumed by mesh_render. Keeping it
free of any pytorch3d type is what lets v2 (UV bake / Blender export) reuse the
extract and deform units unchanged.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass
class ChibiMesh:
    """verts: (V,3) float64 positions. faces: (F,3) int64, 0-based triangle
    indices. rgb: (V,3) float32 per-vertex colour in [0,1]. No opacity — a mesh
    is opaque, and the source data carries none."""
    verts: torch.Tensor
    faces: torch.Tensor
    rgb: torch.Tensor

    def __post_init__(self) -> None:
        assert self.verts.ndim == 2 and self.verts.shape[1] == 3, \
            f"verts must be (V,3), got {tuple(self.verts.shape)}"
        assert self.faces.ndim == 2 and self.faces.shape[1] == 3, \
            f"faces must be (F,3), got {tuple(self.faces.shape)}"
        assert self.rgb.shape == self.verts.shape, \
            f"rgb {tuple(self.rgb.shape)} must match verts {tuple(self.verts.shape)}"
        assert int(self.faces.min()) >= 0 and \
            int(self.faces.max()) < self.verts.shape[0], \
            "face index out of range [0, V)"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_chibi_mesh.py -v`
Expected: PASS — 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/chibi/mesh.py tests/test_chibi_mesh.py
git commit -m "feat(chibi): ChibiMesh dataclass for the mesh pivot"
```

---

### Task 2: `mesh_extract` — OBJ parser + per-frame `.ply` reader

**Files:**
- Create: `src/chibi/mesh_extract.py`
- Test: `tests/test_chibi_mesh.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_chibi_mesh.py`:

```python
import os
import numpy as np
from chibi.mesh_extract import load_chibi_mesh, load_gaussian_ply

_SAMPLE_OBJ = """# sample_textured_mesh.obj
#
v 0.10 0.20 0.30 0.90 0.10 0.05
v 1.10 0.20 0.30 0.10 0.90 0.05
v 0.10 1.20 0.30 0.10 0.10 0.95
v 1.10 1.20 0.30 0.50 0.50 0.50
f 1 2 3
f 2 4 3
"""


def test_load_chibi_mesh_parses_verts_faces_rgb(tmp_path):
    obj = tmp_path / "sample_textured_mesh.obj"
    obj.write_text(_SAMPLE_OBJ)
    m = load_chibi_mesh(str(obj))
    assert m.verts.shape == (4, 3)
    assert m.faces.shape == (2, 3)
    assert m.faces.tolist() == [[0, 1, 2], [1, 3, 2]]
    assert torch.allclose(m.rgb[0], torch.tensor([0.90, 0.10, 0.05]), atol=1e-6)
    assert torch.allclose(m.verts[1], torch.tensor([1.10, 0.20, 0.30],
                                                   dtype=torch.float64), atol=1e-6)


def test_load_chibi_mesh_rejects_obj_without_per_vertex_rgb(tmp_path):
    obj = tmp_path / "plain.obj"
    obj.write_text("v 0.0 0.0 0.0\nv 1.0 0.0 0.0\nv 0.0 1.0 0.0\nf 1 2 3\n")
    with pytest.raises(AssertionError, match="7 tokens"):
        load_chibi_mesh(str(obj))


def test_load_gaussian_ply_returns_verts_and_rgb(tmp_path):
    """A LAM per-frame .ply: PLY 'vertex' element with x,y,z + f_dc_0..2."""
    from plyfile import PlyData, PlyElement
    n = 5
    rng = np.random.default_rng(0)
    xyz = rng.random((n, 3), dtype=np.float32)
    fdc = rng.random((n, 3), dtype=np.float32)
    dtype = [(c, "f4") for c in
             ("x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2")]
    arr = np.empty(n, dtype=dtype)
    for i, c in enumerate(("x", "y", "z")):
        arr[c] = xyz[:, i]
    for i, c in enumerate(("f_dc_0", "f_dc_1", "f_dc_2")):
        arr[c] = fdc[:, i]
    arr["nx"] = arr["ny"] = arr["nz"] = 0.0
    ply = tmp_path / "0000.ply"
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(ply))

    verts, rgb = load_gaussian_ply(str(ply))
    assert verts.shape == (n, 3) and rgb.shape == (n, 3)
    assert np.allclose(verts.numpy(), xyz, atol=1e-6)
    assert np.allclose(rgb.numpy(), fdc, atol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_chibi_mesh.py -k "load_chibi_mesh or load_gaussian_ply" -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chibi.mesh_extract'`

- [ ] **Step 3: Write minimal implementation**

Create `src/chibi/mesh_extract.py`:

```python
"""Read LAM avatar output into the ChibiMesh primitives.

Two readers:
- load_chibi_mesh: the canonical textured-mesh OBJ (verts + per-vertex RGB +
  faces). LAM writes it via mesh_utils.save_obj(texture_type="vertex"); each
  `v` line is `v x y z r g b`, `f` lines are 1-based.
- load_gaussian_ply: a per-frame .ply (the *animated* Gaussian state for one
  frame). LAM runs gs_use_rgb, so the PLY's f_dc_0..2 are sigmoid RGB, and
  x,y,z are the animated per-vertex positions. Topology is constant — faces
  come from the canonical OBJ, not the .ply.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch

from chibi.mesh import ChibiMesh


def load_chibi_mesh(obj_path: str) -> ChibiMesh:
    """Parse a LAM textured-mesh OBJ. Asserts per-vertex RGB (7-token `v`)."""
    verts: list[list[float]] = []
    rgb: list[list[float]] = []
    faces: list[list[int]] = []
    for line in Path(obj_path).read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            assert len(parts) == 7, (
                f"expected per-vertex-colour OBJ ('v x y z r g b', 7 tokens), "
                f"got {len(parts)} tokens: {line!r}")
            verts.append([float(p) for p in parts[1:4]])
            rgb.append([float(p) for p in parts[4:7]])
        elif parts[0] == "f":
            faces.append([int(p.split("/")[0]) - 1 for p in parts[1:4]])
    assert verts, f"no vertices parsed from {obj_path}"
    assert faces, f"no faces parsed from {obj_path}"
    return ChibiMesh(
        verts=torch.tensor(verts, dtype=torch.float64),
        faces=torch.tensor(faces, dtype=torch.int64),
        rgb=torch.tensor(rgb, dtype=torch.float32).clamp(0.0, 1.0),
    )


def load_gaussian_ply(ply_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Parse a LAM per-frame .ply. Returns (verts (V,3) float64,
    rgb (V,3) float32 in [0,1]). f_dc_* is already sigmoid RGB (gs_use_rgb)."""
    from plyfile import PlyData
    el = PlyData.read(ply_path)["vertex"]
    verts = np.stack([el["x"], el["y"], el["z"]], axis=1)
    rgb = np.stack([el["f_dc_0"], el["f_dc_1"], el["f_dc_2"]], axis=1)
    return (torch.tensor(verts, dtype=torch.float64),
            torch.tensor(rgb, dtype=torch.float32).clamp(0.0, 1.0))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_chibi_mesh.py -k "load_chibi_mesh or load_gaussian_ply" -v`
Expected: PASS — 3 passed

- [ ] **Step 5: Verify against real LAM output**

Run:
```bash
python -c "
import sys; sys.path.insert(0, 'src')
from chibi.mesh_extract import load_chibi_mesh, load_gaussian_ply
m = load_chibi_mesh('/home/newub/w/LAM/exps/cano_gs/asian_m_textured_mesh.obj')
print('OBJ  verts', tuple(m.verts.shape), 'faces', tuple(m.faces.shape))
import glob
plys = sorted(glob.glob('/home/newub/w/LAM/exps/images/lam/lam_20k/*/0000.ply'))
if plys:
    v, c = load_gaussian_ply(plys[0])
    print('PLY ', plys[0], 'verts', tuple(v.shape), 'rgb', float(c.min()), float(c.max()))
else:
    print('PLY  none on disk yet — Task 4 step 1 will produce them')
"
```
Expected: `OBJ verts (20018, 3) faces (39904, 3)`; if a `.ply` is present, `verts (20018, 3)` and rgb in `[0,1]`.

- [ ] **Step 6: Commit**

```bash
git add src/chibi/mesh_extract.py tests/test_chibi_mesh.py
git commit -m "feat(chibi): mesh_extract — LAM OBJ + per-frame ply readers"
```

---

### Task 3: `render` — pytorch3d mesh-sequence render

**Files:**
- Create: `src/chibi/mesh_render.py`
- Test: `tests/test_chibi_mesh_render.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_chibi_mesh_render.py`:

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import pytest
import torch

pytest.importorskip("pytorch3d", reason="pytorch3d only in the lam conda env")
from chibi.mesh import ChibiMesh
from chibi.mesh_render import render


def _red_quad():
    verts = torch.tensor([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0],
                          [-1.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
                         dtype=torch.float64)
    faces = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.int64)
    rgb = torch.tensor([[1.0, 0.0, 0.0]] * 4, dtype=torch.float32)
    return ChibiMesh(verts=verts, faces=faces, rgb=rgb)


def test_render_one_mesh_many_angles_shape():
    frames = render([_red_quad()], [0.0, 120.0, 240.0], image_size=64)
    assert frames.shape == (3, 64, 64, 3)
    assert frames.dtype == torch.uint8


def test_render_mesh_sequence_one_angle_shape():
    seq = [_red_quad(), _red_quad()]
    frames = render(seq, [0.0], image_size=64)
    assert frames.shape == (2, 64, 64, 3)


def test_render_front_frame_is_red():
    """azim 0 looks straight at the red quad; centre pixel is red — unlit, so
    the vertex colour comes through directly."""
    frames = render([_red_quad()], [0.0], image_size=64)
    centre = frames[0, 32, 32]
    assert centre[0] > 200, f"red channel too low: {centre.tolist()}"
    assert centre[1] < 80 and centre[2] < 80, f"not red: {centre.tolist()}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_chibi_mesh_render.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chibi.mesh_render'`

- [ ] **Step 3: Write minimal implementation**

Create `src/chibi/mesh_render.py`:

```python
"""Render ChibiMesh frames with pytorch3d.

`render` handles both cases the pivot needs: a mesh *sequence* from one camera
angle (a driven animation — one ChibiMesh per frame), and a single mesh from
*many* angles (a turntable). Whichever list has length 1 is broadcast.

Unlit: AmbientLights with full ambient and no diffuse/specular, so the output
is the per-vertex colour interpolated across triangles — the flat chibi look,
and it isolates geometry from shading for the quality verdict.

Recentre + unit-scale are computed once from the first mesh and reused for the
whole sequence, so a driven animation does not jitter or breathe.
"""
from __future__ import annotations
from collections.abc import Sequence
import torch

from chibi.mesh import ChibiMesh


def render(meshes: Sequence[ChibiMesh], azims: Sequence[float], *,
           image_size: int = 256, dist: float = 2.7, elev: float = 0.0,
           device: str = "cuda") -> torch.Tensor:
    """Render `T = max(len(meshes), len(azims))` frames. The length-1 list is
    broadcast to T. Returns (T, image_size, image_size, 3) uint8 RGB on CPU."""
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        TexturesVertex, FoVPerspectiveCameras, RasterizationSettings,
        MeshRenderer, MeshRasterizer, SoftPhongShader, AmbientLights,
        look_at_view_transform,
    )
    meshes = list(meshes)
    azims = list(azims)
    n = max(len(meshes), len(azims))
    if len(meshes) == 1:
        meshes = meshes * n
    if len(azims) == 1:
        azims = azims * n
    assert len(meshes) == len(azims) == n, \
        "meshes and azims must have equal length or one of them length 1"

    dev = torch.device(device)
    # fixed recentre/scale from frame 0 — keeps a driven animation steady.
    ref = meshes[0].verts.to(torch.float32)
    centre = ref.mean(0, keepdim=True).to(dev)
    scale = (ref.to(dev) - centre).abs().max().clamp_min(1e-6)

    raster = RasterizationSettings(image_size=image_size, blur_radius=0.0,
                                   faces_per_pixel=1)
    lights = AmbientLights(device=dev)            # unlit: flat vertex colour

    frames = []
    for mesh, azim in zip(meshes, azims):
        verts = (mesh.verts.to(torch.float32).to(dev) - centre) / scale
        faces = mesh.faces.to(torch.int64).to(dev)
        rgb = mesh.rgb.to(torch.float32).to(dev)
        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim,
                                      device=dev)
        cameras = FoVPerspectiveCameras(R=R, T=T, device=dev)
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster),
            shader=SoftPhongShader(device=dev, cameras=cameras, lights=lights),
        )
        p3d = Meshes(verts=[verts], faces=[faces],
                     textures=TexturesVertex(verts_features=[rgb]))
        img = renderer(p3d)[0, ..., :3]
        frames.append((img.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu())
    return torch.stack(frames)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_chibi_mesh_render.py -v`
Expected: PASS — 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/chibi/mesh_render.py tests/test_chibi_mesh_render.py
git commit -m "feat(chibi): render — pytorch3d mesh-sequence render"
```

---

### Task 4 — MILESTONE 0 (THE GATE): ARKit-driven mesh quality check

**Goal:** render an existing ARKit take as an animated mesh and confirm we did not catastrophically lose quality versus the splat render. Tasks 5–6 are gated on this verdict.

**Files:**
- Create: `scripts/mesh_quality_check.py`

- [ ] **Step 1: Ensure per-frame `.ply` files exist for one take + anchor**

Check first:
```bash
ls /home/newub/w/LAM/exps/images/lam/lam_20k/asian_m/0000.ply 2>/dev/null && echo PRESENT || echo MISSING
```

If MISSING, run one LAM take inference with `.ply` dumping on (this is a single inference run, not a new pipeline):
```bash
export PATH=/home/newub/miniconda3/envs/lam/bin:$PATH
source /home/newub/miniconda3/etc/profile.d/conda.sh && conda activate lam
export PYTHONPATH=/home/newub/w/LAM XFORMERS_DISABLED=1 LAM_USE_ARKIT=1
export SAVE_PLY=true SAVE_IMG=true
cd /home/newub/w/LAM
bash scripts/inference.sh configs/inference/lam-20k-8gpu.yaml \
    model_zoo/lam_models/releases/lam/lam-20k/step_045500/ \
    /home/newub/w/vamp-interface/exp_output/lam_blender_handoff/anchors/asian_m.png \
    /home/newub/w/vamp-interface/exp_output/lam_bakeoff/take_runs_arkit_600/MySlate_2_arkit/export/MySlate_2/
```
Expected: `exps/images/lam/lam_20k/asian_m/NNNN.ply` for each frame, plus the splat video `exps/videos/lam/lam_20k/asian_m.mp4`. (`SAVE_PLY`/`SAVE_IMG` are read by `scripts/inference.sh` — confirm with `grep SAVE_PLY /home/newub/w/LAM/scripts/inference.sh`; if the script does not export them, prepend them to the `accelerate launch` line it runs.)

- [ ] **Step 2: Write the quality-check script**

Create `scripts/mesh_quality_check.py`:

```python
"""Milestone 0 — the splats->mesh quality gate.

Render an ARKit-driven take as an animated MESH (per-frame .ply -> ChibiMesh
-> pytorch3d) and write it to mp4. Compare the result to the splat render of
the same take to confirm the pivot did not lose quality. No chibi here.

Run in the lam conda env:
  python scripts/mesh_quality_check.py \
      --ply_dir /home/newub/w/LAM/exps/images/lam/lam_20k/asian_m \
      --obj /home/newub/w/LAM/exps/cano_gs/asian_m_textured_mesh.obj \
      --out exp_output/lam_chibi/renders/mesh_v1/driven_asian_m.mp4
"""
from __future__ import annotations
import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import imageio.v2 as imageio  # noqa: E402

from chibi.mesh import ChibiMesh                       # noqa: E402
from chibi.mesh_extract import load_chibi_mesh, load_gaussian_ply  # noqa: E402
from chibi.mesh_render import render                   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply_dir", required=True,
                    help="dir of LAM per-frame NNNN.ply files")
    ap.add_argument("--obj", required=True,
                    help="canonical <stem>_textured_mesh.obj (faces source)")
    ap.add_argument("--out", required=True, help="output mp4 path")
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--fps", type=int, default=25)
    args = ap.parse_args()

    plys = sorted(glob.glob(str(Path(args.ply_dir) / "[0-9]" * 4 + ".ply")))
    assert plys, f"no NNNN.ply files in {args.ply_dir}"
    faces = load_chibi_mesh(args.obj).faces
    print(f"[mesh] {len(plys)} frames, {faces.shape[0]} faces")

    meshes = []
    for p in plys:
        verts, rgb = load_gaussian_ply(p)
        assert verts.shape[0] == int(faces.max()) + 1, (
            f"{p}: vert count {verts.shape[0]} != faces topology "
            f"{int(faces.max()) + 1}")
        meshes.append(ChibiMesh(verts=verts, faces=faces, rgb=rgb))

    # azim 0 = front view, fixed camera for the whole driven sequence.
    frames = render(meshes, [0.0], image_size=args.image_size)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(args.out, list(frames.numpy()), fps=args.fps)
    print(f"[render] driven mesh video -> {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the quality check**

```bash
export PATH=/home/newub/miniconda3/envs/lam/bin:$PATH
cd /home/newub/w/vamp-interface
python scripts/mesh_quality_check.py \
    --ply_dir /home/newub/w/LAM/exps/images/lam/lam_20k/asian_m \
    --obj /home/newub/w/LAM/exps/cano_gs/asian_m_textured_mesh.obj \
    --out exp_output/lam_chibi/renders/mesh_v1/driven_asian_m.mp4
```
Expected: prints frame/face counts and writes `driven_asian_m.mp4`.

- [ ] **Step 4: Build the side-by-side against the splat render**

```bash
ffmpeg -y -i exp_output/lam_chibi/renders/mesh_v1/driven_asian_m.mp4 \
       -i exp_output/lam_blender_handoff/renders/take2__asian_m__arkit600.mp4 \
       -filter_complex "[0:v]scale=512:512[a];[1:v]scale=512:512[b];[a][b]hstack" \
       exp_output/lam_chibi/renders/mesh_v1/sidebyside_driven_asian_m.mp4
```
Expected: `sidebyside_driven_asian_m.mp4` — mesh render (left) vs splat render (right).

- [ ] **Step 5: Record the verdict**

Eyeball `sidebyside_driven_asian_m.mp4`. The gate question: did the mesh render *catastrophically* lose quality versus the splat render? A flatter, lower-fi look is expected and acceptable (it is the v2 motivation). Catastrophic = broken geometry, holes, scrambled colour, motion artifacts. Record the answer in a comment on task #32 via `TaskUpdate`.

- [ ] **Step 6: Commit the script**

```bash
git add scripts/mesh_quality_check.py
git commit -m "feat(chibi): mesh_quality_check — driven-mesh quality gate"
```

- [ ] **Step 7: GATE DECISION** — if the verdict is "catastrophic quality loss", STOP: the mesh pivot needs rework before chibi; escalate to the user with the side-by-side. If acceptable, proceed to Task 5.

---

### Task 5: `apply_chibi` — deform the mesh with `ChibiField` *(gated on Milestone 0)*

**Files:**
- Create: `src/chibi/mesh_deform.py`
- Test: `tests/test_chibi_mesh.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_chibi_mesh.py`:

```python
import math
from chibi.mesh_deform import apply_chibi

MASKS = "/home/newub/w/LAM/model_zoo/human_parametric_models/flame_assets/flame/FLAME_masks.pkl"
needs_flame = pytest.mark.skipif(
    not os.path.exists(MASKS),
    reason="local FLAME assets (FLAME_masks.pkl) not found")


def _flame_template_chibi_mesh():
    """ChibiMesh from the real 5023 FLAME template (dummy rgb — apply_chibi
    must pass rgb through untouched)."""
    from chibi.fit import _load_obj_verts
    from chibi.landmarks import FLAME_TEMPLATE, _template_faces
    v = _load_obj_verts(FLAME_TEMPLATE).double()
    faces = torch.as_tensor(_template_faces(), dtype=torch.int64)
    rgb = torch.full((v.shape[0], 3), 0.5, dtype=torch.float32)
    return ChibiMesh(verts=v, faces=faces, rgb=rgb)


@needs_flame
def test_apply_chibi_identity_field_leaves_verts_unchanged(tmp_path):
    from chibi.field import ChibiField
    from chibi.fit import save_field_params
    mesh = _flame_template_chibi_mesh()
    params = tmp_path / "identity.json"
    save_field_params(ChibiField(y_crown=1.0, y_chin=0.0, z_center=0.0),
                      str(params))
    out = apply_chibi(mesh, str(params), MASKS)
    assert torch.allclose(out.verts, mesh.verts, atol=1e-5)


@needs_flame
def test_apply_chibi_preserves_faces_and_rgb(tmp_path):
    from chibi.field import ChibiField
    from chibi.fit import save_field_params
    mesh = _flame_template_chibi_mesh()
    params = tmp_path / "nontrivial.json"
    f = ChibiField(y_crown=1.0, y_chin=0.0, z_center=0.0)
    with torch.no_grad():
        f.remap_incr.copy_(torch.tensor([0.25, -0.35, 0.1, -0.2, 0.15]))
        f.radial_log.copy_(torch.linspace(0.0, 0.3, 6))
    save_field_params(f, str(params))
    out = apply_chibi(mesh, str(params), MASKS)
    assert torch.equal(out.faces, mesh.faces)
    assert torch.equal(out.rgb, mesh.rgb)
    assert out.verts.shape == mesh.verts.shape
    assert not torch.allclose(out.verts, mesh.verts, atol=1e-3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_chibi_mesh.py -k apply_chibi -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chibi.mesh_deform'`

- [ ] **Step 3: Write minimal implementation**

Create `src/chibi/mesh_deform.py`:

```python
"""Deform a ChibiMesh's vertices with a fitted ChibiField.

The deformation moves verts only — faces and per-vertex rgb pass through
unchanged. That invariance is the point of the mesh pivot: per-vertex colour
on a fixed-topology mesh is deformation-safe (a stretched triangle still
interpolates its three corner colours), unlike a baked splat cloud.

Mirrors scripts/chibi_make_assets.py:deform_with_field, minus the secant-basis
return that only the ARKit asset path needs. Runs in float64: the
searchsorted/interp/exp path in ChibiField carries ~1e-5 error at float32.
"""
from __future__ import annotations
import torch

from chibi.mesh import ChibiMesh
from chibi.fit import load_field_params
from chibi.landmarks import region_falloff_weights, landmark_positions


def apply_chibi(mesh: ChibiMesh, field_params_path: str,
                masks_path: str) -> ChibiMesh:
    """Apply the fitted ChibiField at `field_params_path` to `mesh`.

    The field is re-framed to THIS mesh's own crown/chin (fitted params are
    frame-independent). `mesh` must be FLAME-topology — the first 5023 verts
    are the original FLAME verts, which landmark_positions and the FLAME masks
    index into.
    """
    assert mesh.verts.shape[0] in (5023, 20018), (
        f"apply_chibi expects the 5023 FLAME template or the 20018 baked mesh; "
        f"got {mesh.verts.shape[0]} verts")
    field = load_field_params(field_params_path).double()
    xyz = mesh.verts.to(torch.float64)
    with torch.no_grad():
        field.y_crown.copy_(xyz[:, 1].max())
        field.y_chin.copy_(landmark_positions(xyz[:5023])[8, 1])
        field.z_center.copy_(xyz[:, 2].mean())
    rw = region_falloff_weights(xyz, masks_path)
    with torch.no_grad():
        deformed = field(xyz, region_weights=rw)
    return ChibiMesh(verts=deformed, faces=mesh.faces, rgb=mesh.rgb)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_chibi_mesh.py -k apply_chibi -v`
Expected: PASS — 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/chibi/mesh_deform.py tests/test_chibi_mesh.py
git commit -m "feat(chibi): apply_chibi — deform a ChibiMesh with ChibiField"
```

---

### Task 6: Chibi render + side-by-side verdict — v3 textured mesh *(gated on Milestone 0)*

**Rewritten from the v2 per-vertex-colour path to v3.** The shipped colour path
is the nvdiffrast UV-texture bake (`scripts/bake_uv_texture.py`), whose output
is a `TexturedMesh` (`<stem>_textured.obj` + `.mtl` + `<stem>_texture.png`) —
not a per-vertex-RGB `ChibiMesh`. So Task 6 loads a `TexturedMesh`, deforms its
verts, and renders via `render_textured`. The chibi deform is UV-invariant:
`uv` / `uv_faces` / `texture` pass through untouched.

**Files:**
- Modify: `src/chibi/mesh_extract.py` — add `load_textured_mesh`
- Modify: `src/chibi/mesh_deform.py` — `apply_chibi` accepts a `TexturedMesh`
- Modify: `tests/test_chibi_mesh.py` — append
- Create: `scripts/chibi_mesh_render.py` + `.sh`

- [ ] **Step 1: Write the failing tests for `load_textured_mesh` + `apply_chibi` on a `TexturedMesh`**

Append to `tests/test_chibi_mesh.py`:

```python
from chibi.mesh import TexturedMesh
from chibi.mesh_extract import load_textured_mesh


def test_load_textured_mesh_roundtrips_a_v3_obj(tmp_path):
    import imageio.v2 as imageio, numpy as np
    obj = tmp_path / "t_textured.obj"
    obj.write_text(
        "mtllib t_textured.mtl\nusemtl t_mat\n"
        "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
        "vt 0 0\nvt 1 0\nvt 0 1\n"
        "f 1/1 2/2 3/3\n")
    (tmp_path / "t_textured.mtl").write_text("newmtl t_mat\nmap_Kd t_texture.png\n")
    imageio.imwrite(tmp_path / "t_texture.png",
                    np.full((8, 8, 3), 128, dtype=np.uint8))
    m = load_textured_mesh(str(obj))
    assert isinstance(m, TexturedMesh)
    assert m.verts.shape == (3, 3) and m.faces.shape == (1, 3)
    assert m.uv.shape == (3, 2) and m.uv_faces.shape == (1, 3)
    assert m.texture.shape == (8, 8, 3)
    assert torch.allclose(m.texture, torch.full((8, 8, 3), 128 / 255.0), atol=1e-6)


@needs_flame
def test_apply_chibi_on_textured_mesh_returns_textured_mesh(tmp_path):
    from chibi.field import ChibiField
    from chibi.fit import save_field_params, _load_obj_verts
    from chibi.landmarks import FLAME_TEMPLATE, _template_faces
    from chibi.uv_template import load_flame_uv
    v = _load_obj_verts(FLAME_TEMPLATE).double()
    faces = torch.as_tensor(_template_faces(), dtype=torch.int64)
    fuv = load_flame_uv(FLAME_TEMPLATE)
    tex = torch.full((16, 16, 3), 0.5, dtype=torch.float32)
    mesh = TexturedMesh(verts=v, faces=faces, uv=fuv.uv,
                        uv_faces=fuv.uv_faces, texture=tex)
    params = tmp_path / "f.json"
    f = ChibiField(y_crown=1.0, y_chin=0.0, z_center=0.0)
    with torch.no_grad():
        f.remap_incr.copy_(torch.tensor([0.25, -0.35, 0.1, -0.2, 0.15]))
        f.radial_log.copy_(torch.linspace(0.0, 0.3, 6))
    save_field_params(f, str(params))
    out = apply_chibi(mesh, str(params), MASKS)
    assert isinstance(out, TexturedMesh)
    assert torch.equal(out.faces, mesh.faces)
    assert torch.equal(out.uv, mesh.uv)
    assert torch.equal(out.uv_faces, mesh.uv_faces)
    assert torch.equal(out.texture, mesh.texture)
    assert not torch.allclose(out.verts, mesh.verts, atol=1e-3)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_chibi_mesh.py -k "textured_mesh" -v`
Expected: FAIL — `ImportError: cannot import name 'load_textured_mesh'`

- [ ] **Step 3: Implement `load_textured_mesh`**

Append to `src/chibi/mesh_extract.py`:

```python
def load_textured_mesh(obj_path: str):
    """Parse a v3 UV-textured OBJ (written by scripts/bake_uv_texture.py:
    `v x y z`, `vt u v`, `f v/vt v/vt v/vt`) into a TexturedMesh. The atlas
    PNG is resolved from the sibling `.mtl`'s `map_Kd`."""
    from chibi.mesh import TexturedMesh
    import imageio.v2 as imageio
    obj = Path(obj_path)
    verts: list[list[float]] = []
    uvs: list[list[float]] = []
    faces: list[list[int]] = []
    uv_faces: list[list[int]] = []
    mtl_name = None
    for line in obj.read_text().splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == "v":
            verts.append([float(x) for x in p[1:4]])
        elif p[0] == "vt":
            uvs.append([float(p[1]), float(p[2])])
        elif p[0] == "f":
            faces.append([int(t.split("/")[0]) - 1 for t in p[1:4]])
            uv_faces.append([int(t.split("/")[1]) - 1 for t in p[1:4]])
        elif p[0] == "mtllib":
            mtl_name = p[1]
    assert verts and faces and uvs, f"incomplete textured OBJ: {obj_path}"
    tex_name = None
    if mtl_name and (obj.parent / mtl_name).exists():
        for line in (obj.parent / mtl_name).read_text().splitlines():
            if line.startswith("map_Kd"):
                tex_name = line.split()[1]
    assert tex_name, f"no map_Kd texture found for {obj_path}"
    tex = imageio.imread(obj.parent / tex_name)
    texture = torch.tensor(tex[..., :3], dtype=torch.float32) / 255.0
    return TexturedMesh(
        verts=torch.tensor(verts, dtype=torch.float64),
        faces=torch.tensor(faces, dtype=torch.int64),
        uv=torch.tensor(uvs, dtype=torch.float32),
        uv_faces=torch.tensor(uv_faces, dtype=torch.int64),
        texture=texture,
    )
```

- [ ] **Step 4: Extend `apply_chibi` to dispatch on mesh type**

Replace the body of `apply_chibi` in `src/chibi/mesh_deform.py` so the verts
deform is shared and the return type matches the input type (`ChibiMesh` →
`ChibiMesh`, `TexturedMesh` → `TexturedMesh`). Import `TexturedMesh`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_chibi_mesh.py -k "textured_mesh or apply_chibi" -v`
Expected: PASS — 4 passed (2 from Task 5, 2 new)

- [ ] **Step 6: Write the render driver**

Create `scripts/chibi_mesh_render.py` — load the v3 textured OBJ, render a
baseline turntable, `apply_chibi`, render the chibi turntable, write both mp4s
and a baseline-vs-chibi side-by-side. Create `scripts/chibi_mesh_render.sh` to
activate the `lam` conda env and call it with a `STEM` arg, resolving
`--obj exp_output/lam_chibi/renders/bake_v3/<stem>/<stem>_textured.obj`.

- [ ] **Step 7: Run both anchors**

```bash
bash scripts/chibi_mesh_render.sh asian_m
bash scripts/chibi_mesh_render.sh me_512
```
Expected: `mesh_baseline_<stem>.mp4`, `mesh_chibi_<stem>.mp4`,
`sidebyside_<stem>.mp4` under `exp_output/lam_chibi/renders/mesh_v1/`.

- [ ] **Step 8: Record the verdict and close the task**

Eyeball the outputs: the chibi deform should enlarge the head / shrink features
without tearing the UV texture. Note the verdict in a `TaskUpdate` on #65, then
mark #65 and #32 completed.

- [ ] **Step 9: Commit**

```bash
git add src/chibi/mesh_extract.py src/chibi/mesh_deform.py \
        tests/test_chibi_mesh.py scripts/chibi_mesh_render.py \
        scripts/chibi_mesh_render.sh
git commit -m "feat(chibi): chibi_mesh_render — deform + render the v3 textured mesh"
```

---

## Self-review

**Spec coverage:**
- 4 units behind `ChibiMesh` → Tasks 1–3, 5. ✓
- `extract` is a pure OBJ parser (+ per-frame ply reader for the driven gate) → Task 2. ✓
- pytorch3d `TexturesVertex` render → Task 3. ✓
- `apply_chibi` reuses `ChibiField`, renderer-agnostic → Task 5. ✓
- Driven-mesh quality verification before chibi (user reorder) → Task 4, Milestone 0. ✓
- `me` + `asian_m` rendered; `me` stress case → Task 6 step 3. ✓
- v2-readiness: `ChibiMesh` carries no pytorch3d type; `mesh_deform` has no pytorch3d import → Tasks 1, 5. ✓
- Out of scope (UV bake, Blender, hair shell, GLB export, driven *chibi* animation) → not in any task. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code; every run step shows command + expected output. The one conditional ("if SAVE_PLY not exported, prepend it") is a concrete, bounded instruction with a verification command, not a placeholder. ✓

**Type consistency:** `ChibiMesh(verts, faces, rgb)` used identically in Tasks 1–6. `render(meshes, azims, *, image_size, dist, elev, device)` — called as `render([mesh], azims, ...)` and `render(meshes, [0.0], ...)`, both within the documented broadcast rule. `load_chibi_mesh`/`load_gaussian_ply`/`apply_chibi` signatures consistent between definition and every call site. ✓

**Naming note:** the spec calls the extract function `extract_lam_mesh`; implemented as `load_chibi_mesh` (OBJ) + `load_gaussian_ply` (per-frame ply) — both are pure readers, not LAM-inference wrappers, per the spec's "Spike correction" paragraph.
