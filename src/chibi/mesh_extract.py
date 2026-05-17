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


def load_gaussian_ply(ply_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Parse a LAM per-frame .ply. Returns (verts (V,3) float64,
    rgb (V,3) float32 in [0,1]). f_dc_* is already sigmoid RGB (gs_use_rgb)."""
    from plyfile import PlyData
    el = PlyData.read(ply_path)["vertex"]
    verts = np.stack([el["x"], el["y"], el["z"]], axis=1)
    rgb = np.stack([el["f_dc_0"], el["f_dc_1"], el["f_dc_2"]], axis=1)
    return (torch.tensor(verts, dtype=torch.float64),
            torch.tensor(rgb, dtype=torch.float32).clamp(0.0, 1.0))
