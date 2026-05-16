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
