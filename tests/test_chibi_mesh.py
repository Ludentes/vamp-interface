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
