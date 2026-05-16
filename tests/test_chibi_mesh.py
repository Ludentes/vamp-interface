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
