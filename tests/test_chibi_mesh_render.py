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
