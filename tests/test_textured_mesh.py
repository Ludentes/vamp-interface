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
