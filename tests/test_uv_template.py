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
