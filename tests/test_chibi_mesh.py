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


import os
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
def test_apply_chibi_deforms_and_preserves_faces_rgb(tmp_path):
    mesh = _flame_template_chibi_mesh()
    out = apply_chibi(mesh, None, MASKS)          # None -> pipeline defaults
    assert torch.equal(out.faces, mesh.faces)
    assert torch.equal(out.rgb, mesh.rgb)
    assert out.verts.shape == mesh.verts.shape
    assert not torch.allclose(out.verts, mesh.verts, atol=1e-3)


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
    from chibi.fit import _load_obj_verts
    from chibi.landmarks import FLAME_TEMPLATE, _template_faces
    from chibi.uv_template import load_flame_uv
    v = _load_obj_verts(FLAME_TEMPLATE).double()
    faces = torch.as_tensor(_template_faces(), dtype=torch.int64)
    fuv = load_flame_uv(FLAME_TEMPLATE)
    tex = torch.full((16, 16, 3), 0.5, dtype=torch.float32)
    mesh = TexturedMesh(verts=v, faces=faces, uv=fuv.uv,
                        uv_faces=fuv.uv_faces, texture=tex)
    out = apply_chibi(mesh, None, MASKS)
    assert isinstance(out, TexturedMesh)
    assert torch.equal(out.faces, mesh.faces)
    assert torch.equal(out.uv, mesh.uv)
    assert torch.equal(out.uv_faces, mesh.uv_faces)
    assert torch.equal(out.texture, mesh.texture)
    assert not torch.allclose(out.verts, mesh.verts, atol=1e-3)
