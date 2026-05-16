import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import torch
from chibi.uv_template import FlameUV
from chibi.camera_rig import turntable_views
from chibi.texture_bake import rasterize_uv_attrs, bake_texture, dilate_texture


def _quad():
    """Unit quad in the z=0 plane facing +Z, UV mapped into [0.25,0.75]^2."""
    verts = torch.tensor([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0],
                          [-0.5, 0.5, 0.0], [0.5, 0.5, 0.0]],
                         dtype=torch.float64)
    faces = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.int64)
    uv = torch.tensor([[0.25, 0.25], [0.75, 0.25],
                       [0.25, 0.75], [0.75, 0.75]], dtype=torch.float32)
    fuv = FlameUV(uv=uv, uv_faces=faces.clone(),
                  vt2v=torch.arange(4, dtype=torch.int64))
    return verts, faces, fuv


def test_rasterize_uv_attrs_covers_uv_region():
    verts, faces, fuv = _quad()
    pos_map, nrm_map, mask = rasterize_uv_attrs(verts, faces, fuv, 64)
    assert pos_map.shape == (64, 64, 3)
    assert mask.shape == (64, 64)
    # UV square [0.25,0.75]^2 -> roughly the central quarter of the atlas
    assert mask.float().mean() > 0.15
    assert not mask[0, 0]                      # corner texel is a gutter hole


def test_bake_texture_paints_red():
    """Azim-0 view, solid-red splat image, nothing occludes -> covered texels
    bake red."""
    verts, faces, fuv = _quad()
    views = turntable_views(n_azim=1, elevs=(0.0,), dist=2.7, image_size=64)
    images = torch.zeros(1, 64, 64, 3, dtype=torch.uint8)
    images[..., 0] = 255
    depth = torch.full((1, 64, 64), 1e3)
    texture, filled = bake_texture(verts, faces, fuv, images, depth, views, 64)
    assert texture.shape == (64, 64, 3)
    red = texture[filled]
    assert red[:, 0].min() > 0.9 and red[:, 1].max() < 0.1


def test_dilate_texture_fills_hole():
    texture = torch.zeros(8, 8, 3)
    texture[..., 1] = 1.0                      # solid green
    mask = torch.ones(8, 8, dtype=torch.bool)
    mask[3:5, 3:5] = False                     # punch a 2x2 hole
    texture[~mask] = 0.0
    out = dilate_texture(texture, mask, iters=4)
    assert out[3:5, 3:5, 1].min() > 0.9        # hole filled with green
