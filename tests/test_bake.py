import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import torch
from chibi.mesh import ChibiMesh
from chibi.camera_rig import turntable_views
from chibi.bake import bake_vertex_colors


def _front_quad():
    """A unit quad in the z=0 plane, facing +Z (toward an azim-0 camera)."""
    verts = torch.tensor([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0],
                          [-0.5, 0.5, 0.0], [0.5, 0.5, 0.0]],
                         dtype=torch.float64)
    faces = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.int64)
    rgb = torch.zeros(4, 3, dtype=torch.float32)
    return ChibiMesh(verts=verts, faces=faces, rgb=rgb)


def test_bake_paints_solid_color_from_one_view():
    """One azim-0 view, a solid-red image, infinite depth (nothing occludes):
    every front-facing vertex takes the red."""
    mesh = _front_quad()
    views = turntable_views(n_azim=1, elevs=(0.0,), dist=2.7, image_size=64)
    images = torch.zeros(1, 64, 64, 3, dtype=torch.uint8)
    images[..., 0] = 255                                   # solid red
    depth = torch.full((1, 64, 64), 1e3)                   # nothing in front
    out = bake_vertex_colors(mesh, images, depth, views)
    assert torch.equal(out.faces, mesh.faces)
    assert out.rgb[:, 0].min() > 0.9        # all verts red
    assert out.rgb[:, 1].max() < 0.1


def test_bake_occlusion_rejects_view_behind_surface():
    """If the splat-depth map says the surface is much closer than the vertex,
    the vertex is occluded and does not take that view's color."""
    mesh = _front_quad()
    views = turntable_views(n_azim=1, elevs=(0.0,), dist=2.7, image_size=64)
    images = torch.zeros(1, 64, 64, 3, dtype=torch.uint8)
    images[..., 0] = 255
    depth = torch.full((1, 64, 64), 0.01)   # surface right at the lens
    out = bake_vertex_colors(mesh, images, depth, views,
                             fallback_rgb=(0.0, 0.0, 1.0))
    # every view rejected -> all verts fall back to blue
    assert out.rgb[:, 2].min() > 0.9 and out.rgb[:, 0].max() < 0.1
