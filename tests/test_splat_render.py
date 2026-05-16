import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import pytest
import numpy as np
import torch

pytest.importorskip("diff_gaussian_rasterization",
                    reason="splat rasteriser only in the lam conda env")
from chibi.camera_rig import turntable_views
from chibi.splat_render import load_splat_ply, render_splats


def _one_gaussian_ply(tmp_path):
    """A single fat bright-green Gaussian at the origin."""
    from plyfile import PlyData, PlyElement
    cols = ("x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
            "opacity", "scale_0", "scale_1", "scale_2",
            "rot_0", "rot_1", "rot_2", "rot_3")
    arr = np.zeros(1, dtype=[(c, "f4") for c in cols])
    arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = 0.0, 1.0, 0.0   # green RGB
    arr["opacity"] = 8.0          # inverse_sigmoid-space -> ~1.0 after sigmoid
    arr["scale_0"] = arr["scale_1"] = arr["scale_2"] = -1.0  # log-space -> exp
    arr["rot_0"] = 1.0            # identity quaternion
    ply = tmp_path / "one.ply"
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(ply))
    return ply


def test_load_splat_ply_decodes_fields(tmp_path):
    s = load_splat_ply(str(_one_gaussian_ply(tmp_path)))
    assert s.xyz.shape == (1, 3)
    assert s.rgb.shape == (1, 3)
    assert torch.allclose(s.rgb[0], torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)
    assert 0.99 < float(s.opacity[0]) <= 1.0          # sigmoid(8) ~ 0.9997
    assert torch.allclose(s.scaling[0],
                          torch.full((3,), float(np.exp(-1.0))), atol=1e-5)


def test_render_splats_shape_and_color(tmp_path):
    s = load_splat_ply(str(_one_gaussian_ply(tmp_path)))
    views = turntable_views(n_azim=2, elevs=(0.0,), dist=2.7, image_size=64)
    rgb, depth = render_splats(s, views)
    assert rgb.shape == (2, 64, 64, 3) and rgb.dtype == torch.uint8
    assert depth.shape == (2, 64, 64)
    centre = rgb[0, 32, 32]
    assert centre[1] > 150 and centre[0] < 90 and centre[2] < 90, centre.tolist()
