import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import math
import torch
from chibi.camera_rig import turntable_views, View


def test_turntable_view_count():
    views = turntable_views(n_azim=12, elevs=(-20.0, 0.0, 20.0), dist=2.7)
    assert len(views) == 36
    assert all(isinstance(v, View) for v in views)


def test_view_has_w2c_and_fov():
    v = turntable_views(n_azim=4, elevs=(0.0,), dist=2.7)[0]
    assert v.w2c.shape == (4, 4)
    assert v.w2c.dtype == torch.float32
    assert 0.0 < v.fov_rad < math.pi
    assert v.image_size == 512


def test_camera_centers_on_sphere():
    """Every camera sits at radius `dist` from the origin it looks at."""
    dist = 2.7
    for v in turntable_views(n_azim=8, elevs=(0.0, 30.0), dist=dist):
        c2w = torch.inverse(v.w2c)
        centre = c2w[:3, 3]
        assert abs(float(centre.norm()) - dist) < 1e-3, float(centre.norm())
