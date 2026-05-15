import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import torch
from chibi.field import ChibiField, REALISTIC_KNOTS


def _field():
    return ChibiField(y_crown=1.0, y_chin=0.0, z_center=0.0)


def test_identity_field_leaves_verts_unchanged():
    f = _field()  # default params = 0 → identity
    v = torch.tensor([[0.1, 0.5, 0.2], [-0.3, 0.9, 0.0]])
    out = f(v)
    assert torch.allclose(out, v, atol=1e-5)


def test_remap_is_monotone_for_random_params():
    f = _field()
    with torch.no_grad():
        f.remap_incr.copy_(torch.randn(5))
    u = torch.linspace(0, 1, 50)
    uc = f.remap(u)
    assert torch.all(uc[1:] - uc[:-1] >= -1e-6)
    assert torch.allclose(uc[0], torch.tensor(0.0), atol=1e-5)
    assert torch.allclose(uc[-1], torch.tensor(1.0), atol=1e-5)


def test_realistic_knots_span_unit_interval():
    assert REALISTIC_KNOTS[0] == 0.0 and REALISTIC_KNOTS[-1] == 1.0
    assert len(REALISTIC_KNOTS) == 6
