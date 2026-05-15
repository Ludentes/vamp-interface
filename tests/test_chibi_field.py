import sys, pathlib, math, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pytest
import torch
from chibi.field import ChibiField, REALISTIC_KNOTS

MASKS = "/home/newub/w/LAM/model_zoo/human_parametric_models/flame_assets/flame/FLAME_masks.pkl"
ARKIT = "/home/newub/w/LAM/model_zoo/human_parametric_models/flame_assets/flame_arkit_bs.npy"
# The secant tests need the local FLAME asset checkout; skip gracefully without.
needs_flame = pytest.mark.skipif(
    not (os.path.exists(MASKS) and os.path.exists(ARKIT)),
    reason="local FLAME assets (FLAME_masks.pkl / flame_arkit_bs.npy) not found")


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


def test_region_transform_scales_only_masked_verts():
    f = _field()
    with torch.no_grad():
        f.s_eye_log.copy_(torch.log(torch.tensor(1.5)))
    n = 200
    v = torch.rand(n, 3)
    # region_weights: 1.0 for the first 20 verts (the "eye"), 0 elsewhere.
    w = torch.zeros(n)
    w[:20] = 1.0
    out = f(v, region_weights={"eye": w})
    # verts fully outside the region are unchanged by the region transform.
    base = f(v)
    assert torch.allclose(out[50:], base[50:], atol=1e-5)
    # masked verts moved away from the region centroid (scale 1.5 > 1).
    assert not torch.allclose(out[:20], base[:20], atol=1e-4)


def test_local_jacobian_matches_finite_difference():
    f = _field().double()  # float64: FD of (g1-g0)/eps needs the precision
    with torch.no_grad():
        f.remap_incr.copy_(torch.randn(5) * 0.3)
        f.radial_log.copy_(torch.randn(6) * 0.2)
    v = torch.tensor([[0.12, 0.55, 0.08], [-0.2, 0.3, -0.05]], dtype=torch.float64)
    J = f.local_jacobian(v)                       # (2,3,3)
    eps = 1e-4
    for k in range(2):
        for j in range(3):
            dv = v.clone(); dv[k, j] += eps
            num = (f(dv)[k] - f(v)[k]) / eps
            assert torch.allclose(J[k, :, j], num, atol=1e-3)


def test_local_jacobian_includes_region_transforms():
    f = _field().double()  # float64: FD precision (see test above)
    with torch.no_grad():
        f.s_eye_log.copy_(torch.log(torch.tensor(1.8)))
    n = 120
    v = torch.rand(n, 3, dtype=torch.float64)
    w = torch.zeros(n, dtype=torch.float64); w[:30] = 1.0
    rw = {"eye": w}
    J_full = f.local_jacobian(v, region_weights=rw)   # (n,3,3)
    J_glob = f.local_jacobian(v)
    # Outside the region the two agree; inside they differ (region scaling
    # is now in the Jacobian, so the eye-enlarge shows up in J).
    assert torch.allclose(J_full[60:], J_glob[60:], atol=1e-5)
    assert not torch.allclose(J_full[:30], J_glob[:30], atol=1e-3)
    # FD-check the region-inclusive Jacobian against the same fixed-centroid
    # function jacrev differentiates: global field then eye transform about a
    # detached centroid.
    _, cen = f.forward(v, rw, return_centroids=True)
    cen = cen["eye"].detach()
    s = torch.exp(f.s_eye_log.detach())               # (1,) -> isotropic eye
    eps = 1e-4
    for k in (5, 10):                                 # two in-region verts
        for j in range(3):
            dv = v.clone(); dv[k, j] += eps
            g0 = f.forward(v[k:k+1])[0]
            g1 = f.forward(dv[k:k+1])[0]
            g0 = g0 + w[k] * (g0 - cen) * (s - 1.0)
            g1 = g1 + w[k] * (g1 - cen) * (s - 1.0)
            num = (g1 - g0) / eps
            assert torch.allclose(J_full[k, :, j], num, atol=1e-3)


def _chibi_field_on_template():
    """A non-trivial fitted-like ChibiField on the real 5023 FLAME template:
    lower-face compression, radial skull bulge, collapsed mouth, enlarged eye."""
    from chibi.fit import _load_obj_verts
    from chibi.landmarks import landmark_positions, FLAME_TEMPLATE
    v = _load_obj_verts(FLAME_TEMPLATE).double()
    f = ChibiField(y_crown=float(v[:, 1].max()),
                   y_chin=float(landmark_positions(v)[8, 1]),
                   z_center=float(v[:, 2].mean())).double()
    with torch.no_grad():
        f.remap_incr.copy_(torch.tensor([0.25, -0.35, 0.1, -0.2, 0.15]))
        f.radial_log.copy_(torch.linspace(0.0, 0.3, 6))
        f.s_mouth_y_log.fill_(math.log(0.55))
        f.s_eye_log.fill_(math.log(1.8))
    return f, v


@needs_flame
def test_secant_basis_is_exact_at_unit_amplitude():
    """secant_basis[k] == Phi(v + b_k) - Phi(v) by construction — exact at the
    full per-channel amplitude, where the Jacobian tangent is most wrong."""
    from chibi.landmarks import region_falloff_weights
    f, v = _chibi_field_on_template()
    rw = region_falloff_weights(v, MASKS)
    arkit = torch.as_tensor(np.load(ARKIT), dtype=torch.float64)   # (52,5023,3)
    secant = f.secant_basis(v, arkit, rw)
    base = f(v, region_weights=rw)
    for k in (24, 33, 40):                            # jawOpen + two mouth ch
        true1 = f(v + arkit[k], region_weights=rw) - base
        assert torch.allclose(secant[k], true1, atol=1e-9)


@needs_flame
def test_secant_basis_beats_tangent_on_mouth():
    """Rescaling a finite-amplitude mouth channel through the field's secant
    is far closer to the true field displacement than through the Jacobian
    (tangent at rest), which is ~80% wrong even at small amplitude."""
    from chibi.landmarks import region_falloff_weights
    f, v = _chibi_field_on_template()
    rw = region_falloff_weights(v, MASKS)
    arkit = torch.as_tensor(np.load(ARKIT), dtype=torch.float64)
    jaw = arkit[24]                                   # jawOpen
    base = f(v, region_weights=rw)
    secant = f.secant_basis(v, arkit, rw)
    J = f.local_jacobian(v, region_weights=rw)
    tangent = torch.einsum("vij,vj->vi", J, jaw)
    m = torch.where(rw["mouth"] > 0.05)[0]
    a = 0.5                                           # mid amplitude
    true = f(v + a * jaw, region_weights=rw) - base
    err_tangent = (a * tangent - true)[m].norm(dim=-1).mean()
    err_secant = (a * secant[24] - true)[m].norm(dim=-1).mean()
    assert err_secant < err_tangent * 0.3             # secant >3x closer
