"""Closed-form pose unit tests.

Two layers:
  - euler_to_rotmat: chirality + orthonormality + PersonaLive parity
    (load-bearing — must agree exactly with PersonaLive's
    `liveportrait/camera.get_rotation_matrix` for our R^T row-vector path).
  - compose_kd: F_KP_REF = diag(+1,-1,+1) applied per-frame as
    R_eff = F R F^T (frame conjugation, matching v3's score_combo).
    On the 5_yaw verification clip this drops mean angular distance
    from 0.298 (F=I) to 0.233 rad (-22%); see closed_form_pose.py for
    the full provenance.
"""

import math
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from arkit_bridge.closed_form_pose import (  # noqa: E402
    EULER_SIGNS, F_KP_REF, compose_kd, euler_to_rotmat,
)


# ---- euler_to_rotmat: chirality / orthonormality / PersonaLive parity ----

def test_euler_to_rotmat_zero_is_identity():
    R = euler_to_rotmat(torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0))
    assert torch.allclose(R, torch.eye(3), atol=1e-6)


def test_euler_to_rotmat_orthonormal():
    R = euler_to_rotmat(torch.tensor(0.3), torch.tensor(-0.2), torch.tensor(0.1))
    assert torch.allclose(R @ R.transpose(-1, -2), torch.eye(3), atol=1e-5)
    assert abs(torch.det(R).item() - 1.0) < 1e-5


def test_yaw_rotates_x_axis_to_negative_z():
    """yaw=π/2 about Y in row-vector convention sends +X → −Z. Chirality lock."""
    R = euler_to_rotmat(
        torch.tensor(torch.pi / 2), torch.tensor(0.0), torch.tensor(0.0)
    )
    x_row = torch.tensor([[1.0, 0.0, 0.0]])
    out = x_row @ R
    assert torch.allclose(out, torch.tensor([[0.0, 0.0, -1.0]]), atol=1e-5), out


def test_euler_matches_personalive_get_rotation_matrix():
    """Lock in chirality: our R agrees with PersonaLive's at non-trivial angles.

    PersonaLive's get_rotation_matrix takes degrees and divides by 180*PI;
    we take radians directly. Both use row-vector convention (kp @ R).
    Skipped if PersonaLive isn't checked out (CI-friendly).
    """
    import pytest
    pl = os.path.expanduser("~/w/PersonaLive")
    if not os.path.isdir(pl):
        pytest.skip(f"PersonaLive checkout not found at {pl}")
    if pl not in sys.path:
        sys.path.insert(0, pl)
    try:
        from src.liveportrait.camera import get_rotation_matrix
    except ImportError as e:  # pragma: no cover
        pytest.skip(f"PersonaLive import failed: {e}")
    yaw_rad = 0.3; pitch_rad = -0.2; roll_rad = 0.1
    R_ours = euler_to_rotmat(
        torch.tensor(yaw_rad), torch.tensor(pitch_rad), torch.tensor(roll_rad)
    )
    R_pl = get_rotation_matrix(
        torch.tensor([math.degrees(pitch_rad)]),
        torch.tensor([math.degrees(yaw_rad)]),
        torch.tensor([math.degrees(roll_rad)]),
    ).squeeze(0)
    assert torch.allclose(R_ours, R_pl, atol=1e-5), (R_ours, R_pl)


# ---- F_KP_REF: y-mirror, per render-verified v3 winner ----

def test_F_KP_REF_is_y_mirror():
    expected = torch.tensor([[1., 0., 0.],
                             [0., -1., 0.],
                             [0., 0., 1.]])
    assert torch.allclose(F_KP_REF, expected)
    assert torch.linalg.det(F_KP_REF).item() == -1.0


def test_euler_signs_preserved():
    """v4 must keep the v3-calibration winner: (+1, -1, -1)."""
    assert EULER_SIGNS == (+1.0, -1.0, -1.0)


# ---- compose_kd: F applied once on kp_ref ----

def test_compose_kd_zero_rotation_is_identity_on_kpref():
    """Identity R: F R F^T = I, so k_d = kp_ref (zero translation, unit scale)."""
    kp_ref = torch.randn(1, 21, 3)
    R = torch.eye(3).unsqueeze(0)
    s = torch.ones(1)
    t = torch.zeros(1, 2)
    out = compose_kd(kp_ref, R, s, t)
    assert torch.allclose(out, kp_ref, atol=1e-6)


def test_compose_kd_zero_rotation_with_scale_and_translation():
    """Identity R, nonzero (s, t): out = kp_ref * s + (tx, ty, 0)."""
    kp_ref = torch.randn(1, 21, 3)
    t_ref = torch.tensor([[0.1, -0.2, 0.0]])
    s_ref = torch.tensor([[1.5]])
    R = torch.eye(3).unsqueeze(0)
    k_d = compose_kd(kp_ref, R, s_ref, t_ref)
    expected = kp_ref * 1.5
    expected[..., 0:2] = expected[..., 0:2] + t_ref[:, None, 0:2]
    assert torch.allclose(k_d, expected, atol=1e-5)


def test_compose_kd_nonzero_rotation_conjugates_R_by_F():
    """compose_kd substitutes R with F R F^T (frame conjugation per v3 calibration)."""
    kp_ref = torch.tensor([[[1., 2., 3.]]])  # (1, 1, 3)
    yaw = torch.tensor([0.5]); pitch = torch.zeros(1); roll = torch.zeros(1)
    R = euler_to_rotmat(yaw, pitch, roll)
    s = torch.tensor([1.5])
    t = torch.zeros(1, 2)
    out = compose_kd(kp_ref, R, s, t)
    R_eff = F_KP_REF @ R @ F_KP_REF.transpose(-1, -2)
    expected = (kp_ref @ R_eff) * 1.5
    assert torch.allclose(out, expected, atol=1e-5)


def test_F_KP_REF_is_involution():
    """F = F^T = F^{-1} (diagonal sign matrix). Conjugation F R F^T preserves
    rotations about y, sign-flips x/z components."""
    F = F_KP_REF
    assert torch.allclose(F @ F.transpose(-1, -2), torch.eye(3), atol=1e-6)
    yaw_only = euler_to_rotmat(torch.tensor(0.4), torch.tensor(0.0), torch.tensor(0.0))
    assert torch.allclose(F @ yaw_only @ F.transpose(-1, -2), yaw_only, atol=1e-5)


def test_compose_kd_shape():
    kp_ref = torch.randn(1, 21, 3)
    R = torch.eye(3).unsqueeze(0)
    t_ref = torch.zeros(1, 3); s_ref = torch.ones(1, 1)
    assert compose_kd(kp_ref, R, s_ref, t_ref).shape == (1, 21, 3)
