"""Closed-form pose unit tests including F_KP_REF kp_ref y-mirror."""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from arkit_bridge.closed_form_pose import (  # noqa: E402
    EULER_SIGNS, F_KP_REF, compose_kd, euler_to_rotmat,
)


def test_F_KP_REF_is_y_mirror():
    expected = torch.tensor([[1., 0., 0.],
                             [0., -1., 0.],
                             [0., 0., 1.]])
    assert torch.allclose(F_KP_REF, expected)
    assert torch.linalg.det(F_KP_REF).item() == -1.0


def test_compose_kd_zero_rotation_applies_F_to_kpref():
    """With identity R, compose_kd should apply F_KP_REF to kp_ref."""
    kp_ref = torch.randn(1, 21, 3)
    R = torch.eye(3).unsqueeze(0)
    s = torch.ones(1)
    t = torch.zeros(1, 2)
    out = compose_kd(kp_ref, R, s, t)
    expected = kp_ref @ F_KP_REF
    # tx,ty zero so out == expected exactly
    assert torch.allclose(out, expected, atol=1e-6)


def test_compose_kd_nonzero_rotation():
    """compose_kd applies F to kp_ref, then R, then s, then translates."""
    kp_ref = torch.tensor([[[1., 2., 3.]]])  # (1, 1, 3)
    yaw = torch.tensor([0.5]); pitch = torch.zeros(1); roll = torch.zeros(1)
    R = euler_to_rotmat(yaw, pitch, roll)
    s = torch.tensor([1.5])
    t = torch.zeros(1, 2)
    out = compose_kd(kp_ref, R, s, t)
    expected = (kp_ref @ F_KP_REF) @ R * 1.5
    assert torch.allclose(out, expected, atol=1e-5)


def test_euler_signs_preserved():
    """v4 must keep the v3 winner: (+1, -1, -1)."""
    assert EULER_SIGNS == (+1.0, -1.0, -1.0)
