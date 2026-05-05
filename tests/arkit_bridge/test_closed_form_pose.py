import torch
from arkit_bridge.closed_form_pose import euler_to_rotmat, compose_kd


def test_euler_to_rotmat_zero_is_identity():
    R = euler_to_rotmat(torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0))
    assert torch.allclose(R, torch.eye(3), atol=1e-6)


def test_euler_to_rotmat_orthonormal():
    R = euler_to_rotmat(torch.tensor(0.3), torch.tensor(-0.2), torch.tensor(0.1))
    assert torch.allclose(R @ R.transpose(-1, -2), torch.eye(3), atol=1e-5)
    assert abs(torch.det(R).item() - 1.0) < 1e-5


def test_compose_kd_zero_rotation_returns_scaled_translated_kp():
    kp_ref = torch.randn(1, 21, 3)
    t_ref = torch.tensor([[0.1, -0.2, 0.0]])
    s_ref = torch.tensor([[1.5]])
    R = torch.eye(3).unsqueeze(0)
    k_d = compose_kd(kp_ref, R, s_ref, t_ref)
    expected = kp_ref * 1.5
    expected[..., 0:2] = expected[..., 0:2] + t_ref[:, None, 0:2]
    assert torch.allclose(k_d, expected, atol=1e-5)


def test_euler_matches_personalive_get_rotation_matrix():
    """Lock in chirality: our R agrees with PersonaLive's at non-trivial angles.

    PersonaLive's get_rotation_matrix takes degrees and divides by 180*PI;
    we take radians directly. Both use row-vector convention (kp @ R).
    """
    import sys, os, math
    pl = os.path.expanduser("~/w/PersonaLive")
    if pl not in sys.path:
        sys.path.insert(0, pl)
    from src.liveportrait.camera import get_rotation_matrix
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


def test_yaw_rotates_x_axis_to_negative_z():
    """yaw=π/2 about Y in row-vector convention sends +X -> -Z."""
    R = euler_to_rotmat(
        torch.tensor(torch.pi / 2), torch.tensor(0.0), torch.tensor(0.0)
    )
    x_row = torch.tensor([[1.0, 0.0, 0.0]])
    out = x_row @ R
    assert torch.allclose(out, torch.tensor([[0.0, 0.0, -1.0]]), atol=1e-5), out


def test_compose_kd_shape():
    kp_ref = torch.randn(1, 21, 3)
    R = torch.eye(3).unsqueeze(0)
    t_ref = torch.zeros(1, 3); s_ref = torch.ones(1, 1)
    assert compose_kd(kp_ref, R, s_ref, t_ref).shape == (1, 21, 3)
