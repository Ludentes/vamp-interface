import torch
from arkit_bridge.encoder import ARKitParametricPoseGuider


def test_output_shape_matches_pose_guider():
    student = ARKitParametricPoseGuider()
    b = torch.zeros(2, 61)
    out = student(b)
    assert out.shape == (2, 320, 1, 64, 64), out.shape


def test_output_is_finite_under_random_input():
    student = ARKitParametricPoseGuider()
    b = torch.randn(4, 61)
    out = student(b)
    assert torch.isfinite(out).all()


def test_zero_input_does_not_crash():
    student = ARKitParametricPoseGuider()
    b = torch.zeros(1, 61)
    out = student(b)
    assert out.shape[1:] == (320, 1, 64, 64)


def test_param_count_under_5M():
    student = ARKitParametricPoseGuider()
    n = sum(p.numel() for p in student.parameters())
    assert n < 5_000_000, f"too big: {n}"
