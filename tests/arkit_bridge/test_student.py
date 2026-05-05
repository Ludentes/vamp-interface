import torch
from arkit_bridge.student import MotEncoderStudent


def test_output_shape():
    out = MotEncoderStudent()(torch.zeros(2, 58))
    assert out.shape == (2, 1, 32, 16)


def test_output_finite():
    assert torch.isfinite(MotEncoderStudent()(torch.randn(4, 58))).all()


def test_zero_init_outputs_zero():
    out = MotEncoderStudent()(torch.randn(4, 58))
    assert torch.allclose(out, torch.zeros_like(out))


def test_param_count_under_3M():
    n = sum(p.numel() for p in MotEncoderStudent().parameters())
    assert n < 3_000_000, n
