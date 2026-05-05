import torch
from arkit_bridge.student import MotEncoderStudent


def test_output_shape():
    out = MotEncoderStudent()(torch.zeros(2, 58))
    assert out.shape == (2, 1, 32, 16)


def test_output_finite():
    assert torch.isfinite(MotEncoderStudent()(torch.randn(4, 58))).all()


def test_zero_init_outputs_zero_with_nonzero_trunk():
    """Zero-init must come from head (not vacuous trunk=0): trunk activations
    on random input must be non-zero, but final output must still be zero."""
    s = MotEncoderStudent()
    x = torch.randn(4, 58)
    h = s.trunk(x)
    assert h.abs().sum() > 0, "trunk produced all-zero activations; test is vacuous"
    out = s(x)
    assert torch.allclose(out, torch.zeros_like(out))


def test_view_ordering_l_then_c():
    """Confirm head output reshapes as (B, 1, L=32, C=16): writing a known
    value to head.bias[k] should appear at flat index k under (l, c) order."""
    s = MotEncoderStudent()
    with torch.no_grad():
        s.head.bias.zero_()
        s.head.bias[0] = 1.0           # row 0, col 0
        s.head.bias[16] = 2.0          # row 1, col 0  (only if order is l-major)
        s.head.bias[32 * 16 - 1] = 3.0 # last cell
    out = s(torch.zeros(1, 58))
    assert out.shape == (1, 1, 32, 16)
    assert out[0, 0, 0, 0].item() == 1.0
    assert out[0, 0, 1, 0].item() == 2.0
    assert out[0, 0, 31, 15].item() == 3.0


def test_gradient_decreases_loss():
    """One AdamW step on a fixed batch must reduce MSE — guards against
    detached graphs / frozen params."""
    torch.manual_seed(0)
    s = MotEncoderStudent()
    # Break zero-init so there's a gradient signal from the head too.
    with torch.no_grad():
        s.head.weight.normal_(std=0.01)
    opt = torch.optim.AdamW(s.parameters(), lr=1e-2)
    x = torch.randn(8, 58)
    y = torch.randn(8, 1, 32, 16)
    l0 = torch.nn.functional.mse_loss(s(x), y).item()
    for _ in range(5):
        opt.zero_grad()
        torch.nn.functional.mse_loss(s(x), y).backward()
        opt.step()
    l1 = torch.nn.functional.mse_loss(s(x), y).item()
    assert l1 < l0, (l0, l1)


def test_param_count_under_3M():
    n = sum(p.numel() for p in MotEncoderStudent().parameters())
    assert n < 3_000_000, n
