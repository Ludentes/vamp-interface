"""Distill loss-mode unit tests for v4 (weighted_mse + varnorm_jvp)."""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from arkit_bridge.distill import _make_loss  # noqa: E402


def make_stats(B=8, n_b=58, m_shape=(32, 16), seed=0):
    """Synthetic stats matching v4 stats.npz schema (strict superset of v2)."""
    rng = np.random.default_rng(seed)
    return {
        "teacher_mean": rng.normal(0, 0.1, m_shape).astype(np.float32),
        "teacher_std": np.full(m_shape, 0.1, dtype=np.float32),
        "b_p95": np.full(n_b, 0.5, dtype=np.float32),
        "sample_weights": np.ones(B, dtype=np.float32),
        "paths": np.array([f"f{i}.pkl" for i in range(B)]),
        "freq_k": np.linspace(0.01, 0.5, n_b).astype(np.float32),
        "C": np.abs(rng.normal(0, 0.5, (m_shape[0] * m_shape[1], n_b))).astype(np.float32),
    }


def test_weighted_mse_finite_loss():
    stats = make_stats()
    fn = _make_loss("weighted_mse", stats, "cpu", lam_std=1.0)
    s = torch.zeros(8, 32, 16, requires_grad=True)
    t = torch.randn(8, 32, 16) * 0.1
    b = torch.randn(8, 58)
    loss, parts = fn(s, t, b=b)
    assert torch.isfinite(loss)
    assert "weighted_mse" in parts
    assert "std_match" in parts
    loss.backward()
    assert s.grad is not None
    assert torch.isfinite(s.grad).all()


def test_weighted_mse_rare_channel_amplifies():
    """In a mixed batch the rare-channel sample's gradient must dominate
    the common-channel sample's. We use B=2 because per-batch mean
    normalization (deliberate for training stability) collapses sw=1
    when the batch has a single sample."""
    stats = make_stats()
    stats["freq_k"] = np.array([0.001] + [0.5] * 57, dtype=np.float32)
    fn = _make_loss("weighted_mse", stats, "cpu", lam_std=0.0)

    pred = torch.zeros(2, 32, 16, requires_grad=True)
    target = torch.ones(2, 32, 16) * 0.1
    b = torch.zeros(2, 58)
    b[0, 0] = 0.5    # row 0: rare channel active
    b[1, 50] = 0.5   # row 1: common channel active
    loss, _ = fn(pred, target, b=b)
    loss.backward()
    g_rare = pred.grad[0].norm().item()
    g_common = pred.grad[1].norm().item()
    assert g_rare > g_common * 1.5, f"rare {g_rare} not > 1.5×common {g_common}"


def test_weighted_mse_handles_leading_singleton_dim():
    """PairDataset returns m_f with shape (1, 32, 16) per item, so batched
    tensors are (B, 1, 32, 16). Regression for shape mismatch where err2
    didn't reduce all non-batch dims before the per-sample weight."""
    stats = make_stats()
    fn = _make_loss("weighted_mse", stats, "cpu", lam_std=1.0)
    s = torch.zeros(8, 1, 32, 16, requires_grad=True)
    t = torch.randn(8, 1, 32, 16) * 0.1
    b = torch.randn(8, 58)
    loss, _ = fn(s, t, b=b)
    assert torch.isfinite(loss)
    loss.backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()


def test_anneal_beta_schedule():
    """β=1 at step 0, β=0 at and after anneal_to_step, β=0.5 at half."""
    stats = make_stats()
    fn = _make_loss("weighted_mse_jvp_anneal", stats, "cpu",
                    lam_std=1.0, lam_tail=0.5, lam_jvp=0.1, alpha=0.5,
                    anneal_to_step=1000)
    model = torch.nn.Sequential(
        torch.nn.Linear(58, 32 * 16),
        torch.nn.Unflatten(1, (32, 16)),
    )
    b = torch.randn(4, 58, requires_grad=True)
    pred = model(b)
    targ = torch.randn(4, 32, 16) * 0.1
    _, p0 = fn(pred, targ, b=b, model=model, step=0)
    _, p_half = fn(pred, targ, b=b, model=model, step=500)
    _, p_done = fn(pred, targ, b=b, model=model, step=1000)
    _, p_after = fn(pred, targ, b=b, model=model, step=10000)
    assert abs(p0["beta"] - 1.0) < 1e-6
    assert abs(p_half["beta"] - 0.5) < 1e-6
    assert abs(p_done["beta"]) < 1e-6
    assert abs(p_after["beta"]) < 1e-6


def test_anneal_at_full_aug_grad_flow():
    """β=1: gradient flows through both weighted_mse and JVP regularizer."""
    stats = make_stats()
    fn = _make_loss("weighted_mse_jvp_anneal", stats, "cpu",
                    lam_std=1.0, lam_jvp=0.1, alpha=0.5, anneal_to_step=1000)
    model = torch.nn.Sequential(
        torch.nn.Linear(58, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 32 * 16), torch.nn.Unflatten(1, (32, 16)),
    )
    b = torch.randn(8, 58, requires_grad=True)
    pred = model(b)
    targ = torch.randn(8, 32, 16) * 0.1
    loss, parts = fn(pred, targ, b=b, model=model, step=0)
    assert torch.isfinite(loss) and parts["jvp"] > 0  # JVP active
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()


def test_anneal_handles_leading_singleton_dim():
    """Same (B,1,32,16) student-output shape as MotEncoderStudent emits."""
    stats = make_stats()
    fn = _make_loss("weighted_mse_jvp_anneal", stats, "cpu",
                    lam_std=1.0, lam_jvp=0.1, alpha=0.5, anneal_to_step=1000)
    # tiny dummy "model" emitting (B, 1, 32, 16)
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(58, 32 * 16)
        def forward(self, b):
            return self.fc(b).view(b.shape[0], 1, 32, 16)
    m = M()
    b = torch.randn(8, 58, requires_grad=True)
    pred = m(b)
    targ = torch.randn(8, 1, 32, 16) * 0.1
    loss, _ = fn(pred, targ, b=b, model=m, step=0)
    assert torch.isfinite(loss)
    loss.backward()
    for p in m.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()


def test_anneal_raises_without_anneal_to_step():
    """Guard against silent v2 fallback when user forgets the flag."""
    stats = make_stats()
    try:
        _make_loss("weighted_mse_jvp_anneal", stats, "cpu", anneal_to_step=0)
    except ValueError as e:
        assert "anneal_to_step" in str(e)
    else:
        raise AssertionError("expected ValueError on anneal_to_step=0")


def test_anneal_at_done_equals_varnorm_std_tail():
    """β=0: loss equals plain varnorm_std_tail (JVP fully gone)."""
    stats = make_stats()
    fn_anneal = _make_loss("weighted_mse_jvp_anneal", stats, "cpu",
                           lam_std=1.0, lam_tail=0.5, lam_jvp=0.1,
                           alpha=0.5, anneal_to_step=100)
    fn_v2 = _make_loss("varnorm_std_tail", stats, "cpu",
                       lam_std=1.0, lam_tail=0.5)
    model = torch.nn.Sequential(
        torch.nn.Linear(58, 32 * 16), torch.nn.Unflatten(1, (32, 16)),
    )
    b = torch.randn(8, 58)
    pred = torch.randn(8, 32, 16, requires_grad=True)
    targ = torch.randn(8, 32, 16) * 0.1
    l_anneal, parts = fn_anneal(pred, targ, b=b, model=model, step=200)
    l_v2, _ = fn_v2(pred, targ)
    assert abs(parts["beta"]) < 1e-9
    assert abs(parts["jvp"]) < 1e-9     # JVP path skipped at β=0
    assert torch.allclose(l_anneal, l_v2, atol=1e-6)


def test_varnorm_jvp_finite_loss():
    stats = make_stats()
    fn = _make_loss("varnorm_jvp", stats, "cpu",
                    lam_std=1.0, lam_jvp=0.1, alpha=0.5)
    model = torch.nn.Sequential(
        torch.nn.Linear(58, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32 * 16),
        torch.nn.Unflatten(1, (32, 16)),
    )
    b = torch.randn(8, 58, requires_grad=True)
    pred = model(b)
    t = torch.randn(8, 32, 16) * 0.1
    loss, parts = fn(pred, t, b=b, model=model)
    assert torch.isfinite(loss)
    assert "jvp" in parts
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
