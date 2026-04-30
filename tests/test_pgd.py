"""Unit tests for src/mediapipe_distill/pgd.py."""
from __future__ import annotations

import torch
import torch.nn as nn

from mediapipe_distill.pgd import pgd_perturb


class _ToyModel(nn.Module):
    """A trivial linear-then-sigmoid model so we can verify PGD increases loss."""

    def __init__(self, in_dim: int = 16 * 64 * 64, out_dim: int = 52):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.lin(z.flatten(1)))


def test_pgd_respects_eps_linf_bound():
    torch.manual_seed(0)
    model = _ToyModel().eval()
    z = torch.randn(4, 16, 64, 64)
    y = torch.rand(4, 52)
    eps = 0.05
    delta = pgd_perturb(model, z, y, eps=eps, alpha=eps / 4, k=5)
    assert delta.shape == z.shape
    assert delta.abs().max().item() <= eps + 1e-6, (
        f"PGD δ violated L∞ bound: max |δ|={delta.abs().max().item()} > {eps}"
    )


def test_pgd_increases_mse_loss():
    """The whole point of PGD: bs(z + δ) should be FURTHER from y than bs(z)."""
    torch.manual_seed(0)
    model = _ToyModel().eval()
    z = torch.randn(4, 16, 64, 64)
    y = torch.rand(4, 52)
    eps = 0.05
    delta = pgd_perturb(model, z, y, eps=eps, alpha=eps / 4, k=5)
    with torch.no_grad():
        loss_clean = ((model(z) - y) ** 2).mean().item()
        loss_adv = ((model(z + delta) - y) ** 2).mean().item()
    assert loss_adv > loss_clean, (
        f"PGD failed to increase loss: clean={loss_clean:.6f} adv={loss_adv:.6f}"
    )


def test_pgd_preserves_model_grad_state():
    """PGD must not leave model.requires_grad in a bad state."""
    torch.manual_seed(0)
    model = _ToyModel().train()
    pre_state = [p.requires_grad for p in model.parameters()]
    z = torch.randn(2, 16, 64, 64)
    y = torch.rand(2, 52)
    pgd_perturb(model, z, y, eps=0.05, alpha=0.0125, k=3)
    post_state = [p.requires_grad for p in model.parameters()]
    assert pre_state == post_state, "PGD changed model parameter requires_grad flags"


def test_pgd_zero_steps_returns_zero_delta():
    """Edge case: K=0 should be a no-op (returns zero δ)."""
    model = _ToyModel().eval()
    z = torch.randn(2, 16, 64, 64)
    y = torch.rand(2, 52)
    delta = pgd_perturb(model, z, y, eps=0.05, alpha=0.0125, k=0)
    assert delta.abs().max().item() == 0.0


def test_pgd_zero_eps_returns_zero_delta():
    """At eps=0.0 the attacked path should match the clean path exactly."""
    torch.manual_seed(0)
    model = _ToyModel().eval()
    z = torch.randn(4, 16, 64, 64)
    delta = pgd_perturb(model, z, torch.rand(4, 52), eps=0.0, alpha=0.0, k=5)
    assert delta.abs().max().item() == 0.0, (
        "eps=0 should produce zero δ regardless of K"
    )
