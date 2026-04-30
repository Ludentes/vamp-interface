"""Projected gradient descent attack helper for adversarial training.

Implements PGD-K under L∞ norm bound on the input perturbation. Used by
`train_t_pgd.py` to find worst-case latent perturbations during each
training step; the training loss then includes both the clean-input MSE
and the adversarial-input MSE so the student becomes locally Lipschitz
in latent space.

Reference: Madry et al. 2018, "Towards Deep Learning Models Resistant
to Adversarial Attacks", https://arxiv.org/abs/1706.06083
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def pgd_perturb(
    model: torch.nn.Module,
    z: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    alpha: float,
    k: int,
) -> torch.Tensor:
    """Find an L∞-bounded latent perturbation that maximizes MSE(model(z+δ), y).

    The model's parameter `requires_grad` flags are toggled OFF inside the
    inner loop (we only differentiate w.r.t. δ) and restored before return.
    Model is left in eval mode for the inner loop, then restored.

    Args:
        model: blendshape student (any nn.Module returning (B, 52) sigmoid).
        z: (B, 16, 64, 64) input latent.
        y: (B, 52) teacher labels in [0, 1].
        eps: L∞ bound on per-element |δ|.
        alpha: PGD step size. Typically eps / 4.
        k: number of inner-loop steps. K=0 returns zero δ.

    Returns:
        δ tensor with same shape as z, satisfying ||δ||_∞ ≤ eps. Detached
        from any autograd graph; safe to add to z in the outer training step.
    """
    if k == 0 or eps == 0.0:
        return torch.zeros_like(z)

    was_training = model.training
    model.eval()
    saved_grad_flags = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)

    # Init: random uniform in [-eps, eps] (Madry-style; better than zero init)
    delta = torch.empty_like(z).uniform_(-eps, eps).requires_grad_(True)

    for _ in range(k):
        pred = model(z + delta)
        loss = F.mse_loss(pred, y)
        grad = torch.autograd.grad(loss, delta, only_inputs=True)[0]
        with torch.no_grad():
            delta.add_(alpha * grad.sign())
            delta.clamp_(-eps, eps)
        delta.requires_grad_(True)

    out = delta.detach()

    # restore model state
    for p, flag in zip(model.parameters(), saved_grad_flags):
        p.requires_grad_(flag)
    if was_training:
        model.train()
    return out
