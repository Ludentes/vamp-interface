"""Noise-conditional SigLIP-2 image-embedding student (sg_c variant).

Same trunk as sg_b — LatentStemFull64Native + ResNet-18 layers + pool +
Linear(512, 1152) + F.normalize — but takes (z_t, t) instead of clean z_0,
and threads `t` through every stage via per-block FiLM modulation:

    γ_l, β_l = MLP_l(sinusoidal_emb(t))
    x_l      = block_l(x) * (1 + γ_l) + β_l

`t ∈ [0, 1]` interpreted as rectified-flow blend factor:
    z_t = (1 − t) · z_0 + t · ε,  ε ~ 𝒩(0, I)
i.e. t = 0 is clean, t = 1 is pure noise. Caller passes the same t at
inference that they used for the noisy latent.

FiLM linears are zero-initialised so the noise-conditional student starts
exactly equivalent to a clean-only student (γ ≡ 0, β ≡ 0) and learns to
deviate from there. Total extra params ≈ 600 K on top of the 12 M trunk.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock

from arc_distill.stems import LatentStemFull64Native


VARIANTS = {"sg_c"}
EMB_DIM = 1152


def _make_layer(in_ch: int, out_ch: int, n_blocks: int, stride: int) -> nn.Sequential:
    downsample: nn.Module | None = None
    if stride != 1 or in_ch != out_ch:
        downsample = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        )
    layers: list[nn.Module] = [BasicBlock(in_ch, out_ch, stride=stride, downsample=downsample)]
    for _ in range(1, n_blocks):
        layers.append(BasicBlock(out_ch, out_ch))
    return nn.Sequential(*layers)


class SinusoidalTimeEmbedding(nn.Module):
    """Continuous-t sinusoidal embedding.

    For t ∈ [0, 1], multiply by 1000 before computing sin/cos so the embed
    spans a comparable range to DDPM-style integer-step embeddings (the
    downstream MLP doesn't care about the exact scale, but pinning to a
    standard convention makes hyper-params transferable)."""

    def __init__(self, dim: int = 128):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("dim must be even")
        self.dim = dim
        half = dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half) / half)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) ∈ [0, 1]. Scale to [0, 1000] for DDPM-comparable embeddings.
        freqs: torch.Tensor = self.get_buffer("freqs")
        args = (t.to(freqs) * 1000.0).unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class FiLM(nn.Module):
    """Per-channel affine modulation (γ, β) from a t-embedding.

    Output is identity at init: weights and bias zero → γ ≡ 0, β ≡ 0 →
    `x * 1 + 0 == x`. Network starts behaviourally equivalent to clean-only."""

    def __init__(self, t_dim: int, n_channels: int):
        super().__init__()
        self.proj = nn.Linear(t_dim, 2 * n_channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.n_channels = n_channels

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W); t_emb: (B, t_dim)
        gb = self.proj(t_emb)
        gamma, beta = gb.chunk(2, dim=-1)
        return x * (1.0 + gamma[..., None, None]) + beta[..., None, None]


class SigLIPStudentT(nn.Module):
    """sg_c — noise-conditional SigLIP-2 student.

    forward(z_t, t):
        z_t : (B, 16, 64, 64) — partially noised Flux VAE latent
        t   : (B,)            — noise level in [0, 1] (rectified flow blend)
        out : (B, 1152)       — unit-norm SigLIP-2 image embedding
    """

    def __init__(self, variant: str = "sg_c", t_emb_dim: int = 128, t_proj_dim: int = 256):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant: {variant!r}")
        self.variant = variant
        self.t_emb_dim = t_emb_dim
        self.t_proj_dim = t_proj_dim

        self.t_embed = nn.Sequential(
            SinusoidalTimeEmbedding(t_emb_dim),
            nn.Linear(t_emb_dim, t_proj_dim),
            nn.SiLU(),
            nn.Linear(t_proj_dim, t_proj_dim),
        )

        self.stem = LatentStemFull64Native()
        self.film_stem = FiLM(t_proj_dim, 64)

        self.layer1 = _make_layer(64, 64, n_blocks=2, stride=1)
        self.film1 = FiLM(t_proj_dim, 64)
        self.layer2 = _make_layer(64, 128, n_blocks=2, stride=2)
        self.film2 = FiLM(t_proj_dim, 128)
        self.layer3 = _make_layer(128, 256, n_blocks=2, stride=2)
        self.film3 = FiLM(t_proj_dim, 256)
        self.layer4 = _make_layer(256, 512, n_blocks=2, stride=2)
        self.film4 = FiLM(t_proj_dim, 512)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(512, EMB_DIM)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.t_embed(t)
        x = self.stem(z_t)
        x = self.film_stem(x, t_emb)
        x = self.film1(self.layer1(x), t_emb)
        x = self.film2(self.layer2(x), t_emb)
        x = self.film3(self.layer3(x), t_emb)
        x = self.film4(self.layer4(x), t_emb)
        x = self.pool(x).flatten(1)
        x = self.head(x)
        return F.normalize(x, dim=-1, eps=1e-8)

    def trainable_parameters(self):
        return list(self.parameters())
