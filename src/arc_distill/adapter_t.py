"""Noise-conditional ArcFace adapter (latent_a2_full_native_shallow_t2).

Path 2 from `docs/research/2026-04-30-noise-conditional-distill-design.md`:
same backbone + stem + layer1-unfrozen recipe as the shipped 0.881-cos
clean-only adapter, but with per-stage FiLM modulation conditioned on a
sinusoidal-t embedding. Five FiLM points:

    after stem (Conv_0)    (B,  64, 112, 112)
    after layer1 (Add_17)  (B,  64,  56,  56)
    after layer2 (Add_38)  (B, 128,  28,  28)
    after layer3 (Add_109) (B, 256,  14,  14)
    after layer4 (Add_125) (B, 512,   7,   7)

FiLM is injected via forward hooks on the converted-ONNX modules so the
frozen backbone weights stay frozen — only the FiLM modules (and the
existing trainable stem + layer1) train. FiLM linears are zero-init →
γ ≡ 0, β ≡ 0 → at start the network is exactly equivalent to the clean-only
adapter (modulo whatever weights are loaded from --init-from).

Path 1 retrain (random-t noise into the unmodified adapter) was shown to
saturate at val cos 0.525 at t=0.5 vs 0.869 at t=0 — the frozen IResNet50
cannot handle noisy inputs without per-stage conditioning.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import (
    DEFAULT_ONNX_PATH,
    attach_stem,
    load_frozen_iresnet50,
    mark_stem_trainable,
    unfreeze_layer1,
)
from .stems import LatentStemFull64Native


# Stage-boundary modules in the converted IResNet50 graph; (name, channels).
# Identified empirically — see docs/research/2026-04-30-arc-onnx-stage-probe.md
# (or run `arc_distill.adapter_t._probe_stages` once if regenerating).
FILM_STAGES = [
    ("Conv_0",   64),    # stem output (before native PRelu/BN, both bypassed for latent variants)
    ("Add_17",   64),    # end of layer1
    ("Add_38",  128),    # end of layer2
    ("Add_109", 256),    # end of layer3
    ("Add_125", 512),    # end of layer4
]


class SinusoidalTimeEmbedding(nn.Module):
    """t ∈ [0, 1] → (B, dim) sinusoidal embedding (DDPM-style)."""

    def __init__(self, dim: int = 128):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("dim must be even")
        self.dim = dim
        half = dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half) / half)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        freqs: torch.Tensor = self.get_buffer("freqs")
        args = (t.to(freqs) * 1000.0).unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class FiLM2d(nn.Module):
    """Per-channel (γ, β) modulation of a (B, C, H, W) tensor. Identity at init."""

    def __init__(self, t_dim: int, n_channels: int):
        super().__init__()
        self.proj = nn.Linear(t_dim, 2 * n_channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        gb = self.proj(t_emb)
        gamma, beta = gb.chunk(2, dim=-1)
        return x * (1.0 + gamma[..., None, None]) + beta[..., None, None]


class AdapterStudentT(nn.Module):
    """Noise-conditional adapter wrapping AdapterStudent's
    `latent_a2_full_native_shallow` recipe with per-stage FiLM hooks.

    forward(z_t, t):
        z_t : (B, 16, 64, 64)   partially-noised Flux VAE latent
        t   : (B,)              noise level ∈ [0, 1] (rectified-flow blend)
        out : (B, 512)          unit-norm ArcFace embedding
    """

    VARIANT_NAME = "latent_a2_full_native_shallow_t2"

    def __init__(self, t_emb_dim: int = 128, t_proj_dim: int = 256,
                 onnx_path: Path = DEFAULT_ONNX_PATH):
        super().__init__()
        self.variant = self.VARIANT_NAME
        self.t_emb_dim = t_emb_dim
        self.t_proj_dim = t_proj_dim

        # Sinusoidal-t embedding + 2-layer MLP (matches sg_c).
        self.t_embed = nn.Sequential(
            SinusoidalTimeEmbedding(t_emb_dim),
            nn.Linear(t_emb_dim, t_proj_dim),
            nn.SiLU(),
            nn.Linear(t_proj_dim, t_proj_dim),
        )

        # Build the same recipe as latent_a2_full_native_shallow.
        self.backbone = load_frozen_iresnet50(onnx_path)
        self.stem = LatentStemFull64Native()
        attach_stem(self.backbone, self.stem, bypass_native_prelu_bn=True)
        stem_params = mark_stem_trainable(self.backbone, retrain_prelu_bn=False)
        self.layer1_unfrozen = unfreeze_layer1(self.backbone)
        self._stem_params = stem_params

        # Five FiLM modules — one per stage boundary.
        self.films = nn.ModuleDict({
            mod_name: FiLM2d(t_proj_dim, ch) for mod_name, ch in FILM_STAGES
        })

        # Holder for current batch's t_emb so the forward-hook closures can read it.
        self._cached_t_emb: torch.Tensor | None = None
        self._hooks: list = []
        self._register_film_hooks()

        self.trainable = (
            stem_params + self.layer1_unfrozen
            + list(self.t_embed.parameters())
            + list(self.films.parameters())
        )

    def _register_film_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        for mod_name, _ch in FILM_STAGES:
            mod = getattr(self.backbone, mod_name)
            film_mod = self.films[mod_name]
            assert isinstance(film_mod, FiLM2d)
            hook = mod.register_forward_hook(self._make_film_hook(film_mod))
            self._hooks.append(hook)

    def _make_film_hook(self, film: FiLM2d):
        def hook(_module, _inp, output):
            t_emb = self._cached_t_emb
            if t_emb is None:
                return output
            return film(output, t_emb)
        return hook

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        self._cached_t_emb = self.t_embed(t)
        try:
            z = self.backbone(z_t)
        finally:
            self._cached_t_emb = None
        return F.normalize(z, dim=-1)

    def trainable_parameters(self):
        return list(self.trainable)

    def parameter_groups(self, stem_lr: float, backbone_lr: float) -> list[dict]:
        """Three groups: stem at stem_lr, layer1 at backbone_lr, FiLM+t_embed at stem_lr.

        FiLM modules are entirely new → use the higher stem_lr so they catch up
        to the existing-trained stem fast."""
        film_params = list(self.t_embed.parameters()) + list(self.films.parameters())
        return [
            {"params": self._stem_params, "lr": stem_lr},
            {"params": self.layer1_unfrozen, "lr": backbone_lr},
            {"params": film_params, "lr": stem_lr},
        ]

    def train(self, mode: bool = True):
        """Mirrors AdapterStudent.train: only stem honours mode; deep backbone
        stays in eval; layer1 BN stays frozen."""
        self.training = mode
        self.backbone.eval()
        self.backbone.Conv_0.train(mode)
        nn.Module.train(self.films, mode)
        nn.Module.train(self.t_embed, mode)
        return self
