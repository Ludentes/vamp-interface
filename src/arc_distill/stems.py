"""Trainable stems for the frozen-backbone adapter students.

Each stem produces (B, 64, 112, 112) — the same shape Conv_0 produced in the
native IResNet50 — so the rest of the frozen backbone runs unchanged.

Pixel-A: input (3, 112, 112) RGB. Replaces only the first 3x3 conv; the
  native PRelu_1 + BN_2 are retained (and reset to fresh init).
Latent-A (upsample / native): input (16, 14, 14) Flux VAE latent. Stem
  includes its own activation + BN; we bypass native PRelu_1 + BN_2.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelStem(nn.Module):
    """Single 3x3 conv 3->64 stride 1, mirroring IResNet50's Conv_0 exactly
    (including bias — ArcFace ONNX absorbed the following BN into Conv_0,
    so the converted Conv2d has bias=True)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="leaky_relu")
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class LatentStemUpsample(nn.Module):
    """Bilinear-upsample 14->112, then Conv 16->64, BN, PReLU."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64, eps=1e-5)
        self.act = nn.PReLU(64)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="leaky_relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(112, 112), mode="bilinear", align_corners=False)
        return self.act(self.bn(self.conv(x)))


class LatentStemNative(nn.Module):
    """Native-stride: ConvTranspose2d(s=8) for 14->112 without bilinear interp."""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(16, 64, kernel_size=3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(64, eps=1e-5)
        self.a1 = nn.PReLU(64)
        self.up = nn.ConvTranspose2d(64, 64, kernel_size=8, stride=8, bias=False)
        self.b2 = nn.BatchNorm2d(64, eps=1e-5)
        self.a2 = nn.PReLU(64)
        for w in (self.c1.weight, self.up.weight):
            nn.init.kaiming_normal_(w, mode="fan_out", nonlinearity="leaky_relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.a1(self.b1(self.c1(x)))
        return self.a2(self.b2(self.up(x)))
