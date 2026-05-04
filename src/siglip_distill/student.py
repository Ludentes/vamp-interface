"""SigLIP-2 image-embedding student for Flux VAE latents.

Takes (B, 16, 64, 64) latents → 1152-d real-valued vector. At inference the
caller L2-norms the output to put it on the SigLIP hypersphere.

Architecture (variant `sg_a`, ~12 M params, all trainable) — same trunk as
mediapipe_distill v2c, only the head differs:

  LatentStemFull64Native           (B, 64, 112, 112)
  ResNet-18-without-conv1 head     (B, 512, 14, 14)
  AdaptiveAvgPool2d + Flatten      (B, 512)
  Linear(512, 1152)                (B, 1152) ∈ ℝ
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock

from arc_distill.stems import LatentStemFull64Native


VARIANTS = {"sg_a", "sg_b"}
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


class SigLIPStudent(nn.Module):
    """
    Variants:
      sg_a — raw linear head, output ∈ ℝ¹¹⁵². Pre-L2-norm; magnitude drifts
             during training (v1 ended at pred_norm ≈ 0.88).
      sg_b — sg_a + L2-norm at the end so the output sits on the unit
             hypersphere by construction. Loss simplifies to (1 − cos)
             because MSE between two unit vectors is monotonic in cosine.
             Eliminates the magnitude bias that polluted v1's per-probe R².
    """

    def __init__(self, variant: str = "sg_a"):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant: {variant!r}")
        self.variant = variant
        self.l2_norm_output = (variant == "sg_b")

        self.stem = LatentStemFull64Native()
        self.layer1 = _make_layer(64, 64, n_blocks=2, stride=1)
        self.layer2 = _make_layer(64, 128, n_blocks=2, stride=2)
        self.layer3 = _make_layer(128, 256, n_blocks=2, stride=2)
        self.layer4 = _make_layer(256, 512, n_blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(512, EMB_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        x = self.head(x)
        if self.l2_norm_output:
            x = torch.nn.functional.normalize(x, dim=-1, eps=1e-8)
        return x

    def trainable_parameters(self):
        return list(self.parameters())
