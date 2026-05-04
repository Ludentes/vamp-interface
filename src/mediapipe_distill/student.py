"""MediaPipe-blendshape student for Flux VAE latents.

Takes (B, 16, 64, 64) latents → 52-d blendshape vector in [0, 1].

Architecture (variant `bs_a`, ~11 M params, all trainable):

  LatentStemFull64Native               — same stem as arc_distill (ConvT
                                         stride-2 64→128, crop 112), 0.27 M
        │ (B, 64, 112, 112)
        ▼
  ResNet-18-without-conv1 head         — torchvision BasicBlock layers
                                         {2 × 64ch, 2 × 128ch s2, 2 × 256ch s2,
                                         2 × 512ch s2}, ~10 M
        │ (B, 512, 7, 7)
        ▼
  AdaptiveAvgPool2d + Flatten          → (B, 512)
        │
        ▼
  Linear(512, 52) + Sigmoid             → (B, 52) ∈ [0, 1]

ArcFace embeddings discard expression by design, so the arc_distill
backbone is the wrong feature space for blendshapes — building from
scratch is the right call here. See lessons doc.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock

from arc_distill.stems import LatentStemFull64Native


VARIANTS = {"bs_a"}


def _make_layer(in_ch: int, out_ch: int, n_blocks: int, stride: int) -> nn.Sequential:
    """ResNet18-style stage. First block may downsample (stride>1)."""
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


class BlendshapeStudent(nn.Module):
    def __init__(self, variant: str = "bs_a"):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant: {variant!r}")
        self.variant = variant

        self.stem = LatentStemFull64Native()                # (B, 64, 112, 112)
        self.layer1 = _make_layer(64, 64, n_blocks=2, stride=1)   # (B, 64, 112, 112)
        self.layer2 = _make_layer(64, 128, n_blocks=2, stride=2)  # (B, 128, 56, 56)
        self.layer3 = _make_layer(128, 256, n_blocks=2, stride=2) # (B, 256, 28, 28)
        self.layer4 = _make_layer(256, 512, n_blocks=2, stride=2) # (B, 512, 14, 14)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(512, 52)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return torch.sigmoid(self.head(x))

    def trainable_parameters(self):
        return list(self.parameters())
