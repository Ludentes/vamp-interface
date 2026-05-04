"""v2b — atom-only specialist (variant `bs_v2b`).

Same v2c trunk; head outputs 8 NMF atoms instead of 52 blendshapes. Atoms
can be slightly negative (lstsq projection of the NMF basis), so the head
is plain Linear without sigmoid.

Use case: identity/expression-preserving loss for slider training where
the consumer cares about a small set of decorrelated expression axes
(smile, jaw, anger, surprise, disgust, pucker, lip_press, alpha_interp_attn).
For those axes, atom-prediction is cleaner and more compact than going
through 52-d bs predictions and projecting.

Per-channel bs reconstruction by atom decoding is *capped* by the NMF
reconstruction R² — which is poor for several blendshape channels (the
NMF basis was tuned for high-variance expression axes, not the full 52).
Use v2c if you need per-channel bs predictions; use v2b only for atom
loss terms.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock

from arc_distill.stems import LatentStemFull64Native


VARIANTS = {"bs_v2b"}
N_ATOMS = 8


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


class AtomStudent(nn.Module):
    def __init__(self, variant: str = "bs_v2b"):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant: {variant!r}")
        self.variant = variant

        self.stem = LatentStemFull64Native()
        self.layer1 = _make_layer(64, 64, n_blocks=2, stride=1)
        self.layer2 = _make_layer(64, 128, n_blocks=2, stride=2)
        self.layer3 = _make_layer(128, 256, n_blocks=2, stride=2)
        self.layer4 = _make_layer(256, 512, n_blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(512, N_ATOMS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.head(x)

    def trainable_parameters(self):
        return list(self.parameters())
