"""Two-stage MediaPipe student (variant `bs_v2d`):

  latent (16, 64, 64) → ResNet-18 features (512) ──┬→ head_lmk → (106 × 2) normalized lmks
                                                    │
                                                    └─ concat → head_bs → 52 blendshapes

The landmark head is a regularizer: it forces the shared trunk to encode
explicit geometric structure that the blendshape head can then read off. By
construction, the blendshape head also gets the lmk_pred as a side-channel —
so even if the trunk hasn't fully encoded the geometry, the blendshape head
can lean on it directly.

Loss = λ₁ · MSE(lmk_pred, lmk_target_norm) + λ₂ · MSE(bs_pred, bs_target).
Default λ₁=0.5, λ₂=1.0.

Landmark targets are normalized: pixel coords [0, 512] / 512 → [0, 1] floats.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock

from arc_distill.stems import LatentStemFull64Native


VARIANTS = {"bs_v2d"}
N_LMK = 106
LMK_RES = 512.0  # source resolution used by extract_rendered_landmarks + face_attrs


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


class BlendshapeLandmarkStudent(nn.Module):
    def __init__(self, variant: str = "bs_v2d"):
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

        self.head_lmk = nn.Linear(512, N_LMK * 2)
        self.head_bs = nn.Sequential(
            nn.Linear(512 + N_LMK * 2, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 52),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        feats = self.pool(x).flatten(1)            # (B, 512)
        lmk = torch.sigmoid(self.head_lmk(feats))  # (B, 212) ∈ [0, 1]
        bs = torch.sigmoid(self.head_bs(torch.cat([feats, lmk], dim=-1)))  # (B, 52)
        return bs, lmk

    def trainable_parameters(self):
        return list(self.parameters())
