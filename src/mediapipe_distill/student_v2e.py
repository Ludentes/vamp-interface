"""v2e — U-Net decoder for the landmark head.

Motivation: v2d's lmk head plateaued at the per-landmark population-mean
baseline (≈63 px RMSE on 512² coords). Diagnosis: the trunk's
AdaptiveAvgPool2d collapses spatial structure, so a Linear(512, 212) head
can't recover per-instance landmark positions — only the bias (population
mean) is learnable. The bs head therefore gets a useless lmk_pred side-channel.

v2e fixes this by giving the lmk path access to the trunk's spatial feature
maps via a U-Net decoder with skip connections, then producing per-landmark
heatmaps and applying differentiable softargmax to recover (x, y).

Architecture:

  stem (16,64,64) → (64,112,112)             [skip1]
  layer1                  → (64,112,112)
  layer2 (s2)             → (128,56,56)      [skip2]
  layer3 (s2)             → (256,28,28)      [skip3]
  layer4 (s2)             → (512,14,14)      [bottom]
                              ├─ AvgPool → (512,)
                              │              └─ head_bs (concat with lmk_pred)
                              └─ U-Net decoder
                                   up14→28: ConvTranspose2d, concat skip3 (256), Conv → (256,28,28)
                                   up28→56: ConvTranspose2d, concat skip2 (128), Conv → (128,56,56)
                                   1×1 conv → (106,56,56) heatmap
                                   softmax2d + softargmax → (B,106,2) ∈ [0,1]

Output resolution 56² → pixel ceiling 512/56 ≈ 9 px (more than enough; we
just need spatial gradient flow to the trunk, not subpixel landmarks).

bs head: concat[features (512,), lmk_pred (212,)] → Linear→ReLU→Linear → 52.

Loss = λ_bs · MSE(bs_pred, bs_target) + λ_lmk · MSE(lmk_pred, lmk_target).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock

from arc_distill.stems import LatentStemFull64Native


VARIANTS = {"bs_v2e"}
N_LMK = 106
HEATMAP_RES = 56


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


class _UpBlock(nn.Module):
    """Upsample by 2 + concat skip + 3×3 conv → out_ch."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch // 2 + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # spatial size match (skip and up may differ by 1 px in odd cases)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


def _softargmax_2d(heatmap: torch.Tensor) -> torch.Tensor:
    """heatmap: (B, K, H, W) logits → (B, K, 2) coords ∈ [0, 1].

    softmax across (H, W), expectation over normalized grid.
    """
    B, K, H, W = heatmap.shape
    flat = heatmap.reshape(B, K, H * W)
    prob = F.softmax(flat, dim=-1).reshape(B, K, H, W)
    # normalized grid: y in [0, 1] over H, x in [0, 1] over W
    ys = torch.linspace(0.0, 1.0, H, device=heatmap.device).view(1, 1, H, 1)
    xs = torch.linspace(0.0, 1.0, W, device=heatmap.device).view(1, 1, 1, W)
    y = (prob * ys).sum(dim=(-2, -1))   # (B, K)
    x = (prob * xs).sum(dim=(-2, -1))   # (B, K)
    return torch.stack([x, y], dim=-1)  # (B, K, 2) — order (x, y) matches insightface convention


class BlendshapeLandmarkStudentUNet(nn.Module):
    def __init__(self, variant: str = "bs_v2e"):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant: {variant!r}")
        self.variant = variant

        self.stem = LatentStemFull64Native()                       # (B,64,112,112)
        self.layer1 = _make_layer(64, 64, n_blocks=2, stride=1)    # (B,64,112,112)
        self.layer2 = _make_layer(64, 128, n_blocks=2, stride=2)   # (B,128,56,56)
        self.layer3 = _make_layer(128, 256, n_blocks=2, stride=2)  # (B,256,28,28)
        self.layer4 = _make_layer(256, 512, n_blocks=2, stride=2)  # (B,512,14,14)

        self.up3 = _UpBlock(in_ch=512, skip_ch=256, out_ch=256)    # (B,256,28,28)
        self.up2 = _UpBlock(in_ch=256, skip_ch=128, out_ch=128)    # (B,128,56,56)
        self.heatmap_head = nn.Conv2d(128, N_LMK, kernel_size=1)   # (B,106,56,56)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head_bs = nn.Sequential(
            nn.Linear(512 + N_LMK * 2, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 52),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        f1 = self.layer1(x)            # (B,64,112,112)  — not used for skip (saves params)
        f2 = self.layer2(f1)           # (B,128,56,56)
        f3 = self.layer3(f2)           # (B,256,28,28)
        f4 = self.layer4(f3)           # (B,512,14,14)

        u3 = self.up3(f4, f3)          # (B,256,28,28)
        u2 = self.up2(u3, f2)          # (B,128,56,56)
        heatmap = self.heatmap_head(u2)  # (B,106,56,56)
        lmk = _softargmax_2d(heatmap)  # (B,106,2) ∈ [0,1]

        feats = self.pool(f4).flatten(1)   # (B,512)
        lmk_flat = lmk.reshape(lmk.size(0), -1)
        bs = torch.sigmoid(self.head_bs(torch.cat([feats, lmk_flat], dim=-1)))
        return bs, lmk

    def trainable_parameters(self):
        return list(self.parameters())
