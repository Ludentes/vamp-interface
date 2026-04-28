"""ArcFace-pixel student: torchvision ResNet-18 with 512-d L2-normalised head."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


class ArcStudentResNet18(nn.Module):
    """ResNet-18 backbone, fc → 512, output L2-normalised on the unit sphere."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = tvm.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = tvm.resnet18(weights=weights)
        in_f = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_f, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        return F.normalize(z, dim=-1)


def cosine_distance_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - cos(pred, target). pred is L2-normalised by the model; we re-normalise
    target defensively (insightface buffalo_l outputs are normed but be safe)."""
    target = F.normalize(target, dim=-1)
    return (1.0 - (pred * target).sum(dim=-1)).mean()
