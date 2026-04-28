"""Adapter student: stem + frozen IResNet50 backbone.

Wraps `load_frozen_iresnet50()` with `attach_stem()` + `mark_stem_trainable()`,
and L2-normalises the output (the converted ONNX graph emits un-normalised
512-d; teacher embeddings are L2-normalised so we match teacher distribution).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import (
    DEFAULT_ONNX_PATH,
    attach_stem,
    load_frozen_iresnet50,
    mark_stem_trainable,
)
from .stems import LatentStemNative, LatentStemUpsample, PixelStem


class AdapterStudent(nn.Module):
    """Frozen IResNet50 with a swapped trainable stem; L2-normalised 512-d output."""

    def __init__(
        self,
        variant: str,
        onnx_path: Path = DEFAULT_ONNX_PATH,
    ):
        super().__init__()
        if variant not in {"pixel_a", "latent_a_up", "latent_a_native"}:
            raise ValueError(f"unknown variant: {variant!r}")
        self.variant = variant

        backbone = load_frozen_iresnet50(onnx_path)
        if variant == "pixel_a":
            stem: nn.Module = PixelStem()
            attach_stem(backbone, stem, bypass_native_prelu_bn=False)
            self.trainable = mark_stem_trainable(backbone, retrain_prelu_bn=True)
        elif variant == "latent_a_up":
            stem = LatentStemUpsample()
            attach_stem(backbone, stem, bypass_native_prelu_bn=True)
            self.trainable = mark_stem_trainable(backbone, retrain_prelu_bn=False)
        else:
            stem = LatentStemNative()
            attach_stem(backbone, stem, bypass_native_prelu_bn=True)
            self.trainable = mark_stem_trainable(backbone, retrain_prelu_bn=False)

        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        return F.normalize(z, dim=-1)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def train(self, mode: bool = True):
        """Only the stem (and Pixel-A's PRelu_1+BN_2) honour `mode`; the frozen
        deep backbone stays in eval so its BatchNorm running stats never drift."""
        self.training = mode
        # Whole backbone forced to eval (deep BN running stats locked).
        self.backbone.eval()
        # Re-enable train mode just on the trainable stem subtree.
        self.backbone.Conv_0.train(mode)
        if self.variant == "pixel_a":
            self.backbone.BatchNormalization_2.train(mode)
        return self
