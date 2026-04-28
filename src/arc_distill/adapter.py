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
    unfreeze_layer1,
)
from .stems import LatentStemNative, LatentStemUpsample, PixelStem


VARIANTS = {"pixel_a", "latent_a_up", "latent_a_native", "latent_a2_shallow"}


class AdapterStudent(nn.Module):
    """Frozen IResNet50 with a swapped trainable stem; L2-normalised 512-d output.

    Variants:
      - pixel_a:           PixelStem; only stem + native PRelu_1/BN_2 train.
      - latent_a_up:       Bilinear-up + Conv stem; only stem trains.
      - latent_a_native:   ConvTranspose stride-8 stem; only stem trains.
      - latent_a2_shallow: latent_a_native stem + layer-1 unfrozen (3 IResNet50
                           residual blocks). Returns two parameter groups so the
                           caller can apply a lower LR to the backbone slice.
    """

    def __init__(
        self,
        variant: str,
        onnx_path: Path = DEFAULT_ONNX_PATH,
    ):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant: {variant!r}")
        self.variant = variant
        self.layer1_unfrozen: list[torch.Tensor] = []

        backbone = load_frozen_iresnet50(onnx_path)
        if variant == "pixel_a":
            stem: nn.Module = PixelStem()
            attach_stem(backbone, stem, bypass_native_prelu_bn=False)
            self.trainable = mark_stem_trainable(backbone, retrain_prelu_bn=True)
        elif variant == "latent_a_up":
            stem = LatentStemUpsample()
            attach_stem(backbone, stem, bypass_native_prelu_bn=True)
            self.trainable = mark_stem_trainable(backbone, retrain_prelu_bn=False)
        elif variant == "latent_a_native":
            stem = LatentStemNative()
            attach_stem(backbone, stem, bypass_native_prelu_bn=True)
            self.trainable = mark_stem_trainable(backbone, retrain_prelu_bn=False)
        else:  # latent_a2_shallow
            stem = LatentStemNative()
            attach_stem(backbone, stem, bypass_native_prelu_bn=True)
            stem_params = mark_stem_trainable(backbone, retrain_prelu_bn=False)
            self.layer1_unfrozen = unfreeze_layer1(backbone)
            self.trainable = stem_params + self.layer1_unfrozen
            self._stem_params = stem_params

        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        return F.normalize(z, dim=-1)

    def trainable_parameters(self):
        return list(self.trainable)

    def parameter_groups(self, stem_lr: float, backbone_lr: float) -> list[dict]:
        """For latent_a2_shallow: returns [{stem at stem_lr}, {layer-1 at backbone_lr}].

        For other variants returns a single group at stem_lr (acts like a regular
        opt over trainable_parameters)."""
        if self.variant != "latent_a2_shallow":
            return [{"params": self.trainable_parameters(), "lr": stem_lr}]
        return [
            {"params": self._stem_params, "lr": stem_lr},
            {"params": self.layer1_unfrozen, "lr": backbone_lr},
        ]

    def train(self, mode: bool = True):
        """Only the stem (and Pixel-A's PRelu_1+BN_2) honour `mode`; the frozen
        deep backbone stays in eval so its BatchNorm running stats never drift.

        For latent_a2_shallow: layer-1's BatchNormalizations also stay in eval
        so their running stats stay locked at ArcFace-pretraining values; only
        their affine weight/bias trains. This mirrors standard partial fine-tune
        practice and prevents BN drift on a small dataset."""
        self.training = mode
        self.backbone.eval()
        self.backbone.Conv_0.train(mode)
        if self.variant == "pixel_a":
            self.backbone.BatchNormalization_2.train(mode)
        # latent_a2_shallow: layer-1 BN stays frozen-in-eval (no .train(mode))
        return self
