"""Frozen IResNet50 backbone (buffalo_l ArcFace) loaded via onnx2torch.

The buffalo_l recognition model w600k_r50.onnx is IResNet50 (not torchvision
ResNet-50): stem is `Conv(3,64,3x3,s=1) + PReLU + BN` producing (64,112,112);
downsampling happens inside layer-1's first residual block.

`load_frozen_iresnet50` returns the full converted model with all parameters
frozen. The cut point for stem-swap is `Conv_0`: the converted FX GraphModule
calls `self.Conv_0(input_1)` as its very first op. Replacing that submodule
substitutes the stem; the rest of the network sees the new stem's output.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

DEFAULT_ONNX_PATH = Path.home() / ".insightface" / "models" / "buffalo_l" / "w600k_r50.onnx"


def load_frozen_iresnet50(onnx_path: Path = DEFAULT_ONNX_PATH) -> nn.Module:
    """Convert w600k_r50.onnx to PyTorch via onnx2torch and freeze all parameters."""
    from onnx2torch import convert
    model = convert(str(onnx_path)).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for b in model.buffers():
        b.requires_grad_(False)
    return model


class _StemHook(nn.Module):
    """Replaces `Conv_0` in the converted FX graph: routes the model's input
    through a configurable stem instead of the original 3x3 conv."""

    def __init__(self, stem: nn.Module):
        super().__init__()
        self.stem = stem

    def forward(self, x):
        return self.stem(x)


class _IgnoreSecondArg(nn.Module):
    """Replaces `PRelu_1` for Latent-A: the latent stem already includes
    its own activation, so we pass the stem output through unchanged.
    Signature must match `OnnxPReLU.forward(x, slope)`."""

    def forward(self, x, slope):
        return x


def attach_stem(
    backbone: nn.Module,
    stem: nn.Module,
    *,
    bypass_native_prelu_bn: bool = False,
) -> nn.Module:
    """Mutate `backbone` in-place to use `stem` as the input pathway.

    For Pixel-A: bypass_native_prelu_bn=False. Native PRelu_1 + BN_2 stay
      (and are unfrozen by mark_stem_trainable so they retrain alongside the
      replacement Conv_0).
    For Latent-A: bypass_native_prelu_bn=True. PRelu_1 → ignore-slope passthrough,
      BN_2 → Identity. The latent stem must include its own activation/normalization.
    """
    backbone.Conv_0 = _StemHook(stem)
    if bypass_native_prelu_bn:
        backbone.PRelu_1 = _IgnoreSecondArg()
        backbone.BatchNormalization_2 = nn.Identity()
    return backbone


def mark_stem_trainable(backbone: nn.Module, *, retrain_prelu_bn: bool) -> list[nn.Parameter]:
    """Freeze backbone params, unfreeze stem (always) and Pixel-A's PRelu_1+BN_2 if asked.

    Also resets the unfrozen modules to fresh init so the gate measures
    "can a fresh stem recover R50's input distribution" rather than
    "is the original stem already optimal".
    """
    for p in backbone.parameters():
        p.requires_grad_(False)

    trainable: list[nn.Parameter] = []
    for p in backbone.Conv_0.parameters():
        p.requires_grad_(True)
        trainable.append(p)

    if retrain_prelu_bn:
        # Reset PRelu_1's slope (held in `initializers.onnx_initializer_0`).
        slope = backbone.initializers.onnx_initializer_0
        with torch.no_grad():
            slope.fill_(0.25)
        slope.requires_grad_(True)
        trainable.append(slope)

        bn = backbone.BatchNormalization_2
        with torch.no_grad():
            nn.init.ones_(bn.weight)
            nn.init.zeros_(bn.bias)
            bn.running_mean.zero_()
            bn.running_var.fill_(1.0)
            bn.num_batches_tracked.zero_()
        bn.train()  # so running stats update during training
        for p in bn.parameters():
            p.requires_grad_(True)
            trainable.append(p)

    return trainable
