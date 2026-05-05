"""Load and freeze PersonaLive's PoseGuider as the distillation teacher."""

import os
import sys

import torch

_MOORE = os.path.expanduser("~/w/Moore-AnimateAnyone")
if _MOORE not in sys.path:
    sys.path.insert(0, _MOORE)


def load_frozen_teacher(weights_path: str, device: str = "cuda"):
    """Build a Moore PoseGuider matching PersonaLive's config and load weights."""
    from src.models.pose_guider import PoseGuider  # Moore vendor; sys.path injected above

    pg = PoseGuider(
        conditioning_embedding_channels=320,
        block_out_channels=(16, 32, 96, 256),
    )
    raw = torch.load(weights_path, map_location="cpu")
    sd = {k.replace("conv_out_modify", "conv_out"): v for k, v in raw.items()}
    missing, unexpected = pg.load_state_dict(sd, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys when loading teacher: {unexpected}")
    pg.eval()
    for p in pg.parameters():
        p.requires_grad_(False)
    return pg.to(device)
