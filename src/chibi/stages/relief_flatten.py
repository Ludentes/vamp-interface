"""Stage 3 — flatten facial relief.

Blend the face-panel verts toward their Laplacian-smoothed surface, erasing
the 3-D micro-anatomy (cheekbone volume, nasolabial folds, brow ridge, nose-
bridge ridge) that makes a chibi-proportioned realistic face read as uncanny
(painter rule 8a). The eyeballs are excluded — they are re-primed in stage 4,
and smoothing a sphere into the lid is not wanted here.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch

from chibi.primitives import laplacian_smoothed, mask_weights

PANEL = ["face", "nose", "forehead", "eye_region", "lips"]


@dataclass
class ReliefParams:
    strength: float = 0.8     # blend weight toward the smoothed surface
    iters: int = 12           # Laplacian smoothing iterations
    falloff: float = 0.02     # mask falloff (FLAME units)


def relief_flatten(verts: torch.Tensor, faces: torch.Tensor, masks_path: str,
                   params: ReliefParams) -> torch.Tensor:
    """Return verts with the face panel blended toward its smoothed surface.
    Vert count and faces unchanged."""
    v = verts.to(torch.float64)
    sm = laplacian_smoothed(v, faces, iters=params.iters)
    w = mask_weights(v, masks_path, PANEL, falloff=params.falloff)
    blend = (params.strength * w).clamp(0.0, 1.0).unsqueeze(1)
    return v + blend * (sm - v)
