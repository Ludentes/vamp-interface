"""Stage 1 — reshape the head toward a rounded block.

Fit an axis-aligned box to the head verts (everything but the neck), then
blend each vert toward its superellipsoid projection on that box. The cranium
(scalp+forehead) blends strongly — that is the block silhouette; the face
panel blends mildly — it flattens toward the box front but keeps enough shape
for the later stages to re-prime features on. The neck is left untouched.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch

from chibi.primitives import fit_box, project_superellipsoid, mask_weights

CRANIUM = ["scalp", "forehead"]
FACE = ["face", "nose", "lips", "eye_region"]
NECK = ["neck"]


@dataclass
class HeadBlockParams:
    exponent: float = 8.0          # superellipsoid: 2 ellipsoid .. large box
    strength_cranium: float = 0.9  # blend weight on scalp+forehead
    strength_face: float = 0.3     # blend weight on the face panel


def head_block(verts: torch.Tensor, faces: torch.Tensor, masks_path: str,
               params: HeadBlockParams) -> torch.Tensor:
    """Return verts reshaped toward a superellipsoid block. Vert count and
    faces are unchanged."""
    v = verts.to(torch.float64)
    w_neck = mask_weights(v, masks_path, NECK, falloff=0.02)
    head = v[w_neck < 0.5]                       # box fit excludes the neck
    box = fit_box(head, pct=1.0)
    target = project_superellipsoid(v, box, params.exponent)

    w_cran = mask_weights(v, masks_path, CRANIUM, falloff=0.03)
    w_face = mask_weights(v, masks_path, FACE, falloff=0.03)
    # per-vertex blend strength; neck verts forced to 0.
    strength = (params.strength_cranium * w_cran
                + params.strength_face * w_face).clamp(0.0, 1.0)
    strength = strength * (1.0 - w_neck).clamp(0.0, 1.0)
    return v + strength.unsqueeze(1) * (target - v)
