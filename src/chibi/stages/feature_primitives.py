"""Stage 4 — re-prime eye, nose, mouth as chibi primitives.

On the flattened panel, rebuild each feature toward a chibi target:
  - eye: round the lid aperture toward a circle, per-eye centroid (per-eye is
    the fix for the shared-centroid ballooning of the old field);
  - nose: collapse the bridge toward the local face plane -> a button;
  - mouth: vertical compression toward a strip.
Each is a blend-toward-target with its own strength knob.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch

from chibi.primitives import mask_weights


@dataclass
class FeatureParams:
    eye_round: float = 0.7        # 0..1 blend toward a round aperture
    eye_enlarge: float = 1.6      # in-plane aperture scale
    # eye mask falloff: a wide falloff spreads the enlargement's compression
    # ring over more verts, so a 1.6x enlarge stays fold-safe. At the tight
    # 0.012 falloff the same enlarge folds ~270 faces past the 2% gate.
    eye_falloff: float = 0.03
    bridge_collapse: float = 0.8  # 0..1 collapse of nose-bridge depth
    mouth_compress: float = 0.5   # vertical mouth compression (0..1)


def _round_one_eye(v: torch.Tensor, w: torch.Tensor, params: FeatureParams
                   ) -> torch.Tensor:
    """Round + enlarge one eye's verts (weighted by w) about their own
    centroid, in the x-y (face-tangent) plane only — z is left alone so the
    eye does not balloon in depth."""
    wsum = w.sum().clamp_min(1e-6)
    c = (v * w[:, None]).sum(0) / wsum
    d = v - c
    # current aperture half-extents
    hx = (d[:, 0].abs() * w).sum() / wsum
    hy = (d[:, 1].abs() * w).sum() / wsum
    r = (hx + hy) * 0.5                                  # round target radius
    # blend each axis' scale toward the common radius, then enlarge
    sx = (1.0 + params.eye_round * (r / hx.clamp_min(1e-6) - 1.0))
    sy = (1.0 + params.eye_round * (r / hy.clamp_min(1e-6) - 1.0))
    scale = torch.tensor([sx * params.eye_enlarge,
                          sy * params.eye_enlarge, 1.0], dtype=v.dtype)
    target = c + d * scale
    return v + w[:, None] * (target - v)


def feature_primitives(verts: torch.Tensor, faces: torch.Tensor,
                       masks_path: str, params: FeatureParams) -> torch.Tensor:
    """Return verts with eye/nose/mouth re-primed. Vert count unchanged."""
    v = verts.to(torch.float64)
    # --- eyes: per-eye, so the centroid is each eye's own ---
    for region in (["left_eye_region", "left_eyeball"],
                   ["right_eye_region", "right_eyeball"]):
        w = mask_weights(v, masks_path, region, falloff=params.eye_falloff)
        v = _round_one_eye(v, w, params)
    # --- nose: collapse the bridge toward the local panel plane (z) ---
    w_nose = mask_weights(v, masks_path, ["nose"], falloff=0.012)
    z_panel = (v[:, 2] * w_nose).sum() / w_nose.sum().clamp_min(1e-6)
    v = v.clone()
    v[:, 2] = v[:, 2] + params.bridge_collapse * w_nose * (z_panel - v[:, 2])
    # --- mouth: vertical compression toward the lip-region centroid ---
    w_mouth = mask_weights(v, masks_path, ["lips"], falloff=0.010)
    y_c = (v[:, 1] * w_mouth).sum() / w_mouth.sum().clamp_min(1e-6)
    v[:, 1] = v[:, 1] + params.mouth_compress * w_mouth * (y_c - v[:, 1])
    return v
