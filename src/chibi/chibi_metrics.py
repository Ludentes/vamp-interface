"""Geometric verification metrics for the chibi pipeline stages.

Each stage's gate asserts one of these. They measure the chibi properties
directly — block-ness, flatness, button-ness, eye roundness, mesh integrity —
so a stage that hits its target metric has actually done its job, unlike the
landmark-position targets the superseded ChibiField fit was blind through.
"""
from __future__ import annotations
import torch

from chibi.primitives import Box, project_superellipsoid, laplacian_smoothed
from chibi.landmarks import landmark_positions, GROUPS


def box_residual(verts: torch.Tensor, box: Box, exponent: float) -> float:
    """RMS distance of verts to the superellipsoid surface. Falls toward 0 as
    HeadBlock blends the head onto the block."""
    proj = project_superellipsoid(verts, box, exponent)
    return float(((verts.to(torch.float64) - proj) ** 2).sum(1).mean().sqrt())


def relief_energy(verts: torch.Tensor, faces: torch.Tensor,
                  weights: torch.Tensor, iters: int = 10) -> float:
    """Weighted RMS deviation of verts from their Laplacian-smoothed surface —
    the amount of 3-D micro-relief. ReliefFlatten drives it down."""
    sm = laplacian_smoothed(verts, faces, iters=iters)
    dev = ((verts.to(torch.float64) - sm) ** 2).sum(1)         # (V,)
    w = weights.to(torch.float64)
    return float(((dev * w).sum() / w.sum().clamp_min(1e-9)).sqrt())


def bridge_height(verts: torch.Tensor) -> float:
    """z-extent of the nose-bridge landmarks (68-pt indices 27-30). A button
    nose has a near-zero bridge ridge."""
    lm = landmark_positions(verts.to(torch.float64))
    z = lm[27:31, 2]
    return float(z.max() - z.min())


def eye_aspect(verts: torch.Tensor) -> float:
    """Width/height ratio of the eye landmark group (indices 36-47). A round
    chibi eye trends toward ~1; a realistic almond is ~2-3."""
    lm = landmark_positions(verts.to(torch.float64))
    eye = lm[GROUPS["eye"]]
    w = eye[:, 0].max() - eye[:, 0].min()
    h = (eye[:, 1].max() - eye[:, 1].min()).clamp_min(1e-9)
    return float(w / h)


def foldover_count(base_verts: torch.Tensor, verts: torch.Tensor,
                   faces: torch.Tensor) -> int:
    """Number of faces whose normal flipped relative to `base_verts` — the
    triangle self-fold metric (the wasp-waist class of failure)."""
    def fn(v):
        f = v.to(torch.float64)[faces.to(torch.int64)]
        n = torch.linalg.cross(f[:, 1] - f[:, 0], f[:, 2] - f[:, 0])
        return n / n.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return int(((fn(base_verts) * fn(verts)).sum(1) < 0).sum())
