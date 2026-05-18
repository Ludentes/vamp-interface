"""Geometry primitives for the chibi staged pipeline — no chibi logic here.

A `Box` is an axis-aligned bounding box. `project_superellipsoid` maps verts
radially (from the box centre) onto a superellipsoid inscribed in the box —
exponent 2 is an ellipsoid, large exponent approaches the box. That single
operator is the chibi "block" target. `laplacian_smoothed` is the relief-free
reference surface. `mask_weights` turns FLAME mask regions into smooth
per-vertex blend weights.
"""
from __future__ import annotations
from dataclasses import dataclass
import pickle
import numpy as np
import torch


@dataclass
class Box:
    """Axis-aligned box. center (3,), half (3,) — both float64."""
    center: torch.Tensor
    half: torch.Tensor


def fit_box(points: torch.Tensor, pct: float = 1.0) -> Box:
    """Axis-aligned box from a point cloud. `pct` is the percentile trim per
    axis (1.0 = use the 1st/99th percentile, robust to a few outlier verts;
    0.0 = exact min/max)."""
    p = points.to(torch.float64)
    lo = torch.quantile(p, pct / 100.0, dim=0)
    hi = torch.quantile(p, 1.0 - pct / 100.0, dim=0)
    return Box(center=(lo + hi) * 0.5, half=(hi - lo).clamp_min(1e-6) * 0.5)


def project_superellipsoid(verts: torch.Tensor, box: Box,
                           exponent: float) -> torch.Tensor:
    """Project each vert radially (from box.center) onto the superellipsoid
    inscribed in `box`. exponent=2 -> ellipsoid; large -> box. Returns (V,3)."""
    d = verts.to(torch.float64) - box.center
    # superellipsoid implicit: sum((|d_i|/h_i)^n) = 1 on the surface.
    norm = ((d.abs() / box.half).clamp_min(1e-12) ** exponent).sum(1)
    t = norm.clamp_min(1e-12) ** (-1.0 / exponent)        # scale onto surface
    return box.center + d * t.unsqueeze(1)


def _adjacency(faces: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (src, dst) directed-edge indices for a uniform-Laplacian: every
    triangle edge both ways; double-counting cancels in a per-vertex mean."""
    e = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], 0)
    e = torch.cat([e, e.flip(1)], 0)
    return e[:, 0], e[:, 1]


def laplacian_smoothed(verts: torch.Tensor, faces: torch.Tensor,
                       iters: int = 10, lam: float = 0.5) -> torch.Tensor:
    """Uniform-Laplacian smoothing: v += lam * (mean(1-ring nbrs) - v),
    `iters` times. Returns a new tensor; does not mutate `verts`."""
    v = verts.to(torch.float64).clone()
    src, dst = _adjacency(faces.to(torch.int64), v.shape[0])
    n = v.shape[0]
    for _ in range(iters):
        nbr_sum = torch.zeros_like(v).index_add_(0, src, v[dst])
        deg = torch.zeros(n, dtype=torch.float64).index_add_(
            0, src, torch.ones(src.shape[0], dtype=torch.float64))
        nbr_mean = nbr_sum / deg.clamp_min(1.0).unsqueeze(1)
        v = v + lam * (nbr_mean - v)
    return v


def mask_weights(verts: torch.Tensor, masks_path: str, names: list[str],
                 falloff: float = 0.015) -> torch.Tensor:
    """Smooth per-vertex weight in [0,1]: 1.0 on any vertex in any of the
    named FLAME mask regions, decaying as exp(-(d/falloff)^2) with Euclidean
    distance to the nearest masked vertex. falloff is in FLAME mesh units."""
    with open(masks_path, "rb") as fh:
        masks = pickle.load(fh, encoding="latin1")
    idx = np.unique(np.concatenate([np.asarray(masks[n]) for n in names]))
    v = verts.detach().to(torch.float64)
    core = v[torch.as_tensor(idx, dtype=torch.long)]
    d = torch.cdist(v, core).min(dim=1).values
    return torch.exp(-(d / falloff) ** 2)
