"""The shared camera set for splat render-and-bake.

A `View` is a 3DGS-convention world-to-camera matrix plus a field of view. The
same View list drives both splat_render (rasterises the Gaussians) and bake
(projects mesh vertices) — one convention, so a vertex projects to the pixel it
was rendered at. Cameras orbit the origin on a sphere; the canonical avatar is
recentred onto the origin by the bake driver before rendering.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch


@dataclass
class View:
    """w2c: (4,4) float32 world->camera matrix (3DGS/COLMAP convention,
    row-vector: p_cam_h = p_world_h @ w2c.T). fov_rad: symmetric vertical+
    horizontal FoV in radians (square image). image_size: pixels per side."""
    w2c: torch.Tensor
    fov_rad: float
    image_size: int


def _look_at_w2c(azim_deg: float, elev_deg: float, dist: float) -> torch.Tensor:
    """World->camera matrix for a camera at (azim, elev) on a sphere of radius
    `dist`, looking at the origin, with world +Y up."""
    az = math.radians(azim_deg)
    el = math.radians(elev_deg)
    # camera position in world space
    eye = torch.tensor([
        dist * math.cos(el) * math.sin(az),
        dist * math.sin(el),
        dist * math.cos(el) * math.cos(az),
    ], dtype=torch.float32)
    fwd = (-eye) / eye.norm()                       # look direction (toward origin)
    up0 = torch.tensor([0.0, 1.0, 0.0])
    right = torch.linalg.cross(fwd, up0)
    right = right / right.norm()
    up = torch.linalg.cross(right, fwd)
    # camera axes as rows of the rotation (world -> camera): x=right, y=-up, z=fwd
    R = torch.stack([right, -up, fwd], dim=0)       # (3,3)
    t = -R @ eye                                    # (3,)
    w2c = torch.eye(4, dtype=torch.float32)
    w2c[:3, :3] = R
    w2c[:3, 3] = t
    return w2c


def turntable_views(n_azim: int = 12, elevs=(-20.0, 0.0, 20.0),
                     dist: float = 2.7, fov_deg: float = 40.0,
                     image_size: int = 512) -> list[View]:
    """`n_azim` azimuths evenly around the circle, crossed with each elevation
    in `elevs`. Default 12x3 = 36 views covering a head incl. crown/under-chin."""
    fov = math.radians(fov_deg)
    views: list[View] = []
    for elev in elevs:
        for i in range(n_azim):
            azim = 360.0 * i / n_azim
            views.append(View(w2c=_look_at_w2c(azim, elev, dist),
                              fov_rad=fov, image_size=image_size))
    return views
