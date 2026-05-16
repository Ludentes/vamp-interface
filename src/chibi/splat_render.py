"""Render a canonical LAM Gaussian-splat .ply from a set of Views.

Uses LAM's own diff_gaussian_rasterization + Camera helpers so colour, opacity
and scale conventions match LAM's renders exactly. LAM runs gs_use_rgb, so the
.ply's f_dc_* is direct RGB and is passed as colors_precomp (no SH).
"""
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Sequence
import math
import numpy as np
import torch

from chibi.camera_rig import View


@dataclass
class Splats:
    """Decoded 3DGS state. xyz (N,3), rgb (N,3) in [0,1], opacity (N,1),
    scaling (N,3), rotation (N,4) quaternion — all float32, all activated
    (sigmoid/exp already applied), ready for the rasteriser."""
    xyz: torch.Tensor
    rgb: torch.Tensor
    opacity: torch.Tensor
    scaling: torch.Tensor
    rotation: torch.Tensor


def load_splat_ply(ply_path: str) -> Splats:
    """Parse a LAM 3DGS .ply and decode the stored encodings: opacity is
    inverse-sigmoid'd, scale is log'd, rotation is a raw quaternion, f_dc is
    direct RGB."""
    from plyfile import PlyData
    el = PlyData.read(ply_path)["vertex"]
    names = set(el.data.dtype.names)
    assert {"scale_0", "rot_0", "opacity"} <= names, (
        f"{ply_path} is not a renderable 3DGS ply (missing scale/rot/opacity) "
        f"— did you pass _gs_offset.ply by mistake?")

    def col(*cs):
        return np.stack([el[c] for c in cs], axis=1).astype(np.float32)

    xyz = torch.from_numpy(col("x", "y", "z"))
    rgb = torch.from_numpy(col("f_dc_0", "f_dc_1", "f_dc_2")).clamp(0.0, 1.0)
    opacity = torch.sigmoid(torch.from_numpy(col("opacity")))          # (N,1)
    scaling = torch.exp(torch.from_numpy(col("scale_0", "scale_1", "scale_2")))
    rotation = torch.from_numpy(col("rot_0", "rot_1", "rot_2", "rot_3"))
    return Splats(xyz=xyz, rgb=rgb, opacity=opacity,
                  scaling=scaling, rotation=rotation)


def render_splats(splats: Splats, views: Sequence[View], *,
                  device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """Rasterise `splats` from each View. Returns (rgb (T,H,W,3) uint8 CPU,
    depth (T,H,W) float32 CPU) — depth is camera-space Z from the rasteriser."""
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings, GaussianRasterizer)
    from lam.models.rendering.gs_renderer import Camera

    dev = torch.device(device)
    xyz = splats.xyz.to(dev)
    rgb = splats.rgb.to(dev)
    opacity = splats.opacity.to(dev)
    scaling = splats.scaling.to(dev)
    rotation = splats.rotation.to(dev)
    means2d = torch.zeros_like(xyz, requires_grad=False)

    rgb_out, depth_out = [], []
    for v in views:
        size = v.image_size
        w2c = v.w2c.to(dev)
        # FoVx == FoVy (square image, symmetric); intrinsic only feeds Camera's
        # unused-here fields, so a focal consistent with fov is enough.
        focal = 0.5 * size / math.tan(v.fov_rad * 0.5)
        intrinsic = torch.tensor([[focal, 0, size / 2],
                                  [0, focal, size / 2],
                                  [0, 0, 1]], dtype=torch.float32, device=dev)
        cam = Camera(w2c=w2c, intrinsic=intrinsic, FoVx=v.fov_rad,
                     FoVy=v.fov_rad, height=size, width=size)
        raster = GaussianRasterizationSettings(
            image_height=size, image_width=size,
            tanfovx=math.tan(v.fov_rad * 0.5),
            tanfovy=math.tan(v.fov_rad * 0.5),
            bg=torch.zeros(3, device=dev), scale_modifier=1.0,
            viewmatrix=cam.world_view_transform,
            projmatrix=cam.full_proj_transform.float(),
            sh_degree=0, campos=cam.camera_center,
            prefiltered=False, debug=False)
        rasterizer = GaussianRasterizer(raster_settings=raster)
        with torch.autocast(device_type=dev.type, dtype=torch.float32):
            image, _radii, depth, _alpha = rasterizer(
                means3D=xyz, means2D=means2d, shs=None,
                colors_precomp=rgb, opacities=opacity,
                scales=scaling, rotations=rotation, cov3D_precomp=None)
        img = image.permute(1, 2, 0).clamp(0.0, 1.0)         # (H,W,3)
        rgb_out.append((img * 255.0).round().to(torch.uint8).cpu())
        depth_out.append(depth.squeeze(0).float().cpu())     # (H,W)
    return torch.stack(rgb_out), torch.stack(depth_out)
