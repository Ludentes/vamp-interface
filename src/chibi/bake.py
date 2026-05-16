"""Bake splat-render appearance onto a set of points.

`bake_points` is the core: project each point with each View's w2c +
perspective matrix, sample the splat-render image, occlusion-test against the
splat-render depth, write a normal-weighted average. `bake_vertex_colors` wraps
it for a ChibiMesh (computes vertex normals, fills unseen verts from the nearest
seen vertex). The texture bake calls `bake_points` directly on texel points.
"""
from __future__ import annotations
from collections.abc import Sequence
import math
import torch

from chibi.mesh import ChibiMesh
from chibi.camera_rig import View

_DEPTH_TOL = 0.05      # point counted occluded if this far behind the surface


def _vertex_normals(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Area-weighted vertex normals, (V,3) float32."""
    v = verts.to(torch.float32)
    f = faces.to(torch.int64)
    fn = torch.linalg.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    vn = torch.zeros_like(v)
    for k in range(3):
        vn.index_add_(0, f[:, k], fn)
    return vn / vn.norm(dim=1, keepdim=True).clamp_min(1e-9)


def bake_points(points: torch.Tensor, normals: torch.Tensor,
                images: torch.Tensor, depth: torch.Tensor,
                views: Sequence[View], *,
                fallback_rgb: tuple[float, float, float] = (0.5, 0.5, 0.5)
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """points: (N,3). normals: (N,3) (need not be unit — used only for sign).
    images: (T,H,W,3) uint8. depth: (T,H,W) float32 camera-space Z. views: T.
    Returns (rgb (N,3) float32 in [0,1], seen (N,) bool). Unseen points carry
    `fallback_rgb`; the caller decides how to fill them."""
    assert len(images) == len(views) == len(depth), \
        "images, depth and views must have equal length"
    N = points.shape[0]
    pts = points.to(torch.float32)
    nrm = normals.to(torch.float32)
    accum = torch.zeros(N, 3)
    weight = torch.zeros(N)

    for img, dep, view in zip(images, depth, views):
        size = view.image_size
        w2c = view.w2c                                    # (4,4)
        ph = torch.cat([pts, torch.ones(N, 1)], dim=1)     # (N,4)
        cam = ph @ w2c.T                                   # (N,4) camera space
        cam_z = cam[:, 2]
        tan = math.tan(view.fov_rad * 0.5)
        in_front = cam_z > 1e-4
        x_ndc = cam[:, 0] / (cam_z.clamp_min(1e-4) * tan)
        y_ndc = cam[:, 1] / (cam_z.clamp_min(1e-4) * tan)
        px = ((x_ndc * 0.5 + 0.5) * size).long()
        py = ((y_ndc * 0.5 + 0.5) * size).long()
        on_screen = (px >= 0) & (px < size) & (py >= 0) & (py < size)
        ok = in_front & on_screen
        pxc = px.clamp(0, size - 1)
        pyc = py.clamp(0, size - 1)
        surf = dep[pyc, pxc]
        visible = ok & (cam_z <= surf + _DEPTH_TOL)
        view_dir = w2c[2, :3]                              # world-space fwd
        facing = (-(nrm @ view_dir)).clamp_min(0.0)        # normal toward cam
        w = torch.where(visible, facing, torch.zeros_like(facing))
        sampled = img[pyc, pxc].to(torch.float32) / 255.0  # (N,3)
        accum += sampled * w.unsqueeze(1)
        weight += w

    rgb = torch.empty(N, 3)
    seen = weight > 1e-6
    rgb[seen] = accum[seen] / weight[seen].unsqueeze(1)
    rgb[~seen] = torch.tensor(fallback_rgb, dtype=torch.float32)
    return rgb.clamp(0.0, 1.0).to(torch.float32), seen


def bake_vertex_colors(mesh: ChibiMesh, images: torch.Tensor,
                       depth: torch.Tensor, views: Sequence[View], *,
                       fallback_rgb: tuple[float, float, float] = (0.5, 0.5, 0.5)
                       ) -> ChibiMesh:
    """v2 per-vertex bake. Unseen verts take the nearest seen vertex's colour."""
    normals = _vertex_normals(mesh.verts, mesh.faces)
    rgb, seen = bake_points(mesh.verts, normals, images, depth, views,
                            fallback_rgb=fallback_rgb)
    if (~seen).any() and seen.any():
        verts = mesh.verts.to(torch.float32)
        seen_idx = seen.nonzero(as_tuple=True)[0]
        for i in (~seen).nonzero(as_tuple=True)[0]:
            d = (verts[seen_idx] - verts[i]).norm(dim=1)
            rgb[i] = rgb[seen_idx[int(d.argmin())]]
    return ChibiMesh(verts=mesh.verts, faces=mesh.faces,
                     rgb=rgb.to(torch.float32))
