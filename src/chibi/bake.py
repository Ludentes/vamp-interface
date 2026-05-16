"""Bake splat-render appearance onto a mesh's vertices.

Single 3DGS camera convention — projects each vertex with the View's w2c +
perspective matrix, samples the splat-render image, occlusion-tests against the
splat-render depth, and writes a normal-weighted average into ChibiMesh.rgb.
Pure function over ChibiMesh + tensors; no LAM, no pytorch3d.
"""
from __future__ import annotations
from collections.abc import Sequence
import math
import torch

from chibi.mesh import ChibiMesh
from chibi.camera_rig import View

_DEPTH_TOL = 0.05      # vertex counted occluded if this far behind the surface


def _vertex_normals(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Area-weighted vertex normals, (V,3) float32."""
    v = verts.to(torch.float32)
    f = faces.to(torch.int64)
    fn = torch.linalg.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    vn = torch.zeros_like(v)
    for k in range(3):
        vn.index_add_(0, f[:, k], fn)
    return vn / vn.norm(dim=1, keepdim=True).clamp_min(1e-9)


def bake_vertex_colors(mesh: ChibiMesh, images: torch.Tensor,
                       depth: torch.Tensor, views: Sequence[View], *,
                       fallback_rgb: tuple[float, float, float] = (0.5, 0.5, 0.5)
                       ) -> ChibiMesh:
    """images: (T,H,W,3) uint8. depth: (T,H,W) float32 camera-space Z.
    views: T Views. Returns a new ChibiMesh with baked per-vertex rgb."""
    assert len(images) == len(views) == len(depth), \
        "images, depth and views must have equal length"
    V = mesh.verts.shape[0]
    verts = mesh.verts.to(torch.float32)
    normals = _vertex_normals(mesh.verts, mesh.faces)
    accum = torch.zeros(V, 3)
    weight = torch.zeros(V)

    for img, dep, view in zip(images, depth, views):
        size = view.image_size
        w2c = view.w2c                                   # (4,4)
        vh = torch.cat([verts, torch.ones(V, 1)], dim=1)  # (V,4)
        cam = vh @ w2c.T                                  # (V,4) camera space
        cam_z = cam[:, 2]
        # perspective project: x_ndc = x/(z*tan), in [-1,1]
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
        # occlusion: splat surface at this pixel vs the vertex depth
        surf = dep[pyc, pxc]
        visible = ok & (cam_z <= surf + _DEPTH_TOL)
        # facing weight: w2c's 3x3 rows are the camera axes in world space;
        # row 2 is the forward (look) direction.
        view_dir = w2c[2, :3]                            # world-space fwd
        facing = (-(normals @ view_dir)).clamp_min(0.0)  # normal toward camera
        w = torch.where(visible, facing, torch.zeros_like(facing))
        sampled = img[pyc, pxc].to(torch.float32) / 255.0  # (V,3)
        accum += sampled * w.unsqueeze(1)
        weight += w

    baked = torch.empty(V, 3)
    seen = weight > 1e-6
    baked[seen] = accum[seen] / weight[seen].unsqueeze(1)
    if (~seen).any():
        # fill zero-visibility verts with the nearest seen vertex's colour
        fb = torch.tensor(fallback_rgb, dtype=torch.float32)
        if seen.any():
            seen_idx = seen.nonzero(as_tuple=True)[0]
            for i in (~seen).nonzero(as_tuple=True)[0]:
                d = (verts[seen_idx] - verts[i]).norm(dim=1)
                baked[i] = baked[seen_idx[int(d.argmin())]]
        else:
            baked[:] = fb
    return ChibiMesh(verts=mesh.verts, faces=mesh.faces,
                     rgb=baked.clamp(0.0, 1.0).to(torch.float32))
