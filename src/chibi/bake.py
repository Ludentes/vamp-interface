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


def _erode_mask(fg: torch.Tensor, px: int) -> torch.Tensor:
    """Shrink a (H,W) bool foreground mask inward by `px` pixels (min-pool)."""
    import torch.nn.functional as F
    m = fg.to(torch.float32)[None, None]
    for _ in range(px):
        m = -F.max_pool2d(-m, kernel_size=3, stride=1, padding=1)
    return m[0, 0] > 0.5


def bake_points(points: torch.Tensor, normals: torch.Tensor,
                images: torch.Tensor, depth: torch.Tensor,
                views: Sequence[View], *,
                fallback_rgb: tuple[float, float, float] = (0.5, 0.5, 0.5),
                facing_power: float = 1.0, mode: str = "blend",
                min_facing: float = 0.0, erode_px: int = 0
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """points: (N,3). normals: (N,3) (need not be unit — used only for sign).
    images: (T,H,W,3) uint8. depth: (T,H,W) float32 camera-space Z. views: T.
    The splat image is sampled bilinearly (nearest sampling quantises into
    visible blocks once N exceeds the image pixel count). `facing_power` raises
    the normal-facing weight to a power: >1 concentrates each texel on its most
    head-on views.

    `mode` decides cross-view combination:
      - "blend": normal-weighted average of every visible view. Smooth, but
        averages sub-pixel-misaligned views and so cancels high-frequency skin
        detail into a mottled pattern.
      - "best": winner-take-all — each point takes the single view where it is
        most head-on. Keeps that view's full crispness; the cost is mild seams
        where the winning view changes.

    `min_facing` discards low-confidence samples: a point whose most head-on
    visible view is still more oblique than this cosine is marked unseen (left
    for the caller to fill) instead of trusting a grazing-angle sample. 0.0
    keeps every visible sample.

    `erode_px` shrinks each view's foreground silhouette inward by N pixels
    before sampling, so points near the splat render's anti-aliased edge never
    pull background colour across the silhouette. 0 disables it.

    Returns (rgb (N,3) float32 in [0,1], seen (N,) bool). Unseen points carry
    `fallback_rgb`; the caller decides how to fill them."""
    assert len(images) == len(views) == len(depth), \
        "images, depth and views must have equal length"
    assert mode in ("blend", "best"), f"unknown mode {mode!r}"
    N = points.shape[0]
    pts = points.to(torch.float32)
    nrm = normals.to(torch.float32)
    accum = torch.zeros(N, 3)
    weight = torch.zeros(N)
    best_w = torch.zeros(N)
    best_rgb = torch.zeros(N, 3)
    max_w = torch.zeros(N)

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
        # continuous pixel coords; pixel centres at integer+0.5.
        fx = (x_ndc * 0.5 + 0.5) * size - 0.5
        fy = (y_ndc * 0.5 + 0.5) * size - 0.5
        on_screen = (fx >= -0.5) & (fx < size - 0.5) & \
                    (fy >= -0.5) & (fy < size - 0.5)
        ok = in_front & on_screen
        # occlusion uses the nearest splat-depth texel (depth must not lerp
        # across silhouette edges).
        nx = fx.round().long().clamp(0, size - 1)
        ny = fy.round().long().clamp(0, size - 1)
        surf = dep[ny, nx]
        visible = ok & (cam_z <= surf + _DEPTH_TOL)
        if erode_px > 0:
            # the splat rasteriser leaves un-hit (background) pixels at depth
            # 0, so `dep > 1e-4` is the foreground silhouette; erode it inward.
            fg = _erode_mask(dep > 1e-4, erode_px)
            visible = visible & fg[ny, nx]
        view_dir = w2c[2, :3]                              # world-space fwd
        facing = (-(nrm @ view_dir)).clamp_min(0.0)        # normal toward cam
        w = torch.where(visible, facing.pow(facing_power),
                        torch.zeros_like(facing))
        max_w = torch.maximum(max_w, w)
        # bilinear colour sample.
        x0 = fx.floor().long()
        y0 = fy.floor().long()
        wx = (fx - x0).clamp(0.0, 1.0).unsqueeze(1)
        wy = (fy - y0).clamp(0.0, 1.0).unsqueeze(1)
        x0c = x0.clamp(0, size - 1)
        x1c = (x0 + 1).clamp(0, size - 1)
        y0c = y0.clamp(0, size - 1)
        y1c = (y0 + 1).clamp(0, size - 1)
        imgf = img.to(torch.float32) / 255.0
        c00 = imgf[y0c, x0c]
        c01 = imgf[y0c, x1c]
        c10 = imgf[y1c, x0c]
        c11 = imgf[y1c, x1c]
        sampled = (c00 * (1 - wx) * (1 - wy) + c01 * wx * (1 - wy)
                   + c10 * (1 - wx) * wy + c11 * wx * wy)   # (N,3)
        if mode == "blend":
            accum += sampled * w.unsqueeze(1)
            weight += w
        else:                                              # winner-take-all
            better = w > best_w
            best_w = torch.where(better, w, best_w)
            best_rgb[better] = sampled[better]

    # a point counts as confidently seen only if its best view clears the
    # facing threshold; grazing-only points are left unseen for the caller.
    # `max_w` and the threshold are both in powered units (facing**power), so
    # the comparison is equivalent to facing >= min_facing — keep them in sync.
    conf = max_w >= max(1e-6, min_facing ** facing_power)
    rgb = torch.empty(N, 3)
    if mode == "blend":
        seen = (weight > 1e-6) & conf
        rgb[seen] = accum[seen] / weight[seen].unsqueeze(1)
    else:
        seen = (best_w > 1e-6) & conf
        rgb[seen] = best_rgb[seen]
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
