"""Register a Flux portrait to the canonical mesh's frontal projection.

Detects 2D face landmarks on the portrait (insightface), projects the canonical
mesh's annotated landmark vertices through the frontal View to 2D, fits a
thin-plate-spline warp portrait->canonical, and applies it. The warped portrait
is then ready for the single-view UV bake.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F


def _tps_kernel(r2: torch.Tensor) -> torch.Tensor:
    """U(r) = r^2 log r, evaluated stably from squared distance r2."""
    return 0.5 * r2 * torch.log(r2.clamp_min(1e-12))


@dataclass
class TPS:
    ctrl: torch.Tensor    # (K,2) source control points
    w: torch.Tensor       # (K,2) non-affine weights
    a: torch.Tensor       # (3,2) affine part

    def __call__(self, pts: torch.Tensor) -> torch.Tensor:
        """Map (N,2) source points -> (N,2) target points."""
        d2 = ((pts[:, None, :] - self.ctrl[None, :, :]) ** 2).sum(-1)
        U = _tps_kernel(d2)                       # (N,K)
        ones = torch.ones(pts.shape[0], 1)
        P = torch.cat([ones, pts], dim=1)         # (N,3)
        return U @ self.w + P @ self.a


def fit_tps(src: torch.Tensor, dst: torch.Tensor,
            reg: float = 0.0) -> TPS:
    """Fit a TPS mapping src (K,2) -> dst (K,2). `reg` relaxes exact
    interpolation (0 = exact)."""
    src = src.to(torch.float64)
    dst = dst.to(torch.float64)
    K = src.shape[0]
    d2 = ((src[:, None, :] - src[None, :, :]) ** 2).sum(-1)
    Kmat = _tps_kernel(d2) + reg * torch.eye(K, dtype=torch.float64)
    P = torch.cat([torch.ones(K, 1, dtype=torch.float64), src], dim=1)  # (K,3)
    # Solve [[K, P],[P^T, 0]] [w;a] = [dst;0]
    top = torch.cat([Kmat, P], dim=1)
    bot = torch.cat([P.T, torch.zeros(3, 3, dtype=torch.float64)], dim=1)
    L = torch.cat([top, bot], dim=0)
    rhs = torch.cat([dst, torch.zeros(3, 2, dtype=torch.float64)], dim=0)
    sol = torch.linalg.solve(L, rhs)
    return TPS(ctrl=src.to(torch.float32),
               w=sol[:K].to(torch.float32),
               a=sol[K:].to(torch.float32))


def warp_image(img: torch.Tensor, tps: TPS) -> torch.Tensor:
    """Warp (H,W,3) img in [0,1] by `tps`. The TPS maps portrait->canonical;
    to resample we need canonical->portrait, so the sampling grid pushes each
    output pixel back through tps (tps is near-affine-invertible at the scales
    used; for the small non-rigid part we invert by fixed-point iteration)."""
    H, W, _ = img.shape
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                            torch.arange(W, dtype=torch.float32),
                            indexing="ij")
    grid_pts = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1)  # (HW,2)
    # invert tps by fixed-point: p_{n+1} = p_n - (tps(p_n) - target)
    p = grid_pts.clone()
    for _ in range(20):
        p = p - (tps(p) - grid_pts)
    nx = p[:, 0] / (W - 1) * 2 - 1
    ny = p[:, 1] / (H - 1) * 2 - 1
    samp = torch.stack([nx, ny], dim=1).reshape(1, H, W, 2)
    src = img.permute(2, 0, 1)[None]
    out = F.grid_sample(src, samp, mode="bilinear", align_corners=True,
                        padding_mode="border")
    return out[0].permute(1, 2, 0)
