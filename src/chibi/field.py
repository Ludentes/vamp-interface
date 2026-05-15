"""Differentiable chibi deformation field.

ChibiField is a tiny nn.Module (~13 params) that bends a FLAME head into
chibi proportions. It is the learnable replacement for the hand-tuned
T/SY/SR knots in scripts/chibi_make_assets.py.

Normalized head axis u: u = (y_crown - y) / (y_crown - y_chin), clamped
to [0,1]. u=0 crown, u=1 chin. The field is the composition of:
  1. a monotone vertical remap  u -> u_chibi  (6 knots, 5 free increments)
  2. a per-height radial xz scale  r(u)        (6 log-scale knots)
  3. per-feature region similarity transforms  (eye/nose/mouth)
"""
from __future__ import annotations
import torch
import torch.nn as nn

# Realistic u-positions of the 6 remap knots: crown, brow, eye, nose, mouth, chin.
REALISTIC_KNOTS = (0.0, 0.33, 0.42, 0.67, 0.80, 1.0)


def _interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """Differentiable 1-D linear interpolation (torch has no torch.interp)."""
    idx = torch.searchsorted(xp, x.clamp(xp[0], xp[-1]), right=True)
    idx = idx.clamp(1, len(xp) - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    f0, f1 = fp[idx - 1], fp[idx]
    w = (x - x0) / (x1 - x0).clamp_min(1e-8)
    return f0 + w * (f1 - f0)


class ChibiField(nn.Module):
    def __init__(self, y_crown: float, y_chin: float, z_center: float):
        super().__init__()
        self.register_buffer("y_crown", torch.tensor(float(y_crown)))
        self.register_buffer("y_chin", torch.tensor(float(y_chin)))
        self.register_buffer("z_center", torch.tensor(float(z_center)))
        self.register_buffer("realistic_knots", torch.tensor(REALISTIC_KNOTS))
        # 5 increments between the 6 knots; softplus -> positive; normalized to 1.
        self.remap_incr = nn.Parameter(torch.zeros(5))
        # per-knot radial log-scale; exp(0)=1 -> identity.
        self.radial_log = nn.Parameter(torch.zeros(6))
        # per-feature region log-scales (identity at 0).
        self.s_eye_log = nn.Parameter(torch.zeros(1))      # uniform eye enlarge
        self.s_nose_xy_log = nn.Parameter(torch.zeros(1))  # nose width
        self.s_nose_z_log = nn.Parameter(torch.zeros(1))   # nose depth/bridge
        self.s_mouth_y_log = nn.Parameter(torch.zeros(1))  # mouth vertical

    def chibi_knots(self) -> torch.Tensor:
        """The 6 chibi u-positions. Monotone by construction; identity (==
        REALISTIC_KNOTS) at remap_incr=0 because exp(0)=1.

        Each interior segment is the realistic spacing scaled by exp(incr),
        then the 6 knots are the normalized cumulative sum. exp keeps every
        segment positive (monotone) and unit at rest (identity)."""
        rk = self.realistic_knots
        spacing = rk[1:] - rk[:-1]                 # (5,) sums to 1
        seg = spacing * torch.exp(self.remap_incr)
        cum = torch.cat([torch.zeros(1), torch.cumsum(seg, 0)])
        return cum / cum[-1]

    def remap(self, u: torch.Tensor) -> torch.Tensor:
        """Map realistic u -> chibi u via the piecewise-linear monotone remap."""
        return _interp(u, self.realistic_knots, self.chibi_knots())

    def u_of(self, verts: torch.Tensor) -> torch.Tensor:
        span = (self.y_crown - self.y_chin).clamp_min(1e-6)
        return ((self.y_crown - verts[:, 1]) / span).clamp(0.0, 1.0)

    def _region_scale_vecs(self) -> dict:
        """Per-region diagonal scale vector (sx,sy,sz)."""
        e = torch.exp(self.s_eye_log)
        nxy, nz = torch.exp(self.s_nose_xy_log), torch.exp(self.s_nose_z_log)
        my = torch.exp(self.s_mouth_y_log)
        return {
            "eye": torch.cat([e, e, e]),
            "nose": torch.cat([nxy, nxy, nz]),
            "mouth": torch.cat([torch.ones(1), my, torch.ones(1)]),
        }

    def forward(self, verts: torch.Tensor, region_weights: dict | None = None,
                return_centroids: bool = False):
        """Apply remap + radial scale, then blended region transforms.

        region_weights: {region_name: (N,) tensor in [0,1]} smooth falloff
        masks. region_name in {"eye","nose","mouth"}. None -> global field only.
        return_centroids: also return {name: (3,) centroid} actually used, so
        local_jacobian can replay the region transforms with fixed centroids.
        """
        u = self.u_of(verts)
        u_chibi = self.remap(u)
        new_y = self.y_crown - u_chibi * (self.y_crown - self.y_chin)
        r = torch.exp(_interp(u, self.realistic_knots, self.radial_log))
        new_x = verts[:, 0] * r
        new_z = self.z_center + (verts[:, 2] - self.z_center) * r
        out = torch.stack([new_x, new_y, new_z], dim=1)
        centroids = {}
        if region_weights:
            scales = self._region_scale_vecs()
            for name, w in region_weights.items():
                wsum = w.sum().clamp_min(1e-6)
                centroid = (out * w[:, None]).sum(0) / wsum
                centroids[name] = centroid
                delta = (out - centroid) * (scales[name] - 1.0)
                out = out + w[:, None] * delta
        return (out, centroids) if return_centroids else out

    def local_jacobian(self, verts: torch.Tensor,
                       region_weights: dict | None = None) -> torch.Tensor:
        """Per-vertex 3x3 Jacobian of the COMPLETE field — remap + radial +
        the per-feature region transforms — via autograd. This is the exact
        pushforward that rescales the ARKit basis: bs_chibi[v] = J[v] @ bs[v].

        Including the region transforms is load-bearing: the eye region
        enlarges the eye ~2x, so the blink/squint basis rows must be pushed
        through the eye Jacobian or the lid cannot close the enlarged eye
        (the iris-through-lid leak, rebuilt by geometry). Region centroids are
        held fixed (detached) so each vertex's J is its local linear map; the
        residual centroid coupling is O(1/M) per region and negligible.
        Returns (N,3,3)."""
        if not region_weights:
            def single(v):                   # v: (3,)
                return self.forward(v[None, :])[0]
            return torch.vmap(torch.func.jacrev(single))(verts)
        _, centroids = self.forward(verts, region_weights, return_centroids=True)
        centroids = {k: c.detach() for k, c in centroids.items()}
        scales = {k: s.detach() for k, s in self._region_scale_vecs().items()}
        names = list(region_weights.keys())
        w_stack = torch.stack([region_weights[n] for n in names], dim=1)  # (N,R)

        def single(v, wi):                   # v:(3,) wi:(R,)
            g = self.forward(v[None, :])[0]
            for j, name in enumerate(names):
                g = g + wi[j] * (g - centroids[name]) * (scales[name] - 1.0)
            return g
        return torch.vmap(torch.func.jacrev(single))(verts, w_stack)
