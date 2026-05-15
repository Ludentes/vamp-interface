#!/usr/bin/env python3
"""Procedurally synthesize a pair of anisotropic horn-shaped 3DGS splats.

Each horn is a chain of `n` splats along a curved path. Each splat is
elongated along the local tangent of the path (scale_long >> scale_radial)
with a quaternion that aligns its principal axis with that tangent. The
chain tapers: scale_radial decreases linearly base → tip.

Emits an Inria-format .ply (DC SH only, 17 properties) loadable by LAM's
aux-splat injection hook (LAM_AUX_SPLATS_PLY). Coordinates in FLAME
canonical head frame.

Usage:
    uv run scripts/make_horn_splats.py --output exp_output/lam_edit_spike/horns_aniso.ply
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np


C0 = 0.28209479177387814  # SH band-0 coefficient


def quat_from_axis_to_dir(target_dir: np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z) that rotates (+x) to target_dir (unit vec)."""
    src = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    d = target_dir / (np.linalg.norm(target_dir) + 1e-9)
    cos_t = float(np.dot(src, d))
    if cos_t > 0.99999:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if cos_t < -0.99999:
        # 180°; pick any perpendicular axis
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    axis = np.cross(src, d)
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    theta = np.arccos(cos_t)
    half = theta / 2.0
    s = np.sin(half)
    return np.array([np.cos(half), axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float32)


def make_horn(base_xyz, axis, length, n, base_radius, tip_radius,
              curl, color_rgb, aniso=8.0):
    """One curved, anisotropic horn. Splat principal axis aligned to local
    tangent; long-axis length ≈ segment length × aniso for smooth coverage."""
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    side = np.array([0.0, 0.0, 1.0], dtype=np.float32) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    side = side - axis * np.dot(axis, side)
    side = side / (np.linalg.norm(side) + 1e-9)

    ts = np.linspace(0.0, 1.0, n, dtype=np.float32)
    pts = base_xyz[None] + ts[:, None] * (length * axis[None]) + (ts[:, None] ** 2) * (curl * length * side[None])

    # Local tangent = finite difference along path
    tangents = np.zeros_like(pts)
    tangents[1:-1] = pts[2:] - pts[:-2]
    tangents[0] = pts[1] - pts[0]
    tangents[-1] = pts[-1] - pts[-2]
    tangents = tangents / (np.linalg.norm(tangents, axis=-1, keepdims=True) + 1e-9)

    seg_len = length / max(n - 1, 1)
    radii = base_radius * (1 - ts) + tip_radius * ts
    # Anisotropic: scale_0 long, scale_1/2 small. Use linear (post-activation)
    # values directly — the LAM hook applies exp(log_scale).clip(0,0.2), so
    # store these as log so the activation gives back the linear value.
    long_scale = np.full(n, seg_len * aniso * 0.5, dtype=np.float32)  # half-length per splat
    log_long = np.log(np.maximum(long_scale, 1e-4))
    log_radial = np.log(np.maximum(radii, 1e-4))
    scale = np.stack([log_long, log_radial, log_radial], axis=-1)

    rot = np.stack([quat_from_axis_to_dir(t) for t in tangents], axis=0)

    opacity = np.full((n, 1), 6.0, dtype=np.float32)  # logit; sigmoid(6)≈0.998
    rgb = np.asarray(color_rgb, dtype=np.float32)
    # Optional tip darkening: multiply rgb by (1 - 0.3*ts) for a slight gradient
    rgb_per = rgb[None] * (1.0 - 0.25 * ts[:, None])
    f_dc = (rgb_per - 0.5) / C0
    return dict(xyz=pts.astype(np.float32),
                scale=scale.astype(np.float32),
                rot=rot.astype(np.float32),
                opacity=opacity, f_dc=f_dc.astype(np.float32))


def write_inria_ply(path: Path, fields: dict):
    xyz = fields["xyz"]
    scale = fields["scale"]
    rot = fields["rot"]
    opacity = fields["opacity"]
    f_dc = fields["f_dc"]
    n = xyz.shape[0]
    normals = np.zeros_like(xyz)
    props = ["x", "y", "z", "nx", "ny", "nz",
             "f_dc_0", "f_dc_1", "f_dc_2",
             "opacity",
             "scale_0", "scale_1", "scale_2",
             "rot_0", "rot_1", "rot_2", "rot_3"]
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    header += [f"property float {p}" for p in props]
    header.append("end_header")
    cols = [xyz[:, 0], xyz[:, 1], xyz[:, 2],
            normals[:, 0], normals[:, 1], normals[:, 2],
            f_dc[:, 0], f_dc[:, 1], f_dc[:, 2],
            opacity[:, 0],
            scale[:, 0], scale[:, 1], scale[:, 2],
            rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3]]
    arr = np.stack(cols, axis=-1).astype(np.float32)
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(arr.tobytes(order="C"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--length", type=float, default=0.10)
    ap.add_argument("--base_radius", type=float, default=0.014)
    ap.add_argument("--tip_radius", type=float, default=0.0008)
    ap.add_argument("--n", type=int, default=80, help="splats per horn")
    ap.add_argument("--curl", type=float, default=0.28, help="+=forward, -=back")
    ap.add_argument("--aniso", type=float, default=6.0, help="long/radial ratio")
    ap.add_argument("--color", nargs=3, type=float, default=[0.22, 0.13, 0.08])
    args = ap.parse_args()

    base_y, base_x, base_z = 0.14, 0.055, -0.035
    axis = np.array([0.0, 1.0, -0.25], dtype=np.float32)
    left = make_horn(np.array([-base_x, base_y, base_z], dtype=np.float32),
                     axis, args.length, args.n, args.base_radius,
                     args.tip_radius, args.curl, args.color, aniso=args.aniso)
    right = make_horn(np.array([+base_x, base_y, base_z], dtype=np.float32),
                      axis, args.length, args.n, args.base_radius,
                      args.tip_radius, args.curl, args.color, aniso=args.aniso)
    out = {k: np.concatenate([left[k], right[k]], axis=0) for k in left}
    write_inria_ply(Path(args.output), out)
    print(f"wrote {args.output}: {out['xyz'].shape[0]} splats (aniso={args.aniso}, curl={args.curl})")


if __name__ == "__main__":
    main()
