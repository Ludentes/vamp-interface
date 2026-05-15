#!/usr/bin/env python3
"""Synthesize horn protrusions on a LAM textured_mesh.obj.

Picks the top quartile of vertices by y (head crown), and within that band
finds two clusters in x (left/right). Displaces each cluster outward+upward
with a Gaussian falloff to produce two tapered horns. Preserves vertex order
and per-vertex colour from the source OBJ.

Usage:
    uv run scripts/lam_edit_obj_horns.py --input <anchor>_textured_mesh.obj \
        --output exp_output/lam_edit_spike/asian_m_horns.obj \
        [--height 0.06] [--out 0.04] [--sigma 0.025]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np


def load_obj(path: Path):
    lines = path.read_text().splitlines()
    verts, vert_idx = [], []
    for i, L in enumerate(lines):
        if L.startswith("v "):
            ps = L.split()
            if len(ps) == 7:
                verts.append([float(x) for x in ps[1:]])
                vert_idx.append(i)
    return np.asarray(verts, dtype=np.float32), vert_idx, lines


def save_obj(path: Path, verts: np.ndarray, lines: list[str], vert_idx: list[int]):
    out = list(lines)
    for i, vi in enumerate(vert_idx):
        x, y, z, r, g, b = verts[i]
        out[vi] = f"v {x:.8f} {y:.8f} {z:.8f} {r:.6f} {g:.6f} {b:.6f}"
    path.write_text("\n".join(out) + "\n")


def make_horns(verts: np.ndarray, height: float, out: float, sigma: float):
    """Add two horns. verts is [N,6] (xyz,rgb). Returns edited copy."""
    pos = verts[:, :3].copy()
    rgb = verts[:, 3:].copy()

    y = pos[:, 1]
    y_thresh = np.quantile(y, 0.92)  # top 8% — head crown
    crown_mask = y > y_thresh
    crown = pos[crown_mask]
    if len(crown) < 20:
        raise RuntimeError(f"only {len(crown)} crown verts, need >=20")
    # Split crown into left/right by median x
    x_med = np.median(crown[:, 0])
    # Horn anchors: centroid of each side cluster on the upper crown band
    left_anchor = crown[crown[:, 0] < x_med].mean(axis=0)
    right_anchor = crown[crown[:, 0] >= x_med].mean(axis=0)
    # Place horn tips: anchors displaced outward in x and upward in y
    for anchor, sign in [(left_anchor, -1.0), (right_anchor, +1.0)]:
        # falloff: Gaussian in 3D around anchor
        d2 = ((pos - anchor) ** 2).sum(axis=1)
        w = np.exp(-d2 / (2 * sigma * sigma))
        # Only act on verts above the y_thresh band to avoid pulling face
        w = w * crown_mask.astype(np.float32)
        # Direction: outward (sign*x) + up (+y)
        dx = sign * out
        dy = height
        dz = 0.0
        pos[:, 0] += w * dx
        pos[:, 1] += w * dy
        pos[:, 2] += w * dz
        # Tint the displaced region toward dark brown (horn colour)
        horn_rgb = np.array([0.25, 0.15, 0.10], dtype=np.float32)
        tint = w[:, None] * 0.9
        rgb[:] = (1.0 - tint) * rgb + tint * horn_rgb

    n_moved = ((pos != verts[:, :3]).any(axis=1)).sum()
    print(f"horns: moved {n_moved} verts; left tip ≈ {left_anchor[:3] + np.array([-out, height, 0])}, "
          f"right tip ≈ {right_anchor[:3] + np.array([out, height, 0])}")
    return np.concatenate([pos, rgb], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--height", type=float, default=0.06)
    ap.add_argument("--out", dest="out_dist", type=float, default=0.04)
    ap.add_argument("--sigma", type=float, default=0.025)
    args = ap.parse_args()
    inp, outp = Path(args.input), Path(args.output)
    verts, vert_idx, lines = load_obj(inp)
    print(f"loaded {len(verts)} verts from {inp.name}")
    assert verts.shape[1] == 6
    edited = make_horns(verts, height=args.height, out=args.out_dist, sigma=args.sigma)
    save_obj(outp, edited, lines, vert_idx)
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
