#!/usr/bin/env python3
"""Synthesize an edit on a LAM textured_mesh.obj for spike testing.

Reads `<anchor>_textured_mesh.obj` (FLAME canonical mesh with per-vertex RGB),
applies a deterministic per-vertex colour edit, writes a new OBJ in the same
format. Used to feed the LAM_EDIT_VERTEX_COLORS_OBJ env-var hook in
modeling_lam.py.

Available edits:
  --edit red_hair       : tint vertices above y=threshold red
  --edit pale_skin      : desaturate + brighten everywhere
  --edit blue_eyes      : (no-op for now; FLAME topology not annotated for eyes)
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np


def load_obj(path: Path):
    """Returns (vertex_lines_xyz_rgb [N,6] float, other_lines list[str], vertex_indices list[int])."""
    lines = path.read_text().splitlines()
    verts = []
    vert_idx = []
    other = []
    for i, L in enumerate(lines):
        if L.startswith("v "):
            parts = L.split()
            if len(parts) == 7:
                verts.append([float(x) for x in parts[1:]])
                vert_idx.append(i)
                continue
        other.append((i, L))
    return np.asarray(verts, dtype=np.float32), other, vert_idx, lines


def save_obj(path: Path, verts: np.ndarray, lines: list[str], vert_idx: list[int]):
    out = list(lines)
    for i, vi in enumerate(vert_idx):
        x, y, z, r, g, b = verts[i]
        out[vi] = f"v {x:.8f} {y:.8f} {z:.8f} {r:.6f} {g:.6f} {b:.6f}"
    path.write_text("\n".join(out) + "\n")


def edit_red_hair(verts: np.ndarray) -> np.ndarray:
    """Tint upper-head vertices red. Threshold on y (FLAME canonical, y-up)."""
    out = verts.copy()
    # Choose top quartile of y as 'hair region'. Tune by inspection.
    y = verts[:, 1]
    y_thresh = np.quantile(y, 0.75)
    mask = y > y_thresh
    # Blend toward saturated red.
    target = np.array([0.85, 0.10, 0.10], dtype=np.float32)
    alpha = 0.85
    out[mask, 3:] = alpha * target + (1 - alpha) * verts[mask, 3:]
    print(f"red_hair: tinted {mask.sum()} / {len(verts)} verts above y={y_thresh:.3f}")
    return out


def edit_pale_skin(verts: np.ndarray) -> np.ndarray:
    out = verts.copy()
    rgb = verts[:, 3:]
    gray = rgb.mean(axis=1, keepdims=True)
    pale = 0.7 * gray + 0.3 * 1.0   # mix toward white
    out[:, 3:] = np.clip(pale.repeat(3, axis=1), 0, 1)
    return out


EDITS = {"red_hair": edit_red_hair, "pale_skin": edit_pale_skin}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to <anchor>_textured_mesh.obj")
    ap.add_argument("--output", required=True, help="path to write edited OBJ")
    ap.add_argument("--edit", required=True, choices=list(EDITS))
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    verts, _, vert_idx, lines = load_obj(inp)
    print(f"loaded {len(verts)} verts from {inp.name}")
    assert verts.shape[1] == 6, f"expected x y z r g b, got {verts.shape[1]} columns"
    edited = EDITS[args.edit](verts)
    save_obj(out, edited, lines, vert_idx)
    diff = np.abs(edited[:, 3:] - verts[:, 3:]).sum(axis=1)
    n_changed = (diff > 1e-6).sum()
    print(f"wrote {out}  (changed {n_changed} / {len(verts)} vertex colours)")


if __name__ == "__main__":
    main()
