#!/usr/bin/env python3
"""Re-bake LAM's `<anchor>_textured_mesh.obj` so vertex colours are display RGB.

LAM saves the textured mesh with raw SH band-0 coefficients in the colour
columns (mesh_utils.save_obj writes `shs.squeeze(1)` verbatim). When an
artist imports this OBJ into Blender they see SH values, not display RGB —
they're editing on a wrong palette, and any vertex they DON'T touch comes
back through `RGB2SH` again (= double-encoded) overexposing those regions.

The fix: apply `SH2RGB(x) = x * C0 + 0.5` to each vertex colour. The
output OBJ shows the artist exactly what LAM renders, and our hook's
`RGB2SH` on the way back gives a clean round-trip.

Usage:
    uv run scripts/lam_bake_display_rgb_obj.py --input <anchor>_textured_mesh.obj \
        --output <anchor>_textured_mesh_display.obj
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

C0 = 0.28209479177387814  # SH band-0 coefficient (lam/.../sh_utils.py)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    src = Path(args.input).read_text().splitlines()
    out_lines = list(src)
    n_changed = 0
    for i, L in enumerate(src):
        if not L.startswith("v "):
            continue
        parts = L.split()
        if len(parts) != 7:
            continue
        x, y, z = parts[1], parts[2], parts[3]
        r, g, b = float(parts[4]), float(parts[5]), float(parts[6])
        # SH2RGB
        r2 = r * C0 + 0.5
        g2 = g * C0 + 0.5
        b2 = b * C0 + 0.5
        out_lines[i] = f"v {x} {y} {z} {r2:.6f} {g2:.6f} {b2:.6f}"
        n_changed += 1
    Path(args.output).write_text("\n".join(out_lines) + "\n")
    print(f"baked SH2RGB on {n_changed} verts → {args.output}")


if __name__ == "__main__":
    main()
