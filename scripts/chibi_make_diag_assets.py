"""Build diagnostic per-vertex multiplier assets for iris-through-lid probes.

Emits (N=20018) tensors saved into <out_dir>/diag/ so we can isolate which
splat-attribute is the load-bearing fix for the iris-leak artifact at
chibi_strength=2.0.

Emitted assets:
  axis_boost_ax{0,1,2}_x3.0.npy   (N, 3) float32 — local-axis-K sigma × 3 on
                                  FLAME 'eye_region' verts (lifted to 20018),
                                  1.0 elsewhere. Probes which local Gaussian
                                  axis is the tangent-occlusion direction.

  opacity_eyeball_0.1.npy         (N,)   float32 — opacity × 0.1 on eyeball
                                  verts (left+right), 1.0 elsewhere. If iris
                                  disappears, mechanism is alpha-blending of
                                  partly-transparent lid sheet. If iris
                                  persists, mechanism is geometric splat gap.

  opacity_eye_region_0.3.npy      (N,)   float32 — opacity × 0.3 on eye_region
                                  (lid skin) verts. Sanity-confirms C1: if lid
                                  α drop makes iris MORE visible, alpha is the
                                  mechanism.

Use the existing FLAME mask lift from chibi_make_assets.py (same 5023→20018
midpoint-rule used to build chibi_scale_ratio).
"""
from __future__ import annotations
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chibi_make_assets import (  # noqa: E402
    load_obj, parse_obj_faces, lift_mask_to_subdivided,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True,
                    help="FLAME 5023-vert head_template_mesh.obj")
    ap.add_argument("--flame_masks", required=True,
                    help="FLAME_masks.pkl with eye_region, left_eyeball, right_eyeball")
    ap.add_argument("--out_dir", required=True, help="dir to write *.npy into")
    ap.add_argument("--n_verts", type=int, default=20018)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tpl_verts, tpl_lines, _ = load_obj(Path(args.template))
    tpl_faces = parse_obj_faces(tpl_lines)
    masks = pickle.load(open(args.flame_masks, "rb"), encoding="latin1")

    eye_region_5023 = np.asarray(masks["eye_region"], dtype=np.int64)
    eye_region_20018, _ = lift_mask_to_subdivided(
        tpl_verts[:, :3], tpl_faces, eye_region_5023,
    )
    eyeballs_5023 = np.concatenate([
        np.asarray(masks["left_eyeball"], dtype=np.int64),
        np.asarray(masks["right_eyeball"], dtype=np.int64),
    ])
    eyeballs_20018, _ = lift_mask_to_subdivided(
        tpl_verts[:, :3], tpl_faces, eyeballs_5023,
    )
    print(f"eye_region 5023={len(eye_region_5023)}  20018={len(eye_region_20018)}")
    print(f"eyeballs   5023={len(eyeballs_5023)}  20018={len(eyeballs_20018)}")

    N = args.n_verts

    # Axis boost variants for Exp B.
    for axis in (0, 1, 2):
        ab = np.ones((N, 3), dtype=np.float32)
        ab[eye_region_20018, axis] = 3.0
        path = out / f"axis_boost_ax{axis}_x3.0.npy"
        np.save(path, ab)
        print(f"wrote {path}  (eye_region axis {axis} × 3.0)")

    # All-axis 3× (sanity / matches isotropic eye_region_scale_boost=3.0).
    ab_all = np.ones((N, 3), dtype=np.float32)
    ab_all[eye_region_20018, :] = 3.0
    np.save(out / "axis_boost_all_x3.0.npy", ab_all)
    print(f"wrote {out / 'axis_boost_all_x3.0.npy'}  (eye_region all-axes × 3.0)")

    # Opacity multipliers for Exp C.
    op_eye = np.ones((N,), dtype=np.float32)
    op_eye[eyeballs_20018] = 0.1
    np.save(out / "opacity_eyeball_0.1.npy", op_eye)
    print(f"wrote {out / 'opacity_eyeball_0.1.npy'}  (eyeballs × 0.1)")

    op_lid = np.ones((N,), dtype=np.float32)
    op_lid[eye_region_20018] = 0.3
    np.save(out / "opacity_eye_region_0.3.npy", op_lid)
    print(f"wrote {out / 'opacity_eye_region_0.3.npy'}  (eye_region × 0.3)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
