"""Chibi mesh-pivot driver: v3 textured mesh -> chibi deform -> turntable mp4.

Loads the v3 UV-textured OBJ (output of scripts/bake_uv_texture.py), renders a
baseline turntable, applies the fitted ChibiField, renders the chibi turntable,
and writes both plus a baseline-vs-chibi side-by-side. A static-pose render
(camera orbit) — driven chibi animation reuses the same units, later.

  python scripts/chibi_mesh_render.py \
      --obj   exp_output/lam_chibi/renders/bake_v3/asian_m/asian_m_textured.obj \
      --field exp_output/lam_chibi/diff_geometry/chibi_field_params.json \
      --out   exp_output/lam_chibi/renders/mesh_v1
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np                                  # noqa: E402
import imageio.v2 as imageio                        # noqa: E402

from chibi.mesh_extract import load_textured_mesh    # noqa: E402
from chibi.mesh_render import render_textured        # noqa: E402

MASKS = ("/home/newub/w/LAM/model_zoo/human_parametric_models/"
         "flame_assets/flame/FLAME_masks.pkl")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", required=True, help="<stem>_textured.obj (v3 bake)")
    ap.add_argument("--field", required=True,
                    help="chibi_pipeline_params.json (may be absent)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_frames", type=int, default=72)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--through", default=None,
                    help="stop the pipeline after this stage (inspection)")
    args = ap.parse_args()

    obj = Path(args.obj)
    assert obj.exists(), (
        f"textured-mesh OBJ not found: {obj}\n"
        f"Run scripts/bake_uv_texture.sh on the anchor first to bake it.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = obj.stem.replace("_textured", "")

    mesh = load_textured_mesh(str(obj))
    print(f"[mesh] {stem}: {mesh.verts.shape[0]} verts, "
          f"texture {tuple(mesh.texture.shape)}")
    azims = [360.0 * i / args.n_frames for i in range(args.n_frames)]

    base = render_textured(mesh, azims, image_size=args.image_size)
    imageio.mimwrite(out / f"mesh_baseline_{stem}.mp4", list(base.numpy()), fps=24)
    print(f"[render] baseline -> mesh_baseline_{stem}.mp4")

    from chibi.pipeline import ChibiPipeline
    pipe = ChibiPipeline(MASKS, args.field if Path(args.field).exists() else None)
    deformed = pipe.run(mesh.verts, mesh.faces, through=args.through)
    chibi = type(mesh)(verts=deformed, faces=mesh.faces, uv=mesh.uv,
                       uv_faces=mesh.uv_faces, texture=mesh.texture)
    tag = f"{stem}_{args.through}" if args.through else stem
    cf = render_textured(chibi, azims, image_size=args.image_size)
    imageio.mimwrite(out / f"mesh_chibi_{tag}.mp4", list(cf.numpy()), fps=24)
    print(f"[render] chibi -> mesh_chibi_{tag}.mp4")

    # baseline (left) vs chibi (right) — the verdict frame pair.
    sxs = [np.concatenate([b, c], axis=1)
           for b, c in zip(base.numpy(), cf.numpy())]
    imageio.mimwrite(out / f"sidebyside_{tag}.mp4", sxs, fps=24)
    print(f"[render] side-by-side -> sidebyside_{tag}.mp4")


if __name__ == "__main__":
    main()
