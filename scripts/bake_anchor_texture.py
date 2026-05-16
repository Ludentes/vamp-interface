"""Render-and-bake driver: canonical splat ply + FLAME mesh -> textured mesh.

  python scripts/bake_anchor_texture.py \
      --ply  /home/newub/w/LAM/exps/cano_gs/me_512_cano.ply \
      --mesh /home/newub/w/LAM/exps/cano_gs/me_512_shaped_mesh.obj \
      --out  exp_output/lam_chibi/renders/bake_v1/me

Mesh and splats are recentred onto the origin by TRANSLATION ONLY (rescaling
positions without rescaling Gaussian scales turns the splats into a sparse dot
grid). The camera distance is derived from the recentred extent.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch                                              # noqa: E402
import imageio.v2 as imageio                              # noqa: E402

from chibi.mesh import ChibiMesh                          # noqa: E402
from chibi.camera_rig import turntable_views              # noqa: E402
from chibi.splat_render import load_splat_ply, render_splats  # noqa: E402
from chibi.bake import bake_vertex_colors                 # noqa: E402
from chibi.mesh_render import render                      # noqa: E402


def _load_plain_obj(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Parse a plain `v x y z` + `f` OBJ (trimesh export — no per-vertex rgb)."""
    verts, faces = [], []
    for line in Path(path).read_text().splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == "v":
            verts.append([float(x) for x in p[1:4]])
        elif p[0] == "f":
            faces.append([int(x.split("/")[0]) - 1 for x in p[1:4]])
    return (torch.tensor(verts, dtype=torch.float64),
            torch.tensor(faces, dtype=torch.int64))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True, help="<stem>_cano.ply")
    ap.add_argument("--mesh", required=True, help="<stem>_shaped_mesh.obj")
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--n_azim", type=int, default=12)
    ap.add_argument("--render_size", type=int, default=512)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    verts, faces = _load_plain_obj(args.mesh)
    splats = load_splat_ply(args.ply)
    # recentre mesh + splats onto the origin by the SAME translation (they share
    # one world frame). Translation only — Gaussian scales must not change.
    centre = verts.mean(0, keepdim=True)
    verts = verts - centre
    splats.xyz = (splats.xyz.double() - centre).to(torch.float32)
    extent = float((verts.max(0).values - verts.min(0).values).max())
    dist = extent * 2.2
    mesh = ChibiMesh(verts=verts, faces=faces,
                     rgb=torch.full((verts.shape[0], 3), 0.5,
                                    dtype=torch.float32))

    views = turntable_views(n_azim=args.n_azim, elevs=(-20.0, 0.0, 20.0),
                            dist=dist, image_size=args.render_size)
    images, depth = render_splats(splats, views)
    imageio.imwrite(out / "splat_ref.png", images[args.n_azim // 2].numpy())

    baked = bake_vertex_colors(mesh, images, depth, views)
    frames = render([baked], [0.0, 30.0, 60.0, 90.0], image_size=args.render_size)
    for i, az in enumerate((0, 30, 60, 90)):
        imageio.imwrite(out / f"baked_mesh_{az:03d}.png", frames[i].numpy())
    print(f"[bake] {baked.verts.shape[0]} verts, dist={dist:.3f} -> {out}")


if __name__ == "__main__":
    main()
