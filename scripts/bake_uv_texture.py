"""nvdiffrast UV-texture bake driver: shaped_mesh.obj + cano.ply -> textured OBJ.

  python scripts/bake_uv_texture.py \
      --ply  /home/newub/w/LAM/exps/cano_gs/me_512_cano.ply \
      --mesh /home/newub/w/LAM/exps/cano_gs/me_512_shaped_mesh.obj \
      --out  exp_output/lam_chibi/renders/bake_v3/me_512 --stem me_512

Mesh and splats are recentred onto the origin by TRANSLATION ONLY. Writes
<stem>_textured.obj/.mtl/_texture.png plus verdict turntable renders.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
import imageio.v2 as imageio                                # noqa: E402

from chibi.mesh import TexturedMesh                         # noqa: E402
from chibi.camera_rig import turntable_views                # noqa: E402
from chibi.splat_render import load_splat_ply, render_splats  # noqa: E402
from chibi.uv_template import load_flame_uv                 # noqa: E402
from chibi.texture_bake import (bake_texture, dilate_texture,  # noqa: E402
                                unsharp_mask)
from chibi.mesh_render import render_textured               # noqa: E402

FLAME_TEMPLATE = ("/home/newub/w/LAM/model_zoo/human_parametric_models/"
                  "flame_assets/flame/head_template_mesh.obj")


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


def _write_textured_obj(out: Path, stem: str, verts: torch.Tensor,
                        faces: torch.Tensor, uv: torch.Tensor,
                        uv_faces: torch.Tensor) -> None:
    """Write <stem>_textured.obj + .mtl referencing <stem>_texture.png."""
    obj = [f"mtllib {stem}_textured.mtl", f"usemtl {stem}_mat"]
    for v in verts.tolist():
        obj.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
    for t in uv.tolist():
        obj.append(f"vt {t[0]:.6f} {t[1]:.6f}")
    for fp, ft in zip((faces + 1).tolist(), (uv_faces + 1).tolist()):
        obj.append(f"f {fp[0]}/{ft[0]} {fp[1]}/{ft[1]} {fp[2]}/{ft[2]}")
    (out / f"{stem}_textured.obj").write_text("\n".join(obj) + "\n")
    (out / f"{stem}_textured.mtl").write_text(
        f"newmtl {stem}_mat\nmap_Kd {stem}_texture.png\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True, help="<stem>_cano.ply")
    ap.add_argument("--mesh", required=True, help="<stem>_shaped_mesh.obj")
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--stem", required=True, help="anchor name for filenames")
    ap.add_argument("--n_azim", type=int, default=12)
    ap.add_argument("--splat_size", type=int, default=2048)
    ap.add_argument("--tex_size", type=int, default=2048)
    ap.add_argument("--render_size", type=int, default=512)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    verts, faces = _load_plain_obj(args.mesh)
    splats = load_splat_ply(args.ply)
    centre = verts.mean(0, keepdim=True)
    verts = verts - centre
    splats.xyz = (splats.xyz.double() - centre).to(torch.float32)
    extent = float((verts.max(0).values - verts.min(0).values).max())
    dist = extent * 2.2

    fuv = load_flame_uv(FLAME_TEMPLATE)
    assert fuv.uv_faces.shape[0] == faces.shape[0], \
        "FLAME template / shaped_mesh face count mismatch"

    # five elevation rings: the ±45° rings give concavities (nostrils, under-
    # brow) a near-axial sample instead of only grazing ones.
    elevs = (-45.0, -20.0, 0.0, 20.0, 45.0)
    views = turntable_views(n_azim=args.n_azim, elevs=elevs,
                            dist=dist, image_size=args.splat_size)
    images, depth = render_splats(splats, views)
    # views are elev-major; front = (elev=0 block, azim 0).
    front = args.n_azim * elevs.index(0.0)
    imageio.imwrite(out / "splat_ref.png", images[front].numpy())

    # Seam-free recipe: a single normal-weighted blend across all views, then
    # an unsharp mask to recover crispness. Winner-take-all (mode="best") is
    # crisp but its hard view boundaries leave visible seams on the cheeks and
    # around the eyes; a blend has none, and unsharp is a local op so it adds
    # no seams of its own.
    texture, filled = bake_texture(verts, faces, fuv, images, depth, views,
                                   args.tex_size, mode="blend",
                                   facing_power=2.0)
    texture = dilate_texture(texture, filled, iters=16)
    texture = unsharp_mask(texture, sigma=2.0, amount=0.6)

    tex_png = (texture.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).numpy()
    imageio.imwrite(out / f"{args.stem}_texture.png", tex_png)
    _write_textured_obj(out, args.stem, verts, faces, fuv.uv, fuv.uv_faces)

    import PIL.Image

    tmesh = TexturedMesh(verts=verts, faces=faces, uv=fuv.uv,
                         uv_faces=fuv.uv_faces, texture=texture)
    # 2x supersample: pytorch3d has no silhouette MSAA, so render double and
    # box-downsample to antialias the head outline and texture minification.
    ss = 2
    big = render_textured(tmesh, [0.0, 30.0, 60.0, 90.0],
                          image_size=args.render_size * ss)
    frames = []
    for fr in big:
        im = PIL.Image.fromarray(fr.numpy()[..., :3]).resize(
            (args.render_size, args.render_size), PIL.Image.LANCZOS)
        frames.append(np.asarray(im))
    for i, az in enumerate((0, 30, 60, 90)):
        imageio.imwrite(out / f"textured_mesh_{az:03d}.png", frames[i])

    def _resize(arr: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
        if arr.shape[:2] == hw:
            return arr[..., :3]
        return np.asarray(PIL.Image.fromarray(arr[..., :3]).resize(
            (hw[1], hw[0])))[..., :3]

    b = frames[0]                                      # textured mesh, front
    hw = b.shape[:2]
    # textured mesh vs the photoreal splat render — the bake's true target
    splat = _resize(images[front].numpy(), hw)
    imageio.imwrite(out / "sidebyside_textured_vs_splat.png",
                    np.concatenate([b, splat], axis=1))
    # best-effort side-by-side vs the v2 vertex bake, if it exists
    v2 = Path(f"exp_output/lam_chibi/renders/bake_v1/{args.stem}/"
              "baked_mesh_000.png")
    if v2.exists():
        imageio.imwrite(out / "sidebyside_v2_v3.png",
                        np.concatenate([_resize(imageio.imread(v2), hw), b],
                                       axis=1))
    print(f"[uv-bake] {args.stem}: {int(filled.sum())}/{filled.numel()} texels "
          f"seen, dist={dist:.3f} -> {out}")


if __name__ == "__main__":
    main()
