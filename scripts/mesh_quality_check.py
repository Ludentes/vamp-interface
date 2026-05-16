"""Milestone 0 — the splats->mesh quality gate.

Render an ARKit-driven take as an animated MESH (per-frame .ply -> ChibiMesh
-> pytorch3d) and write it to mp4. Compare the result to the splat render of
the same take to confirm the pivot did not lose quality. No chibi here.

Run in the lam conda env:
  python scripts/mesh_quality_check.py \
      --ply_dir /home/newub/w/LAM/exps/images/lam/lam_20k/asian_m \
      --obj /home/newub/w/LAM/exps/cano_gs/asian_m_textured_mesh.obj \
      --out exp_output/lam_chibi/renders/mesh_v1/driven_asian_m.mp4
"""
from __future__ import annotations
import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import imageio.v2 as imageio  # noqa: E402

from chibi.mesh import ChibiMesh                       # noqa: E402
from chibi.mesh_extract import load_chibi_mesh, load_gaussian_ply  # noqa: E402
from chibi.mesh_render import render                   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply_dir", required=True,
                    help="dir of LAM per-frame NNNN.ply files")
    ap.add_argument("--obj", required=True,
                    help="canonical <stem>_textured_mesh.obj (faces source)")
    ap.add_argument("--out", required=True, help="output mp4 path")
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--fps", type=int, default=25)
    args = ap.parse_args()

    plys = sorted(glob.glob(str(Path(args.ply_dir) / ("[0-9]" * 4 + ".ply"))))
    assert plys, f"no NNNN.ply files in {args.ply_dir}"
    faces = load_chibi_mesh(args.obj).faces
    print(f"[mesh] {len(plys)} frames, {faces.shape[0]} faces")

    meshes = []
    for p in plys:
        verts, rgb = load_gaussian_ply(p)
        assert verts.shape[0] == int(faces.max()) + 1, (
            f"{p}: vert count {verts.shape[0]} != faces topology "
            f"{int(faces.max()) + 1}")
        meshes.append(ChibiMesh(verts=verts, faces=faces, rgb=rgb))

    # azim 0 = front view, fixed camera for the whole driven sequence.
    frames = render(meshes, [0.0], image_size=args.image_size)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(args.out, list(frames.numpy()), fps=args.fps)
    print(f"[render] driven mesh video -> {args.out}")


if __name__ == "__main__":
    main()
