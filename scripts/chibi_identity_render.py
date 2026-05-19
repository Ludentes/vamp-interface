"""End-to-end chibi identity render: anchor -> portrait -> register -> bake
-> textured render.

    uv run python scripts/chibi_identity_render.py \\
        --anchor <id> --identity <face_ref.png> --canny <structure.png> \\
        --out <dir>

Wires the Task 1-5 chibi units into the spike driver used to run S1
(registration quality) and S3 (no double-shading). Stages:

  1. generate_portrait  — Flux+PuLID+Canny via ComfyUI -> <out>/<anchor>.png
  2. detect_portrait_landmarks — insightface-106 2D landmarks on the portrait
  3. fit_tps / warp_image — TPS warp portrait -> canonical frontal projection
  4. bake_portrait_to_uv — single-view projective bake into the Koban UV atlas
  5. render_textured — unlit pytorch3d turntable of the textured canonical mesh

The identity input is a *reference image* PNG (PuLID derives its own face
embedding internally) — there is no embedding-vector path; see
chibi_portrait.py. Both --identity and --canny must already be staged into the
ComfyUI input directory so the workflow's LoadImage nodes can resolve them by
basename.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from chibi.chibi_portrait import generate_portrait  # noqa: E402
from chibi.koban_asset import (  # noqa: E402
    load_koban,
    load_koban_landmarks,
    load_koban_view,
)
from chibi.koban_bake import bake_portrait_to_uv  # noqa: E402
from chibi.mesh import TexturedMesh  # noqa: E402
from chibi.mesh_render import render_textured  # noqa: E402
from chibi.register import fit_tps, warp_image  # noqa: E402

CANON = "exp_output/chibi_meshes/koban_canonical"
AZIMS = [-30, 0, 30]


def detect_portrait_landmarks(png_path: Path) -> torch.Tensor:
    """Detect insightface 2D-106 landmarks on the portrait.

    Returns the full (106, 2) float32 landmark array (image-pixel coords). The
    caller subsets it to the K canonical landmark slots via the index map from
    load_koban_landmarks(..., return_idx=True).
    """
    import insightface

    app = insightface.app.FaceAnalysis(
        allowed_modules=["detection", "landmark_2d_106"])
    app.prepare(ctx_id=0)
    # insightface expects BGR (cv2 convention); imageio gives RGB.
    img_rgb = np.asarray(imageio.imread(png_path))[..., :3]
    faces = app.get(img_rgb[..., ::-1].copy())
    if not faces:
        raise RuntimeError(f"no face detected in {png_path}")
    if len(faces) > 1:
        # Largest detected face wins — the portrait is a single centered head.
        faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return torch.tensor(faces[-1].landmark_2d_106, dtype=torch.float32)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anchor", required=True,
                    help="anchor id; names the portrait PNG")
    ap.add_argument("--identity", required=True,
                    help="face reference PNG for PuLID (must be staged in the "
                         "ComfyUI input dir)")
    ap.add_argument("--canny", default=None,
                    help="optional face-structure Canny PNG (staged in the "
                         "ComfyUI input dir). Required for Task 6 to run.")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--seed", type=int, default=0,
                    help="generation seed (fixed per anchor)")
    ap.add_argument("--comfy-url", default="http://127.0.0.1:8188",
                    help="ComfyUI REST endpoint")
    ap.add_argument("--tex-size", type=int, default=1024,
                    help="UV atlas resolution")
    ap.add_argument("--device", default="cuda",
                    help="render device for render_textured")
    args = ap.parse_args()

    # Task 6 cannot run without both a PuLID identity ref and a Canny
    # structure image — fail clearly here rather than crashing inside ComfyUI.
    identity = Path(args.identity)
    if not identity.exists():
        ap.error(f"--identity not found: {identity}")
    if args.canny is None:
        ap.error("--canny is required: the Flux+PuLID portrait workflow needs "
                 "a Canny structure image. Stage one in the ComfyUI input "
                 "dir and pass its path.")
    canny = Path(args.canny)
    if not canny.exists():
        ap.error(f"--canny not found: {canny}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- Load the canonical Koban asset ------------------------------------
    koban = load_koban(CANON)
    view = load_koban_view(CANON)
    canon_lmk = load_koban_landmarks(CANON)             # (K,2) frontal pixels
    if106_idx = load_koban_landmarks(CANON, return_idx=True)  # (K,) into 106

    # --- Stage 1: portrait -------------------------------------------------
    png = generate_portrait(args.anchor, identity_image=identity,
                            out_dir=out, seed=args.seed,
                            canny_image=canny, comfy_url=args.comfy_url)
    print(f"[chibi_identity] portrait: {png}")

    # --- Stage 2: landmark detection + canonical subset --------------------
    port_lmk_all = detect_portrait_landmarks(png)       # (106,2)
    port_lmk = port_lmk_all[if106_idx]                  # (K,2) — matched order

    # --- Stage 3: TPS registration portrait -> canonical -------------------
    tps = fit_tps(port_lmk, canon_lmk)
    portrait = torch.tensor(
        np.asarray(imageio.imread(png))[..., :3] / 255.0,
        dtype=torch.float32)
    warped = warp_image(portrait, tps)                  # (H,W,3) float[0,1]
    imageio.imwrite(out / "warped.png",
                    (warped.clamp(0, 1) * 255).round().to(torch.uint8).numpy())

    # --- Stage 4: single-view UV bake --------------------------------------
    tex = bake_portrait_to_uv(koban, warped, view, tex_size=args.tex_size)
    imageio.imwrite(out / "texture.png",
                    (tex.clamp(0, 1) * 255).round().to(torch.uint8).numpy())

    # --- Stage 5: textured turntable render --------------------------------
    tmesh = TexturedMesh(verts=koban.verts, faces=koban.faces,
                         uv=koban.uv, uv_faces=koban.uv_faces, texture=tex)
    frames = render_textured(tmesh, AZIMS, device=args.device)  # (T,H,W,3) u8
    for azim, frame in zip(AZIMS, frames):
        imageio.imwrite(out / f"render_{azim}.png", frame.numpy())

    print(f"[chibi_identity] wrote portrait/warped/texture/render_* to {out}")


if __name__ == "__main__":
    main()
