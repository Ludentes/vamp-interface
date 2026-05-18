#!/usr/bin/env python3
"""Face-swap test on the bake-off renders for the zimage / sdxl_lightning arms.

The bake-off verdict rejected sdxl_lightning on the grounds that flat 2D
illustration output is unsuitable for inswapper. This script tests that claim
directly: take representative dolls from each arm, run the swap_core pipeline
(MediaPipe kps -> eye-collapse -> inswapper_128), and montage doll-vs-swap so
the inswapper output can be eyeballed per arm.

Usage:
    python scripts/matryoshka_bakeoff_swap_test.py \\
        --root exp_output/matryoshka_bakeoff \\
        --identities data/importer/identities \\
        --swapper ~/w/ComfyUI/models/insightface/inswapper_128.onnx \\
        --ids id_03 id_11
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from swap_core import detect_source, load_swapper, make_face_app, swap_identity

# representative renders per arm: one per step count, first sampler/seed
ARM_RENDERS = {
    "zimage_turbo": [
        "zimage_turbo_st06_euler_simple_seed74029470.png",
        "zimage_turbo_st08_euler_simple_seed74061146.png",
        "zimage_turbo_st12_euler_simple_seed74092822.png",
    ],
    "sdxl_lightning": [
        "sdxl_lightning_st04_dpmpp_sde_sgm_uniform_cs040_ce040_seed73823576.png",
        "sdxl_lightning_st08_dpmpp_sde_sgm_uniform_cs040_ce040_seed73950280.png",
    ],
}

TILE = 320
PAD = 12
LABEL_H = 24


def _font(size: int):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def thumb(bgr: np.ndarray) -> Image.Image:
    im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    im.thumbnail((TILE, TILE))
    return im


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("exp_output/matryoshka_bakeoff"))
    ap.add_argument("--identities", type=Path,
                    default=Path("data/importer/identities"))
    ap.add_argument("--swapper", type=Path, required=True)
    ap.add_argument("--ids", nargs="+", default=["id_03", "id_11"])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or args.root / "swap_test.png"

    app = make_face_app()
    swapper = load_swapper(str(args.swapper.expanduser()))

    # source identity faces
    sources = []
    for sid in args.ids:
        sp = args.identities / f"{sid}.png"
        src_img = cv2.imread(str(sp))
        face = detect_source(app, src_img) if src_img is not None else None
        if face is None:
            print(f"  [warn] no face in source {sp}")
        sources.append((sid, face))

    rows = []  # (label, [doll_bgr, swap0, swap1, ...])
    for arm, renders in ARM_RENDERS.items():
        for rname in renders:
            rp = args.root / "renders" / rname
            doll = cv2.imread(str(rp))
            if doll is None:
                print(f"  [skip] missing {rp}")
                continue
            tiles = [doll]
            for sid, face in sources:
                if face is None:
                    tiles.append(doll)
                    continue
                res, mode, det = swap_identity(app, swapper, doll, face)
                print(f"  {arm:14s} {rname[:40]:40s} <- {sid}: {mode} det={det:.2f}")
                tiles.append(res)
            step = rname.split("_st")[1][:2]
            rows.append((f"{arm}  {step}st", tiles))

    if not rows:
        print("[swap-test] no rows produced")
        return 1

    ncol = 1 + len(sources)
    col_labels = ["doll"] + [sid for sid, _ in sources]
    grid_w = PAD + ncol * (TILE + PAD)
    grid_h = LABEL_H + PAD + len(rows) * (TILE + LABEL_H + PAD)
    canvas = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(canvas)
    f = _font(15)

    for ci, cl in enumerate(col_labels):
        draw.text((PAD + ci * (TILE + PAD), 4), cl, fill="black", font=f)

    for ri, (label, tiles) in enumerate(rows):
        y0 = LABEL_H + PAD + ri * (TILE + LABEL_H + PAD)
        draw.text((PAD, y0), label, fill="black", font=f)
        for ci, t in enumerate(tiles):
            im = thumb(t)
            x = PAD + ci * (TILE + PAD)
            canvas.paste(im, (x + (TILE - im.width) // 2, y0 + LABEL_H))

    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(f"[swap-test] {out}  ({grid_w}x{grid_h})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
