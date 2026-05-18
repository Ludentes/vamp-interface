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

from swap_core import (_crop_region, detect_source, load_swapper,
                        make_face_app, mediapipe_kps_bbox, swap_identity)

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


def identity_cos(app, result_bgr, source_emb):
    """cos(source ArcFace embedding, output face ArcFace embedding).

    Re-detects the swapped face on the output the same way the swap does --
    crop the face region and upscale to 512 px so SCRFD has enough pixels --
    then takes its normed embedding. Returns NaN if no face is recoverable.
    """
    kps, bbox = mediapipe_kps_bbox(result_bgr)
    if kps is None or bbox is None:
        return float("nan")
    rx0, ry0, rx1, ry1 = _crop_region(bbox, result_bgr.shape)
    crop = result_bgr[ry0:ry1, rx0:rx1]
    if crop.size == 0:
        return float("nan")
    ch, cw = crop.shape[:2]
    s = 512.0 / max(ch, cw)
    up = cv2.resize(crop, (max(1, round(cw * s)), max(1, round(ch * s))),
                    interpolation=cv2.INTER_LANCZOS4)
    faces = app.get(up)
    if not faces:
        return float("nan")
    f = max(faces, key=lambda x: x.det_score)
    return float(np.dot(f.normed_embedding, source_emb))


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

    # source identity faces -- (sid, Face, normed_embedding)
    sources = []
    for sid in args.ids:
        sp = args.identities / f"{sid}.png"
        src_img = cv2.imread(str(sp))
        face = detect_source(app, src_img) if src_img is not None else None
        if face is None:
            print(f"  [warn] no face in source {sp}")
        emb = face.normed_embedding if face is not None else None
        sources.append((sid, face, emb))

    rows = []   # (label, [doll_bgr, swap0, swap1, ...])
    scores = []  # (arm, step, sid, cosine, mode, det_score)
    for arm, renders in ARM_RENDERS.items():
        for rname in renders:
            rp = args.root / "renders" / rname
            doll = cv2.imread(str(rp))
            if doll is None:
                print(f"  [skip] missing {rp}")
                continue
            step = rname.split("_st")[1][:2]
            tiles = [doll]
            for sid, face, emb in sources:
                if face is None:
                    tiles.append(doll)
                    continue
                res, mode, det = swap_identity(app, swapper, doll, face)
                cos = identity_cos(app, res, emb)
                print(f"  {arm:14s} {rname[:40]:40s} <- {sid}: "
                      f"{mode} det={det:.2f} id_cos={cos:.3f}")
                scores.append((arm, step, sid, cos, mode, det))
                tiles.append(res)
            rows.append((f"{arm}  {step}st", tiles))

    if scores:
        csv_path = args.root / "swap_test_scores.csv"
        with open(csv_path, "w") as fh:
            fh.write("arm,step,identity,id_cos,mode,det_score\n")
            for arm, step, sid, cos, mode, det in scores:
                fh.write(f"{arm},{step},{sid},{cos:.4f},{mode},{det:.4f}\n")
        print(f"\n[swap-test] identity-cosine table -> {csv_path}")
        print(f"  {'arm':14s} {'step':4s} {'identity':10s} {'id_cos':>8s} "
              f"{'mode':8s} {'det':>6s}")
        for arm, step, sid, cos, mode, det in scores:
            print(f"  {arm:14s} {step:4s} {sid:10s} {cos:8.3f} "
                  f"{mode:8s} {det:6.2f}")
        finite = [c for _, _, _, c, _, _ in scores if np.isfinite(c)]
        if finite:
            print(f"  median id_cos = {float(np.median(finite)):.3f}  "
                  f"(n={len(finite)})")

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
