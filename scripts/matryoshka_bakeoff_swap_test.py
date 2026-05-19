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

from swap_core import (DEFAULT_SWAPPER, crop_and_upscale, detect_source,
                        load_swapper, make_face_app, mediapipe_kps_bbox,
                        swap_identity)

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
    crop the face region and upscale so SCRFD has enough pixels (shared
    swap_core.crop_and_upscale, so the metric's detection path matches the
    pipeline's). If SCRFD still misses the face, falls back to a MediaPipe
    forced Face + the recognition model directly -- otherwise the metric
    would silently under-report exactly the SCRFD-weak renders. Returns NaN
    only when no face is recoverable at all.
    """
    kps, bbox = mediapipe_kps_bbox(result_bgr)
    if kps is None or bbox is None:
        return float("nan")
    up, _ = crop_and_upscale(result_bgr, bbox)
    if up is None:
        return float("nan")
    faces = app.get(up)
    if faces:
        f = max(faces, key=lambda x: x.det_score)
        return float(np.dot(f.normed_embedding, source_emb))
    # SCRFD missed it -- forced MediaPipe Face through the recognition model
    kps_up, bbox_up = mediapipe_kps_bbox(up)
    rec = app.models.get("recognition")
    if kps_up is None or rec is None:
        return float("nan")
    from insightface.app.common import Face
    f = Face(bbox=bbox_up, kps=kps_up, det_score=1.0)
    rec.get(up, f)
    return float(np.dot(f.normed_embedding, source_emb))


def _swap_stats(app, swapper, dolls, sources):
    """(default_rate, median_id_cos) over doll x identity swaps.

    default_rate is the fraction of swaps where SCRFD detected the face
    (mode == 'default'); median_id_cos is over the finite cosines.
    """
    modes, cosines = [], []
    for doll in dolls:
        for _sid, face, emb in sources:
            if face is None:
                continue
            res, mode, _det = swap_identity(app, swapper, doll, face)
            modes.append(mode)
            cos = identity_cos(app, res, emb)
            if np.isfinite(cos):
                cosines.append(cos)
    default_rate = (modes.count("default") / len(modes)) if modes else 0.0
    median = float(np.median(cosines)) if cosines else float("nan")
    return default_rate, median


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("exp_output/matryoshka_bakeoff"))
    ap.add_argument("--identities", type=Path,
                    default=Path("data/importer/identities"))
    ap.add_argument("--swapper", type=Path, default=Path(DEFAULT_SWAPPER),
                    help="swapper ONNX (default: HyperSwap 1c)")
    ap.add_argument("--ids", nargs="+", default=["id_03", "id_11"])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--refined-root", type=Path, default=None,
                    help="root of refined dolls (<root>/d<NNN>/<name>.png); "
                         "when set, prints a baseline-vs-refined comparison")
    ap.add_argument("--refined-denoise", default="d055",
                    help="denoise subdir under --refined-root to compare")
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
        from collections import Counter
        modes = Counter(m for _, _, _, _, m, _ in scores)
        print(f"  modes: " + "  ".join(f"{m}={n}" for m, n in modes.items()))
        finite = [c for _, _, _, c, _, _ in scores if np.isfinite(c)]
        if finite:
            print(f"  median id_cos = {float(np.median(finite)):.3f}  "
                  f"(n={len(finite)} of {len(scores)} cells; "
                  f"{len(scores) - len(finite)} unrecoverable)")

    if not rows:
        print("[swap-test] no rows produced")
        return 1

    ncol = 1 + len(sources)
    col_labels = ["doll"] + [sid for sid, _, _ in sources]
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

    if args.refined_root is not None:
        print(f"\n[swap-test] baseline vs refined ({args.refined_denoise})")
        print(f"  {'arm':14s} {'set':9s} {'default_rate':>12s} {'median_cos':>11s}")
        for arm, renders in ARM_RENDERS.items():
            base_dolls, ref_dolls = [], []
            for rname in renders:
                bd = cv2.imread(str(args.root / "renders" / rname))
                rp = args.refined_root / args.refined_denoise / rname
                rd = cv2.imread(str(rp))
                if bd is not None:
                    base_dolls.append(bd)
                if rd is not None:
                    ref_dolls.append(rd)
            b_rate, b_med = _swap_stats(app, swapper, base_dolls, sources)
            r_rate, r_med = _swap_stats(app, swapper, ref_dolls, sources)
            print(f"  {arm:14s} {'baseline':9s} {b_rate:12.2f} {b_med:11.3f}")
            print(f"  {arm:14s} {'refined':9s} {r_rate:12.2f} {r_med:11.3f}")

    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(f"[swap-test] {out}  ({grid_w}x{grid_h})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
