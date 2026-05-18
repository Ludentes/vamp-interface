#!/usr/bin/env python3
"""Time-vs-quality montage for the matryoshka fast-model bake-off.

One row per arm, columns sorted by step count. Each tile is a representative
render (first sampler, lowest seed, Canny 0.4/0.4 for canny arms) labelled with
its step count and warm server-side generation time. A caption strip under each
row gives the arm's median warm gen time and cold-load cost.

Usage:
    python scripts/matryoshka_bakeoff_montage.py \\
        --root exp_output/matryoshka_bakeoff \\
        --out  exp_output/matryoshka_bakeoff/montage.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ARM_ORDER = ["flux_krea", "flux_schnell", "sdxl_lightning", "zimage_turbo"]
TILE = 360  # render thumbnail edge (px)
PAD = 12
LABEL_H = 28
CAPTION_H = 30
HEADER_H = 34


def _font(size: int) -> ImageFont.FreeTypeFont:
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def pick(sub: pd.DataFrame) -> pd.Series:
    """Representative cell within an (arm, steps) group."""
    s = sub.copy()
    if s["cn_strength"].notna().any():
        near = s[(s["cn_strength"] == 0.4) & (s["cn_end"] == 0.4)]
        if len(near):
            s = near
    return s.sort_values(["sampler", "scheduler", "seed"]).iloc[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("exp_output/matryoshka_bakeoff"))
    ap.add_argument("--out", type=Path,
                    default=Path("exp_output/matryoshka_bakeoff/montage.png"))
    args = ap.parse_args()

    df = pd.read_parquet(args.root / "manifest.parquet")
    ok = df[(df.status == "ok") & df.steps.notna()].copy()
    ok["is_cold"] = ok["is_cold"].astype(bool)
    renders = args.root / "renders"

    arms = [a for a in ARM_ORDER if a in ok.arm.unique()]
    max_cols = max(ok.groupby("arm").steps.nunique())

    grid_w = PAD + max_cols * (TILE + PAD)
    row_h = HEADER_H + LABEL_H + TILE + CAPTION_H + PAD
    grid_h = PAD + len(arms) * row_h

    canvas = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(canvas)
    f_hdr, f_lbl, f_cap = _font(20), _font(15), _font(14)

    for ri, arm in enumerate(arms):
        sub = ok[ok.arm == arm]
        y0 = PAD + ri * row_h
        warm = sub[~sub.is_cold]
        med = warm.comfy_exec_s.median()
        cold = sub[sub.is_cold].comfy_exec_s
        cold_s = f"{cold.min():.0f}s" if len(cold) else "n/a"
        draw.text((PAD, y0), arm, fill="black", font=f_hdr)

        steps_vals = sorted(sub.steps.unique())
        for ci, st in enumerate(steps_vals):
            row = pick(sub[sub.steps == st])
            x0 = PAD + ci * (TILE + PAD)
            y_img = y0 + HEADER_H + LABEL_H
            img_path = renders / row["render"]
            if img_path.exists():
                im = Image.open(img_path).convert("RGB")
                im.thumbnail((TILE, TILE))
                ox = x0 + (TILE - im.width) // 2
                canvas.paste(im, (ox, y_img))
            else:
                draw.rectangle([x0, y_img, x0 + TILE, y_img + TILE], outline="red")
                draw.text((x0 + 8, y_img + 8), "MISSING", fill="red", font=f_lbl)

            draw.text((x0, y0 + HEADER_H),
                      f"{int(st)} steps  ·  {row['comfy_exec_s']:.1f}s"
                      + ("  (cold)" if row["is_cold"] else ""),
                      fill="black", font=f_lbl)

        draw.text((PAD, y0 + HEADER_H + LABEL_H + TILE + 6),
                  f"median warm gen {med:.1f}s   ·   cold load {cold_s}   "
                  f"·   n={len(sub)} cells",
                  fill="#555555", font=f_cap)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(f"[montage] {args.out}  ({grid_w}x{grid_h})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
