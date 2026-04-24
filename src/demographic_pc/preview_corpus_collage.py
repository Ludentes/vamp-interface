"""Preview collage for corpus inspection — grid of raw rendered PNGs.

No model load — just stitches rendered PNGs from the crossdemo corpus
into a labeled grid (rows=bases, cols=α). Used to eyeball whether a
newly-rendered corpus looks sensible before spending compute on
ArcFace scoring.

Usage:
    uv run python src/demographic_pc/preview_corpus_collage.py \\
        --root output/demographic_pc/fluxspace_metrics/crossdemo_v2 \\
        --axis eye_squint \\
        --seed 2026 \\
        --out models/flux_sliders/collages/preview_eye_squint_v2.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DEFAULT_BASES = [
    "adult_latin_f", "adult_asian_m", "adult_black_f", "adult_european_m",
    "elderly_latin_m", "young_european_f", "elderly_asian_f",
    "young_black_m", "adult_middle_f", "adult_southasian_f",
]
DEFAULT_ALPHAS = [0.0, 0.10, 0.20, 0.30, 0.40]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--axis", required=True)
    p.add_argument("--subdir", default=None,
                   help="default: <axis>_inphase")
    p.add_argument("--bases", nargs="+", default=DEFAULT_BASES)
    p.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--thumb", type=int, default=256,
                   help="thumbnail size in pixels")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    subdir = args.subdir or f"{args.axis}_inphase"
    root = Path(args.root) / args.axis / subdir

    thumb = args.thumb
    pad = 6
    label_h = 26
    title_h = 36
    row_label_w = 200

    rows, cols = len(args.bases), len(args.alphas)
    W = row_label_w + cols * thumb + (cols + 1) * pad
    H = title_h + label_h + rows * thumb + (rows + 1) * pad

    canvas = Image.new("RGB", (W, H), (18, 18, 22))
    draw = ImageDraw.Draw(canvas)

    try:
        font_big = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        font_big = ImageFont.load_default()
        font = ImageFont.load_default()

    title = f"{args.axis} preview — seed={args.seed} — {root}"
    draw.text((pad * 2, 8), title, fill=(235, 235, 235), font=font_big)

    # Column labels (α)
    for ci, a in enumerate(args.alphas):
        x = row_label_w + ci * (thumb + pad) + pad
        draw.text((x, title_h + 4), f"α = {a:.2f}", fill=(210, 210, 210), font=font)

    missing = 0
    for ri, base in enumerate(args.bases):
        y = title_h + label_h + ri * (thumb + pad) + pad
        draw.text((pad * 2, y + thumb // 2 - 8), base,
                  fill=(210, 210, 210), font=font)
        for ci, a in enumerate(args.alphas):
            x = row_label_w + ci * (thumb + pad) + pad
            path = root / base / f"s{args.seed}_a{a:.2f}.png"
            if not path.exists():
                missing += 1
                draw.rectangle([x, y, x + thumb, y + thumb], fill=(40, 20, 20), outline=(120, 60, 60))
                draw.text((x + 6, y + 6), "missing", fill=(200, 120, 120), font=font)
                continue
            img = Image.open(path).convert("RGB")
            if img.size != (thumb, thumb):
                img = img.resize((thumb, thumb), Image.Resampling.LANCZOS)
            canvas.paste(img, (x, y))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(out))
    print(f"[collage] wrote {out} ({rows}×{cols}={rows*cols} cells, {missing} missing)")


if __name__ == "__main__":
    main()
