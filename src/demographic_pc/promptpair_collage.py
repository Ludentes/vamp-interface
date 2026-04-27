"""Collage for a promptpair_iterate iteration.

Rows = variants; columns = scales (from spec.yaml's EVAL_SCALES).
One collage per (base, seed) combo specified.

Usage:
    uv run python -m src.demographic_pc.promptpair_collage \
        --axis smile --iter 01 --base young_european_f --seed 2026
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "output/demographic_pc/promptpair_iterate"

TILE = 320
PAD = 8
LABEL_H = 40


def build(axis: str, iter_label: str, base: str, seed: int) -> Path:
    iter_dir = OUT_ROOT / axis / f"iter_{iter_label}"
    spec = yaml.safe_load((iter_dir / "spec.yaml").read_text())
    variants = list(spec["variants"].keys())
    scales = spec.get("scales", [0.0, 0.5, 1.0])

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    W = PAD + 180 + PAD + len(scales) * (TILE + PAD)   # left label column
    H = LABEL_H + PAD + len(variants) * (TILE + PAD)
    canvas = Image.new("RGB", (W, H), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    # column headers
    for ci, s in enumerate(scales):
        x = PAD + 180 + PAD + ci * (TILE + PAD)
        draw.text((x + TILE // 2 - 30, 8), f"scale={s:+.1f}", fill=(30, 30, 30), font=font)

    # rows
    y = LABEL_H + PAD
    for vi, vkey in enumerate(variants):
        # left label
        draw.text((PAD, y + 4), vkey, fill=(0, 0, 90), font=font)
        pos = spec["variants"][vkey]["pos"]
        draw.text((PAD, y + 24), pos[:28], fill=(60, 60, 60), font=font)
        if len(pos) > 28:
            draw.text((PAD, y + 44), pos[28:56], fill=(60, 60, 60), font=font)

        for ci, s in enumerate(scales):
            png = iter_dir / vkey / base / f"seed{seed}_s{s:+.2f}.png"
            x = PAD + 180 + PAD + ci * (TILE + PAD)
            if not png.exists():
                draw.rectangle([x, y, x + TILE, y + TILE], outline=(200, 0, 0), width=2)
                draw.text((x + 10, y + TILE // 2), "MISSING", fill=(200, 0, 0), font=font)
                continue
            im = Image.open(png).convert("RGB").resize((TILE, TILE), Image.LANCZOS)
            canvas.paste(im, (x, y))
        y += TILE + PAD

    out = iter_dir / f"collage_{base}_seed{seed}.png"
    canvas.save(out)
    print(f"[save] → {out}  ({out.stat().st_size / 1024:.0f} KB)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", required=True)
    ap.add_argument("--iter", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    build(args.axis, args.iter, args.base, args.seed)


if __name__ == "__main__":
    main()
