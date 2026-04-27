"""Quick visual inspection of the 2026-04-22 rebalance corpus.

One PNG per axis: rows = 6 bases, cols = 5 scales at seed=2026 sp=0.15.
Goal: eyeball whether (anger, surprise, disgust, pucker, lip_press) actually
rendered as the intended expression across demographics and scales.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.demographic_pc.render_expression_corpus import (
    AXES, BASES, SCALES, _axis_root,
)

SEED = 2026
SP = 0.15
TILE = 256
PAD = 8
LABEL_COL = 170
LABEL_ROW = 48

OUT = Path(__file__).resolve().parents[2] / "output/demographic_pc/fluxspace_metrics/crossdemo/collages/rebalance"


def _font(size: int):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def build_axis(axis: str) -> Path:
    root = _axis_root(axis)
    n_rows, n_cols = len(BASES), len(SCALES)
    W = LABEL_COL + n_cols * (TILE + PAD) + PAD
    H = LABEL_ROW + n_rows * (TILE + PAD) + PAD
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    big, small = _font(20), _font(15)
    draw.text((PAD, PAD // 2), f"{axis}  seed={SEED}  sp={SP:.2f}  "
              f"rows: bases  cols: s=0.4→2.0", fill="black", font=big)
    for ci, s in enumerate(SCALES):
        draw.text((LABEL_COL + ci * (TILE + PAD) + TILE // 2 - 30,
                   LABEL_ROW - 20), f"s={s:+.2f}", fill="black", font=small)
    for ri, (base, _, _) in enumerate(BASES):
        y = LABEL_ROW + ri * (TILE + PAD)
        draw.text((PAD, y + TILE // 2 - 8), base, fill="black", font=small)
        for ci, s in enumerate(SCALES):
            stem = f"seed{SEED}_sp{SP:.2f}_s{s:+.2f}"
            png = root / base / f"{stem}.png"
            x = LABEL_COL + ci * (TILE + PAD)
            if png.exists():
                im = Image.open(png).convert("RGB").resize((TILE, TILE))
                canvas.paste(im, (x, y))
            else:
                draw.rectangle([x, y, x + TILE, y + TILE], outline="red")
                draw.text((x + 20, y + TILE // 2), "MISSING", fill="red", font=small)
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"{axis}.png"
    canvas.save(dest)
    return dest


def main() -> None:
    for axis in AXES:
        p = build_axis(axis)
        print(f"[collage] {axis} → {p}")


if __name__ == "__main__":
    main()
