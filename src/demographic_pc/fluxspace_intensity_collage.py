"""Assemble intensity-sweep grids for visual inspection.

For each base and each start_pct, build a 4×5 collage (rows=B-ladder faint→manic,
cols=scale 0.4→2.1). Labels on top row (scale) and left column (ladder).
Outputs under intensity/collages/.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.demographic_pc.fluxspace_intensity_sweep import (
    BASES, LADDER, ROOT, SCALES, START_PCTS,
)

TILE = 320
PAD = 10
LABEL_COL = 170  # left column width for ladder labels
LABEL_ROW = 50   # top row height for scale labels

OUT = ROOT / "collages"


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def build_grid(base_name: str, sp: float) -> Path | None:
    n_rows = len(LADDER)
    n_cols = len(SCALES)
    W = LABEL_COL + n_cols * (TILE + PAD) + PAD
    H = LABEL_ROW + n_rows * (TILE + PAD) + PAD
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    big = _font(22)
    small = _font(18)

    draw.text((PAD, PAD),
              f"{base_name}  start_percent={sp:.2f}  (rows: faint→manic, cols: s=0.4→2.1)",
              fill="black", font=big)

    # Column headers
    for ci, s in enumerate(SCALES):
        x = LABEL_COL + ci * (TILE + PAD)
        draw.text((x + TILE // 2 - 30, LABEL_ROW - 28), f"s={s:+.2f}",
                  fill="black", font=small)

    any_found = False
    for ri, (ladder_id, _) in enumerate(LADDER):
        y = LABEL_ROW + ri * (TILE + PAD)
        draw.text((PAD, y + TILE // 2 - 10), ladder_id, fill="black", font=small)
        for ci, s in enumerate(SCALES):
            src = ROOT / base_name / ladder_id / f"sp{sp:.2f}_s{s:+.2f}.png"
            x = LABEL_COL + ci * (TILE + PAD)
            if not src.exists():
                draw.rectangle([x, y, x + TILE, y + TILE], outline="red")
                draw.text((x + 10, y + 10), "missing", fill="red", font=small)
                continue
            any_found = True
            im = Image.open(src).convert("RGB").resize((TILE, TILE), Image.Resampling.LANCZOS)
            canvas.paste(im, (x, y))

    if not any_found:
        return None
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"{base_name}_sp{sp:.2f}.png"
    canvas.save(dest)
    return dest


def main() -> None:
    for base_name, _, _ in BASES:
        for sp in START_PCTS:
            out = build_grid(base_name, sp)
            if out:
                print(f"[collage] {out}")


if __name__ == "__main__":
    main()
