"""Beard-axis inspection collage. Three rows (one per base cell we actually
rendered), four columns (scales 0.00, 0.40, 0.70, 1.00), same seed per row for
readability. Saves at output/demographic_pc/overnight_drift/beard_collage.png.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
BEARD = ROOT / "output/demographic_pc/overnight_drift/beard"
OUT = ROOT / "output/demographic_pc/overnight_drift/beard_collage.png"

ROWS = [
    ("add",    "asian_m",         "add → asian_m"),
    ("add",    "european_m",      "add → european_m"),
    ("remove", "elderly_latin_m", "remove → elderly_latin_m"),
    ("../beard_rebalance/remove", "asian_m_bearded",         "rebal remove → asian_m_bearded"),
    ("../beard_rebalance/remove", "european_m_bearded",      "rebal remove → european_m_bearded"),
    ("../beard_rebalance/remove", "elderly_latin_m_bearded", "rebal remove → elderly_latin_m_bearded"),
]
SCALES = ["+0.00", "+0.40", "+0.70", "+1.00"]
SEED = 2026
TILE = 320
PAD = 8
LABEL_H = 42


def main():
    font = None
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    W = PAD + len(SCALES) * (TILE + PAD)
    H = LABEL_H + PAD + len(ROWS) * (TILE + PAD + LABEL_H)
    canvas = Image.new("RGB", (W, H), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    # Column header row (scales)
    for ci, s in enumerate(SCALES):
        x = PAD + ci * (TILE + PAD)
        draw.text((x + TILE // 2 - 22, 8), f"scale={s}", fill=(30, 30, 30), font=font)

    y = LABEL_H + PAD
    for polarity, base, title in ROWS:
        draw.text((PAD, y), title, fill=(0, 0, 90), font=font)
        y += LABEL_H - 10
        for ci, s in enumerate(SCALES):
            png = BEARD / polarity / base / f"seed{SEED}_s{s}.png"
            if not png.exists():
                x = PAD + ci * (TILE + PAD)
                draw.rectangle([x, y, x + TILE, y + TILE], outline=(200, 0, 0), width=2)
                draw.text((x + 10, y + TILE // 2), f"MISSING\n{png.name}",
                          fill=(200, 0, 0), font=font)
                continue
            im = Image.open(png).convert("RGB").resize((TILE, TILE), Image.LANCZOS)
            canvas.paste(im, (PAD + ci * (TILE + PAD), y))
        y += TILE + PAD

    canvas.save(OUT)
    print(f"[save] → {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
