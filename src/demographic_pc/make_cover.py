"""Build a 3x2 mosaic cover image from diverse Stage 1 outputs.

Pick six faces spanning age levels and demographics, center-crop to square,
resize, arrange at 1200x800 landscape.
"""

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "output" / "demographic_pc" / "stage1" / "samples"
OUT = ROOT / "docs" / "blog" / "images" / "2026-04-20-demographic-pc-sanity-check-cover.png"

# age-gender-eth spread across the grid (sample_id encodes age_idx-gender_idx-eth_idx)
PICKS = [
    "0-1-0-s101000",   # child / woman / East Asian
    "1-0-4-s110400",   # young adult / man / White
    "2-1-3-s121300",   # adult / woman / Black
    "3-1-6-s131600",   # middle-aged / woman / Middle Eastern
    "1-2-2-s112200",   # young adult / non-binary / South Asian
    "4-0-0-s140000",   # elderly / man / East Asian
]

TILE = 400          # each tile
COLS, ROWS = 3, 2
GAP = 4             # px gap between tiles (hairline)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    W = COLS * TILE + (COLS - 1) * GAP
    H = ROWS * TILE + (ROWS - 1) * GAP
    canvas = Image.new("RGB", (W, H), (20, 20, 22))

    for i, sid in enumerate(PICKS):
        p = SAMPLES / f"{sid}.png"
        img = Image.open(p).convert("RGB")
        w, h = img.size
        s = min(w, h)
        img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        img = img.resize((TILE, TILE), Image.LANCZOS)
        col, row = i % COLS, i // COLS
        x = col * (TILE + GAP)
        y = row * (TILE + GAP)
        canvas.paste(img, (x, y))

    canvas.save(OUT, optimize=True)
    print(f"Wrote {OUT} at {canvas.size}")


if __name__ == "__main__":
    main()
