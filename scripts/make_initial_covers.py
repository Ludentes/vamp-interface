"""Build covers for the three initial blog posts from existing artifacts.

- Part 1 (faces-as-data-mining): 10x5 mosaic of Stage-1 portraits.
- Part 2 (math-of-pattern-preservation): Ours-row single-portrait age sweep
  from Stage 4.5 (7 panels, lambda in {-3, -2, -1, 0, +1, +2, +3}).
- Perception-before-training: 8x8 glance-grid; three tiles boxed as anomalies
  to evoke the Level-5 target of the curriculum.

All covers written under docs/blog/images/.
"""

from pathlib import Path
from random import Random

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
STAGE1 = ROOT / "output" / "demographic_pc" / "stage1" / "samples"
STAGE4_5 = ROOT / "output" / "demographic_pc" / "stage4_5" / "renders"
OUT_DIR = ROOT / "docs" / "blog" / "images"


def square_crop(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    return img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))


def portrait_crop(img: Image.Image, aspect: float = 4 / 3) -> Image.Image:
    """Crop to target aspect (h/w), centered."""
    w, h = img.size
    target_h = int(w * aspect)
    if target_h <= h:
        top = (h - target_h) // 2
        return img.crop((0, top, w, top + target_h))
    target_w = int(h / aspect)
    left = (w - target_w) // 2
    return img.crop((left, 0, left + target_w, h))


def part1_cover() -> None:
    """10x5 mosaic of all 50 Stage-1 portraits. Deterministic order by filename
    so the cover is reproducible; shuffled once with a fixed seed so the grid
    reads as diverse at a glance rather than sorted by demographic indices.
    """
    files = sorted(STAGE1.glob("*.png"))
    assert len(files) == 50, f"expected 50 Stage-1 samples, got {len(files)}"
    rng = Random(17)
    rng.shuffle(files)

    tile_w, tile_h = 192, 256
    cols, rows = 10, 5
    gap = 3
    W = cols * tile_w + (cols - 1) * gap
    H = rows * tile_h + (rows - 1) * gap
    canvas = Image.new("RGB", (W, H), (18, 18, 20))

    for i, p in enumerate(files):
        img = Image.open(p).convert("RGB")
        img = portrait_crop(img, aspect=tile_h / tile_w)
        img = img.resize((tile_w, tile_h), Image.LANCZOS)
        col, row = i % cols, i // cols
        x = col * (tile_w + gap)
        y = row * (tile_h + gap)
        canvas.paste(img, (x, y))

    out = OUT_DIR / "2026-04-16-part-1-faces-as-data-mining-cover.png"
    canvas.save(out, optimize=True)
    print(f"wrote {out} at {canvas.size}")


def part2_cover() -> None:
    """Single portrait, Ours row, age sweep. Same portrait p03_0_4_s2003 as the
    Stage 4.5 cover but only the ridge row, drawing the point of Part 2:
    continuous, identity-preserving transformation along a measured axis.
    """
    portrait = "p03_0_4_s2003"
    lams = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    tile_w, tile_h = 256, 342
    gap = 4
    W = len(lams) * tile_w + (len(lams) - 1) * gap
    H = tile_h
    canvas = Image.new("RGB", (W, H), (18, 18, 20))

    for j, lam in enumerate(lams):
        if lam == 0.0:
            p = STAGE4_5 / "baseline" / f"{portrait}__lam+0.00.png"
        else:
            p = STAGE4_5 / "ours" / f"{portrait}__lam{lam:+.2f}.png"
        img = Image.open(p).convert("RGB").resize((tile_w, tile_h), Image.LANCZOS)
        x = j * (tile_w + gap)
        canvas.paste(img, (x, 0))

    out = OUT_DIR / "2026-04-16-part-2-math-of-pattern-preservation-cover.png"
    canvas.save(out, optimize=True)
    print(f"wrote {out} at {canvas.size}")


def perception_cover() -> None:
    """8x8 glance grid of 64 portraits, three boxed as anomalies. Tiles are
    small and roughly uniform so the eye has to sweep — exactly the Level-5
    regime the curriculum is aiming at.
    """
    files = list(sorted(STAGE1.glob("*.png")))  # 50
    baseline = STAGE4_5 / "baseline"
    files += sorted(baseline.glob("*.png"))[:14]
    rng = Random(42)
    rng.shuffle(files)
    files = files[:64]

    tile = 130
    gap = 2
    cols = rows = 8
    W = cols * tile + (cols - 1) * gap
    H = rows * tile + (rows - 1) * gap
    canvas = Image.new("RGB", (W, H), (18, 18, 20))

    anomaly_indices = {9, 31, 52}
    for i, p in enumerate(files):
        img = Image.open(p).convert("RGB")
        img = square_crop(img).resize((tile, tile), Image.LANCZOS)
        col, row = i % cols, i // cols
        x = col * (tile + gap)
        y = row * (tile + gap)
        canvas.paste(img, (x, y))

    draw = ImageDraw.Draw(canvas)
    for i in anomaly_indices:
        col, row = i % cols, i // cols
        x = col * (tile + gap)
        y = row * (tile + gap)
        draw.rectangle(
            [x, y, x + tile - 1, y + tile - 1],
            outline=(230, 80, 60),
            width=3,
        )

    out = OUT_DIR / "2026-04-20-perception-before-training-cover.png"
    canvas.save(out, optimize=True)
    print(f"wrote {out} at {canvas.size}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    part1_cover()
    part2_cover()
    perception_cover()


if __name__ == "__main__":
    main()
