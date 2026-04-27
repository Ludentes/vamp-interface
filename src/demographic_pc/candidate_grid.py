"""Render labeled grids of candidate slider training pairs (anchor | edited).

Reads models/flux_sliders/candidate_balanced.parquet and emits several
sub-grids so the pairs stay readable when the image is viewed at model
resolution (~1568 px max).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
CANDIDATES = ROOT / "models/flux_sliders/candidate_balanced.parquet"
OUT_DIR = ROOT / "models/flux_sliders"

TILE = 400
GAP = 8
LABEL_H = 64
PER_PAGE = 8


def anchor_path(img_path: str) -> Path:
    p = Path(img_path)
    stem = p.stem
    seed_part, _ = stem.split("_a")
    return p.with_name(f"{seed_part}_a0.00.png")


def _font(size):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_page(rows: pd.DataFrame, out_path: Path):
    n = len(rows)
    cols = 2  # 2 pairs per row
    grid_rows = (n + cols - 1) // cols
    pair_w = 2 * TILE + GAP
    pair_h = TILE + LABEL_H
    w = cols * pair_w + (cols + 1) * GAP
    h = grid_rows * pair_h + (grid_rows + 1) * GAP
    img = Image.new("RGB", (w, h), "black")
    draw = ImageDraw.Draw(img)
    f_big = _font(22)
    f_sm = _font(18)
    for i, (_, row) in enumerate(rows.iterrows()):
        c = i % cols
        r = i // cols
        x0 = GAP + c * (pair_w + GAP)
        y0 = GAP + r * (pair_h + GAP)
        apath = ROOT / anchor_path(str(row["img_path"]))
        epath = ROOT / str(row["img_path"])
        for j, p in enumerate([apath, epath]):
            tile = (Image.open(p).convert("RGB").resize((TILE, TILE), Image.Resampling.LANCZOS)
                    if p.exists() else Image.new("RGB", (TILE, TILE), "gray"))
            img.paste(tile, (x0 + j * (TILE + GAP), y0))
        ly = y0 + TILE + 4
        draw.text((x0 + 6, ly),
                  f"[{int(row['global_idx']):02d}] {row['base']}",
                  fill="white", font=f_big)
        draw.text((x0 + 6, ly + 28),
                  f"ed={row['edit_effect']:.2f}  id={row['identity_cos_to_base']:.2f}  "
                  f"Δage={row['dage']:+.1f}y  Δrace={row['drace']:.3f}",
                  fill="#cccccc", font=f_sm)
    img.save(out_path, "PNG")


def main():
    df = pd.read_parquet(CANDIDATES).reset_index(drop=True)
    df["global_idx"] = df.index
    pages = [df.iloc[i : i + PER_PAGE] for i in range(0, len(df), PER_PAGE)]
    for pi, page in enumerate(pages):
        out = OUT_DIR / f"candidate_page_{pi+1}.png"
        render_page(page, out)
        print(f"[saved] {out}  ({len(page)} pairs)")


if __name__ == "__main__":
    main()
