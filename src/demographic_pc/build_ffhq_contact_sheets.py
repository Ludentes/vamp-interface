"""Render FFHQ kept pairs as dense contact sheets for visual triage.

Each sheet is a multi-column grid of [POS|NEG] tiles with the global
rank index as a header so individual pairs can be referenced back to
the manifest. Sorted by J descending so the highest-confidence pairs
appear first.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "output/squint_path_b/pair_manifest.parquet"
IMG_DIR = REPO / "output/ffhq_images"
OUT_DIR = REPO / "output/squint_path_b/contact_sheets"

THUMB = 128
COLS = 4
ROWS = 8
PER_SHEET = COLS * ROWS
HEADER_H = 22


def get_font(size: int = 11) -> ImageFont.ImageFont:
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def load_thumb(sha: str) -> Image.Image:
    p = IMG_DIR / f"{sha}.png"
    if not p.exists():
        return Image.new("RGB", (THUMB, THUMB), (40, 0, 0))
    return Image.open(p).convert("RGB").resize((THUMB, THUMB), Image.Resampling.LANCZOS)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    m = pd.read_parquet(MANIFEST)
    if "source" not in m.columns:
        m["source"] = "ffhq"
    ffhq = m[(m["kept"]) & (m["source"] == "ffhq")].copy()
    ffhq = ffhq.sort_values("J", ascending=False).reset_index(drop=True)
    ffhq["global_id"] = ffhq.index
    print(f"[ffhq] {len(ffhq)} kept pairs")

    font = get_font(11)
    font_h = get_font(10)

    pair_w = THUMB * 2 + 4
    pair_h = THUMB + HEADER_H + 6
    sheet_w = COLS * pair_w + (COLS + 1) * 6
    sheet_h = ROWS * pair_h + (ROWS + 1) * 4

    n_sheets = (len(ffhq) + PER_SHEET - 1) // PER_SHEET
    for s in range(n_sheets):
        page = ffhq.iloc[s * PER_SHEET : (s + 1) * PER_SHEET]
        canvas = Image.new("RGB", (sheet_w, sheet_h), (16, 16, 16))
        draw = ImageDraw.Draw(canvas)
        for i, (_, r) in enumerate(page.iterrows()):
            col = i % COLS
            row = i // COLS
            x0 = 6 + col * (pair_w + 6)
            y0 = 4 + row * (pair_h + 4)
            pos = load_thumb(r["sha_pos"])
            neg = load_thumb(r["sha_neg"])
            canvas.paste(pos, (x0, y0 + HEADER_H))
            canvas.paste(neg, (x0 + THUMB + 4, y0 + HEADER_H))
            cell = f'{r["ff_race"][:8]}/{r["ff_gender"]}/{r["ff_age_bin"]}'
            header = (f'#{int(r["global_id"]):03d} {cell}  '
                      f'Δθ={r["abs_dtheta"]:.2f} J={r["J"]:.3f} '
                      f'arc={r["arc_cos"]:.2f}')
            draw.text((x0 + 2, y0 + 4), header, fill=(220, 220, 220), font=font)
            draw.text((x0 + 2, y0 + HEADER_H + THUMB - 14),
                      "POS", fill=(255, 200, 80), font=font_h)
            draw.text((x0 + THUMB + 6, y0 + HEADER_H + THUMB - 14),
                      "NEG", fill=(120, 220, 255), font=font_h)

        out = OUT_DIR / f"sheet_{s:02d}.png"
        canvas.save(out, format="PNG", optimize=True)
        rng = f"#{s*PER_SHEET}–#{min((s+1)*PER_SHEET-1, len(ffhq)-1)}"
        print(f"  {out.name} ({rng}, {canvas.size[0]}x{canvas.size[1]}, "
              f"{out.stat().st_size/1e6:.1f} MB)")

    # also dump a CSV index so I can map global_id → sha later
    ffhq[["global_id", "sha_pos", "sha_neg", "ff_race", "ff_gender",
          "ff_age_bin", "abs_dtheta", "J", "arc_cos"]].to_csv(
        OUT_DIR / "index.csv", index=False
    )
    print(f"[done] {n_sheets} sheets + index.csv in {OUT_DIR}")


if __name__ == "__main__":
    main()
