"""Analyze start_percent sweep: drift + glasses prob + collage."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.demographic_pc.fluxspace_primary_metrics import (
    _arcface_model, _arc_embed, _clip_glasses_model, _clip_glasses_prob,
)
from src.demographic_pc.fluxspace_metrics import CROSS_DIR

OUT = CROSS_DIR / "startpct"
STARTS = [0.15, 0.20, 0.30, 0.40]

# Baseline (s=0) render per target
BASELINES = {
    "elderly_latin_m": CROSS_DIR / "verify" / "elderly_latin_m" / "s+0.00.png",
    "southasian_f":    CROSS_DIR / "verify" / "southasian_f"    / "s+0.00.png",
    # latin_f s=0 baseline is in the original pair_scale_sweep
    "latin_f": Path("output/demographic_pc/fluxspace_node_test/glasses/pair_scale_sweep/sweep_s+0.00_seed2026.png"),
}


def _font(sz=16):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",):
        if Path(p).exists():
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def main() -> None:
    arc, dev = _arcface_model()
    cm, cp, ctf, cdev = _clip_glasses_model()
    rows = {}
    for name, base_p in BASELINES.items():
        bdir = OUT / name
        if not bdir.exists():
            continue
        if not base_p.exists():
            print(f"[startpct] missing baseline {base_p}")
            continue
        base_emb = _arc_embed(arc, dev, Image.open(base_p))
        info = {}
        for sp in STARTS:
            p = bdir / f"start{sp:.2f}.png"
            if not p.exists():
                continue
            pil = Image.open(p)
            emb = _arc_embed(arc, dev, pil)
            drift = float(1.0 - np.dot(base_emb, emb))
            gp = _clip_glasses_prob(cm, cp, ctf, cdev, pil)
            info[sp] = {"drift": drift, "glasses_prob": gp}
        rows[name] = info

    with (OUT / "summary.json").open("w") as f:
        json.dump(rows, f, indent=2)

    print(f"{'target':<20} {'start':>6} {'drift':>8} {'P(glasses)':>12}")
    for name, info in rows.items():
        for sp, v in info.items():
            print(f"{name:<20} {sp:>6.2f} {v['drift']:>8.4f} {v['glasses_prob']:>12.3f}")

    # Collage: 3 rows (bases), 5 cols (baseline + 4 starts)
    CELL = 256; PAD = 6; LABEL = 36
    font = _font(16)
    rows_img = []
    for name, base_p in BASELINES.items():
        if name not in rows:
            continue
        cells = []
        b = Image.open(base_p).convert("RGB").resize((CELL, CELL), Image.Resampling.LANCZOS)
        cell = Image.new("RGB", (CELL + 2*PAD, CELL + LABEL + 2*PAD), (30, 30, 30))
        cell.paste(b, (PAD, PAD))
        ImageDraw.Draw(cell).text((PAD+4, CELL + PAD + 6), "baseline s=0", fill=(240,240,240), font=font)
        cells.append(cell)
        for sp in STARTS:
            p = OUT / name / f"start{sp:.2f}.png"
            img = Image.open(p).convert("RGB").resize((CELL, CELL), Image.Resampling.LANCZOS)
            cell = Image.new("RGB", (CELL + 2*PAD, CELL + LABEL + 2*PAD), (30, 30, 30))
            cell.paste(img, (PAD, PAD))
            v = rows[name].get(sp, {})
            t = f"sp={sp:.2f}  d={v.get('drift',0):.2f}  g={v.get('glasses_prob',0):.2f}"
            ImageDraw.Draw(cell).text((PAD+4, CELL + PAD + 6), t, fill=(240,240,240), font=font)
            cells.append(cell)
        w = sum(c.width for c in cells)
        h = cells[0].height + LABEL
        row = Image.new("RGB", (w, h), (10, 10, 10))
        ImageDraw.Draw(row).text((12, 8), name, fill=(255,255,255), font=_font(20))
        x = 0
        for c in cells:
            row.paste(c, (x, LABEL))
            x += c.width
        rows_img.append(row)
    if rows_img:
        W = max(r.width for r in rows_img)
        H = sum(r.height for r in rows_img)
        m = Image.new("RGB", (W, H), (0,0,0))
        y = 0
        for r in rows_img:
            m.paste(r, (0, y))
            y += r.height
        m.save(OUT / "collage.png")
        print(f"[startpct] wrote {OUT/'collage.png'}")


if __name__ == "__main__":
    main()
