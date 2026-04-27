"""Collage for cross-demo verification: one row per base, one cell per scale.

For each base prompt:
  - measurement render (s=1, labelled "meas")
  - verification sweep at straddle scales (from predictions.json)
Annotate edges with [lo] and [hi] markers, scale values under each cell.

Usage:
    uv run python -m src.demographic_pc.fluxspace_crossdemo_collage
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
CROSS_ROOT = ROOT / "output" / "demographic_pc" / "fluxspace_metrics" / "crossdemo"
# Overridden per axis in main()
CROSS = CROSS_ROOT
OUT = CROSS / "collages"


def _axis_root(axis: str) -> Path:
    return CROSS_ROOT if axis == "glasses" else CROSS_ROOT / axis
CELL = 256   # thumbnail size
PAD = 6
LABEL_H = 32


def _font(size: int = 16):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _thumb(path: Path) -> Image.Image:
    im = Image.open(path).convert("RGB").resize((CELL, CELL), Image.Resampling.LANCZOS)
    return im


def _cell(img: Image.Image, label: str, highlight: str | None = None) -> Image.Image:
    out = Image.new("RGB", (CELL + 2*PAD, CELL + LABEL_H + 2*PAD), (20, 20, 20))
    if highlight == "edge":
        border = Image.new("RGB", (CELL + 2*PAD, CELL + LABEL_H + 2*PAD), (255, 180, 0))
        out.paste(border, (0, 0))
    elif highlight == "meas":
        border = Image.new("RGB", (CELL + 2*PAD, CELL + LABEL_H + 2*PAD), (40, 140, 220))
        out.paste(border, (0, 0))
    out.paste(img, (PAD, PAD))
    d = ImageDraw.Draw(out)
    d.text((PAD + 4, CELL + PAD + 4), label, fill=(240, 240, 240), font=_font(16))
    return out


def build_row(base: str, predictions: dict) -> Image.Image:
    info = predictions[base]
    s_lo, s_hi = info["s_lo"], info["s_hi"]
    scales = info["scales"]

    cells: list[Image.Image] = []

    meas_p = CROSS / "measurement" / f"{base}_meas.png"
    if meas_p.exists():
        cells.append(_cell(_thumb(meas_p), f"meas s=+1.00", highlight="meas"))

    for s in scales:
        p = CROSS / "verify" / base / f"s{s:+.2f}.png"
        if not p.exists():
            ph = Image.new("RGB", (CELL, CELL), (60, 30, 30))
            d = ImageDraw.Draw(ph)
            d.text((20, CELL//2 - 10), "missing", fill=(220, 100, 100), font=_font(16))
            cells.append(_cell(ph, f"s={s:+.2f}"))
            continue
        highlight = None
        if s_lo is not None and abs(s - s_lo) < 0.025:
            highlight = "edge"
        if s_hi is not None and abs(s - s_hi) < 0.025:
            highlight = "edge"
        label = f"s={s:+.2f}"
        if highlight == "edge":
            which = "lo" if s_lo is not None and abs(s - s_lo) < 0.025 else "hi"
            label += f" [{which}]"
        cells.append(_cell(_thumb(p), label, highlight=highlight))

    cw, ch = cells[0].size
    row = Image.new("RGB", (cw * len(cells), ch + LABEL_H + 2*PAD), (10, 10, 10))
    for i, c in enumerate(cells):
        row.paste(c, (i*cw, LABEL_H + 2*PAD))
    d = ImageDraw.Draw(row)
    lo = f"{s_lo:+.2f}" if s_lo is not None else "?"
    hi = f"{s_hi:+.2f}" if s_hi is not None else "?"
    title = f"{base}    predicted safe ∈ [{lo}, {hi}]    base_max_env={info.get('base_max_env',0):.2f}  T_ratio={info.get('T_ratio',0):.3f}"
    d.text((12, 8), title, fill=(240, 240, 240), font=_font(18))
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", default="glasses", choices=["glasses", "smile"])
    args = ap.parse_args()
    global CROSS, OUT
    CROSS = _axis_root(args.axis)
    OUT = CROSS / "collages"
    pj = CROSS / "predictions.json"
    if not pj.exists():
        print(f"missing {pj} — run --crossdemo-verify first")
        return
    predictions = json.loads(pj.read_text())
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for base in predictions:
        print(f"[collage] {base}")
        row = build_row(base, predictions)
        row.save(OUT / f"{base}.png")
        rows.append(row)

    if rows:
        max_w = max(r.width for r in rows)
        total_h = sum(r.height for r in rows)
        master = Image.new("RGB", (max_w, total_h), (0, 0, 0))
        y = 0
        for r in rows:
            master.paste(r, (0, y))
            y += r.height
        master.save(OUT / "all_bases.png")
        print(f"[collage] wrote {OUT/'all_bases.png'}")


if __name__ == "__main__":
    main()
