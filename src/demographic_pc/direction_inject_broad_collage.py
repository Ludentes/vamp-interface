"""Per-atom scale-sweep collages from the broad validation run.

For each atom dir under `output/demographic_pc/direction_inject_broad/`,
build a (bases × scales) grid showing one seed's rendered output. Useful
for morning triage — find atoms that produce visible edits at low scale.

Auto-discovers atom ids and scale values from the filesystem.
"""

from __future__ import annotations

import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
BROAD = ROOT / "output/demographic_pc/direction_inject_broad"
OUT = BROAD / "_collages"

TILE = 200
PAD = 6
LABEL_L = 180
LABEL_T = 40

STEM_RE = re.compile(r"seed(?P<seed>\d+)_s(?P<scale>[+-]?\d+\.\d+)\.png")


def _font(sz: int):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def build_atom(atom_dir: Path, seed: int = 2026) -> Path | None:
    bases = sorted([d.name for d in atom_dir.iterdir() if d.is_dir()])
    if not bases:
        return None
    scales = set()
    for b in bases:
        for f in (atom_dir / b).glob("*.png"):
            m = STEM_RE.match(f.name)
            if m and int(m.group("seed")) == seed:
                scales.add(float(m.group("scale")))
    scales = sorted(scales)
    if not scales:
        return None

    W = LABEL_L + len(scales) * (TILE + PAD) + PAD
    H = LABEL_T + len(bases) * (TILE + PAD) + PAD
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    big, small = _font(16), _font(12)

    draw.text((PAD, 8), f"{atom_dir.name}  seed={seed}", fill="black", font=big)
    for ci, s in enumerate(scales):
        draw.text((LABEL_L + ci * (TILE + PAD) + TILE // 2 - 25,
                   LABEL_T - 18), f"s={s:+.2f}", fill="black", font=small)

    for ri, base in enumerate(bases):
        y = LABEL_T + ri * (TILE + PAD)
        draw.text((PAD, y + TILE // 2 - 8), base, fill="black", font=small)
        for ci, s in enumerate(scales):
            p = atom_dir / base / f"seed{seed}_s{s:+.2f}.png"
            x = LABEL_L + ci * (TILE + PAD)
            if p.exists():
                im = Image.open(p).convert("RGB").resize((TILE, TILE))
                canvas.paste(im, (x, y))
            else:
                draw.rectangle([x, y, x + TILE, y + TILE], outline="red")
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"{atom_dir.name}_seed{seed}.png"
    canvas.save(dest)
    return dest


def main():
    atom_dirs = sorted(d for d in BROAD.iterdir()
                       if d.is_dir() and d.name.startswith("atom"))
    for ad in atom_dirs:
        out = build_atom(ad)
        if out:
            print(f"[collage] {ad.name} → {out}")


if __name__ == "__main__":
    main()
