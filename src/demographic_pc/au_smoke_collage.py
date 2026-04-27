"""4 atoms × 2 bases × 5 scales collage for the AU-library smoke test.

Rows: atom (smile, brow_lift, brow_furrow, eye_squint)
Cols: scale sweep, with base vs young_european_f stacked 2-high per scale.

Output: docs/research/images/2026-04-23-au-smoke-collage.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "output/demographic_pc/au_library_smoke"
OUT = ROOT / "docs/research/images/2026-04-23-au-smoke-collage.png"

ATOMS = ["smile", "brow_lift", "brow_furrow", "eye_squint"]
BASES = ["elderly_latin_m", "young_european_f"]
SCALES = [0, 100, 1000, 5000, -5000]

def fmt_scale(s):
    return f"{s:+d}" if s != 0 else "+0"

def main():
    rows = len(ATOMS) * len(BASES)
    cols = len(SCALES)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.4),
                             constrained_layout=True)
    for ai, atom in enumerate(ATOMS):
        for bi, base in enumerate(BASES):
            for ci, s in enumerate(SCALES):
                path = SRC / f"{atom}_{base}_s{fmt_scale(s)}.png"
                ax = axes[ai * 2 + bi, ci]
                if path.exists():
                    im = Image.open(path)
                    ax.imshow(im)
                ax.axis("off")
                if bi == 0 and ci == 0:
                    ax.set_ylabel(f"{atom}\n{base}", fontsize=9, rotation=0,
                                   labelpad=50, ha="right", va="center")
                if ai == 0 and bi == 0:
                    ax.set_title(f"scale {s:+d}", fontsize=10)
                # small annotation per row
                ax.text(-0.02, 0.5, f"{atom}\n{base}" if ci == 0 else "",
                        transform=ax.transAxes, ha="right", va="center",
                        fontsize=8)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("AU library smoke test: FluxSpaceDirectionInject, k=1 site at peak block",
                 fontsize=12)
    fig.savefig(OUT, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] → {OUT}")


if __name__ == "__main__":
    main()
