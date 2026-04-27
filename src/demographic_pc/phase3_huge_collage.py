"""Huge-scale collage."""
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "output/demographic_pc/phase3_sweep_huge"
OUT = ROOT / "docs/research/images/2026-04-23-phase3-huge-sweep.png"

BASES = ["elderly_latin_m", "young_european_f"]
SCALES = [10.0, 30.0, 100.0, 300.0]
SEED = 777

fig, axes = plt.subplots(len(BASES), len(SCALES),
                         figsize=(len(SCALES) * 3, len(BASES) * 3),
                         constrained_layout=True)
for bi, base in enumerate(BASES):
    for si, s in enumerate(SCALES):
        ax = axes[bi, si]
        p = SRC / f"{base}_s{SEED}_x{s:+.1f}.png"
        if p.exists():
            ax.imshow(Image.open(p))
        ax.set_title(f"scale={s}", fontsize=10)
        ax.axis("off")
fig.suptitle(f"Phase 3 HUGE scale sweep (renorm ON) — seed={SEED} unseen",
             fontsize=12)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"[fig] → {OUT}")
