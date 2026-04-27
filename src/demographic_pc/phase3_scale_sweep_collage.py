"""2×6 collage for the wide scale sweep."""
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "output/demographic_pc/phase3_scale_sweep"
OUT = ROOT / "docs/research/images/2026-04-23-phase3-scale-sweep.png"

BASES = ["elderly_latin_m", "young_european_f"]
SCALES = [0.0, 2.0, 3.0, 5.0, 7.0, 10.0]
SEED = 777

fig, axes = plt.subplots(len(BASES), len(SCALES),
                         figsize=(len(SCALES) * 3, len(BASES) * 3),
                         constrained_layout=True)
for bi, base in enumerate(BASES):
    for si, s in enumerate(SCALES):
        ax = axes[bi, si]
        p = SRC / f"{base}_s{SEED}_x{s:+.2f}.png"
        if p.exists():
            ax.imshow(Image.open(p))
        ax.set_title(f"scale={s}", fontsize=10)
        ax.axis("off")
    axes[bi, 0].set_ylabel(base, fontsize=11, rotation=0, labelpad=80)
fig.suptitle(f"Phase 3 cache replay — wide scale sweep — seed={SEED} unseen", fontsize=12)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"[fig] → {OUT}")
