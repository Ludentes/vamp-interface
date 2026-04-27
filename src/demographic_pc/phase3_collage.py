"""2×4 collage of phase 3 replay: rows = bases, cols = scales."""
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "output/demographic_pc/phase3_full_replay"
OUT = ROOT / "docs/research/images/2026-04-23-phase3-full-replay.png"

BASES = ["elderly_latin_m", "young_european_f"]
SCALES = [0.0, 0.5, 1.0, 1.5]
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
        ax.set_title(f"{base}\nscale={s}" if bi == 0 else f"scale={s}", fontsize=9)
        ax.axis("off")
for bi, base in enumerate(BASES):
    axes[bi, 0].set_ylabel(base, fontsize=10)
fig.suptitle(f"Phase 3 replay — seed={SEED} (unseen), K=20 sites on single_{{31,33,34}}",
             fontsize=11)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=100, bbox_inches="tight")
plt.close(fig)
print(f"[fig] → {OUT}")
