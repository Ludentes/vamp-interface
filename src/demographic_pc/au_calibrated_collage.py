"""Single-row calibrated smile-scale collage."""
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "output/demographic_pc/au_library_smoke_mean"
OUT = ROOT / "docs/research/images/2026-04-23-au-mean-delta-collage.png"

SCALES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]

fig, axes = plt.subplots(1, len(SCALES), figsize=(len(SCALES) * 3, 3), constrained_layout=True)
for ax, s in zip(axes, SCALES):
    path = SRC / f"smile_elderly_latin_m_s{s:+.4f}.png"
    if path.exists():
        ax.imshow(Image.open(path))
    ax.set_title(f"scale {s}", fontsize=10)
    ax.axis("off")
fig.suptitle("Calibrated smile-atom injection @ K=912 sites on elderly_latin_m", fontsize=11)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=100, bbox_inches="tight")
plt.close(fig)
print(f"[fig] → {OUT}")
