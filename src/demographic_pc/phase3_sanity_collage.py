"""On-latent sanity collage: rows = {baseline, cached replay, live pair},
cols = capture seeds."""
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SANITY = ROOT / "output/demographic_pc/phase3_onlatent_sanity"
LIVE = ROOT / "output/demographic_pc/phase1_full"  # live FluxSpaceEditPair captures
OUT = ROOT / "docs/research/images/2026-04-23-phase3-onlatent-sanity.png"

SEEDS = [2026, 4242, 1337]
ROWS = [
    ("baseline\n(scale 0.0)",  lambda s: SANITY / f"elderly_latin_m_s{s}_x+0.0.png"),
    ("cached replay\n(scale 1.0, renorm)", lambda s: SANITY / f"elderly_latin_m_s{s}_x+1.0.png"),
    ("live FluxSpaceEditPair\n(mix_b=0.5, scale=1.0)", lambda s: LIVE / f"elderly_latin_m_s{s}_mix50.png"),
]

fig, axes = plt.subplots(len(ROWS), len(SEEDS),
                         figsize=(len(SEEDS) * 3, len(ROWS) * 3),
                         constrained_layout=True)
for ri, (label, path_fn) in enumerate(ROWS):
    for si, seed in enumerate(SEEDS):
        ax = axes[ri, si]
        p = path_fn(seed)
        if p.exists():
            ax.imshow(Image.open(p))
        ax.set_title(f"seed={seed}" if ri == 0 else "", fontsize=10)
        ax.axis("off")
    axes[ri, 0].set_ylabel(label, fontsize=10)
fig.suptitle("On-latent sanity: cached replay vs live on SAME (base, seed) as capture",
             fontsize=12)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"[fig] → {OUT}")
