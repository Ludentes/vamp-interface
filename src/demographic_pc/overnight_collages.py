"""Build per-axis collages + a master overview for the overnight 990-render
characterization pass.

Outputs:
  docs/research/images/2026-04-24-overnight-master.png       — 11 axes × 5 alphas at one base
  docs/research/images/2026-04-24-overnight-<axis>.png       — 6 bases × 5 alphas per axis (seed 2026)
"""
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "output/demographic_pc/fluxspace_metrics/crossdemo"
OUT_DIR = ROOT / "docs/research/images"

AXES = [
    "age", "gender", "hair_style", "hair_color", "skin_smoothness",
    "nose_shape", "eye_squint", "brow_lift", "brow_furrow",
    "gaze_horizontal", "mouth_stretch",
]
BASES = ["asian_m", "black_f", "european_m", "elderly_latin_m",
         "young_european_f", "southasian_f"]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
SEED = 2026
MASTER_BASE = "elderly_latin_m"


def path_for(axis: str, base: str, seed: int, alpha: float) -> Path:
    return SRC / axis / f"{axis}_inphase" / base / f"s{seed}_a{alpha:.2f}.png"


def axis_collage(axis: str) -> None:
    fig, axes = plt.subplots(len(BASES), len(ALPHAS),
                             figsize=(len(ALPHAS) * 2.5, len(BASES) * 2.5),
                             constrained_layout=True)
    for bi, base in enumerate(BASES):
        for ai, a in enumerate(ALPHAS):
            ax = axes[bi, ai]
            p = path_for(axis, base, SEED, a)
            if p.exists():
                ax.imshow(Image.open(p))
            ax.set_title(f"α={a}" if bi == 0 else "", fontsize=9)
            ax.axis("off")
        axes[bi, 0].set_ylabel(base, fontsize=9, rotation=90, labelpad=6)
    fig.suptitle(f"axis: {axis} — seed {SEED} (one seed per cell)", fontsize=12)
    dst = OUT_DIR / f"2026-04-24-overnight-{axis}.png"
    fig.savefig(dst, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[axis] {axis} → {dst}")


def master_collage() -> None:
    fig, axes = plt.subplots(len(AXES), len(ALPHAS),
                             figsize=(len(ALPHAS) * 2.5, len(AXES) * 2.5),
                             constrained_layout=True)
    for ri, axis in enumerate(AXES):
        for ai, a in enumerate(ALPHAS):
            ax = axes[ri, ai]
            p = path_for(axis, MASTER_BASE, SEED, a)
            if p.exists():
                ax.imshow(Image.open(p))
            ax.set_title(f"α={a}" if ri == 0 else "", fontsize=9)
            ax.axis("off")
        axes[ri, 0].set_ylabel(axis, fontsize=10, rotation=90, labelpad=6)
    fig.suptitle(f"Overnight 11-axis master — base={MASTER_BASE} seed={SEED}",
                 fontsize=13)
    dst = OUT_DIR / "2026-04-24-overnight-master.png"
    fig.savefig(dst, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[master] → {dst}")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    master_collage()
    for axis in AXES:
        axis_collage(axis)
