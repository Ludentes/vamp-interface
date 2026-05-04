"""Plot blink vs squint distribution on currently kept pos shas to choose
a closed-eye exclusion threshold that doesn't bleed into actual squint.

Joins pair_manifest.kept against reverse_index for bs_eyeBlinkL/R,
bs_eyeSquintL/R, and sg_eyes_closed_margin (SigLIP).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "output/squint_path_b/pair_manifest.parquet"
REVERSE_INDEX = REPO / "output/reverse_index/reverse_index.parquet"
OUT = REPO / "output/squint_path_b/blink_vs_squint.png"


def main() -> None:
    m = pd.read_parquet(MANIFEST)
    kept = m[m["kept"]].copy()
    print(f"[manifest] {len(kept)} kept pairs ({kept['source'].value_counts().to_dict()})")

    ri_cols = ["image_sha256", "bs_eyeBlinkLeft", "bs_eyeBlinkRight",
               "bs_eyeSquintLeft", "bs_eyeSquintRight", "sg_eyes_closed_margin"]
    ri = pd.read_parquet(REVERSE_INDEX, columns=ri_cols)

    pos = kept.merge(ri, left_on="sha_pos", right_on="image_sha256", how="left")
    pos["blink"] = pos["bs_eyeBlinkLeft"] + pos["bs_eyeBlinkRight"]
    pos["squint"] = pos["bs_eyeSquintLeft"] + pos["bs_eyeSquintRight"]
    pos["sg_closed"] = pos["sg_eyes_closed_margin"]

    print("\n[pos blink quantiles]")
    print(pos["blink"].describe(percentiles=[.5, .75, .9, .95, .99]))
    print("\n[pos squint quantiles]")
    print(pos["squint"].describe(percentiles=[.5, .75, .9, .95, .99]))
    print("\n[pos siglip eyes_closed quantiles]")
    print(pos["sg_closed"].describe(percentiles=[.5, .75, .9, .95, .99]))

    # joint table — count of pos shas above thresholds, by source
    for src in pos["source"].unique():
        sub = pos[pos["source"] == src]
        print(f"\n[{src}] n={len(sub)}")
        for bt in (0.4, 0.5, 0.6, 0.7, 0.8):
            n_b = int((sub["blink"] > bt).sum())
            print(f"  blink > {bt}: {n_b}")
        for st in (0.0, 0.5, 1.0, 2.0):
            n_s = int((sub["sg_closed"] > st).sum())
            print(f"  sg_closed > {st}: {n_s}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, src, color in zip(axes[:2],
                              ["ffhq", "flux_solver_a_grid_squint"],
                              ["tab:blue", "tab:orange"]):
        sub = pos[pos["source"] == src]
        ax.scatter(sub["squint"], sub["blink"], s=8, alpha=0.5, c=color)
        ax.axhline(0.5, ls="--", c="red", lw=0.8, label="blink=0.5")
        ax.axhline(0.6, ls=":", c="red", lw=0.8, label="blink=0.6")
        ax.set_xlabel("eyeSquintL+R (pos)")
        ax.set_ylabel("eyeBlinkL+R (pos)")
        ax.set_title(f"{src} (n={len(sub)})")
        ax.set_xlim(0, max(2.0, sub["squint"].max() + 0.1))
        ax.set_ylim(0, max(1.5, sub["blink"].max() + 0.1))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    ax = axes[2]
    for src, color in [("ffhq", "tab:blue"),
                       ("flux_solver_a_grid_squint", "tab:orange")]:
        sub = pos[pos["source"] == src]
        ax.scatter(sub["squint"], sub["sg_closed"], s=8, alpha=0.5,
                   c=color, label=src)
    ax.axhline(0.5, ls="--", c="red", lw=0.8, label="sg=0.5")
    ax.set_xlabel("eyeSquintL+R (pos)")
    ax.set_ylabel("siglip eyes_closed margin (pos)")
    ax.set_title("siglip cross-check")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT, dpi=120)
    print(f"\n[plot] wrote {OUT}")

    # Also dump a small CSV of borderline cases for visual triage
    borderline = pos[(pos["blink"] > 0.4) | (pos["sg_closed"] > 0.0)].copy()
    borderline = borderline.sort_values("blink", ascending=False)
    borderline = borderline[["sha_pos", "source", "ff_race", "ff_gender",
                             "ff_age_bin", "blink", "squint", "sg_closed",
                             "abs_dtheta", "J"]].head(40)
    out_csv = REPO / "output/squint_path_b/blink_borderline_top.csv"
    borderline.to_csv(out_csv, index=False)
    print(f"[csv] wrote {out_csv} ({len(borderline)} rows)")


if __name__ == "__main__":
    main()
