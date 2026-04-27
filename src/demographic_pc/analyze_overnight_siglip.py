"""Per-axis monotonicity analysis of SigLIP-2 probe margins on the
overnight corpus — 6 axes not covered by MediaPipe blendshapes."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
IN = ROOT / "output/demographic_pc/overnight_siglip_probes.parquet"

# Axis → probe whose positive prompt matches α=1 end of the axis.
AXIS_PROBE = {
    "age":             "elderly",
    "gender":          "feminine",
    "hair_style":      "long_hair",
    "hair_color":      "black_hair",
    "skin_smoothness": "rough_skin",
    "nose_shape":      "aquiline_nose",
}

BASES = ["asian_m", "black_f", "european_m", "elderly_latin_m",
         "young_european_f", "southasian_f"]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def main() -> None:
    df = pd.read_parquet(IN)
    print(f"[siglip-analysis] rows={len(df)}")
    df["axis"] = df["rel"].apply(lambda s: s.split("/")[0])
    df["base"] = df["rel"].apply(lambda s: s.split("/")[2])
    df["alpha"] = df["rel"].apply(lambda s: float(s.split("/")[-1].split("_a")[-1].replace(".png", "")))

    for axis, probe in AXIS_PROBE.items():
        col = f"{probe}_margin"
        sub = df[df["axis"] == axis]
        if sub.empty:
            print(f"\n[{axis}] no data"); continue

        # Global means per α.
        global_means = []
        for a in ALPHAS:
            vals = sub[np.isclose(sub["alpha"], a)][col]
            global_means.append(float(vals.mean()) if len(vals) else float("nan"))

        effect = global_means[-1] - global_means[0]
        is_mono = all(global_means[i] <= global_means[i+1] + 1e-3
                      for i in range(len(global_means) - 1))

        print(f"\n[{axis}] probe={probe}  (higher margin = axis α=1 end)")
        for a, m in zip(ALPHAS, global_means):
            print(f"  α={a:.2f}  margin={m:+.4f}")
        print(f"  effect α1−α0 = {effect:+.4f}  monotonic={is_mono}")

        print(f"  per-base effect (α=1 vs α=0):")
        for b in BASES:
            v0 = sub[(sub["base"] == b) & np.isclose(sub["alpha"], 0.0)][col]
            v1 = sub[(sub["base"] == b) & np.isclose(sub["alpha"], 1.0)][col]
            if len(v0) and len(v1):
                print(f"    {b:18s} Δ={v1.mean()-v0.mean():+.4f}")


if __name__ == "__main__":
    main()
