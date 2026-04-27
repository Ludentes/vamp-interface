"""Check whether MediaPipe blendshape scores confirm the visual axis screening.

For each of the 5 blendshape-measurable overnight axes, compute:
  - mean of target blendshape at each α value (averaged over bases × seeds)
  - monotonicity: is the mean strictly increasing in α?
  - effect size: mean(α=1) - mean(α=0)
  - per-base breakdown to catch axes that work on some bases not others
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
JSON = ROOT / "output/demographic_pc/overnight_blendshapes.json"

# Axis → list of target blendshape names. Sum of these per image is the
# "axis score"; compare across α.
AXIS_TARGETS = {
    "brow_lift":       ["browInnerUp", "browOuterUpLeft", "browOuterUpRight"],
    "brow_furrow":     ["browDownLeft", "browDownRight"],
    "eye_squint":      ["eyeSquintLeft", "eyeSquintRight"],
    "gaze_horizontal": ["eyeLookOutLeft", "eyeLookOutRight",
                        "eyeLookInLeft", "eyeLookInRight"],
    # mouth_stretch visual showed open-mouth; check both the target
    # blendshape AND mouthOpen/jawOpen to confirm it's the wrong axis.
    "mouth_stretch":   ["mouthStretchLeft", "mouthStretchRight"],
}
MOUTH_OPEN_CHECK = ["jawOpen"]  # what mouth_stretch prompt actually activates?

BASES = ["asian_m", "black_f", "european_m", "elderly_latin_m",
         "young_european_f", "southasian_f"]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def main() -> None:
    data = json.loads(JSON.read_text())
    print(f"[metrics] {len(data)} scored images")

    for axis, targets in AXIS_TARGETS.items():
        # Collect (alpha, base) → list of per-image sum-of-targets.
        bucket: dict[tuple[float, str], list[float]] = defaultdict(list)
        for rel, scores in data.items():
            parts = rel.split("/")
            # key layout: <axis>/<axis>_inphase/<base>/s<seed>_a<alpha>.png
            if len(parts) != 4 or parts[0] != axis:
                continue
            base = parts[2]
            stem = parts[3].replace(".png", "")
            try:
                alpha = float(stem.split("_a")[-1])
            except ValueError:
                continue
            if base not in BASES or alpha not in ALPHAS:
                continue
            axis_score = sum(scores.get(t, 0.0) for t in targets)
            bucket[(alpha, base)].append(axis_score)

        if not bucket:
            print(f"\n[{axis}] NO DATA")
            continue

        # All-bases mean per α.
        global_means = []
        for a in ALPHAS:
            vals = []
            for b in BASES:
                vals.extend(bucket.get((a, b), []))
            global_means.append(float(np.mean(vals)) if vals else float("nan"))

        effect = global_means[-1] - global_means[0]
        is_mono = all(global_means[i] <= global_means[i+1] + 1e-3
                      for i in range(len(global_means) - 1))
        print(f"\n[{axis}] target = sum({targets})")
        for a, m in zip(ALPHAS, global_means):
            print(f"  α={a:.2f}  mean={m:.4f}")
        print(f"  effect α1−α0 = {effect:+.4f}  monotonic={is_mono}")

        # Per-base diagnostic — which bases respond.
        print(f"  per-base effect (α=1 vs α=0):")
        for b in BASES:
            v0 = bucket.get((0.0, b), [])
            v1 = bucket.get((1.0, b), [])
            if v0 and v1:
                print(f"    {b:18s} Δ={np.mean(v1)-np.mean(v0):+.4f}")

    # Bonus: mouth_stretch cross-check — did "mouth stretch" prompt
    # actually produce jaw-open instead?
    print(f"\n[mouth_stretch cross-check: jawOpen]")
    for a in ALPHAS:
        vals = []
        for b in BASES:
            for rel, scores in data.items():
                parts = rel.split("/")
                if len(parts) != 4 or parts[0] != "mouth_stretch":
                    continue
                if parts[2] != b:
                    continue
                stem = parts[3].replace(".png", "")
                try:
                    alpha = float(stem.split("_a")[-1])
                except ValueError:
                    continue
                if abs(alpha - a) < 1e-3:
                    vals.append(sum(scores.get(t, 0.0) for t in MOUTH_OPEN_CHECK))
        print(f"  α={a:.2f}  jawOpen mean={np.mean(vals) if vals else float('nan'):.4f}")


if __name__ == "__main__":
    main()
