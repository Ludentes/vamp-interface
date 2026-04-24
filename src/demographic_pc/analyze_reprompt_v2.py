"""Compare v2 reprompted corpora (mouth_stretch_v2, brow_furrow_v2) against
v1 blendshape metrics.

v1 problems:
  - mouth_stretch: target mouthStretchL/R *decreased* 0.06 → 0.01; jawOpen
    exploded 0.003 → 0.54 (prompt pulled shocked-open-mouth).
  - brow_furrow: target browDownL/R rose 0.10 → 0.28 but non-monotonic and
    Δ ≈ 0 on asian_m, black_f, southasian_f.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "output/demographic_pc/overnight_blendshapes.json"
V2_PATHS = {
    "mouth_stretch_v2": ROOT / "output/demographic_pc/overnight_blendshapes_mouth_stretch_v2.json",
    "brow_furrow_v2":   ROOT / "output/demographic_pc/overnight_blendshapes_brow_furrow_v2.json",
}

TARGETS = {
    "mouth_stretch": ["mouthStretchLeft", "mouthStretchRight"],
    "brow_furrow":   ["browDownLeft", "browDownRight"],
}
CROSS = {
    "mouth_stretch": ["jawOpen"],
    "brow_furrow":   [],
}

BASES = ["asian_m", "black_f", "european_m", "elderly_latin_m",
         "young_european_f", "southasian_f"]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def _bucket(data: dict, axis: str, targets: list[str], v2: bool):
    bucket: dict[tuple[float, str], list[float]] = defaultdict(list)
    for rel, scores in data.items():
        parts = rel.split("/")
        if v2:
            # layout within v2 JSON: <axis>_v2_inphase/<base>/s<seed>_aN.NN.png
            if len(parts) != 3:
                continue
            base = parts[1]
            stem = parts[2].replace(".png", "")
        else:
            # layout in overnight JSON: <axis>/<axis>_inphase/<base>/s<seed>_aN.NN.png
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
        s = sum(scores.get(t, 0.0) for t in targets)
        bucket[(alpha, base)].append(s)
    return bucket


def _summary(bucket):
    g = []
    for a in ALPHAS:
        vals = []
        for b in BASES:
            vals.extend(bucket.get((a, b), []))
        g.append(float(np.mean(vals)) if vals else float("nan"))
    per_base = {}
    for b in BASES:
        v0 = bucket.get((0.0, b), [])
        v1 = bucket.get((1.0, b), [])
        per_base[b] = (np.mean(v1) - np.mean(v0)) if (v0 and v1) else float("nan")
    mono = all(g[i] <= g[i+1] + 1e-3 for i in range(len(g)-1))
    return g, per_base, mono


def main() -> None:
    v1 = json.loads(V1.read_text())

    for axis_v1, targets in TARGETS.items():
        axis_v2 = f"{axis_v1}_v2"
        v2 = json.loads(V2_PATHS[axis_v2].read_text())

        b1 = _bucket(v1, axis_v1, targets, v2=False)
        b2 = _bucket(v2, axis_v2, targets, v2=True)
        g1, pb1, m1 = _summary(b1)
        g2, pb2, m2 = _summary(b2)

        print(f"\n===== {axis_v1} vs {axis_v2} — target {targets} =====")
        print(f"  α       v1       v2       Δ(v2-v1)")
        for a, x, y in zip(ALPHAS, g1, g2):
            print(f"  {a:.2f}  {x:+.4f}  {y:+.4f}   {y-x:+.4f}")
        print(f"  effect v1: {g1[-1]-g1[0]:+.4f} monotonic={m1}")
        print(f"  effect v2: {g2[-1]-g2[0]:+.4f} monotonic={m2}")
        print(f"  per-base Δ (α=1 − α=0):")
        print(f"    {'base':18s}  v1       v2")
        for b in BASES:
            print(f"    {b:18s} {pb1[b]:+.4f}  {pb2[b]:+.4f}")

        # Cross-check
        for cx in CROSS[axis_v1]:
            b1c = _bucket(v1, axis_v1, [cx], v2=False)
            b2c = _bucket(v2, axis_v2, [cx], v2=True)
            g1c, _, _ = _summary(b1c)
            g2c, _, _ = _summary(b2c)
            print(f"  cross-check [{cx}]")
            for a, x, y in zip(ALPHAS, g1c, g2c):
                print(f"    α={a:.2f}  v1={x:+.4f}  v2={y:+.4f}")
            print(f"    effect v1: {g1c[-1]-g1c[0]:+.4f}  v2: {g2c[-1]-g2c[0]:+.4f}")


if __name__ == "__main__":
    main()
