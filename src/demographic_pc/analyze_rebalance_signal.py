"""Quick-look analysis: did each axis actually move MediaPipe blendshapes?

For each (axis, base), compare mean blendshape vector at s=0.4 (near-baseline)
vs s=1.2 (max valid expression). Report the top 5 channels by |delta| and the
max |delta|. If an axis's max channel delta sits near the noise floor, the
prompt didn't land and the axis is unusable for NMF.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "output/demographic_pc/fluxspace_metrics/crossdemo"
AXES = ["anger", "surprise", "disgust", "pucker", "lip_press"]

LOW_SCALE = "s+0.40"
HIGH_SCALE = "s+1.20"


def load(axis: str) -> dict:
    return json.loads((METRICS / axis / "rebalance/blendshapes.json").read_text())


def vec(scores: dict, names: list[str]) -> np.ndarray:
    return np.array([scores.get(n, 0.0) for n in names])


def main() -> None:
    print(f"{'axis':<10} {'n_low':>5} {'n_high':>6} {'max|Δ|':>8} {'top5 channels (Δ)'}")
    print("-" * 100)
    for axis in AXES:
        scores = load(axis)
        names = None
        low, high = [], []
        for rel, s in scores.items():
            if names is None:
                names = sorted(s.keys())
            v = vec(s, names)
            if LOW_SCALE in rel:
                low.append(v)
            elif HIGH_SCALE in rel:
                high.append(v)
        low = np.array(low)
        high = np.array(high)
        if len(low) == 0 or len(high) == 0:
            print(f"{axis:<10} no data")
            continue
        delta = high.mean(0) - low.mean(0)
        order = np.argsort(-np.abs(delta))[:5]
        top = ", ".join(f"{names[i]}({delta[i]:+.3f})" for i in order)
        print(f"{axis:<10} {len(low):>5} {len(high):>6} {np.abs(delta).max():>8.3f}  {top}")


if __name__ == "__main__":
    main()
