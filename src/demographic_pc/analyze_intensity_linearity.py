"""Phase-4 analysis (redirected) — per-axis linearity in `scale`.

Operates on `intensity_full` (340 scored renders at mix_b=0.5, varying
scale ∈ [0.2, 2.3] and start_pct ∈ {0.15, 0.25, 0.40} across 4 ladder
rungs × 6 bases). Tests whether a single-axis FluxSpace edit is linear
in `scale` once we're above the α-injection threshold.

Reports:
  1. Per (base, ladder, start_pct) trajectory: linear-fit R² of target
     blendshape vs scale; monotonicity (Kendall τ); saturation onset.
  2. Aggregate: does start_pct shift the scale-response? Does the
     response saturate or diverge at large scale?
  3. Cross-channel leakage: does the "smile" ladder move only
     mouthSmile, or does it pull jawOpen along proportionally?
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[2]
BS_JSON = ROOT / "output/demographic_pc/fluxspace_metrics/crossdemo/smile/intensity_full/blendshapes.json"
OUT = ROOT / "output/demographic_pc/fluxspace_metrics/analysis/intensity_linearity.json"

KEY_RE = re.compile(
    r"^(?P<base>[^/]+)/(?P<ladder>\d{2}_\w+)/sp(?P<sp>[0-9.]+)_s(?P<scale>[+-][0-9.]+)$"
)

CHANNELS = {
    "smile": ["mouthSmileLeft", "mouthSmileRight"],
    "jaw": ["jawOpen"],
    "stretch": ["mouthStretchLeft", "mouthStretchRight"],
    "funnel": ["mouthFunnel"],
}


def avg(bs: dict, keys: list[str]) -> float:
    return float(np.mean([bs.get(k, 0.0) for k in keys]))


def parse(rel: str) -> dict | None:
    stem = rel[:-4] if rel.endswith(".png") else rel
    m = KEY_RE.match(stem)
    if not m:
        return None
    return {
        "base": m.group("base"),
        "ladder": m.group("ladder"),
        "sp": float(m.group("sp")),
        "scale": float(m.group("scale")),
    }


def main() -> None:
    bs = json.loads(BS_JSON.read_text())
    print(f"[intensity-lin] {len(bs)} scored renders")

    # group by (base, ladder, sp) → {scale: {channel: value}}
    groups: dict = defaultdict(dict)
    for rel, scores in bs.items():
        tag = parse(rel)
        if tag is None:
            continue
        key = (tag["base"], tag["ladder"], tag["sp"])
        groups[key][tag["scale"]] = {ch: avg(scores, keys)
                                     for ch, keys in CHANNELS.items()}

    print(f"[intensity-lin] {len(groups)} (base, ladder, sp) trajectories")

    # Per-trajectory linearity on the "primary" channel for each ladder
    primary_channel = {
        "01_faint": "smile", "02_warm": "smile",
        "03_broad": "smile", "04_manic": "jaw",
        "05_cackle": "jaw",
    }

    rows = []
    for (base, ladder, sp), scale_pts in groups.items():
        if len(scale_pts) < 3:
            continue
        scales = np.array(sorted(scale_pts.keys()))
        pch = primary_channel[ladder]
        y = np.array([scale_pts[s][pch] for s in scales])
        # Kendall τ for monotonicity
        tau = float(kendalltau(scales, y).statistic)
        # Linear fit
        p = np.polyfit(scales, y, 1)
        y_lin = np.polyval(p, scales)
        rss = ((y - y_lin) ** 2).sum()
        tss = ((y - y.mean()) ** 2).sum() + 1e-12
        r2 = float(1.0 - rss / tss)
        # Saturation: R² drop if we cap at max scale, compare to full
        # (high saturation → curve flattens at large s)
        range_y = float(y.max() - y.min())
        # Cross-axis leakage: all other channels
        leakage = {}
        for ch in CHANNELS:
            if ch == pch:
                continue
            y2 = np.array([scale_pts[s][ch] for s in scales])
            leakage[ch] = float(y2.max() - y2.min())
        rows.append({
            "base": base, "ladder": ladder, "sp": sp,
            "primary": pch,
            "n_scales": len(scales),
            "scales": scales.tolist(),
            "y_primary": y.tolist(),
            "tau": tau, "r2_linear": r2, "range": range_y,
            "leakage": leakage,
        })

    # === aggregate tables ===
    print("\n=== per-ladder, per-sp linearity (mean across 6 bases) ===")
    print(f"  {'ladder':<10} {'sp':>5} {'n':>4} {'<τ>':>6} {'<R²>':>7} "
          f"{'<range>':>8} {'<leak_jaw>':>10} {'<leak_stretch>':>12}")
    agg: dict = defaultdict(list)
    for r in rows:
        agg[(r["ladder"], r["sp"])].append(r)
    for (ladder, sp), rs in sorted(agg.items()):
        taus = [r["tau"] for r in rs]
        r2s = [r["r2_linear"] for r in rs]
        ranges = [r["range"] for r in rs]
        leak_jaw = [r["leakage"].get("jaw", 0.0) for r in rs]
        leak_str = [r["leakage"].get("stretch", 0.0) for r in rs]
        print(f"  {ladder:<10} {sp:>5.2f} {len(rs):>4} "
              f"{np.mean(taus):>6.3f} {np.mean(r2s):>7.3f} "
              f"{np.mean(ranges):>8.3f} {np.mean(leak_jaw):>10.3f} "
              f"{np.mean(leak_str):>12.3f}")

    # Classification per row
    def classify(r):
        if r["tau"] < 0.7:
            return "C-nonmonotonic"
        if r["r2_linear"] >= 0.85:
            return "A-linear"
        if r["r2_linear"] >= 0.5:
            return "B-saturating"
        return "C-nonmonotonic"

    classes = defaultdict(int)
    for r in rows:
        classes[classify(r)] += 1
    print("\n=== trajectory classification ===")
    for c, n in sorted(classes.items()):
        print(f"  {c:<20} {n:>3} / {len(rows)} ({100*n/len(rows):.1f}%)")

    # Start_pct effect: does starting later (sp=0.40) flatten / shift response?
    print("\n=== start_pct effect on response range (mean y_max - y_min) ===")
    print(f"  {'ladder':<10} sp=0.15  sp=0.25  sp=0.40")
    for ladder in sorted({r["ladder"] for r in rows}):
        ranges_by_sp = defaultdict(list)
        for r in rows:
            if r["ladder"] == ladder:
                ranges_by_sp[r["sp"]].append(r["range"])
        line = f"  {ladder:<10}"
        for sp in [0.15, 0.25, 0.40]:
            vals = ranges_by_sp.get(sp, [])
            line += f"  {np.mean(vals) if vals else 0.0:6.3f}"
        print(line)

    # Save
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2))
    print(f"\nsaved → {OUT}")


if __name__ == "__main__":
    main()
