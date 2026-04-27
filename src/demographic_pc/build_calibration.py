"""Per-(atom × axis × base) response curve calibration table.

Uses the sample_index.parquet. For each (atom, axis, base) cell with
varying scale, fit a simple linear model `atom_coeff = a * scale + b`
and report:

    n_samples    total samples for this (atom, axis, base)
    n_scales     number of distinct scales observed
    scale_min    min scale value
    scale_max    max scale value
    mean_low     mean atom at scale_min
    mean_high    mean atom at scale_max
    delta        mean_high - mean_low  (signed atom response)
    slope        least-squares slope atom/scale
    intercept    least-squares intercept
    r2           coefficient of determination
    monotone     bool — is mean atom monotonically increasing in scale

Inverse calibration:
    required_scale(atom, axis, base, target) = (target - intercept) / slope

A cell is "usable" when:
    slope > 0.05 (atom responds to scale)
    r2 > 0.6     (response is approximately linear)
    monotone     (no sign flip across scales)
    n_scales >= 3

Outputs:
    models/blendshape_nmf/calibration_table.parquet — per-(atom, axis, base) row
    models/blendshape_nmf/calibration_summary.json  — aggregate stats
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
NMF_DIR = ROOT / "models/blendshape_nmf"
INDEX = NMF_DIR / "sample_index.parquet"
OUT_PARQUET = NMF_DIR / "calibration_table.parquet"
OUT_JSON = NMF_DIR / "calibration_summary.json"

DEAD_ATOMS = {0, 2, 3, 6}
USABLE_SLOPE_MIN = 0.05
USABLE_R2_MIN = 0.6
USABLE_MIN_SCALES = 3


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return (slope, intercept, r2)."""
    if len(x) < 2 or x.std() == 0:
        return 0.0, float(y.mean()) if len(y) else 0.0, 0.0
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    pred = slope * x + intercept
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return float(slope), float(intercept), float(r2)


def main():
    print(f"[calib] loading {INDEX}")
    df = pd.read_parquet(INDEX)
    print(f"  {len(df)} samples loaded")

    # Filter: rows with a known axis AND a numeric scale, excluding bootstrap.
    df = df[df.axis.notna() & df.scale.notna()].copy()
    print(f"  {len(df)} rows with (axis, scale)")

    # `atom_NN` columns only; the `atom_source` metadata column also starts
    # with "atom_" so filter it out explicitly.
    atom_cols = [c for c in df.columns
                 if c.startswith("atom_") and c != "atom_source"]
    live_atoms = [int(c.removeprefix("atom_")) for c in atom_cols
                  if int(c.removeprefix("atom_")) not in DEAD_ATOMS]

    rows = []
    for (axis, base), group in df.groupby(["axis", "base"]):
        scales = sorted(group.scale.unique())
        if len(scales) < 2:
            continue
        for k in live_atoms:
            col = f"atom_{k:02d}"
            # Mean atom per scale
            means = group.groupby("scale")[col].mean().sort_index()
            xs = means.index.values.astype(float)
            ys = means.values.astype(float)
            slope, intercept, r2 = linear_fit(xs, ys)
            mean_low = float(ys[0])
            mean_high = float(ys[-1])
            delta = mean_high - mean_low
            # Monotonicity: strictly increasing OR strictly decreasing in means
            diffs = np.diff(ys)
            if len(diffs) == 0:
                monotone = True
            elif (diffs >= 0).all() or (diffs <= 0).all():
                monotone = True
            else:
                monotone = False
            usable = (abs(slope) >= USABLE_SLOPE_MIN and r2 >= USABLE_R2_MIN
                      and monotone and len(scales) >= USABLE_MIN_SCALES)
            rows.append({
                "atom": k,
                "axis": axis,
                "base": base,
                "n_samples": int(len(group)),
                "n_scales": len(scales),
                "scale_min": float(xs[0]),
                "scale_max": float(xs[-1]),
                "mean_low": mean_low,
                "mean_high": mean_high,
                "delta": float(delta),
                "slope": slope,
                "intercept": intercept,
                "r2": r2,
                "monotone": monotone,
                "usable": usable,
            })

    cal = pd.DataFrame(rows)
    cal.to_parquet(OUT_PARQUET, index=False, compression="zstd")
    print(f"\n[save] → {OUT_PARQUET}  "
          f"({OUT_PARQUET.stat().st_size / 1024:.1f} KB)")
    print(f"  {len(cal)} cells (atom × axis × base)")

    # Summary
    usable = cal[cal.usable]
    print(f"\n[summary] usable cells: {len(usable)}/{len(cal)} "
          f"(slope≥{USABLE_SLOPE_MIN}, r²≥{USABLE_R2_MIN}, monotone, n_scales≥{USABLE_MIN_SCALES})")

    # Per-axis: how many (atom, base) cells are usable?
    print("\n[per axis] usable (atom, base) cells:")
    pa = cal.groupby("axis").apply(
        lambda g: f"{int(g.usable.sum()):>3}/{len(g):<3}  ({int(g.usable.sum()) / max(len(g), 1) * 100:.0f}%)",
        include_groups=False,
    )
    print(pa.to_string())

    # Per-atom: how many axes can move this atom on how many bases?
    print("\n[per atom] best axis × n_bases_usable:")
    for k in live_atoms:
        sub = cal[(cal.atom == k) & cal.usable]
        if len(sub) == 0:
            continue
        by_axis = sub.groupby("axis").size()
        best_axis = by_axis.idxmax()
        n_bases = int(by_axis.max())
        best_delta = cal[(cal.atom == k) & (cal.axis == best_axis)].delta.mean()
        print(f"  atom_{k:02d}  {best_axis:<18}  {n_bases}/6 bases usable  "
              f"mean Δ={best_delta:+.3f}")

    # "Clean axes" — usable on all 6 bases
    print("\n[clean cells] usable on ≥5/6 bases per (atom, axis):")
    by_atom_axis = usable.groupby(["atom", "axis"]).size().reset_index(name="n_bases")
    clean = by_atom_axis[by_atom_axis.n_bases >= 5].sort_values("n_bases", ascending=False)
    for _, row in clean.iterrows():
        sub = cal[(cal.atom == row.atom) & (cal.axis == row.axis) & cal.usable]
        slope_mean = sub.slope.mean()
        intercept_mean = sub.intercept.mean()
        print(f"  atom_{int(row.atom):02d} × {row.axis:<12}  "
              f"usable on {int(row.n_bases)}/6 bases  "
              f"mean slope={slope_mean:+.3f}  intercept={intercept_mean:+.3f}")

    # Export JSON summary for morning plan
    summary = {
        "n_cells": int(len(cal)),
        "n_usable": int(len(usable)),
        "clean_cells": clean.to_dict("records"),
        "thresholds": {"slope_min": USABLE_SLOPE_MIN, "r2_min": USABLE_R2_MIN,
                       "monotone_required": True, "min_scales": USABLE_MIN_SCALES},
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n[save] → {OUT_JSON}")


if __name__ == "__main__":
    main()
