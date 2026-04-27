"""α-interp linearity analysis.

For each (base, seed), plot blendshape scores vs α ∈ {0.0, 0.1, …, 1.0}.
Fit linear and cubic models; report per-seed R² of the linear fit, deviation
from linearity (nonlinearity ratio), monotonicity, and identify any phase
boundary signatures: plateaus, jumps, or inflection points.

Per manifold theory (Hessian-Geometry paper, survey ref [13]): smooth monotonic
trajectory = within-phase Euclidean interpolation is valid; jumps/plateaus =
phase boundaries where Lipschitz constant diverges.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ALPHA_DIR = ROOT / "output" / "demographic_pc" / "fluxspace_metrics" / "crossdemo" / "smile" / "alpha_interp"
BS_JSON = ALPHA_DIR / "blendshapes.json"
OUT_DIR = ROOT / "output" / "demographic_pc" / "fluxspace_metrics" / "alpha_linearity"

# Composite "smile intensity" target: average of the non-dead blendshape channels
# that loaded onto PC1 in the prior PCA fit.
SMILE_TARGETS = ["mouthSmileLeft", "mouthSmileRight",
                 "mouthStretchLeft", "mouthStretchRight"]
JAW_TARGETS = ["jawOpen"]

FNAME_RE = re.compile(r"^s(?P<seed>\d+)_a(?P<alpha>[0-9.]+)$")


def parse_stem(stem: str) -> dict | None:
    m = FNAME_RE.match(stem)
    return m.groupdict() if m else None


def smile_intensity(bs: dict) -> float:
    return float(np.mean([bs.get(t, 0.0) for t in SMILE_TARGETS]))


def jaw(bs: dict) -> float:
    return float(np.mean([bs.get(t, 0.0) for t in JAW_TARGETS]))


def fit_linear_and_cubic(alpha: np.ndarray, y: np.ndarray) -> dict:
    """R² for linear vs cubic fit; monotonicity check; signed residual pattern."""
    # Linear
    p1 = np.polyfit(alpha, y, 1)
    y_lin = np.polyval(p1, alpha)
    rss_lin = float(((y - y_lin) ** 2).sum())
    tss = float(((y - y.mean()) ** 2).sum())
    r2_lin = 1.0 - rss_lin / (tss + 1e-12)

    # Cubic
    p3 = np.polyfit(alpha, y, 3)
    y_cub = np.polyval(p3, alpha)
    rss_cub = float(((y - y_cub) ** 2).sum())
    r2_cub = 1.0 - rss_cub / (tss + 1e-12)

    # Nonlinearity ratio: how much variance is explained ONLY by cubic terms.
    # High value = linear fit misses curvature; phase-boundary candidate.
    nonlin = max(0.0, r2_cub - r2_lin)

    # Monotonicity: are all consecutive Δy same sign?
    dy = np.diff(y)
    monotonic = bool(np.all(dy >= -1e-4)) or bool(np.all(dy <= 1e-4))

    # Max step: largest single α→α+0.1 jump, relative to total range.
    y_range = float(y.max() - y.min()) + 1e-9
    max_step = float(np.abs(dy).max()) / y_range

    # Signed residuals from linear fit: concave/convex detection.
    # If residuals are U-shaped (concave) or ∩-shaped (convex) the fit curve
    # deviates from a straight line consistently.
    res = y - y_lin
    # Concavity sign: mean of middle residuals minus mean of endpoint residuals.
    # Negative = ∩-shape (concave up from endpoints), positive = U-shape.
    mid = res[len(res) // 3 : 2 * len(res) // 3]
    ends = np.concatenate([res[:2], res[-2:]])
    concavity_sign = float(mid.mean() - ends.mean())

    return {
        "r2_linear": r2_lin,
        "r2_cubic": r2_cub,
        "nonlinearity": nonlin,
        "monotonic": monotonic,
        "max_step_ratio": max_step,
        "concavity_sign": concavity_sign,
        "slope_linear": float(p1[0]),
        "intercept_linear": float(p1[1]),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bs_scores = json.loads(BS_JSON.read_text())
    print(f"[alpha-lin] {len(bs_scores)} blendshape entries")

    # Group by (base, seed) → {alpha: (smile, jaw)}
    grouped: dict = defaultdict(dict)
    for rel_path, bs in bs_scores.items():
        base, fname = rel_path.split("/")
        stem = Path(fname).stem
        tag = parse_stem(stem)
        if tag is None:
            continue
        key = (base, int(tag["seed"]))
        grouped[key][float(tag["alpha"])] = {
            "smile": smile_intensity(bs),
            "jaw": jaw(bs),
        }

    print(f"[alpha-lin] {len(grouped)} (base, seed) groups")

    # Per-group linearity analysis
    rows = []
    per_base_smile_r2 = defaultdict(list)
    per_base_jaw_r2 = defaultdict(list)
    for (base, seed), pts in grouped.items():
        alphas = np.array(sorted(pts.keys()))
        smile_y = np.array([pts[a]["smile"] for a in alphas])
        jaw_y = np.array([pts[a]["jaw"] for a in alphas])

        smile_fit = fit_linear_and_cubic(alphas, smile_y)
        jaw_fit = fit_linear_and_cubic(alphas, jaw_y)
        rows.append({
            "base": base, "seed": seed,
            "smile_y": smile_y.tolist(), "jaw_y": jaw_y.tolist(),
            "smile_fit": smile_fit, "jaw_fit": jaw_fit,
        })
        per_base_smile_r2[base].append(smile_fit["r2_linear"])
        per_base_jaw_r2[base].append(jaw_fit["r2_linear"])

    # Aggregate per-base
    print("\n=== per-base linear-fit quality ===")
    print(f"  {'base':>22}  {'smile_R²(linear)':>20}  {'jaw_R²(linear)':>18}  n_seeds")
    for base in sorted(per_base_smile_r2):
        s = np.array(per_base_smile_r2[base])
        j = np.array(per_base_jaw_r2[base])
        print(f"  {base:>22}  {s.mean():.3f} ± {s.std():.3f}       "
              f"{j.mean():.3f} ± {j.std():.3f}    {len(s)}")

    # Monotonicity + nonlinearity across all groups
    smile_nonlin = np.array([r["smile_fit"]["nonlinearity"] for r in rows])
    smile_mono = np.array([r["smile_fit"]["monotonic"] for r in rows])
    smile_concavity = np.array([r["smile_fit"]["concavity_sign"] for r in rows])
    jaw_nonlin = np.array([r["jaw_fit"]["nonlinearity"] for r in rows])
    jaw_mono = np.array([r["jaw_fit"]["monotonic"] for r in rows])

    print("\n=== aggregate diagnostics ===")
    print(f"  smile: monotonic in {smile_mono.sum()}/{len(smile_mono)} groups")
    print(f"  smile: nonlinearity mean={smile_nonlin.mean():.4f}  p95={np.percentile(smile_nonlin, 95):.4f}  max={smile_nonlin.max():.4f}")
    print(f"  smile: concavity_sign mean={smile_concavity.mean():+.4f}  std={smile_concavity.std():.4f}")
    print(f"  jaw:   monotonic in {jaw_mono.sum()}/{len(jaw_mono)} groups")
    print(f"  jaw:   nonlinearity mean={jaw_nonlin.mean():.4f}  p95={np.percentile(jaw_nonlin, 95):.4f}")

    # Largest jump per group (phase boundary candidates)
    print("\n=== largest single-α jumps (phase-boundary candidates) ===")
    smile_jumps = [(r["base"], r["seed"], r["smile_fit"]["max_step_ratio"]) for r in rows]
    smile_jumps.sort(key=lambda x: -x[2])
    for base, seed, ratio in smile_jumps[:8]:
        print(f"  {base:>22}  s={seed}  max_step_ratio={ratio:.3f}")

    # Save per-group tables
    with (OUT_DIR / "per_group_stats.json").open("w") as f:
        json.dump(rows, f, indent=2)

    # Concavity interpretation per base
    print("\n=== per-base concavity pattern ===")
    print(f"  {'base':>22}  {'mean concavity':>14}  interpretation")
    per_base_concavity = defaultdict(list)
    for r in rows:
        per_base_concavity[r["base"]].append(r["smile_fit"]["concavity_sign"])
    for base, vals in sorted(per_base_concavity.items()):
        m = float(np.mean(vals))
        s = float(np.std(vals))
        tag = ("concave (U-shape, mid above linear)" if m > 0.005
               else "convex (∩-shape, mid below linear)" if m < -0.005
               else "approximately linear")
        print(f"  {base:>22}  {m:+.4f} ± {s:.4f}   {tag}")

    # Seed-vs-seed consistency: for each base, how similar are the curves across seeds?
    print("\n=== cross-seed curve consistency (std of y across seeds at each α) ===")
    print(f"  {'base':>22}  {'avg σ smile':>12}  {'avg σ jaw':>10}  {'avg mean smile':>14}  {'avg mean jaw':>12}")
    for base in sorted(per_base_smile_r2):
        base_rows = [r for r in rows if r["base"] == base]
        smile_arr = np.array([r["smile_y"] for r in base_rows])  # (n_seeds, 11)
        jaw_arr = np.array([r["jaw_y"] for r in base_rows])
        sigma_smile = smile_arr.std(axis=0).mean()
        sigma_jaw = jaw_arr.std(axis=0).mean()
        mean_smile = smile_arr.mean()
        mean_jaw = jaw_arr.mean()
        print(f"  {base:>22}  {sigma_smile:>12.4f}  {sigma_jaw:>10.4f}  {mean_smile:>14.3f}  {mean_jaw:>12.3f}")

    print(f"\nsaved → {OUT_DIR / 'per_group_stats.json'}")


if __name__ == "__main__":
    main()
