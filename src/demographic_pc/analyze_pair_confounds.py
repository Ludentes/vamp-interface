"""Project existing pair-averaging δ onto the ridge basis to quantify
demographic confounds carried by each (axis, base) edit direction.

For each cross-demographic pair measurement pkl (glasses, smile) we extract
delta_mix per (block, step), flatten across keys shared with the ridge basis,
and solve: δ ≈ Σ_k c_k · β[k] + residual. The c_k coefficients expose the
confound direction, residual tells us how much edit signal is outside the
ridge-explained subspace.

Outputs: table of c_k per (axis, base), residual fraction, and a rough
"confound imbalance" score comparing the two extreme bases.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RIDGE_PKL = ROOT / "output" / "demographic_pc" / "fluxspace_metrics" / "ridge_attn" / "ridge_basis.pkl"
CROSS = ROOT / "output" / "demographic_pc" / "fluxspace_metrics" / "crossdemo"

AXES = {
    "glasses": CROSS / "measurement",
    "smile":   CROSS / "smile" / "measurement",
}


def load_delta_mix(pkl_path: Path, keys: list[tuple[str, int]]) -> tuple[np.ndarray, list[int]] | None:
    """Load δ_mix per (block, step) aligned to ridge keys. Returns
    (delta[K_valid, D], valid_indices_into_keys). Pair measurements may only
    cover a sub-range of steps (start_percent>0), so we take the intersection
    rather than returning None on first miss."""
    d = pickle.load(pkl_path.open("rb"))
    rows, valid = [], []
    for idx, (bk, st) in enumerate(keys):
        entry = d["steps"].get(st, {}).get(bk)
        if entry is None or "delta_mix" not in entry:
            continue
        v = entry["delta_mix"]["mean_d"]
        rows.append(v.numpy() if hasattr(v, "numpy") else v)
        valid.append(idx)
    if not rows:
        return None
    return np.stack(rows).astype(np.float32), valid


def project_onto_ridge(delta: np.ndarray, valid: list[int],
                       W: np.ndarray, feat_names: list[str]) -> dict:
    """Least-squares project δ (K_valid, D) onto ridge basis W restricted
    to the same valid indices.

    Flatten each to vectors; solve c = argmin ||δ_vec - Σ c_k β_k_vec||².
    """
    Kf = W.shape[0]
    W_valid = W[:, valid, :]  # (K_feat, K_valid, D)
    delta_vec = delta.flatten()
    B_vec = W_valid.reshape(Kf, -1).T  # (K_valid*D, K_feat)
    # Solve normal equations
    c, *_ = np.linalg.lstsq(B_vec, delta_vec, rcond=None)
    reconstructed = B_vec @ c
    residual = delta_vec - reconstructed
    explained_frac = 1.0 - (residual @ residual) / (delta_vec @ delta_vec + 1e-12)
    return {
        "coefficients": {feat_names[i]: float(c[i]) for i in range(Kf)},
        "delta_norm": float(np.linalg.norm(delta_vec)),
        "explained_fraction": float(explained_frac),
        "residual_norm": float(np.linalg.norm(residual)),
    }


def main() -> None:
    with RIDGE_PKL.open("rb") as f:
        rb = pickle.load(f)
    feat_names = rb["feature_names"]
    keys = rb["keys"]
    W = rb["W"]  # (K_feat, K_keys, D)
    print(f"[confound] ridge basis: features={len(feat_names)} keys={len(keys)} D={W.shape[-1]}")

    results: dict = {"glasses": {}, "smile": {}}
    for axis_name, meas_dir in AXES.items():
        print(f"\n=== {axis_name} axis ===")
        pkls = sorted(meas_dir.glob("*_meas.pkl"))
        for p in pkls:
            base = p.stem.replace("_meas", "")
            loaded = load_delta_mix(p, keys)
            if loaded is None:
                print(f"  {base}: no delta_mix, skip")
                continue
            delta, valid = loaded
            proj = project_onto_ridge(delta, valid, W, feat_names)
            results[axis_name][base] = proj
            top = sorted(proj["coefficients"].items(),
                         key=lambda kv: abs(kv[1]), reverse=True)[:5]
            print(f"  {base}: expl_frac={proj['explained_fraction']:.3f}  "
                  f"‖δ‖={proj['delta_norm']:.1f}  top={[(k, f'{v:+.3f}') for k,v in top]}")

    # For each axis, compare the demographic-feature coefficients across bases.
    # Ideally, (axis-specific common direction) should dominate; demographic
    # coefficients should LOOK LIKE the base's signature (because δ = attn(with
    # attribute) − attn(without) carries the base's demographic profile).
    # If coefficients vary systematically with base demographics, confounds
    # are demographic-specific, not axis-clean.
    print("\n=== demographic confound pattern per axis ===")
    demographic_features = [f for f in feat_names
                            if f.startswith(("age=", "gender=", "eth="))]
    for axis_name, per_base in results.items():
        print(f"\n{axis_name}:")
        for feat in demographic_features:
            row = {b: per_base[b]["coefficients"][feat] for b in per_base}
            vals = np.array(list(row.values()))
            if len(vals) < 2:
                continue
            # Variance across bases: how much does this feature's contribution
            # change across bases? Large variance = confound depends on base.
            print(f"  {feat:>20}  "
                  f"mean={vals.mean():+.3f}  range=[{vals.min():+.3f},{vals.max():+.3f}]  "
                  f"std={vals.std():.3f}")

    # Key question: is there a COMMON axis direction across all bases?
    # Compute mean coefficient per feature across bases per axis. Features
    # with consistent non-zero coefficient across bases are axis-common.
    # Features with mean≈0 but high variance are base-specific confounds.
    print("\n=== axis-common vs base-specific features ===")
    for axis_name, per_base in results.items():
        print(f"\n{axis_name}:")
        common_scores = {}
        for feat in feat_names:
            row = np.array([per_base[b]["coefficients"][feat] for b in per_base])
            if len(row) < 2:
                continue
            # Consistency score: how stable is the coefficient across bases?
            # High mean/std ratio = axis-common. Low mean/std = base-specific.
            consistency = abs(row.mean()) / (row.std() + 1e-6)
            common_scores[feat] = (row.mean(), row.std(), consistency)
        for feat, (m, s, c) in sorted(common_scores.items(),
                                       key=lambda kv: -kv[1][2])[:8]:
            tag = "axis-common" if c > 2.0 else ("mixed" if c > 0.5 else "base-specific")
            print(f"  {feat:>20}  mean={m:+.3f}  std={s:.3f}  consist={c:.2f}  [{tag}]")

    # Save full table
    out = ROOT / "output" / "demographic_pc" / "fluxspace_metrics" / "ridge_attn" / "pair_confounds.pkl"
    with out.open("wb") as f:
        pickle.dump(results, f)
    print(f"\nsaved → {out}")


if __name__ == "__main__":
    main()
