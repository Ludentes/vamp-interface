"""Ridge α × K_sites sweep on top of the residualised directions pipeline.

Reuses the existing per-site screening (already O(912) ridges) and only
varies the per-atom CV-ridge stage. Reports per-atom CV R² for each grid
cell; saves the best-per-atom config to
`models/blendshape_nmf/directions_resid_tuned.npz`.

Excludes the 4 dead atoms (#00/#02/#03/#06) to avoid trivial CV=1.0.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from src.demographic_pc.fit_nmf_directions_resid import (
    load_sources, per_site_screening,
)

ROOT = Path(__file__).resolve().parents[2]
NMF_DIR = ROOT / "models/blendshape_nmf"
OUT_PATH = NMF_DIR / "directions_resid_tuned.npz"

ALPHAS = [1.0, 10.0, 100.0, 1000.0]
K_SITES_GRID = [12, 24, 48]
DEAD_ATOMS = {0, 2, 3, 6}


def cv_r2(delta_mix, y, site_idx, alpha):
    N, S, B, D = delta_mix.shape
    flat = delta_mix.reshape(N, S * B, D)
    X = flat[:, site_idx].reshape(N, -1).astype(np.float32)
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    r2 = []
    for tr, te in kf.split(X):
        m = Ridge(alpha=alpha, random_state=0).fit(X[tr], y[tr])
        pred = m.predict(X[te])
        rss = ((y[te] - pred) ** 2).sum()
        tss = ((y[te] - y[te].mean()) ** 2).sum() + 1e-12
        r2.append(float(1.0 - rss / tss))
    return float(np.mean(r2)), float(np.std(r2))


def main():
    print("[tune] loading caches + residualised atoms")
    delta_mix, _, Y, _, step_keys, block_keys = load_sources()
    n_atoms = Y.shape[1]
    print(f"  N={delta_mix.shape[0]}  atoms={n_atoms}  sites={delta_mix.shape[1]*delta_mix.shape[2]}")

    print("\n[tune] per-site screening (once)")
    r2_site = per_site_screening(delta_mix, Y)

    live_atoms = [k for k in range(n_atoms) if k not in DEAD_ATOMS]
    print(f"\n[tune] grid: α={ALPHAS}  K_sites={K_SITES_GRID}  live={len(live_atoms)} atoms")

    # grid shape (n_live, n_alpha, n_K) of CV R²
    grid = np.zeros((len(live_atoms), len(ALPHAS), len(K_SITES_GRID)))
    for ai, alpha in enumerate(ALPHAS):
        for ki, K in enumerate(K_SITES_GRID):
            top_sites_all = np.argsort(-np.abs(r2_site), axis=1)[:, :K]
            for li, k in enumerate(live_atoms):
                mean, _ = cv_r2(delta_mix, Y[:, k], top_sites_all[k], alpha)
                grid[li, ai, ki] = mean
            mean_best_per_atom = grid[:, ai, ki].mean()
            print(f"  α={alpha:<6}  K={K:<3}  mean-CV={mean_best_per_atom:.3f}  "
                  f"marginal-count(<.65)={(grid[:, ai, ki] < 0.65).sum()}")

    # Per-atom best cell
    print(f"\n[tune] per-atom best (α, K):")
    out = {}
    best_live = []
    for li, k in enumerate(live_atoms):
        best_flat = np.argmax(grid[li].ravel())
        ai, ki = divmod(int(best_flat), len(K_SITES_GRID))
        best_alpha = ALPHAS[ai]
        best_K = K_SITES_GRID[ki]
        best_r2 = float(grid[li, ai, ki])
        baseline = float(grid[li, 0, 1])  # α=1, K=24 — the original
        delta = best_r2 - baseline
        print(f"  #{k:02d}  best α={best_alpha:<6}  K={best_K:<3}  "
              f"CV={best_r2:.3f}  (Δ vs α=1,K=24: {delta:+.3f})")

        # Refit final at best (α, K) and save direction
        top_sites = np.argsort(-np.abs(r2_site[k]))[:best_K]
        N, S, B, D = delta_mix.shape
        X = delta_mix.reshape(N, S * B, D)[:, top_sites].reshape(N, -1).astype(np.float32)
        final = Ridge(alpha=best_alpha, random_state=0).fit(X, Y[:, k])
        out[f"atom_{k:02d}_direction"] = final.coef_.reshape(best_K, D)
        out[f"atom_{k:02d}_sites"] = top_sites
        out[f"atom_{k:02d}_alpha"] = np.array([best_alpha])
        out[f"atom_{k:02d}_K"] = np.array([best_K])
        out[f"atom_{k:02d}_cv_r2"] = np.array([best_r2])
        best_live.append(best_r2)

    out["step_keys"] = np.array(step_keys)
    out["block_keys"] = np.array(block_keys, dtype=object)
    out["live_atoms"] = np.array(live_atoms)
    out["dead_atoms"] = np.array(sorted(DEAD_ATOMS))
    out["grid"] = grid
    out["alphas"] = np.array(ALPHAS)
    out["K_sites_grid"] = np.array(K_SITES_GRID)

    np.savez(OUT_PATH, **out)
    print(f"\n[tune] saved → {OUT_PATH}")
    print(f"  mean CV across live atoms (tuned): {np.mean(best_live):.3f}")
    print(f"  marginal-count(<.65) after tuning: {(np.array(best_live) < 0.65).sum()}")


if __name__ == "__main__":
    main()
