"""Contrast-mean (causal) direction fitter — alternative to ridge.

Problem with `fit_nmf_directions_resid.py`: ridge minimizes prediction loss
on `delta_mix → atom_coeff`, which gives weights that correlate but can live
outside the training-data manifold. Injecting them at inference pushes
attention off-manifold → noise at large scale, no semantic edit at small
scale (smoke test verdict: "normal → blur → noise, no edit").

This script computes an **on-manifold** direction instead:

    direction[site, :] = Δ[delta_mix | high-atom vs low-atom training samples]

Concretely, at each top-importance site, take the mean of `delta_mix` across
the samples where atom coefficient is in the top quantile, minus the mean
over samples where it's in the bottom quantile. Each direction is by
construction a linear combination of observed training-data delta_mix
vectors — staying on-manifold means Flux has "seen" these patterns during
training-pair edits and should respond semantically rather than break down.

Reuses cached features + residualised Y from `fit_nmf_directions_resid`.
Outputs `directions_resid_causal.npz` matching the npz layout read by
`FluxSpaceDirectionInject`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.demographic_pc.fit_nmf_directions_resid import load_sources

ROOT = Path(__file__).resolve().parents[2]
NMF_DIR = ROOT / "models/blendshape_nmf"
OUT_PATH = NMF_DIR / "directions_resid_causal.npz"

K_SITES = 24
HIGH_QUANTILE = 0.80
LOW_QUANTILE = 0.20
DEAD_ATOMS = {0, 2, 3, 6}


def main():
    print("[causal] loading cache + residualised atom targets")
    delta_mix, _, Y, _, step_keys, block_keys = load_sources()
    N, S, B, D = delta_mix.shape
    n_atoms = Y.shape[1]
    print(f"  N={N}  atoms={n_atoms}  sites={S*B}  D={D}")

    # Reuse top-K sites from directions_resid.npz (same screening; saves ~10 min)
    print("\n[causal] reusing top-K sites from directions_resid.npz")
    prior = np.load(NMF_DIR / "directions_resid.npz", allow_pickle=True)
    top_sites = np.stack([prior[f"atom_{k:02d}_sites"][:K_SITES] for k in range(n_atoms)])
    print(f"  top_sites: {top_sites.shape}")

    flat = delta_mix.reshape(N, S * B, D)  # mmap, fp16
    print("\n[causal] fitting contrast-mean direction per atom")
    print(f"  {'#':>3}  {'n_high':>6}  {'n_low':>6}  {'|direction|_fro':>16}  {'max-row-norm':>12}")
    out = {}
    for k in range(n_atoms):
        if k in DEAD_ATOMS:
            # Still emit a zero direction so atom_id=k lookups don't fail
            out[f"atom_{k:02d}_direction"] = np.zeros((K_SITES, D), dtype=np.float32)
            out[f"atom_{k:02d}_sites"] = top_sites[k].astype(np.int64)
            out[f"atom_{k:02d}_cv_r2"] = np.array([0.0])
            continue

        y = Y[:, k]
        hi_cut = np.quantile(y, HIGH_QUANTILE)
        lo_cut = np.quantile(y, LOW_QUANTILE)
        hi_mask = y >= hi_cut
        lo_mask = y <= lo_cut
        sites = top_sites[k]

        # (n_hi, K, D) → mean → (K, D); same for lo.
        # mmap slice; cast to fp32 for the mean reduction.
        hi_slice = flat[hi_mask][:, sites].astype(np.float32)   # (n_hi, K, D)
        lo_slice = flat[lo_mask][:, sites].astype(np.float32)
        direction = hi_slice.mean(axis=0) - lo_slice.mean(axis=0)  # (K, D)

        out[f"atom_{k:02d}_direction"] = direction.astype(np.float32)
        out[f"atom_{k:02d}_sites"] = sites.astype(np.int64)
        fro = float(np.linalg.norm(direction))
        max_row = float(np.linalg.norm(direction, axis=1).max())
        out[f"atom_{k:02d}_max_row_norm"] = np.array([max_row])
        out[f"atom_{k:02d}_fro"] = np.array([fro])
        print(f"  {k:>3}  {int(hi_mask.sum()):>6}  {int(lo_mask.sum()):>6}  "
              f"{fro:>16.4g}  {max_row:>12.4g}")

    out["step_keys"] = np.array(step_keys)
    out["block_keys"] = np.array(block_keys, dtype=object)
    out["K_sites"] = np.array([K_SITES])
    out["high_quantile"] = np.array([HIGH_QUANTILE])
    out["low_quantile"] = np.array([LOW_QUANTILE])
    out["dead_atoms"] = np.array(sorted(DEAD_ATOMS))

    np.savez(OUT_PATH, **out)
    print(f"\n[causal] saved → {OUT_PATH}  (K={K_SITES}, {n_atoms} atoms)")


if __name__ == "__main__":
    main()
