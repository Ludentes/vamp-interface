"""Compare ridge-fit and causal-contrast direction tensors per atom.

For each atom, prints:
  * magnitude ratio (causal / ridge)
  * cosine similarity (flattened directions)
  * overlap in top-K site indices
  * fraction of direction mass in shared vs non-shared sites

Intended as a quick diagnostic after the morning validation run. If ridge
and causal overlap >80% with cos > 0.8, they're approximately the same
direction and the smoke difference was purely magnitude. If cos < 0.3,
they point in qualitatively different directions and validation should
tell us which one points where Flux actually responds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NMF_DIR = ROOT / "models/blendshape_nmf"

DEAD_ATOMS = {0, 2, 3, 6}


def main():
    ridge = np.load(NMF_DIR / "directions_resid.npz", allow_pickle=True)
    causal = np.load(NMF_DIR / "directions_resid_causal.npz", allow_pickle=True)

    n_atoms = 20
    print(f"{'atom':>4}  {'|ridge|':>9}  {'|causal|':>10}  {'ratio':>7}  {'cos':>6}  "
          f"{'top-K shared':>13}")
    print("-" * 70)
    for k in range(n_atoms):
        if k in DEAD_ATOMS:
            continue
        dr = ridge[f"atom_{k:02d}_direction"].astype(np.float32).flatten()
        dc = causal[f"atom_{k:02d}_direction"].astype(np.float32).flatten()
        nr = np.linalg.norm(dr)
        nc = np.linalg.norm(dc)
        ratio = nc / max(nr, 1e-9)
        cos = float(dr @ dc / max(nr * nc, 1e-12))
        sr_arr = ridge[f"atom_{k:02d}_sites"]
        sc_arr = causal[f"atom_{k:02d}_sites"]
        assert len(sr_arr) == len(set(sr_arr.tolist())), \
            f"duplicate sites in ridge atom_{k:02d}_sites"
        assert len(sc_arr) == len(set(sc_arr.tolist())), \
            f"duplicate sites in causal atom_{k:02d}_sites"
        assert dr.shape == dc.shape, \
            f"direction shape mismatch atom_{k:02d}: ridge {dr.shape} vs causal {dc.shape}"
        # cos is meaningful only when the two fits ORDER sites identically.
        # Both scripts build sites via `argsort(-|r2|)[:K]` from the same
        # r2_site matrix, so ordering should match. Verify.
        assert np.array_equal(sr_arr, sc_arr), \
            f"atom_{k:02d}: ridge and causal saved top-K sites in different ORDER — " \
            f"flattened cosine is meaningless. Realign before trusting."
        sr = set(sr_arr.tolist())
        sc = set(sc_arr.tolist())
        shared = len(sr & sc)
        print(f"{k:>4}  {nr:>9.4f}  {nc:>10.4f}  {ratio:>7.2f}  {cos:>+6.3f}  "
              f"{shared:>4}/{len(sr):<2} ({shared/len(sr)*100:.0f}%)")


if __name__ == "__main__":
    main()
