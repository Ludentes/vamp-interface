"""Stability check — run NMF at k=11, alpha=0 across N random seeds,
match atoms via Hungarian assignment on cosine similarity, report
per-atom cross-seed stability.

Rule of thumb: an atom is "stable" if it has cos ≥ 0.95 matching atom
in ≥ 4 of the other seeds.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF

from src.demographic_pc.blendshape_decomposition import (
    classify_atom, load_corpus, prune_channels, ve,
)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "models/blendshape_nmf"
K = 11
N_SEEDS = 5


def fit_nmf_unreg(X: np.ndarray, k: int, seed: int,
                  init: str = "random") -> tuple[np.ndarray, np.ndarray, float]:
    """Default init="random" so different seeds actually diverge. Pass
    init="nndsvda" for the deterministic production fit."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = NMF(n_components=k, init=init, solver="cd",
                beta_loss="frobenius", max_iter=3000, tol=1e-5,
                random_state=seed)
        H = m.fit_transform(X)
        W = m.components_
    return W, H, ve(X, H @ W)


def cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A is (k, C), B is (k, C). Return (k, k) of pairwise cosine."""
    an = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return an @ bn.T


def main() -> None:
    print("[stability] loading corpus")
    X, channels, _ = load_corpus()
    X, channels = prune_channels(X, channels)
    print(f"[stability] X shape = {X.shape}")

    print(f"\n[stability] fitting NMF × {N_SEEDS} seeds @ k={K}, init=random")
    runs = []
    for s in range(N_SEEDS):
        W, H, v = fit_nmf_unreg(X, K, seed=s, init="random")
        norms = np.linalg.norm(W, axis=1)
        print(f"  seed={s}  VE={v:.4f}  atom norms min/max={norms.min():.3f}/{norms.max():.3f}")
        runs.append({"W": W, "H": H, "ve": v, "seed": s})

    # Also run the deterministic nndsvda fit as the reference (canonical basis).
    W_ref, _, ve_ref = fit_nmf_unreg(X, K, seed=0, init="nndsvda")
    print(f"  nndsvda  VE={ve_ref:.4f}  (reference canonical fit)")

    # Anchor = nndsvda reference. Hungarian-match each random run to it.
    anchor = W_ref
    matched_cos: list[np.ndarray] = []
    for other in runs:  # compare all 5 random-init runs against nndsvda anchor
        C = cosine_matrix(anchor, other["W"])
        # Want to MAXIMISE cosine → minimise -cos
        row_ind, col_ind = linear_sum_assignment(-C)
        # row_ind is 0..k-1; col_ind is permutation in other run
        cos_values = np.array([C[r, c] for r, c in zip(row_ind, col_ind)])
        matched_cos.append(cos_values)
        other["perm"] = col_ind.tolist()

    # Per-anchor-atom stability: mean cos across the 4 other seeds
    matched_stack = np.stack(matched_cos, axis=0)  # (n_random_seeds, k)
    n_compares = matched_stack.shape[0]
    mean_cos = matched_stack.mean(axis=0)
    min_cos = matched_stack.min(axis=0)
    n_ge_095 = (matched_stack >= 0.95).sum(axis=0)
    n_ge_080 = (matched_stack >= 0.80).sum(axis=0)

    # Classify atoms from the anchor run
    atom_classes = [classify_atom(anchor[i], channels) for i in range(K)]

    print(f"\n[stability] per-atom match — {n_compares} random-init seeds vs nndsvda anchor")
    print(f"  {'#':>3}  {'mean cos':>9}  {'min cos':>8}  {'n≥0.95':>7}  {'n≥0.80':>7}  "
          f"{'class':<18}  top channels")
    for i in range(K):
        info = atom_classes[i]
        top = ", ".join(info["top_channels"][:3])
        print(f"  {i:>3}  {mean_cos[i]:>9.3f}  {min_cos[i]:>8.3f}  "
              f"{n_ge_095[i]}/{n_compares:>3}    "
              f"{n_ge_080[i]}/{n_compares:>3}    "
              f"{info['classification']:<18}  {top}")

    stable = int((n_ge_095 >= n_compares - 1).sum())  # matches in ≥4 of 5
    moderate = int(((n_ge_095 >= 3) & (n_ge_095 < n_compares - 1)).sum())
    unstable = int((n_ge_095 < 3).sum())
    print(f"\n[stability] summary (cos ≥ 0.95 criterion): "
          f"stable={stable}/{K}  moderate={moderate}/{K}  unstable={unstable}/{K}")

    # Save the anchor basis as canonical
    np.save(OUT_DIR / "W_nmf_k11.npy", anchor)
    stability_report = {
        "k": K,
        "n_seeds": N_SEEDS,
        "ve_per_seed": [r["ve"] for r in runs],
        "anchor_seed": 0,
        "mean_cos": mean_cos.tolist(),
        "min_cos": min_cos.tolist(),
        "n_matches_ge_095": n_ge_095.tolist(),
        "stability_counts": {"stable": stable, "moderate": moderate,
                             "unstable": unstable},
        "anchor_atoms": atom_classes,
    }
    (OUT_DIR / "stability_k11.json").write_text(json.dumps(stability_report, indent=2))
    print(f"\n[stability] wrote → {OUT_DIR / 'W_nmf_k11.npy'}, "
          f"{OUT_DIR / 'stability_k11.json'}")


if __name__ == "__main__":
    main()
