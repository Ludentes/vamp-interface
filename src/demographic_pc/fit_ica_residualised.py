"""PCA→FastICA on z-score-residualised blendshapes for comparison with NMF.

NMF needs non-negative input so the residualised pipeline stacks [X⁺ | X⁻]
(76 columns). ICA is naturally signed, so it runs directly on the 38-column
residual matrix.

Reports:
  * k-sweep (PCA VE → ICA)
  * Atom classification (AU-plausible / composite / noise) like
    `blendshape_decomposition.py`
  * Per-atom top-5 demographic spread (same metric used for NMF atoms)
  * Saves W_ica_resid.npy + H_ica_resid.npy + manifest_ica_resid.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA, FastICA

from src.demographic_pc.fit_nmf_residualised import (
    CORPUS_SOURCES, _load_raw, parse_base, BASE_META,
)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "models/blendshape_nmf"

K_SWEEP = [10, 12, 14, 16, 20]
MIN_CHANNEL_STD = 0.01


def load_residualised():
    samples = _load_raw()
    channels = sorted({k for s in samples.values() for k in s.keys()})
    sample_ids = list(samples.keys())
    X = np.zeros((len(sample_ids), len(channels)))
    bases: list[str] = []
    for i, sid in enumerate(sample_ids):
        parts = sid.split("/")
        base = None
        for start in range(len(parts)):
            base = parse_base("/".join(parts[start:]))
            if base is not None:
                break
        if base is None:
            base = "unknown"
        bases.append(base)
        for j, c in enumerate(channels):
            X[i, j] = samples[sid].get(c, 0.0)
    mask = np.array([b != "unknown" for b in bases])
    X = X[mask]
    sample_ids = [s for s, m in zip(sample_ids, mask) if m]
    bases = [b for b, m in zip(bases, mask) if m]

    unique = sorted(set(bases))
    idx = {b: i for i, b in enumerate(unique)}
    b_arr = np.array([idx[b] for b in bases])
    mu = np.zeros((len(unique), X.shape[1]))
    sigma = np.zeros_like(mu)
    for bi in range(len(unique)):
        Xb = X[b_arr == bi]
        mu[bi] = Xb.mean(0)
        sigma[bi] = Xb.std(0)
    sig_safe = np.where(sigma < 1e-4, 1.0, sigma)
    X_res = (X - mu[b_arr]) / sig_safe[b_arr]

    # Prune low-std channels (after residualisation → channels with no
    # structure left contribute noise to ICA).
    stds = X_res.std(0)
    keep = stds >= MIN_CHANNEL_STD
    X_res_p = X_res[:, keep]
    channels_p = [c for c, k in zip(channels, keep) if k]
    print(f"  [load] N={X_res_p.shape[0]} channels={X_res_p.shape[1]}")
    return X_res_p, channels_p, sample_ids, bases, unique


def evaluate_k(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, float]:
    pca = PCA(n_components=k, random_state=0).fit(X)
    ve = float(pca.explained_variance_ratio_.sum())
    ica = FastICA(n_components=k, random_state=0, whiten="unit-variance",
                  max_iter=500, tol=1e-4).fit(X)
    H = ica.transform(X)
    W = ica.components_
    return W, H, ve


def spread_report(H: np.ndarray, bases: list[str], top_n: int = 5) -> list[int]:
    """Per-atom #distinct bases in top-n activators (by |coeff|)."""
    out = []
    for k in range(H.shape[1]):
        top = np.argsort(-np.abs(H[:, k]))[:top_n]
        out.append(len({bases[int(i)] for i in top}))
    return out


def main():
    print("[ica-resid] loading residualised data")
    X, channels, sample_ids, bases, unique_bases = load_residualised()

    print("\n[ica-resid] k-sweep")
    best_k = None
    best_ve = 0.0
    for k in K_SWEEP:
        W, H, ve = evaluate_k(X, k)
        spread = spread_report(H, bases)
        mean_spread = float(np.mean(spread))
        pinned = sum(1 for s in spread if s == 1)
        print(f"  k={k:<3}  PCA VE={ve:.3f}  mean spread={mean_spread:.2f}  "
              f"pinned(1/5)={pinned}/{k}")
        # Pick the biggest k where we still keep >50% atoms with spread ≥2
        if sum(1 for s in spread if s >= 2) >= k * 0.6:
            best_k = k
            best_ve = ve

    if best_k is None:
        best_k = K_SWEEP[0]
        print(f"[ica-resid] no k passed spread filter — falling back to {best_k}")
    print(f"\n[ica-resid] chosen k* = {best_k}")

    W, H, ve = evaluate_k(X, best_k)
    spread = spread_report(H, bases)
    print(f"  VE={ve:.3f}")
    print(f"\n[ica-resid] atom top-5 loadings + demographic spread")
    for k in range(best_k):
        idx = np.argsort(-np.abs(W[k]))[:5]
        loadings = ", ".join(f"{channels[i]}({W[k, i]:+.2f})" for i in idx)
        print(f"  #{k:02d}  spread={spread[k]}/5  {loadings}")

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "W_ica_resid.npy", W)
    np.save(OUT_DIR / "H_ica_resid.npy", H)
    (OUT_DIR / "manifest_ica_resid.json").write_text(json.dumps({
        "k": best_k,
        "ve": ve,
        "n_samples": int(H.shape[0]),
        "channels": channels,
        "sample_ids": sample_ids,
        "bases": bases,
        "unique_bases": unique_bases,
        "base_meta": {b: list(BASE_META.get(b, ("?", "?", "?"))) for b in unique_bases},
        "spread_top5": spread,
        "corpus_sources": [str(s.relative_to(ROOT)) for s in CORPUS_SOURCES if s.exists()],
    }, indent=2))
    print(f"\n[ica-resid] saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
