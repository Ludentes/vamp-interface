"""Identity-residualised NMF decomposition.

Yesterday's finding: atoms were corpus-correlation-pure; today's rebalanced
corpus fixed expression-smile confounding but atoms still latch onto a single
demographic in their top-K. MediaPipe has per-demographic channel baselines,
so top-activating samples sort by who-has-highest-resting-baseline.

Fix: subtract the per-base mean blendshape vector (captures ethnicity, gender,
and age jointly since each of the 6 bases is one combination). Then NMF fits
on deviation-from-baseline, not absolute channel amplitude.

NMF needs non-negative input. Residuals are signed. Standard trick: stack
X = [X⁺ | X⁻] column-wise (shape N×2C), so each fitted atom carries signed
channel contributions.

Also emits sample-level metadata (base_id, ethnicity, gender, age) so we can
verify post-hoc that atoms span demographics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import NMF

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "output/demographic_pc/fluxspace_metrics"
OUT_DIR = ROOT / "models/blendshape_nmf"

CORPUS_SOURCES = [
    METRICS / "bootstrap_v1/blendshapes.json",
    METRICS / "crossdemo/smile/alpha_interp/blendshapes.json",
    METRICS / "crossdemo/smile/smile_inphase/blendshapes.json",
    METRICS / "crossdemo/smile/jaw_inphase/blendshapes.json",
    METRICS / "crossdemo/smile/intensity_full/blendshapes.json",
    METRICS / "crossdemo/anger/rebalance/blendshapes.json",
    METRICS / "crossdemo/surprise/rebalance/blendshapes.json",
    METRICS / "crossdemo/disgust/rebalance/blendshapes.json",
    METRICS / "crossdemo/pucker/rebalance/blendshapes.json",
    METRICS / "crossdemo/lip_press/rebalance/blendshapes.json",
]

# Base-id → (ethnicity, gender, age). All 6 rendering bases across the corpus.
BASE_META: dict[str, tuple[str, str, str]] = {
    "asian_m":          ("asian",     "m", "adult"),
    "black_f":          ("black",     "f", "adult"),
    "european_m":       ("european",  "m", "adult"),
    "elderly_latin_m":  ("latin",     "m", "elderly"),
    "young_european_f": ("european",  "f", "young"),
    "southasian_f":     ("southasian","f", "adult"),
}

K_SWEEP = [8, 10, 12, 14, 16, 20]
MIN_CHANNEL_STD = 0.01


def parse_base(key: str) -> str | None:
    """Handle both 'base/rest.png' (most sources) and bootstrap-style
    'category/base_XX_tag_sYYYY.png'."""
    if "/" in key:
        top = key.split("/", 1)[0]
        if top in BASE_META:
            return top
    fname = key.split("/")[-1]
    for b in BASE_META:
        if fname.startswith(b + "_"):
            return b
    return None


def _load_raw():
    samples: dict[str, dict[str, float]] = {}
    for src in CORPUS_SOURCES:
        if not src.exists():
            print(f"  [skip] {src}")
            continue
        data = json.loads(src.read_text())
        parts = src.parts
        tag = f"{parts[-3]}/{parts[-2]}" if parts[-2] == "rebalance" else parts[-2]
        for rel, scores in data.items():
            samples[f"{tag}/{rel}"] = scores
    return samples


def load_and_residualise():
    samples = _load_raw()
    channels = sorted({k for s in samples.values() for k in s.keys()})
    sample_ids = list(samples.keys())
    X = np.zeros((len(sample_ids), len(channels)), dtype=np.float64)
    bases: list[str] = []
    drop = []
    for i, sid in enumerate(sample_ids):
        # Tag may be 1 or 2 path components (e.g. "anger/rebalance"). Walk
        # the prefix successively stripping one path piece until we find a
        # base-prefixed path component.
        base = None
        parts = sid.split("/")
        for start in range(len(parts)):
            candidate = "/".join(parts[start:])
            base = parse_base(candidate)
            if base is not None:
                break
        if base is None:
            drop.append(sid)
            base = "unknown"
        bases.append(base)
        for j, ch in enumerate(channels):
            X[i, j] = samples[sid].get(ch, 0.0)

    print(f"  [load] {len(sample_ids)} samples, {len(channels)} channels")
    if drop:
        print(f"  [warn] {len(drop)} samples with unparsed base: e.g. {drop[0]}")

    # Drop samples we couldn't tag — residualisation requires a base.
    mask = np.array([b != "unknown" for b in bases])
    X = X[mask]
    sample_ids = [s for s, m in zip(sample_ids, mask) if m]
    bases = [b for b, m in zip(bases, mask) if m]

    # Per-base mean + std (z-score). Higher-moment (variance) differences
    # between demographics mean the bigger negative residuals
    # mechanically land on whichever base has the widest spread in a
    # channel, so mean-only centering still lets anti-direction atoms
    # lock to one base (verified 2026-04-22 — #01, #08, #12 were single-
    # demographic). Dividing by per-base std removes that asymmetry.
    unique_bases = sorted(set(bases))
    base_idx = {b: i for i, b in enumerate(unique_bases)}
    b_arr = np.array([base_idx[b] for b in bases])
    mu = np.zeros((len(unique_bases), X.shape[1]))
    sigma = np.zeros_like(mu)
    for bi in range(len(unique_bases)):
        Xb = X[b_arr == bi]
        mu[bi] = Xb.mean(axis=0)
        sigma[bi] = Xb.std(axis=0)
    sigma_safe = np.where(sigma < 1e-4, 1.0, sigma)
    X_res = (X - mu[b_arr]) / sigma_safe[b_arr]

    # Channel pruning on residuals (drop near-zero-variance channels)
    stds = X_res.std(axis=0)
    keep = stds >= MIN_CHANNEL_STD
    X_res_p = X_res[:, keep]
    channels_p = [c for c, k in zip(channels, keep) if k]
    print(f"  [prune] {X_res.shape[1]} → {X_res_p.shape[1]} channels")
    print(f"  [base-mean] per-base baseline subtracted ({len(unique_bases)} bases)")

    # Split signed residual into non-negative [X⁺ | X⁻]
    X_pos = np.clip(X_res_p, 0.0, None)
    X_neg = np.clip(-X_res_p, 0.0, None)
    X_stack = np.concatenate([X_pos, X_neg], axis=1)
    channels_stack = [f"{c}(+)" for c in channels_p] + [f"{c}(-)" for c in channels_p]
    print(f"  [stack] X_pos|X_neg shape = {X_stack.shape}")

    meta = {
        "sample_ids": sample_ids,
        "bases": bases,
        "unique_bases": unique_bases,
        "base_meta": {b: BASE_META.get(b, ("?", "?", "?")) for b in unique_bases},
        "channels_raw": channels_p,
        "channels_stack": channels_stack,
        "n_pruned_channels": int((~keep).sum()),
    }
    return X_stack, channels_stack, meta, mu, sigma, channels


def ve(X, Xh):
    num = ((X - Xh) ** 2).sum()
    den = ((X - X.mean(0)) ** 2).sum() + 1e-12
    return float(1.0 - num / den)


def fit(X, k, alpha=0.0, seed=0):
    model = NMF(n_components=k, init="nndsvda", solver="cd",
                beta_loss="frobenius", l1_ratio=0.5,
                alpha_W=alpha, alpha_H=alpha,
                max_iter=2000, tol=1e-5, random_state=seed)
    H = model.fit_transform(X)
    W = model.components_
    return H, W, ve(X, H @ W), float((W > 1e-4).sum(axis=1).mean())


def main():
    print("[resid-nmf] loading + residualising (z-score per base)")
    X, channels_stack, meta, mu, sigma, channels_full = load_and_residualise()

    print("\n[resid-nmf] k-sweep (unregularised)")
    print(f"  {'k':>3}  {'VE':>7}  {'mean_support':>12}")
    ves = {}
    for k in K_SWEEP:
        _, _, v, sup = fit(X, k)
        ves[k] = v
        print(f"  {k:>3}  {v:>7.4f}  {sup:>12.2f}")

    # Pick k by VE knee (same rule as blendshape_decomposition)
    knee_k = K_SWEEP[0]
    for i in range(1, len(K_SWEEP)):
        if ves[K_SWEEP[i]] - ves[K_SWEEP[i - 1]] < 0.015:
            knee_k = K_SWEEP[i - 1]
            break
    else:
        knee_k = K_SWEEP[-1]
    print(f"\n[resid-nmf] chosen k* = {knee_k}")

    print("\n[resid-nmf] alpha sweep at chosen k")
    for a in [0.0, 0.0005, 0.001, 0.005]:
        _, _, v, sup = fit(X, knee_k, alpha=a)
        print(f"  alpha={a:<7.4f} VE={v:.4f} support={sup:.2f}")

    print("\n[resid-nmf] final fit")
    H, W, v, sup = fit(X, knee_k, alpha=0.001)
    print(f"  VE={v:.4f} support={sup:.2f}")

    # Top-signed-channels per atom
    half = len(channels_stack) // 2
    print("\n[resid-nmf] atom signed loadings (top 6)")
    for k in range(knee_k):
        w = W[k]
        idx = np.argsort(-w)[:6]
        lines = ", ".join(f"{channels_stack[i]}({w[i]:.2f})" for i in idx)
        print(f"  #{k:02d}  {lines}")

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "W_nmf_resid.npy", W)
    np.save(OUT_DIR / "H_nmf_resid.npy", H)
    np.save(OUT_DIR / "mu_base_resid.npy", mu)
    np.save(OUT_DIR / "sigma_base_resid.npy", sigma)
    (OUT_DIR / "manifest_resid.json").write_text(json.dumps({
        "k": knee_k,
        "alpha": 0.001,
        "ve": v,
        "mean_support": sup,
        "n_samples": H.shape[0],
        "channels_stack": channels_stack,
        "channels_raw": meta["channels_raw"],
        "channels_full": channels_full,
        "bases": meta["bases"],
        "unique_bases": meta["unique_bases"],
        "base_meta": meta["base_meta"],
        "sample_ids": meta["sample_ids"],
        "corpus_sources": [str(s.relative_to(ROOT)) for s in CORPUS_SOURCES if s.exists()],
    }, indent=2))
    print(f"\n[save] W_nmf_resid.npy, H_nmf_resid.npy, mu_base_resid.npy, "
          f"manifest_resid.json → {OUT_DIR}")


if __name__ == "__main__":
    main()
