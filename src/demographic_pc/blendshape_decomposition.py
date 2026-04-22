"""Phase 2 — sparse NMF decomposition of the MediaPipe 52-channel
blendshape corpus into AU-like atoms.

Recipe in `docs/research/2026-04-22-blendshape-decomp-lit-read.md`
(Tripathi DFECS adapted to our non-negative, already-region-labeled
input).

Steps
-----
  1. Load all scored blendshape JSONs, stack into X ∈ ℝ^(N×52).
  2. Drop near-zero-variance channels; keep non-negativity.
  3. k-sweep: sparse NMF at k ∈ {6, 8, 10, 12, 14, 16, 20}; log VE.
  4. Pick k* at smallest k with VE ≥ 0.95.
  5. Final NMF at k*; PCA→FastICA at same k* for comparison.
  6. Classify atoms (AU-plausible / composite / noise) from top-loading
     channels.
  7. Save basis W, channel names, atom classification to
     `models/blendshape_nmf/`.
"""

from __future__ import annotations

import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import NMF, FastICA, PCA

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "output/demographic_pc/fluxspace_metrics"
OUT_DIR = ROOT / "models/blendshape_nmf"
ANALYSIS_DIR = ROOT / "output/demographic_pc/fluxspace_metrics/analysis"

CORPUS_SOURCES = [
    METRICS / "bootstrap_v1/blendshapes.json",
    METRICS / "crossdemo/smile/alpha_interp/blendshapes.json",
    METRICS / "crossdemo/smile/smile_inphase/blendshapes.json",
    METRICS / "crossdemo/smile/jaw_inphase/blendshapes.json",
    METRICS / "crossdemo/smile/intensity_full/blendshapes.json",
]

K_SWEEP = [6, 8, 10, 12, 14, 16, 20]
VAR_TARGET = 0.95
MIN_CHANNEL_STD = 0.01
ANATOMICAL_GROUPS = {
    "brow": ("brow",),
    "eye": ("eye", "squint"),
    "cheek": ("cheek",),
    "nose": ("nose",),
    "mouth-pull": ("mouthSmile", "mouthDimple", "mouthUpperUp",
                   "mouthLowerDown", "mouthPress", "mouthStretch"),
    "mouth-open": ("jaw", "mouthOpen", "mouthClose", "mouthFunnel",
                   "mouthPucker", "mouthRollLower", "mouthRollUpper",
                   "mouthShrug"),
    "mouth-corner": ("mouthFrown", "mouthLeft", "mouthRight"),
    "tongue": ("tongue",),
}


def load_corpus() -> tuple[np.ndarray, list[str], list[str]]:
    """Return (X, channel_names, sample_ids)."""
    samples: dict[str, dict[str, float]] = {}
    for src in CORPUS_SOURCES:
        if not src.exists():
            print(f"  [skip] missing: {src}")
            continue
        data = json.loads(src.read_text())
        tag = src.parent.name  # e.g. 'alpha_interp', 'intensity_full'
        for rel, scores in data.items():
            samples[f"{tag}/{rel}"] = scores
    print(f"  [load] {len(samples)} scored samples across {len(CORPUS_SOURCES)} sources")

    channels = sorted({k for s in samples.values() for k in s.keys()})
    sample_ids = list(samples.keys())
    X = np.zeros((len(sample_ids), len(channels)), dtype=np.float64)
    for i, sid in enumerate(sample_ids):
        for j, ch in enumerate(channels):
            X[i, j] = samples[sid].get(ch, 0.0)
    return X, channels, sample_ids


def prune_channels(X: np.ndarray, channels: list[str]) -> tuple[np.ndarray, list[str]]:
    stds = X.std(axis=0)
    keep = stds >= MIN_CHANNEL_STD
    X_p = X[:, keep]
    channels_p = [c for c, k in zip(channels, keep) if k]
    dropped = [c for c, k in zip(channels, keep) if not k]
    print(f"  [prune] {X.shape[1]} → {X_p.shape[1]} channels "
          f"(dropped {len(dropped)}: {', '.join(dropped[:6])}{'…' if len(dropped)>6 else ''})")
    return X_p, channels_p


def ve(X: np.ndarray, X_hat: np.ndarray) -> float:
    num = ((X - X_hat) ** 2).sum()
    den = ((X - X.mean(axis=0)) ** 2).sum() + 1e-12
    return float(1.0 - num / den)


def fit_nmf(X: np.ndarray, k: int, seed: int = 0,
            alpha_W: float = 0.0, alpha_H: float = 0.0,
            l1_ratio: float = 0.5) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit NMF. Default is unregularised; pass small alpha for sparsity.

    Our blendshape values live in [0,1] and are typically <0.1 on most
    channels, so alpha above ~0.01 swamps the data gradient and atoms
    collapse to zero.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = NMF(
            n_components=k,
            init="nndsvda",
            solver="cd",
            beta_loss="frobenius",
            l1_ratio=l1_ratio,
            alpha_W=alpha_W,
            alpha_H=alpha_H,
            max_iter=2000,
            tol=1e-5,
            random_state=seed,
        )
        H = model.fit_transform(X)   # (N, k)
        W = model.components_         # (k, C)
    X_hat = H @ W
    return W, H, ve(X, X_hat)


def fit_pca_ica(X: np.ndarray, k: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, float]:
    # Center for PCA
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    pca = PCA(n_components=k, whiten=True, random_state=seed)
    Z = pca.fit_transform(Xc)
    X_hat = pca.inverse_transform(Z) + mean
    # Rotate via FastICA
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ica = FastICA(n_components=k, random_state=seed, max_iter=2000, tol=1e-5)
        _ = ica.fit_transform(Z)
    # Reconstruct ICA atoms in original (channel) space.
    # FastICA decomposes Z ≈ S W_ica; the mixing matrix in original
    # space combines PCA inverse transform with ICA's mixing. For our
    # purposes here we want atom loadings as channel-space vectors.
    # Approximation: rotate PCA components by ICA mixing.
    # PCA.components_: (k, C) principal axes.
    # ICA.mixing_: (k, k).
    W_ica = ica.mixing_.T @ pca.components_  # (k, C)
    return W_ica, Z, ve(X, X_hat)


def _channel_group(ch: str) -> str:
    for group_name, tokens in ANATOMICAL_GROUPS.items():
        if any(tok.lower() in ch.lower() for tok in tokens):
            return group_name
    return "unknown"


def classify_atom(atom: np.ndarray, channels: list[str]) -> dict:
    """Classify by the *weight-weighted* group distribution among dominant
    channels (those with |w| >= 0.25 × max |w|). A handful of small eye
    loadings don't pollute a dominant mouth atom."""
    abs_atom = np.abs(atom)
    if abs_atom.max() < 1e-6:
        return {"top_channels": [], "top_weights": [], "groups_hit": [],
                "dominant_group_mass": 0.0, "support": 0,
                "classification": "dead"}
    # Top-k display (k=4)
    idx = np.argsort(-abs_atom)[:4]
    top_channels = [channels[i] for i in idx]
    top_weights = [float(atom[i]) for i in idx]
    # Dominant channels (for classification): |w| >= 25% of peak
    dom = abs_atom >= 0.25 * abs_atom.max()
    dom_weights = abs_atom[dom]
    dom_groups = [_channel_group(channels[i]) for i in np.where(dom)[0]]
    mass_by_group: dict[str, float] = defaultdict(float)
    for g, w in zip(dom_groups, dom_weights):
        mass_by_group[g] += float(w)
    total_mass = sum(mass_by_group.values())
    dominant_mass_frac = max(mass_by_group.values()) / (total_mass + 1e-12)
    # Support at the usual 5% threshold (reporting)
    support = int((abs_atom >= 0.05 * abs_atom.max()).sum())
    # Classification rule: a single group must carry >=70% of dominant mass
    groups_hit = sorted(mass_by_group.keys())
    if dominant_mass_frac >= 0.70 and len(mass_by_group) <= 2:
        classification = "AU-plausible"
    elif dominant_mass_frac >= 0.50 and len(mass_by_group) <= 3:
        classification = "composite-2region"
    elif len(mass_by_group) >= 4 or support >= 20:
        classification = "noise"
    else:
        classification = "composite-broad"
    return {
        "top_channels": top_channels,
        "top_weights": top_weights,
        "groups_hit": groups_hit,
        "dominant_group_mass": float(dominant_mass_frac),
        "support": support,
        "classification": classification,
    }


def main() -> None:
    print("[nmf] loading corpus")
    X, channels, sample_ids = load_corpus()
    X, channels = prune_channels(X, channels)
    N, C = X.shape
    print(f"[nmf] X shape = ({N}, {C})")

    # --- k-sweep (unregularised NMF to find VE knee) ---
    print("\n[nmf] k-sweep — unregularised NMF (VE knee)")
    print(f"  {'k':>4}  {'VE':>7}  {'mean_support':>12}")
    sweep = {}
    for k in K_SWEEP:
        W, H, v = fit_nmf(X, k, alpha_W=0.0, alpha_H=0.0)
        support = np.mean(((np.abs(W) >= 0.05 * np.abs(W).max(axis=1, keepdims=True)).sum(axis=1)))
        sweep[k] = {"ve": v, "mean_support": float(support)}
        print(f"  {k:>4}  {v:>7.4f}  {support:>12.2f}")

    # pick k*: smallest k with VE >= 0.95, otherwise largest
    k_star = next((k for k in K_SWEEP if sweep[k]["ve"] >= VAR_TARGET), K_SWEEP[-1])
    print(f"\n[nmf] chosen k* = {k_star}  (VE = {sweep[k_star]['ve']:.4f})")

    # --- sparsity sweep at k* to pick alpha ---
    print(f"\n[nmf] sparsity sweep @ k={k_star}")
    print(f"  {'alpha':>7}  {'VE':>7}  {'mean_support':>12}")
    alpha_sweep = {}
    for a in [0.0, 0.0005, 0.001, 0.005, 0.01, 0.02]:
        W, H, v = fit_nmf(X, k_star, alpha_W=a, alpha_H=a)
        support = np.mean(((np.abs(W) >= 0.05 * np.abs(W).max(axis=1, keepdims=True)).sum(axis=1)))
        alpha_sweep[a] = {"ve": v, "mean_support": float(support)}
        print(f"  {a:>7.4f}  {v:>7.4f}  {support:>12.2f}")

    # Pick largest alpha where VE stays within 5% of unregularised baseline.
    ve_baseline = alpha_sweep[0.0]["ve"]
    usable = [(a, d["ve"]) for a, d in alpha_sweep.items()
              if d["ve"] >= ve_baseline - 0.05 and a > 0]
    alpha_star = max(usable, key=lambda x: x[0])[0] if usable else 0.0
    print(f"\n[nmf] chosen alpha* = {alpha_star}  "
          f"(VE = {alpha_sweep[alpha_star]['ve']:.4f}, "
          f"baseline {ve_baseline:.4f})")

    # --- final NMF at k*, alpha* ---
    print(f"\n[nmf] final fit @ k={k_star}, alpha={alpha_star}")
    W_nmf, H_nmf, ve_nmf = fit_nmf(X, k_star, alpha_W=alpha_star, alpha_H=alpha_star)
    # Sort atoms by total energy (sum of weights) for reproducible ordering
    energy = W_nmf.sum(axis=1)
    order = np.argsort(-energy)
    W_nmf = W_nmf[order]
    H_nmf = H_nmf[:, order]

    # --- classify atoms ---
    print(f"\n[nmf] atom classification (top-6 channels per atom)")
    nmf_atoms = []
    for i, atom in enumerate(W_nmf):
        info = classify_atom(atom, channels)
        nmf_atoms.append(info)
        top = ", ".join(f"{c}({w:.2f})" for c, w in zip(info["top_channels"][:4],
                                                         info["top_weights"][:4]))
        print(f"  #{i:02d}  [{info['classification']:<18}] "
              f"groups={info['groups_hit']}  support={info['support']}  "
              f"top: {top}")

    # --- comparison: PCA→ICA at same k ---
    print(f"\n[ica] PCA→FastICA @ k={k_star} (comparison)")
    W_ica, _, ve_ica = fit_pca_ica(X, k_star)
    print(f"  VE = {ve_ica:.4f}")

    ica_atoms = []
    for i, atom in enumerate(W_ica):
        info = classify_atom(atom, channels)
        ica_atoms.append(info)
        top = ", ".join(f"{c}({w:+.2f})" for c, w in zip(info["top_channels"][:4],
                                                          info["top_weights"][:4]))
        print(f"  #{i:02d}  [{info['classification']:<18}] "
              f"groups={info['groups_hit']}  support={info['support']}  "
              f"top: {top}")

    # --- head-to-head summary ---
    print("\n[compare] NMF vs PCA→ICA at matched k")
    nmf_counts = defaultdict(int)
    ica_counts = defaultdict(int)
    for a in nmf_atoms: nmf_counts[a["classification"]] += 1
    for a in ica_atoms: ica_counts[a["classification"]] += 1
    all_classes = sorted(set(nmf_counts) | set(ica_counts))
    print(f"  {'class':<22}  {'NMF':>5}  {'ICA':>5}")
    for c in all_classes:
        print(f"  {c:<22}  {nmf_counts[c]:>5}  {ica_counts[c]:>5}")
    nmf_mean_support = np.mean([a["support"] for a in nmf_atoms])
    ica_mean_support = np.mean([a["support"] for a in ica_atoms])
    print(f"  mean atom support      {nmf_mean_support:>5.2f}  {ica_mean_support:>5.2f}")
    print(f"  VE                     {ve_nmf:>5.4f} {ve_ica:>5.4f}")

    # --- save ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "W_nmf.npy", W_nmf)
    np.save(OUT_DIR / "W_ica.npy", W_ica)
    manifest = {
        "k": k_star,
        "alpha": alpha_star,
        "channels": channels,
        "corpus_size": int(N),
        "corpus_sources": [str(s.relative_to(ROOT)) for s in CORPUS_SOURCES if s.exists()],
        "k_sweep": sweep,
        "alpha_sweep": {str(a): v for a, v in alpha_sweep.items()},
        "nmf_ve": ve_nmf,
        "ica_ve": ve_ica,
        "nmf_atoms": nmf_atoms,
        "ica_atoms": ica_atoms,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (ANALYSIS_DIR / "blendshape_decomposition.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n[save] W_nmf.npy, W_ica.npy, manifest.json → {OUT_DIR}")


if __name__ == "__main__":
    main()
