"""Decompose the 52-d blendshape output space, then regress attention channels
against the discovered components.

Two separate decompositions of Y = (N_all, 52), stacked across all 8 corpora:
  PCA — orthogonal directions ranked by variance.
  ICA — non-Gaussian directions; each component tries to be a single AU
        pattern, not a mixture.

For each discovered component c (unit vector in 52-d blendshape space):
  score_i = Y_i @ c                                        (N,)
  ridge-fit score_i ≈ δ_i @ w_c + b                        (per corpus)
  report CV R² and top blendshape loadings on c.

Also compares PCA components' δ-weight vectors to the existing 11 NMF atoms
(`directions_k11.npz`) to see whether attention-space NMF and blendshape-space
PCA recover the same structure.

Usage:
  uv run python -m src.demographic_pc.analyze_blendshape_components
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "models/blendshape_nmf/attn_cache"
METRICS = ROOT / "output/demographic_pc/fluxspace_metrics"
ATOMS_NPZ = ROOT / "models/blendshape_nmf/directions_k11.npz"
OUT_MD = ROOT / "docs/research/2026-04-23-blendshape-components.md"

TAG_TO_BS = {
    "smile_inphase":       METRICS / "crossdemo/smile/smile_inphase/blendshapes.json",
    "jaw_inphase":         METRICS / "crossdemo/smile/jaw_inphase/blendshapes.json",
    "alpha_interp_attn":   METRICS / "crossdemo/smile/alpha_interp_attn/blendshapes.json",
    "anger_rebalance":     METRICS / "crossdemo/anger/rebalance/blendshapes.json",
    "surprise_rebalance":  METRICS / "crossdemo/surprise/rebalance/blendshapes.json",
    "disgust_rebalance":   METRICS / "crossdemo/disgust/rebalance/blendshapes.json",
    "pucker_rebalance":    METRICS / "crossdemo/pucker/rebalance/blendshapes.json",
    "lip_press_rebalance": METRICS / "crossdemo/lip_press/rebalance/blendshapes.json",
}


def _peak(arr):
    N, S, B, _ = arr.shape
    acc = np.zeros((S, B), dtype=np.float64)
    for i in range(0, N, 32):
        blk = arr[i:i+32].astype(np.float32)
        acc += np.sqrt((blk ** 2).sum(axis=-1)).sum(axis=0)
    acc /= N
    idx = np.unravel_index(np.argmax(acc), acc.shape)
    return int(idx[0]), int(idx[1])


def _cv_r2(X, y, alpha=1.0, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=2026)
    preds = np.zeros_like(y)
    for tr, te in kf.split(X):
        m = Ridge(alpha=alpha).fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    ss_res = ((y - preds) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return 1.0 - (ss_res / max(float(ss_tot), 1e-12))


def load_all_Y():
    """Stack (N_per_tag × 52) blendshape matrices across corpora, aligned with cache order."""
    # Canonical blendshape name order from the first available sample.
    sample_bs = json.loads(next(iter(TAG_TO_BS.values())).read_text())
    names = list(next(iter(sample_bs.values())).keys())

    per_tag_Y = {}
    per_tag_rels = {}
    for tag, bs_path in TAG_TO_BS.items():
        meta = json.load(open(CACHE / tag / "meta.json"))
        rels = meta["rels"]
        data = json.loads(bs_path.read_text())
        Y = np.zeros((len(rels), len(names)), dtype=np.float32)
        for i, r in enumerate(rels):
            d = data.get(r, {})
            for j, name in enumerate(names):
                Y[i, j] = d.get(name, 0.0)
        per_tag_Y[tag] = Y
        per_tag_rels[tag] = rels

    Y_all = np.concatenate(list(per_tag_Y.values()), axis=0)
    print(f"[Y] stacked shape: {Y_all.shape}")
    return names, per_tag_Y, Y_all


def decompose(Y_all, k_pca=10, k_ica=10, seed=2026):
    pca = PCA(n_components=k_pca, random_state=seed).fit(Y_all)
    ica = FastICA(n_components=k_ica, random_state=seed, max_iter=1000,
                  tol=1e-4, whiten="unit-variance").fit(Y_all)
    return pca, ica


def top_loadings(comp: np.ndarray, names: list[str], n: int = 6):
    """Return n largest-|coef| blendshape names with signed values."""
    idx = np.argsort(-np.abs(comp))[:n]
    return [(names[i], float(comp[i])) for i in idx]


def per_tag_r2_for_component(comp, per_tag_Y, tags, alpha=1.0):
    """For each tag, score_i = Y @ comp; regress score against δ at tag's peak block."""
    rows = []
    for tag in tags:
        Y_t = per_tag_Y[tag]
        scores = Y_t @ comp  # (N_t,)
        if scores.std() < 1e-4:
            rows.append((tag, float("nan"), float(scores.std())))
            continue
        arr = np.load(CACHE / tag / "delta_mix.npy", mmap_mode="r")
        s_i, b_i = _peak(arr)
        X = arr[:, s_i, b_i, :].astype(np.float32)
        X = X - X.mean(axis=0, keepdims=True)
        r2 = _cv_r2(X, scores, alpha=alpha)
        rows.append((tag, r2, float(scores.std())))
    return rows


def compare_to_atoms(pca_w_per_tag: dict):
    """For a representative tag (smile_inphase), compare each PCA component's
    ridge-fit w vector (at peak block) to the 11 NMF atom directions (24, 3072)
    at their first site."""
    if not ATOMS_NPZ.exists():
        return None
    data = np.load(ATOMS_NPZ, allow_pickle=True)
    atom_names = [f"atom_{i:02d}" for i in range(11)]
    atom_dirs_first = []
    for i in range(11):
        d = data[f"atom_{i:02d}_direction"]   # (24, 3072)
        s = d.mean(axis=0)                    # collapse sites to a single vector
        s = s / max(float(np.linalg.norm(s)), 1e-9)
        atom_dirs_first.append(s)
    atoms = np.stack(atom_dirs_first)         # (11, 3072)

    # pca_w_per_tag["smile_inphase"] = (k_pca, 3072)
    W = pca_w_per_tag["smile_inphase"]
    Wn = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-9)
    cos = Wn @ atoms.T                         # (k_pca, 11)
    return atom_names, cos


def fit_w_per_tag(comp, per_tag_Y, tags):
    """Fit Ridge(δ → score) per tag and return the 3072-d weight vectors."""
    W = {}
    for tag in tags:
        Y_t = per_tag_Y[tag]
        scores = Y_t @ comp
        if scores.std() < 1e-4:
            W[tag] = np.zeros(3072, dtype=np.float32)
            continue
        arr = np.load(CACHE / tag / "delta_mix.npy", mmap_mode="r")
        s_i, b_i = _peak(arr)
        X = arr[:, s_i, b_i, :].astype(np.float32)
        X = X - X.mean(axis=0, keepdims=True)
        w = Ridge(alpha=1.0).fit(X, scores).coef_.astype(np.float32)
        W[tag] = w
    return W


def write_md(names, pca, ica, pca_r2_rows, ica_r2_rows, atom_cmp):
    lines = []
    lines.append("---")
    lines.append("status: live")
    lines.append("topic: metrics-and-direction-quality")
    lines.append("summary: Decompose the 52-d ARKit blendshape space (PCA + ICA) "
                 "and regress attention channels against the discovered components. "
                 "Better than regressing each of 52 blendshapes separately because "
                 "the output space is highly collinear.")
    lines.append("---\n")
    lines.append("# Blendshape-space decomposition + δ regression — 2026-04-23\n")
    lines.append("The 52-d ARKit blendshape output is strongly collinear (previous "
                 "per-blendshape regression showed top-15 AUs across axes are mostly "
                 "brow/eye). This file first decomposes the blendshape matrix itself, "
                 "then regresses attention against the discovered components.\n")

    # PCA variance
    lines.append("## PCA of blendshape matrix (N_all × 52 stacked across 8 corpora)\n")
    ev = pca.explained_variance_ratio_
    lines.append("| PC | var ratio | cum var |")
    lines.append("|---|---|---|")
    for i, v in enumerate(ev):
        lines.append(f"| PC{i} | {v:.3f} | {ev[:i+1].sum():.3f} |")
    lines.append("")

    lines.append("## PCA components: top blendshape loadings + per-corpus δ-regression R²\n")
    for i in range(pca.components_.shape[0]):
        comp = pca.components_[i]
        tops = top_loadings(comp, names, n=6)
        rows = pca_r2_rows[i]
        loadings_str = ", ".join(f"{n}({v:+.2f})" for n, v in tops)
        lines.append(f"### PC{i} — variance {ev[i]:.3f}\n")
        lines.append(f"Top loadings: {loadings_str}\n")
        lines.append("| corpus | CV R² | σ(score) |")
        lines.append("|---|---|---|")
        for tag, r2, std in rows:
            lines.append(f"| {tag} | {r2:+.3f} | {std:.3f} |")
        lines.append("")

    lines.append("## ICA components: top blendshape loadings + δ R²\n")
    for i in range(ica.components_.shape[0]):
        comp = ica.components_[i]
        comp_n = comp / max(float(np.linalg.norm(comp)), 1e-9)
        tops = top_loadings(comp_n, names, n=6)
        rows = ica_r2_rows[i]
        loadings_str = ", ".join(f"{n}({v:+.2f})" for n, v in tops)
        lines.append(f"### IC{i}\n")
        lines.append(f"Top loadings: {loadings_str}\n")
        lines.append("| corpus | CV R² | σ(score) |")
        lines.append("|---|---|---|")
        for tag, r2, std in rows:
            lines.append(f"| {tag} | {r2:+.3f} | {std:.3f} |")
        lines.append("")

    if atom_cmp is not None:
        atom_names, cos = atom_cmp
        lines.append("## PCA blendshape components vs 11 existing NMF atoms on δ\n")
        lines.append("Cosine between each PCA component's δ-weight vector "
                     "(ridge-fit at smile_inphase peak block) and each NMF atom's "
                     "direction (averaged across its 24 sites, unit-normalised).\n")
        lines.append("| | " + " | ".join(atom_names) + " |")
        lines.append("|---|" + "|".join(["---"] * len(atom_names)) + "|")
        for i in range(cos.shape[0]):
            row = " | ".join(f"{cos[i,j]:+.2f}" for j in range(cos.shape[1]))
            lines.append(f"| PC{i} | {row} |")
        lines.append("")
        off = cos[np.abs(cos) > 0.3]
        lines.append(f"\n{len(off)} cells with |cos|>0.3; max |cos| = "
                     f"{float(np.abs(cos).max()):.3f}.\n")

    lines.append("## How to read this\n")
    lines.append("- **PCA/ICA var ratio tail**: how many independent blendshape "
                 "patterns actually exist. If PC0 eats >50%, the output is nearly rank-1; "
                 "most per-blendshape signal was noise around that one direction.\n")
    lines.append("- **Top loadings**: interpret each component. A single clean AU = "
                 "one blendshape dominant. A pattern ('smile') = multiple blendshapes "
                 "co-load with same sign. Mixed signs = open/close style opposition.\n")
    lines.append("- **R² per corpus**: tells us which corpora encode each pattern in "
                 "attention. If `smile_inphase` scores high on 'smile-pattern' PC but "
                 "anger corpora score low on it, attention is axis-specific.\n")
    lines.append("- **ICA vs PCA**: ICA components should be closer to single AUs "
                 "(non-Gaussian, parts-based). If ICA gives cleaner loadings and "
                 "similar R², prefer ICA as the library basis.\n")
    lines.append("- **PC ↔ NMF atom cos**: |cos|>0.5 means attention-space NMF and "
                 "output-space PCA found the same direction. |cos|<0.3 everywhere "
                 "means the two bases answer different questions.\n")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines))
    print(f"[md] → {OUT_MD}")


def main():
    names, per_tag_Y, Y_all = load_all_Y()
    tags = list(TAG_TO_BS.keys())
    pca, ica = decompose(Y_all, k_pca=8, k_ica=8)
    print(f"\n[PCA] variance ratios: {pca.explained_variance_ratio_}")
    print(f"[PCA] cumulative at k=8: {pca.explained_variance_ratio_.sum():.3f}")

    pca_r2_rows = []
    pca_w_per_tag: dict[str, np.ndarray] = {t: np.zeros((pca.components_.shape[0], 3072),
                                                         dtype=np.float32) for t in tags}
    for i in range(pca.components_.shape[0]):
        comp = pca.components_[i]
        print(f"\n[PC{i}] top loadings:", top_loadings(comp, names, n=5))
        rows = per_tag_r2_for_component(comp, per_tag_Y, tags)
        pca_r2_rows.append(rows)
        for tag, r2, std in rows:
            print(f"  {tag:24s}  R²={r2:+.3f}  σ={std:.3f}")
        W = fit_w_per_tag(comp, per_tag_Y, tags)
        for tag in tags:
            pca_w_per_tag[tag][i] = W[tag]

    ica_r2_rows = []
    for i in range(ica.components_.shape[0]):
        comp = ica.components_[i]
        print(f"\n[IC{i}] top loadings:",
              top_loadings(comp / max(float(np.linalg.norm(comp)), 1e-9), names, n=5))
        rows = per_tag_r2_for_component(comp, per_tag_Y, tags)
        ica_r2_rows.append(rows)
        for tag, r2, std in rows:
            print(f"  {tag:24s}  R²={r2:+.3f}  σ={std:.3f}")

    atom_cmp = compare_to_atoms(pca_w_per_tag)
    write_md(names, pca, ica, pca_r2_rows, ica_r2_rows, atom_cmp)


if __name__ == "__main__":
    main()
