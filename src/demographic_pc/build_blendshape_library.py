"""NMF on the blendshape matrix + persist a library for cached-δ injection.

Blendshape scores are non-negative in [0, 1], so NMF is the natural basis:
  Y ≈ W @ H,   W ≥ 0 (sample loadings),  H ≥ 0 (52-d AU patterns).

Each row of H is one "AU pattern" — a non-negative combination of blendshapes
that co-activate. More interpretable than PCA (orthogonal, signed mixtures)
because it's parts-based: an AU pattern like "smile" should load positively
on mouthSmile + mouthUpperUp and zero elsewhere.

Pipeline:
  1. Load Y_all = (N_total, 52) stacked across 8 corpora.
  2. Fit NMF(k=8) on Y_all.
  3. For each component:
       - per-corpus CV R² of δ at peak block → W column c.
       - per-corpus ridge weight vector w_c (3072,) for the library.
  4. Save library to models/blendshape_nmf/au_library.npz:
       H (k, 52)              — AU patterns (blendshape loadings)
       W_refits (K, T, 3072)  — per-corpus ridge weight vectors
       peak_step_block (T, 2) — peak location per corpus tag
       tags, names            — labels
       per_comp_r2 (K, T)     — CV R² per (component, corpus)

Also writes a markdown summary to docs/research/2026-04-23-au-library.md with
an interpretive pass on each component.

Usage:
  uv run python -m src.demographic_pc.build_blendshape_library
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import NMF
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "models/blendshape_nmf/attn_cache"
METRICS = ROOT / "output/demographic_pc/fluxspace_metrics"
OUT_LIB = ROOT / "models/blendshape_nmf/au_library.npz"
OUT_MD = ROOT / "docs/research/2026-04-23-au-library.md"

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

K_COMPONENTS = 8


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
    sample_bs = json.loads(next(iter(TAG_TO_BS.values())).read_text())
    names = list(next(iter(sample_bs.values())).keys())
    per_tag_Y = {}
    per_tag_bounds = {}
    Y_parts = []
    offset = 0
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
        per_tag_bounds[tag] = (offset, offset + len(rels))
        Y_parts.append(Y)
        offset += len(rels)
    Y_all = np.concatenate(Y_parts, axis=0)
    print(f"[Y] stacked: {Y_all.shape}")
    return names, per_tag_Y, per_tag_bounds, Y_all


def top_loadings(comp, names, n=6):
    idx = np.argsort(-np.abs(comp))[:n]
    return [(names[i], float(comp[i])) for i in idx]


def interpret_component(loadings) -> str:
    """Heuristic semantic tag based on top blendshape names."""
    top_names = [n for n, _ in loadings[:6]]
    joined = " ".join(top_names).lower()
    if "mouthsmile" in joined:
        return "smile"
    if "mouthpucker" in joined:
        return "pucker"
    if "browouterup" in joined and "browinnerup" in joined:
        return "brow_lift"
    if "browdown" in joined:
        return "brow_furrow"
    if "eyelookdown" in joined or "eyelookup" in joined:
        return "gaze_vertical"
    if "eyelookin" in joined or "eyelookout" in joined:
        return "gaze_lateral"
    if "eyeblink" in joined or "eyesquint" in joined:
        return "eye_activity"
    if "jawopen" in joined:
        return "jaw_open"
    if "mouthlowerdown" in joined or "mouthupperup" in joined:
        return "mouth_stretch"
    return "mixed"


def main():
    names, _, per_tag_bounds, Y_all = load_all_Y()
    tags = list(TAG_TO_BS.keys())

    # Drop _neutral (always 0) to avoid degenerate col
    keep_mask = np.array([n != "_neutral" for n in names])
    Y_fit = Y_all[:, keep_mask]

    print(f"[NMF] fitting k={K_COMPONENTS} on {Y_fit.shape}")
    nmf = NMF(n_components=K_COMPONENTS, init="nndsvd",  # type: ignore[arg-type]
              max_iter=2000, random_state=2026, tol=1e-5)
    W_all = nmf.fit_transform(Y_fit)       # (N_all, K)
    H_kept = nmf.components_                # (K, 51)

    # Expand H back to full 52 (zero column for _neutral)
    H_full = np.zeros((K_COMPONENTS, len(names)), dtype=np.float32)
    H_full[:, keep_mask] = H_kept

    # Reconstruction fidelity
    Y_hat = W_all @ H_kept
    resid = Y_fit - Y_hat
    recon_r2 = 1.0 - (resid ** 2).sum() / ((Y_fit - Y_fit.mean(axis=0)) ** 2).sum()
    print(f"[NMF] reconstruction R² on Y: {recon_r2:.3f}")

    # Per-component per-corpus analysis
    per_comp_r2 = np.zeros((K_COMPONENTS, len(tags)), dtype=np.float32)
    W_refits = np.zeros((K_COMPONENTS, len(tags), 3072), dtype=np.float32)
    peak_coords = np.zeros((len(tags), 2), dtype=np.int32)
    peak_step_block = []

    # Precompute δ per tag at peak (cache in-memory, eat the 3072 × N_t memory)
    tag_X = {}
    for ti, tag in enumerate(tags):
        arr = np.load(CACHE / tag / "delta_mix.npy", mmap_mode="r")
        s_i, b_i = _peak(arr)
        peak_coords[ti] = [s_i, b_i]
        meta = json.load(open(CACHE / tag / "meta.json"))
        peak_step_block.append((meta["step_keys"][s_i], meta["block_keys"][b_i]))
        X = arr[:, s_i, b_i, :].astype(np.float32)
        X = X - X.mean(axis=0, keepdims=True)
        tag_X[tag] = X

    for c in range(K_COMPONENTS):
        print(f"\n[C{c}] {top_loadings(H_full[c], names, n=5)}")
        sem = interpret_component(top_loadings(H_full[c], names, n=6))
        print(f"       semantic: {sem}")
        for ti, tag in enumerate(tags):
            lo, hi = per_tag_bounds[tag]
            scores = W_all[lo:hi, c]
            if scores.std() < 1e-4:
                per_comp_r2[c, ti] = float("nan")
                continue
            X = tag_X[tag]
            r2 = _cv_r2(X, scores)
            per_comp_r2[c, ti] = r2
            # refit on full data for the library weight vector
            w = Ridge(alpha=1.0).fit(X, scores).coef_.astype(np.float32)
            W_refits[c, ti] = w
            print(f"  {tag:24s}  R²={r2:+.3f}  σ(score)={scores.std():.3f}")

    # Save library
    OUT_LIB.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_LIB,
        H=H_full,
        W_refits=W_refits,
        peak_coords=peak_coords,
        per_comp_r2=per_comp_r2,
        tags=np.array(tags),
        names=np.array(names),
        reconstruction_r2=np.array(recon_r2),
    )
    print(f"\n[lib] → {OUT_LIB}  (H:{H_full.shape}, W_refits:{W_refits.shape})")

    # Markdown
    lines = []
    lines.append("---")
    lines.append("status: live")
    lines.append("topic: metrics-and-direction-quality")
    lines.append("summary: NMF decomposition of the 52-d ARKit blendshape matrix "
                 f"into K={K_COMPONENTS} parts-based AU patterns, with per-corpus "
                 "ridge fits of δ→component score. Library saved to "
                 "models/blendshape_nmf/au_library.npz for future cached-δ injection.")
    lines.append("---\n")
    lines.append("# AU library from blendshape NMF — 2026-04-23\n")
    lines.append(f"K={K_COMPONENTS}, reconstruction R² on Y = **{recon_r2:.3f}**. "
                 "NMF is parts-based: each component is a non-negative co-activation "
                 "pattern of ARKit blendshapes, not an orthogonal signed direction.\n")

    lines.append("## Components\n")
    lines.append("| C | semantic | top blendshape loadings | mean R² | R² range |")
    lines.append("|---|---|---|---|---|")
    for c in range(K_COMPONENTS):
        tops = top_loadings(H_full[c], names, n=4)
        sem = interpret_component(top_loadings(H_full[c], names, n=6))
        loadings_str = ", ".join(f"{n}({v:.2f})" for n, v in tops)
        valid = per_comp_r2[c][~np.isnan(per_comp_r2[c])]
        mean_r2 = float(valid.mean()) if len(valid) else float("nan")
        rng = f"[{valid.min():+.3f}, {valid.max():+.3f}]" if len(valid) else "—"
        lines.append(f"| C{c} | {sem} | {loadings_str} | {mean_r2:+.3f} | {rng} |")
    lines.append("")

    lines.append("## Per-corpus R² matrix\n")
    lines.append("| component | " + " | ".join(tags) + " |")
    lines.append("|---|" + "|".join(["---"] * len(tags)) + "|")
    for c in range(K_COMPONENTS):
        sem = interpret_component(top_loadings(H_full[c], names, n=6))
        row = " | ".join(
            f"{per_comp_r2[c, ti]:+.3f}" if not np.isnan(per_comp_r2[c, ti]) else "—"
            for ti in range(len(tags))
        )
        lines.append(f"| C{c} ({sem}) | {row} |")
    lines.append("")

    lines.append("## Peak (step, block) per corpus\n")
    lines.append("| corpus | step | block |")
    lines.append("|---|---|---|")
    for ti, tag in enumerate(tags):
        s, b = peak_step_block[ti]
        lines.append(f"| {tag} | {s} | {b} |")
    lines.append("")

    lines.append("## Library file layout\n")
    lines.append(f"`{OUT_LIB.relative_to(ROOT)}` contains:\n")
    lines.append("- `H` — (K={}, 52) AU-pattern matrix, blendshape-space loadings per "
                 "component. Non-negative.".format(K_COMPONENTS))
    lines.append("- `W_refits` — (K, T, 3072) per-(component, corpus) ridge weight "
                 "vectors. These are the cached-δ injection targets: to push component C "
                 "by amount α at corpus T's peak block, add `α · W_refits[C, T]` to δ.\n")
    lines.append("- `peak_coords` — (T, 2) [step_idx, block_idx] in cache order.\n")
    lines.append("- `per_comp_r2` — (K, T) CV R² per (component, corpus), the "
                 "reliability of each library vector.\n")
    lines.append("- `tags`, `names`, `reconstruction_r2` — string labels and the "
                 "blendshape-space fit quality.\n")

    lines.append("## How to read this\n")
    lines.append("- **NMF recon R²** tells us how lossy the basis is. ≥0.9 means the "
                 "52-d blendshape space really is 8-d in disguise. <0.7 means we cut "
                 "too much and need more components.\n")
    lines.append("- **Per-component mean R²**: how reliably each AU pattern is "
                 "encoded in attention at the peak block. Components with mean R²>0.7 "
                 "are library-grade — inject them and expect the pattern to move.\n")
    lines.append("- **R² spread across corpora**: a component that works only on one "
                 "corpus reflects a corpus-specific artefact. One that works on all "
                 "corpora is a general attention direction worth caching.\n")
    lines.append("- **Semantic tag**: heuristic over top loadings. Override by "
                 "inspection if a pattern is mixed.\n")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines))
    print(f"[md]  → {OUT_MD}")


if __name__ == "__main__":
    main()
