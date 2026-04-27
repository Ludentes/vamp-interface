"""Per-(step, block) sensitivity map for each NMF component.

For each component c and each corpus, fit ridge at every (step, block) site:
  score_i = W_all[i, c]
  y = Ridge(δ[i, s, b, :], score_i)  → CV R² per site

Produces:
  1. A (16, 57) R² heatmap per component, stacked into one PNG.
  2. A ranked list of top-24 sites per component (best corpus).
  3. A new inject npz with K=24 sites per atom — the upgraded library for
     a fairer smoke test.

Usage:
  uv run python -m src.demographic_pc.analyze_site_sensitivity
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import NMF
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "models/blendshape_nmf/attn_cache"
METRICS = ROOT / "output/demographic_pc/fluxspace_metrics"
LIB_NPZ = ROOT / "models/blendshape_nmf/au_library.npz"
OUT_NPZ_K24 = ROOT / "models/blendshape_nmf/au_inject_k24.npz"
OUT_LABELS = ROOT / "models/blendshape_nmf/au_inject_k24_labels.json"
OUT_FIG = ROOT / "docs/research/images/2026-04-23-site-sensitivity.png"
OUT_MD = ROOT / "docs/research/2026-04-23-site-sensitivity.md"

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
K_SITES = 24


def _cv_r2(X, y, alpha=1.0, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=2026)
    preds = np.zeros_like(y)
    for tr, te in kf.split(X):
        m = Ridge(alpha=alpha).fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    ss_res = ((y - preds) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return 1.0 - (ss_res / max(float(ss_tot), 1e-12))


def _fit_ridge(X, y, alpha=1.0):
    return Ridge(alpha=alpha).fit(X, y).coef_.astype(np.float32)


def load_all_Y():
    """Stack blendshape matrices across corpora in cache rels order."""
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
    return names, per_tag_Y, per_tag_bounds, Y_all


def main():
    names, per_tag_Y, per_tag_bounds, Y_all = load_all_Y()
    tags = list(TAG_TO_BS.keys())

    # Reload NMF from saved library to keep same basis
    lib = np.load(LIB_NPZ, allow_pickle=True)
    H_full = lib["H"]            # (K, 52)
    per_comp_r2 = lib["per_comp_r2"]  # for best-corpus selection

    # Re-derive NMF W scores since au_library.npz didn't save W_all per sample
    keep_mask = np.array([n != "_neutral" for n in names])
    Y_fit = Y_all[:, keep_mask]
    nmf = NMF(n_components=K_COMPONENTS, init="nndsvd",  # type: ignore[arg-type]
              max_iter=2000, random_state=2026, tol=1e-5)
    W_all = nmf.fit_transform(Y_fit)   # (N_all, K)

    # Canonical step/block keys
    meta = json.load(open(CACHE / "smile_inphase" / "meta.json"))
    step_keys = np.asarray(meta["step_keys"], dtype=np.int64)
    block_keys = np.asarray(meta["block_keys"])
    S, B = len(step_keys), len(block_keys)

    # Per-component best-corpus selection (highest peak-block R² from library)
    best_tags = []
    for c in range(K_COMPONENTS):
        row = per_comp_r2[c]
        valid = ~np.isnan(row)
        scores = np.where(valid, row, -np.inf)
        best_tags.append(tags[int(np.argmax(scores))])

    # Per-component R² map, computed on the best corpus only (keeps cost bounded).
    r2_maps = np.zeros((K_COMPONENTS, S, B), dtype=np.float32)
    top_sites = np.zeros((K_COMPONENTS, K_SITES), dtype=np.int64)
    top_dirs = np.zeros((K_COMPONENTS, K_SITES, 3072), dtype=np.float32)
    top_r2 = np.zeros((K_COMPONENTS, K_SITES), dtype=np.float32)

    for c in range(K_COMPONENTS):
        best_tag = best_tags[c]
        lo, hi = per_tag_bounds[best_tag]
        y = W_all[lo:hi, c].astype(np.float32)
        if y.std() < 1e-4:
            continue

        arr = np.load(CACHE / best_tag / "delta_mix.npy", mmap_mode="r")
        print(f"\n[C{c}] best_corpus={best_tag}  N={hi-lo}  σ(y)={y.std():.3f}")

        # Fit ridge at each (step, block). This is 16 × 57 = 912 fits per component.
        # Each fit is (N, 3072) → scalar y; with 5-fold CV, quick.
        for si in range(S):
            for bi in range(B):
                X = arr[:, si, bi, :].astype(np.float32)
                X = X - X.mean(axis=0, keepdims=True)
                r2 = _cv_r2(X, y)
                r2_maps[c, si, bi] = r2

        flat = r2_maps[c].reshape(-1)
        topk_flat = np.argsort(-flat)[:K_SITES]
        top_sites[c] = topk_flat
        top_r2[c] = flat[topk_flat]
        # Refit per-site ridge (no CV) to get the direction vector we'll inject
        for k_i, flat_idx in enumerate(topk_flat):
            si, bi = int(flat_idx // B), int(flat_idx % B)
            X = arr[:, si, bi, :].astype(np.float32)
            X = X - X.mean(axis=0, keepdims=True)
            w = _fit_ridge(X, y)
            n = float(np.linalg.norm(w))
            top_dirs[c, k_i] = w / max(n, 1e-9)

        # Summary of top-24 sites
        site_strs = []
        for k_i in range(min(8, K_SITES)):
            fl = top_sites[c, k_i]
            si, bi = int(fl // B), int(fl % B)
            site_strs.append(f"(s={step_keys[si]}, {block_keys[bi]}) R²={top_r2[c, k_i]:+.2f}")
        print(f"  top-8 sites: {'  '.join(site_strs)}")
        print(f"  top-{K_SITES} R² range: "
              f"[{top_r2[c].min():+.3f}, {top_r2[c].max():+.3f}]  "
              f"median {float(np.median(top_r2[c])):+.3f}")

    # Save K=24 inject npz
    out = {"step_keys": step_keys, "block_keys": block_keys}
    labels = {}
    for c in range(K_COMPONENTS):
        if np.all(top_dirs[c] == 0):
            continue
        out[f"atom_{c:02d}_direction"] = top_dirs[c].astype(np.float32)
        out[f"atom_{c:02d}_sites"] = top_sites[c].astype(np.int64)
        top_j = int(np.argmax(H_full[c]))
        labels[f"atom_{c:02d}"] = {
            "component": int(c),
            "top_blendshape": str(names[top_j]),
            "best_corpus": best_tags[c],
            "K_sites": K_SITES,
            "r2_range": [float(top_r2[c].min()), float(top_r2[c].max())],
            "r2_median": float(np.median(top_r2[c])),
        }
    OUT_NPZ_K24.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_NPZ_K24, **out)
    OUT_LABELS.write_text(json.dumps(labels, indent=2))
    print(f"\n[npz k24] → {OUT_NPZ_K24}")
    print(f"[labels]  → {OUT_LABELS}")

    # Heatmap figure
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    for c, ax in enumerate(axes.flat):
        im = ax.imshow(r2_maps[c], aspect="auto", cmap="viridis", origin="lower",
                       vmin=-0.2, vmax=1.0)
        sem = labels.get(f"atom_{c:02d}", {}).get("top_blendshape", "?")
        ax.set_title(f"C{c} ({sem}) on {best_tags[c]}", fontsize=10)
        ax.set_xlabel("block idx")
        ax.set_ylabel("step idx")
        fig.colorbar(im, ax=ax, fraction=0.04)
        # mark top-K sites with dots
        for fl in top_sites[c]:
            si, bi = int(fl // B), int(fl % B)
            ax.plot(bi, si, "r.", markersize=4)
    fig.suptitle("Per-(step, block) CV R² of NMF-component score regressed from δ",
                 fontsize=12)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=100)
    plt.close(fig)
    print(f"[fig] → {OUT_FIG}")

    # Write markdown
    lines = []
    lines.append("---")
    lines.append("status: live")
    lines.append("topic: metrics-and-direction-quality")
    lines.append("summary: Site-sensitivity analysis for cached-δ injection. Ridge R² "
                 "at each of 16×57 (step, block) sites per NMF component; top-24 per "
                 "component saved to au_inject_k24.npz for a wider smoke test.")
    lines.append("---\n")
    lines.append("# Site sensitivity — where does each AU component live? — 2026-04-23\n")
    lines.append(f"For each component c in `au_library.npz` and its best corpus, "
                 f"ridge-regress the component score from δ at every (step, block) "
                 f"site independently. R² heatmap below shows where the signal "
                 f"concentrates; top-{K_SITES} sites per component saved into a new "
                 f"k=24 inject npz for the follow-up smoke test.\n")
    lines.append(f"![R² heatmap](images/2026-04-23-site-sensitivity.png)\n")
    lines.append("## Per-component top sites (best corpus)\n")
    for c in range(K_COMPONENTS):
        lab = labels.get(f"atom_{c:02d}")
        if not lab:
            continue
        lines.append(f"### C{c} ({lab['top_blendshape']}) on {lab['best_corpus']}\n")
        lines.append(f"K=24 top sites: R² range {lab['r2_range'][0]:+.3f} "
                     f"→ {lab['r2_range'][1]:+.3f}, median {lab['r2_median']:+.3f}\n")
        lines.append("| rank | step | block | R² |")
        lines.append("|---|---|---|---|")
        for k_i in range(min(12, K_SITES)):
            fl = top_sites[c, k_i]
            si, bi = int(fl // B), int(fl % B)
            lines.append(f"| {k_i+1} | {step_keys[si]} | {block_keys[bi]} | "
                         f"{top_r2[c, k_i]:+.3f} |")
        lines.append("")
    lines.append("## Interpretation\n")
    lines.append("- If top-K are **tightly clustered** (same block across many steps, or "
                 "adjacent blocks at one step) → the signal is localised; multi-site "
                 "injection is packing shifts close together and may just increase "
                 "magnitude. If they are **spread across steps + blocks** → multi-site "
                 "is genuinely covering different parts of the forward pass.\n")
    lines.append("- **R² plateau vs fall-off**: if the top-24 median R² is close to "
                 "peak R², the injection library has many equally-viable options; if "
                 "it collapses fast (rank 1 = 0.9, rank 24 = 0.3), K=24 is adding "
                 "noise beyond the first few sites.\n")
    lines.append("- **Next step**: rerun `au_library_smoke.py` pointing at "
                 "`au_inject_k24.npz` (update DIR_PATH). If pixels still don't move, "
                 "the library is a description-only artefact; full-tensor cache is "
                 "the next test.\n")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines))
    print(f"[md]  → {OUT_MD}")


if __name__ == "__main__":
    main()
