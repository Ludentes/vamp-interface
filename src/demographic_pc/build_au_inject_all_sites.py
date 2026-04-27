"""Build an 'inject at every site' npz.

For each NMF component and its best corpus, fit a ridge direction at EVERY
(step, block) site (16 × 57 = 912). Each site gets its own unit-normalised
3072-d weight vector. No site selection.

This is the maximal-signal smoke test: if injecting everywhere doesn't move
pixels, per-channel injection is a dead end regardless of where we poke.

Outputs:
  models/blendshape_nmf/au_inject_all.npz
  models/blendshape_nmf/au_inject_all_labels.json

Usage:
  uv run python -m src.demographic_pc.build_au_inject_all_sites
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import NMF
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "models/blendshape_nmf/attn_cache"
METRICS = ROOT / "output/demographic_pc/fluxspace_metrics"
LIB_NPZ = ROOT / "models/blendshape_nmf/au_library.npz"
OUT_NPZ = ROOT / "models/blendshape_nmf/au_inject_all.npz"
OUT_LABELS = ROOT / "models/blendshape_nmf/au_inject_all_labels.json"

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


def load_all_Y():
    sample_bs = json.loads(next(iter(TAG_TO_BS.values())).read_text())
    names = list(next(iter(sample_bs.values())).keys())
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
        per_tag_bounds[tag] = (offset, offset + len(rels))
        Y_parts.append(Y)
        offset += len(rels)
    return names, per_tag_bounds, np.concatenate(Y_parts, axis=0)


def main():
    names, per_tag_bounds, Y_all = load_all_Y()
    tags = list(TAG_TO_BS.keys())
    lib = np.load(LIB_NPZ, allow_pickle=True)
    H_full = lib["H"]
    per_comp_r2 = lib["per_comp_r2"]

    keep_mask = np.array([n != "_neutral" for n in names])
    Y_fit = Y_all[:, keep_mask]
    nmf = NMF(n_components=K_COMPONENTS, init="nndsvd",  # type: ignore[arg-type]
              max_iter=2000, random_state=2026, tol=1e-5)
    W_all = nmf.fit_transform(Y_fit)

    meta = json.load(open(CACHE / "smile_inphase" / "meta.json"))
    step_keys = np.asarray(meta["step_keys"], dtype=np.int64)
    block_keys = np.asarray(meta["block_keys"])
    S, B = len(step_keys), len(block_keys)
    K_SITES = S * B

    out = {"step_keys": step_keys, "block_keys": block_keys}
    labels = {}
    for c in range(K_COMPONENTS):
        row = per_comp_r2[c]
        valid = ~np.isnan(row)
        if not valid.any():
            continue
        best_t = int(np.argmax(np.where(valid, row, -np.inf)))
        best_tag = tags[best_t]
        lo, hi = per_tag_bounds[best_tag]
        y = W_all[lo:hi, c].astype(np.float32)
        if y.std() < 1e-4:
            continue

        arr = np.load(CACHE / best_tag / "delta_mix.npy", mmap_mode="r")
        print(f"\n[C{c}] best={best_tag} fitting {K_SITES} sites")
        dirs = np.zeros((K_SITES, 3072), dtype=np.float32)
        sites = np.zeros((K_SITES,), dtype=np.int64)
        idx = 0
        for si in range(S):
            for bi in range(B):
                X = arr[:, si, bi, :].astype(np.float32)
                X = X - X.mean(axis=0, keepdims=True)
                w = Ridge(alpha=1.0).fit(X, y).coef_.astype(np.float32)
                n = float(np.linalg.norm(w))
                dirs[idx] = w / max(n, 1e-9)
                sites[idx] = si * B + bi
                idx += 1
            print(f"  step_idx={si} done")
        out[f"atom_{c:02d}_direction"] = dirs
        out[f"atom_{c:02d}_sites"] = sites

        top_j = int(np.argmax(H_full[c]))
        labels[f"atom_{c:02d}"] = {
            "component": int(c),
            "top_blendshape": str(names[top_j]),
            "best_corpus": best_tag,
            "K_sites": int(K_SITES),
        }

    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_NPZ, **out)
    OUT_LABELS.write_text(json.dumps(labels, indent=2))
    print(f"\n[npz all] → {OUT_NPZ}")
    print(f"[labels]  → {OUT_LABELS}")


if __name__ == "__main__":
    main()
