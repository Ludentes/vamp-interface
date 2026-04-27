"""Regress per-sample ARKit blendshape scores against cached delta_mix channels.

For each (axis corpus, blendshape) pair, at the axis's peak (step, block):
  y_k  = per-sample blendshape_k score       (N,)
  X    = per-sample delta_mix vector         (N, 3072)

Fit ridge regression y_k ≈ X @ w_k. Report:
  - R² (5-fold CV) per blendshape
  - cosine(w_k, v_axis_mean) — does the blendshape direction align with
    the axis's mean delta vector?
  - cosine(w_k, w_k') for top-R² blendshapes — are different AUs living
    in different directions, or collapsing into one?

Emits a ranked table: which blendshapes are most predictable from the
cached delta channels, per axis corpus.

Usage:
  uv run python -m src.demographic_pc.analyze_blendshape_regression
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "models/blendshape_nmf/attn_cache"
METRICS = ROOT / "output/demographic_pc/fluxspace_metrics"
OUT_MD = ROOT / "docs/research/2026-04-23-blendshape-regression.md"

TAG_TO_BLENDSHAPES = {
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


def _ridge_cv_r2(X, y, alpha=1.0, k=5):
    """5-fold CV R² for ridge regression."""
    kf = KFold(n_splits=k, shuffle=True, random_state=2026)
    preds = np.zeros_like(y)
    for tr, te in kf.split(X):
        m = Ridge(alpha=alpha).fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    ss_res = ((y - preds) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return 1.0 - (ss_res / max(ss_tot, 1e-12))


def _ridge_fit(X, y, alpha=1.0):
    return Ridge(alpha=alpha).fit(X, y).coef_


def analyse_tag(tag, top_n=10):
    meta = json.load(open(CACHE / tag / "meta.json"))
    arr = np.load(CACHE / tag / "delta_mix.npy", mmap_mode="r")
    s_i, b_i = _peak(arr)
    X = arr[:, s_i, b_i, :].astype(np.float32)
    N, D = X.shape

    bs_path = TAG_TO_BLENDSHAPES[tag]
    bs_data = json.loads(bs_path.read_text())
    rels = meta["rels"]
    # blendshape names from first sample
    first = next(iter(bs_data.values()))
    names = list(first.keys())
    Y = np.zeros((N, len(names)), dtype=np.float32)
    missing = 0
    for i, rel in enumerate(rels):
        d = bs_data.get(rel)
        if d is None:
            missing += 1
            continue
        for j, name in enumerate(names):
            Y[i, j] = d.get(name, 0.0)

    # Center X and Y for fair comparison
    Xc = X - X.mean(axis=0, keepdims=True)
    v_mean = X.mean(axis=0)
    v_mean_n = v_mean / max(float(np.linalg.norm(v_mean)), 1e-9)

    results = []
    W = np.zeros((len(names), D), dtype=np.float32)
    for j, name in enumerate(names):
        y = Y[:, j]
        if y.std() < 1e-4:
            results.append((name, float("nan"), float(y.std()), float("nan")))
            continue
        r2 = _ridge_cv_r2(Xc, y)
        w = _ridge_fit(Xc, y)
        W[j] = w
        w_norm = float(np.linalg.norm(w))
        cos_axis = float(w @ v_mean_n / max(w_norm, 1e-9)) if w_norm > 1e-9 else 0.0
        results.append((name, float(r2), float(y.std()), cos_axis))

    # Rank by R²
    ranked = sorted(
        [r for r in results if not np.isnan(r[1])],
        key=lambda r: -r[1],
    )

    # Cosine between top-N blendshape weight vectors
    top_names = [r[0] for r in ranked[:top_n]]
    top_idx = [names.index(n) for n in top_names]
    W_top = W[top_idx]
    norms = np.linalg.norm(W_top, axis=1, keepdims=True)
    Wn = W_top / np.maximum(norms, 1e-9)
    cos_top = Wn @ Wn.T

    return {
        "tag": tag, "N": int(N), "missing": missing,
        "step": meta["step_keys"][s_i], "block": meta["block_keys"][b_i],
        "ranked": ranked,
        "top_names": top_names,
        "cos_top": cos_top,
    }


def write_md(all_results):
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("---")
    lines.append("status: live")
    lines.append("topic: metrics-and-direction-quality")
    lines.append("summary: ARKit blendshape regression against cached delta_mix channels. "
                 "Finds which of 52 blendshapes (≈FACS AUs) each axis corpus encodes "
                 "linearly from attention, and how the weight vectors relate across AUs.")
    lines.append("---\n")
    lines.append("# Blendshape / AU regression from cached δ — 2026-04-23\n")
    lines.append("For each axis corpus, at its peak `(step, block)`: ridge-fit each of the "
                 "52 ARKit blendshape scores (MediaPipe FaceLandmarker output; maps to FACS "
                 "AUs) against the 3072-dim delta_mix channel vector. 5-fold CV R² reports "
                 "how much of each blendshape's variation is encoded in the attention "
                 "channel means at that block.\n")

    for r in all_results:
        lines.append(f"## {r['tag']} — peak ({r['step']}, {r['block']}), N={r['N']}\n")
        lines.append("Top-15 blendshapes by CV R² (blendshape / R² / σ(y) / cos(w, v_axis)):\n")
        lines.append("| blendshape | R² | σ(y) | cos(w, v_axis) |")
        lines.append("|---|---|---|---|")
        for name, r2, std, cos_axis in r["ranked"][:15]:
            lines.append(f"| {name} | {r2:+.3f} | {std:.3f} | {cos_axis:+.3f} |")
        lines.append("")
        lines.append(f"Cosine between top-{len(r['top_names'])} blendshape weight vectors:\n")
        lines.append("| | " + " | ".join(r["top_names"]) + " |")
        lines.append("|---|" + "|".join(["---"] * len(r["top_names"])) + "|")
        for i, n in enumerate(r["top_names"]):
            row = " | ".join(f"{r['cos_top'][i, j]:+.2f}" for j in range(len(r["top_names"])))
            lines.append(f"| {n} | {row} |")
        lines.append("")

    lines.append("## How to read this\n")
    lines.append("- **R² > 0.5** → the blendshape is strongly encoded as a linear direction in "
                 "delta_mix at that block. Candidate for an injection target.\n")
    lines.append("- **cos(w, v_axis) near ±1** → the blendshape's direction is the same as the "
                 "axis's mean δ direction. The axis *is* that blendshape.\n")
    lines.append("- **cos(w, v_axis) near 0** → the blendshape is encoded but the axis is "
                 "something else; injecting the axis-mean won't move this blendshape.\n")
    lines.append("- **Top-N cos matrix**: high off-diagonal = blendshapes are collinear in "
                 "attention (not separately steerable). Block-diagonal = distinct AUs live in "
                 "distinct attention directions, so a per-AU edit library is tractable.\n")
    OUT_MD.write_text("\n".join(lines))
    print(f"[md] → {OUT_MD}")


def main():
    all_results = []
    for tag in TAG_TO_BLENDSHAPES:
        if not (CACHE / tag / "meta.json").exists():
            print(f"[skip] {tag}")
            continue
        if not TAG_TO_BLENDSHAPES[tag].exists():
            print(f"[skip] {tag} no blendshapes.json")
            continue
        print(f"[axis] {tag}")
        r = analyse_tag(tag)
        all_results.append(r)
        print(f"  top-15 R² blendshapes:")
        for name, r2, std, cos_ax in r["ranked"][:15]:
            print(f"    {name:22s}  R²={r2:+.3f}  σ={std:.3f}  cos={cos_ax:+.3f}")
    write_md(all_results)


if __name__ == "__main__":
    main()
