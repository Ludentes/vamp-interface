"""Offline fidelity analysis for cached-δ injection.

Question: if we build a small per-axis library (mean vector + top-k PCs) at
peak (step, block), how well does it reconstruct individual per-sample δs?

For each axis:
  1. Locate peak (step, block) from Fro energy (matches recon).
  2. Extract δ matrix X = (N, 3072) at that location.
  3. Fit dictionary: v_mean = mean(X); basis = top-k PCs of centered X.
  4. Per sample: project δ_i onto {v_mean} (rank-1 mean), then {v_mean, pc1..k},
     report captured ||δ_i|| fraction.
  5. Base-holdout: build dictionary from 5 bases, evaluate on held-out base.

Writes markdown to docs/research/2026-04-23-injection-fidelity.md.

Usage:
  uv run python -m src.demographic_pc.analyze_injection_fidelity
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "models/blendshape_nmf/attn_cache"
OUT_MD = ROOT / "docs/research/2026-04-23-injection-fidelity.md"

TAGS = [
    "smile_inphase", "jaw_inphase", "alpha_interp_attn",
    "anger_rebalance", "surprise_rebalance", "disgust_rebalance",
    "pucker_rebalance", "lip_press_rebalance",
]


def _load(tag: str):
    meta = json.load(open(CACHE / tag / "meta.json"))
    arr = np.load(CACHE / tag / "delta_mix.npy", mmap_mode="r")
    return meta, arr


def _peak(arr: np.ndarray) -> tuple[int, int]:
    """Locate peak (step_idx, block_idx) by mean Fro energy."""
    N, S, B, _ = arr.shape
    acc = np.zeros((S, B), dtype=np.float64)
    chunk = 32
    for i in range(0, N, chunk):
        block = arr[i:i+chunk].astype(np.float32)
        fro = np.sqrt((block ** 2).sum(axis=-1))
        acc += fro.sum(axis=0)
    acc /= N
    idx = np.unravel_index(np.argmax(acc), acc.shape)
    return int(idx[0]), int(idx[1])


def _fit_library(X: np.ndarray, k: int):
    """X: (N, D) → (v_mean, basis (k, D)) where basis spans centered X."""
    v_mean = X.mean(axis=0)
    Xc = X - v_mean
    # economy SVD: (N, D) → U (N, r), s (r,), Vt (r, D); r = min(N, D)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    basis = Vt[:k]  # (k, D)
    return v_mean, basis


def _reconstruct(X: np.ndarray, v_mean: np.ndarray, basis: np.ndarray | None):
    """Best-LS reconstruction of each row of X using {v_mean} ∪ basis."""
    # Build dictionary D = [v_mean; basis] (m, D)
    if basis is None or basis.shape[0] == 0:
        D = v_mean[None, :]
    else:
        D = np.vstack([v_mean[None, :], basis])
    # Solve X ≈ C @ D, i.e. C = X @ D.T @ (D D.T)^-1
    G = D @ D.T                                     # (m, m)
    C = X @ D.T @ np.linalg.pinv(G)                 # (N, m)
    X_hat = C @ D                                   # (N, D)
    resid = X - X_hat
    norm_x = np.linalg.norm(X, axis=1)
    norm_r = np.linalg.norm(resid, axis=1)
    # captured fraction of ||δ||: 1 - ||resid|| / ||δ||
    captured = 1.0 - (norm_r / np.maximum(norm_x, 1e-9))
    return captured, X_hat


def _probe_per_axis(tag: str, ks=(0, 1, 3, 10)):
    meta, arr = _load(tag)
    s_i, b_i = _peak(arr)
    X = arr[:, s_i, b_i, :].astype(np.float32)
    N, D = X.shape
    rows = []
    v_mean, basis_full = _fit_library(X, k=max(ks))
    for k in ks:
        basis = basis_full[:k] if k > 0 else None
        captured, _ = _reconstruct(X, v_mean, basis)
        rows.append((k, float(np.median(captured)),
                     float(np.percentile(captured, 25)),
                     float(np.percentile(captured, 75))))
    return {
        "tag": tag, "N": int(N), "D": int(D),
        "step_idx": s_i, "block_idx": b_i,
        "step_key": meta["step_keys"][s_i],
        "block_key": meta["block_keys"][b_i],
        "rows": rows,
    }


def _probe_base_holdout(tag: str, k: int = 3):
    meta, arr = _load(tag)
    s_i, b_i = _peak(arr)
    X = arr[:, s_i, b_i, :].astype(np.float32)
    bases = sorted({r.split("/")[0] for r in meta["rels"]})
    rows = []
    for holdout in bases:
        train_mask = np.array([r.split("/")[0] != holdout for r in meta["rels"]])
        test_mask = ~train_mask
        v_mean, basis = _fit_library(X[train_mask], k=k)
        captured_in, _ = _reconstruct(X[train_mask], v_mean, basis)
        captured_out, _ = _reconstruct(X[test_mask], v_mean, basis)
        rows.append((holdout,
                     float(np.median(captured_in)),
                     float(np.median(captured_out))))
    return {"tag": tag, "k": k, "rows": rows}


def _probe_cross_axis_leakage(tags: list[str]):
    """If we inject v_mean from axis A, how much does it project onto v_mean of B?"""
    libs = {}
    for tag in tags:
        _, arr = _load(tag)
        s_i, b_i = _peak(arr)
        X = arr[:, s_i, b_i, :].astype(np.float32)
        v = X.mean(axis=0)
        n = np.linalg.norm(v)
        libs[tag] = v / n if n > 1e-9 else v
    M = np.stack([libs[t] for t in tags])
    cos = M @ M.T
    return tags, cos


def write_md(per_axis, base_holdout, leakage):
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("---")
    lines.append("status: live")
    lines.append("topic: metrics-and-direction-quality")
    lines.append("summary: Offline fidelity analysis for cached-δ injection. "
                 "Tests whether a small per-axis library (mean + top-k PCs) can "
                 "reconstruct individual sample δs at single_34, whether libraries "
                 "generalise across bases, and how much injecting one axis leaks "
                 "into another.")
    lines.append("---\n")
    lines.append("# Cached-δ injection fidelity — 2026-04-23\n")
    lines.append("Decides whether it is worth building a `FluxSpaceInjectCached` "
                 "ComfyUI node that skips the `2N+1` forward-pass overhead of "
                 "`FluxSpaceEditPair` by consuming a precomputed per-axis vector.\n")
    lines.append("Working at each axis's peak `(step, block)` from the recon. "
                 "Cache only stores channel-reduced summaries (`mean_d` shape 3072, "
                 "one value per channel after averaging over tokens), so this "
                 "analysis says nothing about whether a per-channel shift reproduces "
                 "the live-tensor edit — it only says whether the per-channel δ itself "
                 "is low-rank and portable.\n")

    # Per-axis reconstruction
    lines.append("## Per-axis reconstruction captured fraction\n")
    lines.append("Per-sample `1 − ‖δ − δ̂‖ / ‖δ‖` using `{v_mean} ∪ top-k PCs`. "
                 "Reports median and [p25, p75] over N samples.\n")
    lines.append("| axis | N | peak (step, block) | k=0 (mean-only) | k=1 | k=3 | k=10 |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in per_axis:
        cells = []
        for (_, med, lo, hi) in r["rows"]:
            cells.append(f"{med:.3f} [{lo:.3f}, {hi:.3f}]")
        lines.append(f"| {r['tag']} | {r['N']} | ({r['step_key']}, {r['block_key']}) | "
                     f"{cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
    lines.append("")

    # Base holdout
    lines.append("## Base-holdout portability (k=3)\n")
    lines.append("Build library from 5 bases, evaluate on held-out base. "
                 "`in_median` vs `out_median` = train vs test captured fraction.\n")
    for holdout_data in base_holdout:
        lines.append(f"### {holdout_data['tag']}\n")
        lines.append("| heldout base | in_median | out_median | gap |")
        lines.append("|---|---|---|---|")
        for (b, in_m, out_m) in holdout_data["rows"]:
            lines.append(f"| {b} | {in_m:.3f} | {out_m:.3f} | {in_m - out_m:+.3f} |")
        lines.append("")

    # Leakage
    tags, cos = leakage
    lines.append("## Cross-axis leakage — cosine of axis-mean vectors\n")
    lines.append("If we inject unit vector `v_A`, its projection onto `v_B` is `cos(A, B)`. "
                 "This is the side-effect an injection node would produce on the *other* "
                 "axis's peak channel.\n")
    lines.append("| | " + " | ".join(tags) + " |")
    lines.append("|---|" + "|".join(["---"] * len(tags)) + "|")
    for i, t in enumerate(tags):
        row = " | ".join(f"{cos[i, j]:+.3f}" for j in range(len(tags)))
        lines.append(f"| {t} | {row} |")
    lines.append("")

    # Interpretation
    lines.append("## Interpretation\n")
    lines.append("- **If k=0 (mean-only) already captures ≥0.7 median** → a single "
                 "per-axis vector is enough; rank-1 library is viable. This is the "
                 "strongest possible outcome for the injection-node path.\n")
    lines.append("- **If k=0 is poor but k=1..3 catches up** → we need a small basis, "
                 "not a vector. Still tractable; library entries become `(mean, basis)` pairs.\n")
    lines.append("- **If k=10 still leaves a large residual** → samples carry "
                 "per-sample structure the axis doesn't share. Injection would "
                 "reproduce only the axis-common part, which may or may not be the "
                 "part that moves pixels.\n")
    lines.append("- **Base-holdout gap <0.1** → one library works across bases → "
                 "a flat `{axis → vector}` store is enough. **Gap >0.2** → needs "
                 "per-base entries, matching the dictionary's `(axis, base)` keying.\n")
    lines.append("- **Leakage table**: the high non-smile/non-smile cosines from the "
                 "recon (0.97–0.99) say an injection of one emotion shifts the channel "
                 "mean of the others by almost as much — expected, since the recon "
                 "already showed they share the same direction.\n")
    lines.append("## Caveat\n")
    lines.append("All of this is on channel-reduced summaries. The real question — "
                 "does injecting `scale · v` (broadcast across tokens) at `single_34` "
                 "during a forward pass reproduce the prompt-pair edit's pixel output — "
                 "is not answerable from cache alone. Positive signals here license "
                 "building the ComfyUI node; negative signals kill it. Either outcome "
                 "is a real result.\n")

    OUT_MD.write_text("\n".join(lines))
    print(f"[md] → {OUT_MD}")


def main():
    per_axis = []
    for tag in TAGS:
        print(f"[axis] {tag}")
        per_axis.append(_probe_per_axis(tag))
        r = per_axis[-1]
        for (k_i, med, lo, hi) in r["rows"]:
            print(f"  k={k_i:2d}  captured median={med:.3f}  [p25={lo:.3f}, p75={hi:.3f}]")

    base_holdout = []
    for tag in ["smile_inphase", "anger_rebalance"]:
        print(f"\n[holdout] {tag} (k=3)")
        r = _probe_base_holdout(tag, k=3)
        base_holdout.append(r)
        for (b, in_m, out_m) in r["rows"]:
            print(f"  heldout={b:20s}  in={in_m:.3f}  out={out_m:.3f}  gap={in_m - out_m:+.3f}")

    print("\n[leakage]")
    leakage = _probe_cross_axis_leakage(TAGS)
    tags, cos = leakage
    for i, t in enumerate(tags):
        row = " ".join(f"{cos[i,j]:+.3f}" for j in range(len(tags)))
        print(f"  {t:28s} {row}")

    write_md(per_axis, base_holdout, leakage)


if __name__ == "__main__":
    main()
