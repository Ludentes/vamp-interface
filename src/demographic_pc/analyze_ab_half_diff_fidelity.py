"""Same analysis as analyze_injection_fidelity.py, but on `ab_half_diff`.

ab_half_diff is the A/B pair asymmetry at each (step, block) — what makes
prompt A differ from prompt B. If the axis-specific signal lives here rather
than in delta_mix (which is A+B jointly relative to base), the per-sample
structure should carry more of the axis identity.

Runs Probe A (peak location), Probe B (cross-axis cosine at peak), and
reconstruction fidelity (mean + top-k PCs) on ab_half_diff. Compares directly
to the delta_mix results.

Usage:
  uv run python -m src.demographic_pc.analyze_ab_half_diff_fidelity [tags...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "models/blendshape_nmf/attn_cache"
OUT_MD = ROOT / "docs/research/2026-04-23-ab-half-diff-fidelity.md"

ALL_TAGS = [
    "smile_inphase", "jaw_inphase", "alpha_interp_attn",
    "anger_rebalance", "surprise_rebalance", "disgust_rebalance",
    "pucker_rebalance", "lip_press_rebalance",
]


def _has_ab(tag: str) -> bool:
    return (CACHE / tag / "ab_half_diff.npy").exists()


def _load(tag: str):
    meta = json.load(open(CACHE / tag / "meta.json"))
    arr = np.load(CACHE / tag / "ab_half_diff.npy", mmap_mode="r")
    return meta, arr


def _peak(arr: np.ndarray) -> tuple[int, int]:
    N, S, B, _ = arr.shape
    acc = np.zeros((S, B), dtype=np.float64)
    chunk = 32
    for i in range(0, N, chunk):
        blk = arr[i:i+chunk].astype(np.float32)
        fro = np.sqrt((blk ** 2).sum(axis=-1))
        acc += fro.sum(axis=0)
    acc /= N
    idx = np.unravel_index(np.argmax(acc), acc.shape)
    return int(idx[0]), int(idx[1])


def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _fit_library(X, k):
    v_mean = X.mean(axis=0)
    Xc = X - v_mean
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return v_mean, Vt[:k]


def _reconstruct(X, v_mean, basis):
    if basis is None or basis.shape[0] == 0:
        D = v_mean[None, :]
    else:
        D = np.vstack([v_mean[None, :], basis])
    G = D @ D.T
    C = X @ D.T @ np.linalg.pinv(G)
    X_hat = C @ D
    resid = X - X_hat
    norm_x = np.linalg.norm(X, axis=1)
    norm_r = np.linalg.norm(resid, axis=1)
    captured = 1.0 - (norm_r / np.maximum(norm_x, 1e-9))
    return captured


def probe_axis(tag, ks=(0, 1, 3, 10)):
    meta, arr = _load(tag)
    s_i, b_i = _peak(arr)
    X = arr[:, s_i, b_i, :].astype(np.float32)
    N, D = X.shape
    v_mean, basis_full = _fit_library(X, max(ks))
    rows = []
    for k in ks:
        basis = basis_full[:k] if k > 0 else None
        cap = _reconstruct(X, v_mean, basis)
        rows.append((k, float(np.median(cap)),
                     float(np.percentile(cap, 25)),
                     float(np.percentile(cap, 75))))
    ev_raw = (np.linalg.svd(X - v_mean, full_matrices=False)[1]) ** 2
    ev = ev_raw / ev_raw.sum()
    return {
        "tag": tag, "N": int(N), "D": int(D),
        "step": meta["step_keys"][s_i], "block": meta["block_keys"][b_i],
        "peak_fro": float(np.linalg.norm(v_mean)),
        "rows": rows,
        "ev_top": ev[:3].tolist(),
        "k80": int(np.searchsorted(np.cumsum(ev), 0.80) + 1),
        "k95": int(np.searchsorted(np.cumsum(ev), 0.95) + 1),
        "v_mean_norm": _normalize(v_mean),
    }


def cross_axis_cos(per_axis):
    tags = [r["tag"] for r in per_axis]
    M = np.stack([r["v_mean_norm"] for r in per_axis])
    return tags, M @ M.T


def write_md(per_axis, tags, cos):
    lines = []
    lines.append("---")
    lines.append("status: live")
    lines.append("topic: metrics-and-direction-quality")
    lines.append("summary: Fidelity analysis on `ab_half_diff` (A/B pair asymmetry) "
                 "instead of `delta_mix`. Tests whether emotion axes that looked "
                 "rank-~10 on delta_mix recover axis-specific structure on the "
                 "asymmetry channel.")
    lines.append("---\n")
    lines.append("# ab_half_diff injection fidelity — 2026-04-23\n")
    lines.append("Rerun of the per-channel fidelity probe on `ab_half_diff.mean_d` "
                 "instead of `delta_mix.mean_d`. `ab_half_diff = (attn_a - attn_b) / 2`, "
                 "so it isolates what the A/B pair halves *disagree* on, while "
                 "`delta_mix = (attn_a + attn_b)/2 - attn_base` captures their joint push.\n")

    lines.append("## Per-axis reconstruction captured fraction\n")
    lines.append("| axis | N | peak (step, block) | mean-norm | k=0 | k=1 | k=3 | k=10 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in per_axis:
        cells = [f"{med:.3f} [{lo:.3f}, {hi:.3f}]" for (_, med, lo, hi) in r["rows"]]
        lines.append(
            f"| {r['tag']} | {r['N']} | ({r['step']}, {r['block']}) | {r['peak_fro']:.2f} | "
            f"{cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |"
        )
    lines.append("")

    lines.append("## Effective dimensionality\n")
    lines.append("| axis | ev1 | ev2 | ev3 | k80 | k95 |")
    lines.append("|---|---|---|---|---|---|")
    for r in per_axis:
        ev = r["ev_top"]
        lines.append(f"| {r['tag']} | {ev[0]:.3f} | {ev[1]:.3f} | {ev[2]:.3f} | {r['k80']} | {r['k95']} |")
    lines.append("")

    lines.append("## Cross-axis cosine of axis-mean `v_mean` (normalised)\n")
    lines.append("| | " + " | ".join(tags) + " |")
    lines.append("|---|" + "|".join(["---"] * len(tags)) + "|")
    for i, t in enumerate(tags):
        row = " | ".join(f"{cos[i, j]:+.3f}" for j in range(len(tags)))
        lines.append(f"| {t} | {row} |")
    off = cos[~np.eye(len(tags), dtype=bool)]
    lines.append(f"\nMean off-diagonal cos = **{off.mean():+.3f}** "
                 f"(min {off.min():+.3f}, max {off.max():+.3f})\n")

    lines.append("## Comparison to delta_mix\n")
    lines.append("Delta_mix baseline (from 2026-04-23-injection-fidelity.md): emotion cluster "
                 "shared cos 0.97–0.99, mean-only captured 0.07–0.10, k=10 plateau ~0.80.\n")
    lines.append("This file tests whether `ab_half_diff` separates emotions that "
                 "`delta_mix` collapses.\n")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines))
    print(f"\n[md] → {OUT_MD}")


def main():
    requested = sys.argv[1:] or ALL_TAGS
    ready = [t for t in requested if _has_ab(t)]
    not_ready = [t for t in requested if not _has_ab(t)]
    if not_ready:
        print(f"[wait] not yet cached: {not_ready}")
    if not ready:
        print("[exit] nothing to analyse")
        return
    print(f"[run] analysing: {ready}")
    per_axis = []
    for tag in ready:
        print(f"\n[axis] {tag}")
        r = probe_axis(tag)
        per_axis.append(r)
        print(f"  peak=({r['step']}, {r['block']})  ||v_mean||={r['peak_fro']:.2f}  "
              f"ev1={r['ev_top'][0]:.3f}  k80={r['k80']}  k95={r['k95']}")
        for (k, med, lo, hi) in r["rows"]:
            print(f"  k={k:2d}  captured median={med:.3f}  [p25={lo:.3f}, p75={hi:.3f}]")
    if len(per_axis) >= 2:
        tags, cos = cross_axis_cos(per_axis)
        print("\n[cross-axis cos]")
        for i, t in enumerate(tags):
            row = " ".join(f"{cos[i, j]:+.3f}" for j in range(len(tags)))
            print(f"  {t:28s} {row}")
        write_md(per_axis, tags, cos)
    else:
        write_md(per_axis, [per_axis[0]["tag"]],
                 np.array([[1.0]]))


if __name__ == "__main__":
    main()
