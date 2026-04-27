"""Recon pass on the compact attention cache.

Four probes over the `models/blendshape_nmf/attn_cache/<tag>/` corpora:

  A. Per-axis (step, block) Fro energy map   → where does each axis live?
  B. Cross-axis cosine at each axis's peak   → are axes distinguishable?
  C. Cross-base cosine within one axis       → is direction portable?
  D. PCA at peak block per axis              → effective dimensionality?

Writes one markdown summary + a heatmap PNG.

Usage:
  uv run python -m src.demographic_pc.analyze_attn_cache_recon
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "models/blendshape_nmf/attn_cache"
OUT_MD = ROOT / "docs/research/2026-04-23-cache-recon.md"
OUT_FIG = ROOT / "docs/research/images/2026-04-23-cache-recon-heatmaps.png"

TAGS = [
    "smile_inphase", "jaw_inphase", "alpha_interp_attn",
    "anger_rebalance", "surprise_rebalance", "disgust_rebalance",
    "pucker_rebalance", "lip_press_rebalance",
]


def _load(tag: str):
    meta = json.load(open(CACHE / tag / "meta.json"))
    arr = np.load(CACHE / tag / "delta_mix.npy", mmap_mode="r")  # (N, S, B, D) fp16
    return meta, arr


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


# ── Probe A: per-(step, block) Fro energy, averaged across samples ──────
def probe_a():
    print("\n[A] Per-axis (step × block) Fro energy")
    maps = {}
    for tag in TAGS:
        meta, arr = _load(tag)
        # Read in fp16 but reduce in fp32 for numerical safety.
        # mean over samples of ||delta||_2 per (step, block)
        N, S, B, _ = arr.shape
        # Chunk over N to keep RAM bounded
        acc = np.zeros((S, B), dtype=np.float64)
        chunk = 32
        for i in range(0, N, chunk):
            block = arr[i:i+chunk].astype(np.float32)     # (c, S, B, D)
            fro = np.sqrt((block ** 2).sum(axis=-1))      # (c, S, B)
            acc += fro.sum(axis=0)
        acc /= N
        maps[tag] = {"map": acc, "steps": meta["step_keys"], "blocks": meta["block_keys"]}
        # peak
        idx = np.unravel_index(np.argmax(acc), acc.shape)
        peak_step = meta["step_keys"][idx[0]]
        peak_block = meta["block_keys"][idx[1]]
        print(f"  {tag:28s}  peak @ step={peak_step:2d} block={peak_block:10s}  energy={acc[idx]:.3f}")
    return maps


# ── Probe B: cross-axis cosine at each axis's own peak ──────────────────
def probe_b(maps):
    print("\n[B] Cross-axis cosine at each axis's peak (step, block)")
    # For each axis, compute mean delta vector at its peak (step, block)
    # across all samples.
    peak_vecs: dict[str, np.ndarray] = {}
    peak_coord: dict[str, tuple[int, int]] = {}
    for tag in TAGS:
        meta, arr = _load(tag)
        m = maps[tag]["map"]
        idx = np.unravel_index(np.argmax(m), m.shape)
        s_i, b_i = int(idx[0]), int(idx[1])
        v = arr[:, s_i, b_i, :].astype(np.float32).mean(axis=0)   # (D,)
        peak_vecs[tag] = _normalize(v)
        peak_coord[tag] = (meta["step_keys"][s_i], meta["block_keys"][b_i])

    M = np.stack([peak_vecs[t] for t in TAGS])  # (T, D)
    cos = M @ M.T  # (T, T)
    header = " " * 30 + " ".join(f"{t[:10]:>10s}" for t in TAGS)
    print(header)
    for i, t in enumerate(TAGS):
        row = " ".join(f"{cos[i, j]:>+10.3f}" for j in range(len(TAGS)))
        print(f"  {t:28s} {row}")
    return cos, peak_coord


# ── Probe C: portability across bases within one axis ───────────────────
def probe_c(axis_tag: str, maps):
    print(f"\n[C] Cross-base cosine within '{axis_tag}' at its peak (step, block)")
    meta, arr = _load(axis_tag)
    m = maps[axis_tag]["map"]
    idx = np.unravel_index(np.argmax(m), m.shape)
    s_i, b_i = int(idx[0]), int(idx[1])
    # per-base mean delta at peak
    bases = sorted({r.split("/")[0] for r in meta["rels"]})
    vecs: dict[str, np.ndarray] = {}
    for b in bases:
        mask = np.array([r.split("/")[0] == b for r in meta["rels"]])
        idxs = np.where(mask)[0]
        v = arr[idxs, s_i, b_i, :].astype(np.float32).mean(axis=0)
        vecs[b] = _normalize(v)

    M = np.stack([vecs[b] for b in bases])
    cos = M @ M.T
    print(" " * 22 + " ".join(f"{b[:10]:>12s}" for b in bases))
    for i, b in enumerate(bases):
        row = " ".join(f"{cos[i, j]:>+12.3f}" for j in range(len(bases)))
        print(f"  {b:20s} {row}")
    # summary: mean off-diagonal
    off = cos[~np.eye(len(bases), dtype=bool)]
    print(f"  → mean off-diagonal cos = {off.mean():+.3f}   min = {off.min():+.3f}   max = {off.max():+.3f}")
    return cos, bases


# ── Probe D: PCA at peak block per axis ─────────────────────────────────
def probe_d(maps, k: int = 10):
    print(f"\n[D] PCA at each axis's peak (step, block) — top-{k} variance")
    rows = []
    for tag in TAGS:
        _, arr = _load(tag)
        m = maps[tag]["map"]
        idx = np.unravel_index(np.argmax(m), m.shape)
        s_i, b_i = int(idx[0]), int(idx[1])
        X = arr[:, s_i, b_i, :].astype(np.float32)  # (N, D)
        X = X - X.mean(axis=0, keepdims=True)
        # economy SVD on (N, D) — N ≤ 660 so this is fast
        try:
            _, s, _ = np.linalg.svd(X, full_matrices=False)
            ev = (s ** 2)
            ev = ev / ev.sum()
        except np.linalg.LinAlgError:
            ev = np.zeros(k)
        top = ev[:k]
        cum = np.cumsum(ev)
        k80 = int(np.searchsorted(cum, 0.80) + 1)
        k95 = int(np.searchsorted(cum, 0.95) + 1)
        rows.append((tag, top, k80, k95))
        print(f"  {tag:28s}  ev1={top[0]:.3f} ev2={top[1]:.3f} ev3={top[2]:.3f}"
              f"  |  k80={k80:3d}  k95={k95:3d}")
    return rows


# ── Figure: 8 heatmaps of (step × block) Fro energy ─────────────────────
def write_heatmaps(maps):
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    for ax, tag in zip(axes.flat, TAGS):
        m = maps[tag]["map"]
        im = ax.imshow(m, aspect="auto", cmap="magma", origin="lower")
        ax.set_title(tag, fontsize=10)
        ax.set_xlabel("block idx")
        ax.set_ylabel("step idx")
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle("delta_mix Fro energy per (step, block), averaged across samples",
                 fontsize=12)
    fig.savefig(OUT_FIG, dpi=110)
    plt.close(fig)
    print(f"\n[fig] → {OUT_FIG}")


def write_md(maps, cos_axes, peak_coord, c_cos, c_bases, d_rows):
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("---")
    lines.append("status: live")
    lines.append("topic: metrics-and-direction-quality")
    lines.append("summary: Recon pass on the compact attention cache. "
                 "Locates per-axis energy, measures cross-axis distinguishability, "
                 "portability across bases, and effective dimensionality at peak block.")
    lines.append("---\n")
    lines.append("# Attention-cache recon — 2026-04-23\n")
    lines.append("Ran four probes over `models/blendshape_nmf/attn_cache/<tag>/` "
                 "(8 axis/subtag corpora, delta_mix shape `(N, 16 steps, 57 blocks, 3072 D)` fp16).\n")

    lines.append("## Probe A — peak location per axis\n")
    lines.append("| axis | peak step | peak block | peak Fro |")
    lines.append("|---|---|---|---|")
    for tag in TAGS:
        m = maps[tag]["map"]
        idx = np.unravel_index(np.argmax(m), m.shape)
        ps, pb = peak_coord[tag]
        lines.append(f"| {tag} | {ps} | {pb} | {m[idx]:.3f} |")
    lines.append("\nHeatmap: ![](images/2026-04-23-cache-recon-heatmaps.png)\n")

    lines.append("## Probe B — cross-axis cosine at peaks\n")
    lines.append("Normalized mean δ at each axis's own peak (step, block). "
                 "High off-diagonal = axes share direction; block-diagonal = distinct.\n")
    lines.append("| | " + " | ".join(TAGS) + " |")
    lines.append("|---|" + "|".join(["---"] * len(TAGS)) + "|")
    for i, t in enumerate(TAGS):
        row = " | ".join(f"{cos_axes[i, j]:+.3f}" for j in range(len(TAGS)))
        lines.append(f"| {t} | {row} |")
    off = cos_axes[~np.eye(len(TAGS), dtype=bool)]
    lines.append(f"\nMean off-diagonal cos = **{off.mean():+.3f}** "
                 f"(min {off.min():+.3f}, max {off.max():+.3f}).\n")

    lines.append("## Probe C — portability across bases (smile_inphase)\n")
    lines.append("Per-base mean δ at smile_inphase's peak (step, block), cosine across 6 bases.\n")
    lines.append("| | " + " | ".join(c_bases) + " |")
    lines.append("|---|" + "|".join(["---"] * len(c_bases)) + "|")
    for i, b in enumerate(c_bases):
        row = " | ".join(f"{c_cos[i, j]:+.3f}" for j in range(len(c_bases)))
        lines.append(f"| {b} | {row} |")
    off = c_cos[~np.eye(len(c_bases), dtype=bool)]
    lines.append(f"\nMean off-diagonal cos = **{off.mean():+.3f}** "
                 f"(min {off.min():+.3f}, max {off.max():+.3f}).\n")

    lines.append("## Probe D — effective dimensionality at peak block\n")
    lines.append("PCA on `(N, 3072)` at peak (step, block). `k80` / `k95` = rank needed "
                 "to explain 80% / 95% of variance.\n")
    lines.append("| axis | ev1 | ev2 | ev3 | k80 | k95 |")
    lines.append("|---|---|---|---|---|---|")
    for tag, top, k80, k95 in d_rows:
        lines.append(f"| {tag} | {top[0]:.3f} | {top[1]:.3f} | {top[2]:.3f} | {k80} | {k95} |")
    lines.append("")

    lines.append("## Interpretation hooks\n")
    lines.append("- **If B is block-diagonal** → axes are distinct directions → a per-axis "
                 "library is tractable. **If B shows broad off-diagonal mass** → many "
                 "axes are entangled in the same subspace → portable library is harder.\n")
    lines.append("- **If C mean-off-diag cos is high (>0.7)** → a single cross-base smile "
                 "direction exists → future renders can skip live prompt-pair passes. "
                 "**If low (<0.3)** → `(axis, base)` keying (the dictionary's current choice) is correct.\n")
    lines.append("- **If D `ev1 > 0.6`** → axis is effectively rank-1, one vector stores it. "
                 "**If k80 > 10** → richer structure; library would have to be a basis, "
                 "not a vector.\n")
    lines.append("## Next steps\n")
    lines.append("- Extend `cache_attn_features.py` to cover `promptpair_iterate/` sweeps "
                 "(the adjacency-validated smile/age/race runs) and re-run this recon on the validated corpus.\n")
    lines.append("- If axes look portable (B + C pass thresholds): prototype a "
                 "`FluxSpaceInjectCached` node that consumes a cached δ and reproduces "
                 "the prompt-pair edit without the 2N+1 forward passes.\n")

    OUT_MD.write_text("\n".join(lines))
    print(f"[md]  → {OUT_MD}")


def main():
    maps = probe_a()
    write_heatmaps(maps)
    cos_axes, peak_coord = probe_b(maps)
    c_cos, c_bases = probe_c("smile_inphase", maps)
    d_rows = probe_d(maps)
    write_md(maps, cos_axes, peak_coord, c_cos, c_bases, d_rows)


if __name__ == "__main__":
    main()
