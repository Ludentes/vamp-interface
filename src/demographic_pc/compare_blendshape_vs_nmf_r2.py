"""Side-by-side: per-blendshape ridge R² vs NMF-component ridge R², per corpus.

For each NMF component in au_library.npz, we know:
  - the top blendshape loading (e.g. C6 smile → mouthSmileLeft)
  - the component's per-corpus CV R²

Question: does regressing δ directly against the single dominant blendshape
beat regressing against the NMF component's score? If yes, the blendshape
basis is finer-grained and the library should store per-blendshape vectors
rather than per-component. If no, the NMF is already extracting the signal.

Also runs a full 52-row ridge sweep with a σ(y) > 0.05 filter to keep only
blendshapes that vary meaningfully in at least one corpus.

Emits:
  docs/research/2026-04-23-blendshape-vs-nmf-r2.md
  output/demographic_pc/blendshape_vs_nmf_r2.csv

Usage:
  uv run python -m src.demographic_pc.compare_blendshape_vs_nmf_r2
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "models/blendshape_nmf/attn_cache"
METRICS = ROOT / "output/demographic_pc/fluxspace_metrics"
LIB_NPZ = ROOT / "models/blendshape_nmf/au_library.npz"
OUT_MD = ROOT / "docs/research/2026-04-23-blendshape-vs-nmf-r2.md"
OUT_CSV = ROOT / "output/demographic_pc/blendshape_vs_nmf_r2.csv"

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

# Canonical AU-like blendshapes (the ones with a clear FACS mapping).
CANONICAL_AU_BLENDSHAPES = [
    "mouthSmileLeft", "mouthSmileRight",        # AU12 smile corners
    "mouthPucker",                              # AU18 pucker
    "mouthFunnel",                              # AU22 funnel
    "mouthPressLeft", "mouthPressRight",        # AU24 lip press
    "mouthUpperUpLeft", "mouthUpperUpRight",    # AU10 upper-lip raise
    "mouthLowerDownLeft", "mouthLowerDownRight",# AU16 lower-lip depress
    "jawOpen",                                  # AU26 jaw drop
    "browInnerUp",                              # AU1 inner brow raise
    "browOuterUpLeft", "browOuterUpRight",      # AU2 outer brow raise
    "browDownLeft", "browDownRight",            # AU4 brow lowerer
    "noseSneerLeft", "noseSneerRight",          # AU9 nose wrinkle
    "eyeSquintLeft", "eyeSquintRight",          # AU7 lid tighten
    "eyeBlinkLeft", "eyeBlinkRight",            # AU45 blink
    "cheekPuff",                                # AU34 cheek puff
    "cheekSquintLeft", "cheekSquintRight",      # AU6 cheek raise
]


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


def load_Y_for_tag(tag, names):
    meta = json.load(open(CACHE / tag / "meta.json"))
    rels = meta["rels"]
    data = json.loads(TAG_TO_BS[tag].read_text())
    Y = np.zeros((len(rels), len(names)), dtype=np.float32)
    for i, r in enumerate(rels):
        d = data.get(r, {})
        for j, name in enumerate(names):
            Y[i, j] = d.get(name, 0.0)
    return Y


def load_X_at_peak(tag):
    arr = np.load(CACHE / tag / "delta_mix.npy", mmap_mode="r")
    s_i, b_i = _peak(arr)
    X = arr[:, s_i, b_i, :].astype(np.float32)
    X = X - X.mean(axis=0, keepdims=True)
    return X


def main():
    lib = np.load(LIB_NPZ, allow_pickle=True)
    H = lib["H"]                       # (K, 52)
    per_comp_r2 = lib["per_comp_r2"]   # (K, T)
    tags_arr = lib["tags"]
    names_arr = lib["names"]
    K = H.shape[0]
    tags = list(tags_arr)
    names = list(names_arr)

    # Pre-fit δ and Y per tag
    X_by_tag = {}
    Y_by_tag = {}
    for tag in tags:
        X_by_tag[tag] = load_X_at_peak(tag)
        Y_by_tag[tag] = load_Y_for_tag(tag, names)

    # -------- Part 1: per-blendshape ridge on canonical AUs --------
    print("[part1] per-blendshape ridge on canonical AUs")
    canonical_rows = []      # (bs_name, per-tag R² dict, per-tag σ dict)
    for bs_name in CANONICAL_AU_BLENDSHAPES:
        if bs_name not in names:
            continue
        j = names.index(bs_name)
        r2_by_tag = {}
        sig_by_tag = {}
        for tag in tags:
            y = Y_by_tag[tag][:, j]
            sig = float(y.std())
            sig_by_tag[tag] = sig
            if sig < 0.01:
                r2_by_tag[tag] = float("nan")
                continue
            r2_by_tag[tag] = _cv_r2(X_by_tag[tag], y)
        canonical_rows.append((bs_name, r2_by_tag, sig_by_tag))
        best = max((v for v in r2_by_tag.values() if not np.isnan(v)), default=float("nan"))
        print(f"  {bs_name:22s}  best R²={best:+.3f}")

    # -------- Part 2: side-by-side NMF vs dominant-blendshape per component --------
    print("\n[part2] NMF component vs its dominant blendshape")
    side_by_side = []
    for c in range(K):
        dom_j = int(np.argmax(H[c]))
        dom_name = names[dom_j]
        # Per-tag: component R² vs dominant-blendshape R²
        comp_r2_row = per_comp_r2[c]
        bs_r2_row = np.zeros(len(tags), dtype=np.float32)
        bs_sig_row = np.zeros(len(tags), dtype=np.float32)
        for ti, tag in enumerate(tags):
            y = Y_by_tag[tag][:, dom_j]
            sig = float(y.std())
            bs_sig_row[ti] = sig
            if sig < 0.01:
                bs_r2_row[ti] = float("nan")
                continue
            bs_r2_row[ti] = _cv_r2(X_by_tag[tag], y)
        side_by_side.append((c, dom_name, comp_r2_row, bs_r2_row, bs_sig_row))
        print(f"  C{c} (dom={dom_name})")
        for ti, tag in enumerate(tags):
            cr = comp_r2_row[ti]
            br = bs_r2_row[ti]
            if np.isnan(br):
                delta = float("nan")
            else:
                delta = float(br) - float(cr)
            print(f"    {tag:24s}  comp={cr:+.3f}  bs={br:+.3f}  Δ={delta:+.3f}  σ(bs)={bs_sig_row[ti]:.3f}")

    # -------- Part 3: 52-wide scan filtered by σ --------
    print("\n[part3] full 52 scan (filtered: σ(y) > 0.05 in at least one corpus)")
    full_rows = []
    for j, bs_name in enumerate(names):
        if bs_name == "_neutral":
            continue
        max_sig = max(float(Y_by_tag[tag][:, j].std()) for tag in tags)
        if max_sig < 0.05:
            continue
        r2_by_tag = {}
        sig_by_tag = {}
        for tag in tags:
            y = Y_by_tag[tag][:, j]
            sig = float(y.std())
            sig_by_tag[tag] = sig
            if sig < 0.01:
                r2_by_tag[tag] = float("nan")
                continue
            r2_by_tag[tag] = _cv_r2(X_by_tag[tag], y)
        full_rows.append((bs_name, r2_by_tag, sig_by_tag, max_sig))

    # Rank by best R² across corpora
    full_rows.sort(
        key=lambda row: max((v for v in row[1].values() if not np.isnan(v)),
                             default=-1.0),
        reverse=True,
    )

    # -------- Write CSV + MD --------
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["blendshape", "max_sigma"] + [f"r2_{t}" for t in tags]
                   + [f"sigma_{t}" for t in tags])
        for bs, r2d, sd, ms in full_rows:
            w.writerow([bs, f"{ms:.3f}"]
                       + [f"{r2d[t]:+.3f}" if not np.isnan(r2d[t]) else ""
                          for t in tags]
                       + [f"{sd[t]:.3f}" for t in tags])
    print(f"[csv] → {OUT_CSV}")

    lines = []
    lines.append("---")
    lines.append("status: live")
    lines.append("topic: metrics-and-direction-quality")
    lines.append("summary: Side-by-side comparison of per-blendshape ridge R² against "
                 "NMF-component R² at each corpus's peak block. Tests whether a "
                 "finer-grained per-blendshape library would outperform the 8-component "
                 "NMF library.")
    lines.append("---\n")
    lines.append("# Per-blendshape vs NMF component R² — 2026-04-23\n")
    lines.append("Fitting ridge directly on individual ARKit blendshapes at each "
                 "corpus's peak block, side-by-side with the NMF-component R² from "
                 "`au_library.npz`.\n")

    lines.append("## Canonical AU blendshapes — best R² across corpora\n")
    lines.append("| blendshape | " + " | ".join(tags) + " |")
    lines.append("|---|" + "|".join(["---"] * len(tags)) + "|")
    for bs, r2d, sd in canonical_rows:
        row = " | ".join(
            ("—" if np.isnan(r2d[t]) else f"{r2d[t]:+.3f} (σ={sd[t]:.2f})")
            for t in tags
        )
        lines.append(f"| {bs} | {row} |")
    lines.append("")

    lines.append("## NMF component R² vs its dominant blendshape\n")
    lines.append("For each component, the dominant blendshape is `argmax(H[c])`. "
                 "Δ = per-blendshape R² − per-component R². Positive Δ means the "
                 "direct blendshape direction beats the NMF component at this corpus.\n")
    for c, dom_name, comp_r, bs_r, bs_sig in side_by_side:
        lines.append(f"### C{c} dominant = `{dom_name}`\n")
        lines.append("| corpus | NMF R² | blendshape R² | Δ | σ(bs) |")
        lines.append("|---|---|---|---|---|")
        for ti, tag in enumerate(tags):
            cr = float(comp_r[ti])
            br = float(bs_r[ti]) if not np.isnan(bs_r[ti]) else None
            if br is None:
                lines.append(f"| {tag} | {cr:+.3f} | — | — | {bs_sig[ti]:.2f} |")
            else:
                lines.append(f"| {tag} | {cr:+.3f} | {br:+.3f} | {br - cr:+.3f} | {bs_sig[ti]:.2f} |")
        lines.append("")

    lines.append("## Full 52 scan, σ-filtered, ranked by best R²\n")
    lines.append(f"Kept {len(full_rows)} blendshapes with σ(y) > 0.05 in at least "
                 "one corpus. CSV at `output/demographic_pc/blendshape_vs_nmf_r2.csv`. "
                 "Top 20 shown here.\n")
    lines.append("| blendshape | max σ | " + " | ".join(tags) + " |")
    lines.append("|---|---|" + "|".join(["---"] * len(tags)) + "|")
    for bs, r2d, sd, ms in full_rows[:20]:
        row = " | ".join(
            ("—" if np.isnan(r2d[t]) else f"{r2d[t]:+.3f}") for t in tags
        )
        lines.append(f"| {bs} | {ms:.2f} | {row} |")
    lines.append("")

    lines.append("## How to read\n")
    lines.append("- **Δ mostly positive** → per-blendshape library beats per-component. "
                 "Each blendshape direction is finer than the component's mixture, and "
                 "an injection library should store per-blendshape `w` vectors.\n")
    lines.append("- **Δ mostly zero** → NMF component already captures the same signal. "
                 "Use the 8-component library; it's smaller and cleaner.\n")
    lines.append("- **Δ negative** → the NMF component's broader loading is actually "
                 "more predictable than any single blendshape (averaging noise across "
                 "co-activating AUs helps).\n")
    lines.append("- **σ(bs) small** → per-blendshape R² is dominated by noise; ignore.\n")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines))
    print(f"[md] → {OUT_MD}")


if __name__ == "__main__":
    main()
