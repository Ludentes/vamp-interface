"""For each existing text-pair axis in the rebalance corpus, quantify
which atoms it moves, how strongly, and whether the response is
base-invariant or base-specific.

Data already in hand:
  - rebalance corpus: 5 text-pair axes (anger, surprise, disgust, pucker,
    lip_press) × 6 bases × 5 seeds × 5 scales (only s ∈ {0.4, 0.8, 1.2}
    face-detectable after scale-collapse cut) × 2 start_percents.
  - Each sample has MediaPipe blendshape scores (from score_blendshapes).
  - Atom basis W_nmf_resid.npy + mu_base / sigma_base for per-base z-score.

For each axis we compute:
  * mean atom coefficient at the highest usable scale (1.2)
  * difference vs atom coefficient at the lowest usable scale (0.4)
  * per-base breakdown (are the same atoms moved on every base?)

A clean text-pair-for-atom-k pick has: single dominant atom with large
positive delta across all 6 bases.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "output/demographic_pc/fluxspace_metrics/crossdemo"
NMF_DIR = ROOT / "models/blendshape_nmf"

AXES = ["anger", "surprise", "disgust", "pucker", "lip_press"]
LOW_SCALE_STR = "s+0.40"
HIGH_SCALE_STR = "s+1.20"
DEAD_ATOMS = {0, 2, 3, 6}


def load_nmf():
    W = np.load(NMF_DIR / "W_nmf_resid.npy")
    manifest = json.loads((NMF_DIR / "manifest_resid.json").read_text())
    mu = np.load(NMF_DIR / "mu_base_resid.npy")
    sigma = np.load(NMF_DIR / "sigma_base_resid.npy")
    channels_raw = manifest["channels_raw"]
    channels_full = manifest["channels_full"]
    unique_bases = manifest["unique_bases"]
    raw_set = set(channels_raw)
    prune_mask = np.array([c in raw_set for c in channels_full])
    W_pinv = np.linalg.pinv(W)
    base_idx = {b: i for i, b in enumerate(unique_bases)}
    return W, W_pinv, mu, sigma, channels_raw, channels_full, prune_mask, base_idx


def project_to_atoms(scores: dict, channels_full: list[str],
                     prune_mask: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
                     base_idx_val: int, W_pinv: np.ndarray) -> np.ndarray:
    x = np.array([scores.get(c, 0.0) for c in channels_full])
    sigma_safe = np.where(sigma < 1e-4, 1.0, sigma)
    x_res = (x - mu[base_idx_val]) / sigma_safe[base_idx_val]
    x_res = x_res[prune_mask]
    x_pos = np.clip(x_res, 0.0, None)
    x_neg = np.clip(-x_res, 0.0, None)
    stacked = np.concatenate([x_pos, x_neg])
    y = np.clip(stacked @ W_pinv, 0.0, None)
    return y


def parse_base(key: str, base_idx: dict) -> str | None:
    head = key.split("/", 1)[0]
    return head if head in base_idx else None


def main():
    (W, W_pinv, mu, sigma, channels_raw, channels_full,
     prune_mask, base_idx) = load_nmf()
    n_atoms = W.shape[0]
    unique_bases = sorted(base_idx, key=base_idx.get)
    live_atoms = [k for k in range(n_atoms) if k not in DEAD_ATOMS]

    regions = {r["atom"]: r.get("region", "?")
               for r in json.loads((NMF_DIR / "atom_regions.json").read_text())}

    print(f"\n{'='*100}\nText-pair → atom coverage (high scale 1.2 − low scale 0.4)\n{'='*100}")

    # For each axis, accumulate per-base atom deltas
    axis_per_base_delta = {}  # axis → base → (n_atoms,)
    axis_all_delta = {}       # axis → (n_atoms,)

    for axis in AXES:
        bs_json = METRICS / axis / "rebalance/blendshapes.json"
        if not bs_json.exists():
            continue
        data = json.loads(bs_json.read_text())
        per_base = {b: {"low": [], "high": []} for b in unique_bases}
        for rel, scores in data.items():
            base = parse_base(rel, base_idx)
            if base is None:
                continue
            y = project_to_atoms(scores, channels_full, prune_mask, mu, sigma,
                                 base_idx[base], W_pinv)
            if LOW_SCALE_STR in rel:
                per_base[base]["low"].append(y)
            elif HIGH_SCALE_STR in rel:
                per_base[base]["high"].append(y)
        per_base_delta = {}
        for b in unique_bases:
            if not per_base[b]["low"] or not per_base[b]["high"]:
                continue
            delta_b = (np.array(per_base[b]["high"]).mean(0) -
                       np.array(per_base[b]["low"]).mean(0))
            per_base_delta[b] = delta_b
        all_delta = np.array(list(per_base_delta.values())).mean(0)
        axis_per_base_delta[axis] = per_base_delta
        axis_all_delta[axis] = all_delta

    # Summary table: per axis, top-3 atoms it moves + cross-base consistency
    print(f"\n{'axis':<12} {'top atom':<5} {'Δ mean':>7} {'Δ signed per base':<70}")
    print("-" * 105)
    for axis in AXES:
        if axis not in axis_all_delta:
            continue
        mean_delta = axis_all_delta[axis]
        # Only consider live atoms
        ranked = sorted([(mean_delta[k], k) for k in live_atoms], reverse=True)
        for rank, (val, k) in enumerate(ranked[:3]):
            per_base = axis_per_base_delta[axis]
            per_base_vals = " ".join(f"{b[:5]}:{per_base[b][k]:+5.2f}"
                                     for b in unique_bases if b in per_base)
            region = regions.get(k, "?")
            prefix = f"{axis:<12}" if rank == 0 else " " * 12
            print(f"{prefix} #{k:02d} ({region[:8]:<8}) {val:>+7.3f}  {per_base_vals}")

    # Reverse table: per atom, which axis moves it most
    print(f"\n\n{'='*100}\nAtom → best text-pair (sorted by best Δ)\n{'='*100}")
    print(f"{'atom':<4} {'region':<14} {'best axis':<12} {'Δ':>7}  runners-up")
    print("-" * 90)
    per_atom_scores = []
    for k in live_atoms:
        scores_by_axis = [(axis, axis_all_delta[axis][k])
                          for axis in AXES if axis in axis_all_delta]
        scores_by_axis.sort(key=lambda x: -x[1])
        per_atom_scores.append((k, scores_by_axis))
    # sort by best-axis delta descending
    per_atom_scores.sort(key=lambda x: -x[1][0][1])
    for k, scores_by_axis in per_atom_scores:
        best_axis, best_delta = scores_by_axis[0]
        runners = ", ".join(f"{a}({d:+.2f})" for a, d in scores_by_axis[1:])
        print(f"#{k:02d}  {regions.get(k, '?'):<14} {best_axis:<12} {best_delta:>+7.3f}  {runners}")

    # Save
    out = {
        "axes": AXES,
        "live_atoms": live_atoms,
        "regions": regions,
        "bases": unique_bases,
        "axis_all_delta": {a: axis_all_delta[a].tolist() for a in axis_all_delta},
        "axis_per_base_delta": {a: {b: v.tolist() for b, v in d.items()}
                                for a, d in axis_per_base_delta.items()},
    }
    dest = NMF_DIR / "textpair_atom_coverage.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\n[save] → {dest}")


if __name__ == "__main__":
    main()
