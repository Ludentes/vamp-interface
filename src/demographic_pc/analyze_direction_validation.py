"""Morning analysis: did direction injection actually shift the target atom?

Reads `output/demographic_pc/direction_inject_broad/atom{NN}_*/blendshapes.json`
and compares measured blendshape delta (rendered - scale=0 baseline) against
each atom's target direction vector (from W_nmf_resid after residualisation).

Metric per (atom, base, seed, scale):
    measured_delta_bs    = score_at_scale - score_at_scale_0     (52-d signed)
    target_direction     = W_atom[k] as signed blendshape channels (76-d → 38-d
                           signed via pos-neg difference)
    cos(measured, target) — did we move in the intended direction?
    projection = measured · target / |target|² — "how much of the target was
                 actually delivered"

A healthy atom: cos ≈ +1 at positive scales, cos ≈ -1 at negative scales,
projection monotone in scale.

Null atom (predictor-not-controller failure): cos ≈ 0 or random across scales.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NMF_DIR = ROOT / "models/blendshape_nmf"
BROAD_ROOT = ROOT / "output/demographic_pc/direction_inject_broad"


def load_target_directions():
    """Return per-atom target in 38-channel signed blendshape space.

    W_nmf_resid has shape (k, 2C) with pos/neg stacked channels. Convert to
    (k, C) signed by taking W_pos - W_neg.
    """
    W = np.load(NMF_DIR / "W_nmf_resid.npy")  # (k, 2C)
    manifest = json.loads((NMF_DIR / "manifest_resid.json").read_text())
    channels_raw = manifest["channels_raw"]
    C = len(channels_raw)
    assert W.shape[1] == 2 * C, f"W shape {W.shape} vs 2*C={2*C}"
    W_pos = W[:, :C]
    W_neg = W[:, C:]
    W_signed = W_pos - W_neg
    return W_signed, channels_raw


def project_bs_to_residual(scores: dict, channels_raw: list[str],
                           channels_full: list[str], base: str,
                           mu: np.ndarray, sigma: np.ndarray,
                           base_idx: dict) -> np.ndarray:
    """Project a raw blendshape score dict to 38-channel z-score residual."""
    x = np.array([scores.get(c, 0.0) for c in channels_full])
    bi = base_idx[base]
    sigma_safe = np.where(sigma < 1e-4, 1.0, sigma)
    x_res = (x - mu[bi]) / sigma_safe[bi]
    raw_set = set(channels_raw)
    mask = np.array([c in raw_set for c in channels_full])
    return x_res[mask]


def parse_key(rel: str) -> tuple[str, int, float] | None:
    # rel = "base/seedNNNN_s+X.XX.png"
    parts = rel.split("/")
    if len(parts) != 2:
        return None
    base = parts[0]
    fname = parts[1].rsplit(".", 1)[0]
    try:
        seed_part, scale_part = fname.split("_s")
        seed = int(seed_part.removeprefix("seed"))
        scale = float(scale_part)
    except Exception:
        return None
    return base, seed, scale


def main():
    W_signed, channels_raw = load_target_directions()
    manifest = json.loads((NMF_DIR / "manifest_resid.json").read_text())
    channels_full = manifest["channels_full"]
    unique_bases = manifest["unique_bases"]
    base_idx = {b: i for i, b in enumerate(unique_bases)}
    mu = np.load(NMF_DIR / "mu_base_resid.npy")
    sigma = np.load(NMF_DIR / "sigma_base_resid.npy")

    atom_dirs = sorted(BROAD_ROOT.glob("atom*/blendshapes.json"))
    print(f"[validate] {len(atom_dirs)} atom dirs scored")

    per_atom_summary = []
    for ajson in atom_dirs:
        atom_dir = ajson.parent
        stem = atom_dir.name  # e.g. "atom17_directions_resid_causal"
        atom_id = int(stem.removeprefix("atom").split("_", 1)[0])
        target = W_signed[atom_id]                                   # (C,)
        target_norm = np.linalg.norm(target)
        if target_norm < 1e-6:
            # dead atom
            continue

        scores = json.loads(ajson.read_text())

        # Group by (base, seed), find scale=0 baseline, compute delta per scale
        per_group: dict[tuple[str, int], dict[float, np.ndarray]] = {}
        for rel, s in scores.items():
            parsed = parse_key(rel)
            if parsed is None:
                continue
            base, seed, scale = parsed
            res = project_bs_to_residual(s, channels_raw, channels_full, base,
                                         mu, sigma, base_idx)
            per_group.setdefault((base, seed), {})[scale] = res

        rows = []
        for (base, seed), by_scale in per_group.items():
            if 0.0 not in by_scale:
                continue
            base_res = by_scale[0.0]
            for scale, res in by_scale.items():
                if scale == 0.0:
                    continue
                delta = res - base_res
                delta_norm = np.linalg.norm(delta)
                cos = float(delta @ target / (delta_norm * target_norm)) if delta_norm > 1e-6 else 0.0
                proj = float(delta @ target / (target_norm ** 2))
                rows.append({
                    "base": base, "seed": seed, "scale": scale,
                    "cos": cos, "proj": proj, "delta_norm": float(delta_norm),
                })

        if not rows:
            continue
        # Aggregate per scale across (base, seed)
        scales_sorted = sorted({r["scale"] for r in rows})
        print(f"\natom #{atom_id:02d}  target_norm={target_norm:.3f}  n_obs={len(rows)}")
        print(f"  {'scale':>6}  {'mean cos':>9}  {'mean proj':>10}  {'mean |Δ|':>9}")
        for sc in scales_sorted:
            subset = [r for r in rows if r["scale"] == sc]
            cos_m = float(np.mean([r["cos"] for r in subset]))
            proj_m = float(np.mean([r["proj"] for r in subset]))
            dnorm_m = float(np.mean([r["delta_norm"] for r in subset]))
            print(f"  {sc:>6.2f}  {cos_m:>+9.3f}  {proj_m:>+10.3f}  {dnorm_m:>9.3f}")
        per_atom_summary.append({
            "atom": atom_id,
            "rows": rows,
            "target_norm": target_norm,
        })

    # Verdict: count atoms where largest positive scale gives cos > 0.3
    wins = 0
    for row in per_atom_summary:
        max_scale_rows = [r for r in row["rows"] if r["scale"] == max(rr["scale"] for rr in row["rows"])]
        if max_scale_rows and float(np.mean([r["cos"] for r in max_scale_rows])) > 0.3:
            wins += 1
    print(f"\n[verdict] {wins}/{len(per_atom_summary)} atoms have cos>0.3 at max scale "
          f"(directional injection working)")

    (BROAD_ROOT / "validation_report.json").write_text(
        json.dumps({"atoms": per_atom_summary, "wins": wins,
                    "total": len(per_atom_summary)}, indent=2))
    print(f"[save] → {BROAD_ROOT}/validation_report.json")


if __name__ == "__main__":
    main()
