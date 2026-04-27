"""Paired-contrast ridge fit on bootstrap_v1.

For each axis (smile, age, glasses):
  * Pick a blendshape channel as the continuous target (e.g. mouthSmile for
    smile axis).
  * Compute Δattn per (base, seed) = attn_at_levelX − attn_at_level01 for
    every non-baseline level.
  * Compute Δblendshape similarly.
  * Ridge: direction β (in attn space) ≈ Δattn / Δblendshape per-dim,
    aggregated via cov(x,y)/(var(x)+λ).
  * Compare β to existing pair-averaging δ from crossdemo measurements —
    |cos|_p95 across shared (block, step) keys.

Outputs: direction tensors + comparison table.
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BOOT = ROOT / "output" / "demographic_pc" / "fluxspace_metrics" / "bootstrap_v1"
CROSS = ROOT / "output" / "demographic_pc" / "fluxspace_metrics" / "crossdemo"
OUT_DIR = ROOT / "output" / "demographic_pc" / "fluxspace_metrics" / "paired_ridge"

# Map axis → list of target blendshape channels. Each channel gets its own
# direction; pairwise |cos| between directions is reported as explicit
# entanglement so the user knows which channels can't be independently steered.
AXIS_TARGETS = {
    "smile":   ["mouthSmileLeft", "mouthSmileRight", "jawOpen",
                "cheekSquintLeft", "cheekSquintRight",
                "mouthOpen", "mouthStretchLeft", "mouthStretchRight"],
    "age":     None,   # no direct blendshape analog; would need separate age estimator
    "glasses": None,   # blendshapes don't encode glasses
}

# Pair-δ reference: existing crossdemo measurements. For cos comparison.
PAIR_REF = {
    "smile":   CROSS / "smile" / "measurement",
    "glasses": CROSS / "measurement",
}

FNAME_RE = re.compile(
    r"^(?P<base>[a-z_]+?)_(?P<level_id>\d\d_[a-z_]+)_s(?P<seed>\d+)$"
)


def parse_stem(stem: str) -> dict | None:
    m = FNAME_RE.match(stem)
    return m.groupdict() if m else None


def load_attn(pkl_path: Path, keys: list[tuple[str, int]]) -> np.ndarray | None:
    """Return attn_base.mean_d flattened across provided keys, shape (K_keys, D)."""
    d = pickle.load(pkl_path.open("rb"))
    rows = []
    for bk, st in keys:
        entry = d["steps"].get(st, {}).get(bk)
        if entry is None:
            return None
        v = entry["attn_base"]["mean_d"]
        rows.append(v.numpy() if hasattr(v, "numpy") else v)
    return np.stack(rows).astype(np.float32)


def get_shared_keys(pkl_paths: list[Path]) -> list[tuple[str, int]]:
    """Intersection of (block, step) keys across all pkls."""
    shared = None
    for p in pkl_paths:
        d = pickle.load(p.open("rb"))
        keys = {(bk, st) for st, blocks in d["steps"].items() for bk in blocks}
        shared = keys if shared is None else shared & keys
    return sorted(shared) if shared else []


def load_pair_delta(pkl_path: Path, keys: list[tuple[str, int]]) -> tuple[np.ndarray, list[int]] | None:
    """Load δ_mix per (block, step) aligned to `keys`, return (delta[K_valid, D], valid_indices)."""
    d = pickle.load(pkl_path.open("rb"))
    rows, valid = [], []
    for idx, (bk, st) in enumerate(keys):
        entry = d["steps"].get(st, {}).get(bk)
        if entry is None or "delta_mix" not in entry:
            continue
        v = entry["delta_mix"]["mean_d"]
        rows.append(v.numpy() if hasattr(v, "numpy") else v)
        valid.append(idx)
    if not rows:
        return None
    return np.stack(rows).astype(np.float32), valid


def fit_axis(axis: str, targets: list[str], bs_scores: dict) -> dict:
    axis_dir = BOOT / axis
    pkls = sorted(axis_dir.glob("*.pkl"))
    print(f"\n=== {axis}: {len(pkls)} pkls  targets={targets} ===")

    # Parse and group by (base, seed)
    groups: dict[tuple[str, int], list[dict]] = {}
    for p in pkls:
        tag = parse_stem(p.stem)
        if tag is None:
            continue
        png_rel = f"{axis}/{p.stem}.png"
        bs = bs_scores.get(png_rel)
        if bs is None:
            continue
        # Target vector: one value per blendshape channel.
        target_vec = np.array([bs.get(t, 0.0) for t in targets], dtype=np.float32)
        entry = {"path": p, "base": tag["base"], "seed": int(tag["seed"]),
                 "level_id": tag["level_id"], "target_vec": target_vec}
        groups.setdefault((tag["base"], int(tag["seed"])), []).append(entry)

    keys = get_shared_keys(pkls[:5])
    print(f"  shared keys: {len(keys)}")

    # Build Δattn (N, K_keys*D) and ΔT_vec (N, K_channels) as paired contrasts
    # against the level-01 baseline within each (base, seed) group.
    delta_attn, delta_target, group_tags = [], [], []
    for (base, seed), items in groups.items():
        items.sort(key=lambda e: e["level_id"])
        if not items[0]["level_id"].startswith("01_"):
            continue
        attn0 = load_attn(items[0]["path"], keys)
        if attn0 is None:
            continue
        t0 = items[0]["target_vec"]
        for it in items[1:]:
            attn = load_attn(it["path"], keys)
            if attn is None:
                continue
            delta_attn.append((attn - attn0).reshape(-1))
            delta_target.append(it["target_vec"] - t0)
            group_tags.append((base, seed, it["level_id"]))

    X = np.stack(delta_target, axis=0)      # (N, K_chan)
    Y = np.stack(delta_attn, axis=0)        # (N, K_keys*D)
    Kc = X.shape[1]
    print(f"  N={len(X)}  X shape={X.shape}  Y shape={Y.shape}")
    print(f"  per-channel ΔT mean±std:")
    for i, t in enumerate(targets):
        print(f"    {t:>22}  {X[:, i].mean():+.3f} ± {X[:, i].std():.3f}  "
              f"range=[{X[:, i].min():+.3f},{X[:, i].max():+.3f}]")

    # Multi-output ridge: β = (X'X + λI)⁻¹ X'Y, shape (K_chan, K_keys*D)
    # Relative regularisation: scale λ by tr(X'X)/K_chan.
    XtX = X.T @ X
    lam = 1e-3 * XtX.trace() / Kc
    inv = np.linalg.inv(XtX + lam * np.eye(Kc))
    beta = inv @ (X.T @ Y)                  # (K_chan, K_keys*D)
    K_keys = len(keys)
    D = Y.shape[1] // K_keys
    beta_3d = beta.reshape(Kc, K_keys, D)

    # Overall R² (multi-output)
    pred = X @ beta
    rss = ((Y - pred) ** 2).sum()
    tss = ((Y - Y.mean(axis=0, keepdims=True)) ** 2).sum()
    r2 = 1.0 - rss / tss
    print(f"  R² (multi-output, all Δattn dims): {r2:.3f}")

    # Pairwise |cos| between channel directions (entanglement report).
    # Uses flattened β vectors per channel.
    print(f"\n  pairwise |cos| between channel directions (entanglement):")
    header = "  " + " " * 22 + "  " + "  ".join(f"{t[:7]:>7}" for t in targets)
    print(header)
    flat = beta.reshape(Kc, -1)
    norms = np.linalg.norm(flat, axis=1)
    cos_mat = (flat @ flat.T) / np.outer(norms, norms + 1e-12)
    for i, t in enumerate(targets):
        row = "  " + f"{t:>22}  " + "  ".join(f"{cos_mat[i, j]:+.3f}" for j in range(Kc))
        print(row)

    # Compare each channel direction to pair-averaging δ (smile axis only).
    comparisons: dict = {}
    ref_dir = PAIR_REF.get(axis)
    if ref_dir and ref_dir.exists():
        print(f"\n  per-channel vs pair-averaging δ (all 6 bases, asian_m shown):")
        for p in sorted(ref_dir.glob("*_meas.pkl")):
            base_ref = p.stem.replace("_meas", "")
            loaded = load_pair_delta(p, keys)
            if loaded is None:
                continue
            delta, valid = loaded
            comparisons[base_ref] = {}
            for k in range(Kc):
                beta_v = beta_3d[k, valid]                # (K_valid, D)
                cos_per_key = []
                for kk in range(len(valid)):
                    num = float(beta_v[kk] @ delta[kk])
                    den = float(np.linalg.norm(beta_v[kk]) * np.linalg.norm(delta[kk]) + 1e-12)
                    cos_per_key.append(num / den)
                a = np.array(cos_per_key)
                comparisons[base_ref][targets[k]] = {
                    "abs_mean": float(np.abs(a).mean()),
                    "abs_p95": float(np.percentile(np.abs(a), 95)),
                    "mean": float(a.mean()),
                }
            if base_ref == "asian_m":
                for tname in targets:
                    s = comparisons[base_ref][tname]
                    print(f"    {tname:>22}  |cos|_p95={s['abs_p95']:.3f}  "
                          f"|cos|_mean={s['abs_mean']:.3f}  cos_mean={s['mean']:+.3f}")

        # Best single-channel |cos|_p95 across all 6 bases × channels
        print(f"\n  best |cos|_p95 per base (which channel direction aligns best with pair δ):")
        for base_ref, per_chan in comparisons.items():
            best = max(per_chan.items(), key=lambda kv: kv[1]["abs_p95"])
            print(f"    {base_ref:>20}  best={best[0]:>22}  "
                  f"|cos|_p95={best[1]['abs_p95']:.3f}")

    return {
        "axis": axis,
        "N": len(X),
        "R2": float(r2),
        "targets": targets,
        "beta_shape": list(beta_3d.shape),
        "keys": keys,
        "entanglement_cos": cos_mat.tolist(),
        "comparisons_to_pair": comparisons,
        "beta": beta_3d,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (BOOT / "blendshapes.json").open() as f:
        bs_scores = json.load(f)
    print(f"[paired] loaded {len(bs_scores)} blendshape entries")

    results = {}
    for axis, targets in AXIS_TARGETS.items():
        if targets is None:
            print(f"\n[paired] skipping {axis} (no blendshape target defined)")
            continue
        results[axis] = fit_axis(axis, targets, bs_scores)

    # Save
    with (OUT_DIR / "paired_ridge_results.pkl").open("wb") as f:
        pickle.dump(results, f)
    # JSON summary w/o the big beta tensor
    summary = {
        axis: {k: v for k, v in r.items() if k not in ("beta", "keys")}
        for axis, r in results.items()
    }
    with (OUT_DIR / "paired_ridge_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nsaved → {OUT_DIR / 'paired_ridge_results.pkl'}")
    print(f"summary → {OUT_DIR / 'paired_ridge_summary.json'}")


if __name__ == "__main__":
    main()
