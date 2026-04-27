"""Paired-contrast ridge with PCA-decorrelated blendshape targets +
double-blocks-only analysis.

Two improvements over fit_paired_ridge:
  1. PCA on blendshape target columns before ridge — handles bilateral
     multicollinearity (mouthSmileLeft ≈ mouthSmileRight) by rotating into
     independent components. Direction per PC is unambiguous.
  2. For comparison with FluxSpace pair-averaging δ, restrict to double-block
     keys only. Single blocks carry almost no smile signal per prior analysis.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from src.demographic_pc.fit_paired_ridge import (
    BOOT, CROSS, get_shared_keys, load_attn, load_pair_delta, parse_stem,
)

OUT_DIR = Path("/home/newub/w/vamp-interface/output/demographic_pc/fluxspace_metrics/paired_ridge_pca")

SMILE_TARGETS = [
    "mouthSmileLeft", "mouthSmileRight",
    "jawOpen",
    "mouthStretchLeft", "mouthStretchRight",
    # Drop cheekSquint*, mouthOpen — always ~0 on Flux portraits per prior run.
]


def load_bootstrap_data(axis: str, targets: list[str], bs_scores: dict,
                        keys: list[tuple[str, int]]):
    """Return (X, Y, group_tags) where X = (N, K_chan) Δblendshapes and
    Y = (N, K_keys, D) Δattn against level-01 baseline per (base, seed)."""
    pkls = sorted((BOOT / axis).glob("*.pkl"))
    groups: dict[tuple[str, int], list[dict]] = {}
    for p in pkls:
        tag = parse_stem(p.stem)
        if tag is None:
            continue
        png_rel = f"{axis}/{p.stem}.png"
        bs = bs_scores.get(png_rel)
        if bs is None:
            continue
        target_vec = np.array([bs.get(t, 0.0) for t in targets], dtype=np.float32)
        entry = {"path": p, "base": tag["base"], "seed": int(tag["seed"]),
                 "level_id": tag["level_id"], "target_vec": target_vec}
        groups.setdefault((tag["base"], int(tag["seed"])), []).append(entry)

    dA, dT, tags = [], [], []
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
            dA.append(attn - attn0)
            dT.append(it["target_vec"] - t0)
            tags.append((base, seed, it["level_id"]))
    return np.stack(dT).astype(np.float32), np.stack(dA).astype(np.float32), tags


def pca_targets(dT: np.ndarray, var_threshold: float = 0.99):
    """Standardize then PCA. Return (scores, components, explained_var_ratio,
    mean, scale). Keep top K components covering var_threshold of variance."""
    mean = dT.mean(axis=0)
    # Small jitter to avoid zero-var columns; we already dropped dead channels.
    std = dT.std(axis=0) + 1e-6
    Z = (dT - mean) / std
    _, S, Vt = np.linalg.svd(Z, full_matrices=False)
    var = S ** 2 / (len(dT) - 1)
    var_ratio = var / var.sum()
    cum = np.cumsum(var_ratio)
    K = int(np.searchsorted(cum, var_threshold) + 1)
    scores = Z @ Vt[:K].T       # (N, K)
    return scores, Vt[:K], var_ratio[:K], mean, std


def ridge_multi(X: np.ndarray, Y: np.ndarray, lam_rel: float = 1e-3) -> np.ndarray:
    """Ridge: X (N, K), Y (N, P) → β (K, P)."""
    XtX = X.T @ X
    lam = lam_rel * XtX.trace() / X.shape[1]
    inv = np.linalg.inv(XtX + lam * np.eye(X.shape[1]))
    return inv @ (X.T @ Y)      # (K, P)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (BOOT / "blendshapes.json").open() as f:
        bs_scores = json.load(f)

    pkls = sorted((BOOT / "smile").glob("*.pkl"))
    keys = get_shared_keys(pkls[:5])
    K_keys = len(keys)
    double_mask = np.array([k[0].startswith("double") for k in keys])
    print(f"[pca] keys: total={K_keys}  double={int(double_mask.sum())}  single={int((~double_mask).sum())}")

    dT, dA, _ = load_bootstrap_data("smile", SMILE_TARGETS, bs_scores, keys)
    N = len(dT)
    D = dA.shape[2]
    print(f"[pca] N={N}  K_chan={dT.shape[1]}  K_keys={K_keys}  D={D}")

    # PCA on target matrix
    scores, comps, var_ratio, mean_t, std_t = pca_targets(dT, var_threshold=0.999)
    Kpc = scores.shape[1]
    print(f"[pca] kept {Kpc} PCs  explained_var_ratio={[f'{v:.3f}' for v in var_ratio]}")
    print(f"[pca] PC loadings (standardized blendshape space):")
    for k in range(Kpc):
        row = "  PC" + f"{k+1}: " + "  ".join(
            f"{t[:8]}={comps[k, j]:+.3f}" for j, t in enumerate(SMILE_TARGETS))
        print(row)

    # Fit ridge in PC-target space: scores (N, Kpc) → dA.reshape(N, K_keys*D)
    Y_flat = dA.reshape(N, K_keys * D)
    beta_flat = ridge_multi(scores, Y_flat)    # (Kpc, K_keys*D)
    beta = beta_flat.reshape(Kpc, K_keys, D)

    # Overall R²
    pred = scores @ beta_flat
    rss = ((Y_flat - pred) ** 2).sum()
    tss = ((Y_flat - Y_flat.mean(axis=0, keepdims=True)) ** 2).sum()
    r2 = 1.0 - rss / tss
    print(f"[pca] R² (full keys): {r2:.3f}")

    # R² restricted to double blocks only
    Y_dbl = dA[:, double_mask, :].reshape(N, -1)
    pred_dbl = scores @ ridge_multi(scores, Y_dbl)
    rss_d = ((Y_dbl - pred_dbl) ** 2).sum()
    tss_d = ((Y_dbl - Y_dbl.mean(axis=0, keepdims=True)) ** 2).sum()
    print(f"[pca] R² (double blocks only): {(1 - rss_d / tss_d):.3f}")

    # Compare each PC's direction to pair-averaging δ, restricted to double blocks
    print(f"\n[pca] per-PC vs pair-δ (double-blocks-only, all 6 bases):")
    header = "  " + f"{'base':>22}"
    header += "  " + "  ".join(f"PC{k+1}".rjust(6) for k in range(Kpc))
    print(header + "  " + "best".rjust(6))
    summary = {}
    pair_dir = CROSS / "smile" / "measurement"
    for p in sorted(pair_dir.glob("*_meas.pkl")):
        base_ref = p.stem.replace("_meas", "")
        loaded = load_pair_delta(p, keys)
        if loaded is None:
            continue
        delta, valid = loaded
        valid_dbl = [v for v in valid if double_mask[v]]
        valid_dbl_local = [i for i, v in enumerate(valid) if double_mask[v]]
        if not valid_dbl:
            continue
        delta_dbl = delta[valid_dbl_local]
        row = "  " + f"{base_ref:>22}"
        per_pc = {}
        for k in range(Kpc):
            beta_v = beta[k, valid_dbl]      # (K_valid_dbl, D)
            cos_list = []
            for j in range(len(valid_dbl)):
                num = float(beta_v[j] @ delta_dbl[j])
                den = float(np.linalg.norm(beta_v[j]) * np.linalg.norm(delta_dbl[j]) + 1e-12)
                cos_list.append(num / den)
            a = np.array(cos_list)
            per_pc[f"PC{k+1}"] = {
                "abs_p95": float(np.percentile(np.abs(a), 95)),
                "abs_mean": float(np.abs(a).mean()),
                "mean": float(a.mean()),
            }
            row += "  " + f"{per_pc[f'PC{k+1}']['abs_p95']:+.3f}"
        best_pc = max(per_pc, key=lambda k: per_pc[k]["abs_p95"])
        row += "  " + f"{per_pc[best_pc]['abs_p95']:+.3f}({best_pc})"
        print(row)
        summary[base_ref] = per_pc

    # Scalar-target baseline (for direct comparison) — restricted to double blocks
    scalar_target = scores[:, 0]    # first PC as "overall smile intensity"
    beta_scalar = ridge_multi(scalar_target[:, None], Y_flat).reshape(1, K_keys, D)[0]
    print(f"\n[pca] scalar PC1-target ridge vs pair-δ (double-blocks-only):")
    for p in sorted(pair_dir.glob("*_meas.pkl")):
        base_ref = p.stem.replace("_meas", "")
        loaded = load_pair_delta(p, keys)
        if loaded is None:
            continue
        delta, valid = loaded
        valid_dbl = [v for v in valid if double_mask[v]]
        valid_dbl_local = [i for i, v in enumerate(valid) if double_mask[v]]
        if not valid_dbl:
            continue
        delta_dbl = delta[valid_dbl_local]
        beta_v = beta_scalar[valid_dbl]
        cos_list = []
        for j in range(len(valid_dbl)):
            num = float(beta_v[j] @ delta_dbl[j])
            den = float(np.linalg.norm(beta_v[j]) * np.linalg.norm(delta_dbl[j]) + 1e-12)
            cos_list.append(num / den)
        a = np.array(cos_list)
        print(f"  {base_ref:>22}  |cos|_p95={np.percentile(np.abs(a), 95):.3f}  "
              f"|cos|_mean={np.abs(a).mean():.3f}  cos_mean={a.mean():+.3f}")

    out = {
        "N": N, "K_keys": K_keys, "D": D,
        "smile_targets": SMILE_TARGETS,
        "pc_explained_var": var_ratio.tolist(),
        "pc_loadings": comps.tolist(),
        "target_mean": mean_t.tolist(),
        "target_std": std_t.tolist(),
        "R2_full": float(r2),
        "comparisons_to_pair_double": summary,
    }
    with (OUT_DIR / "pca_ridge_summary.json").open("w") as f:
        json.dump(out, f, indent=2)
    with (OUT_DIR / "pca_ridge_beta.pkl").open("wb") as f:
        pickle.dump({"beta": beta, "beta_scalar": beta_scalar, "keys": keys,
                     "double_mask": double_mask, "pc_loadings": comps,
                     "target_mean": mean_t, "target_std": std_t,
                     "smile_targets": SMILE_TARGETS}, f)
    print(f"\nsaved → {OUT_DIR}")


if __name__ == "__main__":
    main()
