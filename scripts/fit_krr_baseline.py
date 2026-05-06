"""Closed-form Nyström-KRR baseline for the ARKit→PersonaLive distill.

Solves a fixed-design Gaussian-RBF kernel ridge regression
    β = (Φᵀ Φ + λ I)⁻¹ Φᵀ Y
with the kernel-feature design matrix
    Φ[i, j] = exp( -‖b_i - landmark_j‖² / (2σ²) )

Sweeps λ on a log grid, picks λ minimising holdout `ratio_mean`. Metrics
match `arkit_bridge.eval`: per-frame MSE / per-frame Var(target), plus
per-cell R² over the held-out set.

Landmark modes:
  nmf      — fit NMF(n_components=k) on b_expr; landmarks = H[j] * w_j_p75,
             i.e. each atom row scaled by its 75-percentile loading so it
             lands *inside* the data cloud at typical firing intensity
             (raw H rows are direction patterns at unit-loading and sit
             far outside the cloud, giving a degenerate kernel — see
             docs/research/2026-05-06-arkit-bridge-function-class-pareto.md
             open question 1, raised by code-reviewer).
  nmf_raw  — diagnostic: atoms as-is (H[j]); exposes the kernel-saturation
             pathology so we can confirm the active-mode fix is the cause.
  kmeans   — k-means cluster centers in b_expr space (textbook landmarks).
  random   — k uniformly-random training points.
  full     — every training point is a landmark (full-Gram KRR).

Numerical: PtP and PtY computed in float64 once, then PtP+λI is solved
per λ via Cholesky (works regardless of M). For full-Gram (M=N≈16k) the
float64 cast prevents condition-number-driven Cholesky failures.

Outputs (under --out_dir):
  scorecard.json   one entry per λ; best summary; landmark diagnostics
  beta_best.npy    readout β for the λ that won
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_pairs(pairs_dir: Path):
    paths = sorted(pairs_dir.glob("*frame_*.pkl"))
    if not paths:
        raise SystemExit(f"no pkls under {pairs_dir}")
    bs, ms = [], []
    for p in paths:
        with open(p, "rb") as f:
            d = pickle.load(f)
        bs.append(np.asarray(d["b_expr"], dtype=np.float32))
        # m_f stored as (1, 32, 16); reshape to (512,) — C-order matches
        # PyTorch's flatten(1) used in arkit_bridge.eval (verified).
        ms.append(np.asarray(d["m_f"], dtype=np.float32).reshape(-1))
    return np.stack(bs), np.stack(ms)


def median_pairwise_dist(x: np.ndarray, max_n: int = 2000,
                         seed: int = 0, block: int = 256) -> float:
    """Median ‖x_i - x_j‖ over a max_n random subsample, computed in
    row-blocks to avoid the n×n×d broadcast temporary."""
    rng = np.random.default_rng(seed)
    n = min(max_n, len(x))
    xs = x[rng.choice(len(x), n, replace=False)].astype(np.float32)
    sq_x = (xs * xs).sum(axis=1)
    pieces = []
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        cross = xs[i0:i1] @ xs.T                            # (B, n)
        sq = sq_x[i0:i1, None] + sq_x[None, :] - 2.0 * cross
        np.maximum(sq, 0.0, out=sq)
        for r in range(i1 - i0):
            j_start = i0 + r + 1
            if j_start < n:
                pieces.append(np.sqrt(sq[r, j_start:]))
    d = np.concatenate(pieces) if pieces else np.array([])
    d = d[d > 0]
    if d.size == 0 or not np.isfinite(d).any():
        raise SystemExit("degenerate corpus: no positive pairwise distances")
    return float(np.median(d))


def gaussian_kernel(b: np.ndarray, landmarks: np.ndarray, sigma: float,
                    block: int = 1024) -> np.ndarray:
    """Φ[i, j] = exp(-‖b_i - L_j‖² / (2σ²)) computed in row-blocks."""
    n = b.shape[0]
    m = landmarks.shape[0]
    out = np.empty((n, m), dtype=np.float32)
    inv_2s2 = 1.0 / (2.0 * sigma * sigma)
    L_sq = (landmarks * landmarks).sum(axis=1)              # (m,)
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        bb = b[i0:i1]
        b_sq = (bb * bb).sum(axis=1)[:, None]
        cross = bb @ landmarks.T
        sq = b_sq + L_sq[None, :] - 2.0 * cross
        np.maximum(sq, 0.0, out=sq)
        out[i0:i1] = np.exp(-sq * inv_2s2)
    return out


def fit_landmarks(b_train: np.ndarray, mode: str, k: int, seed: int):
    """Returns (landmarks, diag_dict) where diag_dict carries
    mode-specific diagnostics (NMF recon R², w_typ, etc.)."""
    if mode in ("nmf", "nmf_raw"):
        from sklearn.decomposition import NMF
        # b_expr clipped to ≥0 only for the NMF fit (rare ARKit jitter goes
        # slightly negative); kernel design matrix uses unclipped b_train.
        b_nn = np.maximum(b_train, 0.0)
        nmf = NMF(n_components=int(k), init="nndsvd", max_iter=500,
                  tol=1e-5, random_state=seed)
        W = nmf.fit_transform(b_nn)                         # (N, k)
        H = np.asarray(nmf.components_, dtype=np.float32)   # (k, 58)
        recon = W @ H
        ss_res = float(((b_nn - recon) ** 2).sum())
        ss_tot = float(((b_nn - b_nn.mean(axis=0)) ** 2).sum())
        recon_r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        if mode == "nmf":
            # Active-mode scaling: H[j] * percentile(W[:, j], 75) to put
            # each landmark inside the data cloud at typical activation.
            w_p75 = np.percentile(W, 75, axis=0).astype(np.float32)
            landmarks = (H * w_p75[:, None]).astype(np.float32)
            return landmarks, {"nmf_recon_r2": recon_r2,
                               "w_p75": w_p75.tolist()}
        return H.astype(np.float32), {"nmf_recon_r2": recon_r2}
    if mode == "kmeans":
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=int(k), random_state=seed, n_init=10)
        km.fit(b_train)
        return km.cluster_centers_.astype(np.float32), {}
    if mode == "random":
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(b_train), int(k), replace=False)
        return b_train[idx], {}
    if mode == "full":
        return b_train, {}
    raise ValueError(f"unknown landmark mode: {mode}")


def landmark_distance_diag(b: np.ndarray, landmarks: np.ndarray,
                           sigma: float, sample: int = 2048) -> dict:
    """Per-landmark median ‖b - landmark_j‖ / σ. Skipped for very large
    landmark sets where the M×N matrix wouldn't fit comfortably."""
    if landmarks.shape[0] > 256:
        return {"skipped_reason": "M>256 (full-Gram diagnostic skipped)"}
    rng = np.random.default_rng(0)
    n = min(sample, len(b))
    bs = b[rng.choice(len(b), n, replace=False)]
    bs_sq = (bs * bs).sum(axis=1)[:, None]
    L_sq = (landmarks * landmarks).sum(axis=1)
    sq = bs_sq + L_sq[None, :] - 2.0 * (bs @ landmarks.T)
    np.maximum(sq, 0.0, out=sq)
    med_per_landmark = np.sqrt(np.median(sq, axis=0))
    return {
        "median_b_to_landmark_over_sigma": (med_per_landmark / sigma).tolist(),
        "landmark_norms": np.sqrt(L_sq).tolist(),
    }


def metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """ratio_mean = mean over frames of (per-frame MSE / per-frame Var);
    per-cell R² across the held-out set (matches arkit_bridge.eval)."""
    diff_sq = ((pred - target) ** 2).mean(axis=1)
    var = target.var(axis=1).clip(min=1e-8)
    ratio = diff_sq / var
    ss_res = ((pred - target) ** 2).sum(axis=0)
    mean = target.mean(axis=0, keepdims=True)
    ss_tot = ((target - mean) ** 2).sum(axis=0)
    r2 = np.where(ss_tot > 1e-12,
                  1.0 - ss_res / np.maximum(ss_tot, 1e-12), np.nan)
    valid = ~np.isnan(r2)
    above = float((r2[valid] >= 0.7).sum() / max(int(valid.sum()), 1))
    return {
        "ratio_mean": float(ratio.mean()),
        "ratio_p95": float(np.quantile(ratio, 0.95)),
        "r2_median": float(np.nanmedian(r2)),
        "r2_above_0_7_fraction": above,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_dir", required=True)
    ap.add_argument("--holdout_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mode", default="nmf",
                    choices=["nmf", "nmf_raw", "kmeans", "random", "full"])
    ap.add_argument("--k", type=int, default=11,
                    help="landmark count (ignored for --mode full)")
    ap.add_argument("--sigma", type=float, default=None,
                    help="RBF bandwidth; default = median pairwise dist")
    ap.add_argument("--lambdas", type=str,
                    default="1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0",
                    help="comma-separated λ sweep")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lambdas = [float(x) for x in args.lambdas.split(",")]

    print(f"loading {args.pairs_dir}", flush=True)
    b_tr, y_tr = load_pairs(Path(args.pairs_dir))
    print(f"loading {args.holdout_dir}", flush=True)
    b_te, y_te = load_pairs(Path(args.holdout_dir))
    print(f"  train: b{b_tr.shape} y{y_tr.shape}; "
          f"holdout: b{b_te.shape} y{y_te.shape}", flush=True)

    sigma = (args.sigma if args.sigma is not None
             else median_pairwise_dist(b_tr))
    if not np.isfinite(sigma) or sigma < 1e-6:
        raise SystemExit(f"degenerate σ={sigma}")
    print(f"  σ = {sigma:.4f}{' (median)' if args.sigma is None else ''}",
          flush=True)

    print(f"fitting landmarks: mode={args.mode} k={args.k}", flush=True)
    landmarks, lm_diag = fit_landmarks(b_tr, args.mode, args.k, args.seed)
    print(f"  landmarks shape: {landmarks.shape}", flush=True)
    if lm_diag:
        for kk, vv in lm_diag.items():
            if isinstance(vv, list):
                v_arr = np.asarray(vv)
                print(f"  {kk}: min={v_arr.min():.3f} "
                      f"median={np.median(v_arr):.3f} max={v_arr.max():.3f}",
                      flush=True)
            else:
                print(f"  {kk}: {vv}", flush=True)

    dist_diag = landmark_distance_diag(b_tr, landmarks, sigma)
    if "median_b_to_landmark_over_sigma" in dist_diag:
        v = np.asarray(dist_diag["median_b_to_landmark_over_sigma"])
        print(f"  median ‖b-landmark‖/σ: min={v.min():.2f} "
              f"median={np.median(v):.2f} max={v.max():.2f}", flush=True)

    print("building train design matrix Φ_train", flush=True)
    Phi_tr = gaussian_kernel(b_tr, landmarks, sigma)
    print(f"  Φ_train shape: {Phi_tr.shape}; "
          f"Φ_tr stats: min={Phi_tr.min():.3e} mean={Phi_tr.mean():.3e} "
          f"max={Phi_tr.max():.3e}", flush=True)

    print("building holdout design matrix Φ_holdout", flush=True)
    Phi_te = gaussian_kernel(b_te, landmarks, sigma)

    # Float64 normal equations, computed once. PtP+λI is solved per λ.
    print("computing normal equations PtP, PtY (float64)...", flush=True)
    Phi_tr_d = Phi_tr.astype(np.float64)
    Y_d = y_tr.astype(np.float64)
    PtP_d = Phi_tr_d.T @ Phi_tr_d
    PtY_d = Phi_tr_d.T @ Y_d
    del Phi_tr_d, Y_d
    M = PtP_d.shape[0]

    rows = []
    best = None
    best_beta = None
    for lam in lambdas:
        A = PtP_d.copy()
        A[np.diag_indices(M)] += lam
        try:
            L = np.linalg.cholesky(A)
        except np.linalg.LinAlgError as e:
            print(f"  λ={lam:.0e}: Cholesky failed ({e}); skipping",
                  flush=True)
            continue
        z = np.linalg.solve(L, PtY_d)
        beta = np.linalg.solve(L.T, z)                      # float64
        pred_te = (Phi_te.astype(np.float64) @ beta).astype(np.float32)
        m = metrics(pred_te, y_te)
        m["lam"] = lam
        m["mode"] = args.mode
        m["k"] = int(args.k if args.mode != "full" else b_tr.shape[0])
        m["sigma"] = sigma
        rows.append(m)
        print(f"  λ={lam:.0e}: ratio={m['ratio_mean']:.5f} "
              f"r2_median={m['r2_median']:.3f} "
              f"r2_above_0_7={m['r2_above_0_7_fraction']:.3f}", flush=True)
        if best is None or m["ratio_mean"] < best["ratio_mean"]:
            best = m
            best_beta = beta.astype(np.float32)

    if best is None or best_beta is None:
        raise SystemExit("no λ solved successfully")

    np.save(out_dir / "beta_best.npy", best_beta)

    payload = {
        "mode": args.mode,
        "k": int(args.k if args.mode != "full" else b_tr.shape[0]),
        "sigma": sigma,
        "n_train": int(b_tr.shape[0]),
        "n_holdout": int(b_te.shape[0]),
        "landmark_diag": lm_diag,
        "distance_diag": dist_diag,
        "sweep": rows,
        "best": {"lam": best["lam"],
                 "ratio_mean": best["ratio_mean"],
                 "ratio_p95": best["ratio_p95"],
                 "r2_median": best["r2_median"],
                 "r2_above_0_7_fraction": best["r2_above_0_7_fraction"],
                 "beta_path": str(out_dir / "beta_best.npy")},
    }
    with open(out_dir / "scorecard.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {out_dir}/scorecard.json", flush=True)
    print(f"best λ={best['lam']:.0e} ratio={best['ratio_mean']:.5f} "
          f"r2_median={best['r2_median']:.3f}", flush=True)


if __name__ == "__main__":
    main()
