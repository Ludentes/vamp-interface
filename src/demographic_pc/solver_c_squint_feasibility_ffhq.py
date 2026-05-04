"""Solver C feasibility check on FFHQ (squint axis).

Companion to solver_c_squint_feasibility.py (v1.1 Flux). Asks: does
FFHQ's broader photographic distribution contain squint-isolated pairs
that the Flux corpus does not?

Differences from the Flux v1.1 script:

- Source: FFHQ rows from output/reverse_index/reverse_index.parquet,
  arcface non-null (~26k of 70k).
- Cell key: (ff_race, ff_gender, ff_age_bin) — FairFace categories.
- η: 5 blendshape scalars (smile, gaze, blink, brow, cheek) + 16
  ArcFace PCs. No base one-hots (no scene preset on FFHQ); no
  id_drift (no anchor; FFHQ images are independent identities).
- Pair restriction: top-K=50 ArcFace cosine neighbors within cell per
  row. Brute-force matmul, sub-second per cell at this scale.
- Same metric: full Σ_η^{−1} Mahalanobis with Tikhonov reg, σ_θ²
  sweep, Δη eigenvalue diversity check.

Exits with PASS if (count ≥ 200 above p90) ∧ (eff-rank ≥ 8) ∧
(σ_θ² Spearman ≥ 0.90). The eff-rank threshold is raised vs Flux
(was 4) because FFHQ's η is smaller (21 vs 31 dims) and PC0 should
not dominate as severely if the bundle is corpus-distribution-driven
rather than universal.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
REVERSE_INDEX = REPO / "output" / "reverse_index" / "reverse_index.parquet"
OUT_DIR = REPO / "output" / "solver_c_ffhq"
N_BUDGET = 1000
PER_RENDER_CAP = 4
SIGMA_SWEEP = (0.01, 0.05, 0.25)
SIGMA_THETA_FRAC = 0.05
ARC_PCA_DIMS = 16
TIKHONOV_LAMBDA = 1e-3
KNN_K = 50
MIN_CELL_SIZE = 30


def load_ffhq() -> pd.DataFrame:
    cols = [
        "image_sha256",
        "source",
        "arcface_fp32",
        "ff_race",
        "ff_gender",
        "ff_age_bin",
        "bs_eyeSquintLeft",
        "bs_eyeSquintRight",
        "bs_mouthSmileLeft",
        "bs_mouthSmileRight",
        "bs_eyeLookInLeft",
        "bs_eyeLookInRight",
        "bs_eyeLookOutLeft",
        "bs_eyeLookOutRight",
        "bs_eyeLookUpLeft",
        "bs_eyeLookUpRight",
        "bs_eyeLookDownLeft",
        "bs_eyeLookDownRight",
        "bs_eyeBlinkLeft",
        "bs_eyeBlinkRight",
        "bs_browDownLeft",
        "bs_browDownRight",
        "bs_browInnerUp",
        "bs_browOuterUpLeft",
        "bs_browOuterUpRight",
        "bs_cheekSquintLeft",
        "bs_cheekSquintRight",
    ]
    df = pd.read_parquet(REVERSE_INDEX, columns=cols)
    df = df[df["source"] == "ffhq"].drop(columns=["source"])
    before = len(df)
    df = df[df["arcface_fp32"].notna()]
    df = df[df["ff_race"] != ""]
    df = df[df["ff_gender"] != ""]
    df = df[df["ff_age_bin"] != ""]
    df = df.reset_index(drop=True)
    df["row_id"] = df.index
    print(f"[loaded] {len(df)} rows after filters (was {before})")
    return df


def arcface_matrix(df: pd.DataFrame) -> np.ndarray:
    arc = np.stack([np.asarray(v, dtype=np.float32) for v in df["arcface_fp32"]])
    norms = np.linalg.norm(arc, axis=1, keepdims=True)
    arc = arc / np.maximum(norms, 1e-8)
    return arc


def arcface_pca(arc: np.ndarray, k: int) -> np.ndarray:
    centered = arc - arc.mean(axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    pcs = centered @ Vt[:k].T
    var_ret = (S[:k] ** 2).sum() / (S ** 2).sum()
    print(f"[arc-pca] k={k}, retained variance fraction: {var_ret:.3f}")
    return pcs.astype(np.float64)


def build_theta_eta(df: pd.DataFrame, arc_pcs: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    theta = (df["bs_eyeSquintLeft"] + df["bs_eyeSquintRight"]).to_numpy()

    smile = (df["bs_mouthSmileLeft"] + df["bs_mouthSmileRight"]).to_numpy()
    gaze = (
        df["bs_eyeLookInLeft"].abs() + df["bs_eyeLookInRight"].abs()
        + df["bs_eyeLookOutLeft"].abs() + df["bs_eyeLookOutRight"].abs()
        + df["bs_eyeLookUpLeft"].abs() + df["bs_eyeLookUpRight"].abs()
        + df["bs_eyeLookDownLeft"].abs() + df["bs_eyeLookDownRight"].abs()
    ).to_numpy()
    blink = (df["bs_eyeBlinkLeft"] + df["bs_eyeBlinkRight"]).to_numpy()
    brow = (
        df["bs_browDownLeft"] + df["bs_browDownRight"]
        + df["bs_browInnerUp"]
        + df["bs_browOuterUpLeft"] + df["bs_browOuterUpRight"]
    ).to_numpy()
    cheek = (df["bs_cheekSquintLeft"] + df["bs_cheekSquintRight"]).to_numpy()

    cols = [("smile", smile), ("gaze", gaze), ("blink", blink), ("brow", brow), ("cheek", cheek)]
    eta_cols, names = [], []
    for n, v in cols:
        sd = v.std()
        eta_cols.append(v / sd if sd > 0 else v)
        names.append(n)

    arc_sd = arc_pcs.std(axis=0)
    arc_z = arc_pcs / np.where(arc_sd > 0, arc_sd, 1.0)
    for k in range(arc_z.shape[1]):
        eta_cols.append(arc_z[:, k])
        names.append(f"arc_pc{k}")

    return theta, np.stack(eta_cols, axis=1), names


def full_mahalanobis_W(eta: np.ndarray, lam: float) -> np.ndarray:
    cov = np.cov(eta, rowvar=False)
    d = cov.shape[0]
    return np.linalg.inv(cov + lam * np.eye(d))


def knn_pairs_per_cell(
    df: pd.DataFrame, arc: np.ndarray, theta: np.ndarray, eta: np.ndarray,
    W: np.ndarray, sigma_theta_sq: float, k: int
) -> pd.DataFrame:
    """Within each (race, gender, age_bin) cell with ≥MIN_CELL_SIZE rows,
    take top-k ArcFace neighbors per row, score by J."""
    pieces = []
    cells_used = 0
    cells_skipped = 0
    for cell_key, cell in df.groupby(["ff_race", "ff_gender", "ff_age_bin"], sort=False):
        idx = cell["row_id"].to_numpy()
        if len(idx) < MIN_CELL_SIZE:
            cells_skipped += 1
            continue
        cells_used += 1
        arc_cell = arc[idx]                            # (m, 512)
        sim = arc_cell @ arc_cell.T                    # (m, m) cosines (L2-normed)
        np.fill_diagonal(sim, -np.inf)
        kk = min(k, len(idx) - 1)
        # top-k indices per row
        nbrs = np.argpartition(-sim, kk - 1, axis=1)[:, :kk]
        rows = np.repeat(np.arange(len(idx)), kk)
        cols = nbrs.flatten()
        # dedup unordered pairs (i < j)
        i_loc = np.minimum(rows, cols)
        j_loc = np.maximum(rows, cols)
        pair_key = i_loc.astype(np.int64) * len(idx) + j_loc.astype(np.int64)
        _, uniq = np.unique(pair_key, return_index=True)
        i_loc, j_loc = i_loc[uniq], j_loc[uniq]
        ii, jj = idx[i_loc], idx[j_loc]
        dtheta = theta[ii] - theta[jj]
        deta = eta[ii] - eta[jj]
        deta_W = deta @ W
        deta_sq = np.einsum("pi,pi->p", deta_W, deta)
        J = (dtheta * dtheta) / (sigma_theta_sq + deta_sq)
        pieces.append(pd.DataFrame({
            "i": ii, "j": jj,
            "ff_race": cell_key[0], "ff_gender": cell_key[1], "ff_age_bin": cell_key[2],
            "dtheta": dtheta, "abs_dtheta": np.abs(dtheta),
            "deta_sq": deta_sq, "J": J,
        }))
    print(f"[cells] used {cells_used}, skipped {cells_skipped} (size < {MIN_CELL_SIZE})")
    return pd.concat(pieces, ignore_index=True)


def greedy_select(pairs: pd.DataFrame, budget: int, per_render_cap: int) -> np.ndarray:
    order = np.argsort(-pairs["J"].to_numpy())
    chosen = np.zeros(len(pairs), dtype=bool)
    counts: dict[int, int] = {}
    n = 0
    for idx in order:
        i = int(pairs["i"].iat[idx])
        j = int(pairs["j"].iat[idx])
        if counts.get(i, 0) >= per_render_cap or counts.get(j, 0) >= per_render_cap:
            continue
        chosen[idx] = True
        counts[i] = counts.get(i, 0) + 1
        counts[j] = counts.get(j, 0) + 1
        n += 1
        if n >= budget:
            break
    return chosen


def eta_pca_diversity(eta: np.ndarray, sel: pd.DataFrame) -> tuple[np.ndarray, int]:
    if len(sel) == 0:
        return np.array([]), 0
    ii, jj = sel["i"].to_numpy(), sel["j"].to_numpy()
    deta = eta[ii] - eta[jj]
    cov = np.cov(deta, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 1e-12][::-1]
    if eigvals.size == 0:
        return eigvals, 0
    return eigvals / eigvals.sum(), int(round((eigvals.sum() ** 2) / (eigvals ** 2).sum()))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_ffhq()
    arc = arcface_matrix(df)
    arc_pcs = arcface_pca(arc, ARC_PCA_DIMS)
    theta, eta, names = build_theta_eta(df, arc_pcs)
    var_theta = float(theta.var())
    sigma_theta_sq = SIGMA_THETA_FRAC * var_theta

    W = full_mahalanobis_W(eta, TIKHONOV_LAMBDA)
    print(f"[W] full Σ_η^-1, η dim={eta.shape[1]}, cond≈{np.linalg.cond(W):.2e}")

    pairs = knn_pairs_per_cell(df, arc, theta, eta, W, sigma_theta_sq, KNN_K)
    print(f"[paired] {len(pairs):,} candidate pairs")

    p50 = float(np.percentile(pairs["J"], 50))
    p90 = float(np.percentile(pairs["J"], 90))
    p99 = float(np.percentile(pairs["J"], 99))
    n_above_p90 = int((pairs["J"] > p90).sum())

    chosen = greedy_select(pairs, N_BUDGET, PER_RENDER_CAP)
    pairs["selected"] = chosen
    selected = pairs[chosen].sort_values("J", ascending=False).reset_index(drop=True)

    # σ sweep
    from scipy.stats import spearmanr
    sweep = {}
    for frac in SIGMA_SWEEP:
        s2 = frac * var_theta
        denom = s2 + pairs["deta_sq"].to_numpy()
        Js = (pairs["dtheta"].to_numpy() ** 2) / denom
        sweep[frac] = Js
    sweep_rows = []
    for a in SIGMA_SWEEP:
        for b in SIGMA_SWEEP:
            r, _ = spearmanr(sweep[a], sweep[b])
            sweep_rows.append({"sigma_a": a, "sigma_b": b, "spearman": float(r)})
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_pivot = sweep_df.pivot(index="sigma_a", columns="sigma_b", values="spearman")
    min_off_diag = float(sweep_pivot.where(~np.eye(len(sweep_pivot), dtype=bool)).min().min())
    sweep_df.to_csv(OUT_DIR / "ffhq_squint_sigma_sweep.csv", index=False)

    var_ratio, eff_rank = eta_pca_diversity(eta, selected)

    # decompose top-3 selected Δη eigenvectors in eta_names basis
    deta_sel = eta[selected["i"].to_numpy()] - eta[selected["j"].to_numpy()]
    cov_sel = np.cov(deta_sel, rowvar=False)
    vals, vecs = np.linalg.eigh(cov_sel)
    order = np.argsort(-vals)
    vals, vecs = vals[order], vecs[:, order]
    pc_decomp = []
    for k in range(min(3, len(vals))):
        v = vecs[:, k] ** 2
        contrib = sorted(zip(names, v), key=lambda x: -x[1])[:6]
        pc_decomp.append((vals[k] / vals.sum(), contrib))

    sel_dtheta = selected["abs_dtheta"]
    sel_deta = np.sqrt(selected["deta_sq"])
    all_dtheta = pairs["abs_dtheta"]
    all_deta = np.sqrt(pairs["deta_sq"])

    pass_count = n_above_p90 >= 200
    pass_diversity = eff_rank >= 8
    pass_sigma = min_off_diag >= 0.90

    pairs.to_parquet(OUT_DIR / "ffhq_squint_pairs.parquet")

    report = [
        "# Solver C — FFHQ squint feasibility report",
        "",
        f"- FFHQ rows post-filter (arcface + non-empty demographics): {len(df)}",
        f"- candidate pairs (top-{KNN_K} ArcFace neighbors within cell): {len(pairs):,}",
        f"- η dimensions: {eta.shape[1]} = 5 blendshape scalars + {ARC_PCA_DIMS} ArcFace PCs",
        f"- W: full Σ_η^−1 Tikhonov λ={TIKHONOV_LAMBDA}",
        f"- σ_θ² primary: {sigma_theta_sq:.4g} (5% of var θ = {var_theta:.4g})",
        "",
        "## J distribution",
        "",
        f"- p50 / p90 / p99: {p50:.4f} / {p90:.4f} / {p99:.4f}",
        f"- pairs above p90: {n_above_p90:,}",
        f"- pairs above p99: {int((pairs['J'] > p99).sum()):,}",
        "",
        "## Greedy selection (budget 1000, per-render cap 4)",
        "",
        f"- selected: {int(chosen.sum())}",
        f"- median |Δθ| selected vs all: {sel_dtheta.median():.3f} vs {all_dtheta.median():.3f}",
        f"- median √(Δη^T W Δη) selected vs all: {sel_deta.median():.3f} vs {all_deta.median():.3f}",
        "",
        "## σ_θ² sweep (Spearman of J across sweep)",
        "",
        "```",
        sweep_pivot.to_string(float_format=lambda x: f"{x:.4f}"),
        "```",
        f"- min off-diagonal: {min_off_diag:.4f}",
        "",
        "## Δη diversity of selected set",
        "",
        f"- effective rank (participation ratio): {eff_rank} / {eta.shape[1]}",
        f"- top-3 var fractions: {', '.join(f'{v:.3f}' for v in var_ratio[:3])}",
        "",
        "### Top-3 Δη eigenvector decomposition",
        "",
    ]
    for k, (frac, contrib) in enumerate(pc_decomp):
        report.append(f"**PC{k} ({frac:.1%}):**")
        for n, c in contrib:
            report.append(f"  - {n}: {c:.3f}")
        report.append("")

    report += [
        "## Pass criteria",
        "",
        f"- ≥ 200 pairs above p90: **{'PASS' if pass_count else 'FAIL'}** ({n_above_p90})",
        f"- selected Δη eff. rank ≥ 8: **{'PASS' if pass_diversity else 'FAIL'}** ({eff_rank})",
        f"- σ_θ² rank stability ≥ 0.90: **{'PASS' if pass_sigma else 'FAIL'}** ({min_off_diag:.3f})",
        "",
        "## Verdict",
        "",
        f"- Data feasibility on FFHQ: **{'PASS' if (pass_count and pass_diversity and pass_sigma) else 'FAIL'}**",
        "",
    ]
    (OUT_DIR / "ffhq_squint_feasibility_report.md").write_text("\n".join(report))
    print((OUT_DIR / "ffhq_squint_feasibility_report.md").read_text())


if __name__ == "__main__":
    main()
