"""Solver C feasibility check on FFHQ (squint axis) — v2 with
contamination pre-filters.

v1 of this script (solver_c_squint_feasibility_ffhq.py) selected
pairs from all FFHQ rows with non-null ArcFace + demographics. Visual
inspection of the resulting top-1000 surfaced two failure modes:

1. ~20% of pairs included a face wearing glasses (often sunglasses);
   covered eyes confuse MediaPipe's eye region and produce spurious
   bs_eyeSquint readings.
2. ~5–10% of "high-squint" rows were actually mid-blink or sleepy
   frames — eye closure mis-flagged as squint.

v2 applies hard pre-filters to the FFHQ row pool *before* pair
selection, so the top-N is chosen from a clean candidate set rather
than retroactively pruned. Filters:

- sg_glasses_margin <= -0.01    (excludes any glasses; baseline
  median is ~-0.04, so this is just below "no glasses" baseline)
- sg_eyes_closed_margin <= 0    (excludes closed-eye frames)
- bs_eyeBlink(L+R) <= 0.55      (excludes mid-blink frames; real
  squint compresses lids but does not fully close them)

Budget bumped from 1000 → 2000 to give downstream
build_squint_pair_manifest.py room after its |Δsmile| ≤ 0.30 filter.
Output dir is sibling: output/solver_c_ffhq_v2/.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
REVERSE_INDEX = REPO / "output" / "reverse_index" / "reverse_index.parquet"
OUT_DIR = REPO / "output" / "solver_c_ffhq_v2"
N_BUDGET = 2000
PER_RENDER_CAP = 4
SIGMA_SWEEP = (0.01, 0.05, 0.25)
SIGMA_THETA_FRAC = 0.05
ARC_PCA_DIMS = 16
TIKHONOV_LAMBDA = 1e-3
KNN_K = 50
MIN_CELL_SIZE = 30

GLASSES_MARGIN_MAX = -0.01
EYES_CLOSED_MARGIN_MAX = 0.0
BLINK_MAX = 0.55


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
        "sg_glasses_margin",
        "sg_eyes_closed_margin",
    ]
    df = pd.read_parquet(REVERSE_INDEX, columns=cols)
    df = df[df["source"] == "ffhq"].drop(columns=["source"])
    before = len(df)
    df = df[df["arcface_fp32"].notna()]
    df = df[df["ff_race"] != ""]
    df = df[df["ff_gender"] != ""]
    df = df[df["ff_age_bin"] != ""]
    after_demog = len(df)
    print(f"[loaded] {after_demog} rows after demographics+arcface (was {before})")

    blink_total = df["bs_eyeBlinkLeft"] + df["bs_eyeBlinkRight"]
    n_glasses = int((df["sg_glasses_margin"] > GLASSES_MARGIN_MAX).sum())
    n_eyes_closed = int((df["sg_eyes_closed_margin"] > EYES_CLOSED_MARGIN_MAX).sum())
    n_blinking = int((blink_total > BLINK_MAX).sum())
    df = df[df["sg_glasses_margin"] <= GLASSES_MARGIN_MAX]
    df = df[df["sg_eyes_closed_margin"] <= EYES_CLOSED_MARGIN_MAX]
    df = df[(df["bs_eyeBlinkLeft"] + df["bs_eyeBlinkRight"]) <= BLINK_MAX]
    df = df.drop(columns=["sg_glasses_margin", "sg_eyes_closed_margin"])
    df = df.reset_index(drop=True)
    df["row_id"] = df.index
    print(f"[contam-filter] dropped: glasses={n_glasses}, eyes_closed={n_eyes_closed}, blink>{BLINK_MAX}={n_blinking}")
    print(f"[loaded] {len(df)} rows after contamination filters")
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
    W: np.ndarray, sigma_theta_sq: float, k: int,
) -> pd.DataFrame:
    """Within each (race, gender, age_bin) cell with ≥MIN_CELL_SIZE rows,
    take top-k ArcFace cosine neighbors per row and emit their pair J."""
    out_rows = []
    cells_used = cells_skipped = 0
    for cell_key, cell in df.groupby(["ff_race", "ff_gender", "ff_age_bin"], sort=False):
        idx = cell["row_id"].to_numpy()
        if len(idx) < MIN_CELL_SIZE:
            cells_skipped += 1
            continue
        cells_used += 1
        a = arc[idx]
        sim = a @ a.T
        np.fill_diagonal(sim, -np.inf)
        kk = min(k, len(idx) - 1)
        top = np.argpartition(-sim, kk - 1, axis=1)[:, :kk]
        rows = np.repeat(np.arange(len(idx)), kk)
        cols = top.reshape(-1)
        i_loc = np.minimum(rows, cols)
        j_loc = np.maximum(rows, cols)
        pair_key = i_loc.astype(np.int64) * len(idx) + j_loc.astype(np.int64)
        _, uniq = np.unique(pair_key, return_index=True)
        i_loc, j_loc = i_loc[uniq], j_loc[uniq]
        ii, jj = idx[i_loc], idx[j_loc]
        dtheta = theta[ii] - theta[jj]
        deta = eta[ii] - eta[jj]
        deta_sq = np.einsum("ij,jk,ik->i", deta, W, deta)
        J = (dtheta * dtheta) / (sigma_theta_sq + deta_sq)
        out_rows.append(pd.DataFrame({
            "i": ii, "j": jj,
            "ff_race": cell_key[0], "ff_gender": cell_key[1], "ff_age_bin": cell_key[2],
            "dtheta": dtheta, "abs_dtheta": np.abs(dtheta),
            "deta_sq": deta_sq, "J": J,
        }))
    print(f"[cells] used {cells_used}, skipped {cells_skipped} (size < {MIN_CELL_SIZE})")
    return pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()


def greedy_select(pairs: pd.DataFrame, budget: int, per_render_cap: int) -> np.ndarray:
    order = np.argsort(-pairs["J"].to_numpy())
    chosen = np.zeros(len(pairs), dtype=bool)
    counts: dict[int, int] = {}
    n = 0
    i_arr = pairs["i"].to_numpy()
    j_arr = pairs["j"].to_numpy()
    for idx in order:
        i, j = int(i_arr[idx]), int(j_arr[idx])
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

    # Persist sha256 mapping alongside pairs so the manifest builder
    # doesn't have to replicate the contamination filters.
    pairs.attrs["budget"] = N_BUDGET
    df_sha = df[["image_sha256"]].copy()
    df_sha["row_id"] = df_sha.index
    df_sha.to_parquet(OUT_DIR / "ffhq_squint_v2_index.parquet")

    var_ratio, eff_rank = eta_pca_diversity(eta, selected)
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

    pairs.to_parquet(OUT_DIR / "ffhq_squint_pairs.parquet")

    report = [
        "# Solver C v2 — FFHQ squint feasibility report (with contamination filters)",
        "",
        f"- FFHQ rows post all filters: {len(df)}",
        f"- pre-filters applied: glasses_margin ≤ {GLASSES_MARGIN_MAX}, eyes_closed_margin ≤ {EYES_CLOSED_MARGIN_MAX}, blink ≤ {BLINK_MAX}",
        f"- candidate pairs (top-{KNN_K} ArcFace neighbors within cell): {len(pairs):,}",
        f"- η dimensions: {eta.shape[1]} = 5 blendshape scalars + {ARC_PCA_DIMS} ArcFace PCs",
        f"- W: full Σ_η^−1 Tikhonov λ={TIKHONOV_LAMBDA}",
        f"- σ_θ² primary: {sigma_theta_sq:.4g} (5% of var θ = {var_theta:.4g})",
        f"- budget: {N_BUDGET} (bumped from 1000)",
        "",
        "## J distribution",
        "",
        f"- p50 / p90 / p99: {p50:.4f} / {p90:.4f} / {p99:.4f}",
        f"- pairs above p90: {n_above_p90:,}",
        f"- pairs above p99: {int((pairs['J'] > p99).sum()):,}",
        "",
        f"## Greedy selection (budget {N_BUDGET}, per-render cap {PER_RENDER_CAP})",
        "",
        f"- selected: {int(chosen.sum())}",
        f"- median |Δθ| selected vs all: {sel_dtheta.median():.3f} vs {all_dtheta.median():.3f}",
        f"- median √(Δη^T W Δη) selected vs all: {sel_deta.median():.3f} vs {all_deta.median():.3f}",
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
        for n_, c in contrib:
            report.append(f"  - {n_}: {c:.3f}")
        report.append("")

    report += [
        "## Pass criteria",
        "",
        f"- ≥ 200 pairs above p90: **{'PASS' if pass_count else 'FAIL'}** ({n_above_p90})",
        f"- selected Δη eff. rank ≥ 8: **{'PASS' if pass_diversity else 'FAIL'}** ({eff_rank})",
        "",
        "## Verdict",
        "",
        f"- Data feasibility on FFHQ v2: **{'PASS' if (pass_count and pass_diversity) else 'FAIL'}**",
    ]

    (OUT_DIR / "ffhq_squint_feasibility_report.md").write_text("\n".join(report))
    print((OUT_DIR / "ffhq_squint_feasibility_report.md").read_text())


if __name__ == "__main__":
    main()
