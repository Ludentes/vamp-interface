"""Solver C feasibility check on the eye-squint axis (v1.1).

Spec: docs/research/2026-04-28-solver-c-partial-fisher.md.

v1.1 changes vs v1 (per adversarial review 2026-04-28):
- η extended with ArcFace identity features (PCA-8 of arcface_fp32 from
  reverse_index.parquet) — captures identity/shape variance blendshapes
  miss; this is the load-bearing confound for "same identity ± squint".
- W = full Σ_η^{−1} with Tikhonov regularization (was diag(1/σ²_k));
  smile/cheek/brow correlate strongly and double-counted under diagonal.
- σ_θ² sensitivity sweep over {1%, 5%, 25%} of var(θ); reports Spearman
  rank correlation of J across sweeps.
- Replaces "cumulative ΣJ saturation" criterion with PCA-spectrum
  diversity check on selected Δη vectors — collapse to a 1D direction
  flags that selection is exploiting a single confound axis.
- Restricted to Flux corpus (source='flux_corpus_v3'); FFHQ rows omitted
  because they have no `base`/`identity_cos_to_base` and the trainer
  consumes Flux-renderable supervision.

Heuristic disclaimer: J(i,j) = (Δθ)² / (σ_θ² + Δη^T W Δη) is a
finite-difference proxy for partial Fisher information, not an
estimator. It scores pairs in the same `(z_neg − z_pos)/(2w)` algebra
the Concept Sliders image-pair trainer consumes — that alignment is
the operational justification, not asymptotic theory.

Outputs:
    output/solver_c/squint_pairs.parquet
    output/solver_c/squint_feasibility_report.md
    output/solver_c/squint_cumulative_info.png
    output/solver_c/squint_eta_pca_spectrum.png
    output/solver_c/squint_sigma_sweep.csv
    output/solver_c/squint_top50_pairs.png
    output/solver_c/squint_bottom50_pairs.png
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SAMPLE_INDEX = REPO / "models" / "blendshape_nmf" / "sample_index.parquet"
REVERSE_INDEX = REPO / "output" / "reverse_index" / "reverse_index.parquet"
OUT_DIR = REPO / "output" / "solver_c"
N_BUDGET = 1000
PER_RENDER_CAP = 4
SIGMA_THETA_FRAC = 0.05  # primary noise floor (sweep also covers 1% / 25%)
SIGMA_SWEEP = (0.01, 0.05, 0.25)
ARC_PCA_DIMS = 8
TIKHONOV_LAMBDA = 1e-3  # regularizer for Σ_η inversion


def load_corpus() -> pd.DataFrame:
    df = pd.read_parquet(SAMPLE_INDEX).reset_index(drop=True)

    rev = pd.read_parquet(
        REVERSE_INDEX, columns=["image_sha256", "source", "arcface_fp32"]
    )
    rev = rev[rev["source"] == "flux_corpus_v3"].drop(columns=["source"])
    df = df.merge(rev, on="image_sha256", how="left")
    before = len(df)
    df = df[df["arcface_fp32"].notna()].reset_index(drop=True)
    print(f"[loaded] dropped {before - len(df)} rows without arcface; {len(df)} retained")

    df["row_id"] = df.index
    return df


def arcface_pca(df: pd.DataFrame, k: int) -> np.ndarray:
    arc = np.stack([np.asarray(v, dtype=np.float32) for v in df["arcface_fp32"]])
    arc = arc - arc.mean(axis=0, keepdims=True)
    # SVD on (N, 512) — N=7477; cheap.
    _, S, Vt = np.linalg.svd(arc, full_matrices=False)
    pcs = arc @ Vt[:k].T  # (N, k)
    var = (S[:k] ** 2) / (len(arc) - 1)
    print(f"[arc-pca] retained variance fraction: {var.sum() / ((S ** 2).sum() / (len(arc) - 1)):.3f}")
    return pcs.astype(np.float64)


def build_theta_eta(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    theta = (df["bs_eyeSquintLeft"] + df["bs_eyeSquintRight"]).to_numpy()

    smile = (df["bs_mouthSmileLeft"] + df["bs_mouthSmileRight"]).to_numpy()
    gaze = (
        df["bs_eyeLookInLeft"].abs()
        + df["bs_eyeLookInRight"].abs()
        + df["bs_eyeLookOutLeft"].abs()
        + df["bs_eyeLookOutRight"].abs()
        + df["bs_eyeLookUpLeft"].abs()
        + df["bs_eyeLookUpRight"].abs()
        + df["bs_eyeLookDownLeft"].abs()
        + df["bs_eyeLookDownRight"].abs()
    ).to_numpy()
    blink = (df["bs_eyeBlinkLeft"] + df["bs_eyeBlinkRight"]).to_numpy()
    brow = (
        df["bs_browDownLeft"]
        + df["bs_browDownRight"]
        + df["bs_browInnerUp"]
        + df["bs_browOuterUpLeft"]
        + df["bs_browOuterUpRight"]
    ).to_numpy()
    cheek = (df["bs_cheekSquintLeft"] + df["bs_cheekSquintRight"]).to_numpy()
    id_drift = (1.0 - df["identity_cos_to_base"].fillna(1.0)).to_numpy()
    base_oh = pd.get_dummies(df["base"], prefix="base").to_numpy().astype(float)
    arc_pcs = arcface_pca(df, ARC_PCA_DIMS)

    scalar_cols = [
        ("smile", smile),
        ("gaze", gaze),
        ("blink", blink),
        ("brow", brow),
        ("cheek", cheek),
        ("id_drift", id_drift),
    ]
    eta_cols = []
    eta_names = []
    for name, vec in scalar_cols:
        sd = vec.std()
        eta_cols.append(vec / sd if sd > 0 else vec)
        eta_names.append(name)

    K = base_oh.shape[1]
    base_block = base_oh / max(math.sqrt(K), 1.0)
    for k in range(K):
        eta_cols.append(base_block[:, k])
        eta_names.append(f"base_{k}")

    arc_sd = arc_pcs.std(axis=0)
    arc_z = arc_pcs / np.where(arc_sd > 0, arc_sd, 1.0)
    for k in range(arc_z.shape[1]):
        eta_cols.append(arc_z[:, k])
        eta_names.append(f"arc_pc{k}")

    eta = np.stack(eta_cols, axis=1)
    return theta, eta, eta_names


def full_mahalanobis_W(eta: np.ndarray, lam: float) -> np.ndarray:
    """W = (Σ_η + λ·I)^{−1}; Tikhonov-regularized inverse covariance."""
    cov = np.cov(eta, rowvar=False)
    d = cov.shape[0]
    return np.linalg.inv(cov + lam * np.eye(d))


def enumerate_pairs(
    df: pd.DataFrame, theta: np.ndarray, eta: np.ndarray, W: np.ndarray, sigma_theta_sq: float
) -> pd.DataFrame:
    pieces = []
    for cell_key, cell in df.groupby(["ethnicity", "gender", "age"], sort=False):
        idx = cell["row_id"].to_numpy()
        if len(idx) < 2:
            continue
        i_grid, j_grid = np.triu_indices(len(idx), k=1)
        ii, jj = idx[i_grid], idx[j_grid]
        dtheta = theta[ii] - theta[jj]
        deta = eta[ii] - eta[jj]                           # (P, d)
        # Mahalanobis quadratic form: einsum row-wise
        deta_W = deta @ W                                  # (P, d)
        deta_sq = np.einsum("pi,pi->p", deta_W, deta)      # (P,)
        J = (dtheta * dtheta) / (sigma_theta_sq + deta_sq)
        pieces.append(
            pd.DataFrame(
                {
                    "i": ii,
                    "j": jj,
                    "ethnicity": cell_key[0],
                    "gender": cell_key[1],
                    "age": cell_key[2],
                    "dtheta": dtheta,
                    "abs_dtheta": np.abs(dtheta),
                    "deta_sq": deta_sq,
                    "J": J,
                }
            )
        )
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


def sigma_sweep(
    df: pd.DataFrame, theta: np.ndarray, eta: np.ndarray, W: np.ndarray, var_theta: float
) -> pd.DataFrame:
    """Compute J under each σ_θ² ∈ SIGMA_SWEEP; return Spearman rank corr matrix."""
    from scipy.stats import spearmanr

    rankings = {}
    for frac in SIGMA_SWEEP:
        s2 = frac * var_theta
        pairs = enumerate_pairs(df, theta, eta, W, s2)
        rankings[frac] = pairs["J"].to_numpy()

    rows = []
    keys = list(rankings.keys())
    for a in keys:
        for b in keys:
            r, _ = spearmanr(rankings[a], rankings[b])
            rows.append({"sigma_a": a, "sigma_b": b, "spearman": r})
    return pd.DataFrame(rows)


def eta_pca_diversity(eta: np.ndarray, selected_pairs: pd.DataFrame) -> tuple[np.ndarray, int]:
    """PCA spectrum of selected Δη vectors. Returns (variance ratio, effective rank).

    Effective rank = participation ratio of the eigenvalues — a soft
    rank that's high when the selection spans many η directions and
    low when it collapses to a single confound axis.
    """
    if len(selected_pairs) == 0:
        return np.array([]), 0
    ii = selected_pairs["i"].to_numpy()
    jj = selected_pairs["j"].to_numpy()
    deta = eta[ii] - eta[jj]
    cov = np.cov(deta, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 1e-12][::-1]
    if eigvals.size == 0:
        return eigvals, 0
    var_ratio = eigvals / eigvals.sum()
    eff_rank = int(round((eigvals.sum() ** 2) / (eigvals ** 2).sum()))
    return var_ratio, eff_rank


def make_collage(
    df: pd.DataFrame, pair_rows: pd.DataFrame, out_path: Path, title: str, n: int = 50
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    rows = pair_rows.head(n)
    if len(rows) == 0:
        return
    thumb = 128
    cols = 10
    nrow = math.ceil(len(rows) / cols)
    W = cols * (2 * thumb + 8) + 16
    H = nrow * (thumb + 32) + 48
    canvas = Image.new("RGB", (W, H), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, 8), title, fill=(220, 220, 220), font=font)

    for k, (_, row) in enumerate(rows.iterrows()):
        col = k % cols
        rr = k // cols
        x0 = 8 + col * (2 * thumb + 8)
        y0 = 32 + rr * (thumb + 32)
        for offset, render_idx in enumerate((int(row["i"]), int(row["j"]))):
            try:
                img_path = REPO / df["img_path"].iat[render_idx]
                im = Image.open(img_path).convert("RGB").resize((thumb, thumb))
                canvas.paste(im, (x0 + offset * thumb, y0))
            except Exception:
                pass
        sq_i = (
            df["bs_eyeSquintLeft"].iat[int(row["i"])]
            + df["bs_eyeSquintRight"].iat[int(row["i"])]
        )
        sq_j = (
            df["bs_eyeSquintLeft"].iat[int(row["j"])]
            + df["bs_eyeSquintRight"].iat[int(row["j"])]
        )
        draw.text(
            (x0, y0 + thumb + 2),
            f"Δθ={sq_i - sq_j:+.2f} J={row['J']:.2f}",
            fill=(200, 200, 200),
            font=font,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def write_cumulative_plot(J_sorted: np.ndarray, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cum = np.cumsum(J_sorted)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(cum) + 1), cum)
    ax.set_xlabel("|S| (greedy pairs selected)")
    ax.set_ylabel("cumulative Σ J(i,j)")
    ax.set_title("Solver C v1.1 — squint cumulative J over selected set")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_pca_spectrum(var_ratio: np.ndarray, eff_rank: int, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(np.arange(len(var_ratio)), var_ratio)
    ax.set_xlabel("Δη eigenvector index")
    ax.set_ylabel("variance fraction")
    ax.set_title(
        f"Selected-pair Δη PCA spectrum  |  effective rank ≈ {eff_rank}"
    )
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_corpus()
    theta, eta, eta_names = build_theta_eta(df)
    var_theta = float(theta.var())
    sigma_theta_sq = SIGMA_THETA_FRAC * var_theta

    W = full_mahalanobis_W(eta, TIKHONOV_LAMBDA)
    print(f"[W] full Σ_η^-1 with Tikhonov λ={TIKHONOV_LAMBDA}; cond≈{np.linalg.cond(W):.2e}")

    pairs = enumerate_pairs(df, theta, eta, W, sigma_theta_sq)
    print(f"[paired] {len(pairs):,} candidate pairs at σ_θ²={sigma_theta_sq:.4g}")

    p50 = float(np.percentile(pairs["J"], 50))
    p90 = float(np.percentile(pairs["J"], 90))
    p99 = float(np.percentile(pairs["J"], 99))
    n_above_p90 = int((pairs["J"] > p90).sum())

    chosen_mask = greedy_select(pairs, budget=N_BUDGET, per_render_cap=PER_RENDER_CAP)
    pairs["selected"] = chosen_mask
    selected = (
        pairs[chosen_mask].sort_values("J", ascending=False).reset_index(drop=True)
    )
    bottom = pairs.sort_values("J", ascending=True).head(50).reset_index(drop=True)

    pairs.to_parquet(OUT_DIR / "squint_pairs.parquet")

    sweep_df = sigma_sweep(df, theta, eta, W, var_theta)
    sweep_df.to_csv(OUT_DIR / "squint_sigma_sweep.csv", index=False)
    sweep_pivot = sweep_df.pivot(index="sigma_a", columns="sigma_b", values="spearman")
    print("[sigma sweep] Spearman rank corr matrix:")
    print(sweep_pivot.to_string(float_format=lambda x: f"{x:.4f}"))
    min_off_diag = float(
        sweep_pivot.where(~np.eye(len(sweep_pivot), dtype=bool)).min().min()
    )

    var_ratio, eff_rank = eta_pca_diversity(eta, selected)
    write_pca_spectrum(var_ratio, eff_rank, OUT_DIR / "squint_eta_pca_spectrum.png")

    write_cumulative_plot(
        selected["J"].to_numpy(), OUT_DIR / "squint_cumulative_info.png"
    )
    make_collage(
        df, selected, OUT_DIR / "squint_top50_pairs.png", "Solver C v1.1 — top 50 by J"
    )
    make_collage(
        df, bottom, OUT_DIR / "squint_bottom50_pairs.png", "Solver C v1.1 — bottom 50 by J"
    )

    sel_dtheta = selected["abs_dtheta"]
    sel_deta = np.sqrt(selected["deta_sq"])
    all_dtheta = pairs["abs_dtheta"]
    all_deta = np.sqrt(pairs["deta_sq"])

    pass_count = n_above_p90 >= 200
    pass_diversity = eff_rank >= max(4, eta.shape[1] // 4)
    pass_sigma = min_off_diag >= 0.90

    report = [
        "# Solver C — squint feasibility report (v1.1)",
        "",
        "**Disclaimer:** J(i,j) is a finite-difference proxy for partial",
        "Fisher information, not an estimator. The score's operational",
        "justification is alignment with the Concept Sliders image-pair",
        "loss algebra, not asymptotic theory.",
        "",
        f"- Flux corpus rows (post arcface filter): {len(df)}",
        f"- candidate pairs (within ethnicity+gender+age cells): {len(pairs):,}",
        f"- η dimensions: {eta.shape[1]} = 6 blendshape scalars + {sum(1 for n in eta_names if n.startswith('base_'))} base one-hots + {ARC_PCA_DIMS} ArcFace PCs",
        f"- W: full Σ_η^−1 with Tikhonov λ={TIKHONOV_LAMBDA}",
        f"- σ_θ² (primary, 5% of var θ): {sigma_theta_sq:.4g}",
        "",
        "## J distribution (primary σ_θ²)",
        "",
        f"- p50 J: {p50:.4f}",
        f"- p90 J: {p90:.4f}",
        f"- p99 J: {p99:.4f}",
        f"- pairs with J > p90: {n_above_p90:,}",
        f"- pairs with J > p99: {int((pairs['J'] > p99).sum()):,}",
        "",
        "## Greedy selection (budget 1000, per-render cap 4)",
        "",
        f"- pairs selected: {int(chosen_mask.sum())}",
        f"- median |Δθ| selected vs all: {sel_dtheta.median():.3f} vs {all_dtheta.median():.3f}",
        f"- median √(Δη^T W Δη) selected vs all: {sel_deta.median():.3f} vs {all_deta.median():.3f}",
        "",
        "## σ_θ² sensitivity (Spearman rank corr of J across sweep)",
        "",
        "```",
        sweep_pivot.to_string(float_format=lambda x: f"{x:.4f}"),
        "```",
        f"- min off-diagonal Spearman: {min_off_diag:.4f}",
        "",
        "## Δη PCA diversity of selected set",
        "",
        f"- effective rank (participation ratio): {eff_rank} / {eta.shape[1]}",
        f"- top-3 var fractions: {', '.join(f'{v:.3f}' for v in var_ratio[:3])}",
        "",
        "## Pass criteria (v1.1)",
        "",
        f"- ≥ 200 pairs above p90: **{'PASS' if pass_count else 'FAIL'}** ({n_above_p90})",
        f"- selected Δη eff. rank ≥ max(4, η-dim/4): **{'PASS' if pass_diversity else 'FAIL'}** ({eff_rank})",
        f"- σ_θ² rank stability (min off-diag Spearman ≥ 0.90): **{'PASS' if pass_sigma else 'FAIL'}** ({min_off_diag:.3f})",
        "",
        "## Inspect",
        "",
        "- `squint_top50_pairs.png` / `..._bottom50_pairs.png` — visual sanity (subjective; the falsifiable test is the trained slider's eval battery, not these).",
        "- `squint_cumulative_info.png` — cumulative ΣJ over selected.",
        "- `squint_eta_pca_spectrum.png` — diversity of selected Δη; flat-ish spectrum is healthy, single dominant bar means selection collapsed.",
        "- `squint_sigma_sweep.csv` — full Spearman matrix.",
        "",
        "## Verdict",
        "",
        f"- Data feasibility: **{'PASS' if (pass_count and pass_diversity and pass_sigma) else 'FAIL'}**",
        "- Note: data feasibility ≠ training feasibility. If this passes, v9-squint Path B trainer port is the next step; the trained slider's eval battery is the falsifiable test of whether these pairs translate into a working slider.",
        "",
    ]
    (OUT_DIR / "squint_feasibility_report.md").write_text("\n".join(report))
    print((OUT_DIR / "squint_feasibility_report.md").read_text())


if __name__ == "__main__":
    main()
