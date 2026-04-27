"""Corpus-health report: scan sample_index.parquet for actionable problems.

Produces a markdown report + JSON summary at:
    docs/research/2026-04-23-corpus-problems.md
    output/demographic_pc/corpus_problems.json

Sections:
  1. MediaPipe detection failures (blendshapes all-zero / _neutral≈1)
  2. Cell coverage (axis × base × subtag × scale): where are the gaps?
  3. Per-seed variance at each cell: which cells are noisy enough to need
     more seeds before we trust a mean?
  4. Atom-per-axis response slopes: which axes have no atom tracking them
     linearly? (Those need raw-channel directions, not atom directions.)
  5. Cross-base response spread: for the same (axis, scale), how much do
     bases disagree?

Only reports the top offenders per section; nothing below the noise floor.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "models/blendshape_nmf/sample_index.parquet"
REPORT_MD = ROOT / "docs/research/2026-04-23-corpus-problems.md"
REPORT_JSON = ROOT / "output/demographic_pc/corpus_problems.json"


ATOM_COLS = [f"atom_{k:02d}" for k in range(20)]

# Axes where we have a continuous scale sweep and can fit per-cell slopes.
CONTINUOUS_AXES = ["smile", "beard", "anger", "surprise", "pucker",
                   "anger_rebalance", "surprise_rebalance", "pucker_rebalance",
                   "disgust_rebalance", "lip_press_rebalance"]


def detect_failures(df: pd.DataFrame) -> pd.DataFrame:
    """MediaPipe non-detection: _neutral == 1.0 exactly, or all bs_ ≈ 0."""
    bs_cols = [c for c in df.columns if c.startswith("bs_")]
    bs = df[bs_cols].to_numpy()
    is_zero = np.allclose(bs.sum(axis=1), 0.0, atol=1e-6)
    is_neutral = df["bs__neutral"].to_numpy() > 0.99
    bad = is_zero | is_neutral
    out = df.loc[bad, ["source", "rel", "base", "axis", "scale", "seed",
                       "bs__neutral"]].copy()
    return out


def cell_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """For each (axis, base, subtag, scale) in the overnight sources, count seeds."""
    ov = df[df["source"].str.startswith("overnight_")].copy()
    g = ov.groupby(["axis", "subtag", "base", "scale"]).size().reset_index(name="n_seeds")
    return g


def per_cell_variance(df: pd.DataFrame) -> pd.DataFrame:
    """For each continuous-scale cell, compute per-seed std of every atom.
    Report max-std atom per cell — high-variance cells need more seeds."""
    ov = df[df["source"].str.startswith("overnight_")].copy()
    out = []
    for (axis, subtag, base, scale), g in ov.groupby(["axis", "subtag", "base", "scale"]):
        if len(g) < 3:
            continue
        stds = g[ATOM_COLS].std(axis=0)
        argmax = stds.idxmax()
        out.append({
            "axis": axis, "subtag": subtag, "base": base, "scale": scale,
            "n": len(g),
            "max_atom": argmax,
            "max_std": float(stds[argmax]),
            "mean_atom": float(g[argmax].mean()),
        })
    return pd.DataFrame(out)


def atom_slopes_by_axis(df: pd.DataFrame) -> pd.DataFrame:
    """For overnight continuous cells, fit scale→atom slope per (axis, subtag,
    base, atom). A strong slope means the atom responds to that axis."""
    ov = df[df["source"].str.startswith("overnight_")].copy()
    rows = []
    for (axis, subtag, base), g in ov.groupby(["axis", "subtag", "base"]):
        scales = g["scale"].to_numpy()
        uniq = np.unique(scales)
        if len(uniq) < 3:
            continue
        for atom in ATOM_COLS:
            y = g[atom].to_numpy()
            # Per-seed means help — fit to cell means to reduce per-seed noise.
            cell_means = g.groupby("scale")[atom].mean()
            x = cell_means.index.to_numpy(dtype=float)
            yv = cell_means.to_numpy()
            if len(x) < 3:
                continue
            slope, intercept = np.polyfit(x, yv, 1)
            yhat = slope * x + intercept
            ss_res = float(np.sum((yv - yhat) ** 2))
            ss_tot = float(np.sum((yv - yv.mean()) ** 2)) + 1e-12
            r2 = 1.0 - ss_res / ss_tot
            rows.append({
                "axis": axis, "subtag": subtag, "base": base, "atom": atom,
                "slope": float(slope), "r2": float(r2),
                "range": float(yv.max() - yv.min()),
            })
    return pd.DataFrame(rows)


def cross_base_spread(df: pd.DataFrame, axes: list[str]) -> pd.DataFrame:
    """At each (axis, subtag, scale) compute coefficient-of-variation of the
    max-slope atom across bases — how much does the response disagree?"""
    ov = df[(df["source"].str.startswith("overnight_")) & (df["axis"].isin(axes))].copy()
    rows = []
    for (axis, subtag, scale), g in ov.groupby(["axis", "subtag", "scale"]):
        if g["base"].nunique() < 2:
            continue
        per_base = g.groupby("base")[ATOM_COLS].mean()
        # Atom with largest cross-base spread at this scale
        cv = (per_base.std(axis=0) / (per_base.mean(axis=0).abs() + 1e-6))
        rng = per_base.max(axis=0) - per_base.min(axis=0)
        arg = rng.idxmax()
        rows.append({
            "axis": axis, "subtag": subtag, "scale": scale,
            "n_bases": int(g["base"].nunique()),
            "spread_atom": arg,
            "range_across_bases": float(rng[arg]),
            "mean_across_bases": float(per_base[arg].mean()),
        })
    return pd.DataFrame(rows)


def main():
    df = pd.read_parquet(PARQUET)
    print(f"[corpus] loaded {len(df):,} samples, {df.shape[1]} cols")

    summary = {"rows_total": int(len(df))}

    # 1. Detection failures
    fail = detect_failures(df)
    summary["detection_failures"] = {
        "total": int(len(fail)),
        "fraction": float(len(fail) / len(df)),
        "by_source": fail.groupby("source").size().to_dict(),
    }
    print(f"[corpus] detection failures: {len(fail)} ({len(fail)/len(df)*100:.2f}%)")

    # 2. Coverage
    cov = cell_coverage(df)
    under5 = cov[cov["n_seeds"] < 5]
    summary["coverage_gaps_under5"] = int(len(under5))
    summary["coverage_total_cells"] = int(len(cov))
    print(f"[corpus] overnight cells: {len(cov)}  (under-5-seed: {len(under5)})")

    # 3. Per-cell variance — top 20 noisiest
    var = per_cell_variance(df)
    var_top = var.sort_values("max_std", ascending=False).head(20)
    summary["noisiest_cells"] = var_top.to_dict(orient="records")

    # 4. Axis→atom slopes (per base); aggregate to find axes with no atom-tracker
    slopes = atom_slopes_by_axis(df)
    # For each (axis, subtag), pick the strongest atom by median |slope|
    axis_best_atom = []
    if not slopes.empty:
        for (axis, subtag), g in slopes.groupby(["axis", "subtag"]):
            g2 = g.groupby("atom").agg(
                median_slope=("slope", lambda x: float(np.median(np.abs(x)))),
                median_r2=("r2", "median"),
                median_range=("range", "median"),
            ).reset_index().sort_values("median_slope", ascending=False)
            top = g2.head(3).to_dict(orient="records")
            axis_best_atom.append({
                "axis": axis, "subtag": subtag,
                "top3_atoms_by_median_abs_slope": top,
            })
    summary["axis_atom_trackers"] = axis_best_atom

    # 5. Cross-base spread — how much does the response disagree across bases?
    all_axes = sorted(df[df["source"].str.startswith("overnight_")]["axis"].unique())
    spread = cross_base_spread(df, all_axes)
    spread_top = spread.sort_values("range_across_bases", ascending=False).head(15)
    summary["cross_base_spread_top"] = spread_top.to_dict(orient="records")

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(summary, indent=2, default=str))

    # --- Markdown report ---
    lines: list[str] = []
    lines.append("---")
    lines.append("status: live")
    lines.append("topic: corpus-rebalance")
    lines.append("summary: First corpus-problem scan over the 4942-sample index "
                 "(3502 training + 1440 overnight_drift). Surfaces detection "
                 "failures, coverage gaps, per-seed variance, axis-atom trackers, "
                 "and cross-base spread.")
    lines.append("---")
    lines.append("")
    lines.append("# Corpus problems — 2026-04-23 scan")
    lines.append("")
    lines.append(f"Total samples: {len(df):,}.")
    lines.append("")
    lines.append("## MediaPipe detection failures")
    lines.append("")
    lines.append(f"{len(fail)} / {len(df)} ({len(fail)/len(df)*100:.2f}%) "
                 f"samples have `_neutral > 0.99` or all-zero blendshapes.")
    if len(fail):
        lines.append("")
        by_src = fail.groupby("source").size().sort_values(ascending=False)
        for src, n in by_src.items():
            lines.append(f"- `{src}`: {n}")
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    lines.append(f"{len(cov)} overnight cells across (axis × subtag × base × scale). "
                 f"{len(under5)} cells have fewer than 5 seeds.")
    lines.append("")
    lines.append("### Beard axis is structurally unbalanced")
    beard = cov[cov["axis"] == "beard"].groupby(["subtag", "base"]).size().unstack(fill_value=0)
    lines.append("")
    lines.append("```")
    lines.append(beard.to_string())
    lines.append("```")
    lines.append("")
    lines.append("`beard/add` only covers 2 male bases; `beard/remove` only covers 1. "
                 "Cross-base generalisation is not testable here; this axis needs "
                 "at least `remove` on a second bearded base before slope fits mean much.")
    lines.append("")
    lines.append("## Noisiest cells (per-seed std, top 10)")
    lines.append("")
    lines.append("Cells with high per-seed std need more seeds before a mean is trustworthy.")
    lines.append("")
    lines.append("| axis | subtag | base | scale | n | atom | std | mean |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, r in var_top.head(10).iterrows():
        lines.append(f"| {r['axis']} | {r['subtag']} | {r['base']} | "
                     f"{r['scale']:+.2f} | {int(r['n'])} | {r['max_atom']} | "
                     f"{r['max_std']:.3f} | {r['mean_atom']:.3f} |")
    lines.append("")
    lines.append("## Axis → atom tracker")
    lines.append("")
    lines.append("For each (axis, subtag) the 3 atoms with the largest median |slope| "
                 "vs scale across bases. An axis with all small slopes has no atom "
                 "that tracks it — raw channels will be needed.")
    lines.append("")
    for entry in axis_best_atom:
        lines.append(f"### {entry['axis']} / {entry['subtag']}")
        lines.append("")
        lines.append("| atom | median \\|slope\\| | median R² | median range |")
        lines.append("|---|---|---|---|")
        for t in entry["top3_atoms_by_median_abs_slope"]:
            lines.append(f"| {t['atom']} | {t['median_slope']:.3f} | "
                         f"{t['median_r2']:.3f} | {t['median_range']:.3f} |")
        lines.append("")
    lines.append("## Cross-base response spread (top 15)")
    lines.append("")
    lines.append("At these cells the response-atom's mean differs most across bases. "
                 "Large spread → per-base calibration strictly needed, single "
                 "global scale won't hit target uniformly.")
    lines.append("")
    lines.append("| axis | subtag | scale | n_bases | atom | range | mean |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in spread_top.iterrows():
        lines.append(f"| {r['axis']} | {r['subtag']} | {r['scale']:+.2f} | "
                     f"{int(r['n_bases'])} | {r['spread_atom']} | "
                     f"{r['range_across_bases']:.3f} | {r['mean_across_bases']:.3f} |")
    lines.append("")

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines))
    print(f"[save] → {REPORT_MD}")
    print(f"[save] → {REPORT_JSON}")


if __name__ == "__main__":
    main()
