"""Effect-matrix v0 — per (axis, subtag, base) slope report across readouts.

For each overnight edit cell (axis × subtag × base), fit slope vs scale for:

  * target SigLIP probe margin   (primary signal)
  * primary blendshape-atom      (data-driven: max |slope| across 21 atoms)
  * age drift                    (mv_age, ins_age numeric)
  * identity drift               (1 - identity_cos_to_base)
  * total drift                  (1 - siglip_img_cos_to_base)

Outputs:
  output/demographic_pc/effect_matrix_v0.parquet
  docs/research/2026-04-23-effect-matrix-v0.md

Slopes fit by simple linear regression over all seeds × scales in the cell.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "models/blendshape_nmf/sample_index.parquet"
OUT_PARQUET = ROOT / "output/demographic_pc/effect_matrix_v0.parquet"
OUT_MD = ROOT / "docs/research/2026-04-23-effect-matrix-v0.md"

# axis|subtag → target SigLIP probe (None where the axis has no direct probe)
TARGET_PROBE: dict[tuple[str, str], str] = {
    ("smile", "faint"): "siglip_smiling_margin",
    ("smile", "warm"):  "siglip_smiling_margin",
    ("smile", "broad"): "siglip_smiling_margin",
    ("smile", "manic"): "siglip_smiling_margin",
    ("beard", "add"):   "siglip_bearded_margin",
    ("beard", "remove"):"siglip_bearded_margin",
    ("beard_rebalance", "remove"): "siglip_bearded_margin",
    ("anger", "anger"):       "siglip_angry_margin",
    ("surprise", "surprise"): "siglip_surprised_margin",
    ("pucker", "pucker"):     "siglip_puckered_lips_margin",
}

# whether an "add" or "remove" direction is expected: +1 or -1 on the probe
EXPECTED_SIGN: dict[tuple[str, str], int] = {
    ("beard", "add"):   +1,
    ("beard", "remove"):-1,
    ("beard_rebalance", "remove"): -1,
}


def _slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """OLS slope + R² of y ~ a + b·x. Handles NaN drop. Returns (nan, nan) when < 3 points."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(pd.to_numeric(pd.Series(y), errors="coerce"), dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan"), float("nan")
    x = x[m]; y = y[m]
    b, a = np.polyfit(x, y, 1)
    yhat = a + b * x
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return float(b), float(r2)


def _cell_slopes(cell: pd.DataFrame, target_probe: str | None, atom_cols: list[str]) -> dict:
    out = {
        "n_rows": len(cell),
        "n_seeds": int(cell["seed"].nunique()),
        "scale_min": float(cell["scale"].min()),
        "scale_max": float(cell["scale"].max()),
    }
    x = cell["scale"].to_numpy()

    if target_probe and target_probe in cell.columns:
        b, r2 = _slope(x, cell[target_probe].to_numpy())
        out["target_probe"] = target_probe
        out["target_slope"] = b
        out["target_r2"] = r2
    else:
        out["target_probe"] = None
        out["target_slope"] = float("nan")
        out["target_r2"] = float("nan")

    # Primary atom: the atom with max |slope| across the cell
    best_atom, best_slope, best_r2 = None, 0.0, 0.0
    for a in atom_cols:
        b, r2 = _slope(x, cell[a].to_numpy())
        if np.isfinite(b) and abs(b) > abs(best_slope):
            best_atom, best_slope, best_r2 = a, b, r2
    out["primary_atom"] = best_atom
    out["primary_atom_slope"] = float(best_slope)
    out["primary_atom_r2"] = float(best_r2)

    # Age drift (numeric)
    b, _ = _slope(x, cell["mv_age"].to_numpy().astype(float))
    out["mv_age_slope"] = b
    b, _ = _slope(x, cell["ins_age"].to_numpy().astype(float))
    out["ins_age_slope"] = b

    # Identity drift (we want it to stay high → slope is usually negative)
    b, r2 = _slope(x, 1.0 - cell["identity_cos_to_base"].to_numpy().astype(float))
    out["identity_drift_slope"] = b
    out["identity_drift_r2"] = r2

    # Total drift
    b, r2 = _slope(x, 1.0 - cell["siglip_img_cos_to_base"].to_numpy().astype(float))
    out["total_drift_slope"] = b
    out["total_drift_r2"] = r2

    # Gender flip rate at max-scale rows vs min-scale rows (mv_gender)
    x_min, x_max = x.min(), x.max()
    mv_gender = cell["mv_gender"].to_numpy()
    flip_rate = float("nan")
    if x_max > x_min:
        mask_lo = np.isclose(x, x_min)
        mask_hi = np.isclose(x, x_max)
        if mask_lo.sum() and mask_hi.sum():
            g_lo = pd.Series(mv_gender[mask_lo]).mode()
            if len(g_lo):
                g_lo_mode = g_lo.iloc[0]
                flip_rate = float(np.mean(mv_gender[mask_hi] != g_lo_mode))
    out["mv_gender_flip_hi_vs_lo"] = flip_rate

    # max_env at max scale (mean across seeds) — NaN where no attn
    me = cell.loc[np.isclose(x, x_max), "max_env"].dropna()
    out["max_env_at_max_scale"] = float(me.mean()) if len(me) else float("nan")

    return out


def build_matrix() -> pd.DataFrame:
    idx = pd.read_parquet(INDEX)
    overnight = idx[idx["source"].str.startswith("overnight")].copy()
    overnight["axis"] = overnight["axis"].astype(str)
    overnight["subtag"] = overnight["subtag"].astype(str)
    atom_cols = [c for c in overnight.columns if c.startswith("atom_")]

    rows = []
    for (axis, subtag, base), cell in overnight.groupby(["axis", "subtag", "base"]):
        target_probe = TARGET_PROBE.get((axis, subtag))
        rec = {"axis": axis, "subtag": subtag, "base": base}
        rec.update(_cell_slopes(cell, target_probe, atom_cols))
        rec["expected_sign"] = EXPECTED_SIGN.get((axis, subtag), +1)
        rows.append(rec)
    return pd.DataFrame(rows)


def _fmt(v, spec=".3f") -> str:
    try:
        return f"{float(v):{spec}}"
    except Exception:
        return "–"


def write_report(mat: pd.DataFrame) -> None:
    lines = [
        "---",
        "status: live",
        "topic: metrics-and-direction-quality",
        "---",
        "",
        "# Effect matrix v0 (2026-04-23)",
        "",
        "Per (axis, subtag, base) cell, slopes of each readout vs `scale`, fit by OLS "
        "across all seeds × scales in the cell. Built over the overnight_* corpus "
        "(1,536 rows) from the fully populated `models/blendshape_nmf/sample_index.parquet`.",
        "",
        "Artifact: [`output/demographic_pc/effect_matrix_v0.parquet`](../../output/demographic_pc/effect_matrix_v0.parquet).",
        "",
        "## Readouts",
        "",
        "- **target_slope** — slope of the axis's target SigLIP-2 probe margin vs scale. "
        "Positive for \"add\" directions, negative for \"remove\".",
        "- **primary_atom / slope** — the NMF atom (out of 21) with the largest |slope|; "
        "empty if no atom tracks the edit (e.g. beard has no blendshape analog).",
        "- **mv_age_slope, ins_age_slope** — age drift in years per unit scale, "
        "MiVOLO and InsightFace readings.",
        "- **identity_drift_slope** — slope of `1 - cos(ArcFace_this, ArcFace_base)`. "
        "Positive = identity drifts away from base.",
        "- **total_drift_slope** — slope of `1 - cos(SigLIP_img_this, SigLIP_img_base)`. "
        "Catch-all for effects outside blendshapes + probes.",
        "- **mv_gender_flip_hi_vs_lo** — fraction of seeds where the MiVOLO-predicted "
        "gender at max-scale differs from the majority gender at min-scale. 0 = stable.",
        "- **max_env_at_max_scale** — mean collapse envelope at the cell's max scale. "
        "NaN where no attn cache (all overnight_* rows currently lack attn pkls — "
        "stamped only where has_attn=True elsewhere).",
        "",
    ]

    # one section per (axis, subtag)
    for (axis, subtag), grp in mat.groupby(["axis", "subtag"]):
        header = f"## {axis} · {subtag}"
        lines.append(header)
        lines.append("")
        target_probe = grp["target_probe"].iloc[0]
        exp = int(grp["expected_sign"].iloc[0])
        sign_txt = "+" if exp > 0 else "–"
        lines.append(f"Target probe: `{target_probe}`  · expected sign `{sign_txt}`  · "
                     f"cells: {len(grp)}")
        lines.append("")
        lines.append("| base | target | atom (slope) | mv_age/y | ins_age/y | id drift | "
                     "total drift | gender flip |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, r in grp.iterrows():
            atom_txt = f"{r['primary_atom']} ({_fmt(r['primary_atom_slope'])})" \
                       if r['primary_atom'] else "–"
            lines.append(
                f"| {r['base']} | {_fmt(r['target_slope'])} (R²={_fmt(r['target_r2'], '.2f')}) | "
                f"{atom_txt} | "
                f"{_fmt(r['mv_age_slope'], '+.2f')} | "
                f"{_fmt(r['ins_age_slope'], '+.2f')} | "
                f"{_fmt(r['identity_drift_slope'], '+.3f')} | "
                f"{_fmt(r['total_drift_slope'], '+.3f')} | "
                f"{_fmt(r['mv_gender_flip_hi_vs_lo'], '.2f')} |"
            )
        lines.append("")

    # Aggregate takeaways
    lines.append("## Aggregate observations")
    lines.append("")
    by_at = mat.groupby(["axis", "subtag"]).agg(
        target=("target_slope", "mean"),
        target_std=("target_slope", "std"),
        age_mv=("mv_age_slope", "mean"),
        age_ins=("ins_age_slope", "mean"),
        id_drift=("identity_drift_slope", "mean"),
        total_drift=("total_drift_slope", "mean"),
        gender_flip=("mv_gender_flip_hi_vs_lo", "mean"),
    ).round(3)
    lines.append("Mean slopes across bases for each (axis, subtag):")
    lines.append("")
    lines.append("| axis/subtag | target | σ | mv_age/y | ins_age/y | id drift | "
                 "total drift | gender flip |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for (axis, subtag), row in by_at.iterrows():
        lines.append(
            f"| {axis}/{subtag} | "
            f"{_fmt(row['target'])} | {_fmt(row['target_std'], '.2f')} | "
            f"{_fmt(row['age_mv'], '+.2f')} | {_fmt(row['age_ins'], '+.2f')} | "
            f"{_fmt(row['id_drift'], '+.3f')} | {_fmt(row['total_drift'], '+.3f')} | "
            f"{_fmt(row['gender_flip'], '.2f')} |"
        )
    lines.append("")

    lines.append("## Interpretation hooks")
    lines.append("")
    lines.append("- If target slope is small or flips sign across bases, the edit "
                 "direction doesn't transfer — candidate for re-pairing the prompt.")
    lines.append("- Large |mv_age_slope| with a semantic edit flags a classical "
                 "demographic confound (smile makes younger, beard-add makes older). "
                 "These are the entries the composition solver will use to counter-edit.")
    lines.append("- identity_drift >> total_drift means identity features moved more "
                 "than the overall SigLIP distribution — likely a face-specific "
                 "signal. Opposite ordering means the edit changed the non-face "
                 "scene (background, lighting) more than the face.")
    lines.append("- gender_flip > 0.25 at max-scale means the axis crosses a "
                 "demographic boundary often enough to matter — flag for "
                 "preservation-clause work.")
    lines.append("")
    lines.append("## Next")
    lines.append("")
    lines.append("- Compose counter-edits on the axes with biggest non-target drifts: "
                 "expected first targets are smile×manic (largest age drift) and "
                 "beard/add (largest identity drift).")
    lines.append("- Render demographic-edit pairs (age, gender, race) so they become "
                 "composable δs, not just measurements.")
    lines.append("- Verify `max_env` / `T_ratio` on the new overnight corpus once "
                 "attn caches are built for the overnight sources (future batch).")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    mat = build_matrix()
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    mat.to_parquet(OUT_PARQUET, index=False, compression="zstd")
    print(f"[save] → {OUT_PARQUET}  rows={len(mat)}")
    write_report(mat)
    print(f"[save] → {OUT_MD}")
    print()
    # quick console preview
    cols = ["axis", "subtag", "base", "target_slope", "target_r2", "primary_atom",
            "primary_atom_slope", "mv_age_slope", "identity_drift_slope",
            "total_drift_slope", "mv_gender_flip_hi_vs_lo"]
    print(mat[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
