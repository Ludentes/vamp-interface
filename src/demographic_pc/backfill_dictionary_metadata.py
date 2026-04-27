"""Backfill per-row metadata on the effect_matrix_dictionary parquet.

Each dictionary row (axis, iteration_id, variant, base) gets:

  good_scale_range  : (s_lo, s_hi) — scales where target fires AND id_cos survives
  sweet_spot_scale  : scale that maximizes target / (1 + 1.5·|1-id_cos|)
  confounds_above_tau : list of confound-name strings exceeding thresholds
  handles           : list of axis names this pair is good for (v0: just [axis])
  verified_on_bases : list of base names where the row's thresholds pass (v0:
                      [base] if target slope & id drift both meet gate, else [])

Pulls per-(base, seed, scale) rows from each iteration's results.parquet;
pure analysis, no rendering.

Usage:
  uv run python -m src.demographic_pc.backfill_dictionary_metadata
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DICT_PATH = ROOT / "output/demographic_pc/effect_matrix_dictionary.parquet"
ITER_ROOT = ROOT / "output/demographic_pc/promptpair_iterate"

# Axis-to-target-readout map. Pair rows inherit this from their axis column.
AXIS_TARGET_READOUT = {
    "smile": "siglip_smiling_margin",
    "age":   "siglip_wrinkled_margin",
    "race":  None,                       # classifier-label axis, no float target
}

# Confound thresholds (per unit scale at highest scale; signed deltas from s=0).
CONFOUND_THRESHOLDS = {
    "mv_age_abs":       5.0,    # |mv_age(s) − mv_age(0)| > 5 y
    "id_cos_drop":      0.5,    # 1 − identity_cos_to_base > 0.5
    "total_img_drop":   0.25,   # 1 − siglip_img_cos_to_base > 0.25
    "ff_race_change":   True,   # any scale s>0 flips ff_race vs s=0
    "ff_gender_change": True,
}

# Verification gates (all must pass for a (variant, base) row to be "verified"):
# target fires meaningfully AND identity survives.
VERIFY_TARGET_MIN = 0.04          # min |target@s_hi − target@0|
VERIFY_ID_COS_MIN = 0.3           # min identity_cos_to_base at any non-zero scale
VERIFY_MIN_SCORE = 0.02           # composite score > this


def _composite(target: float, id_cos: float) -> float:
    """Same scoring used in promptpair_iterate._rank: target / (1 + 1.5·(1-id_cos))."""
    return float(target) / (1.0 + 1.5 * abs(1.0 - float(id_cos)))


def _compute_row_metadata(results: pd.DataFrame, variant: str, base: str,
                          axis: str) -> dict:
    sub = results[(results["variant"] == variant) & (results["base"] == base)].copy()
    if sub.empty:
        return {}

    sub = sub.sort_values(["seed", "scale"]).reset_index(drop=True)
    target_col = AXIS_TARGET_READOUT.get(axis)
    scales = sorted(sub["scale"].unique().tolist())
    nonzero_scales = [s for s in scales if s > 1e-6]

    # Average readouts across seeds per scale for stable per-scale metrics.
    per_scale = sub.groupby("scale").agg({
        "identity_cos_to_base": "mean",
        "siglip_img_cos_to_base": "mean",
        "mv_age": "mean",
        **({target_col: "mean"} if target_col and target_col in sub.columns else {}),
    }).reset_index()

    base0 = per_scale[per_scale["scale"] == 0.0]
    mv_age_0 = float(base0["mv_age"].iloc[0]) if not base0.empty else np.nan
    target_0 = (float(base0[target_col].iloc[0])
                if target_col and target_col in base0.columns and not base0.empty
                else np.nan)

    # ── sweet_spot_scale + composite score over non-zero scales
    best_scale = None; best_score = -np.inf
    target_delta_at_best = np.nan
    id_cos_at_best = np.nan
    for s in nonzero_scales:
        row = per_scale[per_scale["scale"] == s]
        if row.empty:
            continue
        id_cos = float(row["identity_cos_to_base"].iloc[0])
        if target_col and target_col in row.columns and not np.isnan(target_0):
            target_delta = float(row[target_col].iloc[0]) - target_0
        else:
            target_delta = np.nan
        score = _composite(target_delta if not np.isnan(target_delta) else 0.0, id_cos)
        if score > best_score:
            best_score = score; best_scale = s
            target_delta_at_best = target_delta
            id_cos_at_best = id_cos

    # ── good_scale_range: contiguous scales where |target_delta| >= 0.3 * peak
    # AND identity_cos >= 0.2. For race (no target), fall back to id_cos-only.
    peak = 0.0
    if target_col and target_col in per_scale.columns:
        peak = max((abs(float(per_scale[per_scale["scale"] == s][target_col].iloc[0]) - target_0)
                    for s in nonzero_scales
                    if not per_scale[per_scale["scale"] == s].empty), default=0.0)

    good_lo = None; good_hi = None
    for s in nonzero_scales:
        row = per_scale[per_scale["scale"] == s]
        if row.empty:
            continue
        id_cos = float(row["identity_cos_to_base"].iloc[0])
        if target_col and target_col in row.columns and peak > 1e-6:
            target_delta = abs(float(row[target_col].iloc[0]) - target_0)
            target_ok = target_delta >= 0.3 * peak
        else:
            target_ok = True
        id_ok = id_cos >= 0.2
        if target_ok and id_ok:
            if good_lo is None:
                good_lo = s
            good_hi = s

    # ── confounds_above_tau: check each dim at max scale
    confounds = []
    max_scale = nonzero_scales[-1] if nonzero_scales else None
    if max_scale is not None:
        row_hi = per_scale[per_scale["scale"] == max_scale]
        if not row_hi.empty:
            mv_age_hi = float(row_hi["mv_age"].iloc[0])
            if not np.isnan(mv_age_0) and abs(mv_age_hi - mv_age_0) > CONFOUND_THRESHOLDS["mv_age_abs"]:
                sign = "+" if mv_age_hi > mv_age_0 else "-"
                confounds.append(f"mv_age({sign}{abs(mv_age_hi - mv_age_0):.1f}y)")
            id_hi = float(row_hi["identity_cos_to_base"].iloc[0])
            if (1.0 - id_hi) > CONFOUND_THRESHOLDS["id_cos_drop"]:
                confounds.append(f"identity({1.0 - id_hi:.2f})")
            sig_hi = float(row_hi["siglip_img_cos_to_base"].iloc[0])
            if (1.0 - sig_hi) > CONFOUND_THRESHOLDS["total_img_drop"]:
                confounds.append(f"total_img({1.0 - sig_hi:.2f})")

    # FF race/gender flips: check scale-wise (majority vote across seeds per scale)
    def _majority(col: str, s: float) -> str | None:
        vals = sub[sub["scale"] == s][col].tolist()
        vals = [v for v in vals if v]
        if not vals:
            return None
        return max(set(vals), key=vals.count)
    if 0.0 in scales:
        race_0 = _majority("ff_race", 0.0)
        gender_0 = _majority("ff_gender", 0.0)
        for s in nonzero_scales:
            if _majority("ff_race", s) and _majority("ff_race", s) != race_0:
                confounds.append(f"ff_race({race_0}→{_majority('ff_race', s)}@s={s})")
                break
        for s in nonzero_scales:
            if _majority("ff_gender", s) and _majority("ff_gender", s) != gender_0:
                confounds.append(f"ff_gender({gender_0}→{_majority('ff_gender', s)}@s={s})")
                break

    # ── verify_on_bases (trivial v0: just this base if the row passes gates)
    verified = False
    if not np.isnan(target_delta_at_best) and abs(target_delta_at_best) >= VERIFY_TARGET_MIN \
       and id_cos_at_best >= VERIFY_ID_COS_MIN \
       and best_score >= VERIFY_MIN_SCORE:
        verified = True

    # For race axis, target_delta is NaN. Verify on id_cos alone.
    if target_col is None and id_cos_at_best >= VERIFY_ID_COS_MIN:
        verified = True

    return {
        "good_scale_range": json.dumps([good_lo, good_hi]) if good_lo is not None else json.dumps([None, None]),
        "sweet_spot_scale": float(best_scale) if best_scale is not None else np.nan,
        "sweet_spot_score": float(best_score) if not np.isinf(best_score) else np.nan,
        "target_delta_at_sweet": float(target_delta_at_best) if not np.isnan(target_delta_at_best) else np.nan,
        "id_cos_at_sweet": float(id_cos_at_best) if not np.isnan(id_cos_at_best) else np.nan,
        "confounds_above_tau": json.dumps(confounds),
        "handles": json.dumps([axis]),
        "verified_on_bases": json.dumps([base] if verified else []),
    }


def main() -> None:
    df = pd.read_parquet(DICT_PATH)
    print(f"[load] dictionary: {len(df)} rows")

    # Load each iteration's results.parquet once
    iter_ids = sorted(df["iteration_id"].unique().tolist())
    results_cache: dict[str, pd.DataFrame] = {}
    for it in iter_ids:
        p = ITER_ROOT / it / "results.parquet"
        if p.exists():
            results_cache[it] = pd.read_parquet(p)
        else:
            print(f"[warn] missing results.parquet for {it}")

    new_cols: list[dict] = []
    missing = 0
    for _, row in df.iterrows():
        it = row["iteration_id"]
        res = results_cache.get(it)
        if res is None:
            new_cols.append({}); missing += 1; continue
        meta = _compute_row_metadata(res, row["variant"], row["base"], row["axis"])
        if not meta:
            missing += 1
        new_cols.append(meta)

    meta_df = pd.DataFrame(new_cols)
    # Drop any existing metadata columns to ensure fresh values
    for col in meta_df.columns:
        if col in df.columns:
            df = df.drop(columns=col)
    out = pd.concat([df.reset_index(drop=True), meta_df.reset_index(drop=True)], axis=1)

    # Human summary
    verified_count = sum(1 for r in new_cols if r.get("verified_on_bases", "[]") != "[]")
    with_confound = sum(1 for r in new_cols if r.get("confounds_above_tau", "[]") != "[]")
    with_sweet = sum(1 for r in new_cols if r.get("sweet_spot_scale") is not None
                     and not (isinstance(r["sweet_spot_scale"], float) and np.isnan(r["sweet_spot_scale"])))

    print(f"[compute] {len(new_cols) - missing}/{len(new_cols)} rows enriched "
          f"({missing} missing source data)")
    print(f"[compute] {verified_count} rows verified")
    print(f"[compute] {with_sweet} rows have sweet_spot_scale")
    print(f"[compute] {with_confound} rows have confounds above τ")

    out.to_parquet(DICT_PATH, index=False)
    print(f"[save] → {DICT_PATH}")

    # Quick peek at a verified smile row for sanity
    peek = out[(out["axis"] == "smile") & (out["verified_on_bases"] != "[]")]
    if not peek.empty:
        r = peek.iloc[-1]
        print("\n[sample row]")
        for k in ("iteration_id", "variant", "base", "good_scale_range",
                  "sweet_spot_scale", "sweet_spot_score", "target_delta_at_sweet",
                  "id_cos_at_sweet", "confounds_above_tau", "verified_on_bases"):
            print(f"  {k}: {r[k]}")


if __name__ == "__main__":
    main()
