"""Cherry-pick a slider training manifest from sample_index.parquet.

Scatter–gather: we have many corpora (v1, v2, v3, v3.1) at different
mix_b ranges with different drift profiles. Rather than train on one
corpus, pick rows across all of them that pass:

  - identity gate    : identity_pass_075 == True
  - edit-effect gate : Δ(squint) vs (source,base,seed,α=0) anchor ≥ τ_edit
  - smile confound   : |Δ(smile)| ≤ τ_smile
  - gaze confound    : |Δ(off-axis-gaze)| ≤ τ_gaze

Anchor lookup is per (source, base, seed) — α=0 within the same render
batch so prompt and demographics match. Anchor at α=0 always passes
identity (cos=1.0) and is included in the manifest unconditionally.

Emit:
  models/flux_sliders/training_manifest_<axis>.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "models/blendshape_nmf/sample_index.parquet"
DEFAULT_OUT = ROOT / "models/flux_sliders/training_manifest_eye_squint.parquet"

AXIS_TARGETS = {
    "eye_squint": {
        "edit_cols":   ["bs_eyeSquintLeft", "bs_eyeSquintRight"],
        "smile_cols":  ["bs_mouthSmileLeft", "bs_mouthSmileRight"],
        "gaze_cols":   ["bs_eyeLookOutLeft", "bs_eyeLookOutRight",
                        "bs_eyeLookInLeft",  "bs_eyeLookInRight"],
    },
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--axis", default="eye_squint")
    p.add_argument("--sources", nargs="*", default=None,
                   help="restrict to these source tags (e.g. v3_eye_squint v3_1_eye_squint)")
    p.add_argument("--tau-edit",  type=float, default=0.15,
                   help="min Δ(squint) vs α=0 anchor")
    p.add_argument("--tau-smile", type=float, default=0.15,
                   help="max |Δ(smile)| per cell")
    p.add_argument("--tau-gaze",  type=float, default=0.10,
                   help="max |Δ(off-axis gaze)| per cell")
    p.add_argument("--balance", action="store_true",
                   help="v1.1.5 mode: enforce set-level confound mean ≈ 0 per α bucket")
    p.add_argument("--balance-eps", type=float, default=0.02,
                   help="set-level mean tolerance (default 0.02)")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    return p.parse_args()


def balance_alpha_bucket(rows: pd.DataFrame, eps: float) -> pd.DataFrame:
    """Greedy set-level balancer.

    Drops one cell at a time — the cell whose removal reduces the larger
    of |mean_smile|, |mean_gaze| the most — until both are within ±eps
    OR the bucket is empty. Anchors (alpha == 0) are protected; we only
    touch non-anchor cells.
    """
    if len(rows) == 0:
        return rows
    cur = rows.copy().reset_index(drop=True)
    while True:
        m_s = cur["confound_smile"].mean()
        m_g = cur["confound_gaze"].mean()
        if abs(m_s) <= eps and abs(m_g) <= eps:
            return cur
        if len(cur) <= 1:
            return cur
        n = len(cur)
        # For each candidate row i, compute new means after removal:
        # new_mean = (sum - x_i) / (n - 1)  ⇒  new_score = max(|new_mean_s|, |new_mean_g|)
        s_sum = cur["confound_smile"].sum()
        g_sum = cur["confound_gaze"].sum()
        new_s = (s_sum - cur["confound_smile"].values) / (n - 1)
        new_g = (g_sum - cur["confound_gaze"].values) / (n - 1)
        new_score = np.maximum(np.abs(new_s), np.abs(new_g))
        cur_score = max(abs(m_s), abs(m_g))
        # Pick the row whose removal most reduces score; bail if no improvement.
        i_drop = int(np.argmin(new_score))
        if new_score[i_drop] >= cur_score - 1e-9:
            return cur
        cur = cur.drop(index=i_drop).reset_index(drop=True)


def compute_deltas(df: pd.DataFrame, cols: dict[str, list[str]]) -> pd.DataFrame:
    """Add edit_effect, confound_smile, confound_gaze columns as Δ vs
    (source, base, seed, α=0) anchor."""
    df = df.copy()
    df["_edit_raw"]  = df[cols["edit_cols"]].sum(axis=1)
    df["_smile_raw"] = df[cols["smile_cols"]].sum(axis=1)
    df["_gaze_raw"]  = df[cols["gaze_cols"]].abs().sum(axis=1)

    # anchor (α=0) per (source, base, seed)
    anchors = (df[df["alpha"].abs() < 1e-6]
               .groupby(["source", "base", "seed"], dropna=False)
               [["_edit_raw", "_smile_raw", "_gaze_raw"]]
               .first()
               .rename(columns=lambda c: c + "_anchor"))

    df = df.merge(anchors, on=["source", "base", "seed"], how="left")
    df["edit_effect"]    = df["_edit_raw"]  - df["_edit_raw_anchor"]
    df["confound_smile"] = df["_smile_raw"] - df["_smile_raw_anchor"]
    df["confound_gaze"]  = df["_gaze_raw"]  - df["_gaze_raw_anchor"]
    return df.drop(columns=[c for c in df.columns if c.endswith("_raw") or c.endswith("_anchor")])


def main():
    args = parse_args()
    if args.axis not in AXIS_TARGETS:
        raise SystemExit(f"unknown axis: {args.axis}")

    df = pd.read_parquet(INDEX)
    print(f"[index] {len(df)} rows")

    # restrict to rows that match this axis
    df = df[df["axis"] == args.axis].reset_index(drop=True)
    print(f"[axis={args.axis}] {len(df)} rows")

    if args.sources:
        df = df[df["source"].isin(args.sources)].reset_index(drop=True)
        print(f"[sources={args.sources}] {len(df)} rows")

    # blendshapes present?
    missing = df[AXIS_TARGETS[args.axis]["edit_cols"][0]].isna().sum()
    if missing:
        print(f"[warn] {missing} rows lack blendshapes — dropping")
        df = df.dropna(subset=AXIS_TARGETS[args.axis]["edit_cols"]).reset_index(drop=True)

    df = compute_deltas(df, AXIS_TARGETS[args.axis])

    # gate
    is_anchor = df["alpha"].abs() < 1e-6
    edit_ok   = df["edit_effect"]    >= args.tau_edit
    smile_ok  = df["confound_smile"].abs() <= args.tau_smile
    gaze_ok   = df["confound_gaze"].abs()  <= args.tau_gaze
    id_ok     = df["identity_pass_075"].astype(bool)

    keep = is_anchor | (edit_ok & smile_ok & gaze_ok & id_ok)
    df["keep"] = keep
    print(f"[gate] τ_edit={args.tau_edit}  τ_smile={args.tau_smile}  τ_gaze={args.tau_gaze}")
    print(f"[gate] anchors:        {is_anchor.sum()}")
    print(f"[gate] edit_ok:        {edit_ok.sum()}")
    print(f"[gate] smile_ok:       {smile_ok.sum()}")
    print(f"[gate] gaze_ok:        {gaze_ok.sum()}")
    print(f"[gate] identity_ok:    {id_ok.sum()}")
    print(f"[gate] keep (union with anchors): {keep.sum()} / {len(df)}")

    # breakdown
    print("\n[breakdown] keep-rate by (source, alpha):")
    pivot = (df.groupby(["source", "alpha"])["keep"]
             .agg(["sum", "count"])
             .assign(rate=lambda x: x["sum"] / x["count"]))
    print(pivot.to_string())

    # final manifest — rows we keep, with paths + α + deltas
    out_cols = ["source", "base", "seed", "alpha", "img_path",
                "edit_effect", "confound_smile", "confound_gaze",
                "identity_cos_to_base", "corpus_version"]
    manifest = df.loc[keep, out_cols].reset_index(drop=True)

    # v1.1.5 — set-level balancing per α bucket
    if args.balance:
        print(f"\n[balance] enforcing |mean confound| ≤ {args.balance_eps} per α (anchors protected)")
        before_per_alpha = manifest["alpha"].value_counts().sort_index()
        is_anchor_m = manifest["alpha"].abs() < 1e-6
        anchors_m = manifest[is_anchor_m].copy()
        non_anchors = manifest[~is_anchor_m].copy()
        balanced_chunks: list[pd.DataFrame] = []
        for alpha, grp in non_anchors.groupby("alpha", sort=True):
            ms_before = grp["confound_smile"].mean()
            mg_before = grp["confound_gaze"].mean()
            balanced = balance_alpha_bucket(grp, args.balance_eps)
            ms_after = balanced["confound_smile"].mean() if len(balanced) else 0.0
            mg_after = balanced["confound_gaze"].mean() if len(balanced) else 0.0
            print(f"  α={alpha:.2f}: {len(grp):3d} → {len(balanced):3d}   "
                  f"smile {ms_before:+.3f} → {ms_after:+.3f}   "
                  f"gaze {mg_before:+.3f} → {mg_after:+.3f}")
            balanced_chunks.append(balanced)
        manifest = pd.concat([anchors_m, *balanced_chunks], ignore_index=True) \
                       .sort_values(["alpha", "source", "base", "seed"]).reset_index(drop=True)
        print(f"[balance] final manifest: {len(manifest)} rows "
              f"(was {before_per_alpha.sum()} before balancing)")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(out_path, index=False)
    print(f"\n[save] → {out_path}  ({len(manifest)} rows)")

    # report keep counts per α
    print("\n[manifest] α distribution:")
    print(manifest["alpha"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
