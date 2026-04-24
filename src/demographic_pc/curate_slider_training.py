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
                   help="max |Δ(smile)|")
    p.add_argument("--tau-gaze",  type=float, default=0.10,
                   help="max |Δ(off-axis gaze)|")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    return p.parse_args()


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
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(out_path, index=False)
    print(f"\n[save] → {out_path}  ({len(manifest)} rows)")

    # report keep counts per α
    print("\n[manifest] α distribution:")
    print(manifest["alpha"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
