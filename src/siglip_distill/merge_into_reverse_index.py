"""Merge per-source SigLIP-feature sidecar parquets into reverse_index.parquet.

Inputs (one or more, will be concatenated, dedup by image_sha256 keeping first):
  --sidecar  <path>   Repeatable. Each must have columns
                      image_sha256, source, siglip_img_emb_fp16.

Output:
  Writes back to --reverse-index (atomic via tmp file). Adds column
  `siglip_img_emb_fp16` (ndarray (1152,) fp16) where SHA matches; rows with no
  match get NaN sentinel (None in the column, since arrays can't carry NaN cleanly).

Run with --dry-run to preview coverage stats without writing.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_sidecars(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_parquet(p)
        missing = {"image_sha256", "source", "siglip_img_emb_fp16"} - set(df.columns)
        if missing:
            raise SystemExit(f"{p} missing columns: {missing}")
        frames.append(df)
        print(f"  {p}: {len(df)} rows (sources: {df['source'].value_counts().to_dict()})")
    cat = pd.concat(frames, ignore_index=True)
    before = len(cat)
    cat = cat.drop_duplicates(subset=["image_sha256"], keep="first").reset_index(drop=True)
    print(f"  combined: {before} -> {len(cat)} after dedup on image_sha256")
    out = cat[["image_sha256", "siglip_img_emb_fp16"]]
    assert isinstance(out, pd.DataFrame)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidecar", type=Path, action="append", required=True,
                    help="Sidecar parquet (repeatable)")
    ap.add_argument("--reverse-index", type=Path,
                    default=Path("output/reverse_index/reverse_index.parquet"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"[load] sidecars ({len(args.sidecar)})")
    sidecar = load_sidecars(args.sidecar)

    print(f"[load] {args.reverse_index}")
    rev = pd.read_parquet(args.reverse_index)
    print(f"  reverse_index: {len(rev)} rows, sources: {rev['source'].value_counts().to_dict()}")

    print("[merge] left-joining on image_sha256")
    if "siglip_img_emb_fp16" in rev.columns:
        print("  WARN: column already exists; will be replaced")
        rev = rev.drop(columns=["siglip_img_emb_fp16"])
    merged = rev.merge(sidecar, on="image_sha256", how="left")

    has = merged["siglip_img_emb_fp16"].notna().sum()
    miss = len(merged) - has
    print(f"[coverage] matched={has}/{len(merged)} ({has/len(merged):.1%}), missing={miss}")
    for src, sub in merged.groupby("source"):
        h = sub["siglip_img_emb_fp16"].notna().sum()
        print(f"  {src}: matched={h}/{len(sub)} ({h/len(sub):.1%})")

    # Sanity: any matched embedding has the expected shape
    sample = merged["siglip_img_emb_fp16"].dropna().head(1).values
    if len(sample):
        v = np.asarray(sample[0])
        print(f"[shape] sample emb shape={v.shape} dtype={v.dtype} L2={np.linalg.norm(v.astype(np.float32)):.4f}")

    if args.dry_run:
        print("[dry-run] not writing")
        return

    tmp = args.reverse_index.with_suffix(".tmp.parquet")
    merged.to_parquet(tmp, index=False)
    tmp.replace(args.reverse_index)
    print(f"[write] {args.reverse_index} ({args.reverse_index.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
