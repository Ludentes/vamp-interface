"""Merge SigLIP-2 probe margins into sample_index.parquet.

Join key: strip `output/demographic_pc/overnight_drift/` from `img_path` to
get the same key SigLIP used (`rel_from_overnight`). Non-overnight rows get
NaN for clip_* columns — they'll be filled if/when we score the pre-overnight
renders.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "models/blendshape_nmf/sample_index.parquet"
SIGLIP = ROOT / "output/demographic_pc/clip_probes_siglip2.parquet"
OVERNIGHT_PREFIX = "output/demographic_pc/overnight_drift/"


def main() -> None:
    idx = pd.read_parquet(INDEX)
    sip = pd.read_parquet(SIGLIP)
    margin_cols = [c for c in sip.columns if c.endswith("_margin")]
    siglip_cols = [f"siglip_{c}" for c in margin_cols]
    sip = sip.rename(columns=dict(zip(margin_cols, siglip_cols)))

    # Drop any existing siglip_* from a prior run so we re-merge cleanly.
    for c in siglip_cols:
        if c in idx.columns:
            idx = idx.drop(columns=c)

    idx["_join_key"] = idx["img_path"].str.removeprefix(OVERNIGHT_PREFIX)
    merged = idx.merge(
        sip.rename(columns={"rel_from_overnight": "_join_key"}),
        on="_join_key", how="left",
    ).drop(columns="_join_key")

    n_filled = merged[siglip_cols[0]].notna().sum()
    print(f"[merge] rows={len(merged)}  siglip-filled={n_filled}  "
          f"({100 * n_filled / len(merged):.1f}%)")
    print(f"[merge] new cols: {siglip_cols}")
    merged.to_parquet(INDEX, index=False, compression="zstd")
    print(f"[save] → {INDEX}  "
          f"({INDEX.stat().st_size / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
