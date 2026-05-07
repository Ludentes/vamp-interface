"""Build anchors.parquet — one row per anchor portrait.

Columns:
  anchor_stem (str), anchor_path (str), mp_blendshapes (list<f32>[51]),
  ypr (list<f32>[3])

Run on the Phase-2 anchors used by bridge / teacher_full / personalive_rgb.
Default: just the canonical asian_m anchor we use everywhere.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _mp_blendshape import make_landmarker, extract_one  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", nargs="+",
                    default=["data/llf-phase2/asian_m__06_neutral.midframe.png"])
    ap.add_argument("--out", default="exp_output/arkit_bridge/parquet/anchors.parquet")
    args = ap.parse_args()

    lm = make_landmarker()
    rows = []
    for p in args.anchors:
        path = Path(p)
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        bs, ypr = extract_one(lm, rgb)
        rows.append({
            "anchor_stem": path.stem,
            "anchor_path": str(path.resolve()),
            "mp_blendshapes": bs.tolist(),
            "ypr": ypr.tolist(),
        })
        print(f"  {path.stem}: bs nan={int(np.isnan(bs).any())}  ypr={ypr}")

    df = pl.DataFrame(rows, schema={
        "anchor_stem": pl.String,
        "anchor_path": pl.String,
        "mp_blendshapes": pl.List(pl.Float32),
        "ypr": pl.List(pl.Float32),
    })
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    print(f"wrote {out}  rows={len(df)}")


if __name__ == "__main__":
    main()
