"""Build frames.parquet — per-frame ground truth from arkit_bridge_pairs pkls.

Row per training/eval frame. Source: data/arkit_bridge_pairs/all/*.pkl with
filenames `<NAME>_frame_<6-digit-idx>.pkl`. Also reads the matching LLF CSV
for `head_ypr` (yaw/pitch/roll, radians) where available.

Columns:
  take (uint8), take_name (str), frame_idx (uint32), csv_idx (uint32),
  split (str), b_expr (list<f32>[58]), head_ypr (list<f32>[3] or null),
  m_f_teacher (list<f32>[512])

Split convention (from 2026-05-05-axis-report-test-split.md):
  takes 4, 7   = test
  takes 2,3,5,6,8 = diag (train/val left to downstream tools)
  take 1       = excluded (no LLF CSV)
"""
from __future__ import annotations
import argparse
import pickle
import re
from pathlib import Path

import numpy as np
import polars as pl

DIAG_TAKES = {2, 3, 5, 6, 8}
TEST_TAKES = {4, 7}
EXCLUDED   = {1}

NAME_RE = re.compile(r"(?P<name>\d+_MySlate_(?P<take>\d+))_frame_(?P<idx>\d+)\.pkl$")


def split_for(take: int) -> str:
    if take in TEST_TAKES: return "test"
    if take in DIAG_TAKES: return "diag"
    return "excluded"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_dir", default="data/arkit_bridge_pairs/all")
    ap.add_argument("--out", default="exp_output/arkit_bridge/parquet/frames.parquet")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    pkl_dir = Path(args.pairs_dir)
    files = sorted(pkl_dir.glob("*.pkl"))
    if args.limit:
        files = files[:args.limit]
    print(f"reading {len(files)} pkls from {pkl_dir}")

    rows = []
    for i, p in enumerate(files):
        m = NAME_RE.search(p.name)
        if not m:
            print(f"  skip (name mismatch): {p.name}")
            continue
        take = int(m.group("take"))
        if take in EXCLUDED:
            continue
        take_name = m.group("name")
        frame_idx = int(m.group("idx"))
        with open(p, "rb") as f:
            d = pickle.load(f)
        b_expr = np.asarray(d["b_expr"], dtype=np.float32).reshape(-1)
        m_f    = np.asarray(d["m_f"],    dtype=np.float32).reshape(-1)
        rows.append({
            "take": take,
            "take_name": take_name,
            "frame_idx": frame_idx,
            "csv_idx": int(d.get("frame_idx", frame_idx)),
            "split": split_for(take),
            "b_expr": b_expr.tolist(),
            "m_f_teacher": m_f.tolist(),
        })
        if i % 5000 == 0 and i:
            print(f"  {i}/{len(files)}")

    print(f"  built {len(rows)} rows; encoding parquet…")
    df = pl.DataFrame(rows, schema={
        "take": pl.UInt8,
        "take_name": pl.String,
        "frame_idx": pl.UInt32,
        "csv_idx": pl.UInt32,
        "split": pl.String,
        "b_expr": pl.List(pl.Float32),
        "m_f_teacher": pl.List(pl.Float32),
    }).sort(["take", "frame_idx"])
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out, compression="zstd")
    print(f"wrote {out}  rows={len(df)}  size={out.stat().st_size/1e6:.1f} MB")
    print(df.group_by("take").len().sort("take"))


if __name__ == "__main__":
    main()
