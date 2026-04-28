"""One-shot builder for a small FFHQ smoke fixture.

Reads a FFHQ parquet shard and saves the first N rows to a small parquet
under tests/fixtures/. Run once, commit the parquet.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path,
                    default=Path(__file__).parent / "ffhq_shard0.parquet",
                    help="path to a FFHQ train-*.parquet shard "
                         "(default: tests/fixtures/ffhq_shard0.parquet — "
                         "kept locally, gitignored)")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "ffhq_smoke.parquet")
    args = ap.parse_args()

    table = pq.read_table(args.src)
    sliced = table.slice(0, args.n)
    pq.write_table(sliced, args.out)
    print(f"wrote {args.out} ({sliced.num_rows} rows, "
          f"{args.out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
