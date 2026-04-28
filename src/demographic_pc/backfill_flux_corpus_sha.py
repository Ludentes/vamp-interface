"""Add image_sha256 to the existing Flux-corpus reverse-index parquets.

For each parquet that has a path column pointing at an existing PNG, compute
the sha256 of the raw bytes and add a column. Writes back in place after a
backup. Idempotent: skips parquets that already have image_sha256.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# (parquet_path, path_column, path_root_for_resolution)
TARGETS = [
    (ROOT / "models/blendshape_nmf/sample_index.parquet",
     "img_path", ROOT),
    (ROOT / "output/demographic_pc/classifier_scores.parquet",
     "img_path", ROOT),
    (ROOT / "output/demographic_pc/clip_probes_siglip2.parquet",
     "rel_from_overnight",
     ROOT / "output/demographic_pc/overnight_drift"),
]


def sha256_file(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def backfill_one(parquet_path: Path, path_col: str, path_root: Path) -> None:
    if not parquet_path.exists():
        print(f"  SKIP (missing): {parquet_path}")
        return
    df = pd.read_parquet(parquet_path)
    if "image_sha256" in df.columns:
        print(f"  SKIP (already has image_sha256): {parquet_path}")
        return
    if path_col not in df.columns:
        print(f"  SKIP (no '{path_col}' column): {parquet_path}")
        return

    backup = parquet_path.with_suffix(parquet_path.suffix + ".bak.before_sha")
    if not backup.exists():
        shutil.copy2(parquet_path, backup)
        print(f"  backed up to {backup.name}")

    sha_col: list[str | None] = []
    missing = 0
    for rel in df[path_col]:
        if rel is None or (isinstance(rel, float)):
            sha_col.append(None); missing += 1; continue
        p = (path_root / rel).resolve()
        if not p.exists():
            sha_col.append(None); missing += 1; continue
        sha_col.append(sha256_file(p))
    df["image_sha256"] = sha_col
    df.to_parquet(parquet_path, index=False)
    print(f"  wrote {parquet_path} ({len(df)} rows, {missing} unresolved paths)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for parquet_path, path_col, path_root in TARGETS:
        print(f"backfill {parquet_path.relative_to(ROOT)}")
        if args.dry_run:
            print("  (dry-run; skipping write)")
            continue
        backfill_one(parquet_path, path_col, path_root)


if __name__ == "__main__":
    main()
