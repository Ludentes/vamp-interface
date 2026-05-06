#!/usr/bin/env python3
"""Curate a small (race, age, gender) subset across both photoreal pools.

Why: the full 42×2 = 84 photoreal anchor pool is overkill for the bridge
scorecard. We want a small, demographically informative subset that
stresses different priors:

  1. white__young__f         — easy baseline (well-represented prior)
  2. east_asian__adult__m    — race + gender + age all shifted
  3. south_asian__elderly__f — extreme age + underrepresented race

Same three cells across both pools so Flux-rendered vs real-photo are
directly A/B-comparable on identical demographic targets.

Effect: prune the staged PNGs in data/anchors/{photoreal_grid,
photoreal_ffhq}/ to the chosen subset, and rewrite each manifest.parquet
to only the surviving rows. The full pool can be regenerated any time by
re-running the upstream selector (select_photoreal_anchors.py /
select_ffhq_anchors.py) — we keep manifests ground-truth-ish at
manifest.full.parquet siblings.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]

# Curated demographic spread. Same triple for both pools so the renders
# A/B-compare cleanly on identical demographic targets.
SUBSET = [
    ("white",       "young",   "f"),
    ("east_asian",  "adult",   "m"),
    ("south_asian", "elderly", "f"),
]

POOLS = [
    REPO / "data" / "anchors" / "photoreal_grid",
    REPO / "data" / "anchors" / "photoreal_ffhq",
]


def curate_pool(pool_dir: Path, subset: list[tuple[str, str, str]],
                dry_run: bool) -> None:
    manifest_path = pool_dir / "manifest.parquet"
    if not manifest_path.exists():
        print(f"[skip] no manifest at {manifest_path}")
        return

    full = pl.read_parquet(manifest_path)
    print(f"\n[{pool_dir.name}] full manifest: {len(full)} rows")

    subset_df = pl.DataFrame(
        subset,
        schema=["race", "age", "gender"],
        orient="row",
    )
    keep = full.join(subset_df, on=["race", "age", "gender"], how="inner")
    print(f"[{pool_dir.name}] subset matched: {len(keep)}/{len(subset)}")

    if len(keep) != len(subset):
        missing = (
            subset_df.join(keep, on=["race", "age", "gender"], how="anti")
            .to_dicts()
        )
        print(f"[warn] missing cells: {missing}")

    keep_paths = set(Path(p).name for p in keep.get_column("anchor_path"))
    drop_pngs = [p for p in pool_dir.glob("*.png") if p.name not in keep_paths]
    print(f"[{pool_dir.name}] would drop {len(drop_pngs)} PNGs, keep {len(keep_paths)}")

    if dry_run:
        return

    # Preserve full manifest for re-expansion.
    full_backup = manifest_path.with_name("manifest.full.parquet")
    if not full_backup.exists():
        shutil.copy2(manifest_path, full_backup)
        print(f"[{pool_dir.name}] backed up full manifest → {full_backup.name}")

    for png in drop_pngs:
        png.unlink()

    keep.write_parquet(manifest_path)
    print(f"[{pool_dir.name}] rewrote manifest.parquet ({len(keep)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    print("Subset:")
    for r, a, g in SUBSET:
        print(f"  {r:14s} {a:8s} {g}")

    for pool in POOLS:
        curate_pool(pool, SUBSET, args.dry_run)


if __name__ == "__main__":
    main()
