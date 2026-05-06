#!/usr/bin/env python3
"""Pick one best-neutral anchor per (race, age, gender) cell from the
solver_a_squint_grid corpus and stage it under data/anchors/photoreal_grid/.

Selection rule per cell (16 seeds available):
  1. Require anchor_face_detected = True (drops ~0–2 per cell typically).
  2. Drop seeds where SigLIP probes flag occluders/expression
     (glasses, eyes_closed, smiling, open_mouth, wrinkled, surprised,
     puckered_lips, angry) above a margin threshold. We want a clean
     neutral baseline.
  3. Among survivors, minimise:
        score = anchor_squint + anchor_smile_bs + |anchor_brow|
     i.e. choose the seed closest to a neutral expression in
     blendshape space.
  4. If no survivors, fall back to lowest score across all 16 seeds.

Output: data/anchors/photoreal_grid/<race>__<age>__<gender>__seed<N>.png
        data/anchors/photoreal_grid/manifest.parquet (one row per cell).

The PNGs are copied (not symlinked) so the anchor pool stays usable when
the archive drive is unmounted.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = Path(
    "/media/newub/Seagate Hub/vamp-interface-archive/output/demographic_pc/"
    "solver_a_squint_grid"
)
SCORES_PARQUET = ARCHIVE / "scores.parquet"
DST_ROOT = REPO / "data" / "anchors" / "photoreal_grid"

# Margin thresholds: SigLIP probe outputs are normalised cos-like scores
# centred near 0 with light positive bias; > 0.05 reliably means
# "the probe says yes" in this corpus.
# Probes that flag *expression* / *occluder* — these we always want to filter.
# 'wrinkled' is intentionally NOT here: it correlates with the elderly-age
# demographic this cell is meant to capture, so filtering on it biases the
# elderly bin toward atypically-smooth faces. (Reviewed 2026-05-06.)
OCCLUDER_PROBES = [
    "anchor_siglip_glasses",
    "anchor_siglip_eyes_closed",
    "anchor_siglip_smiling",
    "anchor_siglip_open_mouth",
    "anchor_siglip_surprised",
    "anchor_siglip_puckered_lips",
    "anchor_siglip_angry",
]
# 0.05: empirically separates "probe says yes" from base-rate noise on this
# corpus's SigLIP probe outputs (centred near 0, light positive bias). Verify
# by histogramming any new probe column before re-using on another corpus.
PROBE_THRESHOLD = 0.05

MANIFEST_FORMAT_VERSION = 1


def neutrality_score(df: pl.DataFrame) -> pl.DataFrame:
    """Lower = more neutral.

    `squint` and `smile_bs` are non-negative blendshape activations — taken raw.
    `brow` is signed (can be raised or furrowed) — abs(), since both polarities
    are non-neutral. Don't symmetrize the first two; they're already one-sided.
    """
    return df.with_columns(
        (
            pl.col("anchor_squint")
            + pl.col("anchor_smile_bs")
            + pl.col("anchor_brow").abs()
        ).alias("neutrality_score")
    )


def filter_clean(df: pl.DataFrame) -> pl.DataFrame:
    cond = pl.col("anchor_face_detected")
    for col in OCCLUDER_PROBES:
        cond = cond & (pl.col(col) < PROBE_THRESHOLD)
    return df.filter(cond)


def select_per_cell(df: pl.DataFrame) -> pl.DataFrame:
    df = neutrality_score(df)
    clean = filter_clean(df)

    # Cells where the strict filter wiped out all 16 seeds — relax to
    # face_detected only and warn loudly so the operator can eyeball them.
    cell_keys = ["race", "age", "gender"]
    full_cells = df.select(cell_keys).unique()
    clean_cells = clean.select(cell_keys).unique()
    missing = full_cells.join(clean_cells, on=cell_keys, how="anti")
    missing_rows = missing.to_dicts()

    if missing_rows:
        print(f"[fallback] {len(missing_rows)} cell(s) had no clean seed — "
              f"falling back to face_detected only:")
        for r in missing_rows:
            print(f"  {r['race']:16s} {r['age']:8s} {r['gender']}")

    fallback = (
        df.filter(pl.col("anchor_face_detected"))
        .join(missing, on=cell_keys, how="inner")
        .with_columns(pl.lit(True).alias("__fallback__"))
    )
    clean = clean.with_columns(pl.lit(False).alias("__fallback__"))

    pool = pl.concat([clean, fallback], how="vertical_relaxed")
    # Sort on (score, seed) so ties resolve to lowest seed deterministically.
    winners = (
        pool.sort(["neutrality_score", "seed"])
        .group_by(cell_keys, maintain_order=True)
        .head(1)
    )
    return winners.sort(cell_keys)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, default=SCORES_PARQUET)
    ap.add_argument("--dst_root", type=Path, default=DST_ROOT)
    ap.add_argument(
        "--archive_root",
        type=Path,
        default=Path("/media/newub/Seagate Hub/vamp-interface-archive"),
        help="Root for resolving anchor_png relative paths stored as "
             "'output/demographic_pc/...' in scores.parquet.",
    )
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    if not args.scores.exists():
        raise SystemExit(f"scores parquet missing: {args.scores}\n"
                         f"  is the archive drive mounted at "
                         f"/media/newub/Seagate Hub/?")

    df = pl.read_parquet(args.scores)
    print(f"[scores] {len(df)} rows over {df.select(['race','age','gender']).unique().height} cells")

    winners = select_per_cell(df)
    print(f"[select] {len(winners)} cells (target 42)")
    if len(winners) != 42:
        print(f"[warn] expected 42 cells, got {len(winners)}")

    args.dst_root.mkdir(parents=True, exist_ok=True)

    # Wipe stale anchor PNGs from prior runs so a re-run with a different
    # selection rule doesn't leave orphans (different seed → different name).
    if not args.dry_run:
        for stale in args.dst_root.glob("*.png"):
            stale.unlink()

    n_copied = 0
    rows = []
    for r in winners.iter_rows(named=True):
        rel = r["anchor_png"]
        # rel looks like "output/demographic_pc/solver_a_squint_grid/<cell>/seed<N>_anchor.png"
        src = args.archive_root / rel
        dst_name = f"{r['race']}__{r['age']}__{r['gender']}__seed{r['seed']}.png"
        dst = args.dst_root / dst_name
        if not src.exists():
            print(f"[miss] {src}")
            continue
        if not args.dry_run:
            shutil.copy2(src, dst)
            n_copied += 1
        rows.append({
            "format_version": MANIFEST_FORMAT_VERSION,
            "anchor_path": str(dst),
            "race": r["race"],
            "age": r["age"],
            "gender": r["gender"],
            "seed": r["seed"],
            "neutrality_score": r["neutrality_score"],
            "anchor_face_detected": r["anchor_face_detected"],
            "fallback_used": bool(r.get("__fallback__", False)),
            "source_archive_path": rel,
        })

    if not args.dry_run:
        manifest = pl.DataFrame(rows)
        manifest.write_parquet(args.dst_root / "manifest.parquet")
        print(f"[done] copied {n_copied} anchors → {args.dst_root}")
        print(f"[done] manifest → {args.dst_root / 'manifest.parquet'}")
    else:
        print(f"[dry] would copy {len(rows)} anchors")


if __name__ == "__main__":
    main()
