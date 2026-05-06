#!/usr/bin/env python3
"""Lift one best-neutral FFHQ photo per (race, age, gender) cell into
data/anchors/photoreal_ffhq/. Real-photo counterpart to
select_photoreal_anchors.py (which uses Flux-rendered grid anchors).

Two-stage pipeline:

  Stage 1 — sha → (shard, row) lookup (cached):
    Walk per-shard metrics .pt files at /media/newub/Seagate Hub/arc_distill/
    metrics/. Each .pt stores image_sha256 in the same order as rows in the
    matching FFHQ shard parquet, so position k → image k. Cache to
    output/reverse_index/ffhq_sha_lookup.parquet (one row per FFHQ image,
    cols: image_sha256, shard_path, row_idx, png_filename).

  Stage 2 — selection + extraction:
    Filter the unified reverse_index (source='ffhq') to ffhq + face_detected.
    Map FairFace labels to the squint-grid taxonomy:
      race:    White→white, Black→black, East Asian→east_asian,
               Southeast Asian→southeast_asian, Indian→south_asian,
               Middle Eastern→middle_eastern, Latino_Hispanic→latino
      age_bin: 0-2,3-9,10-19,20-29 → young
               30-39,40-49         → adult
               50-59,60-69,70+     → elderly
      gender:  M→m, F→f

    For each (race, age, gender) cell:
      1. drop SigLIP probes flagging glasses, eyes_closed, smiling,
         open_mouth, surprised, puckered_lips, angry > 0.05
         ('wrinkled' deliberately excluded — see select_photoreal_anchors.py)
      2. minimise neutrality_score = bs_squint + bs_smile + |bs_brow|
         where bs_squint = max eyeSquint L/R, bs_smile = max mouthSmile L/R,
               bs_brow = mean of inner/outerUp - mean of brow_down
      3. tie-break on classifier confidence (mv_gender_conf desc) then sha

    Read the chosen rows' PNG bytes via the shard lookup, save to
    data/anchors/photoreal_ffhq/<race>__<age>__<gender>__<sha8>.png plus
    manifest.parquet (format_version: 1).
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import torch

REPO = Path(__file__).resolve().parents[1]
ARC_DISTILL = Path("/media/newub/Seagate Hub/arc_distill")
SHARD_DIR = ARC_DISTILL / "ffhq_parquet" / "data"
METRICS_DIR = ARC_DISTILL / "metrics"
LOOKUP_PARQUET = REPO / "output" / "reverse_index" / "ffhq_sha_lookup.parquet"
REVERSE_INDEX = REPO / "output" / "reverse_index" / "reverse_index.parquet"
DST_ROOT = REPO / "data" / "anchors" / "photoreal_ffhq"
MANIFEST_FORMAT_VERSION = 1

RACE_MAP = {
    "White": "white",
    "Black": "black",
    "East Asian": "east_asian",
    "Southeast Asian": "southeast_asian",
    "Indian": "south_asian",
    "Middle Eastern": "middle_eastern",
    "Latino_Hispanic": "latino",
}
GENDER_MAP = {"M": "m", "F": "f"}
# FairFace age_bin → coarse {young, adult, elderly}
AGE_MAP = {
    "0-2": "young", "3-9": "young", "10-19": "young", "20-29": "young",
    "30-39": "adult", "40-49": "adult",
    "50-59": "elderly", "60-69": "elderly", "70+": "elderly",
}

OCCLUDER_PROBES = [
    "sg_glasses_margin",
    "sg_eyes_closed_margin",
    "sg_smiling_margin",
    "sg_open_mouth_margin",
    "sg_surprised_margin",
    "sg_puckered_lips_margin",
    "sg_angry_margin",
]
PROBE_THRESHOLD = 0.05


def build_sha_lookup(metrics_dir: Path, lookup_path: Path, force: bool) -> pl.DataFrame:
    """Walk metrics .pt files, emit (sha, shard_path, row_idx) per FFHQ image."""
    if lookup_path.exists() and not force:
        df = pl.read_parquet(lookup_path)
        print(f"[lookup] cached: {len(df)} rows from {lookup_path}")
        return df

    print(f"[lookup] building from {metrics_dir}")
    pts = sorted(metrics_dir.glob("train-*-of-*.pt"))
    if not pts:
        raise SystemExit(f"no metrics .pt files at {metrics_dir} — is the drive mounted?")
    print(f"[lookup] {len(pts)} shard metrics files")

    rows: list[dict] = []
    for i, pt in enumerate(pts):
        p = torch.load(pt, map_location="cpu", weights_only=False)
        shard_name = p.get("shard_name") or pt.with_suffix(".parquet").name
        shard_path = str(SHARD_DIR / shard_name)
        shas = p["image_sha256"]
        for k, sha in enumerate(shas):
            rows.append({
                "image_sha256": sha,
                "shard_path": shard_path,
                "row_idx": k,
            })
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(pts)}] {len(rows)} sha so far")

    df = pl.DataFrame(rows).with_columns(
        pl.col("row_idx").cast(pl.UInt32),
    )
    lookup_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(lookup_path)
    print(f"[lookup] wrote {lookup_path}: {len(df)} rows")
    return df


def neutrality_components(df: pl.DataFrame) -> pl.DataFrame:
    """Compute squint / smile / brow magnitudes from per-blendshape cols.

    `_squint` and `_smile` use max(L,R) so a one-sided expression still
    counts. `_brow` is signed: positive = raised brows, negative = furrowed.
    The brow formula collapses L/R outer-up to a mean, then averages with
    the midline inner-up so each anatomical site contributes equally:
        up   = (innerUp + (outerUpL + outerUpR)/2) / 2
        down = (downL + downR) / 2
        _brow = up - down
    abs() taken downstream so both polarities count as non-neutral.
    """
    up = (
        pl.col("bs_browInnerUp")
        + (pl.col("bs_browOuterUpLeft") + pl.col("bs_browOuterUpRight")) / 2
    ) / 2
    down = (pl.col("bs_browDownLeft") + pl.col("bs_browDownRight")) / 2
    return df.with_columns(
        pl.max_horizontal("bs_eyeSquintLeft", "bs_eyeSquintRight").alias("_squint"),
        pl.max_horizontal("bs_mouthSmileLeft", "bs_mouthSmileRight").alias("_smile"),
        (up - down).alias("_brow"),
    ).with_columns(
        (
            pl.col("_squint") + pl.col("_smile") + pl.col("_brow").abs()
        ).alias("neutrality_score")
    )


def select_per_cell(ri_path: Path) -> pl.DataFrame:
    ri = pl.read_parquet(ri_path)
    ffhq = ri.filter(
        (pl.col("source") == "ffhq")
        & pl.col("ff_detected")
        & pl.col("bs_detected")
        & pl.col("ff_race").is_in(list(RACE_MAP.keys()))
        & pl.col("ff_gender").is_in(list(GENDER_MAP.keys()))
        & pl.col("ff_age_bin").is_in(list(AGE_MAP.keys()))
    )
    print(f"[ffhq] {len(ffhq)} rows after detect+demographic filter")

    ffhq = ffhq.with_columns(
        pl.col("ff_race").replace(RACE_MAP).alias("race"),
        pl.col("ff_age_bin").replace(AGE_MAP).alias("age"),
        pl.col("ff_gender").replace(GENDER_MAP).alias("gender"),
    )

    # Occluder filter
    cond = pl.lit(True)
    for col in OCCLUDER_PROBES:
        cond = cond & (pl.col(col).fill_null(0.0) < PROBE_THRESHOLD)
    clean = ffhq.filter(cond)
    print(f"[ffhq] {len(clean)} rows after occluder filter")

    clean = neutrality_components(clean)

    cell_keys = ["race", "age", "gender"]
    # Deterministic tie-break: lower neutrality first, then higher confidence,
    # then sha alphabetical. Explicit `descending=` flag avoids polars
    # optimizer surprises with negation tricks.
    winners = (
        clean.with_columns(pl.col("mv_gender_conf").fill_null(0.0))
        .sort(
            ["neutrality_score", "mv_gender_conf", "image_sha256"],
            descending=[False, True, False],
            nulls_last=True,
        )
        .group_by(cell_keys, maintain_order=True)
        .head(1)
    )
    return winners.sort(cell_keys)


def extract_png(shard_path: Path, row_idx: int) -> bytes:
    table = pq.read_table(shard_path, columns=["image"])
    img = table.column("image")[row_idx].as_py()
    if not isinstance(img, dict) or "bytes" not in img:
        raise ValueError(f"unexpected image row at {shard_path}:{row_idx}: {type(img)}")
    return img["bytes"]


def verify_lookup_alignment(lookup: pl.DataFrame) -> None:
    """Spot-check: re-hash row 0 of the first shard and compare to the cached
    sha. Catches a silent build-script row-shuffle / row-filter regression.

    Per project memory `feedback_verify_training_data_first.md`: gate any
    corpus-derived pipeline on a hash check before trusting it.
    """
    first = lookup.filter(pl.col("row_idx") == 0).head(1)
    if len(first) == 0:
        raise SystemExit("[verify] lookup has no row_idx=0 entries")
    r = first.row(0, named=True)
    expected_sha = r["image_sha256"]
    shard = Path(r["shard_path"])
    if not shard.exists():
        print(f"[verify] skipping (shard not mounted): {shard}")
        return
    png_bytes = extract_png(shard, 0)
    actual_sha = hashlib.sha256(png_bytes).hexdigest()
    if actual_sha != expected_sha:
        raise SystemExit(
            f"[verify] sha mismatch on {shard.name} row 0:\n"
            f"  cached:    {expected_sha}\n"
            f"  re-hashed: {actual_sha}\n"
            f"  → metrics .pt row order does NOT match parquet shard row order;\n"
            f"    rebuild the lookup with --rebuild_lookup or fix the metrics build script"
        )
    print(f"[verify] sha alignment OK on {shard.name} row 0 ({expected_sha[:16]}...)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_dir", type=Path, default=METRICS_DIR)
    ap.add_argument("--lookup", type=Path, default=LOOKUP_PARQUET)
    ap.add_argument("--reverse_index", type=Path, default=REVERSE_INDEX)
    ap.add_argument("--dst_root", type=Path, default=DST_ROOT)
    ap.add_argument("--rebuild_lookup", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    if not args.metrics_dir.exists():
        raise SystemExit(f"metrics dir missing: {args.metrics_dir}\n"
                         f"  is the Seagate drive mounted?")

    lookup = build_sha_lookup(args.metrics_dir, args.lookup, args.rebuild_lookup)
    verify_lookup_alignment(lookup)
    winners = select_per_cell(args.reverse_index)
    print(f"[select] {len(winners)} cells (target up to 42)")

    # Join winners with shard lookup
    winners = winners.join(lookup, on="image_sha256", how="left")
    missing = winners.filter(pl.col("shard_path").is_null())
    if len(missing) > 0:
        print(f"[warn] {len(missing)} winners missing from sha lookup:")
        print(missing.select(["race","age","gender","image_sha256"]).head(10))
        winners = winners.filter(pl.col("shard_path").is_not_null())

    args.dst_root.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        for stale in args.dst_root.glob("*.png"):
            stale.unlink()

    rows = []
    n_copied = 0
    for r in winners.iter_rows(named=True):
        sha8 = r["image_sha256"][:8]
        dst_name = f"{r['race']}__{r['age']}__{r['gender']}__{sha8}.png"
        dst = args.dst_root / dst_name
        if not args.dry_run:
            png_bytes = extract_png(Path(r["shard_path"]), r["row_idx"])
            dst.write_bytes(png_bytes)
            n_copied += 1
        rows.append({
            "format_version": MANIFEST_FORMAT_VERSION,
            "anchor_path": str(dst),
            "race": r["race"],
            "age": r["age"],
            "gender": r["gender"],
            "image_sha256": r["image_sha256"],
            "neutrality_score": r["neutrality_score"],
            # Components saved so winner-shuffles after a formula tweak are
            # debuggable without re-running the selector.
            "neutrality_squint": r["_squint"],
            "neutrality_smile": r["_smile"],
            "neutrality_brow": r["_brow"],
            "ff_age_bin": r["ff_age_bin"],
            "mv_gender_conf": r["mv_gender_conf"],
            "shard_path": r["shard_path"],
            "row_idx": r["row_idx"],
        })

    if not args.dry_run:
        pl.DataFrame(rows).write_parquet(args.dst_root / "manifest.parquet")
        print(f"[done] copied {n_copied} anchors → {args.dst_root}")
    else:
        print(f"[dry] would copy {len(rows)} anchors")


if __name__ == "__main__":
    main()
