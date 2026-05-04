"""Reindex the Solver A squint grid renders into reverse_index.parquet.

The grid produced 336 PNGs (168 cells × {anchor, edit}) under
output/demographic_pc/solver_a_squint_grid/, with a slim scores.parquet
covering only the pass-card metrics. This script:

  1. Builds an in-process HF-format parquet shard from the 336 PNGs,
     so we can reuse extract_ffhq_metrics.process_shard() unchanged.
  2. Runs the full extractor stack (MiVOLO, FairFace, InsightFace +
     ArcFace embeddings, SigLIP-2 probes, MediaPipe blendshapes + NMF
     atoms).
  3. Converts the resulting .pt to a DataFrame matching the
     reverse_index schema, tags source='flux_solver_a_grid_squint',
     adds grid-only metadata (cell, seed, kind, scale).
  4. Concatenates with the existing reverse_index.parquet.

After this lands, Solver C v2 (or any future axis selector) can pull
these renders into the candidate pool the same way it pulls FFHQ rows.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

REPO = Path(__file__).resolve().parents[2]
GRID_DIR = REPO / "output" / "demographic_pc" / "solver_a_squint_grid"
REVERSE_INDEX = REPO / "output" / "reverse_index" / "reverse_index.parquet"
AU_LIBRARY = REPO / "models" / "blendshape_nmf" / "au_library.npz"
MEDIAPIPE_MODEL = REPO / "models" / "mediapipe" / "face_landmarker.task"
SOURCE_TAG = "flux_solver_a_grid_squint"


def collect_pngs() -> pd.DataFrame:
    scores = pd.read_parquet(GRID_DIR / "scores.parquet")
    rows: list[dict] = []
    for _, r in scores.iterrows():
        for kind, png in (("anchor", r["anchor_png"]), ("edit", r["edit_png"])):
            rows.append({
                "png_path": str(png),
                "race": r["race"],
                "gender": r["gender"],
                "age": r["age"],
                "seed": int(r["seed"]),
                "kind": kind,
                "scale": 0.0 if kind == "anchor" else 1.0,
            })
    out = pd.DataFrame(rows)
    print(f"[grid] collected {len(out)} PNGs ({len(scores)} pairs)")
    return out


def build_hf_shard(grid_rows: pd.DataFrame, out_path: Path) -> None:
    """Pack PNG bytes into HF-format parquet (column 'image' =
    {bytes, path}) so extract_ffhq_metrics.process_shard accepts it."""
    images: list[dict] = []
    for _, r in grid_rows.iterrows():
        b = Path(r["png_path"]).read_bytes()
        images.append({"bytes": b, "path": str(r["png_path"])})
    table = pa.Table.from_pydict({"image": images})
    pq.write_table(table, out_path)
    print(f"[shard] wrote {out_path} ({out_path.stat().st_size/1e6:.1f} MB, {len(images)} rows)")


def run_extractor(shard_path: Path, out_pt: Path, log: logging.Logger) -> None:
    from demographic_pc.classifiers import (
        MiVOLOClassifier, FairFaceClassifier, InsightFaceClassifier,
    )
    from demographic_pc.score_clip_probes import Siglip2Backend
    from demographic_pc.extract_ffhq_metrics import (
        build_landmarker, encode_siglip_probes, process_shard,
    )

    H = np.load(AU_LIBRARY)["H"].astype(np.float32)
    log.info(f"[au-lib] H shape {H.shape}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"[device] {device}")

    log.info("[load] MiVOLO");      mv = MiVOLOClassifier(device=device)
    log.info("[load] FairFace");    ff = FairFaceClassifier(device=device, use_hog=True)
    log.info("[load] InsightFace + ArcFace recognition")
    ins = InsightFaceClassifier(ctx_id=0, with_embedding=True)
    log.info("[load] SigLIP-2");    siglip = Siglip2Backend(device=device)
    log.info("[load] FaceLandmarker"); landmarker, mp = build_landmarker(MEDIAPIPE_MODEL)

    log.info("[siglip] encoding probes")
    probe_feats = encode_siglip_probes(siglip)

    process_shard(
        shard_path, out_pt,
        mv=mv, ff=ff, ins=ins,
        siglip=siglip, probe_feats=probe_feats,
        landmarker=landmarker, mp=mp,
        H=H, resolution=512, log=log,
    )


def pt_to_dataframe(pt_path: Path, grid_rows: pd.DataFrame) -> pd.DataFrame:
    p = torch.load(pt_path, weights_only=False)
    n = len(p["image_sha256"])
    if n != len(grid_rows):
        raise RuntimeError(f"row count mismatch: pt={n}, grid={len(grid_rows)}")

    df = pd.DataFrame({"image_sha256": list(p["image_sha256"])})
    for k, v in p.items():
        if k in ("image_sha256", "shard_name", "format_version", "resolution"):
            continue
        if isinstance(v, np.ndarray) and v.ndim > 1:
            df[k] = list(v)
        else:
            df[k] = list(v) if isinstance(v, list) else v

    df["source"] = SOURCE_TAG
    df["atom_source"] = "exact"
    # Normalize ndarray-typed object columns to lists with float64 elements
    # so pyarrow doesn't complain about mixed float32/float64 dtypes when
    # concatenated with the existing reverse_index.
    for col in ("ff_age_probs", "ff_gender_probs", "ff_race_probs", "arcface_fp32"):
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: (None if v is None else
                           list(np.asarray(v, dtype=np.float32).astype(float).tolist()))
            )
    # extend atom columns to match reverse_index width (atom_00..atom_19);
    # k=8 library populates 0..7, the rest stay NaN — same heterogeneity
    # already present in the corpus.
    for k in range(8, 20):
        col = f"atom_{k:02d}"
        if col not in df.columns:
            df[col] = np.nan

    df["grid_race"] = grid_rows["race"].to_numpy()
    df["grid_gender"] = grid_rows["gender"].to_numpy()
    df["grid_age"] = grid_rows["age"].to_numpy()
    df["grid_seed"] = grid_rows["seed"].to_numpy()
    df["grid_kind"] = grid_rows["kind"].to_numpy()
    df["grid_scale"] = grid_rows["scale"].to_numpy()
    return df


def append_to_reverse_index(new_df: pd.DataFrame) -> None:
    ri = pd.read_parquet(REVERSE_INDEX)
    print(f"[reverse-index] {len(ri)} existing rows")

    # drop prior rows of this source so the script is idempotent
    if (ri["source"] == SOURCE_TAG).any():
        n_drop = int((ri["source"] == SOURCE_TAG).sum())
        ri = ri[ri["source"] != SOURCE_TAG].reset_index(drop=True)
        print(f"[reverse-index] dropped {n_drop} prior rows tagged {SOURCE_TAG!r}")

    # backup before write
    backup = REVERSE_INDEX.with_suffix(".parquet.bak")
    if not backup.exists():
        ri.to_parquet(backup)
        print(f"[reverse-index] backup -> {backup}")

    # align columns: add NaN for any missing-in-new cols, drop extras with warning
    extra_in_new = set(new_df.columns) - set(ri.columns)
    missing_in_new = set(ri.columns) - set(new_df.columns)
    if missing_in_new:
        for c in missing_in_new:
            new_df[c] = np.nan
    if extra_in_new:
        print(f"[reverse-index] new columns added by grid extraction: {sorted(extra_in_new)}")
    new_df = new_df[list(ri.columns) + sorted(extra_in_new)]
    # extend ri with new cols too (NaN-filled) so concat schema-aligns
    for c in extra_in_new:
        ri[c] = np.nan
    ri = ri[new_df.columns]

    merged = pd.concat([ri, new_df], ignore_index=True)
    merged.to_parquet(REVERSE_INDEX)
    print(f"[reverse-index] wrote {REVERSE_INDEX} ({len(merged)} rows, +{len(new_df)})")
    by_source = merged["source"].value_counts().to_dict()
    print(f"[reverse-index] by source: {by_source}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("extract_grid")
    t0 = time.time()

    grid_rows = collect_pngs()

    with tempfile.TemporaryDirectory(prefix="grid_extract_") as tmp:
        shard = Path(tmp) / "grid_shard.parquet"
        build_hf_shard(grid_rows, shard)

        out_pt = Path(tmp) / "grid_shard.pt"
        run_extractor(shard, out_pt, log)

        new_df = pt_to_dataframe(out_pt, grid_rows)
        log.info(f"[df] {len(new_df)} rows, {len(new_df.columns)} cols")

    append_to_reverse_index(new_df)

    log.info(f"[done] total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
