"""Compute identity_cos_to_base and siglip_img_cos_to_base per row.

Two passes:
1. `--cache-siglip`: SigLIP-2 image-feature cache over every unique img_path
   in sample_index.parquet. Writes siglip_img_features.parquet (resumable).
2. `--compute`: for each row, resolve anchor (same base + seed at scale=0
   preferred, fallback = mean over all scale=0 renders of same base),
   compute cosine(ArcFace emb, anchor ArcFace mean) and cosine(SigLIP feat,
   anchor SigLIP mean). Merge into main index.

ArcFace embeddings come from `ins_embedding` already in the index (from
score_classifiers). SigLIP features come from the cache we build here.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.demographic_pc.score_clip_probes import Siglip2Backend

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "models/blendshape_nmf/sample_index.parquet"
SIGLIP_FEAT = ROOT / "output/demographic_pc/siglip_img_features.parquet"
CHECKPOINT_EVERY = 500


def cache_siglip_features() -> None:
    idx = pd.read_parquet(INDEX)
    todo = idx["img_path"].drop_duplicates().tolist()
    done: set[str] = set()
    existing_rows: list[dict] = []
    if SIGLIP_FEAT.exists():
        prior = pd.read_parquet(SIGLIP_FEAT)
        done = set(prior["img_path"])
        existing_rows = prior.to_dict("records")
        print(f"[resume] {len(done)}/{len(todo)} already cached")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] SigLIP-2 on {device}")
    backend = Siglip2Backend(device)

    rows = list(existing_rows)
    t0 = time.time()
    n_new = 0
    for rel in todo:
        if rel in done:
            continue
        abs_path = ROOT / rel
        try:
            feat = backend.encode_image(abs_path).squeeze(0).cpu().numpy().astype(np.float32)
            rows.append({"img_path": rel, "siglip_img_feat": feat.tolist()})
        except Exception as e:
            print(f"  [warn] {rel}: {type(e).__name__}: {e}")
            rows.append({"img_path": rel, "siglip_img_feat": None})
        n_new += 1
        if n_new % CHECKPOINT_EVERY == 0:
            dt = time.time() - t0
            print(f"  [{n_new}/{len(todo) - len(done)}] {n_new / dt:.1f} img/s")
            pd.DataFrame(rows).to_parquet(SIGLIP_FEAT, index=False, compression="zstd")
    pd.DataFrame(rows).to_parquet(SIGLIP_FEAT, index=False, compression="zstd")
    print(f"[save] → {SIGLIP_FEAT}  rows={len(rows)}")


def _list_to_arr(col: pd.Series, dim: int) -> np.ndarray:
    """Turn a list-column of fixed-dim vectors into an (N, dim) array; NaN rows → zeros."""
    out = np.zeros((len(col), dim), dtype=np.float32)
    for i, v in enumerate(col):
        if v is None:
            continue
        out[i] = np.asarray(v, dtype=np.float32)
    return out


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine between two (N, D) arrays, both assumed non-zero rows => finite; else NaN."""
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    num = (a * b).sum(axis=1)
    denom = na * nb
    cos = np.full(len(a), np.nan, dtype=np.float32)
    ok = (denom > 1e-9)
    cos[ok] = num[ok] / denom[ok]
    return cos


def compute() -> None:
    idx = pd.read_parquet(INDEX)
    sip = pd.read_parquet(SIGLIP_FEAT)
    merged = idx.merge(sip, on="img_path", how="left")
    assert len(merged) == len(idx), f"join mismatch: {len(merged)} vs {len(idx)}"

    arc = _list_to_arr(merged["ins_embedding"], 512)      # already L2-normed per InsightFace
    sig = _list_to_arr(merged["siglip_img_feat"], 1152)   # SigLIP2 So400m/16 feature dim

    arc_has = np.array([v is not None for v in merged["ins_embedding"]])
    sig_has = np.array([v is not None for v in merged["siglip_img_feat"]])

    # Build per-(base, seed) anchors if a scale≈0 row exists; else fall back
    # to a per-base anchor averaged over all scale≈0 rows of that base.
    mask0 = np.isclose(merged["scale"].fillna(1e9), 0.0)
    bs_anchor_arc: dict[tuple, np.ndarray] = {}
    bs_anchor_sig: dict[tuple, np.ndarray] = {}
    b_anchor_arc: dict[str, np.ndarray] = {}
    b_anchor_sig: dict[str, np.ndarray] = {}

    zero_rows = merged[mask0]
    for (base, seed), grp in zero_rows.groupby(["base", "seed"]):
        i = grp.index.to_numpy()
        a = arc[i][arc_has[i]]
        s = sig[i][sig_has[i]]
        if len(a) > 0:
            bs_anchor_arc[(base, seed)] = a.mean(axis=0)
        if len(s) > 0:
            bs_anchor_sig[(base, seed)] = s.mean(axis=0)

    for base, grp in zero_rows.groupby("base"):
        i = grp.index.to_numpy()
        a = arc[i][arc_has[i]]
        s = sig[i][sig_has[i]]
        if len(a) > 0:
            b_anchor_arc[base] = a.mean(axis=0)
        if len(s) > 0:
            b_anchor_sig[base] = s.mean(axis=0)

    # Resolve per-row anchor (preferred > fallback > None).
    N = len(merged)
    anchor_arc = np.zeros((N, 512), dtype=np.float32)
    anchor_sig = np.zeros((N, 1152), dtype=np.float32)
    anchor_arc_has = np.zeros(N, dtype=bool)
    anchor_sig_has = np.zeros(N, dtype=bool)
    anchor_kind = np.empty(N, dtype=object)

    bases = merged["base"].to_numpy()
    seeds = merged["seed"].to_numpy()
    for i in range(N):
        key = (bases[i], int(seeds[i]))
        if key in bs_anchor_arc:
            anchor_arc[i] = bs_anchor_arc[key]; anchor_arc_has[i] = True
        elif bases[i] in b_anchor_arc:
            anchor_arc[i] = b_anchor_arc[bases[i]]; anchor_arc_has[i] = True
        if key in bs_anchor_sig:
            anchor_sig[i] = bs_anchor_sig[key]; anchor_sig_has[i] = True
        elif bases[i] in b_anchor_sig:
            anchor_sig[i] = b_anchor_sig[bases[i]]; anchor_sig_has[i] = True
        anchor_kind[i] = (
            "seed" if key in bs_anchor_arc else
            "base" if bases[i] in b_anchor_arc else
            "none"
        )

    id_cos = _cos(arc, anchor_arc)
    id_cos[~arc_has | ~anchor_arc_has] = np.nan
    sig_cos = _cos(sig, anchor_sig)
    sig_cos[~sig_has | ~anchor_sig_has] = np.nan

    merged["identity_cos_to_base"] = id_cos
    merged["siglip_img_cos_to_base"] = sig_cos
    merged["drift_anchor_kind"] = anchor_kind

    # Drop the heavy intermediate columns before save.
    out_cols = [c for c in merged.columns if c != "siglip_img_feat"]
    merged = merged[out_cols]

    merged.to_parquet(INDEX, index=False, compression="zstd")
    print(f"[save] → {INDEX}  ({INDEX.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"[stats] identity_cos:  filled={np.isfinite(id_cos).sum()}/{N}  mean={np.nanmean(id_cos):.3f}")
    print(f"[stats] siglip_cos:    filled={np.isfinite(sig_cos).sum()}/{N}  mean={np.nanmean(sig_cos):.3f}")
    print(f"[stats] anchor kinds:  {pd.Series(anchor_kind).value_counts().to_dict()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-siglip", action="store_true")
    ap.add_argument("--compute", action="store_true")
    args = ap.parse_args()
    if args.cache_siglip:
        cache_siglip_features()
    if args.compute:
        compute()
    if not (args.cache_siglip or args.compute):
        ap.print_help()


if __name__ == "__main__":
    main()
