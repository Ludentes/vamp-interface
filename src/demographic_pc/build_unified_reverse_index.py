"""Concatenate per-shard FFHQ extractor outputs + Flux-corpus reverse-index
parquets into one long parquet keyed by image_sha256.

source column values:
    "flux_corpus_v3"  - synthetic renders (existing reverse-index parquets)
    "ffhq"            - real-world FFHQ images (per-shard .pt outputs)

Columns missing in one side are filled with NaN/empty.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]


def load_ffhq_shard(pt_path: Path) -> pd.DataFrame:
    p = torch.load(pt_path, weights_only=False)
    n = len(p["image_sha256"])
    out: dict = {"source": ["ffhq"] * n, "image_sha256": list(p["image_sha256"])}
    for k, v in p.items():
        if k in {"image_sha256", "shard_name", "format_version", "resolution"}:
            continue
        if isinstance(v, np.ndarray) and v.ndim == 1:
            out[k] = v.tolist()
        elif isinstance(v, np.ndarray) and v.ndim == 2:
            out[k] = [row.tolist() for row in v]
        elif isinstance(v, list):
            out[k] = v
        elif isinstance(v, torch.Tensor):
            out[k] = v.tolist()
    return pd.DataFrame(out)


def load_ffhq_arcface(pt_dir: Path) -> pd.DataFrame:
    """Pull the ArcFace 512-d embeddings produced by encode_ffhq.py."""
    rows = []
    for pt_path in sorted(pt_dir.glob("*.pt")):
        if pt_path.name.endswith(".tmp"):
            continue
        p = torch.load(pt_path, weights_only=False)
        for sha, emb, det in zip(p["image_sha256"],
                                 p["arcface_fp32"].numpy(),
                                 p["detected"].numpy()):
            rows.append({
                "image_sha256": sha,
                "arcface_fp32": emb.tolist() if det else None,
                "ffhq_arcface_detected": bool(det),
            })
    return pd.DataFrame(rows)


def load_flux_corpus(root: Path) -> pd.DataFrame:
    """Merge the three backfilled Flux-corpus parquets on image_sha256."""
    paths = {
        "sample_index":  root / "models/blendshape_nmf/sample_index.parquet",
        "classifier":    root / "output/demographic_pc/classifier_scores.parquet",
        "siglip":        root / "output/demographic_pc/clip_probes_siglip2.parquet",
    }
    dfs = {k: pd.read_parquet(p) for k, p in paths.items() if p.exists()}
    base = dfs["classifier"][[
        "image_sha256",
        "mv_age", "mv_gender", "mv_gender_conf",
        "ff_age_bin", "ff_gender", "ff_race",
        "ff_age_probs", "ff_gender_probs", "ff_race_probs",
        "ff_detected",
        "ins_age", "ins_gender", "ins_detected",
        "ins_embedding",
    ]].copy()
    base = base.rename(columns={"ins_embedding": "arcface_fp32"})
    base["source"] = "flux_corpus_v3"

    bs_cols = [c for c in dfs["sample_index"].columns if c.startswith(("bs_", "atom_"))]
    base = base.merge(
        dfs["sample_index"][["image_sha256"] + bs_cols],
        on="image_sha256", how="left",
    )

    sg = dfs["siglip"].copy()
    sg = sg.rename(columns={c: f"sg_{c}" for c in sg.columns if c.endswith("_margin")})
    sg_cols = [c for c in sg.columns if c.startswith("sg_") and c.endswith("_margin")]
    base = base.merge(sg[["image_sha256"] + sg_cols], on="image_sha256", how="left")

    base = base[base["image_sha256"].notna()].reset_index(drop=True)
    return base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ffhq-metrics-dir", type=Path,
                    default=ROOT / "output/ffhq_metrics/metrics")
    ap.add_argument("--ffhq-encoded-dir", type=Path,
                    default=ROOT / "output/ffhq_metrics/encoded")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "output/reverse_index/reverse_index.parquet")
    ap.add_argument("--log", type=Path,
                    default=ROOT / "output/reverse_index/build_log.json")
    args = ap.parse_args()

    print("[reverse-index] loading FFHQ metrics shards")
    ffhq_metrics = pd.concat(
        [load_ffhq_shard(p) for p in sorted(args.ffhq_metrics_dir.glob("*.pt"))
         if not p.name.endswith(".tmp")],
        ignore_index=True,
    )
    print(f"  ffhq_metrics: {len(ffhq_metrics)} rows")

    print("[reverse-index] loading FFHQ ArcFace shards")
    ffhq_arc = load_ffhq_arcface(args.ffhq_encoded_dir)
    print(f"  ffhq_arcface: {len(ffhq_arc)} rows")

    print("[reverse-index] joining FFHQ metrics + ArcFace on image_sha256")
    ffhq = ffhq_metrics.merge(ffhq_arc, on="image_sha256", how="left")
    print(f"  ffhq joined: {len(ffhq)} rows")

    print("[reverse-index] loading Flux corpus")
    flux = load_flux_corpus(ROOT)
    print(f"  flux_corpus_v3: {len(flux)} rows")

    print("[reverse-index] concatenating")
    unified = pd.concat([flux, ffhq], ignore_index=True, sort=False)
    print(f"  unified: {len(unified)} rows, {len(unified.columns)} cols")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    unified.to_parquet(args.out, index=False)
    print(f"[reverse-index] wrote {args.out} ({args.out.stat().st_size/1e6:.1f} MB)")

    log = {
        "n_total": len(unified),
        "n_ffhq": int((unified["source"] == "ffhq").sum()),
        "n_flux": int((unified["source"] == "flux_corpus_v3").sum()),
        "columns": sorted(unified.columns.tolist()),
        "ffhq_metrics_dir": str(args.ffhq_metrics_dir),
        "ffhq_encoded_dir": str(args.ffhq_encoded_dir),
    }
    with args.log.open("w") as f:
        json.dump(log, f, indent=2)
    print(f"[reverse-index] log -> {args.log}")


if __name__ == "__main__":
    main()
