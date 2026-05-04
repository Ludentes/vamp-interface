"""Extract 1152-d SigLIP-2 image embeddings keyed by image_sha256.

Standalone (no MiVOLO/FairFace/InsightFace/MediaPipe) — only loads SigLIP-2 SO400M.
Designed to run on the shard against the same image sources `extract_ffhq_metrics.py`
and `extract_grid_to_reverse_index.py` use, then have its sidecar parquet pulled
locally and merged into reverse_index.parquet.

Three source modes:

  --source ffhq           --shards-dir <dir of HF parquets with `image:{bytes,path}`>
  --source flux_corpus_v3 --sample-index <models/blendshape_nmf/sample_index.parquet>
  --source flux_solver_a  --scores-parquet <output/.../solver_a_squint_grid/scores.parquet>

All modes write the same sidecar schema, appending to --out-parquet:

    image_sha256: str
    source:       str        (mode tag, for audit)
    siglip_img_emb_fp16: ndarray (1152,) fp16, L2-normed

Resumable: any image_sha256 already present in --out-parquet is skipped.
The script flushes every --flush-every rows so a kill mid-run loses at most that many.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from PIL import Image


SIGLIP2_MODEL = "google/siglip2-so400m-patch16-384"
EMB_DIM = 1152


def compute_sha256(png_bytes: bytes) -> str:
    return hashlib.sha256(png_bytes).hexdigest()


def load_siglip(device: str):
    from transformers import AutoModel, AutoProcessor
    model = AutoModel.from_pretrained(SIGLIP2_MODEL).to(device).eval()
    processor = AutoProcessor.from_pretrained(SIGLIP2_MODEL)
    return model, processor


def _as_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    for attr in ("pooler_output", "last_hidden_state"):
        v = getattr(x, attr, None)
        if isinstance(v, torch.Tensor):
            return v
    raise TypeError(f"cannot extract tensor from {type(x)}")


@torch.no_grad()
def encode_batch(model, processor, pil_images: list[Image.Image], device: str) -> np.ndarray:
    inputs = processor(images=pil_images, return_tensors="pt").to(device)
    feat = _as_tensor(model.get_image_features(**inputs))
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.to(torch.float16).cpu().numpy()


def existing_shas(out_parquet: Path) -> set[str]:
    if not out_parquet.exists():
        return set()
    df = pd.read_parquet(out_parquet, columns=["image_sha256"])
    return set(df["image_sha256"].tolist())


def append_rows(out_parquet: Path, rows: list[dict]) -> None:
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    if out_parquet.exists():
        existing = pd.read_parquet(out_parquet)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_parquet.with_suffix(".tmp.parquet")
    combined.to_parquet(tmp_path, index=False)
    tmp_path.replace(out_parquet)


def iter_ffhq_shards(shards_dir: Path):
    """Yield (sha, PIL.Image) from HF-format parquet shards (column `image: {bytes, path}`).

    Hashes the raw PNG bytes (matching extract_ffhq_metrics' compute_image_sha256).
    """
    for shard_path in sorted(shards_dir.glob("*.parquet")):
        table = pq.read_table(shard_path, columns=["image"])
        col = table.column("image").to_pylist()
        for entry in col:
            png_bytes = entry["bytes"]
            sha = compute_sha256(png_bytes)
            img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
            yield sha, img, shard_path.name


def iter_flux_corpus_v3(sample_index: Path, repo_root: Path):
    """Yield (sha, PIL.Image) from sample_index.parquet which has image_sha256 + img_path."""
    df = pd.read_parquet(sample_index, columns=["image_sha256", "img_path"])
    df = df[df["image_sha256"].notna() & df["img_path"].notna()]
    for _, row in df.iterrows():
        sha = str(row["image_sha256"])
        path = repo_root / str(row["img_path"])
        if not path.exists():
            continue
        png_bytes = path.read_bytes()
        # Sanity: confirm hash matches; sample_index SHAs were computed on PNG bytes.
        actual = compute_sha256(png_bytes)
        if actual != sha:
            # Some samples may have been re-saved; trust the on-disk hash.
            sha = actual
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        yield sha, img, path.name


def iter_flux_solver_a(scores_parquet: Path):
    """Yield (sha, PIL.Image) from solver_a_squint_grid/scores.parquet (anchor_png/edit_png)."""
    df = pd.read_parquet(scores_parquet)
    paths: list[Path] = []
    for _, row in df.iterrows():
        for col in ("anchor_png", "edit_png"):
            paths.append(Path(str(row[col])))
    for p in paths:
        if not p.exists():
            continue
        png_bytes = p.read_bytes()
        sha = compute_sha256(png_bytes)
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        yield sha, img, p.name


def run(args, log: logging.Logger) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"device={device}")
    log.info(f"loading {SIGLIP2_MODEL}...")
    model, processor = load_siglip(device)

    seen = existing_shas(args.out_parquet)
    log.info(f"resume: {len(seen)} SHAs already in {args.out_parquet}")

    if args.source == "ffhq":
        if not args.shards_dir:
            raise SystemExit("--shards-dir required for source=ffhq")
        gen = iter_ffhq_shards(args.shards_dir)
    elif args.source == "flux_corpus_v3":
        if not args.sample_index:
            raise SystemExit("--sample-index required for source=flux_corpus_v3")
        gen = iter_flux_corpus_v3(args.sample_index, args.repo_root)
    elif args.source == "flux_solver_a":
        if not args.scores_parquet:
            raise SystemExit("--scores-parquet required for source=flux_solver_a")
        gen = iter_flux_solver_a(args.scores_parquet)
    else:
        raise SystemExit(f"unknown source {args.source}")

    pending: list[Image.Image] = []
    pending_shas: list[str] = []
    pending_origins: list[str] = []
    buffered_rows: list[dict] = []
    n_processed = 0
    n_skipped = 0
    t0 = time.time()

    def flush_batch():
        nonlocal pending, pending_shas, pending_origins, buffered_rows, n_processed
        if not pending:
            return
        feats = encode_batch(model, processor, pending, device)
        for sha, feat in zip(pending_shas, feats):
            buffered_rows.append({
                "image_sha256": sha,
                "source": args.source,
                "siglip_img_emb_fp16": feat,
            })
            seen.add(sha)
        n_processed += len(pending)
        pending = []
        pending_shas = []
        pending_origins = []

    def flush_disk():
        nonlocal buffered_rows
        if not buffered_rows:
            return
        append_rows(args.out_parquet, buffered_rows)
        log.info(f"  flushed {len(buffered_rows)} rows to {args.out_parquet} "
                 f"(total processed={n_processed}, skipped={n_skipped}, "
                 f"rate={n_processed / max(time.time() - t0, 1e-3):.1f} img/s)")
        buffered_rows = []

    for sha, img, origin in gen:
        if sha in seen:
            n_skipped += 1
            continue
        pending.append(img)
        pending_shas.append(sha)
        pending_origins.append(origin)
        if len(pending) >= args.batch_size:
            flush_batch()
        if len(buffered_rows) >= args.flush_every:
            flush_disk()

    flush_batch()
    flush_disk()
    log.info(f"done: processed={n_processed}, skipped={n_skipped}, "
             f"elapsed={time.time() - t0:.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    choices=["ffhq", "flux_corpus_v3", "flux_solver_a"])
    ap.add_argument("--shards-dir", type=Path,
                    help="HF-format parquet dir (source=ffhq)")
    ap.add_argument("--sample-index", type=Path,
                    help="sample_index.parquet (source=flux_corpus_v3)")
    ap.add_argument("--scores-parquet", type=Path,
                    help="solver_a scores parquet (source=flux_solver_a)")
    ap.add_argument("--out-parquet", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, default=Path.cwd(),
                    help="root for resolving relative img_path entries")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--flush-every", type=int, default=512)
    ap.add_argument("--log", type=Path)
    args = ap.parse_args()

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log, mode="a"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=handlers)
    log = logging.getLogger("siglip_extract")
    log.info("=" * 60)
    log.info(f"start source={args.source} out={args.out_parquet}")
    run(args, log)


if __name__ == "__main__":
    main()
