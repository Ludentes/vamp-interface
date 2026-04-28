"""Targeted FFHQ image fetcher (run on the machine that has the shards).

Reads `needed_shas.txt` (one sha256 per line), scans HuggingFace-format
FFHQ parquet shards (each row a dict ``{bytes, path}`` in column
``image``), decodes only the matching rows, resizes to 512x512 LANCZOS
to match the extractor convention used everywhere else in this repo,
and writes ``<out_dir>/<sha>.png``.

Usage on PC 25 (videocard@192.168.87.25):

    python fetch_ffhq_pngs.py \
        --shards-dir /path/to/ffhq/shards \
        --shas needed_shas.txt \
        --out-dir ffhq_images \
        --resolution 512

After it finishes, rsync ``ffhq_images/`` back to this box at
``output/ffhq_images/``.

Stdlib + pyarrow + Pillow only. No GPU, no torch. Stop-and-resume
safe: existing PNGs are skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards-dir", type=Path, required=True,
                    help="Directory containing HF FFHQ parquet shards (column 'image' = {bytes, path}).")
    ap.add_argument("--shas", type=Path, required=True,
                    help="Text file with one image_sha256 per line.")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Output dir for PNGs (one per sha).")
    ap.add_argument("--resolution", type=int, default=512,
                    help="Resize to (R, R) using LANCZOS (matches encode_ffhq.py + reverse_index).")
    ap.add_argument("--shard-glob", default="*.parquet")
    ap.add_argument("--limit-shards", type=int, default=0, help="0 = all shards")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    wanted = {s.strip() for s in args.shas.read_text().splitlines() if s.strip()}
    if not wanted:
        print(f"[fatal] {args.shas} is empty", file=sys.stderr)
        return 1
    print(f"[fetch] need {len(wanted)} unique sha256s")

    # Skip already-fetched shas to support resume.
    existing = {p.stem for p in args.out_dir.glob("*.png")}
    todo = wanted - existing
    print(f"[fetch] {len(existing)} already on disk, {len(todo)} remaining")
    if not todo:
        print("[fetch] nothing to do")
        return 0

    shards = sorted(args.shards_dir.glob(args.shard_glob))
    if args.limit_shards:
        shards = shards[: args.limit_shards]
    if not shards:
        print(f"[fatal] no shards matched {args.shards_dir}/{args.shard_glob}", file=sys.stderr)
        return 1
    print(f"[fetch] scanning {len(shards)} shard(s)")

    found = 0
    t0 = time.time()
    for s_idx, shard in enumerate(shards, 1):
        if not todo:
            break
        try:
            table = pq.read_table(shard, columns=["image"])
        except Exception as e:
            print(f"[warn] {shard.name}: cannot read 'image' column ({e}); skipping")
            continue
        rows = table.column("image").to_pylist()
        n_match_shard = 0
        for row in rows:
            if row is None:
                continue
            png_bytes = row.get("bytes") if isinstance(row, dict) else None
            if not png_bytes:
                continue
            sha = sha256_hex(png_bytes)
            if sha not in todo:
                continue
            try:
                img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
                img = img.resize((args.resolution, args.resolution), Image.Resampling.LANCZOS)
                img.save(args.out_dir / f"{sha}.png", format="PNG", optimize=False)
            except Exception as e:
                print(f"[warn] decode/save failed for {sha[:12]}: {e}")
                continue
            todo.discard(sha)
            found += 1
            n_match_shard += 1
            if not todo:
                break
        elapsed = time.time() - t0
        print(f"[{s_idx}/{len(shards)}] {shard.name}: matched {n_match_shard} "
              f"(total {found}/{len(wanted)} in {elapsed:.1f}s, {len(todo)} remaining)")

    if todo:
        miss = sorted(todo)
        miss_path = args.out_dir / "_missing.txt"
        miss_path.write_text("\n".join(miss) + "\n")
        print(f"[done] fetched {found}/{len(wanted)}; {len(todo)} not found "
              f"(written to {miss_path})", file=sys.stderr)
        return 2
    print(f"[done] fetched {found}/{len(wanted)} in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
