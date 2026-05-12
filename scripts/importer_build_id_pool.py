"""Build a demographic-balanced FFHQ identity pool for the importer bridge.

Stratified sample 20 FFHQ portraits (10M / 10F, varied age and race) with both
ff_detected and ins_detected = True. Resolves sha256 → shard+row via
ffhq_sha_lookup and writes 1024×1024 PNGs to data/importer/identities/.

Usage:
    uv run python scripts/importer_build_id_pool.py [--n 20] [--seed 42]
"""
from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow.compute as pc
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
REVERSE_INDEX = REPO / "output/reverse_index/reverse_index.parquet"
SHA_LOOKUP = REPO / "output/reverse_index/ffhq_sha_lookup.parquet"
OUT_DIR = REPO / "data/importer/identities"


def stratified_sample(n: int, seed: int) -> list[dict]:
    """Pick n FFHQ rows balanced across gender and age, both detectors confirming."""
    t = pq.read_table(
        REVERSE_INDEX,
        columns=[
            "image_sha256", "source",
            "ff_detected", "ins_detected",
            "ff_gender", "ff_age_bin", "ff_race",
        ],
    )
    # FFHQ rows only, both detectors confirm
    mask = pc.and_(
        pc.and_(
            pc.equal(t["source"], "ffhq"),
            pc.equal(t["ff_detected"], True),
        ),
        pc.equal(t["ins_detected"], True),
    )
    t = t.filter(mask)
    rows = t.to_pylist()

    # Stratify: target buckets across gender × age (20-29, 30-39, 40-49)
    import random
    rng = random.Random(seed)
    rng.shuffle(rows)

    target_ages = ["20-29", "30-39", "40-49"]
    target_genders = ["M", "F"]
    per_bucket = max(1, n // (len(target_ages) * len(target_genders)))

    picked: list[dict] = []
    buckets: dict[tuple[str, str], int] = {}
    for r in rows:
        key = (r["ff_gender"], r["ff_age_bin"])
        if r["ff_age_bin"] not in target_ages or r["ff_gender"] not in target_genders:
            continue
        if buckets.get(key, 0) >= per_bucket:
            continue
        picked.append(r)
        buckets[key] = buckets.get(key, 0) + 1
        if len(picked) >= n:
            break

    # If under-quota due to thin buckets, fill from remainder
    if len(picked) < n:
        seen = {r["image_sha256"] for r in picked}
        for r in rows:
            if r["image_sha256"] in seen:
                continue
            picked.append(r)
            if len(picked) >= n:
                break

    return picked[:n]


def load_image_bytes(sha: str, lookup: dict[str, tuple[str, int]]) -> bytes:
    shard_path, row_idx = lookup[sha]
    shard = pq.read_table(shard_path, columns=["image"])
    img_struct = shard["image"][row_idx].as_py()
    return img_struct["bytes"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"[id_pool] stratified-sampling n={args.n}, seed={args.seed}")
    picked = stratified_sample(args.n, args.seed)
    print(f"[id_pool] selected {len(picked)} identities")

    # Build sha → (shard_path, row_idx) lookup
    look_t = pq.read_table(SHA_LOOKUP)
    lookup = {
        s: (p, int(r))
        for s, p, r in zip(
            look_t["image_sha256"].to_pylist(),
            look_t["shard_path"].to_pylist(),
            look_t["row_idx"].to_pylist(),
        )
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_DIR / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id_idx", "filename", "sha256", "gender", "age_bin", "race"])
        for i, r in enumerate(picked):
            sha = r["image_sha256"]
            if sha not in lookup:
                print(f"  [skip] {sha[:8]} not in ffhq_sha_lookup")
                continue
            img_bytes = load_image_bytes(sha, lookup)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            # Ensure 1024x1024; FFHQ is already 1024
            if img.size != (1024, 1024):
                img = img.resize((1024, 1024), Image.LANCZOS)
            fn = f"id_{i:02d}.png"
            img.save(OUT_DIR / fn, "PNG")
            w.writerow([i, fn, sha, r["ff_gender"], r["ff_age_bin"], r["ff_race"]])
            print(f"  [ok] id_{i:02d}  {r['ff_gender']} {r['ff_age_bin']} {r['ff_race'][:12]}")

    print(f"[id_pool] wrote {len(picked)} PNGs + manifest.csv to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
