"""Build the image_sha256 -> (shard_idx, row_idx) index over FFHQ-70k.

The FFHQ-70000 dataset lives as 190 HuggingFace parquet shards on the Seagate
drive; reverse_index keys rows on image_sha256 but the shards carry no sha.
This index is the join key. Resumable per shard. Run once:
    PYTHONPATH=src /home/newub/miniconda3/bin/python -m arkit_controlnet.build_ffhq_index

sha convention (resolved empirically 2026-05-18): the parquet stores
already-PIL-canonical PNG bytes — sha256(raw bytes) == sha256(re-encoded PNG)
for all sampled rows. Hashing the raw stored bytes is therefore canonical and
matches reverse_index 200/200 on the first shard.
"""
import glob
import hashlib
from pathlib import Path

import pandas as pd

SHARD_GLOB = "/media/newub/Seagate Hub/arc_distill/ffhq_parquet/data/train-*.parquet"
OUT = Path("output/ffhq_index/ffhq_sha_index.parquet")


def canonical_sha(image_bytes: bytes) -> str:
    """sha256 of the parquet-stored image bytes (the project's sha convention)."""
    return hashlib.sha256(image_bytes).hexdigest()


def build() -> None:
    shards = sorted(glob.glob(SHARD_GLOB))
    if not shards:
        raise FileNotFoundError(
            f"no FFHQ shards at {SHARD_GLOB} — is the Seagate drive mounted?")
    OUT.parent.mkdir(parents=True, exist_ok=True)

    done_shards: set[int] = set()
    if OUT.exists():
        done_shards = set(pd.read_parquet(OUT, columns=["shard_idx"])["shard_idx"])

    rows = []
    for shard_idx, shard in enumerate(shards):
        if shard_idx in done_shards:
            continue
        df = pd.read_parquet(shard, columns=["image"])
        for row_idx, cell in enumerate(df["image"]):
            rows.append({
                "image_sha256": canonical_sha(cell["bytes"]),
                "shard_idx": shard_idx,
                "row_idx": row_idx,
            })
        print(f"shard {shard_idx}/{len(shards)} done")
    if not rows:
        print("index already complete")
        return
    new = pd.DataFrame(rows)
    if OUT.exists():
        new = pd.concat([pd.read_parquet(OUT), new], ignore_index=True)
    new.to_parquet(OUT)
    print(f"wrote {len(new)} rows to {OUT}")


if __name__ == "__main__":
    build()
