import pandas as pd
from arkit_controlnet.build_ffhq_index import canonical_sha, SHARD_GLOB


def test_canonical_sha_matches_reverse_index():
    """The first shard's images hash to shas present in reverse_index."""
    import glob
    ri = set(pd.read_parquet("output/reverse_index/reverse_index.parquet",
                             columns=["image_sha256"])["image_sha256"])
    shard = sorted(glob.glob(SHARD_GLOB))[0]
    df = pd.read_parquet(shard)
    hits = sum(canonical_sha(cell["bytes"]) in ri for cell in df["image"].iloc[:100])
    assert hits >= 95, f"only {hits}/100 shard images matched reverse_index"
