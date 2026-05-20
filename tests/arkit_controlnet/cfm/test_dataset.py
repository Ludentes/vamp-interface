import pytest
import torch
from pathlib import Path


@pytest.fixture(scope="module")
def cached_sha():
    if not Path("output/cfm_precompute/meta.parquet").exists():
        pytest.skip("precompute cache not built")
    import pandas as pd
    meta = pd.read_parquet("output/cfm_precompute/meta.parquet")
    ok = meta[(meta.vae_ok) & (meta.id_ok)]
    if ok.empty:
        pytest.skip("no fully-precomputed rows")
    return ok.iloc[0].image_sha256


def test_split_assigns_each_sha_to_exactly_one():
    from arkit_controlnet.cfm.dataset import CfmPairDataset
    eval_ds = CfmPairDataset(split="eval", eval_size=1024)
    train_ds = CfmPairDataset(split="train", eval_size=1024)
    eval_shas = set(eval_ds.df.image_sha256)
    train_shas = set(train_ds.df.image_sha256)
    assert not (eval_shas & train_shas)
    assert len(eval_shas) > 0 and len(train_shas) > 0


def test_split_is_deterministic():
    from arkit_controlnet.cfm.dataset import CfmPairDataset
    a = CfmPairDataset(split="eval", eval_size=1024)
    b = CfmPairDataset(split="eval", eval_size=1024)
    assert list(a.df.image_sha256) == list(b.df.image_sha256)


def test_getitem_shapes(cached_sha):
    from arkit_controlnet.cfm.dataset import CfmPairDataset
    ds = CfmPairDataset(split="train", eval_size=1024)
    ds.df = ds.df[ds.df.image_sha256 == cached_sha].reset_index(drop=True)
    if len(ds.df) == 0:
        ds = CfmPairDataset(split="eval", eval_size=1024)
        ds.df = ds.df[ds.df.image_sha256 == cached_sha].reset_index(drop=True)
    item = ds[0]
    assert item["photo_latent"].shape == (16, 64, 64)
    assert item["id_tokens"].shape == (8, 4096)
    assert item["control_rgb"].shape == (3, 512, 512)
    assert item["control_rgb"].dtype == torch.float32
    assert item["sha"] == cached_sha


def test_bs_column_order_matches_basis():
    """The bs_* column order must align with BASIS_CHANNEL_NAMES (minus tongueOut)."""
    from arkit_controlnet.cfm.dataset import BS_COLUMNS
    from arkit_controlnet.flame_render import BASIS_CHANNEL_NAMES
    expected = [f"bs_{n}" for n in BASIS_CHANNEL_NAMES if n != "tongueOut"]
    assert BS_COLUMNS == expected
