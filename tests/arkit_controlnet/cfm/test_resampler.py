import torch
from arkit_controlnet.cfm.resampler import Resampler


def test_resampler_io_shapes():
    """InfiniteYou config: 512-d ArcFace → (1, 8, 4096) identity tokens."""
    r = Resampler(dim=1280, depth=4, dim_head=64, heads=20,
                  num_queries=8, embedding_dim=512, output_dim=4096,
                  ff_mult=4)
    x = torch.randn(1, 1, 512)
    out = r(x)
    assert out.shape == (1, 8, 4096)


def test_resampler_loads_pretrained_weights():
    """The released InfiniteYou image_proj_model.bin loads cleanly."""
    import os
    p = "data/infiniteyou_dl_bf16/infu_flux_v1.0/sim_stage1/image_proj_model.bin"
    if not os.path.exists(p):
        import pytest
        pytest.skip("InfiniteYou weights not downloaded")
    sd = torch.load(p, map_location="cpu", weights_only=False)
    r = Resampler(dim=1280, depth=4, dim_head=64, heads=20,
                  num_queries=8, embedding_dim=512, output_dim=4096,
                  ff_mult=4)
    r.load_state_dict(sd["image_proj"])
