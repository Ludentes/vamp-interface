from pathlib import Path

import pandas as pd
import pytest
import torch


def test_face_crop_resize_returns_512_rgb():
    """`face_crop_resize` returns (512, 512, 3) uint8 from a 1024² photo."""
    import numpy as np
    from arkit_controlnet.cfm.precompute import face_crop_resize
    photo = (np.random.rand(1024, 1024, 3) * 255).astype(np.uint8)
    out = face_crop_resize(photo, bbox_cx=0.5, bbox_cy=0.5,
                           bbox_w=0.3, bbox_h=0.4, out_size=512)
    assert out.shape == (512, 512, 3) and out.dtype == np.uint8


@pytest.mark.slow
def test_precompute_one_row_round_trip(tmp_path):
    """Precompute one known sha; verify outputs exist with correct shapes."""
    if not Path("output/flame_pose_cache/pose_cache.parquet").exists():
        pytest.skip("pose cache absent")
    if not Path("data/infiniteyou_dl_bf16/infu_flux_v1.0/sim_stage1/"
                "image_proj_model.bin").exists():
        pytest.skip("InfiniteYou resampler weights absent")
    if not Path("/media/newub/Seagate Hub/arc_distill/ffhq_parquet/data").is_dir():
        pytest.skip("Seagate drive not mounted")
    from arkit_controlnet.cfm.precompute import run

    pc = pd.read_parquet("output/flame_pose_cache/pose_cache.parquet",
                         columns=["image_sha256", "pose_detected"])
    candidates = pc[pc.pose_detected].image_sha256.tolist()[:8]

    out_dir = tmp_path / "cfm_precompute"
    run(shas=candidates, out_dir=str(out_dir))

    meta = pd.read_parquet(out_dir / "meta.parquet")
    # At least one of the first few pose-detected shas should also yield an
    # ArcFace embedding; the precompute is per-image best-effort.
    ok = meta[meta.id_ok & meta.pl_ok]
    assert len(ok) >= 1, f"no id+pl rows in {meta}"
    sha = ok.iloc[0].image_sha256
    pl = out_dir / "photo_latents" / f"{sha}.pt"
    it = out_dir / "id_tokens" / f"{sha}.pt"
    assert pl.exists() and it.exists()
    assert torch.load(pl, map_location="cpu").shape == (16, 64, 64)
    assert torch.load(it, map_location="cpu").shape == (8, 4096)
