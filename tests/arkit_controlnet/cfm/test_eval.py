import pytest
import torch
from pathlib import Path


@pytest.mark.slow
@pytest.mark.skipif(
    not Path("output/cfm_precompute/text_embeds.pt").exists()
    or not Path("output/cfm_precompute/meta.parquet").exists(),
    reason="precompute not built")
def test_dump_samples_writes_collage(tmp_path):
    from arkit_controlnet.cfm.model import build_model
    from arkit_controlnet.cfm.eval import dump_samples
    model = build_model()
    out = tmp_path / "cfm_train" / "test_run"
    dump_samples(model, step=0, out_dir=str(out), n=2,
                 sample_steps=4)  # tiny for the smoke
    collage = out / "samples" / "step_000000.png"
    assert collage.exists()
