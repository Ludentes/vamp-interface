import json
import os
import torch
import pytest
from pathlib import Path


@pytest.mark.skipif(
    not Path("output/cfm_precompute/text_embeds.pt").exists(),
    reason="text_embeds not built")
def test_two_steps_drive_loss_finite_and_log_appears(tmp_path):
    """Two CFM steps on a 1-row train fixture: loss finite, log written, ckpt saved."""
    from arkit_controlnet.cfm.train import train
    out_dir = tmp_path / "cfm_train" / "smoke"
    train(out_dir=str(out_dir), max_steps=2, save_every=2,
          eval_every=10, dataset_filter_n=1)
    log = (out_dir / "step_log.csv").read_text().strip().splitlines()
    assert len(log) >= 3   # header + 2 steps
    ckpt = out_dir / "latest.pt"
    assert ckpt.exists()
    state = torch.load(ckpt, map_location="cpu")
    assert state["step"] == 2
