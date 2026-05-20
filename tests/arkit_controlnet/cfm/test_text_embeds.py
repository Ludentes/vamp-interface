import os
import torch
from arkit_controlnet.cfm.text_embeds import build_text_embeds, FIXED_PROMPT

OUT = "output/cfm_precompute/text_embeds.pt"


def test_fixed_prompt_is_one_string():
    assert isinstance(FIXED_PROMPT, str) and len(FIXED_PROMPT) > 10


def test_build_text_embeds_writes_expected_shapes(tmp_path):
    target = tmp_path / "text_embeds.pt"
    build_text_embeds(out_path=str(target))
    d = torch.load(target, map_location="cpu")
    assert d["t5"].shape == (1, 512, 4096)
    assert d["pooled"].shape == (1, 768)
    assert d["t5"].dtype == torch.bfloat16
    assert d["pooled"].dtype == torch.bfloat16


def test_build_text_embeds_skips_if_exists(tmp_path):
    target = tmp_path / "text_embeds.pt"
    target.write_bytes(b"sentinel")
    build_text_embeds(out_path=str(target))
    assert target.read_bytes() == b"sentinel"
