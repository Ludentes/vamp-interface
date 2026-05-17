from pathlib import Path

import pytest

from arkit_controlnet.eval_spike import ARKIT_BLENDSHAPE_NAMES, arcface_cos, bs_read

FIXTURE = Path("tests/arkit_controlnet/fixtures/face.png")  # any clear single-face photo


@pytest.mark.skipif(not FIXTURE.exists(), reason="needs a face fixture image")
def test_bs_read_returns_52_named_channels():
    bs = bs_read(FIXTURE)
    assert set(bs) == set(ARKIT_BLENDSHAPE_NAMES)
    assert all(0.0 <= v <= 1.0 for v in bs.values())


@pytest.mark.skipif(not FIXTURE.exists(), reason="needs a face fixture image")
def test_arcface_cos_self_is_one():
    assert arcface_cos(FIXTURE, FIXTURE) > 0.99
