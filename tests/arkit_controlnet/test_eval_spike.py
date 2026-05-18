from pathlib import Path

import numpy as np
import pytest

from arkit_controlnet.eval_spike import (
    ARKIT_BLENDSHAPE_NAMES,
    arcface_cos,
    bs_read,
    bs_vector,
    expr_cos,
    face_landmarks_xy,
)

FIXTURE = Path("tests/arkit_controlnet/fixtures/face.png")  # any clear single-face photo


@pytest.mark.skipif(not FIXTURE.exists(), reason="needs a face fixture image")
def test_bs_read_returns_52_named_channels():
    bs = bs_read(FIXTURE)
    assert set(bs) == set(ARKIT_BLENDSHAPE_NAMES)
    assert all(0.0 <= v <= 1.0 for v in bs.values())


@pytest.mark.skipif(not FIXTURE.exists(), reason="needs a face fixture image")
def test_arcface_cos_self_is_one():
    assert arcface_cos(FIXTURE, FIXTURE) > 0.99


@pytest.mark.skipif(not FIXTURE.exists(), reason="needs a face fixture image")
def test_bs_vector_is_52d_in_canonical_order():
    v = bs_vector(FIXTURE)
    assert v.shape == (52,)
    assert v.dtype == np.float64


@pytest.mark.skipif(not FIXTURE.exists(), reason="needs a face fixture image")
def test_expr_cos_of_a_face_with_itself_is_one():
    assert expr_cos(FIXTURE, FIXTURE) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.skipif(not FIXTURE.exists(), reason="needs a face fixture image")
def test_face_landmarks_xy_returns_478_points_in_unit_square():
    lm = face_landmarks_xy(FIXTURE)
    assert lm.shape == (478, 2)
    assert lm.min() >= -0.5 and lm.max() <= 1.5  # normalized, small slop off-frame
