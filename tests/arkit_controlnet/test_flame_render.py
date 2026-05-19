import numpy as np
from arkit_controlnet.flame_render import (
    BASIS_CHANNEL_NAMES, mediapipe_to_basis_vector,
)
from arkit_controlnet.eval_spike import ARKIT_BLENDSHAPE_NAMES


def test_basis_channel_names_are_52_sorted_arkit():
    assert len(BASIS_CHANNEL_NAMES) == 52
    assert BASIS_CHANNEL_NAMES == sorted(BASIS_CHANNEL_NAMES)
    assert "tongueOut" in BASIS_CHANNEL_NAMES
    assert "_neutral" not in BASIS_CHANNEL_NAMES


def test_basis_covers_all_mediapipe_expression_names():
    mp_expr = [n for n in ARKIT_BLENDSHAPE_NAMES if n != "_neutral"]
    assert set(mp_expr).issubset(set(BASIS_CHANNEL_NAMES))


def test_mediapipe_to_basis_vector_places_jawopen():
    """A unit jawOpen in MediaPipe order lands on the basis's jawOpen channel."""
    mp = {n: 0.0 for n in ARKIT_BLENDSHAPE_NAMES}
    mp["jawOpen"] = 1.0
    vec = mediapipe_to_basis_vector(mp)
    assert vec.shape == (52,)
    j = BASIS_CHANNEL_NAMES.index("jawOpen")
    assert vec[j] == 1.0
    assert np.count_nonzero(vec) == 1
