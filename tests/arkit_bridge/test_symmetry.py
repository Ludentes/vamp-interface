"""Tests for the b_expr horizontal-flip transform."""
import numpy as np

from arkit_bridge.symmetry import flip_b_expr, _BS_PAIRS


def test_flip_is_involution():
    rng = np.random.default_rng(0)
    b = rng.standard_normal(58).astype(np.float32)
    b2 = flip_b_expr(flip_b_expr(b))
    np.testing.assert_allclose(b, b2, atol=1e-6)


def test_pair_swap():
    b = np.zeros(58, dtype=np.float32)
    b[0] = 0.7   # eyeBlinkLeft
    out = flip_b_expr(b)
    assert out[0] == 0.0
    assert out[7] == 0.7   # eyeBlinkRight


def test_eye_yaw_negates_and_swaps():
    b = np.zeros(58, dtype=np.float32)
    b[52] = 0.3   # LeftEyeYaw
    b[55] = -0.1  # RightEyeYaw
    out = flip_b_expr(b)
    assert np.isclose(out[52], 0.1)   # = -(-0.1)
    assert np.isclose(out[55], -0.3)  # = -0.3


def test_eye_pitch_swaps_no_sign():
    b = np.zeros(58, dtype=np.float32)
    b[53] = 0.2; b[56] = 0.4
    out = flip_b_expr(b)
    assert np.isclose(out[53], 0.4)
    assert np.isclose(out[56], 0.2)


def test_symmetric_unchanged():
    """jawOpen, browInnerUp, etc. should be unchanged."""
    b = np.zeros(58, dtype=np.float32)
    b[14] = 0.5  # jawForward
    b[17] = 0.8  # jawOpen
    b[43] = 0.3  # browInnerUp
    b[51] = 0.9  # tongueOut
    out = flip_b_expr(b)
    for idx in [14, 17, 43, 51]:
        assert out[idx] == b[idx], f"idx {idx} changed"


def test_batched_shape():
    rng = np.random.default_rng(1)
    b = rng.standard_normal((4, 58)).astype(np.float32)
    out = flip_b_expr(b)
    assert out.shape == b.shape
    # involution holds
    np.testing.assert_allclose(flip_b_expr(out), b, atol=1e-6)


def test_pairs_are_disjoint():
    seen = set()
    for i, j in _BS_PAIRS:
        assert i not in seen and j not in seen
        seen.add(i); seen.add(j)
