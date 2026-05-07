"""Smoke-only test: import path, signature, and provider-mode argument validation."""
import inspect

import numpy as np
import pytest

from arkit_bridge.seam_install import _provider_fetch, install_arkit_seams


def test_signature():
    sig = inspect.signature(install_arkit_seams)
    params = list(sig.parameters.keys())
    assert "pipe" in params
    assert "b_seq" in params
    assert "ypr_seq" in params
    assert "student" in params
    assert "provider" in params  # new keyword for streaming mode


def test_provider_and_arrays_mutually_exclusive():
    """Array-mode and provider-mode are mutually exclusive at install time.

    We can't actually install (would need a Pose2VideoPipeline_Stream), but
    the input-mode assertions fire before any pipe attribute access, so
    passing pipe=None is enough to exercise them.
    """
    # Provider mode: b_seq/ypr_seq must be None.
    with pytest.raises(AssertionError):
        install_arkit_seams(
            pipe=None, b_seq=object(), ypr_seq=object(),
            student=None, device="cpu", dtype=None,
            provider=lambda start, n: (None, None),
        )
    # Array mode: b_seq and ypr_seq must both be supplied.
    with pytest.raises(AssertionError):
        install_arkit_seams(
            pipe=None, b_seq=None, ypr_seq=None,
            student=None, device="cpu", dtype=None,
        )


def _make_array_provider():
    """Return (provider, b_full, ypr_full) where provider serves contiguous
    (start, n) windows from full arrays. Lets us cross-check provider-mode
    fetch against array-mode b_seq[indices] / ypr_seq[indices]."""
    rng = np.random.default_rng(0)
    b_full = rng.standard_normal((32, 58)).astype(np.float32)
    ypr_full = rng.standard_normal((32, 3)).astype(np.float32)

    calls = []

    def provider(start, n):
        calls.append((start, n))
        return b_full[start:start + n], ypr_full[start:start + n]

    return provider, b_full, ypr_full, calls


def test_provider_fetch_contiguous():
    provider, b, ypr, calls = _make_array_provider()
    idx = np.arange(5, 13)
    b_out, y_out = _provider_fetch(provider, idx)
    np.testing.assert_array_equal(b_out, b[idx])
    np.testing.assert_array_equal(y_out, ypr[idx])
    assert calls == [(5, 8)]


def test_provider_fetch_single():
    provider, b, ypr, calls = _make_array_provider()
    idx = np.array([7])
    b_out, y_out = _provider_fetch(provider, idx)
    np.testing.assert_array_equal(b_out, b[idx])
    np.testing.assert_array_equal(y_out, ypr[idx])
    assert calls == [(7, 1)]


def test_provider_fetch_leading_zero_pad():
    """Pattern [0]*pad_k + [0, 1, ..., m-1] used by patched_interpolate_kps_online."""
    provider, b, ypr, calls = _make_array_provider()
    pad_k = 2
    m = 4
    idx = np.array([0] * pad_k + list(range(m)))  # [0, 0, 0, 1, 2, 3]
    b_out, y_out = _provider_fetch(provider, idx)
    expected_b = np.concatenate([np.repeat(b[:1], pad_k, axis=0), b[:m]], axis=0)
    expected_y = np.concatenate([np.repeat(ypr[:1], pad_k, axis=0), ypr[:m]], axis=0)
    np.testing.assert_array_equal(b_out, expected_b)
    np.testing.assert_array_equal(y_out, expected_y)


def test_provider_fetch_zero_pad_one():
    """num_interp=2 → pad_k=1: [0, 0, 1, 2, ...] still maps correctly."""
    provider, b, ypr, calls = _make_array_provider()
    idx = np.array([0, 0, 1, 2, 3])
    b_out, y_out = _provider_fetch(provider, idx)
    expected = np.concatenate([b[:1], b[:4]], axis=0)
    np.testing.assert_array_equal(b_out, expected)


def test_provider_fetch_no_pad_arange_from_zero():
    """num_interp=1 → no pad, just [0, 1, 2, ...]; takes the contiguous fast path."""
    provider, b, ypr, calls = _make_array_provider()
    idx = np.arange(0, 5)
    b_out, y_out = _provider_fetch(provider, idx)
    np.testing.assert_array_equal(b_out, b[:5])
    assert calls == [(0, 5)]


def test_provider_fetch_unsupported_pattern():
    """Anything other than contiguous-arange or leading-zero-pad is rejected."""
    provider, _, _, _ = _make_array_provider()
    with pytest.raises(AssertionError):
        _provider_fetch(provider, np.array([3, 1, 4, 1, 5]))
