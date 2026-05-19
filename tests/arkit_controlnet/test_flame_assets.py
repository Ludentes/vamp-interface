from pathlib import Path

import numpy as np
import pytest

from arkit_controlnet.prep_flame_assets import _install_chumpy_shim, _to_array

NPZ = Path("output/flame_assets/flame_base.npz")
BASIS = Path("output/flame_assets/flame_arkit_bs.npy")


def test_flame_base_shapes():
    d = np.load(NPZ)
    assert d["v_template"].shape == (5023, 3)
    assert d["v_template"].dtype == np.float32
    assert d["faces"].ndim == 2 and d["faces"].shape[1] == 3
    assert int(d["faces"].max()) < 5023            # face indices in range


def test_basis_copied():
    b = np.load(BASIS)
    assert b.shape == (52, 5023, 3)


def test_to_array_passes_through_plain_ndarray():
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert _to_array(arr) is not None
    np.testing.assert_array_equal(_to_array(arr), arr)


def test_ch_stub_materialises_value_from_pickled_state():
    """A Ch stub with pickled-style state exposes its array via `.r`, and
    `_to_array` recovers it. Covers the chumpy-leaf unpickle path."""
    _install_chumpy_shim()
    import chumpy as ch_mod

    arr = np.arange(9, dtype=np.float64).reshape(3, 3)
    leaf = ch_mod.Ch.__new__(ch_mod.Ch)
    leaf.__setstate__({"x": arr})
    np.testing.assert_array_equal(_to_array(leaf), arr)


def test_ch_stub_raises_clear_error_without_x():
    """A Ch stub lacking the documented `x` leaf attribute raises a specific
    error rather than silently guessing."""
    _install_chumpy_shim()
    import chumpy as ch_mod

    leaf = ch_mod.Ch.__new__(ch_mod.Ch)
    leaf.__setstate__({"some_other_key": 1})
    with pytest.raises(AttributeError, match="no `x` attribute"):
        _ = leaf.r
