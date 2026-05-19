from pathlib import Path

import numpy as np

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
