"""swap_core._crop_region / _feathered_mask -- pure geometry, no models."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from swap_core import _crop_region, _feathered_mask


def test_crop_region_expands_and_centres():
    # 100x100 bbox centred in a 1000x1000 image, 0.45 margin -> half = 100*1.9/2...
    bbox = (450, 450, 550, 550)
    x0, y0, x1, y1 = _crop_region(bbox, (1000, 1000, 3))
    # half = max(100,100) * (0.5 + 0.45) = 95; centre 500
    assert (x0, y0, x1, y1) == (405, 405, 595, 595)


def test_crop_region_clamps_at_border():
    bbox = (10, 10, 60, 60)  # near top-left corner
    x0, y0, x1, y1 = _crop_region(bbox, (200, 200, 3))
    assert x0 == 0 and y0 == 0
    assert 0 <= x1 <= 200 and 0 <= y1 <= 200


def test_feathered_mask_interior_one_edges_zero():
    m = _feathered_mask(100, 100)
    assert m.shape == (100, 100)
    assert m[50, 50] == 1.0
    assert m[0, 0] < 0.5
    assert m.min() >= 0.0 and m.max() <= 1.0


def test_feathered_mask_tiny_region_all_one():
    m = _feathered_mask(4, 4)
    assert np.all(m == 1.0)
