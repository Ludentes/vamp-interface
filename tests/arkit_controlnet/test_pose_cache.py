import numpy as np
import pytest
from arkit_controlnet.build_pose_cache import pose_from_image


def test_pose_from_image_on_a_known_face():
    """A materialized FFHQ portrait yields a rotation and a plausible bbox."""
    import glob
    from PIL import Image
    pngs = sorted(glob.glob("output/ffhq_images/*.png"))
    if not pngs:
        pytest.skip("no materialized FFHQ images")
    png = pngs[0]
    arr = np.asarray(Image.open(png).convert("RGB"))
    rot, bbox, detected = pose_from_image(arr)
    assert detected is True
    assert rot.shape == (3, 3)
    assert np.isclose(np.linalg.det(rot), 1.0, atol=0.1)   # proper rotation
    cx, cy, bw, bh = bbox
    assert 0.0 < bw < 1.0 and 0.0 < bh < 1.0
    assert 0.0 < cx < 1.0 and 0.0 < cy < 1.0


def test_pose_from_image_no_face():
    rot, bbox, detected = pose_from_image(np.zeros((256, 256, 3), np.uint8))
    assert detected is False
