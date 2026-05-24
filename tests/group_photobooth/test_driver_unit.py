import numpy as np

from group_photobooth import Person
from group_photobooth.driver import _crop_face, _seed_for_face


def test_seed_for_face_is_deterministic():
    s1 = _seed_for_face("photo_a.png", 0)
    s2 = _seed_for_face("photo_a.png", 0)
    assert s1 == s2
    s3 = _seed_for_face("photo_a.png", 1)
    assert s3 != s1
    s4 = _seed_for_face("photo_b.png", 0)
    assert s4 != s1


def test_crop_face_returns_bgr_with_margin():
    img = np.full((400, 600, 3), 128, np.uint8)
    person = Person(body_bbox=(100, 50, 300, 380),
                    body_mask=np.zeros((400, 600), np.uint8),
                    face_bbox=(180, 90, 240, 160))
    crop = _crop_face(img, person, margin_frac=0.3)
    assert crop.ndim == 3 and crop.shape[2] == 3
    fw, fh = 240 - 180, 160 - 90
    assert crop.shape[0] > fh
    assert crop.shape[1] > fw
