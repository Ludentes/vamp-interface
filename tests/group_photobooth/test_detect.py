import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from group_photobooth import Person

FIXTURES = Path(__file__).parent / "fixtures"
COMFY_URL = os.environ.get("COMFY_URL")
needs_comfy = pytest.mark.skipif(not COMFY_URL,
                                 reason="set COMFY_URL=http://... to run")


@needs_comfy
def test_detect_single_person_returns_one_person():
    from group_photobooth.detect import detect_people
    img = cv2.imread(str(FIXTURES / "single_person.png"))
    assert img is not None
    persons = detect_people(img, comfy_url=COMFY_URL)
    assert len(persons) == 1
    p = persons[0]
    assert isinstance(p, Person)
    x1, y1, x2, y2 = p.body_bbox
    assert 0 <= x1 < x2 <= img.shape[1]
    assert 0 <= y1 < y2 <= img.shape[0]
    assert p.body_mask.shape == img.shape[:2]
    assert p.body_mask.dtype.name == "uint8"
    crop = p.body_mask[y1:y2, x1:x2]
    assert (crop > 0).mean() > 0.10
    assert p.face_bbox is not None


@needs_comfy
def test_detect_returns_empty_on_no_person():
    from group_photobooth.detect import detect_people
    img = np.full((400, 400, 3), 128, np.uint8)
    persons = detect_people(img, comfy_url=COMFY_URL)
    assert persons == []
