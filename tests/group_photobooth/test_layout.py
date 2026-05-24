import numpy as np

from group_photobooth import Person, Placement
from group_photobooth.layout import solve_placements


def _person(x1, y1, x2, y2, photo_hw):
    h, w = photo_hw
    m = np.zeros((h, w), np.uint8)
    m[y1:y2, x1:x2] = 255
    return Person(body_bbox=(x1, y1, x2, y2), body_mask=m,
                  face_bbox=(x1 + 5, y1 + 5, x1 + 25, y1 + 25))


def test_painter_order_sorts_back_to_front():
    photo_hw = (600, 800)
    bg_hw = (600, 800)
    persons = [
        _person(100, 100, 250, 400, photo_hw),
        _person(450, 200, 600, 500, photo_hw),
    ]
    placements = solve_placements(persons, photo_hw, bg_hw)
    assert len(placements) == 2
    assert placements[0].z_order < placements[1].z_order
    assert placements[0].y_bottom < placements[1].y_bottom


def test_letterbox_scales_correctly():
    photo_hw = (800, 400)
    bg_hw = (800, 800)
    persons = [_person(100, 200, 200, 600, photo_hw)]
    placements = solve_placements(persons, photo_hw, bg_hw)
    assert len(placements) == 1
    p = placements[0]
    assert p.x_center == 350
    assert p.y_bottom == 600
    assert p.height == 400


def test_empty_persons_returns_empty_placements():
    assert solve_placements([], (600, 800), (600, 800)) == []
