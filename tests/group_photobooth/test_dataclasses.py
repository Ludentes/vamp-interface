from group_photobooth import Person, Placement
import numpy as np

def test_person_dataclass():
    p = Person(body_bbox=(10, 20, 100, 200),
               body_mask=np.zeros((300, 200), dtype=np.uint8),
               face_bbox=(30, 25, 70, 75))
    assert p.body_bbox == (10, 20, 100, 200)
    assert p.body_mask.shape == (300, 200)
    assert p.face_bbox == (30, 25, 70, 75)

def test_placement_dataclass():
    pl = Placement(x_center=512, y_bottom=900, height=400, z_order=0)
    assert pl.x_center == 512
    assert pl.height == 400
