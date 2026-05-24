import cv2
import numpy as np

from group_photobooth import Placement
from group_photobooth.composite import composite, lab_match, drop_shadow


def _solid_doll(h, w, color):
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., :3] = color
    rgba[..., 3] = 255
    return rgba


def test_composite_pastes_at_foot_anchor():
    bg = np.full((600, 800, 3), 128, np.uint8)
    doll = _solid_doll(400, 200, (200, 100, 50))
    placement = Placement(x_center=400, y_bottom=600, height=400, z_order=0)
    out = composite(bg, [doll], [placement])
    assert out.shape == bg.shape
    assert (out[200:600, 300:500] != 128).all()
    assert (out[:200, :] == 128).all()


def test_painter_order_front_doll_occludes_back_doll():
    bg = np.full((600, 800, 3), 0, np.uint8)
    back = _solid_doll(400, 200, (255, 0, 0))
    front = _solid_doll(400, 200, (0, 255, 0))
    placements = [
        Placement(x_center=400, y_bottom=400, height=400, z_order=0),
        Placement(x_center=400, y_bottom=600, height=400, z_order=1),
    ]
    out = composite(bg, [back, front], placements)
    overlap_px = out[300:400, 300:500]
    assert (overlap_px[..., 1] > overlap_px[..., 0]).mean() > 0.95


def test_lab_match_shifts_mean_toward_target():
    src = np.full((100, 100, 3), [50, 100, 200], np.uint8)
    alpha = np.full((100, 100), 255, np.uint8)
    target_lab = (60.0, 0.0, 0.0)
    out = lab_match(src, alpha, target_lab, max_dL=8.0)
    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
    out_lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    dL_src = src_lab[..., 0].mean() - target_lab[0]
    dL_out = out_lab[..., 0].mean() - target_lab[0]
    assert abs(dL_out) < abs(dL_src)


def test_drop_shadow_darkens_area_under_doll():
    bg = np.full((600, 800, 3), 200, np.uint8)
    placement = Placement(x_center=400, y_bottom=400, height=200, z_order=0)
    out = drop_shadow(bg, placement, doll_w=100, opacity=0.5,
                      offset_yx=(8, 0), blur_px=12)
    foot = out[395:410, 380:420].mean()
    corner = out[0:30, 0:30].mean()
    assert foot < corner
