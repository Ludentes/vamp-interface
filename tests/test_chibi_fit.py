import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import torch
from chibi.fit import fit_chibi_field, chibi_loss
from chibi.field import ChibiField
from chibi.landmarks import landmark_lines, region_falloff_weights, FLAME_TEMPLATE

MASKS = "/home/newub/w/LAM/model_zoo/human_parametric_models/flame_assets/flame/FLAME_masks.pkl"


def _template_verts():
    verts = []
    for L in pathlib.Path(FLAME_TEMPLATE).read_text().splitlines():
        if L.startswith("v "):
            p = L.split(); verts.append([float(p[1]), float(p[2]), float(p[3])])
    return torch.tensor(verts, dtype=torch.float32)


def test_loss_is_lower_after_fit():
    v = _template_verts()
    field = ChibiField(y_crown=float(v[:,1].max()), y_chin=0.0, z_center=0.0)
    rw = region_falloff_weights(v, MASKS)
    # Compare the LANDMARK component: 'total' at identity carries zero smooth
    # and reg, so total-vs-total compares a reg-free point to a reg-bearing
    # one. The landmark term is the proportion fit the optimizer is for.
    before = chibi_loss(field, v, rw)["landmark"].item()
    fitted = fit_chibi_field(v, MASKS, n_steps=120, lr=0.05, verbose=False)
    after = chibi_loss(fitted, v, rw)["landmark"].item()
    assert after < before * 0.5


def test_fit_lands_eye_and_mouth_near_quarter_grid():
    v = _template_verts()
    fitted = fit_chibi_field(v, MASKS, n_steps=300, lr=0.05, verbose=False)
    rw = region_falloff_weights(v, MASKS)
    with torch.no_grad():
        deformed = fitted(v, region_weights=rw)
        lines = landmark_lines(deformed)
    assert abs(float(lines["eye"]) - 0.50) < 0.03
    assert abs(float(lines["mouth"]) - 0.75) < 0.03
