import pickle

import numpy as np
from arkit_controlnet.flame_render import (
    BASIS_CHANNEL_NAMES, deform, load_flame_assets, mediapipe_to_basis_vector,
    render,
)
from arkit_controlnet.eval_spike import ARKIT_BLENDSHAPE_NAMES


def test_basis_channel_names_are_52_sorted_arkit():
    assert len(BASIS_CHANNEL_NAMES) == 52
    assert BASIS_CHANNEL_NAMES == sorted(BASIS_CHANNEL_NAMES)
    assert "tongueOut" in BASIS_CHANNEL_NAMES
    assert "_neutral" not in BASIS_CHANNEL_NAMES


def test_basis_covers_all_mediapipe_expression_names():
    mp_expr = [n for n in ARKIT_BLENDSHAPE_NAMES if n != "_neutral"]
    assert set(mp_expr).issubset(set(BASIS_CHANNEL_NAMES))


def test_mediapipe_to_basis_vector_places_jawopen():
    """A unit jawOpen in MediaPipe order lands on the basis's jawOpen channel."""
    mp = {n: 0.0 for n in ARKIT_BLENDSHAPE_NAMES}
    mp["jawOpen"] = 1.0
    vec = mediapipe_to_basis_vector(mp)
    assert vec.shape == (52,)
    j = BASIS_CHANNEL_NAMES.index("jawOpen")
    assert vec[j] == 1.0
    assert np.count_nonzero(vec) == 1


def test_deform_neutral_is_template():
    a = load_flame_assets()
    out = deform(np.zeros(52))
    assert np.allclose(out, a.v_template, atol=1e-6)
    assert out.shape == (5023, 3)


def test_deform_jawopen_drops_lower_lip():
    """jawOpen must move the lower-lip vertices down (FLAME -Y is down)."""
    with open("/home/newub/w/LAM/model_zoo/human_parametric_models/"
              "flame_vhap/FLAME_masks.pkl", "rb") as f:
        masks = pickle.load(f, encoding="latin1")
    lips = np.asarray(masks["lips"]).ravel()
    mp = {n: 0.0 for n in ARKIT_BLENDSHAPE_NAMES}
    mp["jawOpen"] = 1.0
    template = load_flame_assets().v_template
    moved = deform(mediapipe_to_basis_vector(mp))
    dy = (moved[lips, 1] - template[lips, 1]).mean()
    assert dy < -1e-4, f"jawOpen did not drop the lips downward (dy={dy})"


def test_mediapipe_to_basis_vector_full_permutation():
    """Every MediaPipe expression name lands on its own basis channel."""
    mp_expr = [n for n in ARKIT_BLENDSHAPE_NAMES if n != "_neutral"]
    for i, name in enumerate(mp_expr):
        mp = {n: 0.0 for n in ARKIT_BLENDSHAPE_NAMES}
        mp[name] = float(i + 1)
        vec = mediapipe_to_basis_vector(mp)
        j = BASIS_CHANNEL_NAMES.index(name)
        assert vec[j] == float(i + 1), f"{name} mis-placed"
        assert np.count_nonzero(vec) == 1


def test_render_neutral_fills_bbox():
    verts = deform(np.zeros(52))
    R = np.eye(3)
    bcx, bcy, bw, bh = 0.5, 0.45, 0.4, 0.5   # cx, cy, w, h — normalized
    img = render(verts, R, (bcx, bcy, bw, bh), modality="normals", H=512, W=512)
    assert img.shape == (512, 512, 3) and img.dtype == np.uint8
    nonblack = (img.sum(axis=2) > 10)
    ys, xs = np.where(nonblack)
    assert nonblack.mean() > 0.02, "render is nearly empty"
    # centroid sits at the bbox centre on both axes
    cx, cy = xs.mean() / 512, ys.mean() / 512
    assert abs(cx - bcx) < 0.12, f"face not centred at bbox cx (got {cx:.3f})"
    assert abs(cy - bcy) < 0.12, f"face not centred at bbox cy (got {cy:.3f})"
    # the face stays inside the requested bbox (isotropic fit, small margin)
    margin = 0.04
    assert xs.min() / 512 >= bcx - bw / 2 - margin, "face overflows bbox left"
    assert xs.max() / 512 <= bcx + bw / 2 + margin, "face overflows bbox right"
    assert ys.min() / 512 >= bcy - bh / 2 - margin, "face overflows bbox top"
    assert ys.max() / 512 <= bcy + bh / 2 + margin, "face overflows bbox bottom"


def test_render_degenerate_bbox_raises():
    verts = deform(np.zeros(52))
    try:
        render(verts, np.eye(3), (0.5, 0.5, 0.0, 0.5), modality="normals",
               H=256, W=256)
        assert False, "expected ValueError on zero-width bbox"
    except ValueError:
        pass
