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
    """The FLAME face region (not the whole skull) sits inside the bbox.

    With face-region-fit, the rendered canvas legitimately contains skull
    pixels above the bbox (forehead, crown). The test invariant moves from
    `every non-black pixel is inside the bbox` (which encoded the old
    whole-skull-fit bug) to `the projected face-region vertices fill the
    bbox`.
    """
    from arkit_controlnet.flame_render import _project, load_flame_assets
    verts = deform(np.zeros(52))
    R = np.eye(3)
    bcx, bcy, bw, bh = 0.5, 0.45, 0.4, 0.5
    H = W = 512
    img = render(verts, R, (bcx, bcy, bw, bh), modality="normals", H=H, W=W)
    assert img.shape == (H, W, 3) and img.dtype == np.uint8
    assert (img.sum(axis=2) > 10).mean() > 0.02, "render is nearly empty"

    # Face-region verts sit inside the bbox (isotropic fit fills one axis).
    a = load_flame_assets()
    px = _project(verts, R, (bcx, bcy, bw, bh), H, W)
    face_px = px[a.face_region_idx]
    margin = 2.0   # pixels
    assert face_px[:, 0].min() >= (bcx - bw / 2) * W - margin
    assert face_px[:, 0].max() <= (bcx + bw / 2) * W + margin
    assert face_px[:, 1].min() >= (bcy - bh / 2) * H - margin
    assert face_px[:, 1].max() <= (bcy + bh / 2) * H + margin

    # Face-region centroid sits near the bbox centre.
    fcx, fcy = face_px.mean(axis=0) / np.array([W, H])
    assert abs(fcx - bcx) < 0.02, f"face cx={fcx:.3f}, bbox cx={bcx}"
    assert abs(fcy - bcy) < 0.02, f"face cy={fcy:.3f}, bbox cy={bcy}"


def test_face_region_idx_loaded():
    """Assets carry the FLAME face-region vertex index list."""
    from arkit_controlnet.flame_render import load_flame_assets
    a = load_flame_assets()
    assert a.face_region_idx.shape == (1787,)
    assert a.face_region_idx.dtype == np.int32
    assert a.face_region_idx.min() >= 0
    assert a.face_region_idx.max() < 5023


def test_projection_fits_face_region_to_bbox():
    """The face-region vertices — not the whole skull — fill the bbox.

    Isotropic fit: one axis (the limiting one) spans the bbox exactly; the
    other spans <= bbox. Both must fit within the bbox.
    """
    from arkit_controlnet.flame_render import (
        load_flame_assets, deform, _project)
    a = load_flame_assets()
    verts = deform(np.zeros(52))
    bbox = (0.5, 0.5, 0.4, 0.5)          # cx, cy, w, h normalized
    H = W = 512
    px = _project(verts, np.eye(3), bbox, H, W)
    face_px = px[a.face_region_idx]
    lo, hi = face_px.min(axis=0), face_px.max(axis=0)
    span_x, span_y = hi - lo
    bbox_w_px, bbox_h_px = bbox[2] * W, bbox[3] * H
    fill_x, fill_y = span_x / bbox_w_px, span_y / bbox_h_px
    # one axis fills exactly (within 1%); both fit within the bbox
    assert max(fill_x, fill_y) > 0.99 and max(fill_x, fill_y) <= 1.001, \
        f"limiting axis should fill bbox exactly: fill_x={fill_x:.3f}, fill_y={fill_y:.3f}"
    assert min(fill_x, fill_y) <= 1.001, \
        f"other axis must fit in bbox: fill_x={fill_x:.3f}, fill_y={fill_y:.3f}"
    # the full skull extends ABOVE the face box (no longer squashed in)
    assert px[:, 1].min() < lo[1]


def test_render_landmark_aligned_signature():
    """render_landmark_aligned takes verts, rotation, landmarks_px, H, W."""
    from arkit_controlnet.flame_render import (
        deform, render_landmark_aligned)
    verts = deform(np.zeros(52))
    R = np.eye(3)
    H = W = 256
    lm = np.full((478, 2), 128.0, dtype=np.float64)
    lm[1] = [128, 140]
    lm[33] = [110, 130]
    lm[263] = [146, 130]
    lm[61] = [118, 150]
    lm[291] = [138, 150]
    lm[152] = [128, 165]
    out = render_landmark_aligned(verts, R, lm, H=H, W=W)
    assert out.shape == (H, W, 3) and out.dtype == np.uint8
    assert (out.sum(axis=2) > 10).mean() > 0.02


def test_render_degenerate_bbox_raises():
    verts = deform(np.zeros(52))
    try:
        render(verts, np.eye(3), (0.5, 0.5, 0.0, 0.5), modality="normals",
               H=256, W=256)
        assert False, "expected ValueError on zero-width bbox"
    except ValueError:
        pass
