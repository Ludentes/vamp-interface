import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import torch
from chibi.landmarks import (load_landmark_embedding, landmark_lines,
                             QUARTER_GRID_TARGETS, FLAME_TEMPLATE)

EMB = "/home/newub/w/LAM/model_zoo/human_parametric_models/flame_assets/flame/landmark_embedding_with_eyes.npy"


def _template_verts():
    verts = []
    for L in pathlib.Path(FLAME_TEMPLATE).read_text().splitlines():
        if L.startswith("v "):
            p = L.split()
            verts.append([float(p[1]), float(p[2]), float(p[3])])
    return torch.tensor(verts, dtype=torch.float32)


def test_embedding_loads_with_expected_shapes():
    faces_idx, bary = load_landmark_embedding(EMB)
    assert faces_idx.shape == (70,)
    assert bary.shape == (70, 3)


def test_realistic_template_landmark_lines_match_known_u():
    v = _template_verts()
    lines = landmark_lines(v)  # uses faces from FLAME template internally
    # On the undeformed FLAME head, eye line sits above the nose, nose above
    # mouth, mouth above chin (u increases downward).
    assert lines["brow"] < lines["eye"] < lines["nose"] < lines["mouth"] < lines["chin"]
    assert 0.9 < lines["chin"] <= 1.01


def test_quarter_grid_targets_present():
    for k in ("eye", "nose", "mouth"):
        assert k in QUARTER_GRID_TARGETS["lines"]
    assert abs(QUARTER_GRID_TARGETS["lines"]["eye"] - 0.50) < 1e-9
    assert abs(QUARTER_GRID_TARGETS["lines"]["mouth"] - 0.75) < 1e-9
