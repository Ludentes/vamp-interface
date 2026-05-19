import json
import pathlib

import pytest

from chibi.koban_asset import load_koban, load_koban_landmarks, ARKIT_52

CANON = pathlib.Path("exp_output/chibi_meshes/koban_canonical")

pytestmark = pytest.mark.skipif(
    not (CANON / "koban.obj").exists(),
    reason="koban.obj not generated; run scripts/koban_prep.py first",
)


def test_koban_obj_parses_with_uv():
    k = load_koban(CANON)
    assert k.verts.shape[1] == 3 and k.faces.shape[1] == 3
    assert k.uv.shape[1] == 2 and k.uv_faces.shape == k.faces.shape
    assert 0.0 <= float(k.uv.min()) and float(k.uv.max()) <= 1.0001


def test_arkit_52_present():
    keys = set(json.loads((CANON / "arkit_keys.json").read_text()))
    missing = [n for n in ARKIT_52 if n not in keys]
    assert not missing, f"missing ARKit keys: {missing}"


def test_koban_landmarks_load():
    pts = load_koban_landmarks(CANON)
    assert pts.shape[1] == 2 and pts.shape[0] >= 12
    idx = load_koban_landmarks(CANON, return_idx=True)
    assert all(0 <= int(i) <= 105 for i in idx)
