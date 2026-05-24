import os
from pathlib import Path

import cv2
import numpy as np
import pytest

COMFY_URL = os.environ.get("COMFY_URL")
pytestmark = pytest.mark.skipif(not COMFY_URL,
                                reason="set COMFY_URL=http://... to run")


def _build_two_person_synth(out_path: Path) -> Path:
    ROOT = Path(__file__).resolve().parents[2]
    a = cv2.imread(str(ROOT / "data/importer/identities/id_00.png"))
    b = cv2.imread(str(ROOT / "data/importer/identities/id_11.png"))
    h = max(a.shape[0], b.shape[0])
    a = cv2.resize(a, (int(a.shape[1] * h / a.shape[0]), h))
    b = cv2.resize(b, (int(b.shape[1] * h / b.shape[0]), h))
    canvas = np.full((h, a.shape[1] + b.shape[1] + 100, 3), 220, np.uint8)
    canvas[:, :a.shape[1]] = a
    canvas[:, a.shape[1] + 100:] = b
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    return out_path


def test_n1_path_runs_end_to_end(tmp_path):
    from group_photobooth.driver import run

    ROOT = Path(__file__).resolve().parents[2]
    photo = ROOT / "data/importer/identities/id_00.png"
    out = run(photo, "blank_studio", tmp_path, comfy_url=COMFY_URL)
    assert out.exists()
    img = cv2.imread(str(out))
    assert img is not None
    assert img.shape == (1024, 1024, 3)
    diff = np.abs(img.astype(int) - 200).mean()
    assert diff > 5.0, "result looks like the untouched background"


def test_n2_synth_runs_end_to_end(tmp_path):
    from group_photobooth.driver import run

    photo = _build_two_person_synth(tmp_path / "synth.png")
    out = run(photo, "blank_studio", tmp_path / "run", comfy_url=COMFY_URL)
    assert out.exists()
    people = sorted((tmp_path / "run" / "people").iterdir())
    assert len(people) == 2
    for pdir in people:
        assert (pdir / "doll.png").exists()
        assert (pdir / "doll_rgba.png").exists()
