import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from group_photobooth.comfy_io import (
    WORKFLOWS, post_workflow, upload_image)

COMFY_URL = os.environ.get("COMFY_URL")
pytestmark = pytest.mark.skipif(not COMFY_URL,
                                reason="set COMFY_URL=http://... to run")


def test_sam2_workflow_returns_mask_image():
    ROOT = Path(__file__).resolve().parents[2]
    bgr = cv2.imread(str(ROOT / "data/importer/identities/id_00.png"))
    name = upload_image(COMFY_URL, bgr, "test_sam2.png")
    h, w = bgr.shape[:2]
    # BBoxFromJSON helper node parses this string into the BBOX list.
    bboxes_json = f"[[{w // 8}, {h // 8}, {w * 7 // 8}, {h * 7 // 8}]]"
    out = post_workflow(COMFY_URL, WORKFLOWS / "group_sam2_mask.api.json",
                        subs={"$$IMAGE": name, "$$BBOX_JSON": bboxes_json},
                        output_node_id="5")
    assert out is not None
    assert out.shape[:2] == bgr.shape[:2]


def test_cutout_workflow_returns_rgba():
    ROOT = Path(__file__).resolve().parents[2]
    portrait_p = ROOT / "exp_output/photobooth_phase3/cells/id_00__cfg004/refined.png"
    if not portrait_p.exists():
        pytest.skip(f"missing portrait fixture {portrait_p}")
    bgr = cv2.imread(str(portrait_p))
    name = upload_image(COMFY_URL, bgr, "test_cutout.png")
    out = post_workflow(COMFY_URL, WORKFLOWS / "group_cutout.api.json",
                        subs={"$$IMAGE": name},
                        output_node_id="3", unchanged=True)
    assert out.shape[-1] == 4
    assert out.shape[:2] == bgr.shape[:2]
    alpha = out[..., 3]
    assert (alpha > 0).any() and (alpha == 0).any()
