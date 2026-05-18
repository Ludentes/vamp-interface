"""build_inpaint_workflow -- deterministic generation->inpaint transform."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_inpaint_workflow import build_inpaint_workflow

WF_DIR = Path(__file__).resolve().parents[1] / "comfyui" / "workflows"


def _ksampler(wf):
    return next(n for n in wf.values() if n["class_type"] == "KSampler")


def _by_class(wf, cls):
    return [n for n in wf.values() if n["class_type"] == cls]


def test_zimage_transform_inserts_inpaint_chain():
    base = json.loads((WF_DIR / "matryoshka_zimage_turbo.api.json").read_text())
    wf = build_inpaint_workflow(base)
    snm = _by_class(wf, "SetLatentNoiseMask")
    assert len(snm) == 1
    ks = _ksampler(wf)
    snm_id = next(i for i, n in wf.items() if n["class_type"] == "SetLatentNoiseMask")
    assert ks["inputs"]["latent_image"] == [snm_id, 0]
    assert ks["inputs"]["denoise"] == "$$DENOISE"
    loads = {n["inputs"].get("image") for n in _by_class(wf, "LoadImage")}
    assert "$$DOLL_FILENAME" in loads
    assert _by_class(wf, "LoadImageMask")[0]["inputs"]["image"] == "$$MASK_FILENAME"
    assert len(_by_class(wf, "VAEEncode")) == 1


def test_sdxl_transform_bypasses_controlnet():
    base = json.loads((WF_DIR / "matryoshka_sdxl_lightning.api.json").read_text())
    wf = build_inpaint_workflow(base)
    ks = _ksampler(wf)
    pos_src = wf[ks["inputs"]["positive"][0]]
    assert pos_src["class_type"] != "ControlNetApplyAdvanced"
    assert len(_by_class(wf, "SetLatentNoiseMask")) == 1
    assert ks["inputs"]["denoise"] == "$$DENOISE"
