"""Single-face photobooth call — Phase 3 heavy mix, locked.

Wraps the existing photobooth machinery (preprocess + cn_workflow +
swap_core.swap_identity) so the group pipeline sees one function:
`render_doll(face_crop_bgr, comfy_url, seed, demo) -> portrait_bgr`.

Phase 3 "heavy mix" verdict (cfg004): swap_weight=0.10, embedding pulled
~28% toward doll, natural_1024 / canny / soft, demo_inject on, refine off.
See docs/superpowers/specs/2026-05-21-group-photobooth-pipeline-design.md
section "Locked decisions".
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))

import swap_core  # noqa: E402
from photobooth_sweep import driver as ph_driver  # noqa: E402
from photobooth_sweep import preprocess as ph_pp  # noqa: E402

HEAVY_MIX_CFG: dict[str, Any] = {
    "face_pixel_budget": "natural_1024",
    "cn_condition": "canny",
    "cn_strength": 0.85,
    "canny_preset": "soft",
    "refine_denoise": 0.00,
    "demo_inject": "on",
    "swap_weight": 0.10,
}


def render_doll(face_crop_bgr: np.ndarray, comfy_url: str, *,
                seed: int, demo: dict[str, str]) -> np.ndarray:
    """Run the Phase 3 heavy mix end-to-end on one face crop.

    Returns the post-swap doll portrait as BGR. Refine is off
    (refine_denoise=0), so the swap output is the final portrait.

    `demo` must carry keys consumed by photobooth_sweep.driver.make_prompt:
    `gender` ("M"|"F"|other), `age_bin`, `race`. If `demo_inject` is "off",
    `demo` may be empty.
    """
    cfg = HEAVY_MIX_CFG
    prompt = ph_driver.make_prompt("group", demo, cfg["demo_inject"])
    app = swap_core.make_face_app()
    swapper = swap_core.load_swapper()
    src_face = swap_core.detect_source(app, face_crop_bgr)
    if src_face is None:
        raise ValueError("no face detected in crop")

    ctrl, render_hw = ph_pp.build_control(
        app, comfy_url, face_crop_bgr,
        cfg["cn_condition"], cfg["canny_preset"], cfg["face_pixel_budget"])
    ctrl_name = ph_pp.upload_control(comfy_url, ctrl, "groupbooth")
    wf = ph_driver.cn_workflow(
        comfy_url, ctrl_name=ctrl_name, prompt=prompt,
        render_hw=render_hw, cn_strength=cfg["cn_strength"], seed=seed,
        prefix=f"groupbooth_{int(time.time()*1000)}")
    render = ph_driver.comfy_submit(comfy_url, wf)
    swap, _mode, _score = swap_core.swap_identity(
        app, swapper, render, src_face, collapse=True, restore=False,
        swap_weight=cfg["swap_weight"])
    return swap
