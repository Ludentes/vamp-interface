"""Doll-portrait cutout via the group_cutout ComfyUI workflow.

The workflow uses 1038lab/ComfyUI-RMBG's INSPYRENET path. Swap to
BiRefNet-portrait or BEN2 by editing the workflow JSON's RMBG node
without touching this file.
"""
from __future__ import annotations

import numpy as np

from group_photobooth.comfy_io import (
    WORKFLOWS, post_workflow, upload_image)


def cutout(bgr: np.ndarray, *, comfy_url: str) -> np.ndarray:
    """Cut foreground from `bgr`, return a BGRA numpy array.

    Posts `group_cutout.api.json` to ComfyUI. The workflow's SaveImage
    node id is "3" and produces a 4-channel PNG.
    """
    name = upload_image(comfy_url, bgr)
    rgba = post_workflow(
        comfy_url, WORKFLOWS / "group_cutout.api.json",
        subs={"$$IMAGE": name},
        output_node_id="3",
        unchanged=True)
    if rgba.ndim != 3 or rgba.shape[-1] != 4:
        raise RuntimeError(
            f"group_cutout workflow returned shape {rgba.shape}, expected (H,W,4)")
    return rgba
