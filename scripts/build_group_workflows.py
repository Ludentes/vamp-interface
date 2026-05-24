"""Construct the group-photobooth ComfyUI workflows programmatically.

Mirrors the pattern in scripts/build_inpaint_workflow.py: dict-literal
node definitions, written to comfyui/workflows/*.api.json. Re-run when
the graphs change. The `$$IMAGE` / `$$BBOX_JSON` placeholders are
substituted at call time by group_photobooth.comfy_io.post_workflow.

Run:
    uv run --no-project python scripts/build_group_workflows.py
"""
from __future__ import annotations

import json
from pathlib import Path

_WF_DIR = Path(__file__).resolve().parents[1] / "comfyui" / "workflows"


def build_sam2_mask_workflow() -> dict:
    """LoadImage + BBoxFromJSON -> Sam2Segmentation -> MaskToImage -> SaveImage.

    ComfyUI rejects literals for BBOX-typed inputs, so the bbox JSON is
    routed through our `BBoxFromJSON` helper (custom node at
    /home/newub/w/ComfyUI/custom_nodes/group_photobooth_helpers/).

    `$$BBOX_JSON` is substituted at call time with a JSON string like
    `"[[x1,y1,x2,y2]]"` (the helper node parses it back into a list).
    """
    return {
        "1": {"class_type": "LoadImage",
              "inputs": {"image": "$$IMAGE"}},
        "2": {"class_type": "DownloadAndLoadSAM2Model",
              "inputs": {"model": "sam2_hiera_base_plus.safetensors",
                         "segmentor": "single_image",
                         "device": "cuda",
                         "precision": "fp16"}},
        "6": {"class_type": "BBoxFromJSON",
              "inputs": {"bboxes_json": "$$BBOX_JSON"}},
        "3": {"class_type": "Sam2Segmentation",
              "inputs": {"sam2_model": ["2", 0],
                         "image": ["1", 0],
                         "bboxes": ["6", 0],
                         "individual_objects": False,
                         "keep_model_loaded": True}},
        "4": {"class_type": "MaskToImage",
              "inputs": {"mask": ["3", 0]}},
        "5": {"class_type": "SaveImage",
              "inputs": {"images": ["4", 0],
                         "filename_prefix": "group_sam2"}},
    }


def build_cutout_workflow() -> dict:
    """LoadImage -> RMBG (INSPYRENET) -> SaveImage with alpha."""
    return {
        "1": {"class_type": "LoadImage",
              "inputs": {"image": "$$IMAGE"}},
        "2": {"class_type": "RMBG",
              "inputs": {"image": ["1", 0],
                         "model": "INSPYRENET",
                         "sensitivity": 1.0,
                         "process_res": 1024,
                         "mask_blur": 0,
                         "mask_offset": 0,
                         "invert_output": False,
                         "refine_foreground": True,
                         "background": "Alpha"}},
        "3": {"class_type": "SaveImage",
              "inputs": {"images": ["2", 0],
                         "filename_prefix": "group_cutout"}},
    }


def main() -> None:
    _WF_DIR.mkdir(parents=True, exist_ok=True)
    pairs = [
        ("group_sam2_mask.api.json", build_sam2_mask_workflow()),
        ("group_cutout.api.json", build_cutout_workflow()),
    ]
    for name, wf in pairs:
        path = _WF_DIR / name
        path.write_text(json.dumps(wf, indent=2) + "\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
