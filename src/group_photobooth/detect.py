"""YOLOv8-person (Python) + buffalo_l face (Python) + SAM2 mask (ComfyUI).

YOLO is fast on CPU (~100 ms / 1024² for yolov8n). buffalo_l is already a
process-cached dependency via swap_core. SAM2 runs as a ComfyUI workflow
so model weights stay loaded across calls.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from json import dumps
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from group_photobooth import Person
from group_photobooth.comfy_io import (
    WORKFLOWS, post_workflow, upload_image)

# swap_core lives in scripts/, not src/ — add to path once
_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


@lru_cache(maxsize=1)
def _yolo() -> YOLO:
    return YOLO("yolov8n.pt")


@lru_cache(maxsize=1)
def _face_app():
    from swap_core import make_face_app  # type: ignore
    return make_face_app()


def _sam2_mask(comfy_url: str, image_name: str,
               bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Call group_sam2_mask.api.json with the uploaded image and a single
    bbox. Returns an HxW uint8 mask (0/255).

    BBoxFromJSON helper parses the JSON string into the BBOX type.
    """
    bbox_json = dumps([[int(bbox[0]), int(bbox[1]),
                        int(bbox[2]), int(bbox[3])]])
    mask_bgr = post_workflow(
        comfy_url, WORKFLOWS / "group_sam2_mask.api.json",
        subs={"$$IMAGE": image_name, "$$BBOX_JSON": bbox_json},
        output_node_id="5")
    gray = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY) if mask_bgr.ndim == 3 else mask_bgr
    return ((gray > 127).astype(np.uint8) * 255)


def _faces_in_full_image(photo_bgr: np.ndarray, app) -> list:
    """Run buffalo_l face detection on the full image.

    Running on the full image rather than per-body-crop avoids det failures:
    buffalo_l's det_10g model can silently miss faces when the crop is
    non-square or has very different dimensions from the detector's training
    distribution (empirically verified on 1024×1024 id_00 crop 990×968).
    """
    rgb = cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2RGB)
    return app.get(rgb)


def _best_face_for_body(
    faces: list,
    body_bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    """Return the face bbox (in photo coords) whose center lies inside the
    body bbox and has the largest area. Returns None if no match."""
    bx1, by1, bx2, by2 = body_bbox
    best = None
    best_area = 0.0
    for f in faces:
        fx1, fy1, fx2, fy2 = f.bbox
        cx, cy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
        if bx1 <= cx <= bx2 and by1 <= cy <= by2:
            area = (fx2 - fx1) * (fy2 - fy1)
            if area > best_area:
                best_area = area
                best = (int(fx1), int(fy1), int(fx2), int(fy2))
    return best


def detect_people(photo_bgr: np.ndarray, *,
                  comfy_url: str,
                  conf: float = 0.35,
                  iou: float = 0.45) -> list[Person]:
    """Detect every person in `photo_bgr`. SAM2 segmentation runs as a
    ComfyUI workflow at `comfy_url`. Persons without a detectable face are
    skipped (and logged).

    Face detection runs on the full image rather than per-body-crop: buffalo_l's
    det_10g model can fail on non-square crops that differ from its training
    distribution.
    """
    results = _yolo().predict(photo_bgr, classes=[0], conf=conf, iou=iou,
                              verbose=False)
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return []
    app = _face_app()
    all_faces = _faces_in_full_image(photo_bgr, app)
    image_name = upload_image(comfy_url, photo_bgr)
    persons: list[Person] = []
    for box in results[0].boxes.xyxy.cpu().numpy().astype(int):
        x1, y1, x2, y2 = box.tolist()
        face_bbox = _best_face_for_body(all_faces, (x1, y1, x2, y2))
        if face_bbox is None:
            print(f"[detect] skipping bbox ({x1},{y1},{x2},{y2}): no face")
            continue
        mask = _sam2_mask(comfy_url, image_name, (x1, y1, x2, y2))
        if mask.shape != photo_bgr.shape[:2]:
            mask = cv2.resize(mask, (photo_bgr.shape[1], photo_bgr.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        persons.append(Person(body_bbox=(x1, y1, x2, y2),
                              body_mask=mask, face_bbox=face_bbox))
    return persons
