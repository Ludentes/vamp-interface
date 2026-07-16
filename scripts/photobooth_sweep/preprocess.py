"""Control-image preprocessing for the photobooth sweep.

- canny via OpenCV, three presets (soft/default/aggressive)
- depth via the remote ComfyUI `DepthAnythingV2Preprocessor` node
- face_pixel_budget determines render H×W and how tight the source face is
  framed inside it
"""
from __future__ import annotations

import io
import json
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent.parent

# (render_h, render_w, crop_mode):
# - "natural": scale source so min(H,W) = render_dim, center-crop on face
# - "tight":   crop a tight_zoom*face-bbox region centered on face, scale to render
FACE_PIXEL_BUDGETS = {
    "natural_1024": (1024, 1024, "natural"),
    "tight_1024":   (1024, 1024, "tight"),
    "natural_768":  (768, 768, "natural"),
}
TIGHT_ZOOM = 1.4  # crop region = TIGHT_ZOOM * max(face_w, face_h)

# (low_thresh, high_thresh, blur_sigma)
CANNY_PRESETS = {
    "soft":       (50, 150, 1.5),
    "default":    (100, 200, 0.0),
    "aggressive": (150, 250, 0.0),
}


def detect_face_rect(app, bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return (x0,y0,x1,y1) of the highest-scoring face via insightface."""
    faces = app.get(bgr)
    if not faces:
        return None
    f = max(faces, key=lambda x: x.det_score)
    x0, y0, x1, y1 = f.bbox.astype(int).tolist()
    return int(x0), int(y0), int(x1), int(y1)


def frame_face(src_bgr: np.ndarray, face_rect: tuple[int, int, int, int],
               render_hw: tuple[int, int], crop_mode: str
               ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Build a render-sized canvas filled entirely with source pixels — no
    padding, no reflection. Two modes:

    - "natural": scale source so min(src_h, src_w) == render_dim; center-crop
      the resulting image on the face center, clamped to image bounds.
    - "tight":   take a TIGHT_ZOOM*max(face_w, face_h) square around the face
      center, clamped to source bounds; scale to render_dim.

    Returns (canvas, (face_x0, face_y0, face_w, face_h)) in canvas coords.
    """
    x0, y0, x1, y1 = face_rect
    fcx_src = (x0 + x1) / 2.0
    fcy_src = (y0 + y1) / 2.0
    fw_src = x1 - x0
    fh_src = y1 - y0
    H, W = render_hw
    assert H == W, "square render only"
    sh, sw = src_bgr.shape[:2]

    if crop_mode == "natural":
        scale = H / min(sh, sw)
        new_h, new_w = int(round(sh * scale)), int(round(sw * scale))
        scaled = cv2.resize(
            src_bgr, (new_w, new_h),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        # face center in scaled coords
        fcx = int(round(fcx_src * scale))
        fcy = int(round(fcy_src * scale))
        # window around the face center, clamped to scaled bounds
        x = max(0, min(new_w - W, fcx - W // 2))
        y = max(0, min(new_h - H, fcy - H // 2))
        canvas = scaled[y:y + H, x:x + W].copy()
        face_w = int(round(fw_src * scale))
        face_h = int(round(fh_src * scale))
        fx0 = fcx - face_w // 2 - x
        fy0 = fcy - face_h // 2 - y
        return canvas, (fx0, fy0, face_w, face_h)

    if crop_mode == "tight":
        side = TIGHT_ZOOM * max(fw_src, fh_src)
        half = side / 2.0
        # clamp window to source bounds — if it would overflow, slide it in
        x = int(round(fcx_src - half))
        y = int(round(fcy_src - half))
        side_i = int(round(side))
        x = max(0, min(sw - side_i, x))
        y = max(0, min(sh - side_i, y))
        # if side_i > sw or sh, shrink to fit
        side_i = min(side_i, sw - x, sh - y)
        crop = src_bgr[y:y + side_i, x:x + side_i]
        scale = H / side_i
        canvas = cv2.resize(
            crop, (W, H),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        face_w = int(round(fw_src * scale))
        face_h = int(round(fh_src * scale))
        fx0 = int(round((fcx_src - x) * scale)) - face_w // 2
        fy0 = int(round((fcy_src - y) * scale)) - face_h // 2
        return canvas, (fx0, fy0, face_w, face_h)

    raise ValueError(f"unknown crop_mode {crop_mode!r}")


def build_canny(framed_bgr: np.ndarray, preset: str) -> np.ndarray:
    """Canny on the framed source canvas (option-B framing: no hard border)."""
    t_lo, t_hi, sigma = CANNY_PRESETS[preset]
    g = cv2.cvtColor(framed_bgr, cv2.COLOR_BGR2GRAY)
    if sigma > 0:
        g = cv2.GaussianBlur(g, (0, 0), sigma)
    edges = cv2.Canny(g, t_lo, t_hi)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def _upload(comfy_url: str, bgr: np.ndarray, name: str) -> str:
    _, buf = cv2.imencode(".png", bgr)
    r = requests.post(f"{comfy_url}/upload/image",
                      files={"image": (name, io.BytesIO(buf.tobytes()),
                                       "image/png")},
                      data={"overwrite": "true"}, timeout=30)
    r.raise_for_status()
    j = r.json()
    return f"{j['subfolder']}/{j['name']}" if j.get("subfolder") else j["name"]


def _submit(comfy_url: str, wf: dict) -> np.ndarray:
    cid = str(uuid.uuid4())
    r = requests.post(f"{comfy_url}/prompt",
                      json={"prompt": wf, "client_id": cid}, timeout=30)
    r.raise_for_status()
    pid = r.json()["prompt_id"]
    for _ in range(600):
        h = requests.get(f"{comfy_url}/history/{pid}", timeout=10).json()
        if pid in h:
            st = h[pid]["status"]
            if st.get("status_str") != "success":
                errs = [m for m in st.get("messages", []) if m[0] == "execution_error"]
                raise RuntimeError(f"depth preprocess failed: {errs[:1]}")
            for o in h[pid].get("outputs", {}).values():
                for im in o.get("images", []):
                    rr = requests.get(f"{comfy_url}/view", params={
                        "filename": im["filename"],
                        "subfolder": im.get("subfolder", ""),
                        "type": "output"}, timeout=60)
                    return cv2.imdecode(np.frombuffer(rr.content, np.uint8),
                                        cv2.IMREAD_COLOR)
            raise RuntimeError("depth preprocess: no image")
        time.sleep(0.5)
    raise RuntimeError("depth preprocess: timeout")


_DEPTH_WF = {
    "1": {"class_type": "LoadImage", "inputs": {"image": "$$INPUT"}},
    "2": {"class_type": "DepthAnythingV2Preprocessor",
          "inputs": {"image": ["1", 0], "ckpt_name": "depth_anything_v2_vitl.pth",
                     "resolution": 1024}},
    "3": {"class_type": "SaveImage",
          "inputs": {"images": ["2", 0], "filename_prefix": "phb_depth"}},
}


def build_depth(comfy_url: str, framed_bgr: np.ndarray,
                tag: str = "depth_in") -> np.ndarray:
    """Run DepthAnythingV2 on the framed face canvas via remote ComfyUI.
    Returns an RGB depth map sized to the input framing."""
    name = _upload(comfy_url, framed_bgr, f"phb_{tag}_{int(time.time()*1000)}.png")
    wf = json.loads(json.dumps(_DEPTH_WF).replace("$$INPUT", name))
    out = _submit(comfy_url, wf)
    if out.shape[:2] != framed_bgr.shape[:2]:
        out = cv2.resize(out, (framed_bgr.shape[1], framed_bgr.shape[0]))
    return out


def build_control(app, comfy_url: str, src_bgr: np.ndarray,
                  cn_condition: str, canny_preset: str | None,
                  face_pixel_budget: str
                  ) -> tuple[np.ndarray, tuple[int, int]]:
    """End-to-end: source photo → framed → canny|depth control image.
    Returns (control_bgr, (render_h, render_w))."""
    H, W, crop_mode = FACE_PIXEL_BUDGETS[face_pixel_budget]
    rect = detect_face_rect(app, src_bgr)
    if rect is None:
        raise RuntimeError("no source face")
    framed, _ = frame_face(src_bgr, rect, (H, W), crop_mode)
    if cn_condition == "canny":
        if canny_preset is None:
            raise ValueError("canny_preset required when cn_condition=canny")
        ctrl = build_canny(framed, canny_preset)
    elif cn_condition == "depth":
        ctrl = build_depth(comfy_url, framed)
    else:
        raise ValueError(f"unknown cn_condition: {cn_condition}")
    return ctrl, (H, W)


def upload_control(comfy_url: str, ctrl_bgr: np.ndarray, tag: str) -> str:
    return _upload(comfy_url, ctrl_bgr,
                   f"phb_ctrl_{tag}_{int(time.time()*1000)}.png")
