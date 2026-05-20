"""Photobooth Phase-1 spike. End-to-end on a single source photo.

Validates: (a) workflow JSON wiring, (b) canny preprocess, (c) skin mask,
(d) full cell render→swap→color-match→refine→score on the 4 phase-1 photos.
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
import uuid
from pathlib import Path

import cv2
import insightface
import numpy as np
import requests

_APP = None


def face_app():
    global _APP
    if _APP is None:
        _APP = insightface.app.FaceAnalysis(
            name="buffalo_l", providers=["CPUExecutionProvider"])
        _APP.prepare(ctx_id=0, det_size=(640, 640))
    return _APP

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

URL = "http://127.0.0.1:8188"
OUT = ROOT / "exp_output" / "photobooth_spike"
OUT.mkdir(parents=True, exist_ok=True)

PHOTO_IDS = ["id_00", "id_11", "id_16", "id_01"]
PROMPT = ("a vibrant traditional Russian matryoshka nesting doll, glossy red "
          "and gold lacquer, ornate floral painting, with a realistic "
          "photographic human face, soft three-dimensional shading, correct "
          "facial proportions, defined nose and lips, natural-sized eyes, "
          "centered frontal face, wooden doll, plain background")


def src_path(pid: str) -> Path:
    return ROOT / "data" / "importer" / "identities" / f"{pid}.png"


def detect_face_rect(bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    faces = face_app().get(bgr)
    if not faces:
        return None
    f = max(faces, key=lambda x: x.det_score)
    x0, y0, x1, y1 = f.bbox.astype(int).tolist()
    return x0, y0, x1, y1


def build_canny_control(src_bgr: np.ndarray, render_hw=(1024, 1024),
                        face_frac=0.42, t_low=80, t_high=160) -> np.ndarray:
    """Place a tight source-face canny inside a black canvas of render size,
    framing the face at `face_frac` of canvas height, centered."""
    fr = detect_face_rect(src_bgr)
    if fr is None:
        raise RuntimeError("no face")
    x0, y0, x1, y1 = fr
    m = int(0.25 * max(x1 - x0, y1 - y0))
    sh, sw = src_bgr.shape[:2]
    face = src_bgr[max(0, y0 - m):min(sh, y1 + m),
                   max(0, x0 - m):min(sw, x1 + m)]
    H, W = render_hw
    target_h = int(face_frac * H)
    fh, fw = face.shape[:2]
    target_w = int(target_h * fw / fh)
    face = cv2.resize(face, (target_w, target_h))
    edges = cv2.Canny(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY), t_low, t_high)
    canvas = np.zeros((H, W, 3), np.uint8)
    cx, cy = W // 2 - target_w // 2, int(0.30 * H)
    canvas[cy:cy + target_h, cx:cx + target_w] = cv2.cvtColor(
        edges, cv2.COLOR_GRAY2BGR)
    return canvas


def upload(bgr: np.ndarray, name: str) -> str:
    ok, buf = cv2.imencode(".png", bgr)
    r = requests.post(f"{URL}/upload/image",
                      files={"image": (name, io.BytesIO(buf.tobytes()),
                                       "image/png")},
                      data={"overwrite": "true"}, timeout=30)
    r.raise_for_status()
    j = r.json()
    return f"{j['subfolder']}/{j['name']}" if j.get("subfolder") else j["name"]


def submit_and_wait(wf: dict) -> tuple[np.ndarray | None, str]:
    cid = str(uuid.uuid4())
    r = requests.post(f"{URL}/prompt", json={"prompt": wf, "client_id": cid},
                      timeout=30)
    if r.status_code != 200:
        return None, f"submit_fail {r.status_code}: {r.text[:400]}"
    pid = r.json()["prompt_id"]
    for _ in range(600):
        h = requests.get(f"{URL}/history/{pid}", timeout=10).json()
        if pid in h:
            st = h[pid]["status"]
            if st.get("status_str") != "success":
                errs = [m for m in st.get("messages", [])
                        if m[0] == "execution_error"]
                return None, f"exec_fail: {errs[:1]}"
            for o in h[pid].get("outputs", {}).values():
                for im in o.get("images", []):
                    rr = requests.get(f"{URL}/view", params={
                        "filename": im["filename"],
                        "subfolder": im.get("subfolder", ""),
                        "type": "output"}, timeout=60)
                    return cv2.imdecode(np.frombuffer(rr.content, np.uint8),
                                        cv2.IMREAD_COLOR), "ok"
            return None, "no image"
        time.sleep(1)
    return None, "timeout"


def render_one(pid: str, *, cn_strength=0.85, steps=6, seed=12345,
               render_hw=(1024, 1024)) -> tuple[np.ndarray | None, str]:
    src = cv2.imread(str(src_path(pid)))
    if src is None:
        return None, f"no source for {pid}"
    ctrl = build_canny_control(src, render_hw=render_hw)
    ctrl_name = upload(ctrl, f"spike_ctrl_{pid}_{int(time.time())}.png")
    wf = json.load(
        open(ROOT / "comfyui/workflows/photobooth_zimage_cn.api.json"))
    subs = {"$$POSITIVE_PROMPT": PROMPT, "$$WIDTH": render_hw[1],
            "$$HEIGHT": render_hw[0], "$$CTRL_IMAGE": ctrl_name,
            "$$CN_STRENGTH": cn_strength, "$$SEED": seed, "$$STEPS": steps,
            "$$OUTPUT_PREFIX": f"photobooth_spike_{pid}"}
    for n in wf.values():
        for k, v in n.get("inputs", {}).items():
            if isinstance(v, str) and v in subs:
                n["inputs"][k] = subs[v]
    out, msg = submit_and_wait(wf)
    if out is not None:
        cv2.imwrite(str(OUT / f"ctrl_{pid}.png"), ctrl)
        cv2.imwrite(str(OUT / f"render_{pid}.png"), out)
    return out, msg


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "id_00"
    if target == "all":
        for pid in PHOTO_IDS:
            t0 = time.time()
            out, msg = render_one(pid)
            print(f"{pid}: {msg} ({time.time()-t0:.1f}s)")
    else:
        t0 = time.time()
        out, msg = render_one(target)
        print(f"{target}: {msg} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
