"""Whole-image low-denoise refine via remote ComfyUI."""
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
WORKFLOW = ROOT / "comfyui/workflows/photobooth_zimage_refine.api.json"


def _upload(comfy_url: str, bgr: np.ndarray, name: str) -> str:
    _, buf = cv2.imencode(".png", bgr)
    r = requests.post(f"{comfy_url}/upload/image",
                      files={"image": (name, io.BytesIO(buf.tobytes()),
                                       "image/png")},
                      data={"overwrite": "true"}, timeout=30)
    r.raise_for_status()
    j = r.json()
    return f"{j['subfolder']}/{j['name']}" if j.get("subfolder") else j["name"]


def refine(comfy_url: str, swap_bgr: np.ndarray, *, prompt: str,
           denoise: float, seed: int, tag: str = "refine") -> np.ndarray:
    """If denoise == 0, returns swap_bgr untouched (skip the network round-trip).
    Otherwise loads the refine workflow, fills params, submits, fetches."""
    if denoise <= 0:
        return swap_bgr
    name = _upload(comfy_url, swap_bgr,
                   f"phb_swap_{tag}_{int(time.time()*1000)}.png")
    wf = json.load(open(WORKFLOW))
    subs = {"$$POSITIVE_PROMPT": prompt, "$$INPUT_IMAGE": name,
            "$$SEED": int(seed), "$$DENOISE": float(denoise),
            "$$OUTPUT_PREFIX": f"phb_refined_{tag}"}
    for n in wf.values():
        for k, v in n.get("inputs", {}).items():
            if isinstance(v, str) and v in subs:
                n["inputs"][k] = subs[v]

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
                errs = [m for m in st.get("messages", [])
                        if m[0] == "execution_error"]
                raise RuntimeError(f"refine failed: {errs[:1]}")
            for o in h[pid].get("outputs", {}).values():
                for im in o.get("images", []):
                    rr = requests.get(f"{comfy_url}/view", params={
                        "filename": im["filename"],
                        "subfolder": im.get("subfolder", ""),
                        "type": "output"}, timeout=60)
                    return cv2.imdecode(np.frombuffer(rr.content, np.uint8),
                                        cv2.IMREAD_COLOR)
            raise RuntimeError("refine: no image")
        time.sleep(0.5)
    raise RuntimeError("refine: timeout")
