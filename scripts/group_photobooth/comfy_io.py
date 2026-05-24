"""HTTP helpers for posting workflows to a ComfyUI server.

Mirrors the pattern used in scripts/photobooth_sweep/driver.py but
generalized: any workflow can be posted by giving (path, substitutions,
output_node_id) and gets back a numpy BGR (or BGRA) image.
"""
from __future__ import annotations

import copy
import io
import json
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / "comfyui" / "workflows"


def substitute_template(workflow: dict, subs: dict[str, str]) -> dict:
    """Walk a workflow JSON and replace every leaf string that matches
    any key in `subs` exactly. Returns a deep copy; input unchanged."""
    def walk(v):
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items()}
        if isinstance(v, list):
            return [walk(x) for x in v]
        if isinstance(v, str) and v in subs:
            return subs[v]
        return v
    return walk(copy.deepcopy(workflow))


def upload_image(comfy_url: str, bgr: np.ndarray, name: str | None = None) -> str:
    """POST an image to ComfyUI /upload/image. Return the server filename."""
    if name is None:
        name = f"group_{uuid.uuid4().hex[:12]}.png"
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("imencode failed")
    files = {"image": (name, buf.tobytes(), "image/png")}
    r = requests.post(f"{comfy_url}/upload/image",
                      files=files, data={"overwrite": "1"})
    r.raise_for_status()
    return r.json()["name"]


def post_workflow(comfy_url: str, workflow_path: Path,
                  subs: dict[str, str], output_node_id: str,
                  timeout_s: float = 120.0,
                  unchanged: bool = False) -> np.ndarray:
    """POST a substituted workflow, poll for completion, fetch the output
    image from `output_node_id`. Returns BGR or BGRA numpy array depending
    on the saved PNG channel count."""
    with open(workflow_path) as f:
        tpl = json.load(f)
    wf = substitute_template(tpl, subs)
    prompt_id = uuid.uuid4().hex
    r = requests.post(f"{comfy_url}/prompt",
                      json={"prompt": wf, "client_id": prompt_id})
    r.raise_for_status()
    pid = r.json()["prompt_id"]
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        time.sleep(0.5)
        h = requests.get(f"{comfy_url}/history/{pid}").json()
        if pid not in h:
            continue
        outputs = h[pid].get("outputs", {})
        if output_node_id not in outputs:
            continue
        imgs = outputs[output_node_id].get("images", [])
        if not imgs:
            raise RuntimeError(f"workflow: node {output_node_id} produced no image")
        meta = imgs[0]
        fn = meta["filename"]
        sub = meta.get("subfolder", "")
        typ = meta.get("type", "output")
        img_r = requests.get(f"{comfy_url}/view",
                             params={"filename": fn, "subfolder": sub, "type": typ})
        img_r.raise_for_status()
        arr = np.frombuffer(img_r.content, np.uint8)
        flag = cv2.IMREAD_UNCHANGED if unchanged else cv2.IMREAD_COLOR
        out = cv2.imdecode(arr, flag)
        if out is None:
            raise RuntimeError("decoded empty image from /view")
        return out
    raise TimeoutError(f"workflow timed out after {timeout_s}s")
