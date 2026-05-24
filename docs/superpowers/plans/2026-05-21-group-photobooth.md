# Group Photobooth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a photo of 1-5 people into a matryoshka group portrait composited onto a chosen pre-rendered background.

**Architecture:** YOLOv8 person detection + buffalo_l face detection run **in Python**; SAM2 body-mask prediction and INSPYRENET doll cutout run **as ComfyUI workflows** posted via HTTP so models stay hot across runs and can be node-swapped without code edits. The per-face photobooth (Phase 3 heavy mix, `swap_weight=0.10`) is called verbatim, also via HTTP. Pillow alpha-composite is in Python. All orchestration (loop over N persons, per-file resumable cache, letterbox + painter's-order math) is in Python.

**Tech Stack:** Python 3.12, ultralytics YOLOv8 (CPU), opencv-python, numpy, Pillow, requests. ComfyUI custom nodes: `kijai/ComfyUI-segment-anything-2` (SAM2) and `1038lab/ComfyUI-RMBG` (INSPYRENET). Reuses `scripts/photobooth_sweep/` and `scripts/swap_core.py` verbatim.

**Spec:** `docs/superpowers/specs/2026-05-21-group-photobooth-pipeline-design.md`
**Tooling research:** `docs/research/2026-05-21-comfyui-compositor-toolkit.md`

---

## File structure

```
scripts/group_photobooth/
  __init__.py           # exports Person, Placement, dataclasses
  comfy_io.py           # HTTP helper: upload image, post workflow, fetch output
  detect.py             # YOLOv8-person + buffalo_l + ComfyUI-SAM2 → list[Person]
  silhouette.py         # ComfyUI-INSPYRENET cutout → RGBA
  layout.py             # photo coords → background-canvas placements + painter z-order
  composite.py          # alpha blend + optional Lab match + drop shadow
  face_renderer.py      # thin wrapper around photobooth Phase-3 heavy mix
  driver.py             # CLI orchestrator, resumable per-person cache
comfyui/workflows/
  group_sam2_mask.api.json   # image + bbox → single-person body mask
  group_cutout.api.json      # doll portrait → RGBA cutout
assets/backgrounds/
  manifest.json         # array of {id, description, dims, palette_lab_mean, lighting_hint}
  blank_studio/bg.png   # shipped seed background
tests/group_photobooth/
  __init__.py
  conftest.py           # shared fixtures (synthetic photo, background)
  test_comfy_io.py
  test_detect.py
  test_silhouette.py
  test_layout.py
  test_composite.py
  test_driver_smoke.py
```

---

## Task 1: Bootstrap package skeleton and dataclasses

**Files:**
- Create: `scripts/group_photobooth/__init__.py`
- Create: `tests/group_photobooth/__init__.py`
- Create: `tests/group_photobooth/conftest.py`

- [ ] **Step 1: Write the failing test**

`tests/group_photobooth/conftest.py`:
```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
```

`tests/group_photobooth/test_dataclasses.py`:
```python
from group_photobooth import Person, Placement
import numpy as np

def test_person_dataclass():
    p = Person(body_bbox=(10, 20, 100, 200),
               body_mask=np.zeros((300, 200), dtype=np.uint8),
               face_bbox=(30, 25, 70, 75))
    assert p.body_bbox == (10, 20, 100, 200)
    assert p.body_mask.shape == (300, 200)
    assert p.face_bbox == (30, 25, 70, 75)

def test_placement_dataclass():
    pl = Placement(x_center=512, y_bottom=900, height=400, z_order=0)
    assert pl.x_center == 512
    assert pl.height == 400
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-project pytest tests/group_photobooth/test_dataclasses.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'group_photobooth'`

- [ ] **Step 3: Create the package**

`scripts/group_photobooth/__init__.py`:
```python
"""Group photobooth — multi-person → matryoshka group portrait.

Approach A from docs/superpowers/specs/2026-05-21-group-photobooth-pipeline-design.md:
per-face render via the existing single-face photobooth, then rembg + composite
onto a pre-rendered background in painter's order.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Person:
    """A detected person in the input photo.

    body_bbox: (x1, y1, x2, y2) in photo pixel coords.
    body_mask: uint8 HxW binary mask (0 or 255), same dims as the input photo.
    face_bbox: (x1, y1, x2, y2) face crop in photo pixel coords, None if no
        face was detected (person is then skipped by the driver).
    """
    body_bbox: tuple[int, int, int, int]
    body_mask: np.ndarray
    face_bbox: tuple[int, int, int, int] | None


@dataclass(frozen=True)
class Placement:
    """A doll's placement on the background canvas.

    x_center, y_bottom: foot anchor in background-canvas pixel coords.
    height: doll's height in background-canvas pixels.
    z_order: painter's-algorithm order. Lower = further back, painted first.
    """
    x_center: int
    y_bottom: int
    height: int
    z_order: int


__all__ = ["Person", "Placement"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-project pytest tests/group_photobooth/test_dataclasses.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/group_photobooth/__init__.py tests/group_photobooth/
git commit -m "feat(group-photobooth): package skeleton + Person/Placement dataclasses"
```

---

## Task 2: ComfyUI custom nodes + workflow scaffolding

**Files:**
- Create: `comfyui/workflows/group_sam2_mask.api.json`
- Create: `comfyui/workflows/group_cutout.api.json`
- Create: `scripts/group_photobooth/comfy_io.py`
- Create: `tests/group_photobooth/test_comfy_io.py`

**Prerequisite: install custom nodes (one-time, on the ComfyUI machine).**

```bash
cd /home/newub/w/ComfyUI/custom_nodes/
git clone https://github.com/kijai/ComfyUI-segment-anything-2
git clone https://github.com/1038lab/ComfyUI-RMBG
# Restart ComfyUI so the new nodes register. SAM2 + INSPYRENET model
# weights auto-download to ComfyUI/models/sam2/ and ComfyUI/models/RMBG/
# on first workflow run.
```

- [ ] **Step 1: Author `group_sam2_mask.api.json` in the ComfyUI UI**

In the ComfyUI web UI, build a graph with these nodes:
1. `LoadImage` (id `1`) — `image: "$$IMAGE"` (templated input filename)
2. `(Down)load SAM2 Model` (Kijai) (id `2`) — `model: "sam2_hiera_base_plus.safetensors"`, `precision: "fp16"`, `device: "cuda"`
3. `Sam2Segmentation` (id `3`) — `sam2_model: 2.0`, `image: 1.0`, `mask_or_coordinates: "$$BBOX_JSON"` (a JSON string like `[[x1,y1,x2,y2]]`), `keep_model_loaded: true`, `individual_objects: false`
4. `MaskToImage` (id `4`) — `mask: 3.0`
5. `SaveImage` (id `5`) — `images: 4.0`, `filename_prefix: "group_sam2"`

Export via **Save (API Format)** to `comfyui/workflows/group_sam2_mask.api.json`. The `$$IMAGE` and `$$BBOX_JSON` literal strings will be substituted by `comfy_io.post_workflow` at call time.

- [ ] **Step 2: Author `group_cutout.api.json` in the ComfyUI UI**

In the ComfyUI web UI, build:
1. `LoadImage` (id `1`) — `image: "$$IMAGE"`
2. `RMBG (1038lab)` (id `2`) — `image: 1.0`, `model: "INSPYRENET"`, `sensitivity: 1.0`, `process_res: 1024`, `mask_blur: 0`, `mask_offset: 0`, `invert_output: false`, `refine_foreground: true`, `background: "Alpha"`
3. `SaveImage` (id `3`) — `images: 2.0`, `filename_prefix: "group_cutout"`

Export via **Save (API Format)** to `comfyui/workflows/group_cutout.api.json`. Verify the saved JSON references RGBA-capable output (PNG with alpha).

- [ ] **Step 3: Write the failing test for the HTTP helper**

`tests/group_photobooth/test_comfy_io.py`:
```python
import json
from pathlib import Path

from group_photobooth.comfy_io import substitute_template


def test_substitute_replaces_string_placeholders():
    tpl = {"nodes": {"1": {"inputs": {"image": "$$IMAGE",
                                       "ignored": "no marker"}}}}
    out = substitute_template(tpl, {"$$IMAGE": "photo_abc.png"})
    assert out["nodes"]["1"]["inputs"]["image"] == "photo_abc.png"
    assert out["nodes"]["1"]["inputs"]["ignored"] == "no marker"


def test_substitute_replaces_inside_nested_lists():
    tpl = {"nodes": {"3": {"inputs": {"mask_or_coordinates": "$$BBOX_JSON"}}}}
    out = substitute_template(tpl, {"$$BBOX_JSON": "[[10,20,30,40]]"})
    assert out["nodes"]["3"]["inputs"]["mask_or_coordinates"] == "[[10,20,30,40]]"


def test_substitute_leaves_unknown_markers_alone():
    tpl = {"x": "$$UNKNOWN"}
    out = substitute_template(tpl, {"$$OTHER": "foo"})
    assert out["x"] == "$$UNKNOWN"
```

- [ ] **Step 4: Run test to verify it fails**

Run: `uv run --no-project pytest tests/group_photobooth/test_comfy_io.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'group_photobooth.comfy_io'`

- [ ] **Step 5: Implement comfy_io.py**

`scripts/group_photobooth/comfy_io.py`:
```python
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
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run --no-project pytest tests/group_photobooth/test_comfy_io.py -v`
Expected: 3 passed

- [ ] **Step 7: Smoke-test workflow round-trips (gated on live ComfyUI)**

`tests/group_photobooth/test_comfy_io_e2e.py`:
```python
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
    # Bbox covering most of the image.
    h, w = bgr.shape[:2]
    bbox_json = f"[[{w//8},{h//8},{w*7//8},{h*7//8}]]"
    out = post_workflow(COMFY_URL, WORKFLOWS / "group_sam2_mask.api.json",
                        subs={"$$IMAGE": name, "$$BBOX_JSON": bbox_json},
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
    # RGBA expected.
    assert out.shape[-1] == 4
    assert out.shape[:2] == bgr.shape[:2]
    alpha = out[..., 3]
    assert (alpha > 0).any() and (alpha == 0).any()
```

Run: `COMFY_URL=http://127.0.0.1:8188 uv run --no-project pytest tests/group_photobooth/test_comfy_io_e2e.py -v -s`
Expected: 2 passed (each ~5-15s including upload + first-time model load).

- [ ] **Step 8: Commit**

```bash
git add comfyui/workflows/group_sam2_mask.api.json \
        comfyui/workflows/group_cutout.api.json \
        scripts/group_photobooth/comfy_io.py \
        tests/group_photobooth/test_comfy_io.py \
        tests/group_photobooth/test_comfy_io_e2e.py
git commit -m "feat(group-photobooth): ComfyUI workflows for SAM2 + cutout + HTTP helper"
```

---

## Task 3: Detection — YOLOv8 + buffalo_l (Python) + SAM2 via ComfyUI

**Files:**
- Create: `scripts/group_photobooth/detect.py`
- Create: `tests/group_photobooth/test_detect.py`
- Create: `tests/group_photobooth/fixtures/single_person.png`

**Dependencies install (one-time, before tests):**
```bash
uv pip install ultralytics opencv-python requests
# Pre-download YOLOv8n weights to ~/.cache/ultralytics/:
uv run --no-project python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

YOLO (CPU) and buffalo_l (already used by the photobooth via `swap_core.make_face_app()`) run in-process; SAM2 mask prediction is one HTTP call per person to ComfyUI using the workflow authored in Task 2.

- [ ] **Step 1: Stage the fixture**

```bash
cp data/importer/identities/id_00.png tests/group_photobooth/fixtures/single_person.png
```

- [ ] **Step 2: Write the failing test (gated on live ComfyUI for the SAM2 path)**

`tests/group_photobooth/test_detect.py`:
```python
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from group_photobooth import Person

FIXTURES = Path(__file__).parent / "fixtures"
COMFY_URL = os.environ.get("COMFY_URL")
needs_comfy = pytest.mark.skipif(not COMFY_URL,
                                 reason="set COMFY_URL=http://... to run")


@needs_comfy
def test_detect_single_person_returns_one_person():
    from group_photobooth.detect import detect_people
    img = cv2.imread(str(FIXTURES / "single_person.png"))
    assert img is not None
    persons = detect_people(img, comfy_url=COMFY_URL)
    assert len(persons) == 1
    p = persons[0]
    assert isinstance(p, Person)
    x1, y1, x2, y2 = p.body_bbox
    assert 0 <= x1 < x2 <= img.shape[1]
    assert 0 <= y1 < y2 <= img.shape[0]
    assert p.body_mask.shape == img.shape[:2]
    assert p.body_mask.dtype.name == "uint8"
    crop = p.body_mask[y1:y2, x1:x2]
    assert (crop > 0).mean() > 0.10
    assert p.face_bbox is not None


@needs_comfy
def test_detect_returns_empty_on_no_person():
    from group_photobooth.detect import detect_people
    img = np.full((400, 400, 3), 128, np.uint8)
    persons = detect_people(img, comfy_url=COMFY_URL)
    assert persons == []
```

- [ ] **Step 3: Run test to verify it fails**

Run: `COMFY_URL=http://127.0.0.1:8188 uv run --no-project pytest tests/group_photobooth/test_detect.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'group_photobooth.detect'`

- [ ] **Step 4: Implement detect.py**

`scripts/group_photobooth/detect.py`:
```python
"""YOLOv8-person (Python) + buffalo_l face (Python) + SAM2 mask (ComfyUI).

YOLO is fast on CPU (~100 ms / 1024² for yolov8n). buffalo_l is already a
process-cached dependency via swap_core. SAM2 runs as a ComfyUI workflow
so model weights stay loaded across calls — far cheaper than re-creating
a Python SAM2Predictor per driver run.
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

# Reach into scripts/ for buffalo_l (already used by photobooth).
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))


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
    bbox. Returns an HxW uint8 mask (0/255)."""
    bbox_json = dumps([[int(bbox[0]), int(bbox[1]),
                        int(bbox[2]), int(bbox[3])]])
    mask_bgr = post_workflow(
        comfy_url, WORKFLOWS / "group_sam2_mask.api.json",
        subs={"$$IMAGE": image_name, "$$BBOX_JSON": bbox_json},
        output_node_id="5")
    # MaskToImage emits a 3-channel grayscale; collapse to single channel.
    gray = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY) if mask_bgr.ndim == 3 else mask_bgr
    return ((gray > 127).astype(np.uint8) * 255)


def _detect_face_in_crop(crop_bgr: np.ndarray, app) -> tuple[int, int, int, int] | None:
    if crop_bgr.size == 0:
        return None
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    faces = app.get(rgb)
    if not faces:
        return None
    f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    return tuple(int(v) for v in f.bbox)


def detect_people(photo_bgr: np.ndarray, *,
                  comfy_url: str,
                  conf: float = 0.35,
                  iou: float = 0.45) -> list[Person]:
    """Detect every person in `photo_bgr`. SAM2 segmentation runs as a
    ComfyUI workflow at `comfy_url`. Persons without a detectable face are
    skipped (and logged)."""
    results = _yolo().predict(photo_bgr, classes=[0], conf=conf, iou=iou,
                              verbose=False)
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return []
    app = _face_app()
    # Upload the photo ONCE; reuse the server-side filename for every SAM2 call.
    image_name = upload_image(comfy_url, photo_bgr)
    persons: list[Person] = []
    for box in results[0].boxes.xyxy.cpu().numpy().astype(int):
        x1, y1, x2, y2 = box.tolist()
        crop = photo_bgr[y1:y2, x1:x2]
        face_local = _detect_face_in_crop(crop, app)
        if face_local is None:
            print(f"[detect] skipping bbox ({x1},{y1},{x2},{y2}): no face")
            continue
        mask = _sam2_mask(comfy_url, image_name, (x1, y1, x2, y2))
        if mask.shape != photo_bgr.shape[:2]:
            # Defensive: SAM2 workflow should preserve dims; resize if not.
            mask = cv2.resize(mask, (photo_bgr.shape[1], photo_bgr.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        fx1, fy1, fx2, fy2 = face_local
        face_bbox = (x1 + fx1, y1 + fy1, x1 + fx2, y1 + fy2)
        persons.append(Person(body_bbox=(x1, y1, x2, y2),
                              body_mask=mask, face_bbox=face_bbox))
    return persons
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `COMFY_URL=http://127.0.0.1:8188 uv run --no-project pytest tests/group_photobooth/test_detect.py -v -s`
Expected: 2 passed (first run ~5-10s for YOLO + SAM2 cold start; subsequent runs are sub-second per detection).

- [ ] **Step 6: Confirm skip behavior without ComfyUI**

Run: `uv run --no-project pytest tests/group_photobooth/test_detect.py -v`
Expected: 2 skipped

- [ ] **Step 7: Commit**

```bash
git add scripts/group_photobooth/detect.py tests/group_photobooth/test_detect.py tests/group_photobooth/fixtures/
git commit -m "feat(group-photobooth): person detection (YOLO+buffalo_l Python, SAM2 via ComfyUI)"
```

---

## Task 4: Silhouette — INSPYRENET cutout via ComfyUI

**Files:**
- Create: `scripts/group_photobooth/silhouette.py`
- Create: `tests/group_photobooth/test_silhouette.py`
- Create: `tests/group_photobooth/fixtures/doll_portrait.png`

INSPYRENET (from `1038lab/ComfyUI-RMBG`) is the recommended model for human/portrait foreground per `docs/research/2026-05-21-comfyui-compositor-toolkit.md`. Running it as a ComfyUI workflow keeps weights resident across calls and lets the engineer swap the model (BiRefNet-portrait, BEN2, etc.) by editing the workflow JSON's node settings rather than the Python code.

- [ ] **Step 1: Stage the fixture from a real photobooth output**

```bash
cp exp_output/photobooth_phase3/cells/id_00__cfg004/refined.png \
   tests/group_photobooth/fixtures/doll_portrait.png
```

- [ ] **Step 2: Write the failing test (gated on live ComfyUI)**

`tests/group_photobooth/test_silhouette.py`:
```python
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

FIXTURES = Path(__file__).parent / "fixtures"
COMFY_URL = os.environ.get("COMFY_URL")
needs_comfy = pytest.mark.skipif(not COMFY_URL,
                                 reason="set COMFY_URL=http://... to run")


@needs_comfy
def test_cutout_returns_rgba_same_dims():
    from group_photobooth.silhouette import cutout
    bgr = cv2.imread(str(FIXTURES / "doll_portrait.png"))
    rgba = cutout(bgr, comfy_url=COMFY_URL)
    assert rgba.shape == (bgr.shape[0], bgr.shape[1], 4)
    assert rgba.dtype == np.uint8
    alpha = rgba[..., 3]
    assert (alpha > 0).any()
    assert (alpha == 0).any()


@needs_comfy
def test_cutout_idempotent_on_already_cut_input():
    from group_photobooth.silhouette import cutout
    bgr = cv2.imread(str(FIXTURES / "doll_portrait.png"))
    rgba1 = cutout(bgr, comfy_url=COMFY_URL)
    rgba2 = cutout(cv2.cvtColor(rgba1, cv2.COLOR_BGRA2BGR), comfy_url=COMFY_URL)
    a1, a2 = rgba1[..., 3] > 128, rgba2[..., 3] > 128
    iou = (a1 & a2).sum() / max(1, (a1 | a2).sum())
    assert iou > 0.85
```

- [ ] **Step 3: Run test to verify it fails**

Run: `COMFY_URL=http://127.0.0.1:8188 uv run --no-project pytest tests/group_photobooth/test_silhouette.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement silhouette.py**

`scripts/group_photobooth/silhouette.py`:
```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `COMFY_URL=http://127.0.0.1:8188 uv run --no-project pytest tests/group_photobooth/test_silhouette.py -v -s`
Expected: 2 passed

- [ ] **Step 6: Confirm skip behavior without ComfyUI**

Run: `uv run --no-project pytest tests/group_photobooth/test_silhouette.py -v`
Expected: 2 skipped

- [ ] **Step 7: Commit**

```bash
git add scripts/group_photobooth/silhouette.py tests/group_photobooth/test_silhouette.py tests/group_photobooth/fixtures/doll_portrait.png
git commit -m "feat(group-photobooth): INSPYRENET cutout via ComfyUI workflow"
```

---

## Task 5: Layout — photo coords → background-canvas placements

**Files:**
- Create: `scripts/group_photobooth/layout.py`
- Create: `tests/group_photobooth/test_layout.py`

- [ ] **Step 1: Write the failing test**

`tests/group_photobooth/test_layout.py`:
```python
import numpy as np

from group_photobooth import Person, Placement
from group_photobooth.layout import solve_placements


def _person(x1, y1, x2, y2, photo_hw):
    h, w = photo_hw
    m = np.zeros((h, w), np.uint8)
    m[y1:y2, x1:x2] = 255
    return Person(body_bbox=(x1, y1, x2, y2), body_mask=m,
                  face_bbox=(x1 + 5, y1 + 5, x1 + 25, y1 + 25))


def test_painter_order_sorts_back_to_front():
    photo_hw = (600, 800)
    bg_hw = (600, 800)  # same — identity transform
    persons = [
        _person(100, 100, 250, 400, photo_hw),   # higher up = farther back
        _person(450, 200, 600, 500, photo_hw),   # lower bottom = closer
    ]
    placements = solve_placements(persons, photo_hw, bg_hw)
    assert len(placements) == 2
    # Painter's order: lower z_order is painted first (back).
    assert placements[0].z_order < placements[1].z_order
    # The person with the smaller y_bottom is farther back.
    assert placements[0].y_bottom < placements[1].y_bottom


def test_letterbox_scales_correctly():
    # Photo 400x800 (portrait), background 800x800 (square) — letterbox.
    photo_hw = (800, 400)
    bg_hw = (800, 800)
    # A person 100..200 horizontally, 200..600 vertically (height=400).
    persons = [_person(100, 200, 200, 600, photo_hw)]
    placements = solve_placements(persons, photo_hw, bg_hw)
    assert len(placements) == 1
    p = placements[0]
    # Scale s = min(800/400, 800/800) = 1.0 (height-bound for portrait photo).
    # x offset (letterbox) = (800 - 400*1.0)/2 = 200.
    # Person bbox center_x in photo = 150 → in canvas = 200 + 150 = 350.
    assert p.x_center == 350
    # Person y_bottom in photo = 600 → in canvas = 0 + 600 = 600.
    assert p.y_bottom == 600
    # Height in canvas = 400 * 1.0 = 400.
    assert p.height == 400


def test_empty_persons_returns_empty_placements():
    assert solve_placements([], (600, 800), (600, 800)) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-project pytest tests/group_photobooth/test_layout.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement layout.py**

`scripts/group_photobooth/layout.py`:
```python
"""Map per-person photo-coord geometry to background-canvas placements.

Letterboxes the photo's aspect into the background's aspect (preserves
proportions, never stretches). Painter's order = ascending y_bottom so
closer-to-camera dolls (lower in image) paint last.
"""
from __future__ import annotations

from group_photobooth import Person, Placement


def solve_placements(persons: list[Person],
                     photo_hw: tuple[int, int],
                     bg_hw: tuple[int, int]) -> list[Placement]:
    """Project each person's body bbox to background-canvas coords.

    photo_hw, bg_hw are (height, width). The photo is letterboxed into the
    background — uniform scale by `min(bg_h/photo_h, bg_w/photo_w)` and
    centered on the off-axis. Doll height matches person bbox height in
    the projected coords.
    """
    if not persons:
        return []
    ph, pw = photo_hw
    bh, bw = bg_hw
    s = min(bh / ph, bw / pw)
    dx = (bw - pw * s) / 2.0  # horizontal letterbox offset
    dy = (bh - ph * s) / 2.0  # vertical letterbox offset

    indexed: list[tuple[Person, int, int, int]] = []
    for person in persons:
        x1, y1, x2, y2 = person.body_bbox
        cx_photo = (x1 + x2) / 2.0
        yb_photo = float(y2)
        h_photo = float(y2 - y1)
        x_center = int(round(dx + cx_photo * s))
        y_bottom = int(round(dy + yb_photo * s))
        height = int(round(h_photo * s))
        indexed.append((person, x_center, y_bottom, height))

    # Painter's order: lower y_bottom → farther back → painted first → smaller z.
    indexed.sort(key=lambda t: t[2])
    return [Placement(x_center=xc, y_bottom=yb, height=h, z_order=z)
            for z, (_, xc, yb, h) in enumerate(indexed)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-project pytest tests/group_photobooth/test_layout.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/group_photobooth/layout.py tests/group_photobooth/test_layout.py
git commit -m "feat(group-photobooth): layout solver with letterbox + painter's order"
```

---

## Task 6: Composite — painter's-order alpha blend + optional polish

**Files:**
- Create: `scripts/group_photobooth/composite.py`
- Create: `tests/group_photobooth/test_composite.py`

- [ ] **Step 1: Write the failing test**

`tests/group_photobooth/test_composite.py`:
```python
import cv2
import numpy as np

from group_photobooth import Placement
from group_photobooth.composite import composite, lab_match, drop_shadow


def _solid_doll(h, w, color):
    """Return a fully-opaque RGBA solid-color doll silhouette."""
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., :3] = color
    rgba[..., 3] = 255
    return rgba


def test_composite_pastes_at_foot_anchor():
    bg = np.full((600, 800, 3), 128, np.uint8)
    doll = _solid_doll(400, 200, (200, 100, 50))  # 200w × 400h
    placement = Placement(x_center=400, y_bottom=600, height=400, z_order=0)
    out = composite(bg, [doll], [placement])
    assert out.shape == bg.shape
    # The doll should overwrite the background at (300..500, 200..600).
    assert (out[200:600, 300:500] != 128).all()
    # Outside that region, untouched.
    assert (out[:200, :] == 128).all()


def test_painter_order_front_doll_occludes_back_doll():
    bg = np.full((600, 800, 3), 0, np.uint8)
    back = _solid_doll(400, 200, (255, 0, 0))   # blue (BGR)
    front = _solid_doll(400, 200, (0, 255, 0))  # green
    placements = [
        Placement(x_center=400, y_bottom=400, height=400, z_order=0),  # back
        Placement(x_center=400, y_bottom=600, height=400, z_order=1),  # front
    ]
    out = composite(bg, [back, front], placements)
    # In the overlap (300..500 x, ~200..400 y), the front (green) wins.
    overlap_px = out[300:400, 300:500]
    # Green > Blue
    assert (overlap_px[..., 1] > overlap_px[..., 0]).mean() > 0.95


def test_lab_match_shifts_mean_toward_target():
    src = np.full((100, 100, 3), [50, 100, 200], np.uint8)  # bluish
    alpha = np.full((100, 100), 255, np.uint8)
    target_lab = (60.0, 0.0, 0.0)  # neutral gray-ish in Lab
    out = lab_match(src, alpha, target_lab, max_dL=8.0)
    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
    out_lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    # The mean L must have shifted toward target_lab[0]=60, capped at ±8.
    dL_src = src_lab[..., 0].mean() - target_lab[0]
    dL_out = out_lab[..., 0].mean() - target_lab[0]
    assert abs(dL_out) < abs(dL_src)


def test_drop_shadow_darkens_area_under_doll():
    bg = np.full((600, 800, 3), 200, np.uint8)
    placement = Placement(x_center=400, y_bottom=400, height=200, z_order=0)
    out = drop_shadow(bg, placement, doll_w=100, opacity=0.5,
                      offset_yx=(8, 0), blur_px=12)
    # Background under the foot anchor should be darker than the corners.
    foot = out[395:410, 380:420].mean()
    corner = out[0:30, 0:30].mean()
    assert foot < corner
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-project pytest tests/group_photobooth/test_composite.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement composite.py**

`scripts/group_photobooth/composite.py`:
```python
"""Painter's-order alpha-composite RGBA dolls onto an opaque background.

Polish stages — Lab color matching and elliptical drop shadow — are
exposed as standalone functions so the driver can opt in/out per run.
"""
from __future__ import annotations

import cv2
import numpy as np

from group_photobooth import Placement


def composite(background_bgr: np.ndarray,
              dolls_rgba: list[np.ndarray],
              placements: list[Placement]) -> np.ndarray:
    """Paste each doll onto a copy of the background at its placement.

    `dolls_rgba` and `placements` must be parallel lists. Placements are
    sorted by `z_order` ascending and pasted in that order.
    """
    assert len(dolls_rgba) == len(placements), \
        "dolls and placements must be parallel lists"
    out = background_bgr.copy()
    bh, bw = out.shape[:2]
    paired = sorted(zip(dolls_rgba, placements), key=lambda p: p[1].z_order)
    for doll, pl in paired:
        # Resize doll so its height matches pl.height; width preserves aspect.
        dh, dw = doll.shape[:2]
        new_h = max(1, pl.height)
        new_w = max(1, int(round(dw * new_h / dh)))
        dol = cv2.resize(doll, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # Foot-anchor: doll's bottom-center at (x_center, y_bottom).
        x0 = pl.x_center - new_w // 2
        y0 = pl.y_bottom - new_h
        # Clip against canvas bounds.
        sx0 = max(0, -x0); sy0 = max(0, -y0)
        sx1 = new_w - max(0, (x0 + new_w) - bw)
        sy1 = new_h - max(0, (y0 + new_h) - bh)
        if sx0 >= sx1 or sy0 >= sy1:
            continue
        dx0 = max(0, x0); dy0 = max(0, y0)
        dx1 = dx0 + (sx1 - sx0); dy1 = dy0 + (sy1 - sy0)
        src = dol[sy0:sy1, sx0:sx1]
        alpha = src[..., 3:4].astype(np.float32) / 255.0
        out[dy0:dy1, dx0:dx1] = (
            src[..., :3].astype(np.float32) * alpha
            + out[dy0:dy1, dx0:dx1].astype(np.float32) * (1.0 - alpha)
        ).astype(np.uint8)
    return out


def lab_match(bgr: np.ndarray, alpha: np.ndarray,
              target_lab: tuple[float, float, float],
              max_dL: float = 8.0,
              max_dab: float = 4.0) -> np.ndarray:
    """Shift mean Lab of foreground pixels (alpha>0) toward `target_lab`.

    Caps the shift to (max_dL, max_dab, max_dab) so dolls keep their
    character.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    fg = alpha > 0
    if not fg.any():
        return bgr
    mean = np.array([lab[..., c][fg].mean() for c in range(3)])
    diff = np.array(target_lab) - mean
    diff[0] = np.clip(diff[0], -max_dL, max_dL)
    diff[1] = np.clip(diff[1], -max_dab, max_dab)
    diff[2] = np.clip(diff[2], -max_dab, max_dab)
    for c in range(3):
        lab[..., c] = np.where(fg, lab[..., c] + diff[c], lab[..., c])
    lab[..., 0] = np.clip(lab[..., 0], 0, 100)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def drop_shadow(background_bgr: np.ndarray, placement: Placement,
                doll_w: int, opacity: float = 0.4,
                offset_yx: tuple[int, int] = (10, 0),
                blur_px: int = 16) -> np.ndarray:
    """Render an elliptical drop shadow under a doll's foot anchor.

    `doll_w` is the doll's pixel width on the canvas, used to size the
    ellipse. The shadow is darkened-multiply over the background.
    """
    out = background_bgr.copy()
    bh, bw = out.shape[:2]
    cx = placement.x_center + offset_yx[1]
    cy = placement.y_bottom + offset_yx[0]
    ax = max(4, doll_w // 2)
    ay = max(2, doll_w // 6)
    mask = np.zeros((bh, bw), np.uint8)
    cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    if blur_px > 0:
        k = 2 * blur_px + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    factor = 1.0 - (mask.astype(np.float32) / 255.0) * opacity
    out = (out.astype(np.float32) * factor[..., None]).astype(np.uint8)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-project pytest tests/group_photobooth/test_composite.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/group_photobooth/composite.py tests/group_photobooth/test_composite.py
git commit -m "feat(group-photobooth): painter's-order composite + lab_match + drop_shadow"
```

---

## Task 7: Background library + manifest

**Files:**
- Create: `assets/backgrounds/manifest.json`
- Create: `assets/backgrounds/blank_studio/bg.png`
- Create: `scripts/group_photobooth/backgrounds.py`
- Create: `tests/group_photobooth/test_backgrounds.py`

- [ ] **Step 1: Generate the seed background**

```python
# scripts/group_photobooth/_make_bg_blank.py (one-shot, do not commit)
import cv2, numpy as np
from pathlib import Path

W, H = 1024, 1024
# Smooth vertical gradient — neutral studio backdrop.
g = np.linspace(220, 180, H, dtype=np.float32)
bg = np.tile(g[:, None, None], (1, W, 3)).astype(np.uint8)
out = Path("assets/backgrounds/blank_studio/bg.png")
out.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out), bg)
```

Run once: `uv run --no-project python scripts/group_photobooth/_make_bg_blank.py` then delete the script.

- [ ] **Step 2: Author the manifest**

`assets/backgrounds/manifest.json`:
```json
[
  {
    "id": "blank_studio",
    "description": "Neutral studio backdrop — soft vertical gray gradient.",
    "dims": [1024, 1024],
    "palette_lab_mean": [80.0, 0.0, 0.0],
    "lighting_hint": "cool-flat"
  }
]
```

- [ ] **Step 3: Write the failing test**

`tests/group_photobooth/test_backgrounds.py`:
```python
from group_photobooth.backgrounds import load_background, list_backgrounds


def test_list_backgrounds_includes_blank_studio():
    ids = [b["id"] for b in list_backgrounds()]
    assert "blank_studio" in ids


def test_load_background_returns_image_and_metadata():
    img, meta = load_background("blank_studio")
    assert img.shape == (1024, 1024, 3)
    assert meta["id"] == "blank_studio"
    assert tuple(meta["dims"]) == (1024, 1024)
    assert "palette_lab_mean" in meta


def test_load_background_unknown_id_raises():
    import pytest
    with pytest.raises(KeyError):
        load_background("does_not_exist")
```

- [ ] **Step 4: Run test to verify it fails**

Run: `uv run --no-project pytest tests/group_photobooth/test_backgrounds.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 5: Implement backgrounds.py**

`scripts/group_photobooth/backgrounds.py`:
```python
"""Background library — pre-rendered scenes + metadata.

Manifest lives at `assets/backgrounds/manifest.json`. Each background's
PNG is at `assets/backgrounds/<id>/bg.png`.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BG_DIR = ROOT / "assets" / "backgrounds"


@lru_cache(maxsize=1)
def _manifest() -> list[dict]:
    with open(BG_DIR / "manifest.json") as f:
        return json.load(f)


def list_backgrounds() -> list[dict]:
    """Return the manifest list as-is."""
    return list(_manifest())


def load_background(bg_id: str) -> tuple[np.ndarray, dict]:
    """Return (image_bgr, metadata_dict) for `bg_id`.

    Raises KeyError if the id is not in the manifest.
    """
    for entry in _manifest():
        if entry["id"] == bg_id:
            img = cv2.imread(str(BG_DIR / bg_id / "bg.png"))
            if img is None:
                raise FileNotFoundError(
                    f"manifest lists {bg_id} but bg.png is missing")
            return img, dict(entry)
    raise KeyError(f"unknown background id: {bg_id}")
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run --no-project pytest tests/group_photobooth/test_backgrounds.py -v`
Expected: 3 passed

- [ ] **Step 7: Commit**

```bash
git add assets/backgrounds/ scripts/group_photobooth/backgrounds.py tests/group_photobooth/test_backgrounds.py
git commit -m "feat(group-photobooth): background library + blank_studio seed"
```

---

## Task 8: Face renderer — thin wrapper around Phase 3 heavy mix

**Files:**
- Create: `scripts/group_photobooth/face_renderer.py`
- Create: `tests/group_photobooth/test_face_renderer.py`

This module reuses the photobooth's existing render + swap by calling its
lower-level functions, NOT `driver.run_cell` (which is sweep-flavoured).
The Phase 3 heavy-mix config is locked here as a module-level constant.

- [ ] **Step 1: Write the failing test**

`tests/group_photobooth/test_face_renderer.py`:
```python
from group_photobooth.face_renderer import HEAVY_MIX_CFG


def test_heavy_mix_cfg_matches_phase3_cfg004():
    # The spec locks: swap_weight=0.10, natural_1024, canny soft,
    # cn_strength=0.85, refine_denoise=0.00, demo_inject=on.
    assert HEAVY_MIX_CFG["swap_weight"] == 0.10
    assert HEAVY_MIX_CFG["face_pixel_budget"] == "natural_1024"
    assert HEAVY_MIX_CFG["cn_condition"] == "canny"
    assert HEAVY_MIX_CFG["canny_preset"] == "soft"
    assert HEAVY_MIX_CFG["cn_strength"] == 0.85
    assert HEAVY_MIX_CFG["refine_denoise"] == 0.00
    assert HEAVY_MIX_CFG["demo_inject"] == "on"
```

(The end-to-end render-one-face test is gated behind a live ComfyUI server
and is exercised by the smoke test in Task 9, not here.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-project pytest tests/group_photobooth/test_face_renderer.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement face_renderer.py**

`scripts/group_photobooth/face_renderer.py`:
```python
"""Single-face photobooth call — Phase 3 heavy mix, locked.

Wraps the existing photobooth machinery (preprocess + cn_workflow +
swap_core.swap_identity) so the group pipeline sees one function:
`render_doll(face_crop_bgr, comfy_url, seed, demo) -> portrait_bgr`.

Phase 3 "heavy mix" verdict (cfg004): swap_weight=0.10, embedding pulled
28% toward doll, natural_1024 / canny / soft, demo_inject on, refine off.
See docs/superpowers/specs/2026-05-21-group-photobooth-pipeline-design.md
section "Locked decisions".
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Reach into scripts/ for the photobooth sweep modules.
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

    `demo` is the demographic dict ({"gender": ..., "age": ..., "race": ...})
    used for the prompt injection.
    """
    cfg = HEAVY_MIX_CFG
    prompt = ph_driver.make_prompt("group", demo, cfg["demo_inject"])
    app = swap_core.make_face_app()
    swapper = swap_core.load_swapper()
    src_face = swap_core.detect_source(app, face_crop_bgr)
    if src_face is None:
        raise ValueError("no face detected in crop")

    ctrl, render_hw = ph_pp.build_control(
        app, comfy_url, face_crop_bgr, cfg["face_pixel_budget"],
        cfg["cn_condition"], cfg["canny_preset"])
    ctrl_name = ph_pp.upload_control(comfy_url, ctrl, "groupbooth")
    wf = ph_driver.cn_workflow(comfy_url, ctrl_name=ctrl_name, prompt=prompt,
                               render_hw=render_hw,
                               cn_strength=cfg["cn_strength"], seed=seed)
    render = ph_driver.comfy_submit(comfy_url, wf)
    swap, _det_mode, _det_pre = swap_core.swap_identity(
        app, swapper, render, src_face, collapse=True, restore=False,
        swap_weight=cfg["swap_weight"])
    return swap
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-project pytest tests/group_photobooth/test_face_renderer.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/group_photobooth/face_renderer.py tests/group_photobooth/test_face_renderer.py
git commit -m "feat(group-photobooth): face_renderer wraps Phase 3 heavy mix"
```

---

## Task 9: Driver — orchestrator with resumable per-person cache

**Files:**
- Create: `scripts/group_photobooth/driver.py`
- Create: `tests/group_photobooth/test_driver_unit.py`

- [ ] **Step 1: Write the failing unit test**

`tests/group_photobooth/test_driver_unit.py`:
```python
import numpy as np

from group_photobooth import Person, Placement
from group_photobooth.driver import _seed_for_face, _crop_face


def test_seed_for_face_is_deterministic():
    s1 = _seed_for_face("photo_a.png", 0)
    s2 = _seed_for_face("photo_a.png", 0)
    assert s1 == s2
    s3 = _seed_for_face("photo_a.png", 1)
    assert s3 != s1
    s4 = _seed_for_face("photo_b.png", 0)
    assert s4 != s1


def test_crop_face_returns_bgr_with_margin():
    img = np.full((400, 600, 3), 128, np.uint8)
    person = Person(body_bbox=(100, 50, 300, 380),
                    body_mask=np.zeros((400, 600), np.uint8),
                    face_bbox=(180, 90, 240, 160))
    crop = _crop_face(img, person, margin_frac=0.3)
    assert crop.ndim == 3 and crop.shape[2] == 3
    # Crop must be larger than the face bbox by the margin.
    fw = 240 - 180; fh = 160 - 90
    assert crop.shape[0] > fh
    assert crop.shape[1] > fw
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-project pytest tests/group_photobooth/test_driver_unit.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement driver.py**

`scripts/group_photobooth/driver.py`:
```python
"""Group photobooth orchestrator.

CLI:
    uv run --no-project python -m scripts.group_photobooth.driver \\
        --photo path/to/group.jpg --background blank_studio \\
        --out exp_output/group_photobooth/run01.png

Pipeline (Approach A from the spec):
    1. detect.detect_people(photo) → list[Person]
    2. for each person: face_renderer.render_doll(face_crop) → portrait
    3. for each portrait: silhouette.cutout(portrait) → RGBA
    4. layout.solve_placements(persons, photo_hw, bg_hw) → list[Placement]
    5. composite.composite(bg, dolls_rgba, placements)
    6. optional polish: lab_match per doll, drop_shadow per placement

All intermediates cache to <out_dir>/people/<i>/ so reruns skip work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from group_photobooth import Person
from group_photobooth.backgrounds import load_background
from group_photobooth.composite import composite, drop_shadow, lab_match
from group_photobooth.detect import detect_people
from group_photobooth.face_renderer import render_doll
from group_photobooth.layout import solve_placements
from group_photobooth.silhouette import cutout


def _seed_for_face(photo_id: str, face_index: int) -> int:
    h = hashlib.md5(f"{photo_id}__{face_index}".encode()).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


def _crop_face(photo_bgr: np.ndarray, person: Person,
               margin_frac: float = 0.35) -> np.ndarray:
    assert person.face_bbox is not None
    x1, y1, x2, y2 = person.face_bbox
    fw, fh = x2 - x1, y2 - y1
    m = int(round(max(fw, fh) * margin_frac))
    H, W = photo_bgr.shape[:2]
    cx1 = max(0, x1 - m); cy1 = max(0, y1 - m)
    cx2 = min(W, x2 + m); cy2 = min(H, y2 + m)
    return photo_bgr[cy1:cy2, cx1:cx2].copy()


def run(photo_path: Path, background_id: str, out_dir: Path,
        comfy_url: str, demo: dict[str, str] | None = None,
        polish: bool = False) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    people_dir = out_dir / "people"
    people_dir.mkdir(exist_ok=True)

    photo = cv2.imread(str(photo_path))
    if photo is None:
        raise FileNotFoundError(f"cannot read photo: {photo_path}")
    persons = detect_people(photo, comfy_url=comfy_url)
    print(f"[group_photobooth] detected {len(persons)} person(s)")
    if not persons:
        bg, _meta = load_background(background_id)
        out_path = out_dir / "result.png"
        cv2.imwrite(str(out_path), bg)
        return out_path

    photo_id = photo_path.stem
    demo = demo or {"gender": "person", "age": "adult", "race": ""}

    dolls_rgba: list[np.ndarray] = []
    for i, person in enumerate(persons):
        pdir = people_dir / f"{i:02d}"
        pdir.mkdir(exist_ok=True)
        face_p = pdir / "face.png"
        doll_p = pdir / "doll.png"
        rgba_p = pdir / "doll_rgba.png"

        if not face_p.exists():
            cv2.imwrite(str(face_p), _crop_face(photo, person))
        if not doll_p.exists():
            face_crop = cv2.imread(str(face_p))
            seed = _seed_for_face(photo_id, i)
            doll = render_doll(face_crop, comfy_url, seed=seed, demo=demo)
            cv2.imwrite(str(doll_p), doll)
        if not rgba_p.exists():
            doll = cv2.imread(str(doll_p))
            rgba = cutout(doll, comfy_url=comfy_url)
            cv2.imwrite(str(rgba_p), rgba)
        dolls_rgba.append(cv2.imread(str(rgba_p), cv2.IMREAD_UNCHANGED))

    bg, meta = load_background(background_id)
    placements = solve_placements(persons, photo.shape[:2], bg.shape[:2])

    if polish:
        target_lab = tuple(meta["palette_lab_mean"])
        for k, doll in enumerate(dolls_rgba):
            alpha = doll[..., 3]
            doll[..., :3] = lab_match(doll[..., :3], alpha, target_lab)
        canvas = bg.copy()
        for pl in placements:
            doll_h = pl.height
            doll_idx = next(i for i, p in enumerate(placements) if p is pl)
            doll_w = int(round(dolls_rgba[doll_idx].shape[1]
                               * doll_h / dolls_rgba[doll_idx].shape[0]))
            canvas = drop_shadow(canvas, pl, doll_w=doll_w)
        result = composite(canvas, dolls_rgba, placements)
    else:
        result = composite(bg, dolls_rgba, placements)

    out_path = out_dir / "result.png"
    cv2.imwrite(str(out_path), result)
    print(f"[group_photobooth] wrote {out_path}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--photo", type=Path, required=True)
    p.add_argument("--background", type=str, required=True)
    p.add_argument("--out", type=Path, required=True,
                   help="output directory (intermediates + result.png)")
    p.add_argument("--comfy-url", type=str, default="http://127.0.0.1:8188")
    p.add_argument("--demo-json", type=str, default=None,
                   help="JSON with gender/age/race for prompt injection")
    p.add_argument("--polish", action="store_true",
                   help="enable Lab match + drop shadow polish")
    args = p.parse_args()
    demo = json.loads(args.demo_json) if args.demo_json else None
    run(args.photo, args.background, args.out, args.comfy_url,
        demo=demo, polish=args.polish)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-project pytest tests/group_photobooth/test_driver_unit.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/group_photobooth/driver.py tests/group_photobooth/test_driver_unit.py
git commit -m "feat(group-photobooth): driver with resumable per-person cache + CLI"
```

---

## Task 10: End-to-end smoke tests (gated on live ComfyUI)

**Files:**
- Create: `tests/group_photobooth/test_smoke_e2e.py`

These tests need a live ComfyUI (the existing local 5090 instance on
`http://127.0.0.1:8188`). They are skipped when `COMFY_URL` env var is
unset, so they don't break CI/lint runs.

- [ ] **Step 1: Write the smoke tests**

`tests/group_photobooth/test_smoke_e2e.py`:
```python
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

COMFY_URL = os.environ.get("COMFY_URL")
pytestmark = pytest.mark.skipif(not COMFY_URL,
                                reason="set COMFY_URL=http://... to run")


def _build_two_person_synth(out_path: Path) -> Path:
    """Stitch two `data/importer/identities/*.png` side-by-side on a blank
    canvas to make a synthetic 2-person photo. Real faces, real bodies."""
    ROOT = Path(__file__).resolve().parents[2]
    a = cv2.imread(str(ROOT / "data/importer/identities/id_00.png"))
    b = cv2.imread(str(ROOT / "data/importer/identities/id_11.png"))
    h = max(a.shape[0], b.shape[0])
    a = cv2.resize(a, (int(a.shape[1] * h / a.shape[0]), h))
    b = cv2.resize(b, (int(b.shape[1] * h / b.shape[0]), h))
    canvas = np.full((h, a.shape[1] + b.shape[1] + 100, 3), 220, np.uint8)
    canvas[:, :a.shape[1]] = a
    canvas[:, a.shape[1] + 100:] = b
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    return out_path


def test_n1_path_runs_end_to_end(tmp_path):
    from group_photobooth.driver import run

    ROOT = Path(__file__).resolve().parents[2]
    photo = ROOT / "data/importer/identities/id_00.png"
    out = run(photo, "blank_studio", tmp_path, comfy_url=COMFY_URL)
    assert out.exists()
    img = cv2.imread(str(out))
    assert img is not None
    assert img.shape == (1024, 1024, 3)
    # Background was gray ~200; if a doll was composited, some pixels
    # must differ substantially.
    diff = np.abs(img.astype(int) - 200).mean()
    assert diff > 5.0, "result looks like the untouched background"


def test_n2_synth_runs_end_to_end(tmp_path):
    from group_photobooth.driver import run

    photo = _build_two_person_synth(tmp_path / "synth.png")
    out = run(photo, "blank_studio", tmp_path / "run", comfy_url=COMFY_URL)
    assert out.exists()
    # Two per-person dirs must exist.
    people = sorted((tmp_path / "run" / "people").iterdir())
    assert len(people) == 2
    for pdir in people:
        assert (pdir / "doll.png").exists()
        assert (pdir / "doll_rgba.png").exists()
```

- [ ] **Step 2: Run tests (with live ComfyUI)**

Run:
```bash
COMFY_URL=http://127.0.0.1:8188 uv run --no-project \
    pytest tests/group_photobooth/test_smoke_e2e.py -v -s
```
Expected: 2 passed (each takes ~30-60 s due to actual rendering)

- [ ] **Step 3: Run tests (without ComfyUI) to confirm skip behavior**

Run: `uv run --no-project pytest tests/group_photobooth/test_smoke_e2e.py -v`
Expected: 2 skipped

- [ ] **Step 4: Commit**

```bash
git add tests/group_photobooth/test_smoke_e2e.py
git commit -m "test(group-photobooth): end-to-end smoke tests (N=1, N=2)"
```

---

## Task 11: Manual visual verification + runbook

**Files:**
- Create: `docs/research/2026-05-21-group-photobooth-architecture.md`

- [ ] **Step 1: Run the pipeline on a real two-person photo**

Stage a real input under `data/group_photobooth_inputs/` and run:
```bash
COMFY_URL=http://127.0.0.1:8188 uv run --no-project \
    python -m scripts.group_photobooth.driver \
    --photo data/group_photobooth_inputs/test_two_people.jpg \
    --background blank_studio \
    --out exp_output/group_photobooth/test_two_people/
```

Open `exp_output/group_photobooth/test_two_people/result.png` and verify:
- N dolls present, one per person in the source.
- Dolls are sized to person heights in the input photo.
- Painter's order is correct (lower person occludes higher).
- No dolls are clipped at the canvas edge.
- Per-person `people/0/face.png`, `doll.png`, `doll_rgba.png` exist.

- [ ] **Step 2: Run again with polish**

```bash
COMFY_URL=http://127.0.0.1:8188 uv run --no-project \
    python -m scripts.group_photobooth.driver \
    --photo data/group_photobooth_inputs/test_two_people.jpg \
    --background blank_studio \
    --out exp_output/group_photobooth/test_two_people_polish/ \
    --polish
```

Verify dolls now have drop shadows and a subtle color shift toward the
background palette.

- [ ] **Step 3: Author the runbook**

`docs/research/2026-05-21-group-photobooth-architecture.md`:
- Frontmatter: `status: live, topic: photobooth-sweep`.
- Sections: pipeline overview (paste the spec's ASCII diagram), CLI
  reference (the two commands above), failure modes observed (no-face
  skipped, lying-down warning, edge clipping), per-person cache layout,
  background-library schema.

- [ ] **Step 4: Update the topic index**

Append to `docs/research/_topics/photobooth-sweep.md` under "Spec / plan":
- Plan: `docs/superpowers/plans/2026-05-21-group-photobooth.md`
- Runbook: `docs/research/2026-05-21-group-photobooth-architecture.md`

- [ ] **Step 5: Commit**

```bash
git add docs/research/2026-05-21-group-photobooth-architecture.md \
        docs/research/_topics/photobooth-sweep.md \
        exp_output/group_photobooth/ 2>/dev/null || true
git commit -m "docs(group-photobooth): runbook + manual verification artifacts"
```

---

## Self-review

**Spec coverage:**
- Inputs/outputs → Task 9 (driver CLI takes photo + background id, writes result.png).
- Locked decisions (heavy mix, library backgrounds, side-by-side, SAM mask) → Tasks 3, 7, 8.
- Approach A pipeline → Tasks 2, 3, 4, 5, 6, 9.
- Components table — `comfy_io.py` + workflow JSONs (Task 2), `detect.py` (Task 3), `silhouette.py` (Task 4), `layout.py` (Task 5), `composite.py` (Task 6), `driver.py` (Task 9) — all present. Added `face_renderer.py` (Task 8) and `backgrounds.py` (Task 7); both are thin and serve clear responsibilities.
- ComfyUI custom-node installs (Kijai SAM2, 1038lab RMBG) and workflow scaffolding → Task 2.
- Layout solver letterbox + painter's order → Task 5.
- Occlusion ordering → Task 5 + Task 6 (z_order tested with overlapping dolls).
- N=1 case → Task 10 test_n1_path.
- Lying-down / no-face person handling → driver.py skips person if `face_bbox is None`; detect.py logs `[detect] skipping bbox … no face` and emits no Person.
- Polish stage off by default → Task 9 CLI `--polish` flag.
- Background library schema → Task 7.
- Resumability → Task 9 driver skip-if-exists per file.
- Testing checklist (3 smokes) → Task 10 covers N=1 and N=2; occlusion is covered by unit test in Task 6.
- Cross-reference to chibi pivot — preserved in spec only, no plan task (chibi is out of scope).

**Placeholder scan:** No TBD/TODO. Every code step has a complete block. Every test step has expected output.

**Type consistency:**
- `Person.body_bbox: tuple[int, int, int, int]` — used in detect.py, layout.py, driver._crop_face. Consistent.
- `Person.face_bbox: tuple[int, int, int, int] | None` — None case handled in detect.py (skip) and driver (assertion only after detect filters).
- `Placement(x_center, y_bottom, height, z_order)` — used in layout.py producer and composite.py consumer. Consistent.
- `render_doll(face_crop_bgr, comfy_url, *, seed, demo) -> np.ndarray` — face_renderer.py producer, driver.py consumer. Consistent.
- `detect_people(photo_bgr, *, comfy_url, conf, iou) -> list[Person]` — detect.py producer, driver.py consumer. Consistent.
- `cutout(bgr, *, comfy_url) -> np.ndarray (BGRA)` — silhouette.py producer, driver.py consumer. Consistent.
- `composite(bg_bgr, dolls_rgba, placements) -> np.ndarray` — composite.py producer, driver.py consumer. Consistent.
- `post_workflow(comfy_url, workflow_path, subs, output_node_id, *, timeout_s, unchanged) -> np.ndarray` and `upload_image(comfy_url, bgr, name=None) -> str` — comfy_io.py producers, detect.py / silhouette.py consumers. Consistent.
