# Group Photobooth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a photo of 1-5 people into a matryoshka group portrait composited onto a chosen pre-rendered background.

**Architecture:** Per-person YOLO + SAM2 detection/segmentation → existing single-face photobooth (Phase 3 heavy mix, `swap_weight=0.10`) for each face → rembg/INSPYRENET cutout of each doll → Pillow alpha-composite onto a background from `assets/backgrounds/` in painter's order. All new work runs CPU-side except the existing photobooth call, which already brokers GPU via the ComfyUI HTTP API.

**Tech Stack:** Python 3.12, ultralytics YOLOv8, segment-anything-2 (PyTorch), rembg (INSPYRENET), Pillow, opencv-python, numpy, scipy. Reuses `scripts/photobooth_sweep/` and `scripts/swap_core.py` verbatim.

**Spec:** `docs/superpowers/specs/2026-05-21-group-photobooth-pipeline-design.md`
**Tooling research:** `docs/research/2026-05-21-comfyui-compositor-toolkit.md`

---

## File structure

```
scripts/group_photobooth/
  __init__.py           # exports Person, Placement, dataclasses
  detect.py             # YOLOv8-person + SAM2 → list[Person]
  silhouette.py         # rembg/INSPYRENET cutout → RGBA
  layout.py             # photo coords → background-canvas placements + painter z-order
  composite.py          # alpha blend + optional Lab match + drop shadow
  face_renderer.py      # thin wrapper around photobooth Phase-3 heavy mix
  driver.py             # CLI orchestrator, resumable per-person cache
assets/backgrounds/
  manifest.json         # array of {id, description, dims, palette_lab_mean, lighting_hint}
  blank_studio/bg.png   # shipped seed background
tests/group_photobooth/
  __init__.py
  conftest.py           # shared fixtures (synthetic photo, background)
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

## Task 2: Detection — YOLOv8 person + SAM2 body mask

**Files:**
- Create: `scripts/group_photobooth/detect.py`
- Create: `tests/group_photobooth/test_detect.py`
- Create: `tests/group_photobooth/fixtures/synthetic_two_people.png`

**Dependencies install (one-time, before tests):**
```bash
uv pip install ultralytics opencv-python rembg "onnxruntime>=1.18"
# SAM2 (Meta) — install via the kijai fork-compatible package:
uv pip install "git+https://github.com/facebookresearch/segment-anything-2.git"
```

Auto-downloaded model weights cache to `~/.cache/ultralytics/` (YOLO) and `~/.cache/segment-anything-2/` (SAM2). Pre-download to avoid network calls in tests:
```bash
uv run --no-project python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

- [ ] **Step 1: Create the synthetic two-person fixture**

```python
# scripts/group_photobooth/_make_fixtures.py (one-shot, do not commit)
import cv2, numpy as np
from pathlib import Path

W, H = 800, 600
img = np.full((H, W, 3), 200, np.uint8)  # light gray background
# Two solid-color "people" silhouettes — actual YOLO+SAM2 will run on real
# photos; the fixture only needs to exercise the function shapes.
cv2.rectangle(img, (100, 150), (250, 550), (60, 80, 100), -1)
cv2.rectangle(img, (450, 200), (600, 550), (80, 60, 100), -1)
out = Path("tests/group_photobooth/fixtures/synthetic_two_people.png")
out.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out), img)
```

Run once: `uv run --no-project python scripts/group_photobooth/_make_fixtures.py` then delete the script.

Note: YOLO will not detect rectangles as persons. The detection test uses a **real** photo. Copy one from existing assets:

```bash
cp data/importer/identities/id_00.png tests/group_photobooth/fixtures/single_person.png
```

- [ ] **Step 2: Write the failing test**

`tests/group_photobooth/test_detect.py`:
```python
import cv2
from pathlib import Path

from group_photobooth import Person
from group_photobooth.detect import detect_people

FIXTURES = Path(__file__).parent / "fixtures"


def test_detect_single_person_returns_one_person():
    img = cv2.imread(str(FIXTURES / "single_person.png"))
    assert img is not None
    persons = detect_people(img)
    assert len(persons) == 1
    p = persons[0]
    assert isinstance(p, Person)
    x1, y1, x2, y2 = p.body_bbox
    assert 0 <= x1 < x2 <= img.shape[1]
    assert 0 <= y1 < y2 <= img.shape[0]
    assert p.body_mask.shape == img.shape[:2]
    assert p.body_mask.dtype.name == "uint8"
    # Mask must overlap the bbox region.
    crop = p.body_mask[y1:y2, x1:x2]
    assert (crop > 0).mean() > 0.10


def test_detect_returns_empty_on_no_person():
    # Solid gray image, no people.
    img = (cv2.imread(str(FIXTURES / "single_person.png")) * 0 + 128).astype("uint8")
    persons = detect_people(img)
    assert persons == []
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run --no-project pytest tests/group_photobooth/test_detect.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'group_photobooth.detect'`

- [ ] **Step 4: Implement detect.py**

`scripts/group_photobooth/detect.py`:
```python
"""YOLOv8-person + SAM2 → per-person body mask + face bbox.

Detection is CPU-fast (yolov8n ≈ 100 ms / 1024² on CPU). SAM2 mask
prediction from a bbox prompt is ~1-5 s / image on CPU depending on size;
we accept that for the offline group-photobooth.
"""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
from ultralytics import YOLO

from group_photobooth import Person


@lru_cache(maxsize=1)
def _yolo():
    # yolov8n is the smallest/fastest variant — sufficient for "person" class.
    # Auto-downloads to ~/.cache/ultralytics/ on first use.
    return YOLO("yolov8n.pt")


@lru_cache(maxsize=1)
def _sam2_predictor():
    # SAM2 base-plus weights, CPU device. Set to None until first use to
    # avoid the import cost when only YOLO is needed.
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    sam = build_sam2(
        "configs/sam2/sam2_hiera_b+.yaml",
        "facebook/sam2-hiera-base-plus",
        device="cpu",
    )
    return SAM2ImagePredictor(sam)


def _detect_face_in_crop(crop_bgr: np.ndarray, app) -> tuple[int, int, int, int] | None:
    """Run buffalo_l face detection on a body crop, return bbox in crop coords."""
    if crop_bgr.size == 0:
        return None
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    faces = app.get(rgb)
    if not faces:
        return None
    f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    return tuple(int(v) for v in f.bbox)


def detect_people(photo_bgr: np.ndarray, conf: float = 0.35,
                  iou: float = 0.45) -> list[Person]:
    """Detect every person in `photo_bgr`, return body bbox + mask + face bbox.

    Skips a detected person if no face is found inside their body bbox.
    """
    yolo = _yolo()
    results = yolo.predict(photo_bgr, classes=[0], conf=conf, iou=iou,
                           verbose=False)
    persons: list[Person] = []
    if not results:
        return persons
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return persons

    sam = _sam2_predictor()
    rgb = cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2RGB)
    sam.set_image(rgb)

    # Import here to keep buffalo_l init out of module-import time.
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from swap_core import make_face_app  # type: ignore
    app = make_face_app()

    for box in boxes.xyxy.cpu().numpy().astype(int):
        x1, y1, x2, y2 = box.tolist()
        masks, _, _ = sam.predict(box=np.array([x1, y1, x2, y2]),
                                  multimask_output=False)
        # SAM2 returns (1, H, W) float; threshold + uint8.
        mask = (masks[0] > 0.5).astype(np.uint8) * 255
        crop = photo_bgr[y1:y2, x1:x2]
        face_local = _detect_face_in_crop(crop, app)
        if face_local is None:
            continue
        fx1, fy1, fx2, fy2 = face_local
        face_bbox = (x1 + fx1, y1 + fy1, x1 + fx2, y1 + fy2)
        persons.append(Person(body_bbox=(x1, y1, x2, y2),
                              body_mask=mask, face_bbox=face_bbox))
    return persons
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run --no-project pytest tests/group_photobooth/test_detect.py -v`
Expected: 2 passed (may take ~20s on first run due to model downloads)

- [ ] **Step 6: Commit**

```bash
git add scripts/group_photobooth/detect.py tests/group_photobooth/test_detect.py tests/group_photobooth/fixtures/
git commit -m "feat(group-photobooth): YOLOv8 + SAM2 person detection with face anchor"
```

---

## Task 3: Silhouette — rembg/INSPYRENET cutout of doll portraits

**Files:**
- Create: `scripts/group_photobooth/silhouette.py`
- Create: `tests/group_photobooth/test_silhouette.py`
- Create: `tests/group_photobooth/fixtures/doll_portrait.png`

- [ ] **Step 1: Stage the fixture from a real photobooth output**

```bash
cp exp_output/photobooth_phase3/cells/id_00__cfg004/refined.png \
   tests/group_photobooth/fixtures/doll_portrait.png
```

- [ ] **Step 2: Write the failing test**

`tests/group_photobooth/test_silhouette.py`:
```python
import cv2
import numpy as np
from pathlib import Path

from group_photobooth.silhouette import cutout

FIXTURES = Path(__file__).parent / "fixtures"


def test_cutout_returns_rgba_same_dims():
    bgr = cv2.imread(str(FIXTURES / "doll_portrait.png"))
    rgba = cutout(bgr)
    assert rgba.shape == (bgr.shape[0], bgr.shape[1], 4)
    assert rgba.dtype == np.uint8
    # Some alpha must be > 0 (the doll is there) and some must be 0 (bg cut).
    alpha = rgba[..., 3]
    assert (alpha > 0).any()
    assert (alpha == 0).any()


def test_cutout_idempotent_on_already_cut_input():
    # Run twice — second run on the RGB channels of the first must produce
    # a comparable foreground area (within 10% IoU).
    bgr = cv2.imread(str(FIXTURES / "doll_portrait.png"))
    rgba1 = cutout(bgr)
    rgba2 = cutout(cv2.cvtColor(rgba1, cv2.COLOR_BGRA2BGR))
    a1, a2 = rgba1[..., 3] > 128, rgba2[..., 3] > 128
    iou = (a1 & a2).sum() / max(1, (a1 | a2).sum())
    assert iou > 0.85
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run --no-project pytest tests/group_photobooth/test_silhouette.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement silhouette.py**

`scripts/group_photobooth/silhouette.py`:
```python
"""rembg cutout for doll portraits.

INSPYRENET is the recommended model for human-portrait foreground per
docs/research/2026-05-21-comfyui-compositor-toolkit.md. For doll portraits
(Z-Image renders against flat backgrounds) plain u2net is sufficient and
~3× faster on CPU — we default to u2net and expose INSPYRENET via the
`model` kwarg for callers who need matting-quality edges.
"""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np


@lru_cache(maxsize=2)
def _session(model: str):
    from rembg import new_session
    return new_session(model)


def cutout(bgr: np.ndarray, model: str = "u2net") -> np.ndarray:
    """Cut foreground from `bgr`, return BGRA where alpha is the cutout mask.

    `model` is any rembg-supported model name. Defaults to "u2net" because
    doll backgrounds are uniform. Use "isnet-general-use" or
    "inspyrenet" (if installed) for matting-quality human portraits.
    """
    from rembg import remove

    session = _session(model)
    # rembg expects RGB or RGBA; pass RGB, get RGBA back.
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    cut_rgba = remove(rgb, session=session)
    # rembg returns numpy RGBA when given numpy input.
    if cut_rgba.shape[-1] == 4:
        bgra = cv2.cvtColor(cut_rgba, cv2.COLOR_RGBA2BGRA)
    else:
        # Defensive: synthesise an opaque alpha if rembg ever returns RGB.
        bgr_out = cv2.cvtColor(cut_rgba, cv2.COLOR_RGB2BGR)
        alpha = np.full(bgr_out.shape[:2], 255, np.uint8)
        bgra = np.dstack([bgr_out, alpha])
    return bgra
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run --no-project pytest tests/group_photobooth/test_silhouette.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add scripts/group_photobooth/silhouette.py tests/group_photobooth/test_silhouette.py tests/group_photobooth/fixtures/doll_portrait.png
git commit -m "feat(group-photobooth): rembg cutout wrapper with u2net default"
```

---

## Task 4: Layout — photo coords → background-canvas placements

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

## Task 5: Composite — painter's-order alpha blend + optional polish

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

## Task 6: Background library + manifest

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

## Task 7: Face renderer — thin wrapper around Phase 3 heavy mix

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

## Task 8: Driver — orchestrator with resumable per-person cache

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
    persons = detect_people(photo)
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
            rgba = cutout(doll)
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

## Task 9: End-to-end smoke tests (gated on live ComfyUI)

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

## Task 10: Manual visual verification + runbook

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
- Inputs/outputs → Task 8 (driver CLI takes photo + background id, writes result.png).
- Locked decisions (heavy mix, library backgrounds, side-by-side, SAM mask) → Tasks 2, 6, 7.
- Approach A pipeline → Tasks 2, 3, 4, 5, 8.
- Components table — `detect.py` (Task 2), `silhouette.py` (Task 3), `layout.py` (Task 4), `composite.py` (Task 5), `driver.py` (Task 8) — all present. Added `face_renderer.py` (Task 7) and `backgrounds.py` (Task 6); both are thin and serve clear responsibilities.
- Layout solver letterbox + painter's order → Task 4.
- Occlusion ordering → Task 4 + Task 5 (z_order tested with overlapping dolls).
- N=1 case → Task 9 test_n1_path.
- Lying-down / no-face person handling → driver.py skips person if `face_bbox is None`; warning behavior is implicit in detect.py (no Person emitted), explicit logging deferred to Task 10 runbook.
- Polish stage off by default → Task 8 CLI `--polish` flag.
- Background library schema → Task 6.
- Resumability → Task 8 driver skip-if-exists per file.
- Testing checklist (3 smokes) → Task 9 covers N=1 and N=2; occlusion is covered by unit test in Task 5.
- Cross-reference to chibi pivot — preserved in spec only, no plan task (chibi is out of scope).

**Placeholder scan:** No TBD/TODO. Every code step has a complete block. Every test step has expected output.

**Type consistency:**
- `Person.body_bbox: tuple[int, int, int, int]` — used in detect.py, layout.py, driver._crop_face. Consistent.
- `Person.face_bbox: tuple[int, int, int, int] | None` — None case handled in detect.py (skip) and driver (assertion only after detect filters).
- `Placement(x_center, y_bottom, height, z_order)` — used in layout.py producer and composite.py consumer. Consistent.
- `render_doll(face_crop_bgr, comfy_url, *, seed, demo) -> np.ndarray` — face_renderer.py producer, driver.py consumer. Consistent.
- `cutout(bgr, model="u2net") -> np.ndarray (BGRA)` — silhouette.py producer, driver.py consumer. Consistent.
- `composite(bg_bgr, dolls_rgba, placements) -> np.ndarray` — composite.py producer, driver.py consumer. Consistent.
