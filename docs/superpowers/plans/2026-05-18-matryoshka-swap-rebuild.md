# Matryoshka Identity-Swap Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a real human identity actually read on the small painted matryoshka-doll face by rebuilding the downstream swap as crop → upscale → swap → restore → feathered paste-back.

**Architecture:** `swap_core.py` keeps its public `swap_identity` entry point but internally crops the doll face, upscales the crop to 512 px (so SCRFD can detect it and the restorer has real signal), runs `inswapper_128` on the isolated crop, restores it with GFPGAN via a new `face_restore.py` module, then composites the result back under a feathered mask. The swap stays on CPU; restoration loads lazily on GPU and degrades gracefully to a no-op if unavailable.

**Tech Stack:** Python 3.12, uv, OpenCV, NumPy, insightface (`buffalo_l` + `inswapper_128`), MediaPipe FaceLandmarker, GFPGAN, pytest.

**Spec:** `docs/superpowers/specs/2026-05-18-matryoshka-swap-rebuild-design.md`

---

## File Structure

- `scripts/face_restore.py` — **new.** Lazy-loaded GFPGAN restorer. One public function `restore_face`. Graceful no-op fallback.
- `scripts/swap_core.py` — **modified.** Two new private helpers (`_crop_region`, `_feathered_mask`); `swap_identity` rewritten to the crop pipeline. `make_face_app`, `load_swapper`, `detect_source`, `mediapipe_kps_bbox`, `collapse_eyes` unchanged.
- `scripts/matryoshka_bakeoff_swap_test.py` — **modified.** Adds an ArcFace identity-cosine metric and a printed table.
- `tests/test_swap_geometry.py` — **new.** Unit tests for `_crop_region` and `_feathered_mask` (pure functions, synthetic images).
- `tests/test_face_restore.py` — **new.** Tests the `restore_face` graceful-fallback contract.

Consumers `scripts/matryoshka_swap_sweep.py` and `scripts/matryoshka_bakeoff_swap_test.py` call `swap_identity(app, swapper, doll, source_face)` — that signature is preserved (new `collapse`/`restore` args default to current behaviour).

---

## Task 1: Install GFPGAN and download weights

**Files:** none (environment + asset setup)

GFPGAN pulls `basicsr`, which imports `torchvision.transforms.functional_tensor` — removed in modern torchvision (the venv has torchvision 0.26). The import must be patched or every GFPGAN import crashes.

- [ ] **Step 1: Install gfpgan**

Run: `uv pip install gfpgan`
Expected: installs `gfpgan`, `basicsr`, `facexlib`, `realesrgan` and deps.

- [ ] **Step 2: Verify the basicsr import break exists**

Run: `python3 -c "import gfpgan"`
Expected: FAIL with `ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'`. (If it does NOT fail, skip Step 3.)

- [ ] **Step 3: Patch the basicsr import**

Run:
```bash
F=$(python3 -c "import basicsr,os; print(os.path.join(os.path.dirname(basicsr.__file__),'data','degradations.py'))")
sed -i 's/from torchvision.transforms.functional_tensor import/from torchvision.transforms.functional import/' "$F"
echo "patched: $F"
```
Expected: prints `patched: .../basicsr/data/degradations.py`.

- [ ] **Step 4: Verify gfpgan imports cleanly**

Run: `python3 -c "from gfpgan import GFPGANer; print('gfpgan ok')"`
Expected: prints `gfpgan ok` with no traceback.

- [ ] **Step 5: Download the GFPGAN v1.4 weights**

Run:
```bash
mkdir -p ~/w/ComfyUI/models/facerestore
curl -L -o ~/w/ComfyUI/models/facerestore/GFPGANv1.4.pth \
  https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth
ls -la ~/w/ComfyUI/models/facerestore/GFPGANv1.4.pth
```
Expected: a file of ~333 MB (348632874 bytes).

- [ ] **Step 6: Commit**

No code changed; record the environment step in the next task's commit. Proceed to Task 2.

---

## Task 2: Create `face_restore.py`

**Files:**
- Create: `scripts/face_restore.py`
- Test: `tests/test_face_restore.py`

`restore_face` must never be a hard dependency: if GFPGAN or its weights are missing, it logs once and returns the input unchanged. That fallback is the only behaviour testable without the 333 MB model, so it is what the unit test pins.

- [ ] **Step 1: Write the failing test**

Create `tests/test_face_restore.py`:
```python
"""Tests for face_restore.restore_face -- the graceful-fallback contract."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import face_restore  # noqa: E402


def test_restore_face_returns_input_when_restorer_unavailable(monkeypatch):
    """If the restorer cannot load, restore_face returns its input unchanged."""
    # force the lazy loader to report "unavailable"
    monkeypatch.setattr(face_restore, "_restorer", None)
    monkeypatch.setattr(face_restore, "_load_failed", False)
    monkeypatch.setattr(face_restore, "_load", lambda: None)

    img = np.full((64, 64, 3), 123, dtype=np.uint8)
    out = face_restore.restore_face(img)
    assert np.array_equal(out, img), "fallback must return input unchanged"


def test_restore_face_preserves_dtype_and_shape(monkeypatch):
    """The fallback output is a uint8 array of the same shape."""
    monkeypatch.setattr(face_restore, "_load", lambda: None)
    img = np.zeros((48, 32, 3), dtype=np.uint8)
    out = face_restore.restore_face(img)
    assert out.shape == img.shape and out.dtype == np.uint8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/newub/w/vamp-interface && python3 -m pytest tests/test_face_restore.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'face_restore'`.

- [ ] **Step 3: Implement `face_restore.py`**

Create `scripts/face_restore.py`:
```python
"""Face restoration for the matryoshka swap pipeline -- GFPGAN, lazy + optional.

restore_face turns a low-detail swapped face crop into a sharp, realistic one.
GFPGAN is the most conservative restorer (best identity preservation -- it is
least likely to drift to a different person). If the restorer cannot be loaded
-- missing package or missing weights -- restore_face logs once and returns its
input unchanged: restoration is an enhancement, never a hard dependency.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_GFPGAN_WEIGHTS = Path.home() / "w/ComfyUI/models/facerestore/GFPGANv1.4.pth"

_restorer = None          # cached GFPGANer once loaded
_load_failed = False      # set True after a failed load so we log only once


def _load():
    """Lazy-load GFPGANer once. Returns the restorer or None on failure."""
    global _restorer, _load_failed
    if _restorer is not None or _load_failed:
        return _restorer
    try:
        from gfpgan import GFPGANer
        _restorer = GFPGANer(
            model_path=str(_GFPGAN_WEIGHTS), upscale=1,
            arch="clean", channel_multiplier=2, bg_upsampler=None)
    except Exception as e:                                   # noqa: BLE001
        print(f"[face_restore] GFPGAN unavailable, restoration skipped: {e}")
        _load_failed = True
    return _restorer


def restore_face(crop_bgr: np.ndarray) -> np.ndarray:
    """Restore a swapped face crop. Returns a BGR uint8 image of the same size.

    Falls back to the unchanged input if the restorer is unavailable or the
    enhance call fails or finds no face.
    """
    restorer = _load()
    if restorer is None:
        return crop_bgr
    try:
        _, _, restored = restorer.enhance(
            crop_bgr, has_aligned=False, only_center_face=True,
            paste_back=True)
    except Exception as e:                                   # noqa: BLE001
        print(f"[face_restore] enhance failed, returning input: {e}")
        return crop_bgr
    if restored is None:
        return crop_bgr
    return np.ascontiguousarray(restored, dtype=np.uint8)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/newub/w/vamp-interface && python3 -m pytest tests/test_face_restore.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Smoke-test the real restorer**

Run:
```bash
cd /home/newub/w/vamp-interface && python3 -c "
import sys; sys.path.insert(0,'scripts')
import numpy as np, face_restore
out = face_restore.restore_face(np.full((300,300,3),128,np.uint8))
print('restore ok, shape', out.shape, 'loader failed:', face_restore._load_failed)
"
```
Expected: `restore ok, shape (300, 300, 3) loader failed: False` — GFPGAN loaded (facexlib auto-downloads its detection/parsing weights on first run; allow a minute).

- [ ] **Step 6: Commit**

```bash
cd /home/newub/w/vamp-interface
git add scripts/face_restore.py tests/test_face_restore.py
git commit -m "feat(matryoshka): GFPGAN face-restore module with graceful fallback"
```

---

## Task 3: `_crop_region` helper in `swap_core.py`

**Files:**
- Modify: `scripts/swap_core.py`
- Test: `tests/test_swap_geometry.py`

`_crop_region` returns a square region around the MediaPipe face bbox, expanded by a margin, clamped to the image.

- [ ] **Step 1: Write the failing test**

Create `tests/test_swap_geometry.py`:
```python
"""Tests for swap_core crop + paste-back geometry helpers."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from swap_core import _crop_region  # noqa: E402


def test_crop_region_is_centered_and_expanded():
    """A central bbox yields a larger square region around the same centre."""
    bbox = np.array([400, 500, 500, 620], dtype=np.float32)  # 100x120
    x0, y0, x1, y1 = _crop_region(bbox, (1152, 864, 3), margin_frac=0.45)
    # square
    assert (x1 - x0) == (y1 - y0), f"region not square: {x1-x0}x{y1-y0}"
    # larger than the bbox (side 120 expanded by 2*0.45)
    assert (x1 - x0) > 120
    # centred on the bbox centre (450, 560)
    assert abs((x0 + x1) / 2 - 450) <= 1
    assert abs((y0 + y1) / 2 - 560) <= 1


def test_crop_region_clamps_to_image_bounds():
    """A bbox near a corner yields a region clamped inside the image."""
    bbox = np.array([10, 10, 60, 60], dtype=np.float32)
    x0, y0, x1, y1 = _crop_region(bbox, (256, 256, 3), margin_frac=0.45)
    assert x0 >= 0 and y0 >= 0 and x1 <= 256 and y1 <= 256
    assert x1 > x0 and y1 > y0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/newub/w/vamp-interface && python3 -m pytest tests/test_swap_geometry.py -v`
Expected: FAIL with `ImportError: cannot import name '_crop_region'`.

- [ ] **Step 3: Implement `_crop_region`**

In `scripts/swap_core.py`, after the `_EYE_*` constants block and before `collapse_eyes`, add:
```python
def _crop_region(bbox, img_shape, margin_frac=0.45):
    """Square crop region around a face bbox, expanded by margin, clamped.

    bbox is [x0,y0,x1,y1]. Returns integer (x0,y0,x1,y1) inside img_shape.
    The region is square before clamping; clamping at an image border may
    leave it non-square, which the caller handles by aspect-preserving resize.
    """
    x0, y0, x1, y1 = (float(v) for v in bbox)
    h, w = img_shape[:2]
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half = max(x1 - x0, y1 - y0) * (0.5 + margin_frac)
    rx0 = int(max(0, round(cx - half)))
    ry0 = int(max(0, round(cy - half)))
    rx1 = int(min(w, round(cx + half)))
    ry1 = int(min(h, round(cy + half)))
    return rx0, ry0, rx1, ry1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/newub/w/vamp-interface && python3 -m pytest tests/test_swap_geometry.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
cd /home/newub/w/vamp-interface
git add scripts/swap_core.py tests/test_swap_geometry.py
git commit -m "feat(matryoshka): _crop_region helper for swap pipeline"
```

---

## Task 4: `_feathered_mask` helper in `swap_core.py`

**Files:**
- Modify: `scripts/swap_core.py`
- Test: `tests/test_swap_geometry.py`

`_feathered_mask` returns a float32 mask: 1.0 in the interior, Gaussian-feathered to 0.0 at the edges, for seam-free paste-back.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_swap_geometry.py`:
```python
from swap_core import _feathered_mask  # noqa: E402


def test_feathered_mask_center_is_one_edges_fade():
    """Mask is ~1 at the centre and lower near the border."""
    m = _feathered_mask(200, 200, feather_frac=0.12)
    assert m.shape == (200, 200) and m.dtype == np.float32
    assert m[100, 100] > 0.95, "centre should be ~1"
    assert m[2, 2] < m[100, 100], "corner should be faded below centre"
    assert 0.0 <= m.min() and m.max() <= 1.0


def test_feathered_mask_tiny_image_is_safe():
    """A tiny region produces a valid mask, no crash."""
    m = _feathered_mask(6, 8, feather_frac=0.12)
    assert m.shape == (6, 8) and m.dtype == np.float32
    assert 0.0 <= m.min() and m.max() <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/newub/w/vamp-interface && python3 -m pytest tests/test_swap_geometry.py -v`
Expected: FAIL with `ImportError: cannot import name '_feathered_mask'`.

- [ ] **Step 3: Implement `_feathered_mask`**

In `scripts/swap_core.py`, directly after `_crop_region`, add:
```python
def _feathered_mask(h, w, feather_frac=0.12):
    """Float32 [h,w] mask: 1.0 interior, Gaussian-feathered to 0.0 at edges."""
    mask = np.zeros((h, w), dtype=np.float32)
    inset = int(round(min(h, w) * feather_frac))
    inset = max(0, min(inset, min(h, w) // 2 - 1))
    if inset <= 0:
        mask[:] = 1.0
        return mask
    mask[inset:h - inset, inset:w - inset] = 1.0
    k = 2 * inset + 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    return np.clip(mask, 0.0, 1.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/newub/w/vamp-interface && python3 -m pytest tests/test_swap_geometry.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
cd /home/newub/w/vamp-interface
git add scripts/swap_core.py tests/test_swap_geometry.py
git commit -m "feat(matryoshka): _feathered_mask helper for seam-free paste-back"
```

---

## Task 5: Rewrite `swap_identity` to the crop pipeline

**Files:**
- Modify: `scripts/swap_core.py` (the `swap_identity` function and the imports)

`swap_identity` keeps its signature plus a new `restore=True` flag. New flow: crop the doll face → upscale to 512 px longer-side → eye-collapse on the crop → SCRFD-or-forced detection on the crop → swap → restore → feathered paste-back.

- [ ] **Step 1: Add the `face_restore` import**

At the top of `scripts/swap_core.py`, after `import numpy as np`, add:
```python
from face_restore import restore_face
```

- [ ] **Step 2: Replace `swap_identity`**

Replace the entire existing `swap_identity` function with:
```python
def swap_identity(app, swapper, doll_bgr, source_face,
                  collapse=True, restore=True):
    """Swap source_face's identity onto the doll via a crop-isolated pipeline.

    The doll face is cropped, upscaled to 512 px (so SCRFD can see it and the
    restorer has real signal), swapped, restored, and composited back under a
    feathered mask. When collapse is True the doll's oversized painted eyes are
    shrunk first (see collapse_eyes). When restore is True the swapped crop is
    sharpened by GFPGAN (see face_restore.restore_face).

    Returns (result_bgr, mode, det_score). mode is 'default' (SCRFD found the
    face on the upscaled crop), 'forced' (MediaPipe synthetic kps), or 'failed'
    (no detectable face -- the doll is returned unchanged).
    """
    kps_full, bbox_full = mediapipe_kps_bbox(doll_bgr)
    if kps_full is None or bbox_full is None:
        return doll_bgr, "failed", 0.0

    rx0, ry0, rx1, ry1 = _crop_region(bbox_full, doll_bgr.shape)
    crop = doll_bgr[ry0:ry1, rx0:rx1]
    if crop.size == 0:
        return doll_bgr, "failed", 0.0

    ch, cw = crop.shape[:2]
    scale = 512.0 / max(ch, cw)
    up = cv2.resize(crop, (max(1, round(cw * scale)), max(1, round(ch * scale))),
                    interpolation=cv2.INTER_LANCZOS4)

    kps_up, bbox_up = mediapipe_kps_bbox(up)
    work = up
    if collapse and kps_up is not None:
        work = collapse_eyes(up, kps_up)

    det = app.get(work)
    if det:
        target = max(det, key=lambda f: f.det_score)
        mode, det_score = "default", float(target.det_score)
    elif kps_up is not None:
        from insightface.app.common import Face
        target = Face(bbox=bbox_up, kps=kps_up, det_score=1.0)
        mode, det_score = "forced", 0.0
    else:
        return doll_bgr, "failed", 0.0

    swapped = swapper.get(work.copy(), target, source_face, paste_back=True)
    if restore:
        swapped = restore_face(swapped)

    patch = cv2.resize(swapped, (rx1 - rx0, ry1 - ry0),
                       interpolation=cv2.INTER_LANCZOS4)
    mask = _feathered_mask(ry1 - ry0, rx1 - rx0)[..., None]
    out = doll_bgr.copy()
    region = out[ry0:ry1, rx0:rx1].astype(np.float32)
    blended = patch.astype(np.float32) * mask + region * (1.0 - mask)
    out[ry0:ry1, rx0:rx1] = np.clip(blended, 0, 255).astype(np.uint8)
    return out, mode, det_score
```

- [ ] **Step 3: Run the existing collapse-eyes tests (regression guard)**

Run: `cd /home/newub/w/vamp-interface && python3 -m pytest tests/test_collapse_eyes.py tests/test_swap_geometry.py tests/test_face_restore.py -v`
Expected: PASS — `collapse_eyes` was not touched, the helpers and restore module still pass.

- [ ] **Step 4: Smoke-test the rewired pipeline on one render**

Run:
```bash
cd /home/newub/w/vamp-interface/scripts && python3 -c "
import cv2
from swap_core import make_face_app, load_swapper, detect_source, swap_identity
app = make_face_app()
sw = load_swapper('/home/newub/w/ComfyUI/models/insightface/inswapper_128.onnx')
src = detect_source(app, cv2.imread('../data/importer/identities/id_03.png'))
doll = cv2.imread('../exp_output/matryoshka_bakeoff/renders/zimage_turbo_st06_euler_simple_seed74029470.png')
res, mode, score = swap_identity(app, sw, doll, src)
assert res.shape == doll.shape, res.shape
print('swap ok: mode=%s score=%.2f shape=%s' % (mode, score, res.shape))
"
```
Expected: prints `swap ok: mode=... score=... shape=(1152, 864, 3)` with no traceback. `mode` may now be `default` if the 512 crop let SCRFD detect the face.

- [ ] **Step 5: Commit**

```bash
cd /home/newub/w/vamp-interface
git add scripts/swap_core.py
git commit -m "feat(matryoshka): crop-upscale-restore swap pipeline in swap_identity"
```

---

## Task 6: Add the ArcFace identity metric to the eval harness

**Files:**
- Modify: `scripts/matryoshka_bakeoff_swap_test.py`

The harness currently only montages doll-vs-swap. Add a quantitative identity score: `cos(source_embedding, output_face_embedding)`. The swapped face is small in the full output, so the metric re-crops and upscales the output face region the same way the swap pipeline does, then detects there.

- [ ] **Step 1: Add the identity-cosine helper**

In `scripts/matryoshka_bakeoff_swap_test.py`, after the imports, add `mediapipe_kps_bbox` and `_crop_region` to the `swap_core` import line so it reads:
```python
from swap_core import (detect_source, load_swapper, make_face_app,
                       mediapipe_kps_bbox, _crop_region, swap_identity)
```
Then add this function after `thumb`:
```python
def identity_cos(app, result_bgr, source_emb):
    """cos(source_embedding, swapped-face embedding) measured on result_bgr.

    The swapped face is small in the full output, so re-crop + upscale the
    face region (mirroring swap_identity) before detecting, else SCRFD misses
    it. Returns a float in [-1,1], or float('nan') if no face is detectable.
    """
    if source_emb is None:
        return float("nan")
    kps, bbox = mediapipe_kps_bbox(result_bgr)
    if bbox is None:
        return float("nan")
    x0, y0, x1, y1 = _crop_region(bbox, result_bgr.shape)
    crop = result_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return float("nan")
    s = 512.0 / max(crop.shape[:2])
    up = cv2.resize(crop, (max(1, round(crop.shape[1] * s)),
                           max(1, round(crop.shape[0] * s))),
                    interpolation=cv2.INTER_LANCZOS4)
    faces = app.get(up)
    if not faces:
        return float("nan")
    emb = max(faces, key=lambda f: f.det_score).normed_embedding
    return float(np.dot(emb, source_emb))
```

- [ ] **Step 2: Record the cosine in the swap loop**

In `main`, the source list currently stores `(sid, face)`. Change it to also keep the normed embedding, and collect per-cell scores. Replace the `sources` build loop with:
```python
    # source identity faces + their normed embeddings
    sources = []
    for sid in args.ids:
        sp = args.identities / f"{sid}.png"
        src_img = cv2.imread(str(sp))
        face = detect_source(app, src_img) if src_img is not None else None
        if face is None:
            print(f"  [warn] no face in source {sp}")
        emb = face.normed_embedding if face is not None else None
        sources.append((sid, face, emb))
```
Then in the render loop, change the inner `for sid, face in sources:` to
`for sid, face, emb in sources:` and, right after the `swap_identity` call and its `print`, add:
```python
                cos = identity_cos(app, res, emb)
                scores.append({"arm": arm, "render": rname, "id": sid,
                               "mode": mode, "det": det, "cos": cos})
                print(f"      identity cos = {cos:.3f}")
```
Add `scores = []` just before the `rows = []` line.

- [ ] **Step 3: Print the identity table before saving the montage**

Just before `out.parent.mkdir(...)` near the end of `main`, add:
```python
    if scores:
        import pandas as pd
        sdf = pd.DataFrame(scores)
        print("\n=== identity cosine (higher = identity reads better) ===")
        print(sdf.to_string(index=False))
        print(f"\nmedian cos = {sdf['cos'].median():.3f}  "
              f"(n={sdf['cos'].notna().sum()} measured, "
              f"{sdf['cos'].isna().sum()} undetected)")
        sdf.to_csv(args.root / "swap_test_scores.csv", index=False)
```

- [ ] **Step 4: Run the eval harness**

Run:
```bash
cd /home/newub/w/vamp-interface/scripts && python3 matryoshka_bakeoff_swap_test.py \
  --root ../exp_output/matryoshka_bakeoff \
  --identities ../data/importer/identities \
  --swapper ~/w/ComfyUI/models/insightface/inswapper_128.onnx \
  --ids id_03 id_07 id_11 2>&1 | grep -E 'identity cos|median|arm|swap-test|<- id'
```
Expected: a per-cell `identity cos` line for every (render × id), then a printed table and a `median cos` line. The medians should be materially above 0 (the pre-rebuild pipeline scored near 0).

- [ ] **Step 5: Commit**

```bash
cd /home/newub/w/vamp-interface
git add scripts/matryoshka_bakeoff_swap_test.py
git commit -m "feat(matryoshka): ArcFace identity-cosine metric in swap eval harness"
```

---

## Task 7: End-to-end verification and review

**Files:**
- Create: none (verification + montage inspection)

- [ ] **Step 1: Regenerate the montages and inspect**

Run the Task 6 Step 4 command without the `grep` filter so `swap_test.png` and `swap_zoom.png` are rewritten. Open `exp_output/matryoshka_bakeoff/swap_test.png` and `swap_zoom.png` and confirm the swapped faces visibly read as the source identities (sharper, real-skin detail) rather than blurred generic faces.

- [ ] **Step 2: Compare against the pre-rebuild baseline**

The pre-rebuild swap montage was committed in `c72bc03`. Eyeball old vs new: the rebuilt pipeline should show a clear improvement in how strongly each identity reads. Note the median cosine from Task 6 in the verdict update (Step 4).

- [ ] **Step 3: Run the full unit-test suite for the touched files**

Run: `cd /home/newub/w/vamp-interface && python3 -m pytest tests/test_collapse_eyes.py tests/test_swap_geometry.py tests/test_face_restore.py -v`
Expected: all PASS.

- [ ] **Step 4: Update the verdict doc**

Append a short "Swap rebuild (2026-05-18)" section to `docs/research/2026-05-18-matryoshka-bakeoff-verdict.md` recording the median identity cosine before vs after, per arm, and whether identity now reads. Commit:
```bash
cd /home/newub/w/vamp-interface
git add docs/research/2026-05-18-matryoshka-bakeoff-verdict.md
git commit -m "docs(matryoshka): record swap-rebuild identity-cosine results"
```

- [ ] **Step 5: Code review**

Per the standing project rule (`feedback_run_code_review` memory), dispatch the `superpowers:code-reviewer` agent on the diff since `c72bc03` — `scripts/face_restore.py`, the `swap_core.py` changes, and the eval-harness changes — reviewed against this plan and the spec. Address any issues it raises before declaring the task done.

---

## Self-Review Notes

- **Spec coverage:** crop→upscale (Task 5), GFPGAN restore (Tasks 1-2, 5), feathered paste-back (Tasks 4-5), SCRFD-on-crop "help insightface" (Task 5 detection branch), identity-cosine eval (Task 6), no-face regression guard (Task 5 `failed` path; exercised in Task 7). The img2img glue is explicitly out of scope per the spec — no task, correctly.
- **Signature consistency:** `swap_identity(app, swapper, doll_bgr, source_face, collapse=True, restore=True)` is used identically in Task 5 and called by both consumers with the first four positional args. `_crop_region(bbox, img_shape, margin_frac=0.45)` and `_feathered_mask(h, w, feather_frac=0.12)` match between definition (Tasks 3-4) and use (Tasks 5-6). `restore_face(crop_bgr)` matches between definition (Task 2) and use (Task 5).
- **basicsr hazard:** handled explicitly in Task 1 Steps 2-4 rather than left implicit.
