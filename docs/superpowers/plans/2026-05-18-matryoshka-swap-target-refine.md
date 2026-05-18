# Matryoshka Swap-Target Refine Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pre-swap face-region inpaint pass that re-renders a matryoshka doll's flat painted face into a detectable, face-like structure so `inswapper_128` has a good swap target.

**Architecture:** A generation-side mask builder (`face_region.py`) detects the doll face via MediaPipe and emits a feathered inpaint mask. A deterministic transform (`build_inpaint_workflow.py`) turns the existing bake-off generation workflows into masked-img2img inpaint workflows. A driver (`matryoshka_refine.py`) runs the inpaint pass over existing doll PNGs via ComfyUI. The swap-test harness is extended to report SCRFD detection rate and ArcFace `id_cos` before vs after refinement.

**Tech Stack:** Python 3.12, OpenCV, MediaPipe (via existing `swap_core`), ComfyUI REST API, insightface, pytest.

---

Design: `docs/superpowers/specs/2026-05-18-matryoshka-swap-target-refine-design.md`.

**Context for the implementer:**
- `scripts/swap_core.py` is the CPU detect-and-swap module. Reuse its `mediapipe_kps_bbox(img_bgr) -> (kps[5,2]|None, bbox[4]|None)` and `_crop_region(bbox, img_shape, margin_frac=0.45) -> (x0,y0,x1,y1)`.
- Existing ComfyUI generation workflows live in `comfyui/workflows/matryoshka_*.api.json`. Each is a node-id→node dict; placeholder strings like `$$SEED` are substituted by the driver.
- Driver pattern to copy: `scripts/matryoshka_bakeoff_sweep.py` (`queue`, `wait`, `download`, `_retry`, `build_workflow`, atomic writes, resumable skip-if-exists).
- Tests put `scripts/` on the path: `sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))`.
- Run all tests with the project venv: `python3 -m pytest tests/ -q`.
- Bake-off doll renders to refine: `exp_output/matryoshka_bakeoff/renders/`.

---

### Task 1: Face-region mask builder

**Files:**
- Create: `scripts/face_region.py`
- Test: `tests/test_face_region.py`

- [ ] **Step 1: Write the failing test**

```python
"""face_region.build_face_mask -- inpaint mask geometry, no real models."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import face_region


def test_build_face_mask_marks_face_region(monkeypatch):
    # synthetic 400x300 doll; force a known central face bbox
    doll = np.full((400, 300, 3), 200, dtype=np.uint8)
    kps = np.zeros((5, 2), dtype=np.float32)
    bbox = np.array([120, 150, 180, 230], dtype=np.float32)
    monkeypatch.setattr(face_region, "mediapipe_kps_bbox", lambda img: (kps, bbox))
    mask = face_region.build_face_mask(doll)
    assert mask is not None
    assert mask.shape == (400, 300)
    assert mask.dtype == np.uint8
    assert mask[190, 150] == 255          # bbox centre is white
    assert mask[0, 0] == 0                # far corner is black
    assert set(np.unique(mask)).issubset(set(range(256)))


def test_build_face_mask_returns_none_when_no_face(monkeypatch):
    doll = np.full((400, 300, 3), 200, dtype=np.uint8)
    monkeypatch.setattr(face_region, "mediapipe_kps_bbox", lambda img: (None, None))
    assert face_region.build_face_mask(doll) is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/test_face_region.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'face_region'`.

- [ ] **Step 3: Write the implementation**

Create `scripts/face_region.py`:

```python
"""Face-region inpaint mask for the matryoshka swap-target refine pass.

The generation arm produces a doll with a flat painted face. Before the
downstream inswapper swap, the face region is re-diffused into a face-like
structure (see scripts/matryoshka_refine.py). This module builds the mask
that confines that re-diffusion to the face -- the doll body and lacquer are
left untouched.
"""
from __future__ import annotations

import cv2
import numpy as np

from swap_core import _crop_region, mediapipe_kps_bbox


def build_face_mask(doll_bgr: np.ndarray, feather_frac: float = 0.10,
                    margin_frac: float = 0.35) -> np.ndarray | None:
    """Single-channel uint8 mask: white over the (feathered) doll face region.

    Detects the doll face with MediaPipe, expands its bbox by margin_frac
    (a smaller margin than the swap crop -- the inpaint should not bleed into
    the headscarf), and feathers the rectangle edge so the inpaint has no hard
    seam. Returns None if no face is detected, so the caller can pass the doll
    through unrefined.
    """
    kps, bbox = mediapipe_kps_bbox(doll_bgr)
    if kps is None or bbox is None:
        return None
    h, w = doll_bgr.shape[:2]
    x0, y0, x1, y1 = _crop_region(bbox, doll_bgr.shape, margin_frac=margin_frac)
    mask = np.zeros((h, w), dtype=np.float32)
    mask[y0:y1, x0:x1] = 1.0
    inset = int(round(min(x1 - x0, y1 - y0) * feather_frac))
    if inset > 0:
        k = 2 * inset + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    return (np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 -m pytest tests/test_face_region.py -q`
Expected: PASS, 2 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/face_region.py tests/test_face_region.py
git commit -m "feat(matryoshka): add face_region.build_face_mask — inpaint mask builder"
```

---

### Task 2: Inpaint-workflow transform + generated workflow JSONs

**Files:**
- Create: `scripts/build_inpaint_workflow.py`
- Create (generated, committed): `comfyui/workflows/matryoshka_zimage_inpaint.api.json`, `comfyui/workflows/matryoshka_sdxl_inpaint.api.json`
- Test: `tests/test_build_inpaint_workflow.py`

A bake-off generation workflow ends with an empty-latent node feeding `KSampler.latent_image`. The transform replaces that with `LoadImage(doll) → VAEEncode → SetLatentNoiseMask(mask)`, parameterizes `KSampler.denoise` as `$$DENOISE`, and bypasses any `ControlNetApplyAdvanced` between the prompts and the KSampler (face refinement needs no silhouette control). The orphaned empty-latent / ControlNet nodes are harmless — ComfyUI ignores unreferenced nodes.

- [ ] **Step 1: Write the failing test**

```python
"""build_inpaint_workflow -- deterministic generation->inpaint transform."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_inpaint_workflow import build_inpaint_workflow

WF_DIR = Path(__file__).resolve().parents[1] / "comfyui" / "workflows"


def _ksampler(wf):
    return next(n for n in wf.values() if n["class_type"] == "KSampler")


def _by_class(wf, cls):
    return [n for n in wf.values() if n["class_type"] == cls]


def test_zimage_transform_inserts_inpaint_chain():
    base = json.loads((WF_DIR / "matryoshka_zimage_turbo.api.json").read_text())
    wf = build_inpaint_workflow(base)
    snm = _by_class(wf, "SetLatentNoiseMask")
    assert len(snm) == 1
    ks = _ksampler(wf)
    snm_id = next(i for i, n in wf.items() if n["class_type"] == "SetLatentNoiseMask")
    assert ks["inputs"]["latent_image"] == [snm_id, 0]
    assert ks["inputs"]["denoise"] == "$$DENOISE"
    loads = {n["inputs"].get("image") for n in _by_class(wf, "LoadImage")}
    assert "$$DOLL_FILENAME" in loads
    assert _by_class(wf, "LoadImageMask")[0]["inputs"]["image"] == "$$MASK_FILENAME"
    assert len(_by_class(wf, "VAEEncode")) == 1


def test_sdxl_transform_bypasses_controlnet():
    base = json.loads((WF_DIR / "matryoshka_sdxl_lightning.api.json").read_text())
    wf = build_inpaint_workflow(base)
    ks = _ksampler(wf)
    pos_src = wf[ks["inputs"]["positive"][0]]
    assert pos_src["class_type"] != "ControlNetApplyAdvanced"
    assert len(_by_class(wf, "SetLatentNoiseMask")) == 1
    assert ks["inputs"]["denoise"] == "$$DENOISE"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/test_build_inpaint_workflow.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'build_inpaint_workflow'`.

- [ ] **Step 3: Write the implementation**

Create `scripts/build_inpaint_workflow.py`:

```python
"""Transform a matryoshka bake-off generation workflow into an inpaint one.

A generation workflow synthesizes a doll from an empty latent. The face-region
refine pass instead re-diffuses only the masked face of an existing doll. This
module performs that transform deterministically so the inpaint workflows stay
in sync with the generation workflows they derive from -- rather than being
hand-authored and drifting.

Run as a script to (re)generate the committed inpaint workflow JSONs:
    python scripts/build_inpaint_workflow.py
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

_WF_DIR = Path(__file__).resolve().parents[1] / "comfyui" / "workflows"
_ARMS = {
    "matryoshka_zimage_turbo.api.json": "matryoshka_zimage_inpaint.api.json",
    "matryoshka_sdxl_lightning.api.json": "matryoshka_sdxl_inpaint.api.json",
}


def build_inpaint_workflow(base_wf: dict) -> dict:
    """Return a copy of base_wf rewired for masked face-region img2img.

    The empty-latent node feeding KSampler.latent_image is replaced by
    LoadImage(doll) -> VAEEncode -> SetLatentNoiseMask(mask). KSampler.denoise
    becomes the placeholder "$$DENOISE". A ControlNetApplyAdvanced feeding the
    KSampler conditioning is bypassed (its own positive/negative sources are
    wired straight into KSampler). New nodes load "$$DOLL_FILENAME" /
    "$$MASK_FILENAME"; orphaned nodes are left for ComfyUI to ignore.
    """
    wf = copy.deepcopy(base_wf)
    ks_id = next(i for i, n in wf.items() if n["class_type"] == "KSampler")
    ks = wf[ks_id]

    vae_dec = next(n for n in wf.values() if n["class_type"] == "VAEDecode")
    vae_ref = list(vae_dec["inputs"]["vae"])

    for slot in ("positive", "negative"):
        src = wf[ks["inputs"][slot][0]]
        if src["class_type"] == "ControlNetApplyAdvanced":
            ks["inputs"][slot] = list(src["inputs"][slot])

    nid = max(int(i) for i in wf) + 1
    doll_id, mask_id, enc_id, snm_id = (str(nid + k) for k in range(4))
    wf[doll_id] = {"class_type": "LoadImage",
                   "inputs": {"image": "$$DOLL_FILENAME"}}
    wf[mask_id] = {"class_type": "LoadImageMask",
                   "inputs": {"image": "$$MASK_FILENAME", "channel": "red"}}
    wf[enc_id] = {"class_type": "VAEEncode",
                  "inputs": {"pixels": [doll_id, 0], "vae": vae_ref}}
    wf[snm_id] = {"class_type": "SetLatentNoiseMask",
                  "inputs": {"samples": [enc_id, 0], "mask": [mask_id, 0]}}
    ks["inputs"]["latent_image"] = [snm_id, 0]
    ks["inputs"]["denoise"] = "$$DENOISE"
    return wf


def main() -> int:
    for src, dst in _ARMS.items():
        base = json.loads((_WF_DIR / src).read_text())
        out = build_inpaint_workflow(base)
        (_WF_DIR / dst).write_text(json.dumps(out, indent=2) + "\n")
        print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 -m pytest tests/test_build_inpaint_workflow.py -q`
Expected: PASS, 2 passed.

- [ ] **Step 5: Generate and inspect the inpaint workflows**

Run: `python3 scripts/build_inpaint_workflow.py`
Expected: prints `wrote matryoshka_zimage_inpaint.api.json` and `wrote matryoshka_sdxl_inpaint.api.json`.

Run: `python3 -c "import json; wf=json.load(open('comfyui/workflows/matryoshka_zimage_inpaint.api.json')); print(sorted({n['class_type'] for n in wf.values()}))"`
Expected: the list includes `LoadImage`, `LoadImageMask`, `SetLatentNoiseMask`, `VAEEncode`, `KSampler`.

- [ ] **Step 6: Commit**

```bash
git add scripts/build_inpaint_workflow.py tests/test_build_inpaint_workflow.py \
        comfyui/workflows/matryoshka_zimage_inpaint.api.json \
        comfyui/workflows/matryoshka_sdxl_inpaint.api.json
git commit -m "feat(matryoshka): inpaint-workflow transform + generated zimage/sdxl inpaint graphs"
```

---

### Task 3: Refine driver

**Files:**
- Create: `scripts/matryoshka_refine.py`
- Test: `tests/test_matryoshka_refine.py`

The driver takes a directory of doll PNGs, builds a face mask per doll, runs the ComfyUI inpaint workflow for the doll's arm at one or more `denoise` values, and writes refined dolls to `--out/d<NNN>/<name>.png`. Resumable, atomic writes. ComfyUI helpers (`queue`/`wait`/`download`/`_retry`) are copied from `matryoshka_bakeoff_sweep.py`. The unit test covers only the model-free logic: arm detection from filename and the denoise→subdir naming.

- [ ] **Step 1: Write the failing test**

```python
"""matryoshka_refine -- model-free driver logic."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from matryoshka_refine import arm_of, denoise_subdir


def test_arm_of_classifies_by_filename_prefix():
    assert arm_of("zimage_turbo_st06_euler_simple_seed74029470.png") == "zimage_turbo"
    assert arm_of("sdxl_lightning_st04_dpmpp_sde_seed73823576.png") == "sdxl_lightning"


def test_arm_of_returns_none_for_unknown_arm():
    assert arm_of("flux_krea_seed73000000.png") is None


def test_denoise_subdir_formats_three_digits():
    assert denoise_subdir(0.55) == "d055"
    assert denoise_subdir(0.4) == "d040"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/test_matryoshka_refine.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'matryoshka_refine'`.

- [ ] **Step 3: Write the implementation**

Create `scripts/matryoshka_refine.py`:

```python
"""Matryoshka swap-target refine pass.

Takes generated doll PNGs (e.g. the bake-off renders) and re-diffuses only the
face region into a face-like structure, so the downstream inswapper swap has a
detectable, in-distribution target. Identity is NOT added here -- the inpaint
prompt is identity-blind; the swap stays the identity vehicle.

Per doll: detect the face -> build a feathered mask -> run the arm's inpaint
workflow at the given denoise -> save to <out>/d<NNN>/<name>.png. Resumable
(skip-if-exists), atomic writes. Dolls with no detectable face are copied
through unrefined.

Usage (cwd = repo root; run on the box where ComfyUI is bound):
    python scripts/matryoshka_refine.py \\
        --comfy-url http://127.0.0.1:8188 \\
        --renders exp_output/matryoshka_bakeoff/renders \\
        --out exp_output/matryoshka_bakeoff/refined \\
        --comfy-input-dir /home/newub/w/ComfyUI/input \\
        --denoise 0.4 --denoise 0.55 --denoise 0.7
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import requests

from face_region import build_face_mask

REFINE_PROMPT = (
    "a realistic painted human face on a wooden doll, soft three-dimensional "
    "shading, correct facial proportions, defined nose and lips, "
    "natural-sized eyes, rosy cheeks, centered frontal face"
)
NEGATIVE_PROMPT = (
    "flat painted face, oversized eyes, cartoon, deformed, distorted, "
    "watermark, text, low quality, blurry"
)
REFINE_SEED = 88_000_001

# arm -> (inpaint workflow file, steps, sampler, scheduler).
# Recipe = the bake-off winners; only denoise is swept.
ARMS: dict[str, dict] = {
    "zimage_turbo": {"workflow": "matryoshka_zimage_inpaint.api.json",
                     "steps": 6, "sampler": "euler", "scheduler": "simple"},
    "sdxl_lightning": {"workflow": "matryoshka_sdxl_inpaint.api.json",
                       "steps": 4, "sampler": "dpmpp_sde",
                       "scheduler": "sgm_uniform"},
}


def arm_of(filename: str) -> str | None:
    """Arm name a render filename belongs to, or None if not a refine arm."""
    for arm in ARMS:
        if filename.startswith(arm):
            return arm
    return None


def denoise_subdir(denoise: float) -> str:
    """Stable subdir name for a denoise value: 0.55 -> 'd055'."""
    return f"d{int(round(denoise * 100)):03d}"


def build_workflow(template: dict, *, doll_file: str, mask_file: str,
                   denoise: float, arm_cfg: dict, output_prefix: str) -> dict:
    subs: dict[str, Any] = {
        "$$POSITIVE_PROMPT": REFINE_PROMPT,
        "$$NEGATIVE_PROMPT": NEGATIVE_PROMPT,
        "$$DOLL_FILENAME": doll_file,
        "$$MASK_FILENAME": mask_file,
        "$$DENOISE": float(denoise),
        "$$SEED": REFINE_SEED,
        "$$STEPS": int(arm_cfg["steps"]),
        "$$SAMPLER": arm_cfg["sampler"],
        "$$SCHEDULER": arm_cfg["scheduler"],
        "$$OUTPUT_PREFIX": output_prefix,
    }

    def _sub(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: _sub(v) for k, v in node.items() if not k.startswith("_")}
        if isinstance(node, list):
            return [_sub(v) for v in node]
        if isinstance(node, str) and node in subs:
            return subs[node]
        return node

    return _sub(copy.deepcopy(template))


def _retry(fn, *, tries: int = 3, what: str = ""):
    for attempt in range(tries):
        try:
            return fn()
        except (requests.RequestException, ConnectionError) as e:
            if attempt == tries - 1:
                raise
            print(f"  [retry {attempt+1}/{tries}] {what}: {e}")
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError("unreachable")


def queue(sess: requests.Session, url: str, wf: dict, client_id: str) -> str:
    def _do():
        r = sess.post(f"{url}/prompt", json={"prompt": wf, "client_id": client_id},
                      timeout=30)
        r.raise_for_status()
        return r.json()["prompt_id"]
    return _retry(_do, what="queue")


def wait(sess: requests.Session, url: str, pid: str, timeout: float = 300) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = _retry(lambda: sess.get(f"{url}/history/{pid}", timeout=10),
                   what="history")
        if r is not None and r.status_code == 200 and pid in r.json():
            return r.json()[pid]
        time.sleep(1.0)
    raise TimeoutError(f"prompt {pid} did not complete within {timeout}s")


def download(sess: requests.Session, url: str, entry: dict,
             out_path: Path) -> bool:
    status = entry.get("status", {})
    if status.get("status_str") not in (None, "success"):
        print(f"  [fail] {out_path.stem}: status={status.get('status_str')}")
        return False
    for node_out in entry.get("outputs", {}).values():
        for img in node_out.get("images", []):
            r = _retry(lambda: sess.get(f"{url}/view", params={
                "filename": img["filename"],
                "subfolder": img.get("subfolder", ""),
                "type": img.get("type", "output")}, timeout=30), what="view")
            if r is None or r.status_code != 200 or \
                    r.content[:8] != b"\x89PNG\r\n\x1a\n":
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = out_path.with_suffix(".png.tmp")
            tmp.write_bytes(r.content)
            os.replace(tmp, out_path)
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    ap.add_argument("--renders", type=Path, required=True,
                    help="dir of generated doll PNGs to refine")
    ap.add_argument("--out", type=Path, required=True,
                    help="output root; refined dolls go to <out>/d<NNN>/")
    ap.add_argument("--workflow-dir", type=Path,
                    default=Path("comfyui/workflows"))
    ap.add_argument("--comfy-input-dir", type=Path, required=True,
                    help="ComfyUI input dir to stage doll + mask into")
    ap.add_argument("--denoise", type=float, action="append", required=True,
                    help="denoise value(s) to refine at (repeatable)")
    ap.add_argument("--limit", type=int, default=0,
                    help="refine at most N dolls per denoise (0 = all)")
    args = ap.parse_args()

    templates: dict[str, dict] = {}
    for arm, cfg in ARMS.items():
        tpl = json.loads((args.workflow_dir / cfg["workflow"]).read_text())
        classes = {n["class_type"] for n in tpl.values()}
        if "SetLatentNoiseMask" not in classes:
            raise SystemExit(f"{arm}: {cfg['workflow']} has no SetLatentNoiseMask "
                             f"-- regenerate with build_inpaint_workflow.py")
        templates[arm] = tpl

    args.comfy_input_dir.mkdir(parents=True, exist_ok=True)
    dolls = sorted(p for p in args.renders.glob("*.png")
                   if arm_of(p.name) is not None)
    print(f"[refine] {len(dolls)} dolls x {len(args.denoise)} denoise values")

    sess = requests.Session()
    client_id = str(uuid.uuid4())
    done = skipped = failed = passthrough = 0

    for denoise in args.denoise:
        sub = args.out / denoise_subdir(denoise)
        for n, doll_path in enumerate(dolls):
            if args.limit and n >= args.limit:
                break
            out_png = sub / doll_path.name
            if out_png.exists():
                skipped += 1
                continue
            arm = arm_of(doll_path.name)
            assert arm is not None  # filtered above
            doll_bgr = cv2.imread(str(doll_path))
            if doll_bgr is None:
                print(f"  [fail] unreadable {doll_path.name}")
                failed += 1
                continue
            mask = build_face_mask(doll_bgr)
            if mask is None:
                # no detectable face -> pass the doll through unrefined
                out_png.parent.mkdir(parents=True, exist_ok=True)
                tmp = out_png.with_suffix(".png.tmp")
                cv2.imwrite(str(tmp), doll_bgr)
                os.replace(tmp, out_png)
                passthrough += 1
                continue

            stem = doll_path.stem
            doll_file = f"refine_{stem}.png"
            mask_file = f"refine_{stem}_mask.png"
            cv2.imwrite(str(args.comfy_input_dir / doll_file), doll_bgr)
            cv2.imwrite(str(args.comfy_input_dir / mask_file), mask)

            wf = build_workflow(templates[arm], doll_file=doll_file,
                                mask_file=mask_file, denoise=denoise,
                                arm_cfg=ARMS[arm],
                                output_prefix=f"matryoshka_refine/{stem}")
            try:
                pid = queue(sess, args.comfy_url, wf, client_id)
                entry = wait(sess, args.comfy_url, pid)
                ok = download(sess, args.comfy_url, entry, out_png)
            except Exception as e:
                print(f"  [fail] {stem} d={denoise}: {e}")
                ok = False
            if ok:
                done += 1
                print(f"  [ok] {denoise_subdir(denoise)}/{doll_path.name} "
                      f"({done} done, {skipped} skipped, {passthrough} "
                      f"passthrough, {failed} failed)")
            else:
                failed += 1

    print(f"[refine] complete: {done} done, {skipped} skipped, "
          f"{passthrough} passthrough, {failed} failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 -m pytest tests/test_matryoshka_refine.py -q`
Expected: PASS, 3 passed.

- [ ] **Step 5: Syntax-check the whole module**

Run: `python3 -c "import ast; ast.parse(open('scripts/matryoshka_refine.py').read()); print('ok')"`
Expected: `ok`.

- [ ] **Step 6: Commit**

```bash
git add scripts/matryoshka_refine.py tests/test_matryoshka_refine.py
git commit -m "feat(matryoshka): add matryoshka_refine.py — face-region inpaint driver"
```

---

### Task 4: Swap-test before/after evaluation

**Files:**
- Modify: `scripts/matryoshka_bakeoff_swap_test.py`

The swap test already swaps bake-off dolls and reports `id_cos` + a per-mode count. Add an optional `--refined-root` argument: when given, for each render it also locates the refined doll under `<refined-root>/<denoise-subdir>/<name>.png`, swaps that too, and prints a baseline-vs-refined comparison (SCRFD `mode=default` rate and median `id_cos` per arm).

- [ ] **Step 1: Add the `--refined-root` argument**

In `scripts/matryoshka_bakeoff_swap_test.py`, in `main()`, after the existing `ap.add_argument("--out", ...)` line, add:

```python
    ap.add_argument("--refined-root", type=Path, default=None,
                    help="root of refined dolls (<root>/d<NNN>/<name>.png); "
                         "when set, prints a baseline-vs-refined comparison")
    ap.add_argument("--refined-denoise", default="d055",
                    help="denoise subdir under --refined-root to compare")
```

- [ ] **Step 2: Add the comparison helper**

In `scripts/matryoshka_bakeoff_swap_test.py`, after the `identity_cos` function, add:

```python
def _swap_stats(app, swapper, dolls, sources):
    """(default_rate, median_id_cos) over doll x identity swaps.

    default_rate is the fraction of swaps where SCRFD detected the face
    (mode == 'default'); median_id_cos is over the finite cosines.
    """
    modes, cosines = [], []
    for doll in dolls:
        for _sid, face, emb in sources:
            if face is None:
                continue
            res, mode, _det = swap_identity(app, swapper, doll, face)
            modes.append(mode)
            cos = identity_cos(app, res, emb)
            if np.isfinite(cos):
                cosines.append(cos)
    default_rate = (modes.count("default") / len(modes)) if modes else 0.0
    median = float(np.median(cosines)) if cosines else float("nan")
    return default_rate, median
```

- [ ] **Step 3: Add the comparison block**

In `scripts/matryoshka_bakeoff_swap_test.py`, immediately before the final `out.parent.mkdir(...)` / `canvas.save(out)` lines in `main()`, add:

```python
    if args.refined_root is not None:
        print(f"\n[swap-test] baseline vs refined ({args.refined_denoise})")
        print(f"  {'arm':14s} {'set':9s} {'default_rate':>12s} {'median_cos':>11s}")
        for arm, renders in ARM_RENDERS.items():
            base_dolls, ref_dolls = [], []
            for rname in renders:
                bd = cv2.imread(str(args.root / "renders" / rname))
                rp = args.refined_root / args.refined_denoise / rname
                rd = cv2.imread(str(rp))
                if bd is not None:
                    base_dolls.append(bd)
                if rd is not None:
                    ref_dolls.append(rd)
            b_rate, b_med = _swap_stats(app, swapper, base_dolls, sources)
            r_rate, r_med = _swap_stats(app, swapper, ref_dolls, sources)
            print(f"  {arm:14s} {'baseline':9s} {b_rate:12.2f} {b_med:11.3f}")
            print(f"  {arm:14s} {'refined':9s} {r_rate:12.2f} {r_med:11.3f}")
```

- [ ] **Step 4: Syntax-check**

Run: `python3 -c "import ast; ast.parse(open('scripts/matryoshka_bakeoff_swap_test.py').read()); print('ok')"`
Expected: `ok`.

- [ ] **Step 5: Verify the baseline path still runs unchanged**

Run: `python3 scripts/matryoshka_bakeoff_swap_test.py --swapper ~/w/ComfyUI/models/insightface/inswapper_128.onnx --ids id_03 id_11`
Expected: the existing identity-cosine table prints and `swap_test.png` is written; no comparison block (no `--refined-root`).

- [ ] **Step 6: Commit**

```bash
git add scripts/matryoshka_bakeoff_swap_test.py
git commit -m "feat(matryoshka): swap-test baseline-vs-refined comparison (--refined-root)"
```

---

### Task 5: End-to-end refine run + verdict

**Files:**
- Modify: `docs/research/2026-05-18-matryoshka-bakeoff-verdict.md`

This task needs ComfyUI running on the box (local 5090 or the Windows 3090). If the GPU is occupied, free it first; the refine pass is short.

- [ ] **Step 1: Confirm ComfyUI is reachable**

Run: `curl -s http://127.0.0.1:8188/system_stats | head -c 80`
Expected: a JSON blob. If it fails, start ComfyUI before continuing.

- [ ] **Step 2: Smoke-test one refine cell**

Run:
```bash
python3 scripts/matryoshka_refine.py \
    --renders exp_output/matryoshka_bakeoff/renders \
    --out exp_output/matryoshka_bakeoff/refined \
    --comfy-input-dir ~/w/ComfyUI/input \
    --denoise 0.55 --limit 1
```
Expected: `[ok] d055/<name>.png` printed; `exp_output/matryoshka_bakeoff/refined/d055/<name>.png` exists.

- [ ] **Step 3: Run the denoise ladder on the swap-test renders**

The swap test uses 5 specific renders (`ARM_RENDERS` in `matryoshka_bakeoff_swap_test.py`). Refine all bake-off renders at three denoise values:
```bash
python3 scripts/matryoshka_refine.py \
    --renders exp_output/matryoshka_bakeoff/renders \
    --out exp_output/matryoshka_bakeoff/refined \
    --comfy-input-dir ~/w/ComfyUI/input \
    --denoise 0.4 --denoise 0.55 --denoise 0.7
```
Expected: `[refine] complete:` with most dolls `done`, a few `passthrough`, `0 failed`.

- [ ] **Step 4: Compare baseline vs refined at each denoise**

For each of `d040`, `d055`, `d070`:
```bash
python3 scripts/matryoshka_bakeoff_swap_test.py \
    --swapper ~/w/ComfyUI/models/insightface/inswapper_128.onnx \
    --ids id_03 id_11 --refined-root exp_output/matryoshka_bakeoff/refined \
    --refined-denoise d055
```
Expected: the `baseline vs refined` table prints `default_rate` and `median_cos` for each arm. Record the numbers for all three denoise values.

- [ ] **Step 5: Update the verdict doc**

Append a `## Swap-target refine pass (2026-05-18)` section to `docs/research/2026-05-18-matryoshka-bakeoff-verdict.md` stating, with the recorded numbers: the SCRFD `default_rate` and median `id_cos` for baseline vs refined per arm at each denoise, which `denoise` won, and whether the refine pass moved cells out of `forced` mode. If no denoise improves both metrics, state that plainly — the pass is then falsified and the verdict records it.

- [ ] **Step 6: Run the full test suite**

Run: `python3 -m pytest tests/ -q`
Expected: all tests pass, including the new `test_face_region.py`, `test_build_inpaint_workflow.py`, `test_matryoshka_refine.py`.

- [ ] **Step 7: Commit**

```bash
git add docs/research/2026-05-18-matryoshka-bakeoff-verdict.md
git commit -m "docs(matryoshka): swap-target refine pass results + verdict"
```

- [ ] **Step 8: Final code review**

Dispatch the `superpowers:code-reviewer` agent over the diff since the design-doc commit, covering `face_region.py`, `build_inpaint_workflow.py`, `matryoshka_refine.py`, and the swap-test extension. Address any Critical/Important findings before declaring the plan done.

---

## Self-Review

**Spec coverage:**
- "Face-region inpaint mask" → Task 1 (`face_region.build_face_mask`). ✓
- "ComfyUI inpaint workflow" → Task 2 (transform + generated JSONs). ✓
- "`matryoshka_refine.py` driver" → Task 3. ✓
- "Extend the swap test: detection rate + id_cos before/after" → Task 4. ✓
- "denoise sweep (0.4/0.55/0.7)" → Task 5 Step 3. ✓
- "identity-blind inpaint prompt" → `REFINE_PROMPT` in Task 3. ✓
- Error handling: no-face passthrough (Task 3 driver), ComfyUI failure skip (Task 3 `download`/except), missing `SetLatentNoiseMask` guard (Task 3 `main`). ✓

**Placeholder scan:** No TBD/TODO; every code step has complete code; commands have expected output.

**Type consistency:** `build_face_mask` returns `np.ndarray | None` — Task 3 checks `if mask is None`. `arm_of` returns `str | None` — Task 3 filters on `is not None` then `assert`. `build_inpaint_workflow(dict) -> dict` — used consistently. `_swap_stats` returns `(float, float)` — printed as such. Placeholder names (`$$DOLL_FILENAME`, `$$MASK_FILENAME`, `$$DENOISE`) match between Task 2 (workflow) and Task 3 (`build_workflow` subs). ✓
