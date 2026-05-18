# ARKit Landmark-Control Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find out, with zero training, whether a MediaPipe face-mesh control image fed into InfiniteYou's InfuseNet spatial-control slot steers FLUX's output expression while ArcFace identity holds.

**Architecture:** Select real expression exemplars from `reverse_index.parquet` by their stored ARKit blendshape coefficients; extract each exemplar's 478-landmark MediaPipe mesh; normalize and render it as the InfuseNet control image; run an InfiniteYou ComfyUI graph (identity tokens + mesh control slot, no FluxSpace); score outputs by ArcFace cosine and a blendshape-vector cosine against the exemplar.

**Tech Stack:** Python 3.12, uv (`pythonpath=["src"]` — bare imports), pandas, MediaPipe FaceLandmarker, insightface buffalo_l, OpenCV, ComfyUI REST via `demographic_pc.comfy_flux.ComfyClient`, pytest.

**Design:** `docs/superpowers/specs/2026-05-18-arkit-landmark-control-spike-design.md`

**Context for the engineer:**
- This repo puts `src/` on the path (`pyproject.toml` `pythonpath`). Import as `from arkit_controlnet.axes import AXES`, never `from src.arkit_controlnet...`. Bare `python3` needs `PYTHONPATH=src`; pytest adds it automatically.
- `src/arkit_controlnet/axes.py` already defines a frozen dataclass `Axis` with fields `name`, `edit_prompt_a`, `edit_prompt_b`, `mix_b`, `scale_band`, `target_channels` (a `list[str]` of ARKit channel names like `"mouthSmileLeft"`), and `AXES: dict[str, Axis]` with keys `smile`, `pucker`, `surprise`.
- `src/arkit_controlnet/eval_spike.py` already defines `ARKIT_BLENDSHAPE_NAMES` (52 names, `_neutral` at index 0), `bs_read(path) -> dict[str,float]`, `arcface_cos`, `bs_delta`, and private helpers `_get_landmarker()`, `_imread(path)`.
- The parquet at `output/reverse_index/reverse_index.parquet` has one row per image with columns `image_sha256`, `source`, `bs_detected` (bool/None), and 52 columns `bs_<channelName>` (e.g. `bs_mouthSmileLeft`, `bs__neutral`). FFHQ rows have `source == "ffhq"`; FFHQ PNGs live at `output/ffhq_images/{image_sha256}.png`.
- `comfyui/workflows/arkit_controlnet_spike.json` is the prior spike's graph — clone its node structure. ComfyUI workflows here are API-format JSON dicts (string node IDs, `["node_id", slot]` wiring) with `$$PLACEHOLDER` strings, submitted via `ComfyClient`. Never built in the ComfyUI UI.
- `src/arkit_controlnet/run_spike.py` defines `select_identities(n) -> list[Path]` and the `ComfyClient` usage pattern (`async with ComfyClient() as client`, `client.upload_image(path)`, `client.generate(workflow, dest)`). Reuse `select_identities` directly.

---

### Task 1: eval_spike — blendshape vector, expression cosine, landmark extraction

**Files:**
- Modify: `src/arkit_controlnet/eval_spike.py`
- Test: `tests/arkit_controlnet/test_eval_spike.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/arkit_controlnet/test_eval_spike.py` (the file already has a module-level `FIXTURE = Path("tests/arkit_controlnet/fixtures/face.png")` and `pytestmark = pytest.mark.skipif(not FIXTURE.exists(), ...)` — reuse them; do not redefine):

```python
import numpy as np
from arkit_controlnet.eval_spike import bs_vector, expr_cos, face_landmarks_xy


def test_bs_vector_is_52d_in_canonical_order():
    v = bs_vector(FIXTURE)
    assert v.shape == (52,)
    assert v.dtype == np.float64


def test_expr_cos_of_a_face_with_itself_is_one():
    assert expr_cos(FIXTURE, FIXTURE) == pytest.approx(1.0, abs=1e-6)


def test_face_landmarks_xy_returns_478_points_in_unit_square():
    lm = face_landmarks_xy(FIXTURE)
    assert lm.shape == (478, 2)
    assert lm.min() >= -0.5 and lm.max() <= 1.5  # normalized, small slop off-frame
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/arkit_controlnet/test_eval_spike.py -v`
Expected: FAIL — `ImportError: cannot import name 'bs_vector'`.

- [ ] **Step 3: Implement the three functions**

Append to `src/arkit_controlnet/eval_spike.py`:

```python
def bs_vector(image_path: Path) -> np.ndarray:
    """52-d ARKit blendshape vector, ordered by ARKIT_BLENDSHAPE_NAMES."""
    got = bs_read(image_path)
    return np.array([got[n] for n in ARKIT_BLENDSHAPE_NAMES], dtype=np.float64)


def expr_cos(path_a: Path, path_b: Path) -> float:
    """Cosine of two 52-d blendshape vectors, excluding the `_neutral` channel.

    Measures expression similarity. `_neutral` (index 0) is dropped — it is an
    inverse summary of all the others and would wash out the signal. Returns
    -1.0 if either vector is all-zero (no face detected -> bs_read all zeros).
    """
    a = bs_vector(path_a)[1:]
    b = bs_vector(path_b)[1:]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return -1.0
    return float(np.dot(a, b) / (na * nb))


def face_landmarks_xy(image_path: Path) -> np.ndarray:
    """(478, 2) array of normalized [0,1] (x, y) MediaPipe face landmarks.

    Raises ValueError if no face is detected — callers select among several
    candidates, so a hard failure is correct here.
    """
    arr = cv2.cvtColor(_imread(image_path), cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
    res = _get_landmarker().detect(mp_image)
    if not res.face_landmarks:
        raise ValueError(f"no face detected in {image_path}")
    return np.array([[p.x, p.y] for p in res.face_landmarks[0]], dtype=np.float64)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/arkit_controlnet/test_eval_spike.py -v`
Expected: PASS (all tests, including the pre-existing ones).

- [ ] **Step 5: Commit**

```bash
git add src/arkit_controlnet/eval_spike.py tests/arkit_controlnet/test_eval_spike.py
git commit -m "feat(arkit-cn): blendshape vector + expr cosine + landmark extraction"
```

---

### Task 2: landmark_control — expression-exemplar selection from the parquet

**Files:**
- Create: `src/arkit_controlnet/landmark_control.py`
- Test: `tests/arkit_controlnet/test_landmark_control.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/arkit_controlnet/test_landmark_control.py`:

```python
from pathlib import Path

import pandas as pd
import pytest

from arkit_controlnet.landmark_control import (
    REVERSE_INDEX, select_exemplars, select_neutral,
)

pytestmark = pytest.mark.skipif(
    not REVERSE_INDEX.exists(), reason="reverse_index parquet not present"
)


def test_select_exemplars_returns_k_existing_pngs():
    paths = select_exemplars("smile", k=3)
    assert len(paths) == 3
    assert all(p.exists() and p.suffix == ".png" for p in paths)


def test_smile_exemplars_score_above_corpus_median():
    # the picked exemplars must actually be high-smile, not arbitrary rows
    df = pd.read_parquet(
        REVERSE_INDEX, columns=["image_sha256", "bs_mouthSmileLeft", "bs_mouthSmileRight"]
    )
    median = (df["bs_mouthSmileLeft"] + df["bs_mouthSmileRight"]).median()
    picked = {p.stem for p in select_exemplars("smile", k=3)}
    rows = df[df["image_sha256"].isin(picked)]
    score = rows["bs_mouthSmileLeft"] + rows["bs_mouthSmileRight"]
    assert (score > median).all()


def test_select_neutral_returns_k_existing_pngs():
    paths = select_neutral(k=3)
    assert len(paths) == 3
    assert all(p.exists() for p in paths)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/arkit_controlnet/test_landmark_control.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'arkit_controlnet.landmark_control'`.

- [ ] **Step 3: Implement selection**

Create `src/arkit_controlnet/landmark_control.py`:

```python
"""Expression-exemplar selection and MediaPipe mesh rendering for the
landmark-control spike.

Exemplars are real FFHQ images chosen by their stored ARKit blendshape
coefficients (`bs_*` columns in reverse_index.parquet) — we do NOT synthesize
blendshape geometry. The selected image's MediaPipe face mesh becomes the
InfuseNet spatial-control image.
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION

from arkit_controlnet.axes import AXES
from arkit_controlnet.eval_spike import ARKIT_BLENDSHAPE_NAMES, face_landmarks_xy

REVERSE_INDEX = Path("output/reverse_index/reverse_index.parquet")
FFHQ_IMAGES = Path("output/ffhq_images")

# the 51 expression channels (every bs_ column except the `_neutral` summary)
_EXPR_COLS = [f"bs_{n}" for n in ARKIT_BLENDSHAPE_NAMES if n != "_neutral"]


def _ffhq_frame() -> pd.DataFrame:
    """FFHQ rows with a detected face and a PNG on disk; bs columns only."""
    on_disk = {p.stem for p in FFHQ_IMAGES.glob("*.png")}
    df = pd.read_parquet(
        REVERSE_INDEX,
        columns=["image_sha256", "source", "bs_detected", *_EXPR_COLS],
    )
    return df[
        (df["source"] == "ffhq")
        & (df["bs_detected"] == True)  # noqa: E712 — pandas mask, not identity
        & (df["image_sha256"].isin(list(on_disk)))
    ].copy()


def _paths(shas) -> list[Path]:
    return [FFHQ_IMAGES / f"{s}.png" for s in shas]


def select_exemplars(axis_name: str, k: int = 3) -> list[Path]:
    """Top-k FFHQ images by the sum of `axis`'s ARKit target channels.

    Returns k candidates (not 1) so the caller can fall through if MediaPipe
    fails to re-detect an exemplar at mesh-render time.
    """
    cols = [f"bs_{c}" for c in AXES[axis_name].target_channels]
    df = _ffhq_frame()
    df["score"] = df[cols].sum(axis=1)
    return _paths(df.nlargest(k, "score")["image_sha256"])


def select_neutral(k: int = 3) -> list[Path]:
    """The k FFHQ images with the least total expression energy (baseline)."""
    df = _ffhq_frame()
    df["energy"] = df[_EXPR_COLS].abs().sum(axis=1)
    return _paths(df.nsmallest(k, "energy")["image_sha256"])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/arkit_controlnet/test_landmark_control.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/arkit_controlnet/landmark_control.py tests/arkit_controlnet/test_landmark_control.py
git commit -m "feat(arkit-cn): expression-exemplar selection from parquet"
```

---

### Task 3: landmark_control — render the normalized face-mesh control image

**Files:**
- Modify: `src/arkit_controlnet/landmark_control.py`
- Test: `tests/arkit_controlnet/test_landmark_control.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/arkit_controlnet/test_landmark_control.py`:

```python
import numpy as np
from arkit_controlnet.landmark_control import render_landmark_mesh

_FIXTURE = Path("tests/arkit_controlnet/fixtures/face.png")


@pytest.mark.skipif(not _FIXTURE.exists(), reason="fixture face.png not present")
def test_render_landmark_mesh_shape_and_ink():
    mesh = render_landmark_mesh(_FIXTURE)
    assert mesh.shape == (1152, 864, 3)        # (H, W, 3)
    assert mesh.dtype == np.uint8
    white = int((mesh > 0).any(axis=2).sum())
    assert white > 2000                         # mesh edges actually drawn


@pytest.mark.skipif(not _FIXTURE.exists(), reason="fixture face.png not present")
def test_render_is_centered_in_canvas():
    # the drawn mesh's centroid sits near the canvas centre, not a corner
    mesh = render_landmark_mesh(_FIXTURE)
    ys, xs = np.where((mesh > 0).any(axis=2))
    assert abs(xs.mean() - 864 / 2) < 864 * 0.15
    assert abs(ys.mean() - 1152 * 0.42) < 1152 * 0.15
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/arkit_controlnet/test_landmark_control.py -v -k render`
Expected: FAIL — `ImportError: cannot import name 'render_landmark_mesh'`.

- [ ] **Step 3: Implement the renderer**

Append to `src/arkit_controlnet/landmark_control.py`:

```python
# control-image canvas — matches EmptyLatentImage in the spike workflow
_CANVAS_W, _CANVAS_H = 864, 1152
_FACE_FRAC = 0.45   # face bbox height as a fraction of canvas height
_CENTER_Y = 0.42    # vertical placement of the face centre (portrait framing)


def render_landmark_mesh(image_path: Path) -> np.ndarray:
    """Render an image's MediaPipe face mesh as a normalized control image.

    The mesh is recentred and isotropically scaled so the face occupies a
    fixed fraction of an 864x1152 canvas — so the control image also pins head
    size and position, not just expression. White tessellation edges on black.
    Returns an (H, W, 3) uint8 array. Raises ValueError if no face is detected.
    """
    lm = face_landmarks_xy(image_path)            # (478, 2) in [0, 1]
    xs, ys = lm[:, 0], lm[:, 1]
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    bh = ys.max() - ys.min()
    scale = (_FACE_FRAC * _CANVAS_H) / max(bh, 1e-6)
    px = (xs - cx) * scale + _CANVAS_W / 2
    py = (ys - cy) * scale + _CENTER_Y * _CANVAS_H
    pts = np.stack([px, py], axis=1).astype(np.int32)

    canvas = np.zeros((_CANVAS_H, _CANVAS_W, 3), dtype=np.uint8)
    for a, b in FACEMESH_TESSELATION:
        cv2.line(canvas, tuple(pts[a]), tuple(pts[b]), (255, 255, 255), 1,
                 lineType=cv2.LINE_AA)
    return canvas
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/arkit_controlnet/test_landmark_control.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add src/arkit_controlnet/landmark_control.py tests/arkit_controlnet/test_landmark_control.py
git commit -m "feat(arkit-cn): render normalized MediaPipe mesh control image"
```

---

### Task 4: the ComfyUI graph — InfuseNet identity + mesh control slot

**Files:**
- Create: `comfyui/workflows/arkit_landmark_spike.json`

- [ ] **Step 1: Create the workflow JSON**

This is `arkit_controlnet_spike.json` with the FluxSpace nodes (13/14/15) removed — `KSampler.model` wires straight from `UNETLoader` — and node 8 changed from a blank `EmptyImage` to a `LoadImage` of the rendered mesh. Create `comfyui/workflows/arkit_landmark_spike.json`:

```json
{
  "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "FLUX1/flux1-dev-fp8.safetensors", "weight_dtype": "default"}},
  "2": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": "t5/t5xxl_fp8_e4m3fn.safetensors", "clip_name2": "clip_l.safetensors", "type": "flux"}},
  "3": {"class_type": "VAELoader", "inputs": {"vae_name": "FLUX1/ae.safetensors"}},
  "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 864, "height": 1152, "batch_size": 1}},
  "5": {"class_type": "LoadImage", "inputs": {"image": "$$IDENTITY_FILENAME"}},
  "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a portrait photograph of a person, plain background", "clip": ["2", 0]}},
  "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
  "8": {"class_type": "LoadImage", "inputs": {"image": "$$CONTROL_FILENAME"}},
  "9": {"class_type": "IDEmbeddingModelLoader", "inputs": {"image_proj_model_name": "sim_stage1/image_proj_model.bin", "image_proj_num_tokens": 8, "face_analysis_provider": "CUDA", "face_analysis_det_size": "AUTO"}},
  "10": {"class_type": "ExtractIDEmbedding", "inputs": {"face_detector": ["9", 0], "arcface_model": ["9", 1], "image_proj_model": ["9", 2], "image": ["5", 0]}},
  "11": {"class_type": "InfuseNetLoader", "inputs": {"controlnet_name": "sim_stage1/infusenet_sim_fp8e4m3fn.safetensors"}},
  "12": {"class_type": "InfuseNetApply", "inputs": {"positive": ["6", 0], "negative": ["7", 0], "id_embedding": ["10", 0], "control_net": ["11", 0], "image": ["8", 0], "vae": ["3", 0], "strength": "$$STRENGTH", "start_percent": 0.0, "end_percent": 1.0}},
  "16": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["12", 0], "negative": ["12", 1], "latent_image": ["4", 0], "seed": "$$SEED", "steps": 25, "cfg": 1.0, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}},
  "17": {"class_type": "VAEDecode", "inputs": {"samples": ["16", 0], "vae": ["3", 0]}},
  "18": {"class_type": "SaveImage", "inputs": {"images": ["17", 0], "filename_prefix": "$$OUTPUT_PREFIX"}}
}
```

Note: node IDs 13/14/15 are intentionally absent — ComfyUI does not require contiguous IDs. `$$STRENGTH` is the swept InfuseNet control strength.

- [ ] **Step 2: Verify the JSON parses and has no dangling wires**

Run:
```bash
PYTHONPATH=src python3 -c "
import json
wf = json.load(open('comfyui/workflows/arkit_landmark_spike.json'))
ids = set(wf)
for nid, node in wf.items():
    for v in node['inputs'].values():
        if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
            assert v[0] in ids, f'{nid} wires to missing node {v[0]}'
print('nodes:', sorted(ids, key=int), '-- wiring OK')
"
```
Expected: `wiring OK` and the node list `['1','2','3','4','5','6','7','8','9','10','11','12','16','17','18']`.

- [ ] **Step 3: Commit**

```bash
git add comfyui/workflows/arkit_landmark_spike.json
git commit -m "feat(arkit-cn): landmark-control spike workflow (mesh in InfuseNet slot)"
```

---

### Task 5: the driver — sweep identities x axes x strength

**Files:**
- Create: `src/arkit_controlnet/run_landmark_spike.py`

- [ ] **Step 1: Write the driver**

Create `src/arkit_controlnet/run_landmark_spike.py`:

```python
"""Drive the landmark-control spike: 3 identities x 3 axes x strength band.

For each axis, picks one high-coefficient FFHQ exemplar, renders its MediaPipe
mesh as the InfuseNet control image, and generates each identity under that
control. A neutral-expression control gives the per-identity baseline. Scores
identity drift (ArcFace cosine) and expression match (blendshape cosine vs the
exemplar). Resumable: skips a fresh output PNG. See
docs/superpowers/specs/2026-05-18-arkit-landmark-control-spike-design.md.
"""
import asyncio
import copy
import json
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

from arkit_controlnet.eval_spike import arcface_cos, expr_cos
from arkit_controlnet.landmark_control import (
    render_landmark_mesh, select_exemplars, select_neutral,
)
from arkit_controlnet.run_spike import select_identities
from demographic_pc.comfy_flux import ComfyClient

WORKFLOW = Path("comfyui/workflows/arkit_landmark_spike.json")
OUT_DIR = Path("exp_output/arkit_landmark_spike")
SEED = 2026
STRENGTHS = [0.6, 1.0]
AXES_TO_RUN = ["smile", "pucker", "surprise"]
_MIN_PNG_BYTES = 1024


def _is_fresh(png: Path) -> bool:
    return png.exists() and png.stat().st_size >= _MIN_PNG_BYTES


def build_workflow(identity_filename: str, control_filename: str,
                   strength: float, out_prefix: str) -> dict:
    """Substitute the $$-placeholders in the workflow template."""
    subs: dict[str, Any] = {
        "$$IDENTITY_FILENAME": identity_filename,
        "$$CONTROL_FILENAME": control_filename,
        "$$STRENGTH": float(strength),
        "$$SEED": int(SEED),
        "$$OUTPUT_PREFIX": out_prefix,
    }

    def _sub(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: _sub(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_sub(v) for v in node]
        if isinstance(node, str) and node in subs:
            return subs[node]
        return node

    return _sub(copy.deepcopy(json.loads(WORKFLOW.read_text())))


def _prepare_controls() -> dict[str, dict[str, Path]]:
    """Render one control mesh PNG per axis (+ neutral) into OUT_DIR.

    Returns {axis: {"mesh": Path, "exemplar": Path}}. Falls through the k=3
    exemplar candidates if MediaPipe re-detection fails on the first.
    """
    controls: dict[str, dict[str, Path]] = {}
    specs = [(ax, select_exemplars(ax, k=3)) for ax in AXES_TO_RUN]
    specs.append(("neutral", select_neutral(k=3)))
    for axis, candidates in specs:
        for exemplar in candidates:
            try:
                mesh = render_landmark_mesh(exemplar)
            except ValueError:
                continue
            mesh_png = OUT_DIR / f"control_{axis}.png"
            cv2.imwrite(str(mesh_png), mesh)
            controls[axis] = {"mesh": mesh_png, "exemplar": exemplar}
            break
        else:
            raise RuntimeError(f"no exemplar yielded a mesh for axis {axis}")
    return controls


async def _run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    identities = select_identities(3)
    controls = _prepare_controls()
    rows = []
    async with ComfyClient() as client:
        for ident in identities:
            id_name = await client.upload_image(ident)
            for axis in [*AXES_TO_RUN, "neutral"]:
                ctl = controls[axis]
                ctl_name = await client.upload_image(ctl["mesh"])
                for strength in STRENGTHS:
                    tag = f"{ident.stem}__{axis}__str{strength:.2f}"
                    out_png = OUT_DIR / f"{tag}.png"
                    if not _is_fresh(out_png):
                        wf = build_workflow(id_name, ctl_name, strength, tag)
                        try:
                            await client.generate(wf, out_png)
                        except Exception as exc:
                            print(f"  FAILED {tag}: {exc}")
                            continue
                    rows.append({
                        "identity": ident.stem, "axis": axis, "strength": strength,
                        "arcface_cos": arcface_cos(out_png, ident),
                        "expr_cos": expr_cos(out_png, ctl["exemplar"]),
                    })
    df = pd.DataFrame(rows)
    df.to_parquet(OUT_DIR / "metrics.parquet")
    print(df.to_string(index=False))


def run() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    run()
```

- [ ] **Step 2: Dry-run the non-GPU parts**

Run:
```bash
PYTHONPATH=src python3 -c "
from arkit_controlnet.run_landmark_spike import build_workflow, _prepare_controls
ctl = _prepare_controls()
print('controls:', {k: v['mesh'].name for k, v in ctl.items()})
wf = build_workflow('id.png', 'control_smile.png', 0.6, 'tag')
import json
left = [v for v in json.dumps(wf).split('\"') if v.startswith('\$\$')]
print('placeholders left:', left)
assert not left
print('OK')
"
```
Expected: prints a control mesh per axis (`smile`, `pucker`, `surprise`, `neutral`), `placeholders left: []`, `OK`. Four `control_*.png` files appear under `exp_output/arkit_landmark_spike/`.

- [ ] **Step 3: Commit**

```bash
git add src/arkit_controlnet/run_landmark_spike.py
git commit -m "feat(arkit-cn): landmark-control spike driver"
```

---

### Task 6: review, run, verdict

**Files:**
- Create: `docs/research/2026-05-18-arkit-landmark-control-spike-verdict.md`
- Modify: `docs/research/_topics/arkit-controlnet.md`

- [ ] **Step 1: Code review**

Dispatch the `superpowers:code-reviewer` agent over the diff of Tasks 1-5 against the design doc. Fix any real bugs it finds; commit fixes as `fix(arkit-cn): apply code review to landmark spike`.

- [ ] **Step 2: Confirm ComfyUI is up with the InfiniteYou nodes**

Run:
```bash
curl -s http://127.0.0.1:8188/object_info | python3 -c "
import sys, json
d = json.load(sys.stdin)
for n in ['InfuseNetLoader','InfuseNetApply','IDEmbeddingModelLoader','ExtractIDEmbedding']:
    print(('OK  ' if n in d else 'MISS'), n)
"
```
Expected: all four `OK`. If any `MISS`, restart ComfyUI (`/home/newub/w/ComfyUI/`).

- [ ] **Step 3: Run the spike**

Run: `PYTHONPATH=src python3 -m arkit_controlnet.run_landmark_spike`
Expected: 24 cases (3 identities x 4 controls x 2 strengths), a printed metrics table, and `exp_output/arkit_landmark_spike/metrics.parquet`. ~25-30 min on the RTX 5090.

- [ ] **Step 4: Build the collage**

Reuse the collage script from the prior spike (`exp_output/arkit_controlnet_spike/collage.png` was built by an inline script — adapt it: rows = identities, columns = source + each control mesh + the output per axis at the chosen strength). Save to `exp_output/arkit_landmark_spike/collage.png`.

- [ ] **Step 5: Write the verdict doc**

Create `docs/research/2026-05-18-arkit-landmark-control-spike-verdict.md` with frontmatter `status: live`, `topic: arkit-controlnet`. State, against the design's success criteria: (a) did identity hold (`arcface_cos`), (b) did `expr_cos(output, exemplar)` beat the neutral-control baseline by >= 0.05, (c) eyeball verdict from the collage. If negative, recommend the documented fallback (stock depth ControlNet stacked, or sparse-contour modality). Include the collage.

- [ ] **Step 6: Update the topic index and commit**

Add a dated subsection to `docs/research/_topics/arkit-controlnet.md` summarizing the verdict, and mark the Path 1 open question resolved. Commit:

```bash
git add docs/research/2026-05-18-arkit-landmark-control-spike-verdict.md docs/research/_topics/arkit-controlnet.md
git commit -m "docs(arkit-cn): landmark-control spike verdict"
```

---

## Notes for the engineer

- `landmark_control.py` redefines `REVERSE_INDEX` / `FFHQ_IMAGES` rather than importing them from `run_spike.py` — this keeps the selection module free of a dependency on the runner. The two copies are intentional and acceptable for a spike.
- The `df["bs_detected"] == True` comparison triggers a linter `E712`; it is a pandas boolean mask, not a Python identity check, and the `# noqa` is deliberate.
- If `_prepare_controls` raises "no exemplar yielded a mesh", raise `k` in `select_exemplars`/`select_neutral` — MediaPipe failing on 3 straight top-ranked FFHQ faces would itself be a finding worth noting.
