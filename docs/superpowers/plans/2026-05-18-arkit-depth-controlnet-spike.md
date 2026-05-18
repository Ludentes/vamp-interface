# ARKit Depth-ControlNet Stacked Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether a stock FLUX Depth ControlNet, driven by a landmark-rasterized depth map and stacked alongside identity-only InfuseNet, steers FLUX output expression with zero training.

**Architecture:** One ComfyUI graph stacks two controls on shared FLUX conditioning — InstantX ControlNet-Union in depth mode (expression) chained before InfuseNet identity tokens (identity, fed an all-black spatial control). A depth map is rasterized from MediaPipe's 478 3D landmarks via a z-sorted painter's algorithm. A sweep runner scores ArcFace identity drift and blendshape expression match across 3 identities × 4 axes × 2 depth strengths.

**Tech Stack:** Python 3.12, uv, MediaPipe FaceLandmarker, OpenCV, ComfyUI API, pandas/pyarrow, pytest.

---

The design doc this plan implements: `docs/superpowers/specs/2026-05-18-arkit-depth-controlnet-spike-design.md`. Read it first.

## File Structure

- `src/arkit_controlnet/eval_spike.py` — **modify**: add `face_landmarks_xyz` (3D landmark accessor). Existing `face_landmarks_xy`, metrics untouched.
- `src/arkit_controlnet/landmark_control.py` — **modify**: add `render_depth_map`. Existing selection + mesh renderer untouched.
- `comfyui/workflows/arkit_depth_spike.json` — **create**: stacked depth-ControlNet + InfuseNet workflow.
- `src/arkit_controlnet/run_depth_spike.py` — **create**: the sweep runner.
- `tests/arkit_controlnet/test_eval_spike.py` — **modify**: add `face_landmarks_xyz` test.
- `tests/arkit_controlnet/test_landmark_control.py` — **modify**: add `render_depth_map` tests.
- `docs/research/2026-05-18-arkit-depth-controlnet-spike-verdict.md` — **create**: verdict.

Tests are gated behind a `face.png` fixture / `reverse_index` parquet existence check, matching the existing files.

---

### Task 1: 3D landmark accessor

**Files:**
- Modify: `src/arkit_controlnet/eval_spike.py` (append after `face_landmarks_xy`, ends line 128)
- Test: `tests/arkit_controlnet/test_eval_spike.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/arkit_controlnet/test_eval_spike.py` — extend the import block to include `face_landmarks_xyz`, then append:

```python
@pytest.mark.skipif(not FIXTURE.exists(), reason="needs a face fixture image")
def test_face_landmarks_xyz_returns_478_points_with_xy_matching_2d():
    xyz = face_landmarks_xyz(FIXTURE)
    assert xyz.shape == (478, 3)
    # x,y columns are exactly the 2D accessor's output
    np.testing.assert_allclose(xyz[:, :2], face_landmarks_xy(FIXTURE))
    # z is a real depth signal, not a constant
    assert xyz[:, 2].std() > 0.0
```

The import block becomes:

```python
from arkit_controlnet.eval_spike import (
    ARKIT_BLENDSHAPE_NAMES,
    arcface_cos,
    bs_read,
    bs_vector,
    expr_cos,
    face_landmarks_xy,
    face_landmarks_xyz,
)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/arkit_controlnet/test_eval_spike.py::test_face_landmarks_xyz_returns_478_points_with_xy_matching_2d -v`
Expected: FAIL — `ImportError: cannot import name 'face_landmarks_xyz'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/arkit_controlnet/eval_spike.py`:

```python
def face_landmarks_xyz(image_path: Path) -> np.ndarray:
    """(478, 3) MediaPipe face landmarks: normalized [0,1] (x, y) plus raw
    head-centred relative depth z (more negative = closer to the camera).

    Raises ValueError if no face is detected — callers select among several
    candidates, so a hard failure is correct here.
    """
    arr = cv2.cvtColor(_imread(image_path), cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
    res = _get_landmarker().detect(mp_image)
    if not res.face_landmarks:
        raise ValueError(f"no face detected in {image_path}")
    return np.array([[p.x, p.y, p.z] for p in res.face_landmarks[0]],
                    dtype=np.float64)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/arkit_controlnet/test_eval_spike.py -v`
Expected: PASS (all eval_spike tests, including the new one).

- [ ] **Step 5: Commit**

```bash
git add src/arkit_controlnet/eval_spike.py tests/arkit_controlnet/test_eval_spike.py
git commit -m "feat(arkit-cn): face_landmarks_xyz 3D landmark accessor"
```

---

### Task 2: Depth-map rasterizer

**Files:**
- Modify: `src/arkit_controlnet/landmark_control.py` (add `render_depth_map` after `render_landmark_mesh`, ends line 93; extend the `eval_spike` import on line 17)
- Test: `tests/arkit_controlnet/test_landmark_control.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/arkit_controlnet/test_landmark_control.py` — and extend its import to add `render_depth_map`:

```python
def test_render_depth_map_shape_and_dtype():
    depth = render_depth_map(_FIXTURE)
    assert depth.shape == (1152, 864, 3)
    assert depth.dtype == np.uint8


def test_render_depth_map_has_near_and_far_pixels():
    depth = render_depth_map(_FIXTURE)
    g = depth[:, :, 0]
    assert g.max() > 200          # a near (nose-tip-bright) region exists
    assert (g == 0).sum() > 0     # black far background exists


def test_render_depth_map_blob_is_centred():
    depth = render_depth_map(_FIXTURE)
    ys, _ = np.where(depth[:, :, 0] > 30)
    # the painted face blob sits near _CENTER_Y (0.42) of the 1152-px canvas
    assert 0.30 < ys.mean() / 1152 < 0.55
```

Import block becomes:

```python
from arkit_controlnet.landmark_control import (
    REVERSE_INDEX,
    render_depth_map,
    render_landmark_mesh,
    select_exemplars,
    select_neutral,
)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/arkit_controlnet/test_landmark_control.py::test_render_depth_map_shape_and_dtype -v`
Expected: FAIL — `ImportError: cannot import name 'render_depth_map'`.

- [ ] **Step 3: Write minimal implementation**

In `src/arkit_controlnet/landmark_control.py`, change the `eval_spike` import line (currently line 17) to:

```python
from arkit_controlnet.eval_spike import (
    ARKIT_BLENDSHAPE_NAMES, face_landmarks_xy, face_landmarks_xyz,
)
```

Then append after `render_landmark_mesh`:

```python
def render_depth_map(image_path: Path) -> np.ndarray:
    """Render an image's MediaPipe face mesh as a grayscale depth control image.

    The 478 landmarks' x,y are recentred and isotropically scaled with the same
    framing constants as render_landmark_mesh, so the depth map pins head size
    and position identically. The tessellation triangles are flat-shaded by
    their mean MediaPipe z (nearest -> white, farthest -> black) and painted
    farthest-first (painter's algorithm) so nearer geometry occludes farther.
    Background stays black (far). Returns an (H, W, 3) uint8 array. Raises
    ValueError if no face is detected.
    """
    lm = face_landmarks_xyz(image_path)           # (478, 3)
    xs, ys, zs = lm[:, 0], lm[:, 1], lm[:, 2]
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    bh = ys.max() - ys.min()
    scale = (_FACE_FRAC * _CANVAS_H) / max(bh, 1e-6)
    px = (xs - cx) * scale + _CANVAS_W / 2
    py = (ys - cy) * scale + _CENTER_Y * _CANVAS_H
    pts = np.stack([px, py], axis=1).astype(np.int32)

    # per-vertex grayscale depth: nearest (most negative z) -> 255
    z_lo, z_hi = zs.min(), zs.max()
    grays = 255.0 * (z_hi - zs) / max(z_hi - z_lo, 1e-6)

    tris = np.array(_DEPTH_TRIANGLES, dtype=np.int32)        # (n_tri, 3)
    tri_z = zs[tris].mean(axis=1)                            # mean depth / tri
    order = np.argsort(-tri_z)                               # farthest first

    canvas = np.zeros((_CANVAS_H, _CANVAS_W), dtype=np.uint8)
    for i in order:
        tri = tris[i]
        shade = int(grays[tri].mean())
        cv2.fillConvexPoly(canvas, pts[tri], shade, lineType=cv2.LINE_AA)
    return np.repeat(canvas[:, :, None], 3, axis=2)
```

`FACEMESH_TESSELATION` is already imported (line 14:
`from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION`)
— no new import needed. The tessellation ships as an edge set, not triangles, so
build the triangle list once at import. Below the `_CENTER_Y` constant block add:

```python
# FACEMESH_TESSELATION is an edge set; recover the triangle faces once. Each
# triangle is three mutually-connected vertices — collect 3-cliques from the
# adjacency. MediaPipe's tessellation is a triangle mesh, so every face shows
# up as a closed 3-cycle of edges.
def _tessellation_triangles() -> list[tuple[int, int, int]]:
    adj: dict[int, set[int]] = {}
    for a, b in FACEMESH_TESSELATION:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    tris: set[tuple[int, int, int]] = set()
    for a, b in FACEMESH_TESSELATION:
        for c in adj[a] & adj[b]:
            tris.add(tuple(sorted((a, b, c))))
    return sorted(tris)


_DEPTH_TRIANGLES = _tessellation_triangles()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/arkit_controlnet/test_landmark_control.py -v`
Expected: PASS (all landmark_control tests, including the three new ones).

- [ ] **Step 5: Commit**

```bash
git add src/arkit_controlnet/landmark_control.py tests/arkit_controlnet/test_landmark_control.py
git commit -m "feat(arkit-cn): landmark depth-map rasterizer"
```

---

### Task 3: Stacked depth-ControlNet workflow JSON

**Files:**
- Create: `comfyui/workflows/arkit_depth_spike.json`

This is a static asset — no test code; verification is a structural check in Step 2.

- [ ] **Step 1: Write the workflow file**

Create `comfyui/workflows/arkit_depth_spike.json`. It extends `arkit_landmark_spike.json`: node 8 (the mesh `LoadImage`) is dropped; nodes 19–23 add the depth ControlNet and the two new images. InfuseNet (node 12) now consumes the ControlNet's conditioning and an **all-black** spatial control, at a **fixed strength 0.6**.

```json
{
  "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "FLUX1/flux1-dev-fp8.safetensors", "weight_dtype": "default"}},
  "2": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": "t5/t5xxl_fp8_e4m3fn.safetensors", "clip_name2": "clip_l.safetensors", "type": "flux"}},
  "3": {"class_type": "VAELoader", "inputs": {"vae_name": "FLUX1/ae.safetensors"}},
  "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 864, "height": 1152, "batch_size": 1}},
  "5": {"class_type": "LoadImage", "inputs": {"image": "$$IDENTITY_FILENAME"}},
  "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a portrait photograph of a person, plain background", "clip": ["2", 0]}},
  "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
  "9": {"class_type": "IDEmbeddingModelLoader", "inputs": {"image_proj_model_name": "sim_stage1/image_proj_model.bin", "image_proj_num_tokens": 8, "face_analysis_provider": "CUDA", "face_analysis_det_size": "AUTO"}},
  "10": {"class_type": "ExtractIDEmbedding", "inputs": {"face_detector": ["9", 0], "arcface_model": ["9", 1], "image_proj_model": ["9", 2], "image": ["5", 0]}},
  "11": {"class_type": "InfuseNetLoader", "inputs": {"controlnet_name": "sim_stage1/infusenet_sim_fp8e4m3fn.safetensors"}},
  "19": {"class_type": "ControlNetLoader", "inputs": {"control_net_name": "FLUX.1/instantx-union/diffusion_pytorch_model.safetensors"}},
  "20": {"class_type": "SetUnionControlNetType", "inputs": {"control_net": ["19", 0], "type": "depth"}},
  "21": {"class_type": "LoadImage", "inputs": {"image": "$$BLACK_FILENAME"}},
  "22": {"class_type": "LoadImage", "inputs": {"image": "$$DEPTH_FILENAME"}},
  "23": {"class_type": "ControlNetApplyAdvanced", "inputs": {"positive": ["6", 0], "negative": ["7", 0], "control_net": ["20", 0], "image": ["22", 0], "strength": "$$DEPTH_STRENGTH", "start_percent": 0.0, "end_percent": 1.0, "vae": ["3", 0]}},
  "12": {"class_type": "InfuseNetApply", "inputs": {"positive": ["23", 0], "negative": ["23", 1], "id_embedding": ["10", 0], "control_net": ["11", 0], "image": ["21", 0], "vae": ["3", 0], "strength": 0.6, "start_percent": 0.0, "end_percent": 1.0}},
  "16": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["12", 0], "negative": ["12", 1], "latent_image": ["4", 0], "seed": "$$SEED", "steps": 25, "cfg": 1.0, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}},
  "17": {"class_type": "VAEDecode", "inputs": {"samples": ["16", 0], "vae": ["3", 0]}},
  "18": {"class_type": "SaveImage", "inputs": {"images": ["17", 0], "filename_prefix": "$$OUTPUT_PREFIX"}}
}
```

- [ ] **Step 2: Verify the workflow is structurally sound**

Run:

```bash
uv run python -c "
import json
w = json.load(open('comfyui/workflows/arkit_depth_spike.json'))
ids = set(w)
for nid, node in w.items():
    for v in node['inputs'].values():
        if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
            assert v[0] in ids, f'{nid} wires to missing node {v[0]}'
assert w['12']['inputs']['positive'] == ['23', 0], 'InfuseNet must consume ControlNet conditioning'
assert w['12']['inputs']['image'] == ['21', 0], 'InfuseNet spatial control must be the black image'
assert w['12']['inputs']['strength'] == 0.6, 'InfuseNet strength fixed at 0.6'
assert w['23']['inputs']['strength'] == '\$\$DEPTH_STRENGTH'
assert w['20']['inputs']['type'] == 'depth'
print('workflow OK:', len(w), 'nodes')
"
```

Expected: `workflow OK: 16 nodes`.

- [ ] **Step 3: Commit**

```bash
git add comfyui/workflows/arkit_depth_spike.json
git commit -m "feat(arkit-cn): stacked depth-ControlNet + InfuseNet workflow"
```

---

### Task 4: Sweep runner

**Files:**
- Create: `src/arkit_controlnet/run_depth_spike.py`

The runner mirrors `run_landmark_spike.py`. No unit test — it is a thin
ComfyUI-driving script; correctness is established by Task 5's actual run.
Verification here is import + a dry structural check.

- [ ] **Step 1: Write the runner**

Create `src/arkit_controlnet/run_depth_spike.py`:

```python
"""Drive the depth-ControlNet stacked spike: 3 identities x 4 axes x strength.

For each axis, picks one high-coefficient FFHQ exemplar, rasterizes its
MediaPipe face mesh into a depth map, and drives a stock FLUX Depth ControlNet
with it while identity-only InfuseNet (black spatial control, fixed strength
0.6) holds the face. A neutral-expression depth map gives the per-identity
baseline. Scores identity drift (ArcFace cosine) and expression match
(blendshape cosine vs the exemplar). Resumable: skips a fresh output PNG. See
docs/superpowers/specs/2026-05-18-arkit-depth-controlnet-spike-design.md.
"""
import asyncio
import copy
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from arkit_controlnet.eval_spike import arcface_cos, expr_cos
from arkit_controlnet.landmark_control import (
    render_depth_map, select_exemplars, select_neutral,
)
from arkit_controlnet.run_spike import select_identities
from demographic_pc.comfy_flux import ComfyClient

WORKFLOW = Path("comfyui/workflows/arkit_depth_spike.json")
OUT_DIR = Path("exp_output/arkit_depth_spike")
SEED = 2026
DEPTH_STRENGTHS = [0.5, 0.8]
AXES_TO_RUN = ["smile", "pucker", "surprise"]
_CANVAS_W, _CANVAS_H = 864, 1152
_MIN_PNG_BYTES = 1024


def _is_fresh(png: Path) -> bool:
    return png.exists() and png.stat().st_size >= _MIN_PNG_BYTES


def build_workflow(identity_filename: str, depth_filename: str,
                   black_filename: str, strength: float,
                   out_prefix: str) -> dict:
    """Substitute the $$-placeholders in the workflow template."""
    subs: dict[str, Any] = {
        "$$IDENTITY_FILENAME": identity_filename,
        "$$DEPTH_FILENAME": depth_filename,
        "$$BLACK_FILENAME": black_filename,
        "$$DEPTH_STRENGTH": float(strength),
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


def _prepare_inputs() -> tuple[dict[str, dict[str, Path]], Path]:
    """Render one depth map per axis (+ neutral) and one black image into
    OUT_DIR. Returns ({axis: {"depth": Path, "exemplar": Path}}, black_png).
    Falls through the k=3 exemplar candidates if MediaPipe re-detection fails.
    """
    inputs: dict[str, dict[str, Path]] = {}
    specs = [(ax, select_exemplars(ax, k=3)) for ax in AXES_TO_RUN]
    specs.append(("neutral", select_neutral(k=3)))
    for axis, candidates in specs:
        for exemplar in candidates:
            try:
                depth = render_depth_map(exemplar)
            except ValueError:
                continue
            depth_png = OUT_DIR / f"depth_{axis}.png"
            cv2.imwrite(str(depth_png), depth)
            inputs[axis] = {"depth": depth_png, "exemplar": exemplar}
            break
        else:
            raise RuntimeError(f"no exemplar yielded a depth map for axis {axis}")
    black_png = OUT_DIR / "black.png"
    cv2.imwrite(str(black_png),
                np.zeros((_CANVAS_H, _CANVAS_W, 3), dtype=np.uint8))
    return inputs, black_png


async def _run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    identities = select_identities(3)
    inputs, black_png = _prepare_inputs()
    rows = []
    # neutral first: a collapsed neutral row means the InfuseNet black-image
    # control failed, which makes the depth result inconclusive (see design).
    axis_order = ["neutral", *AXES_TO_RUN]
    async with ComfyClient() as client:
        black_name = await client.upload_image(black_png)
        for ident in identities:
            id_name = await client.upload_image(ident)
            for axis in axis_order:
                spec = inputs[axis]
                depth_name = await client.upload_image(spec["depth"])
                for strength in DEPTH_STRENGTHS:
                    tag = f"{ident.stem}__{axis}__str{strength:.2f}"
                    out_png = OUT_DIR / f"{tag}.png"
                    if not _is_fresh(out_png):
                        wf = build_workflow(id_name, depth_name, black_name,
                                            strength, tag)
                        try:
                            await client.generate(wf, out_png)
                        except Exception as exc:
                            print(f"  FAILED {tag}: {exc}")
                            continue
                    try:
                        af = arcface_cos(out_png, ident)
                        ec = expr_cos(out_png, spec["exemplar"])
                    except Exception as exc:
                        print(f"  METRIC FAILED {tag}: {exc}")
                        af, ec = -1.0, -1.0
                    rows.append({
                        "identity": ident.stem, "axis": axis,
                        "strength": strength, "arcface_cos": af,
                        "expr_cos": ec,
                    })
    df = pd.DataFrame(rows)
    df.to_parquet(OUT_DIR / "metrics.parquet")
    print(df.to_string(index=False))


def run() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    run()
```

- [ ] **Step 2: Verify it imports and the template substitutes cleanly**

Run:

```bash
uv run python -c "
from arkit_controlnet.run_depth_spike import build_workflow
wf = build_workflow('id.png', 'depth.png', 'black.png', 0.5, 'tag')
assert wf['23']['inputs']['strength'] == 0.5
assert wf['22']['inputs']['image'] == 'depth.png'
assert wf['21']['inputs']['image'] == 'black.png'
assert wf['5']['inputs']['image'] == 'id.png'
assert wf['16']['inputs']['seed'] == 2026
print('runner OK')
"
```

Expected: `runner OK`.

- [ ] **Step 3: Commit**

```bash
git add src/arkit_controlnet/run_depth_spike.py
git commit -m "feat(arkit-cn): depth-ControlNet spike sweep runner"
```

---

### Task 5: Run the spike and write the verdict

**Files:**
- Create: `docs/research/2026-05-18-arkit-depth-controlnet-spike-verdict.md`
- Modify: `docs/research/_topics/arkit-controlnet.md`

- [ ] **Step 1: Confirm ComfyUI is up**

Run: `curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8188/system_stats`
Expected: `200`. If not, start ComfyUI before continuing.

- [ ] **Step 2: Run the sweep**

Run: `uv run python -m arkit_controlnet.run_depth_spike`
Expected: a 24-row table printed; `exp_output/arkit_depth_spike/metrics.parquet` and 24 `*.png` written. Resumable — re-run if it stops partway.

- [ ] **Step 3: Build the comparison collage**

Run:

```bash
uv run python -c "
from pathlib import Path
import cv2, numpy as np
d = Path('exp_output/arkit_depth_spike')
ids = sorted({p.name.split('__')[0] for p in d.glob('*__*__str0.50.png')})
axes = ['neutral', 'smile', 'pucker', 'surprise']
rows = []
for i in ids:
    cells = []
    for ax in axes:
        p = d / f'{i}__{ax}__str0.50.png'
        img = cv2.imread(str(p)) if p.exists() else np.zeros((1152,864,3),np.uint8)
        cells.append(cv2.resize(img, (288, 384)))
    rows.append(np.hstack(cells))
cv2.imwrite(str(d / 'collage.png'), np.vstack(rows))
print('collage written')
"
```

Expected: `collage.png` written (rows = identities, columns = neutral/smile/pucker/surprise at depth strength 0.5).

- [ ] **Step 4: Compute baseline-relative expression deltas**

Run:

```bash
uv run python -c "
import pandas as pd
from pathlib import Path
from arkit_controlnet.eval_spike import expr_cos
from arkit_controlnet.landmark_control import select_exemplars
d = Path('exp_output/arkit_depth_spike')
df = pd.read_parquet(d / 'metrics.parquet')
exemplar = {ax: select_exemplars(ax, k=3) for ax in ['smile','pucker','surprise']}
# resolve the actual exemplar used (first that produced a depth map)
import json
for ident in sorted(df.identity.unique()):
    for ax in ['smile','pucker','surprise']:
        for s in sorted(df.strength.unique()):
            row = df[(df.identity==ident)&(df.axis==ax)&(df.strength==s)]
            neu = df[(df.identity==ident)&(df.axis=='neutral')&(df.strength==s)]
            if row.empty or neu.empty: continue
            print(ident[:8], ax, s,
                  'arcface=%.3f' % row.arcface_cos.iloc[0],
                  'expr=%.3f' % row.expr_cos.iloc[0],
                  'neutral_expr=%.3f' % neu.expr_cos.iloc[0])
"
```

This prints raw `arcface_cos` and `expr_cos` per cell plus the neutral cell's `expr_cos` at the same strength. The success delta is judged by re-scoring each output's blendshape vector against the *axis* exemplar versus the *neutral* output — note that `metrics.parquet`'s `expr_cos` already compares each output to its own axis exemplar, so for axis rows the delta vs the neutral row at the same strength is the criterion-2 quantity. Record which cells clear `arcface_cos ≥ 0.55` AND a `≥ 0.05` lift over neutral.

- [ ] **Step 5: Write the verdict doc**

Create `docs/research/2026-05-18-arkit-depth-controlnet-spike-verdict.md` with frontmatter `status: live`, `topic: arkit-controlnet`. Structure (mirror `2026-05-18-arkit-landmark-control-spike-verdict.md`):

- What was tested — stacked stock FLUX Depth ControlNet (InstantX Union, depth mode) + identity-only InfuseNet, landmark-rasterized depth map, zero training. 3 identities × 3 axes + neutral × depth strength {0.5, 0.8}.
- Artifacts — workflow, runner, `render_depth_map`, `exp_output/arkit_depth_spike/`.
- Verdict — state positive or negative against the three success criteria; include the per-cell pass/fail table (identities × axes) populated from Step 4; report whether neutral cells held identity (the InfuseNet black-image checkpoint from the design).
- Implication — if negative, all zero-train routes are exhausted and the thread proceeds to the CFM training run; if positive, depth-ControlNet is a viable zero-train expression channel and the next step is widening the axis/strength sweep.

- [ ] **Step 6: Update the topic index**

In `docs/research/_topics/arkit-controlnet.md`, add a subsection "Approach C — depth-ControlNet stacked spike (2026-05-18)" summarizing the verdict, and update the "Open questions" — strike the depth-ControlNet bullet, leaving CFM as the remaining path (if negative) or a sweep-widening follow-up (if positive).

- [ ] **Step 7: Commit**

```bash
git add docs/research/2026-05-18-arkit-depth-controlnet-spike-verdict.md docs/research/_topics/arkit-controlnet.md exp_output/arkit_depth_spike/metrics.parquet
git commit -m "docs(arkit-cn): depth-ControlNet stacked spike verdict"
```

(`exp_output/` PNGs are not committed — only the metrics parquet, matching the prior spike.)

---

### Task 6: Code review

**Files:** all created/modified in Tasks 1–4.

- [ ] **Step 1: Run the code reviewer**

Dispatch the `superpowers:code-reviewer` agent against the diff of this branch since the design-doc commit, with the design doc as the reference. Standing project rule: non-trivial code must pass code review before the task is declared done.

- [ ] **Step 2: Apply fixes**

Apply any "important" / "critical" findings inline; commit each fix with a `fix(arkit-cn): ...` message. Defer "nice-to-have" notes unless trivial.

- [ ] **Step 3: Final commit**

```bash
git add -A && git commit -m "fix(arkit-cn): address code review on depth-ControlNet spike" || echo "no review fixes needed"
```

---

## Self-Review

**Spec coverage:**
- `face_landmarks_xyz` → Task 1. ✓
- `render_depth_map` (z-sorted painter's algorithm, framing constants) → Task 2. ✓
- `arkit_depth_spike.json` (ControlNet stack, depth union type, black InfuseNet image, fixed 0.6) → Task 3. ✓
- `run_depth_spike.py` (3×4×2 sweep, neutral-first, resumable, try/except metrics) → Task 4. ✓
- Success criteria + collage + baseline-relative deltas → Task 5 Steps 3–4. ✓
- InfuseNet black-image collapse checkpoint → Task 4 (neutral-first ordering) + Task 5 Step 5 (verdict reports it). ✓
- Tests for `face_landmarks_xyz` and `render_depth_map` → Tasks 1–2. ✓
- Code review (standing rule) → Task 6. ✓

**Placeholder scan:** No TBD/TODO. The `$$` strings are intentional workflow-template markers, substituted by `build_workflow`.

**Type consistency:** `render_depth_map(Path) -> np.ndarray (1152,864,3) uint8`, `face_landmarks_xyz(Path) -> np.ndarray (478,3)` — used consistently across Tasks 1, 2, 4. `build_workflow` signature `(identity, depth, black, strength, prefix)` matches its Task 4 Step 2 call. Workflow placeholder names match between Task 3 JSON and Task 4 `subs` dict.
