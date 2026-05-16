# ARKit-ControlNet Zero-Train Feasibility Spike — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove, with zero training, that InfiniteYou (identity, residual injection) and FluxSpace (expression, attention editing) compose on FLUX.1-dev — `f(identity_photo, expression_axis, scale) → image` that holds identity and wears the expression.

**Architecture:** A ComfyUI graph stacks the InfU identity node and the project's `FluxSpaceEditPair` node on one frozen FLUX.1-dev. A Python eval harness measures identity drift (ArcFace cosine) and expression change (MediaPipe blendshapes). A runner sweeps 5 identities × the smile axis × edit scale and writes a metrics grid.

**Tech Stack:** ComfyUI, FLUX.1-dev, InfiniteYou (InfU), the `demographic_pc_fluxspace` custom nodes, insightface `buffalo_l`, MediaPipe FaceLandmarker, pandas.

**Spec:** `docs/superpowers/specs/2026-05-16-arkit-controlnet-spike-design.md`

---

## File Structure

- `src/arkit_controlnet/__init__.py` — new package.
- `src/arkit_controlnet/axes.py` — FluxSpace expression axis definitions (prompt pairs, scale bands).
- `src/arkit_controlnet/eval_spike.py` — ArcFace + blendshape readers and the two metrics.
- `src/arkit_controlnet/run_spike.py` — the runner: select identities, drive ComfyUI, collect metrics, write grid.
- `comfyui/workflows/arkit_controlnet_spike.json` — the InfU + FluxSpace ComfyUI graph (API format).
- `tests/arkit_controlnet/test_axes.py`, `tests/arkit_controlnet/test_eval_spike.py` — unit tests.
- `models/mediapipe/face_landmarker.task` — MediaPipe model asset (download in Task 0).

---

## Task 0: Environment and inventory

**Files:**
- Create: `docs/research/2026-05-16-arkit-spike-env-notes.md` (running notes)

- [ ] **Step 1: Confirm the FluxSpace nodes load**

Run: `ls /home/newub/w/ComfyUI/custom_nodes/demographic_pc_fluxspace/__init__.py`
Then start ComfyUI and confirm `FluxSpaceEditPair` appears in the node list.
Expected: file exists; node `FluxSpaceEditPair` registered (it defines
`INPUT_TYPES` with `edit_conditioning_a`, `edit_conditioning_b`, `scale`,
`mix_b`, `start_percent`, `end_percent`; `RETURN_TYPES = ("MODEL",)`).

- [ ] **Step 2: Install an InfiniteYou ComfyUI integration**

InfiniteYou ships official diffusers weights + code at
`github.com/bytedance/InfiniteYou` (model `ByteDance/InfiniteYou` on HF).
Research the current community ComfyUI node options (search for
"ComfyUI InfiniteYou"), pick one, and install it into
`/home/newub/w/ComfyUI/custom_nodes/`. Record the exact repo + commit in the
env-notes doc. If no maintained ComfyUI node exists, fall back to running InfU
via diffusers in a wrapper node — record that decision.
Expected: an InfU identity node appears in ComfyUI; it takes an identity image
+ FLUX model and returns a patched MODEL (or conditioning).

- [ ] **Step 3: Download weights**

Download FLUX.1-dev, the InfU/InfuseNet weights, and the MediaPipe model:
```bash
huggingface-cli download black-forest-labs/FLUX.1-dev --local-dir ~/w/ComfyUI/models/diffusers/FLUX.1-dev
huggingface-cli download ByteDance/InfiniteYou --local-dir ~/w/ComfyUI/models/infiniteyou
mkdir -p models/mediapipe
curl -L -o models/mediapipe/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```
Expected: weights present; `face_landmarker.task` ~3.8 MB.

- [ ] **Step 4: Commit env notes**

```bash
git add docs/research/2026-05-16-arkit-spike-env-notes.md
git commit -m "chore(arkit-spike): record env setup — InfU node, weights, mediapipe asset"
```

---

## Task 1: FluxSpace expression axis config

**Files:**
- Create: `src/arkit_controlnet/__init__.py`
- Create: `src/arkit_controlnet/axes.py`
- Test: `tests/arkit_controlnet/test_axes.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/arkit_controlnet/test_axes.py
from src.arkit_controlnet.axes import AXES, Axis

def test_smile_axis_is_defined_and_well_formed():
    smile = AXES["smile"]
    assert isinstance(smile, Axis)
    assert smile.edit_prompt_a and smile.edit_prompt_b
    assert smile.mix_b == 0.5
    # scale band is ascending and inside the verified smile window
    assert smile.scale_band == sorted(smile.scale_band)
    assert min(smile.scale_band) >= 0.0 and max(smile.scale_band) <= 2.3
    # the ARKit channels this axis is expected to move
    assert "mouthSmileLeft" in smile.target_channels
    assert "mouthSmileRight" in smile.target_channels

def test_three_axes_present():
    assert set(AXES) == {"smile", "pucker", "surprise"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/arkit_controlnet/test_axes.py -v`
Expected: FAIL — `ModuleNotFoundError: src.arkit_controlnet.axes`.

- [ ] **Step 3: Write the config**

```python
# src/arkit_controlnet/__init__.py
# (empty — package marker)
```

```python
# src/arkit_controlnet/axes.py
"""FluxSpace expression axes for the ARKit-ControlNet spike.

`smile` uses the verified prompt pair from the 2026-04-21 FluxSpace smile-axis
experiment (docs/research/2026-04-21-fluxspace-smile-axis.md). `pucker` and
`surprise` are built by analogy on the same template and are UNCHARACTERISED —
their scale bands are provisional and must be eyeball-checked in Task 2.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Axis:
    name: str
    # FluxSpaceEditPair averages two edit-prompt attention caches at mix_b.
    edit_prompt_a: str          # generic edit prompt
    edit_prompt_b: str          # portrait-template edit prompt
    mix_b: float                # pair-averaging weight (0.5 = balanced)
    scale_band: list[float]     # edit scales to sweep
    target_channels: list[str]  # ARKit channels the axis should move


AXES: dict[str, Axis] = {
    "smile": Axis(
        name="smile",
        edit_prompt_a="A person smiling warmly.",
        edit_prompt_b=(
            "A photorealistic portrait photograph of a person smiling warmly, "
            "plain grey background, studio lighting, sharp focus."
        ),
        mix_b=0.5,
        scale_band=[0.5, 1.0, 1.5, 2.0],
        target_channels=["mouthSmileLeft", "mouthSmileRight"],
    ),
    "pucker": Axis(
        name="pucker",
        edit_prompt_a="A person puckering their lips.",
        edit_prompt_b=(
            "A photorealistic portrait photograph of a person puckering their "
            "lips, plain grey background, studio lighting, sharp focus."
        ),
        mix_b=0.5,
        scale_band=[0.5, 1.0, 1.5],
        target_channels=["mouthPucker", "mouthFunnel"],
    ),
    "surprise": Axis(
        name="surprise",
        edit_prompt_a="A person with a surprised expression, mouth open.",
        edit_prompt_b=(
            "A photorealistic portrait photograph of a person with a surprised "
            "expression and open mouth, plain grey background, studio lighting, "
            "sharp focus."
        ),
        mix_b=0.5,
        scale_band=[0.5, 1.0, 1.5],
        target_channels=["jawOpen", "browInnerUp", "eyeWideLeft", "eyeWideRight"],
    ),
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/arkit_controlnet/test_axes.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/arkit_controlnet/__init__.py src/arkit_controlnet/axes.py tests/arkit_controlnet/test_axes.py
git commit -m "feat(arkit-spike): FluxSpace expression axis config (smile verified, pucker/surprise by analogy)"
```

---

## Task 2: ComfyUI workflow

**Files:**
- Create: `comfyui/workflows/arkit_controlnet_spike.json`

- [ ] **Step 1: Build the graph in the ComfyUI UI**

Construct, left to right:
1. Load FLUX.1-dev (checkpoint / UNet + CLIP + VAE loaders, fp8).
2. The InfU identity node — input: an identity image (Load Image) + the FLUX
   MODEL. Output: a patched MODEL carrying the ArcFace residual injection.
3. Two `CLIPTextEncode` (or the FluxSpace conditioning encoders) for the axis's
   `edit_prompt_a` and `edit_prompt_b`.
4. `FluxSpaceEditPair` — inputs: the InfU-patched MODEL,
   `edit_conditioning_a`, `edit_conditioning_b`, `scale`, `mix_b=0.5`,
   `start_percent=0.0`, `end_percent=1.0`. Output: doubly-patched MODEL.
5. A neutral base prompt (`CLIPTextEncode`, e.g. `"a portrait photograph of a
   person, plain background"`) → positive conditioning.
6. `KSampler` (FLUX scheduler, fixed seed 2026) → `VAEDecode` → `SaveImage`.

- [ ] **Step 2: Smoke-test in the UI**

Run the graph once with the smile axis at `scale=1.0` on one FFHQ photo.
Expected: an image saves; the face resembles the identity photo and shows a
visible smile. If the InfU node and FluxSpaceEditPair cannot both patch the
MODEL in series, record the failure in the env-notes doc — this is a real spike
result (composition failure), not a blocker to debug indefinitely.

- [ ] **Step 3: Export and save the API-format workflow**

Export via "Save (API Format)" to `comfyui/workflows/arkit_controlnet_spike.json`.
Confirm the JSON has addressable nodes for: identity image path, edit prompt a,
edit prompt b, scale, seed, output filename.

- [ ] **Step 4: Commit**

```bash
git add comfyui/workflows/arkit_controlnet_spike.json
git commit -m "feat(arkit-spike): ComfyUI workflow — InfU identity + FluxSpaceEditPair expression"
```

---

## Task 3: Eval harness

**Files:**
- Create: `src/arkit_controlnet/eval_spike.py`
- Test: `tests/arkit_controlnet/test_eval_spike.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/arkit_controlnet/test_eval_spike.py
from pathlib import Path
import pytest
from src.arkit_controlnet.eval_spike import bs_read, arcface_cos, ARKIT_BLENDSHAPE_NAMES

FIXTURE = Path("tests/arkit_controlnet/fixtures/face.png")  # any clear single-face photo

@pytest.mark.skipif(not FIXTURE.exists(), reason="needs a face fixture image")
def test_bs_read_returns_52_named_channels():
    bs = bs_read(FIXTURE)
    assert set(bs) == set(ARKIT_BLENDSHAPE_NAMES)
    assert all(0.0 <= v <= 1.0 for v in bs.values())

@pytest.mark.skipif(not FIXTURE.exists(), reason="needs a face fixture image")
def test_arcface_cos_self_is_one():
    assert arcface_cos(FIXTURE, FIXTURE) > 0.99
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/arkit_controlnet/test_eval_spike.py -v`
Expected: FAIL — `ModuleNotFoundError: src.arkit_controlnet.eval_spike`.

- [ ] **Step 3: Write the eval harness**

```python
# src/arkit_controlnet/eval_spike.py
"""Spike metrics: identity drift (ArcFace cosine) and expression (blendshapes).

Both encoders are applied fresh to both images. We do NOT compare against the
stored reverse_index `arcface_fp32` column — that column and this harness may
use different ArcFace models, and cosine is only valid within one encoder.
"""
from pathlib import Path
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from insightface.app import FaceAnalysis

ARKIT_BLENDSHAPE_NAMES = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight", "cheekPuff",
    "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft",
    "mouthFrownRight", "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
]
assert len(ARKIT_BLENDSHAPE_NAMES) == 52

_MP_MODEL = Path("models/mediapipe/face_landmarker.task")
_landmarker = None
_arcface = None


def _get_landmarker():
    global _landmarker
    if _landmarker is None:
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(_MP_MODEL)),
            output_face_blendshapes=True,
            num_faces=1,
        )
        _landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
    return _landmarker


def _get_arcface():
    global _arcface
    if _arcface is None:
        app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"])
        app.prepare(ctx_id=0)
        _arcface = app
    return _arcface


def bs_read(image_path: Path) -> dict[str, float]:
    """52-d ARKit blendshapes from an image. Missing detection -> all zeros."""
    arr = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
    res = _get_landmarker().detect(mp_image)
    if not res.face_blendshapes:
        return {n: 0.0 for n in ARKIT_BLENDSHAPE_NAMES}
    got = {c.category_name: float(c.score) for c in res.face_blendshapes[0]}
    return {n: got.get(n, 0.0) for n in ARKIT_BLENDSHAPE_NAMES}


def arcface_cos(path_a: Path, path_b: Path) -> float:
    """Cosine similarity of buffalo_l embeddings. -1.0 if either has no face."""
    app = _get_arcface()
    fa = app.get(cv2.imread(str(path_a)))
    fb = app.get(cv2.imread(str(path_b)))
    if not fa or not fb:
        return -1.0
    ea, eb = fa[0].normed_embedding, fb[0].normed_embedding
    return float(np.dot(ea, eb))


def bs_delta(output_path: Path, identity_path: Path, channels: list[str]) -> float:
    """Mean increase in the axis's target channels, output minus identity."""
    out, idn = bs_read(output_path), bs_read(identity_path)
    return float(np.mean([out[c] - idn[c] for c in channels]))
```

- [ ] **Step 4: Add a fixture image**

Copy any clear single-face photo to `tests/arkit_controlnet/fixtures/face.png`
(an FFHQ sample is fine). Commit it so the tests are not skipped in CI.

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/arkit_controlnet/test_eval_spike.py -v`
Expected: PASS (2 tests) — `bs_read` returns 52 channels, self-cosine > 0.99.

- [ ] **Step 6: Commit**

```bash
git add src/arkit_controlnet/eval_spike.py tests/arkit_controlnet/test_eval_spike.py tests/arkit_controlnet/fixtures/face.png
git commit -m "feat(arkit-spike): eval harness — ArcFace cosine + MediaPipe blendshape delta"
```

---

## Task 4: Spike runner

**Files:**
- Create: `src/arkit_controlnet/run_spike.py`

- [ ] **Step 1: Write the runner**

```python
# src/arkit_controlnet/run_spike.py
"""Drive the ARKit-ControlNet spike: 5 identities x smile axis x scale band.

Selects identity photos from reverse_index, posts the ComfyUI workflow once per
(identity, axis, scale), then evaluates. Resumable: skips a case whose output
PNG already exists.
"""
import json
import time
import uuid
from pathlib import Path
import urllib.request
import pandas as pd

from src.arkit_controlnet.axes import AXES
from src.arkit_controlnet.eval_spike import arcface_cos, bs_delta

COMFY = "http://127.0.0.1:8188"
WORKFLOW = Path("comfyui/workflows/arkit_controlnet_spike.json")
OUT_DIR = Path("exp_output/arkit_controlnet_spike")
REVERSE_INDEX = Path("output/reverse_index/reverse_index.parquet")
SHA_LOOKUP = Path("output/reverse_index/ffhq_sha_lookup.parquet")
SEED = 2026


def select_identities(n: int = 5) -> list[Path]:
    """n FFHQ photos spread across the FairFace race buckets."""
    ri = pd.read_parquet(REVERSE_INDEX)
    lut = pd.read_parquet(SHA_LOOKUP)
    ffhq = ri[ri["source"] == "ffhq"].merge(lut, on="image_sha256", how="inner")
    race_col = next(c for c in ffhq.columns if "fairface" in c and "race" in c)
    picks = ffhq.groupby(race_col, group_keys=False).head(1).head(n)
    return [Path(p) for p in picks["shard_path"]]  # resolve to actual PNGs as needed


def submit(identity_png: Path, axis_name: str, scale: float, out_name: str) -> None:
    """Patch the workflow JSON's addressable fields and POST to ComfyUI."""
    wf = json.loads(WORKFLOW.read_text())
    axis = AXES[axis_name]
    # NOTE: the exact node ids below come from the Task 2 export — fill them
    # from comfyui/workflows/arkit_controlnet_spike.json before first run.
    patch = {
        "identity_image": str(identity_png),
        "edit_prompt_a": axis.edit_prompt_a,
        "edit_prompt_b": axis.edit_prompt_b,
        "scale": scale,
        "mix_b": axis.mix_b,
        "seed": SEED,
        "filename_prefix": out_name,
    }
    _apply_patch(wf, patch)  # small helper: walk wf, set widgets by node title
    body = json.dumps({"prompt": wf, "client_id": str(uuid.uuid4())}).encode()
    req = urllib.request.Request(f"{COMFY}/prompt", data=body,
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req).read()


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    identities = select_identities(5)
    rows = []
    for ident in identities:
        for axis_name in ["smile"]:                      # primary pass
            for scale in AXES[axis_name].scale_band:
                tag = f"{ident.stem}__{axis_name}__s{scale:.2f}"
                out_png = OUT_DIR / f"{tag}.png"
                if not out_png.exists():
                    submit(ident, axis_name, scale, tag)
                    _wait_for(out_png, timeout=600)
                rows.append({
                    "identity": ident.stem, "axis": axis_name, "scale": scale,
                    "arcface_cos": arcface_cos(out_png, ident),
                    "bs_delta": bs_delta(out_png, ident,
                                         AXES[axis_name].target_channels),
                })
    df = pd.DataFrame(rows)
    df.to_parquet(OUT_DIR / "metrics.parquet")
    print(df.to_string(index=False))


if __name__ == "__main__":
    run()
```

- [ ] **Step 2: Fill in node ids and the two helpers**

Open `comfyui/workflows/arkit_controlnet_spike.json` from Task 2; implement
`_apply_patch` (set each workflow widget by node title) and `_wait_for` (poll
for the output PNG) against the actual node ids. These are mechanical and
depend on the exported JSON — do them with the file open.

- [ ] **Step 3: Dry-run identity selection**

Run: `python -c "from src.arkit_controlnet.run_spike import select_identities; print(select_identities(5))"`
Expected: 5 distinct FFHQ image paths printed, spanning race buckets.

- [ ] **Step 4: Commit**

```bash
git add src/arkit_controlnet/run_spike.py
git commit -m "feat(arkit-spike): runner — 5 identities x smile axis x scale sweep"
```

---

## Task 5: Run the spike and write the verdict

**Files:**
- Create: `docs/research/2026-05-16-arkit-controlnet-spike-verdict.md`
- Modify: `docs/research/_topics/arkit-controlnet.md`

- [ ] **Step 1: Run the smile pass**

Start ComfyUI, then run: `python -m src.arkit_controlnet.run_spike`
Expected: 5 × 4 = 20 output PNGs in `exp_output/arkit_controlnet_spike/`,
plus `metrics.parquet`. Resumable if interrupted.

- [ ] **Step 2: Build the result collage**

Assemble a grid PNG (identities as rows, scales as columns) with the identity
photo in the leftmost column for visual comparison. Save to
`exp_output/arkit_controlnet_spike/collage.png`.

- [ ] **Step 3: Check the success criteria**

Against the spec's gates: `arcface_cos ≥ 0.65` (identity holds), `bs_delta`
monotone increasing in scale (expression follows), no scale-collapse jump
(continuity). If smile passes, optionally run the pucker/surprise extended
pass by adding them to the `axis_name` loop.

- [ ] **Step 4: Write the verdict doc**

Record in `docs/research/2026-05-16-arkit-controlnet-spike-verdict.md`: the
metrics table, the collage, whether InfU and FluxSpace composed, the
identity/expression trade-off across scale, and a go / no-go on Path 2 (the
full ARKit-InfuseNet train). Add frontmatter (`status: live`,
`topic: arkit-controlnet`).

- [ ] **Step 5: Update the topic index and commit**

Update `docs/research/_topics/arkit-controlnet.md` — move the Path-1 spike from
"open question" to its result; adjust the current-belief summary.

```bash
git add docs/research/2026-05-16-arkit-controlnet-spike-verdict.md docs/research/_topics/arkit-controlnet.md exp_output/arkit_controlnet_spike/collage.png
git commit -m "docs(arkit-controlnet): spike verdict — InfU+FluxSpace composition result"
```

---

## Self-review

**Spec coverage:** Goal (Tasks 2–5), success criteria — identity/expression/
continuity (Task 5 Step 3); FluxSpace axis set (Task 1); ComfyUI workflow
(Task 2); eval harness (Task 3); spike runner (Task 4); verdict doc (Task 5).
All spec sections map to a task.

**Placeholder scan:** Task 4 Step 2 deliberately defers `_apply_patch` /
`_wait_for` node-id wiring because the node ids only exist after the Task 2
export — this is a real dependency, not a placeholder, and Step 2 is an
explicit task step with a concrete instruction.

**Type consistency:** `Axis` dataclass fields (`edit_prompt_a`, `edit_prompt_b`,
`mix_b`, `scale_band`, `target_channels`) are used identically in `axes.py`,
`run_spike.py`, and the tests. `bs_read` / `arcface_cos` / `bs_delta`
signatures match between `eval_spike.py` and `run_spike.py`. `ARKIT_BLENDSHAPE_NAMES`
(52 entries, `_neutral` first) is defined once in `eval_spike.py`.
