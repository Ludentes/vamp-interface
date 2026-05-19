# CFM Conditioning Render Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the FLAME-render side of the CFM training pair — an FFHQ sha-index, FLAME assets, a `flame_render` module that turns a row's 52 ARKit blendshapes into an aligned control image, and a MediaPipe pose cache.

**Architecture:** A one-time `image_sha256 → (shard, row)` index over the 190-shard FFHQ-70k parquet. A one-time FLAME asset extraction (`v_template`, faces) into a chumpy-free npz. A pure geometry/raster module `flame_render` (deform + render, no I/O). A resumable `build_pose_cache` batch job that re-runs MediaPipe over the FFHQ images to recover head pose. Renders happen live in the future CFM dataloader; only the pose is cached.

**Tech Stack:** Python 3.12, numpy, pandas/pyarrow, OpenCV (`cv2`), MediaPipe Tasks (FaceLandmarker), pytest. Runs under the miniconda python (`/home/newub/miniconda3/bin/python`), like the existing spike runners.

**Spec:** `docs/superpowers/specs/2026-05-18-cfm-render-cache-design.md`

**Conventions for every task below:**
- Run commands with `PYTHONPATH=src /home/newub/miniconda3/bin/python …`. `pyproject.toml` sets `pythonpath=["src"]` so pytest finds `src/`.
- Conventional commits. End every commit message with the `Co-Authored-By` trailer used elsewhere in this repo.
- Source code under `src/arkit_controlnet/`, tests under `tests/arkit_controlnet/`.

---

## Task 1: FFHQ sha-index

Builds `output/ffhq_index/ffhq_sha_index.parquet` — the join key from
`reverse_index.image_sha256` to image bytes in the FFHQ-70k parquet shards.

**Files:**
- Create: `src/arkit_controlnet/build_ffhq_index.py`
- Test: `tests/arkit_controlnet/test_ffhq_index.py`

**Background the engineer needs:**
- FFHQ-70k parquet: `/media/newub/Seagate Hub/arc_distill/ffhq_parquet/data/train-*.parquet`, 190 shards. Each shard has an `image` column; each cell is a HuggingFace image dict `{"bytes": <encoded image bytes>, "path": <str or None>}`.
- `output/reverse_index/reverse_index.parquet` has 70,000 `source == "ffhq"` rows keyed by `image_sha256`.
- The sha convention is **unknown**: the 2,725 PNGs in `output/ffhq_images/` are named `{sha}.png`, but it is not known whether `sha` hashes the parquet's stored bytes or a re-encoded PNG. Step 1 resolves this empirically.

- [ ] **Step 1: Resolve the sha convention (diagnostic, no test)**

Run this throwaway diagnostic and read its output:

```python
# scratch — do not commit
import hashlib, io, glob
import pandas as pd
from PIL import Image

ri = set(pd.read_parquet("output/reverse_index/reverse_index.parquet",
                          columns=["image_sha256"])["image_sha256"])
shard = sorted(glob.glob("/media/newub/Seagate Hub/arc_distill/ffhq_parquet/data/train-*.parquet"))[0]
df = pd.read_parquet(shard)
raw_hits = enc_hits = 0
for cell in df["image"].iloc[:200]:
    b = cell["bytes"]
    if hashlib.sha256(b).hexdigest() in ri:
        raw_hits += 1
    im = Image.open(io.BytesIO(b)).convert("RGB")
    buf = io.BytesIO(); im.save(buf, format="PNG")
    if hashlib.sha256(buf.getvalue()).hexdigest() in ri:
        enc_hits += 1
print(f"raw-bytes hits: {raw_hits}/200   re-encoded-PNG hits: {enc_hits}/200")
```

Expected: exactly one of the two counts is high (≈200), the other ≈0. Record which — call the winner the **canonical hash**. The implementation in Step 3 uses it. If *both* are ≈0, stop and escalate: the FFHQ parquet is a different image set than `reverse_index` was built from.

- [ ] **Step 2: Write the failing test**

```python
# tests/arkit_controlnet/test_ffhq_index.py
from pathlib import Path
import pandas as pd
from arkit_controlnet.build_ffhq_index import canonical_sha, SHARD_GLOB

def test_canonical_sha_matches_reverse_index():
    """The first shard's images hash to shas present in reverse_index."""
    import glob
    ri = set(pd.read_parquet("output/reverse_index/reverse_index.parquet",
                             columns=["image_sha256"])["image_sha256"])
    shard = sorted(glob.glob(SHARD_GLOB))[0]
    df = pd.read_parquet(shard)
    hits = sum(canonical_sha(cell["bytes"]) in ri for cell in df["image"].iloc[:100])
    assert hits >= 95, f"only {hits}/100 shard images matched reverse_index"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `PYTHONPATH=src /home/newub/miniconda3/bin/python -m pytest tests/arkit_controlnet/test_ffhq_index.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'arkit_controlnet.build_ffhq_index'`.

- [ ] **Step 4: Implement `build_ffhq_index.py`**

Use the canonical hash chosen in Step 1. The code below shows the **raw-bytes** variant; if Step 1 chose re-encoded PNG, replace `canonical_sha` with the re-encode form (decode with PIL, save as PNG to a `BytesIO`, hash that) — keep the function name and signature identical.

```python
"""Build the image_sha256 -> (shard_idx, row_idx) index over FFHQ-70k.

The FFHQ-70000 dataset lives as 190 HuggingFace parquet shards on the Seagate
drive; reverse_index keys rows on image_sha256 but the shards carry no sha.
This index is the join key. Resumable per shard. Run once:
    PYTHONPATH=src /home/newub/miniconda3/bin/python -m arkit_controlnet.build_ffhq_index
"""
import glob
import hashlib
from pathlib import Path

import pandas as pd

SHARD_GLOB = "/media/newub/Seagate Hub/arc_distill/ffhq_parquet/data/train-*.parquet"
OUT = Path("output/ffhq_index/ffhq_sha_index.parquet")


def canonical_sha(image_bytes: bytes) -> str:
    """sha256 of the parquet-stored image bytes (the project's sha convention)."""
    return hashlib.sha256(image_bytes).hexdigest()


def build() -> None:
    shards = sorted(glob.glob(SHARD_GLOB))
    if not shards:
        raise FileNotFoundError(
            f"no FFHQ shards at {SHARD_GLOB} — is the Seagate drive mounted?")
    OUT.parent.mkdir(parents=True, exist_ok=True)

    done_shards: set[int] = set()
    if OUT.exists():
        done_shards = set(pd.read_parquet(OUT, columns=["shard_idx"])["shard_idx"])

    rows = []
    for shard_idx, shard in enumerate(shards):
        if shard_idx in done_shards:
            continue
        df = pd.read_parquet(shard, columns=["image"])
        for row_idx, cell in enumerate(df["image"]):
            rows.append({
                "image_sha256": canonical_sha(cell["bytes"]),
                "shard_idx": shard_idx,
                "row_idx": row_idx,
            })
        print(f"shard {shard_idx}/{len(shards)} done")
    if not rows:
        print("index already complete")
        return
    new = pd.DataFrame(rows)
    if OUT.exists():
        new = pd.concat([pd.read_parquet(OUT), new], ignore_index=True)
    new.to_parquet(OUT)
    print(f"wrote {len(new)} rows to {OUT}")


if __name__ == "__main__":
    build()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `PYTHONPATH=src /home/newub/miniconda3/bin/python -m pytest tests/arkit_controlnet/test_ffhq_index.py -v`
Expected: PASS.

- [ ] **Step 6: Build the full index**

Run: `PYTHONPATH=src /home/newub/miniconda3/bin/python -m arkit_controlnet.build_ffhq_index`
Expected: 190 shard lines, final `wrote 70000 rows`. Then verify coverage:

```bash
PYTHONPATH=src /home/newub/miniconda3/bin/python -c "
import pandas as pd
idx = set(pd.read_parquet('output/ffhq_index/ffhq_sha_index.parquet')['image_sha256'])
ri  = pd.read_parquet('output/reverse_index/reverse_index.parquet', columns=['image_sha256','source'])
ff  = set(ri[ri['source']=='ffhq']['image_sha256'])
print('ffhq rows', len(ff), 'indexed', len(ff & idx))
"
```
Expected: `indexed` within a few of 70000.

- [ ] **Step 7: Commit**

```bash
git add src/arkit_controlnet/build_ffhq_index.py tests/arkit_controlnet/test_ffhq_index.py
git commit -m "feat(arkit-cn): FFHQ-70k sha-index for the CFM corpus"
```

(`output/` is not committed — it is generated data.)

---

## Task 2: FLAME asset prep

Extracts the FLAME neutral mesh and faces into a plain, chumpy-free npz.

**Files:**
- Create: `src/arkit_controlnet/prep_flame_assets.py`
- Test: `tests/arkit_controlnet/test_flame_assets.py`

**Background:**
- FLAME model: `/home/newub/w/LAM/model_zoo/human_parametric_models/flame_vhap/flame2023.pkl`. It pickles `chumpy` arrays; `chumpy` is not installed and references numpy aliases removed in modern numpy.
- ARKit basis: `/home/newub/w/LAM/model_zoo/human_parametric_models/flame_assets/flame_arkit_bs.npy`, shape `(52, 5023, 3)` float64 — already plain numpy.
- Output: `output/flame_assets/flame_base.npz` with `v_template` (5023, 3) float32 and `faces` (n_faces, 3) int32; plus a copy of the basis at `output/flame_assets/flame_arkit_bs.npy`.

- [ ] **Step 1: Write the failing test**

```python
# tests/arkit_controlnet/test_flame_assets.py
from pathlib import Path
import numpy as np

NPZ = Path("output/flame_assets/flame_base.npz")
BASIS = Path("output/flame_assets/flame_arkit_bs.npy")

def test_flame_base_shapes():
    d = np.load(NPZ)
    assert d["v_template"].shape == (5023, 3)
    assert d["v_template"].dtype == np.float32
    assert d["faces"].ndim == 2 and d["faces"].shape[1] == 3
    assert int(d["faces"].max()) < 5023            # face indices in range

def test_basis_copied():
    b = np.load(BASIS)
    assert b.shape == (52, 5023, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src /home/newub/miniconda3/bin/python -m pytest tests/arkit_controlnet/test_flame_assets.py -v`
Expected: FAIL — `FileNotFoundError` on the npz.

- [ ] **Step 3: Implement `prep_flame_assets.py`**

```python
"""Extract FLAME neutral mesh + faces into a chumpy-free npz.

flame2023.pkl stores chumpy arrays; this loads it behind a numpy-alias shim and
a minimal chumpy stub so the pickle resolves, then writes plain numpy. Run once:
    PYTHONPATH=src /home/newub/miniconda3/bin/python -m arkit_controlnet.prep_flame_assets
"""
import pickle
import shutil
import sys
import types
from pathlib import Path

import numpy as np

LAM = Path("/home/newub/w/LAM/model_zoo/human_parametric_models")
FLAME_PKL = LAM / "flame_vhap" / "flame2023.pkl"
BASIS_SRC = LAM / "flame_assets" / "flame_arkit_bs.npy"
OUT_DIR = Path("output/flame_assets")


def _install_chumpy_shim() -> None:
    """Make `import chumpy` resolve to a stub whose Ch objects unpickle as
    plain ndarrays. flame2023.pkl only needs the array values, not chumpy's
    autodiff graph."""
    for alias, real in [("bool", np.bool_), ("int", np.int_),
                        ("float", np.float64), ("object", np.object_),
                        ("str", np.str_), ("complex", np.complex128)]:
        if not hasattr(np, alias):
            setattr(np, alias, real)

    class Ch(np.ndarray):
        """A chumpy.Ch stand-in: unpickles, then behaves as its ndarray value."""
        def __setstate__(self, state):
            pass  # state is restored via __reduce__/array path; value is enough

    chumpy = types.ModuleType("chumpy")
    chumpy.Ch = Ch
    ch_mod = types.ModuleType("chumpy.ch")
    ch_mod.Ch = Ch
    chumpy.ch = ch_mod
    sys.modules.setdefault("chumpy", chumpy)
    sys.modules.setdefault("chumpy.ch", ch_mod)


def _to_array(x) -> np.ndarray:
    """chumpy arrays expose `.r` for their raw ndarray value; plain arrays don't."""
    return np.asarray(getattr(x, "r", x))


def build() -> None:
    if not FLAME_PKL.exists():
        raise FileNotFoundError(
            f"{FLAME_PKL} missing — is the FLAME model zoo in place?")
    _install_chumpy_shim()
    with open(FLAME_PKL, "rb") as f:
        model = pickle.load(f, encoding="latin1")

    v_template = _to_array(model["v_template"]).astype(np.float32).reshape(-1, 3)
    faces = np.asarray(model["f"]).astype(np.int32)
    assert v_template.shape == (5023, 3), v_template.shape
    assert faces.ndim == 2 and faces.shape[1] == 3, faces.shape

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_DIR / "flame_base.npz", v_template=v_template, faces=faces)
    shutil.copy(BASIS_SRC, OUT_DIR / "flame_arkit_bs.npy")
    print(f"wrote {OUT_DIR/'flame_base.npz'} — {len(v_template)} verts, "
          f"{len(faces)} faces; copied basis")


if __name__ == "__main__":
    build()
```

If `pickle.load` still raises despite the shim (chumpy state too entangled),
fall back: run the extraction once under LAM's own Python environment (LAM
loads this file in production), then `np.savez` the same arrays. Do not add
`chumpy` as a project dependency.

- [ ] **Step 4: Run the prep script, then the test**

Run: `PYTHONPATH=src /home/newub/miniconda3/bin/python -m arkit_controlnet.prep_flame_assets`
Expected: `wrote output/flame_assets/flame_base.npz — 5023 verts, 9976 faces; copied basis` (face count may differ slightly — any 2-manifold count near ~9976 is fine).

Run: `PYTHONPATH=src /home/newub/miniconda3/bin/python -m pytest tests/arkit_controlnet/test_flame_assets.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/arkit_controlnet/prep_flame_assets.py tests/arkit_controlnet/test_flame_assets.py
git commit -m "feat(arkit-cn): extract FLAME neutral mesh to a chumpy-free npz"
```

---

## Task 3: ARKit → FLAME basis channel mapping

The FLAME basis's 52 channels and MediaPipe's 52 `bs_*` columns are **not in the
same order**, and do not even cover the same set. This task builds and tests the
explicit permutation. Get this wrong and every render is geometrically corrupt.

**Files:**
- Create: `src/arkit_controlnet/flame_render.py` (mapping section only)
- Test: `tests/arkit_controlnet/test_flame_render.py` (mapping tests only)

**Background (load-bearing — verified facts):**
- `flame_arkit_bs.npy` axis 0 is the **52 ARKit blendshapes in alphabetical order**, per the `targetNames` list in LAM_WebRender's `skin.glb` (`docs/research/2026-05-13-arkit-flame-mapping-extracted.md`).
- MediaPipe's `ARKIT_BLENDSHAPE_NAMES` (`src/arkit_controlnet/eval_spike.py`) is 52 entries: `_neutral` plus 51 expression names. MediaPipe does **not** emit ARKit's `tongueOut`.
- So the basis's 52 = ARKit's standard 52 = MediaPipe's 51 non-`_neutral` names **plus** `tongueOut`. `_neutral` maps to nothing; `tongueOut` always gets coefficient 0 (MediaPipe never produces it).

- [ ] **Step 1: Write the failing test**

```python
# tests/arkit_controlnet/test_flame_render.py
import numpy as np
from arkit_controlnet.flame_render import (
    BASIS_CHANNEL_NAMES, mediapipe_to_basis_vector,
)
from arkit_controlnet.eval_spike import ARKIT_BLENDSHAPE_NAMES

def test_basis_channel_names_are_52_sorted_arkit():
    assert len(BASIS_CHANNEL_NAMES) == 52
    assert BASIS_CHANNEL_NAMES == sorted(BASIS_CHANNEL_NAMES)
    assert "tongueOut" in BASIS_CHANNEL_NAMES
    assert "_neutral" not in BASIS_CHANNEL_NAMES

def test_basis_covers_all_mediapipe_expression_names():
    mp_expr = [n for n in ARKIT_BLENDSHAPE_NAMES if n != "_neutral"]
    assert set(mp_expr).issubset(set(BASIS_CHANNEL_NAMES))

def test_mediapipe_to_basis_vector_places_jawopen():
    """A unit jawOpen in MediaPipe order lands on the basis's jawOpen channel."""
    mp = {n: 0.0 for n in ARKIT_BLENDSHAPE_NAMES}
    mp["jawOpen"] = 1.0
    vec = mediapipe_to_basis_vector(mp)
    assert vec.shape == (52,)
    j = BASIS_CHANNEL_NAMES.index("jawOpen")
    assert vec[j] == 1.0
    assert np.count_nonzero(vec) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src /home/newub/miniconda3/bin/python -m pytest tests/arkit_controlnet/test_flame_render.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'arkit_controlnet.flame_render'`.

- [ ] **Step 3: Create `flame_render.py` with the mapping section**

```python
"""FLAME expression render for the CFM conditioning channel.

Pure geometry + rasterization: deform the FLAME template by 52 ARKit
blendshapes, pose it, and flat-shade it to a control image. No photo I/O, no
MediaPipe. See docs/superpowers/specs/2026-05-18-cfm-render-cache-design.md.
"""
from pathlib import Path

import cv2
import numpy as np

from arkit_controlnet.eval_spike import ARKIT_BLENDSHAPE_NAMES

# The FLAME basis axis-0 order: ARKit's 52 blendshapes, alphabetical. MediaPipe
# emits `_neutral` + 51 expression names and never `tongueOut`; the basis is
# those 51 plus `tongueOut`, sorted.
_MP_EXPR = [n for n in ARKIT_BLENDSHAPE_NAMES if n != "_neutral"]
BASIS_CHANNEL_NAMES = sorted([*_MP_EXPR, "tongueOut"])
assert len(BASIS_CHANNEL_NAMES) == 52

# index in BASIS_CHANNEL_NAMES for each MediaPipe expression name
_MP_TO_BASIS = {n: BASIS_CHANNEL_NAMES.index(n) for n in _MP_EXPR}


def mediapipe_to_basis_vector(mp_blendshapes: dict[str, float]) -> np.ndarray:
    """Reorder a MediaPipe blendshape dict into a 52-d basis-channel vector.

    `_neutral` is dropped; `tongueOut` stays 0 (MediaPipe never emits it).
    """
    vec = np.zeros(52, dtype=np.float64)
    for name, basis_idx in _MP_TO_BASIS.items():
        vec[basis_idx] = float(mp_blendshapes.get(name, 0.0))
    return vec
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src /home/newub/miniconda3/bin/python -m pytest tests/arkit_controlnet/test_flame_render.py -v`
Expected: PASS (3 tests).

> **Caveat for the engineer:** `sorted()` here assumes the `skin.glb` ordering is a plain case-sensitive alphabetical sort of the standard ARKit names. Task 4's `test_deform_jawopen_drops_lower_lip` is the real check — if it fails, the alphabetical assumption is wrong and `BASIS_CHANNEL_NAMES` must be replaced with the literal `targetNames` list read out of `/home/newub/w/LAM/.../skin.glb` (a glTF file: parse its `meshes[*].primitives[*].extras.targetNames`). Do not proceed past Task 4 with a failing jawOpen test.

- [ ] **Step 5: Commit**

```bash
git add src/arkit_controlnet/flame_render.py tests/arkit_controlnet/test_flame_render.py
git commit -m "feat(arkit-cn): ARKit-MediaPipe to FLAME basis channel mapping"
```

---

## Task 4: `deform()` — blendshapes to FLAME vertices

**Files:**
- Modify: `src/arkit_controlnet/flame_render.py` (add asset loading + `deform`)
- Test: `tests/arkit_controlnet/test_flame_render.py` (add deform tests)

**Background:**
- FLAME region masks: `/home/newub/w/LAM/model_zoo/human_parametric_models/flame_vhap/FLAME_masks.pkl` — a dict of region-name → vertex-index arrays (keys include `lips`). Used by the test to check the right vertices moved.
- `deform` applies the basis: `verts = v_template + Σ_k coeff[k] · basis[k]`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/arkit_controlnet/test_flame_render.py
import pickle
from arkit_controlnet.flame_render import deform, load_flame_assets

def test_deform_neutral_is_template():
    a = load_flame_assets()
    out = deform(np.zeros(52))
    assert np.allclose(out, a.v_template, atol=1e-6)
    assert out.shape == (5023, 3)

def test_deform_jawopen_drops_lower_lip():
    """jawOpen must move the lower-lip vertices down (FLAME -Y is down)."""
    from arkit_controlnet.eval_spike import ARKIT_BLENDSHAPE_NAMES
    from arkit_controlnet.flame_render import mediapipe_to_basis_vector
    with open("/home/newub/w/LAM/model_zoo/human_parametric_models/"
              "flame_vhap/FLAME_masks.pkl", "rb") as f:
        masks = pickle.load(f, encoding="latin1")
    lips = np.asarray(masks["lips"]).ravel()
    mp = {n: 0.0 for n in ARKIT_BLENDSHAPE_NAMES}
    mp["jawOpen"] = 1.0
    template = load_flame_assets().v_template
    moved = deform(mediapipe_to_basis_vector(mp))
    dy = (moved[lips, 1] - template[lips, 1]).mean()
    assert abs(dy) > 1e-4, f"jawOpen barely moved the lips (dy={dy})"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src /home/newub/miniconda3/bin/python -m pytest tests/arkit_controlnet/test_flame_render.py -v`
Expected: FAIL — `load_flame_assets`/`deform` not defined.

- [ ] **Step 3: Add asset loading + `deform` to `flame_render.py`**

```python
# add to flame_render.py — imports `Path`, `np` already present
from dataclasses import dataclass

_ASSETS_DIR = Path("output/flame_assets")


@dataclass
class FlameAssets:
    v_template: np.ndarray   # (5023, 3) float32
    faces: np.ndarray        # (n_faces, 3) int32
    arkit_basis: np.ndarray  # (52, 5023, 3) float64


_assets: FlameAssets | None = None


def load_flame_assets() -> FlameAssets:
    """Load and module-cache the FLAME template, faces, and ARKit basis."""
    global _assets
    if _assets is None:
        npz = _ASSETS_DIR / "flame_base.npz"
        basis = _ASSETS_DIR / "flame_arkit_bs.npy"
        if not npz.exists() or not basis.exists():
            raise FileNotFoundError(
                f"{npz} / {basis} missing — run "
                "`python -m arkit_controlnet.prep_flame_assets` first")
        d = np.load(npz)
        _assets = FlameAssets(v_template=d["v_template"], faces=d["faces"],
                              arkit_basis=np.load(basis))
    return _assets


def deform(basis_coeffs: np.ndarray) -> np.ndarray:
    """FLAME vertices for a 52-d basis-channel coefficient vector.

    `basis_coeffs` must be ordered per BASIS_CHANNEL_NAMES (use
    `mediapipe_to_basis_vector`). Returns (5023, 3) float32.
    """
    coeffs = np.asarray(basis_coeffs, dtype=np.float64)
    if coeffs.shape != (52,):
        raise ValueError(f"expected 52 coeffs, got {coeffs.shape}")
    if not np.all(np.isfinite(coeffs)):
        raise ValueError("non-finite blendshape coefficients")
    a = load_flame_assets()
    disp = np.einsum("k,kij->ij", coeffs, a.arkit_basis)
    return (a.v_template + disp).astype(np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src /home/newub/miniconda3/bin/python -m pytest tests/arkit_controlnet/test_flame_render.py -v`
Expected: PASS (5 tests). If `test_deform_jawopen_drops_lower_lip` fails, see the Task 3 Step 4 caveat — fix `BASIS_CHANNEL_NAMES` from `skin.glb` before continuing.

- [ ] **Step 5: Commit**

```bash
git add src/arkit_controlnet/flame_render.py tests/arkit_controlnet/test_flame_render.py
git commit -m "feat(arkit-cn): flame_render.deform — blendshapes to FLAME mesh"
```

---

## Task 5: `render()` — pose, project, rasterize

Turns deformed vertices + a head pose + a target face bbox into a control image.

**Files:**
- Modify: `src/arkit_controlnet/flame_render.py` (add `render`)
- Test: `tests/arkit_controlnet/test_flame_render.py` (add render tests)

**Background:**
- Reuse the painter's-algorithm flat-shade already proven in `landmark_control.render_depth_map`: sort faces by mean camera-z, far first, `cv2.fillConvexPoly` with `LINE_8` (never `LINE_AA` — anti-aliasing corrupts a flat-shaded raster).
- `rotation` is the 3×3 block of MediaPipe's `facial_transformation_matrix`. Only rotation is reused; MediaPipe's translation/scale live in a different metric space. In-plane placement and scale come from fitting the projected mesh's bbox onto the photo's face `bbox`.
- `modality="normals"`: shade each face by its unit normal mapped to RGB, `rgb = (normal * 0.5 + 0.5) * 255`. This is the recommended default; `depth` and `flat` are out of scope for this task (add later if the CFM run wants them).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/arkit_controlnet/test_flame_render.py
from arkit_controlnet.flame_render import render

def test_render_neutral_fills_bbox():
    verts = deform(np.zeros(52))
    R = np.eye(3)
    bbox = (0.5, 0.45, 0.4, 0.5)   # cx, cy, w, h — normalized
    img = render(verts, R, bbox, modality="normals", H=512, W=512)
    assert img.shape == (512, 512, 3) and img.dtype == np.uint8
    # the face occupies roughly the requested bbox region, not the whole frame
    nonblack = (img.sum(axis=2) > 10)
    ys, xs = np.where(nonblack)
    assert nonblack.mean() > 0.02, "render is nearly empty"
    cx = xs.mean() / 512
    assert abs(cx - 0.5) < 0.12, f"face not centred at bbox cx (got {cx:.3f})"

def test_render_degenerate_bbox_raises():
    verts = deform(np.zeros(52))
    try:
        render(verts, np.eye(3), (0.5, 0.5, 0.0, 0.5), modality="normals",
               H=256, W=256)
        assert False, "expected ValueError on zero-width bbox"
    except ValueError:
        pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src /home/newub/miniconda3/bin/python -m pytest tests/arkit_controlnet/test_flame_render.py -v`
Expected: FAIL — `render` not defined.

- [ ] **Step 3: Add `render` to `flame_render.py`**

```python
# add to flame_render.py — `cv2` already imported

def _face_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Unit normal per face from its three vertices."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.clip(norm, 1e-12, None)


def render(verts: np.ndarray, rotation: np.ndarray,
           bbox: tuple[float, float, float, float],
           modality: str = "normals", H: int = 512, W: int = 512) -> np.ndarray:
    """Rasterize posed FLAME geometry into an (H, W, 3) uint8 control image.

    verts:    (5023, 3) FLAME-space vertices (from `deform`).
    rotation: (3, 3) head rotation (MediaPipe transformation-matrix block).
    bbox:     (cx, cy, w, h) target face box, normalized to the [0,1] image.
    modality: "normals" — per-face normal mapped to RGB.
    """
    if modality != "normals":
        raise ValueError(f"unsupported modality {modality!r}")
    cx, cy, bw, bh = bbox
    if bw <= 0 or bh <= 0:
        raise ValueError(f"degenerate bbox {bbox}")

    a = load_flame_assets()
    faces = a.faces
    vr = verts @ np.asarray(rotation, dtype=np.float64).T   # rotate into camera

    # orthographic projection: x,y to pixels; fit mesh xy-extent into the bbox
    xy = vr[:, :2].copy()
    xy[:, 1] *= -1.0                       # FLAME +Y up -> image +Y down
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    extent = np.maximum(hi - lo, 1e-9)
    scale = min(bw * W / extent[0], bh * H / extent[1])    # isotropic, fits box
    px = (xy - (lo + hi) / 2) * scale
    px[:, 0] += cx * W
    px[:, 1] += cy * H
    pts = px.astype(np.int32)

    normals = _face_normals(vr, faces)
    shade = ((normals * 0.5 + 0.5) * 255).astype(np.uint8)   # (n_faces, 3) RGB
    order = np.argsort(vr[faces, 2].mean(axis=1))            # near (small z) last

    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    for i in order:
        tri = faces[i]
        col = (int(shade[i, 2]), int(shade[i, 1]), int(shade[i, 0]))  # cv2 BGR
        cv2.fillConvexPoly(canvas, pts[tri], col, lineType=cv2.LINE_8)
    return canvas
```

> The painter's z-order assumes MediaPipe's camera convention (smaller z = nearer). If Task 7's overlay shows the mesh inside-out, flip `order` to `np.argsort(-…)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src /home/newub/miniconda3/bin/python -m pytest tests/arkit_controlnet/test_flame_render.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add src/arkit_controlnet/flame_render.py tests/arkit_controlnet/test_flame_render.py
git commit -m "feat(arkit-cn): flame_render.render — posed normal-map raster"
```

---

## Task 6: `build_pose_cache.py`

Re-runs MediaPipe over the FFHQ images to recover per-image head pose.

**Files:**
- Create: `src/arkit_controlnet/build_pose_cache.py`
- Test: `tests/arkit_controlnet/test_pose_cache.py`

**Background:**
- `eval_spike._get_landmarker()` builds a FaceLandmarker **without**
  `output_facial_transformation_matrixes` — this task needs its own options
  with that flag set.
- MediaPipe model: `models/mediapipe/face_landmarker.task` (the path
  `eval_spike` uses).
- Images come from the FFHQ shards via `ffhq_sha_index.parquet` (Task 1).
  Decode bytes with PIL → RGB numpy array.
- A `FaceLandmarkerResult` exposes `facial_transformation_matrixes` (list of
  4×4 arrays) and `face_landmarks`. No detection → both empty.

- [ ] **Step 1: Write the failing test**

```python
# tests/arkit_controlnet/test_pose_cache.py
import numpy as np
from arkit_controlnet.build_pose_cache import pose_from_image

def test_pose_from_image_on_a_known_face():
    """A materialized FFHQ portrait yields a rotation and a plausible bbox."""
    import glob
    from PIL import Image
    png = sorted(glob.glob("output/ffhq_images/*.png"))[0]
    arr = np.asarray(Image.open(png).convert("RGB"))
    rot, bbox, detected = pose_from_image(arr)
    assert detected is True
    assert rot.shape == (3, 3)
    assert np.isclose(np.linalg.det(rot), 1.0, atol=0.1)   # proper rotation
    cx, cy, bw, bh = bbox
    assert 0.0 < bw < 1.0 and 0.0 < bh < 1.0
    assert 0.0 < cx < 1.0 and 0.0 < cy < 1.0

def test_pose_from_image_no_face():
    rot, bbox, detected = pose_from_image(np.zeros((256, 256, 3), np.uint8))
    assert detected is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src /home/newub/miniconda3/bin/python -m pytest tests/arkit_controlnet/test_pose_cache.py -v`
Expected: FAIL — `build_pose_cache` not defined.

- [ ] **Step 3: Implement `build_pose_cache.py`**

```python
"""Re-run MediaPipe over FFHQ to cache each image's head pose.

The CFM conditioning render must be posed to match the photo; reverse_index
stores blendshapes but no head pose. This caches the rotation + face bbox per
image_sha256. Resumable per shard. Run under the miniconda python:
    PYTHONPATH=src /home/newub/miniconda3/bin/python -m arkit_controlnet.build_pose_cache
"""
import glob
import io
from pathlib import Path

import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image

from arkit_controlnet.build_ffhq_index import SHARD_GLOB

_MP_MODEL = Path("models/mediapipe/face_landmarker.task")
INDEX = Path("output/ffhq_index/ffhq_sha_index.parquet")
OUT = Path("output/flame_pose_cache/pose_cache.parquet")

_landmarker = None


def _get_landmarker():
    global _landmarker
    if _landmarker is None:
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(_MP_MODEL)),
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        _landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
    return _landmarker


def pose_from_image(rgb: np.ndarray):
    """(rotation 3x3, bbox (cx,cy,w,h) normalized, detected bool) for an RGB array.

    Non-detection returns (eye(3), (0,0,0,0), False)."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=np.ascontiguousarray(rgb))
    res = _get_landmarker().detect(mp_image)
    if not res.facial_transformation_matrixes or not res.face_landmarks:
        return np.eye(3), (0.0, 0.0, 0.0, 0.0), False
    mat = np.asarray(res.facial_transformation_matrixes[0], dtype=np.float64)
    rot = mat[:3, :3]
    lm = np.array([[p.x, p.y] for p in res.face_landmarks[0]], dtype=np.float64)
    lo, hi = lm.min(axis=0), lm.max(axis=0)
    cx, cy = (lo + hi) / 2
    bw, bh = hi - lo
    return rot, (float(cx), float(cy), float(bw), float(bh)), True


def build() -> None:
    if not INDEX.exists():
        raise FileNotFoundError(
            f"{INDEX} missing — run `python -m arkit_controlnet.build_ffhq_index`")
    idx = pd.read_parquet(INDEX)
    shards = sorted(glob.glob(SHARD_GLOB))
    OUT.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    if OUT.exists():
        done = set(pd.read_parquet(OUT, columns=["image_sha256"])["image_sha256"])

    rows = []
    for shard_idx, group in idx.groupby("shard_idx"):
        todo = group[~group["image_sha256"].isin(done)]
        if todo.empty:
            continue
        df = pd.read_parquet(shards[shard_idx], columns=["image"])
        for _, r in todo.iterrows():
            cell = df["image"].iloc[r["row_idx"]]
            rgb = np.asarray(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))
            rot, bbox, detected = pose_from_image(rgb)
            rows.append({
                "image_sha256": r["image_sha256"],
                "rotation": rot.ravel().tolist(),
                "bbox_cx": bbox[0], "bbox_cy": bbox[1],
                "bbox_w": bbox[2], "bbox_h": bbox[3],
                "pose_detected": detected,
            })
        print(f"shard {shard_idx} done ({len(todo)} images)")
    if not rows:
        print("pose cache already complete")
        return
    new = pd.DataFrame(rows)
    if OUT.exists():
        new = pd.concat([pd.read_parquet(OUT), new], ignore_index=True)
    new.to_parquet(OUT)
    print(f"wrote {len(new)} rows; detected "
          f"{int(new['pose_detected'].sum())}/{len(new)}")


if __name__ == "__main__":
    build()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src /home/newub/miniconda3/bin/python -m pytest tests/arkit_controlnet/test_pose_cache.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Build the full pose cache**

Run: `PYTHONPATH=src /home/newub/miniconda3/bin/python -m arkit_controlnet.build_pose_cache`
Expected: 190 shard lines, final `wrote 70000 rows; detected NNNNN/70000`.
Then sanity-check detection against `reverse_index.bs_detected`:

```bash
PYTHONPATH=src /home/newub/miniconda3/bin/python -c "
import pandas as pd
pc = pd.read_parquet('output/flame_pose_cache/pose_cache.parquet')
ri = pd.read_parquet('output/reverse_index/reverse_index.parquet',
                     columns=['image_sha256','bs_detected'])
m = pc.merge(ri, on='image_sha256')
agree = (m['pose_detected'] == m['bs_detected']).mean()
print(f'pose/bs detection agreement: {agree:.3f}')
"
```
Expected: agreement > 0.97 (the two MediaPipe passes should largely concur).

- [ ] **Step 6: Commit**

```bash
git add src/arkit_controlnet/build_pose_cache.py tests/arkit_controlnet/test_pose_cache.py
git commit -m "feat(arkit-cn): MediaPipe head-pose cache over FFHQ-70k"
```

---

## Task 7: Manual verification collage

Per the standing post-task rule, produce an eyeball check that the render is
posed and expressive correctly.

**Files:**
- Create: `src/arkit_controlnet/verify_flame_render.py`
- Create: `docs/research/2026-05-18-cfm-render-cache-verification.md`

- [ ] **Step 1: Implement the verification script**

```python
"""Eyeball check: render FFHQ rows' FLAME meshes and overlay on their photos.

Picks 8 FFHQ rows spanning low-to-high expression energy, renders each row's
FLAME mesh from its blendshapes + cached pose, alpha-overlays it on the photo,
and writes a collage. Run:
    PYTHONPATH=src /home/newub/miniconda3/bin/python -m arkit_controlnet.verify_flame_render
"""
import io
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from arkit_controlnet.build_ffhq_index import SHARD_GLOB
from arkit_controlnet.eval_spike import ARKIT_BLENDSHAPE_NAMES
from arkit_controlnet.flame_render import deform, mediapipe_to_basis_vector, render

OUT_DIR = Path("exp_output/flame_render_check")
_EXPR_COLS = [f"bs_{n}" for n in ARKIT_BLENDSHAPE_NAMES if n != "_neutral"]


def main() -> None:
    import glob
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ri = pd.read_parquet("output/reverse_index/reverse_index.parquet",
                         columns=["image_sha256", "source", "bs_detected",
                                  *_EXPR_COLS])
    ri = ri[(ri["source"] == "ffhq") & (ri["bs_detected"])].copy()
    ri["energy"] = ri[_EXPR_COLS].abs().sum(axis=1)
    ri = ri.sort_values("energy")
    picks = pd.concat([ri.head(4), ri.tail(4)])      # 4 calm, 4 expressive

    idx = pd.read_parquet("output/ffhq_index/ffhq_sha_index.parquet")
    pc = pd.read_parquet("output/flame_pose_cache/pose_cache.parquet")
    shards = sorted(glob.glob(SHARD_GLOB))

    tiles = []
    for _, row in picks.iterrows():
        sha = row["image_sha256"]
        loc = idx[idx["image_sha256"] == sha].iloc[0]
        cell = pd.read_parquet(shards[loc["shard_idx"]],
                               columns=["image"])["image"].iloc[loc["row_idx"]]
        photo = np.asarray(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))
        H, W = photo.shape[:2]

        pose = pc[pc["image_sha256"] == sha].iloc[0]
        rot = np.array(pose["rotation"], dtype=np.float64).reshape(3, 3)
        bbox = (pose["bbox_cx"], pose["bbox_cy"], pose["bbox_w"], pose["bbox_h"])

        mp_bs = {n: float(row.get(f"bs_{n}", 0.0)) for n in ARKIT_BLENDSHAPE_NAMES}
        verts = deform(mediapipe_to_basis_vector(mp_bs))
        ctrl = render(verts, rot, bbox, modality="normals", H=H, W=W)

        mask = (ctrl.sum(axis=2) > 10)[:, :, None]
        overlay = np.where(mask, (0.5 * photo + 0.5 * ctrl).astype(np.uint8),
                           photo)
        tiles.append(np.hstack([photo, ctrl, overlay]))

    collage = np.vstack(tiles)
    cv2.imwrite(str(OUT_DIR / "collage.png"),
                cv2.cvtColor(collage, cv2.COLOR_RGB2BGR))
    print(f"wrote {OUT_DIR/'collage.png'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `PYTHONPATH=src /home/newub/miniconda3/bin/python -m arkit_controlnet.verify_flame_render`
Expected: `wrote exp_output/flame_render_check/collage.png`. Open the collage:
each row is photo | normal-map render | overlay. Check — the mesh sits on the
face (pose), expressive rows show open mouth / raised brows (expression), the
overlay is not inside-out (else flip `order` in `render`, Task 5 Step 3 note).

- [ ] **Step 3: Write the verification doc**

Create `docs/research/2026-05-18-cfm-render-cache-verification.md` with
frontmatter (`status: live`, `topic: arkit-controlnet`) recording: the
detection rate from Task 6 Step 5, the pose/bs agreement number, and a
sentence per collage row pair on whether pose + expression read correctly.
Embed the collage. State plainly if any row fails.

- [ ] **Step 4: Commit**

```bash
git add src/arkit_controlnet/verify_flame_render.py docs/research/2026-05-18-cfm-render-cache-verification.md
git commit -m "feat(arkit-cn): FLAME render verification collage + notes"
```

- [ ] **Step 5: Update the topic index**

Add a section to `docs/research/_topics/arkit-controlnet.md` noting the CFM
render-cache is built (sha-index, pose cache, `flame_render`), with the
detection rate and a pointer to the verification doc and the design spec.
Commit with `docs(arkit-cn): topic index — CFM render-cache built`.

---

## Self-Review

**Spec coverage:** sha-index → Task 1; FLAME asset prep / chumpy → Task 2;
basis channel permutation → Task 3; `deform` → Task 4; `render` (pose, project,
bbox-fit, normals raster) → Task 5; `build_pose_cache` (MediaPipe transformation
matrix, resumable, `pose_detected`) → Task 6; error handling (no-detect, missing
asset, non-finite coeffs, degenerate bbox) covered in Tasks 4–6; testing +
manual verification → Task 7. The spec's "parquet-read, no PNG dump" is honored
— no task writes 70k PNGs. No spec gaps.

**Placeholder scan:** every code step has complete code; commands have expected
output. The one deliberate open branch — the `skin.glb` fallback if the
alphabetical ordering is wrong — is gated by a concrete failing test
(`test_deform_jawopen_drops_lower_lip`) with explicit instructions, not a TODO.

**Type consistency:** `mediapipe_to_basis_vector` → 52-d vector → `deform` →
(5023,3) → `render`; `BASIS_CHANNEL_NAMES`, `load_flame_assets`, `FlameAssets`,
`SHARD_GLOB`, `canonical_sha`, `pose_from_image` names are consistent across
tasks. `pose_from_image` returns `(rotation, bbox, detected)` and the test and
`build` both consume that shape.
