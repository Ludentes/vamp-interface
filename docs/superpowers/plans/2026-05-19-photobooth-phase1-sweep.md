# Photobooth Phase 1 Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and run the Phase 1 photobooth-pipeline sweep: 4 stratified source photos × 40 LHS configs over 8 axes on the Windows 3090 shard, producing `manifest.parquet` + `scores.parquet` for axis pruning.

**Architecture:** New module `scripts/photobooth_sweep/` with one file per responsibility (axes, color_match, features, refine, scorer, driver). Two parametric ComfyUI workflow JSONs. Driver talks to remote ComfyUI over HTTP (the cn_grid_sweep pattern), resumable per cell via append-only parquet. Swap stage held constant on `swap_core.swap_identity` with HyperSwap-1c.

**Tech Stack:** Python 3.12, ComfyUI 0.18 (Z-Image Turbo + Z-Image-Turbo-Fun-CN-Union), HyperSwap-1c ONNX (CPU), insightface buffalo_l, mediapipe FaceLandmarker, open_clip ViT-B/32, scipy.stats.qmc LHS, pyarrow parquet, pytest.

**Spec:** `docs/superpowers/specs/2026-05-19-photobooth-phase1-sweep-design.md`

**Runbook for the shard:** `docs/runbooks/comfyui-windows-shard.md`

---

## File map

| Path | Responsibility |
|---|---|
| `scripts/photobooth_sweep/__init__.py` | package marker |
| `scripts/photobooth_sweep/axes.py` | axis catalog + LHS sampler + manifest writer |
| `scripts/photobooth_sweep/color_match.py` | `lab_transfer`, `reinhard_transfer` (mask-scoped) |
| `scripts/photobooth_sweep/features.py` | per-photo features + canonical skin-mask builder |
| `scripts/photobooth_sweep/preprocess.py` | Canny presets + Depth preprocessor (runs on Linux, uploaded as control image) |
| `scripts/photobooth_sweep/refine.py` | masked img2img refine via remote ComfyUI |
| `scripts/photobooth_sweep/scorer.py` | id_cos / det_mode / lab_delta_ab / clip_style / face_frac |
| `scripts/photobooth_sweep/driver.py` | main loop, resume, error handling, parquet append |
| `scripts/photobooth_sweep/README.md` | how to run / resume |
| `comfyui/workflows/photobooth_zimage_cn.api.json` | parametric Z-Image Turbo + Fun-CN |
| `comfyui/workflows/photobooth_zimage_refine.api.json` | parametric masked-img2img refine |
| `tests/photobooth_sweep/test_axes.py` | LHS shape, encoding, manifest schema |
| `tests/photobooth_sweep/test_color_match.py` | Lab + Reinhard correctness on synthetic input |
| `tests/photobooth_sweep/test_features.py` | feature extraction on the 4 sample photos |
| `tests/photobooth_sweep/test_scorer.py` | scorer's lab_delta_ab + face_frac on a known swap |
| `tests/photobooth_sweep/test_preprocess.py` | Canny preset determinism + Depth shape |

Total new code estimate: ~1100 lines + 2 workflow JSONs.

---

## Task 1: Scaffold + axes catalog (TDD)

**Files:**
- Create: `scripts/photobooth_sweep/__init__.py`
- Create: `scripts/photobooth_sweep/axes.py`
- Create: `tests/photobooth_sweep/__init__.py`
- Create: `tests/photobooth_sweep/test_axes.py`

- [ ] **Step 1.1: Empty package markers**

```bash
mkdir -p scripts/photobooth_sweep tests/photobooth_sweep
touch scripts/photobooth_sweep/__init__.py tests/photobooth_sweep/__init__.py
```

- [ ] **Step 1.2: Write failing test for axis catalog shape**

`tests/photobooth_sweep/test_axes.py`:

```python
import pyarrow.parquet as pq
import pytest

from scripts.photobooth_sweep.axes import AXES, lhs_sample, write_manifest


def test_axes_catalog_has_eight_live_axes():
    names = [a.name for a in AXES]
    assert names == [
        "color_match", "face_pixel_budget", "cn_condition",
        "cn_strength", "cn_start_percent", "canny_preset",
        "refine_denoise", "demo_inject",
    ]


def test_lhs_sample_returns_n_configs():
    cfgs = lhs_sample(n_configs=40, seed=0)
    assert len(cfgs) == 40
    assert all(set(c) >= set(a.name for a in AXES) for c in cfgs)


def test_canny_preset_nulled_when_condition_is_depth():
    cfgs = lhs_sample(n_configs=200, seed=0)
    depth_only = [c for c in cfgs if c["cn_condition"] == "depth"]
    assert depth_only, "LHS should sample depth-only at n=200"
    assert all(c["canny_preset"] is None for c in depth_only)


def test_cn_strength_within_bounds():
    cfgs = lhs_sample(n_configs=200, seed=0)
    assert all(0.80 <= c["cn_strength"] <= 1.00 for c in cfgs)


def test_write_manifest_round_trips(tmp_path):
    cfgs = lhs_sample(n_configs=8, seed=0)
    out = tmp_path / "manifest.parquet"
    write_manifest(out, photo_ids=["id_00", "id_11"], configs=cfgs)
    tbl = pq.read_table(out)
    assert tbl.num_rows == 16  # 2 photos x 8 configs
    cols = set(tbl.column_names)
    assert {"photo_id", "cfg_id", "seed", "cn_strength"} <= cols
```

- [ ] **Step 1.3: Run tests, expect ImportError**

```bash
uv run pytest tests/photobooth_sweep/test_axes.py -v
```

Expected: FAIL — `ModuleNotFoundError: scripts.photobooth_sweep.axes`.

- [ ] **Step 1.4: Implement `axes.py`**

```python
"""Axis catalog + Latin-hypercube sampler for the photobooth Phase 1 sweep.

Each axis is a categorical or continuous knob in the [0,1]^k LHS cube; the
encoder maps a unit-cube coordinate to the real-world level. `canny_preset`
is conditionally NULL when `cn_condition == "depth"` -- the manifest records
this so analysis joins are clean.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import qmc


@dataclass(frozen=True)
class Axis:
    name: str
    encode: Callable[[float], object]   # [0,1] -> level


def _categorical(levels: Sequence[object]):
    def enc(u: float) -> object:
        idx = min(int(u * len(levels)), len(levels) - 1)
        return levels[idx]
    return enc


def _continuous(lo: float, hi: float):
    def enc(u: float) -> float:
        return round(lo + u * (hi - lo), 4)
    return enc


AXES: list[Axis] = [
    Axis("color_match",         _categorical(["off", "lab", "reinhard"])),
    Axis("face_pixel_budget",   _categorical(["tight_1024", "med_1024", "tight_768"])),
    Axis("cn_condition",        _categorical(["canny", "depth", "canny+depth"])),
    Axis("cn_strength",         _continuous(0.80, 1.00)),
    Axis("cn_start_percent",    _categorical([0.00, 0.10, 0.20])),
    Axis("canny_preset",        _categorical(["soft", "default", "aggressive"])),
    Axis("refine_denoise",      _categorical(["off", 0.30, 0.50])),
    Axis("demo_inject",         _categorical([False, True])),
]


def lhs_sample(n_configs: int, seed: int = 0) -> list[dict]:
    sampler = qmc.LatinHypercube(d=len(AXES), optimization="random-cd",
                                 seed=seed)
    cube = sampler.random(n=n_configs)
    cfgs = []
    for i, row in enumerate(cube):
        cfg = {ax.name: ax.encode(u) for ax, u in zip(AXES, row)}
        if cfg["cn_condition"] == "depth":
            cfg["canny_preset"] = None
        cfg["cfg_id"] = f"cfg_{i:03d}"
        cfgs.append(cfg)
    return cfgs


def write_manifest(out_path: Path | str,
                   photo_ids: Sequence[str],
                   configs: Sequence[dict]) -> None:
    rows = []
    for photo_id in photo_ids:
        photo_idx = int(photo_id.split("_")[-1])
        for cfg in configs:
            row = {"photo_id": photo_id,
                   "cfg_id": cfg["cfg_id"],
                   "seed": 91_000_000 + photo_idx,
                   **{k: v for k, v in cfg.items() if k != "cfg_id"}}
            rows.append(row)
    tbl = pa.Table.from_pylist(rows)
    pq.write_table(tbl, str(out_path))
```

- [ ] **Step 1.5: Run tests, expect pass**

```bash
uv run pytest tests/photobooth_sweep/test_axes.py -v
```

Expected: 5 passed.

- [ ] **Step 1.6: Commit**

```bash
git add scripts/photobooth_sweep/__init__.py scripts/photobooth_sweep/axes.py \
        tests/photobooth_sweep/__init__.py tests/photobooth_sweep/test_axes.py
git commit -m "feat(photobooth): axis catalog + LHS sampler + manifest writer"
```

---

## Task 2: Skin mask + per-photo features (TDD)

**Files:**
- Create: `scripts/photobooth_sweep/features.py`
- Create: `tests/photobooth_sweep/test_features.py`

Existing `face_region.py` and `swap_core.py` already build face masks for other paths — *do not* reuse them blindly. The Phase 1 spec defines a *canonical* skin-mask (FaceLandmarker oval, 5 px dilation, Lab gate `L*` ∈ [25, 95], `a*` > 5, `b*` > 8) — implement it here once and import from `color_match.py` and `scorer.py`.

- [ ] **Step 2.1: Write failing tests**

`tests/photobooth_sweep/test_features.py`:

```python
import json
import cv2
import numpy as np
import pytest

from scripts.photobooth_sweep.features import (
    build_skin_mask, extract_source_features, PHASE1_PHOTOS,
)


@pytest.fixture
def sample_face_bgr():
    # one of the calibration photos -- known to detect cleanly
    return cv2.imread("data/importer/identities/id_01.png")


def test_phase1_photos_constant():
    assert PHASE1_PHOTOS == ["id_00", "id_11", "id_16", "id_01"]


def test_skin_mask_excludes_lips_and_eyes(sample_face_bgr):
    mask = build_skin_mask(sample_face_bgr)
    h, w = sample_face_bgr.shape[:2]
    assert mask.shape == (h, w)
    assert mask.dtype == np.uint8
    # mask should be sparse-positive: between 5% and 35% of the image
    coverage = float(mask.sum()) / (mask.size * 255)
    assert 0.05 <= coverage <= 0.35, f"coverage={coverage:.3f}"


def test_extract_source_features_keys(sample_face_bgr, tmp_path):
    src_path = tmp_path / "id_01.png"
    cv2.imwrite(str(src_path), sample_face_bgr)
    feats = extract_source_features(src_path)
    expected = {"skin_lab_L", "skin_lab_a", "skin_lab_b",
                "hair_density", "has_glasses",
                "race", "gender", "age_bin",
                "source_face_resolution"}
    assert expected <= set(feats)
    assert 0 <= feats["skin_lab_L"] <= 100
```

- [ ] **Step 2.2: Run, expect failure**

```bash
uv run pytest tests/photobooth_sweep/test_features.py -v
```

Expected: FAIL — module not found.

- [ ] **Step 2.3: Implement `features.py`**

```python
"""Per-photo features + canonical skin mask for the photobooth sweep.

The skin mask is shared between color_match and scorer: any change here
applies to both. PHASE1_PHOTOS is the fixed Phase-1 stratum from the spec.
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

PHASE1_PHOTOS = ["id_00", "id_11", "id_16", "id_01"]

IDENTITIES_DIR = Path("data/importer/identities")
_LANDMARKER_TASK = Path("scripts/face_landmarker.task")

# mediapipe face-oval indices (the contour around the face, ~36 points)
_FACE_OVAL_IDX = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
    365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
    132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]


def _landmarker():
    base = mp.tasks.BaseOptions(model_asset_path=str(_LANDMARKER_TASK))
    opts = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base, num_faces=1)
    return mp.tasks.vision.FaceLandmarker.create_from_options(opts)


def build_skin_mask(bgr: np.ndarray) -> np.ndarray:
    """Canonical skin mask: face oval ∩ Lab skin gate.

    Returns uint8 0/255 of same H,W as input. Empty (all zeros) if no face.
    """
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    with _landmarker() as lm:
        res = lm.detect(mp_img)
    if not res.face_landmarks:
        return np.zeros((h, w), np.uint8)
    pts = res.face_landmarks[0]
    poly = np.array([[int(pts[i].x * w), int(pts[i].y * h)]
                     for i in _FACE_OVAL_IDX], np.int32)
    oval = np.zeros((h, w), np.uint8)
    cv2.fillPoly(oval, [poly], 255)
    oval = cv2.dilate(oval, np.ones((5, 5), np.uint8), iterations=1)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.int32)
    L, a, b = lab[..., 0], lab[..., 1] - 128, lab[..., 2] - 128
    # convert L* (0..255 OpenCV scale) to 0..100
    L_real = L * (100.0 / 255.0)
    skin_gate = ((L_real >= 25) & (L_real <= 95) &
                 (a >= 5) & (b >= 8)).astype(np.uint8) * 255

    return cv2.bitwise_and(oval, skin_gate)


def _read_manifest_row(idn: str) -> dict:
    with (IDENTITIES_DIR / "manifest.csv").open() as f:
        for row in csv.DictReader(f):
            if row["filename"] == f"{idn}.png":
                return row
    raise KeyError(idn)


def extract_source_features(photo_path: Path | str) -> dict:
    """Lab skin stats + hair density + glasses heuristic + manifest tags."""
    photo_path = Path(photo_path)
    bgr = cv2.imread(str(photo_path))
    if bgr is None:
        raise FileNotFoundError(photo_path)
    mask = build_skin_mask(bgr)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    Lp = lab[..., 0] * (100.0 / 255.0)
    ap = lab[..., 1].astype(np.int32) - 128
    bp = lab[..., 2].astype(np.int32) - 128

    if mask.any():
        m = mask > 0
        skin_L = float(Lp[m].mean())
        skin_a = float(ap[m].mean())
        skin_b = float(bp[m].mean())
    else:
        skin_L = skin_a = skin_b = float("nan")

    # Face bbox via mediapipe oval extent
    if mask.any():
        ys, xs = np.where(mask > 0)
        bb = (int(xs.min()), int(ys.min()),
              int(xs.max()), int(ys.max()))
        face_area = (bb[2] - bb[0]) * (bb[3] - bb[1])
        # hair_density proxy: 1 - (skin_pixels / face_bbox_area)
        hair_density = 1.0 - float(mask[bb[1]:bb[3], bb[0]:bb[2]].astype(bool)
                                   .sum()) / max(face_area, 1)
        face_res = face_area
    else:
        hair_density = float("nan")
        face_res = 0

    # has_glasses heuristic: defer to insightface attribute if available,
    # else False. Cheap and conservative.
    has_glasses = False

    idn = photo_path.stem
    manifest = _read_manifest_row(idn)

    return {
        "skin_lab_L": skin_L, "skin_lab_a": skin_a, "skin_lab_b": skin_b,
        "hair_density": hair_density, "has_glasses": has_glasses,
        "race": manifest["race"], "gender": manifest["gender"],
        "age_bin": manifest["age_bin"],
        "source_face_resolution": face_res,
    }


def extract_phase1_features(out_path: Path | str) -> dict:
    """Run extract_source_features on the four Phase-1 photos; persist JSON."""
    import json
    feats = {idn: extract_source_features(IDENTITIES_DIR / f"{idn}.png")
             for idn in PHASE1_PHOTOS}
    Path(out_path).write_text(json.dumps(feats, indent=2))
    return feats
```

- [ ] **Step 2.4: Run, expect pass**

```bash
uv run pytest tests/photobooth_sweep/test_features.py -v
```

Expected: 3 passed.

- [ ] **Step 2.5: Commit**

```bash
git add scripts/photobooth_sweep/features.py tests/photobooth_sweep/test_features.py
git commit -m "feat(photobooth): canonical skin mask + per-photo feature extraction"
```

---

## Task 3: Color-match transfers (TDD)

**Files:**
- Create: `scripts/photobooth_sweep/color_match.py`
- Create: `tests/photobooth_sweep/test_color_match.py`

- [ ] **Step 3.1: Failing tests**

`tests/photobooth_sweep/test_color_match.py`:

```python
import cv2
import numpy as np
import pytest

from scripts.photobooth_sweep.color_match import (
    apply, lab_transfer, reinhard_transfer,
)


def _solid(bgr, h=64, w=64):
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = bgr
    return img


@pytest.fixture
def source():
    return _solid((90, 130, 180))  # warm tan


@pytest.fixture
def swap():
    return _solid((140, 150, 160))  # washed-out


@pytest.fixture
def full_mask(swap):
    return np.full(swap.shape[:2], 255, np.uint8)


def test_lab_transfer_moves_swap_toward_source(source, swap, full_mask):
    out = lab_transfer(swap, source, full_mask)
    # closer to source than original swap was
    d_orig = np.linalg.norm(swap.mean(axis=(0, 1)) - source.mean(axis=(0, 1)))
    d_new = np.linalg.norm(out.mean(axis=(0, 1)) - source.mean(axis=(0, 1)))
    assert d_new < d_orig


def test_reinhard_transfer_moves_swap_toward_source(source, swap, full_mask):
    out = reinhard_transfer(swap, source, full_mask)
    d_orig = np.linalg.norm(swap.mean(axis=(0, 1)) - source.mean(axis=(0, 1)))
    d_new = np.linalg.norm(out.mean(axis=(0, 1)) - source.mean(axis=(0, 1)))
    assert d_new < d_orig


def test_zero_mask_returns_input_unchanged(source, swap):
    out = lab_transfer(swap, source, np.zeros(swap.shape[:2], np.uint8))
    assert np.array_equal(out, swap)


def test_apply_off_returns_input_unchanged(source, swap, full_mask):
    out = apply("off", swap, source, full_mask)
    assert np.array_equal(out, swap)


def test_apply_dispatches_by_name(source, swap, full_mask):
    out_lab = apply("lab", swap, source, full_mask)
    out_rh = apply("reinhard", swap, source, full_mask)
    assert not np.array_equal(out_lab, swap)
    assert not np.array_equal(out_rh, swap)
```

- [ ] **Step 3.2: Run, expect failure**

```bash
uv run pytest tests/photobooth_sweep/test_color_match.py -v
```

Expected: FAIL — module not found.

- [ ] **Step 3.3: Implement `color_match.py`**

```python
"""Mask-scoped post-swap color transfers.

Both lab_transfer and reinhard_transfer match mean+std of channels between
swap and source -- only inside the mask -- and leave masked-out pixels
untouched. lab_transfer preserves the L (luminance) channel; reinhard
moves all three.
"""
from __future__ import annotations

import cv2
import numpy as np


def _mask_stats(img_lab: np.ndarray, mask: np.ndarray, ch: int
                ) -> tuple[float, float]:
    pixels = img_lab[..., ch][mask > 0].astype(np.float32)
    if pixels.size == 0:
        return 0.0, 1.0
    return float(pixels.mean()), float(pixels.std() + 1e-6)


def _transfer(swap_bgr: np.ndarray,
              source_bgr: np.ndarray,
              mask: np.ndarray,
              channels: tuple[int, ...]) -> np.ndarray:
    if not mask.any():
        return swap_bgr.copy()
    sw_lab = cv2.cvtColor(swap_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    sr_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    out = sw_lab.copy()
    for c in channels:
        sw_mu, sw_sd = _mask_stats(sw_lab, mask, c)
        sr_mu, sr_sd = _mask_stats(sr_lab, mask, c)
        out[..., c] = (sw_lab[..., c] - sw_mu) * (sr_sd / sw_sd) + sr_mu
    out = np.clip(out, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(out, cv2.COLOR_LAB2BGR)
    m3 = (mask > 0)[..., None]
    return np.where(m3, bgr, swap_bgr)


def lab_transfer(swap_bgr, source_bgr, mask):
    """Match a/b mean+std; preserve L."""
    return _transfer(swap_bgr, source_bgr, mask, channels=(1, 2))


def reinhard_transfer(swap_bgr, source_bgr, mask):
    """Match L,a,b mean+std (classical Reinhard)."""
    return _transfer(swap_bgr, source_bgr, mask, channels=(0, 1, 2))


def apply(mode: str, swap_bgr, source_bgr, mask):
    if mode == "off":
        return swap_bgr.copy()
    if mode == "lab":
        return lab_transfer(swap_bgr, source_bgr, mask)
    if mode == "reinhard":
        return reinhard_transfer(swap_bgr, source_bgr, mask)
    raise ValueError(f"unknown color_match mode: {mode}")
```

- [ ] **Step 3.4: Tests pass**

```bash
uv run pytest tests/photobooth_sweep/test_color_match.py -v
```

Expected: 5 passed.

- [ ] **Step 3.5: Commit**

```bash
git add scripts/photobooth_sweep/color_match.py tests/photobooth_sweep/test_color_match.py
git commit -m "feat(photobooth): Lab + Reinhard mask-scoped color transfers"
```

---

## Task 4: Control-image preprocessor (Canny presets + Depth)

**Files:**
- Create: `scripts/photobooth_sweep/preprocess.py`
- Create: `tests/photobooth_sweep/test_preprocess.py`

The CN input is built on the Linux box (where DepthAnything-V2 is already cached locally) and uploaded as one PNG to the shard. `cn_condition="canny+depth"` is composited into a 2-channel image (R=Canny edges, G=Depth, B=0); the photobooth_zimage_cn workflow consumes this as a single image — verify on the shard during smoke test (Task 7) that the Fun-CN-Union node accepts the composite. Fallback if it doesn't: switch `canny+depth` to use Canny only and log the limitation in the open-questions doc.

- [ ] **Step 4.1: Failing tests**

`tests/photobooth_sweep/test_preprocess.py`:

```python
import cv2
import numpy as np
import pytest

from scripts.photobooth_sweep.preprocess import (
    CANNY_PRESETS, build_control_image,
)


@pytest.fixture
def src():
    return cv2.imread("data/importer/identities/id_01.png")


def test_canny_presets_are_three():
    assert set(CANNY_PRESETS) == {"soft", "default", "aggressive"}


def test_canny_preset_determinism(src):
    img1 = build_control_image(src, condition="canny", canny_preset="default",
                               face_pixel_budget="med_1024")
    img2 = build_control_image(src, condition="canny", canny_preset="default",
                               face_pixel_budget="med_1024")
    assert np.array_equal(img1, img2)


def test_canny_preset_soft_yields_fewer_edges_than_aggressive(src):
    soft = build_control_image(src, condition="canny", canny_preset="soft",
                               face_pixel_budget="med_1024")
    aggr = build_control_image(src, condition="canny", canny_preset="aggressive",
                               face_pixel_budget="med_1024")
    # soft preset blurs and uses lower thresholds -> fewer hard edges
    n_soft = int((soft[..., 0] > 0).sum())
    n_aggr = int((aggr[..., 0] > 0).sum())
    assert n_soft < n_aggr * 1.5  # not strict <, just not dramatically more


def test_face_pixel_budget_changes_output_size(src):
    img_1024 = build_control_image(src, condition="canny",
                                   canny_preset="default",
                                   face_pixel_budget="med_1024")
    img_768 = build_control_image(src, condition="canny",
                                  canny_preset="default",
                                  face_pixel_budget="tight_768")
    assert img_1024.shape[:2] != img_768.shape[:2]
    assert img_768.shape[0] <= 1024 and img_768.shape[1] <= 1024


def test_depth_path_produces_grayscale_like(src):
    img = build_control_image(src, condition="depth", canny_preset=None,
                              face_pixel_budget="med_1024")
    # depth path writes to G channel; R should be empty
    assert img[..., 0].max() == 0
    assert img[..., 1].max() > 0


def test_canny_plus_depth_composite(src):
    img = build_control_image(src, condition="canny+depth",
                              canny_preset="default",
                              face_pixel_budget="med_1024")
    assert img[..., 0].max() > 0  # canny in R
    assert img[..., 1].max() > 0  # depth in G
```

- [ ] **Step 4.2: Run, expect failure**

```bash
uv run pytest tests/photobooth_sweep/test_preprocess.py -v
```

Expected: FAIL.

- [ ] **Step 4.3: Implement `preprocess.py`**

```python
"""Build the CN control image for the photobooth workflow.

Composite layout (always BGR uint8):
    R channel = Canny edges (if condition includes 'canny')
    G channel = Depth (if condition includes 'depth')
    B channel = 0

The face_pixel_budget axis determines the canvas size + face placement:
    tight_1024 -> 1024 canvas, face crop ~85% of height
    med_1024   -> 1024 canvas, face crop ~65% of height
    tight_768  ->  768 canvas, face crop ~85% of height
"""
from __future__ import annotations

import cv2
import numpy as np

CANNY_PRESETS = {
    "soft":       {"low":  50, "high": 150, "blur": 1.5},
    "default":    {"low": 100, "high": 200, "blur": 0.0},
    "aggressive": {"low": 150, "high": 250, "blur": 0.0},
}

FACE_PIXEL_BUDGETS = {
    # (canvas_w, canvas_h, face_height_frac)
    "tight_1024": (1024, 1024, 0.85),
    "med_1024":   (1024, 1024, 0.65),
    "tight_768":  (768,  768,  0.85),
}

_DEPTH = None  # lazy DepthAnything-V2 pipeline


def _detect_face_bbox(bgr: np.ndarray) -> tuple[int, int, int, int]:
    """SCRFD face bbox via swap_core's insightface app. Returns x0,y0,x1,y1."""
    import sys
    sys.path.insert(0, "scripts")
    from swap_core import make_face_app
    app = make_face_app()
    faces = app.get(bgr)
    if not faces:
        h, w = bgr.shape[:2]
        return 0, 0, w, h
    f = max(faces, key=lambda x: x.det_score)
    return tuple(map(int, f.bbox))


def _crop_face(bgr: np.ndarray, margin_frac: float = 0.35) -> np.ndarray:
    x0, y0, x1, y1 = _detect_face_bbox(bgr)
    h, w = bgr.shape[:2]
    m = int(margin_frac * max(x1 - x0, y1 - y0))
    return bgr[max(0, y0 - m):min(h, y1 + m),
               max(0, x0 - m):min(w, x1 + m)]


def _canny(face_bgr: np.ndarray, preset: str) -> np.ndarray:
    p = CANNY_PRESETS[preset]
    g = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    if p["blur"] > 0:
        g = cv2.GaussianBlur(g, (0, 0), p["blur"])
    return cv2.Canny(g, p["low"], p["high"])


def _depth(face_bgr: np.ndarray) -> np.ndarray:
    """DepthAnything-V2 small -> uint8 0..255. Falls back to Sobel proxy."""
    global _DEPTH
    if _DEPTH is None:
        try:
            from transformers import pipeline
            _DEPTH = pipeline("depth-estimation",
                              model="depth-anything/Depth-Anything-V2-Small-hf",
                              device=-1)
        except Exception:
            _DEPTH = "fallback"
    if _DEPTH == "fallback":
        g = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.convertScaleAbs(cv2.Laplacian(g, cv2.CV_32F, ksize=5))
    from PIL import Image
    pil = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
    res = _DEPTH(pil)
    return np.array(res["depth"].convert("L"))


def build_control_image(source_bgr: np.ndarray, *,
                        condition: str,
                        canny_preset: str | None,
                        face_pixel_budget: str) -> np.ndarray:
    """Compose Canny / Depth / both into the workflow's control image."""
    canvas_w, canvas_h, face_h_frac = FACE_PIXEL_BUDGETS[face_pixel_budget]
    face = _crop_face(source_bgr)
    target_h = int(canvas_h * face_h_frac)
    fh, fw = face.shape[:2]
    target_w = int(fw * (target_h / fh))
    face = cv2.resize(face, (target_w, target_h))

    canvas = np.zeros((canvas_h, canvas_w, 3), np.uint8)
    y0 = (canvas_h - target_h) // 2
    x0 = (canvas_w - target_w) // 2
    face_padded = np.zeros_like(canvas)
    face_padded[y0:y0 + target_h, x0:x0 + target_w] = face

    if "canny" in condition:
        canny = _canny(face_padded, canny_preset or "default")
        canvas[..., 2] = canny  # R channel (BGR)
    if "depth" in condition:
        depth = _depth(face_padded)
        canvas[..., 1] = depth  # G channel
    return canvas
```

- [ ] **Step 4.4: Tests pass**

```bash
uv run pytest tests/photobooth_sweep/test_preprocess.py -v
```

Expected: 6 passed.

- [ ] **Step 4.5: Commit**

```bash
git add scripts/photobooth_sweep/preprocess.py tests/photobooth_sweep/test_preprocess.py
git commit -m "feat(photobooth): Canny preset + Depth + composite control image"
```

---

## Task 5: Scorer (TDD on synthetic + cn_grid sample)

**Files:**
- Create: `scripts/photobooth_sweep/scorer.py`
- Create: `tests/photobooth_sweep/test_scorer.py`

- [ ] **Step 5.1: Failing tests**

```python
# tests/photobooth_sweep/test_scorer.py
import cv2
import numpy as np
import pytest

from scripts.photobooth_sweep.scorer import (
    face_frac, lab_delta_ab, score_cell,
)


def test_face_frac_zero_when_no_face():
    img = np.zeros((512, 512, 3), np.uint8)
    assert face_frac(img) == 0.0


def test_face_frac_positive_on_known_doll():
    img = cv2.imread("exp_output/cn_grid/swaps/id_01_str90_st6.png")
    if img is None:
        pytest.skip("cn_grid sample not present")
    assert 0.01 < face_frac(img) < 0.6


def test_lab_delta_ab_zero_for_identical_image():
    img = cv2.imread("data/importer/identities/id_01.png")
    from scripts.photobooth_sweep.features import build_skin_mask
    mask = build_skin_mask(img)
    delta = lab_delta_ab(img, img, mask)
    assert delta < 0.5  # numerical-noise floor
```

- [ ] **Step 5.2: Implement `scorer.py`**

```python
"""Per-cell metrics: id_cos, det_mode, lab_delta_ab, clip_style, face_frac."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "scripts")
from swap_core import crop_and_upscale, mediapipe_kps_bbox

from .features import build_skin_mask

_CLIP = None  # lazy open_clip


def _clip_model():
    global _CLIP
    if _CLIP is None:
        try:
            import open_clip
            import torch
            model, _, prep = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k")
            model.eval()
            _CLIP = (model, prep, torch)
        except Exception as e:
            print(f"[scorer] CLIP unavailable: {e}", file=sys.stderr)
            _CLIP = "unavailable"
    return _CLIP


def face_frac(bgr: np.ndarray) -> float:
    """SCRFD face-bbox area / image area, or 0 if no detection."""
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"],
                       providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    faces = app.get(bgr)
    if not faces:
        return 0.0
    f = max(faces, key=lambda x: x.det_score)
    x0, y0, x1, y1 = f.bbox
    return float((x1 - x0) * (y1 - y0)) / (bgr.shape[0] * bgr.shape[1])


def lab_delta_ab(source_bgr: np.ndarray, final_bgr: np.ndarray,
                 source_mask: np.ndarray) -> float:
    """sqrt(Δa² + Δb²) between source skin mean and final skin mean."""
    final_mask = build_skin_mask(final_bgr)
    if not source_mask.any() or not final_mask.any():
        return float("nan")
    src_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    fin_lab = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    src_mu = src_lab[source_mask > 0].mean(axis=0)  # L,a,b
    fin_mu = fin_lab[final_mask > 0].mean(axis=0)
    da = (fin_mu[1] - src_mu[1])
    db = (fin_mu[2] - src_mu[2])
    return float(np.sqrt(da * da + db * db))


def clip_style(doll_bgr: np.ndarray, anchor_bgr: np.ndarray) -> float:
    """open_clip ViT-B/32 cosine between doll and curated anchor."""
    clip = _clip_model()
    if clip == "unavailable":
        return float("nan")
    model, prep, torch = clip
    from PIL import Image
    a = prep(Image.fromarray(cv2.cvtColor(doll_bgr, cv2.COLOR_BGR2RGB))
             ).unsqueeze(0)
    b = prep(Image.fromarray(cv2.cvtColor(anchor_bgr, cv2.COLOR_BGR2RGB))
             ).unsqueeze(0)
    with torch.no_grad():
        ea = model.encode_image(a)
        eb = model.encode_image(b)
    ea = ea / ea.norm(dim=-1, keepdim=True)
    eb = eb / eb.norm(dim=-1, keepdim=True)
    return float((ea @ eb.T).item())


def id_cos(app, bgr, src_emb) -> tuple[float, str, float]:
    """id_cos, det_mode, det_score. Mirrors cn_grid_sweep.id_cos exactly."""
    kps, bbox = mediapipe_kps_bbox(bgr)
    if bbox is None:
        return float("nan"), "failed", 0.0
    up, _ = crop_and_upscale(bgr, bbox)
    if up is None:
        return float("nan"), "failed", 0.0
    faces = app.get(up)
    if faces:
        f = max(faces, key=lambda x: x.det_score)
        return (float(np.dot(f.normed_embedding, src_emb)),
                "default", float(f.det_score))
    kps_up, bbox_up = mediapipe_kps_bbox(up)
    rec = app.models.get("recognition")
    if kps_up is None or rec is None:
        return float("nan"), "failed", 0.0
    from insightface.app.common import Face
    f = Face(bbox=bbox_up, kps=kps_up, det_score=1.0)
    rec.get(up, f)
    return (float(np.dot(f.normed_embedding, src_emb)), "forced", 1.0)


def score_cell(*, app, source_bgr, source_mask, source_emb,
               doll_bgr, final_bgr, anchor_bgr) -> dict:
    cos, det_mode, det = id_cos(app, final_bgr, source_emb)
    return {
        "id_cos": None if cos != cos else cos,
        "det_mode": det_mode,
        "det_score": det,
        "lab_delta_ab": lab_delta_ab(source_bgr, final_bgr, source_mask),
        "clip_style": clip_style(doll_bgr, anchor_bgr),
        "face_frac": face_frac(final_bgr),
    }
```

- [ ] **Step 5.3: Tests pass + commit**

```bash
uv run pytest tests/photobooth_sweep/test_scorer.py -v
git add scripts/photobooth_sweep/scorer.py tests/photobooth_sweep/test_scorer.py
git commit -m "feat(photobooth): per-cell scorer (id_cos, lab_delta, clip_style, face_frac)"
```

---

## Task 6: Photobooth CN workflow JSON

**Files:**
- Create: `comfyui/workflows/photobooth_zimage_cn.api.json`

- [ ] **Step 6.1: Write the parametric workflow**

`comfyui/workflows/photobooth_zimage_cn.api.json`:

```json
{
  "1":  {"class_type": "UNETLoader", "inputs": {"unet_name": "z_image_turbo_bf16.safetensors", "weight_dtype": "default"}},
  "2":  {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "lumina2", "device": "default"}},
  "3":  {"class_type": "VAELoader", "inputs": {"vae_name": "z_image_ae.safetensors"}},
  "4":  {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": "$$POSITIVE_PROMPT"}},
  "5":  {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
  "6":  {"class_type": "EmptySD3LatentImage", "inputs": {"width": "$$WIDTH", "height": "$$HEIGHT", "batch_size": 1}},
  "7":  {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 3.0}},
  "20": {"class_type": "ModelPatchLoader", "inputs": {"name": "Z-Image-Turbo-Fun-Controlnet-Union.safetensors"}},
  "21": {"class_type": "LoadImage", "inputs": {"image": "$$CONTROL_IMAGE"}},
  "22": {"class_type": "ZImageFunControlnet", "inputs": {"model": ["7", 0], "model_patch": ["20", 0], "vae": ["3", 0], "strength": "$$CN_STRENGTH", "start_percent": "$$CN_START_PERCENT", "end_percent": 1.0, "image": ["21", 0]}},
  "8":  {"class_type": "KSampler", "inputs": {"model": ["22", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0], "seed": "$$SEED", "steps": 6, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
  "9":  {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
  "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "$$OUTPUT_PREFIX"}}
}
```

Substitutions to perform at runtime: `$$POSITIVE_PROMPT`, `$$WIDTH`, `$$HEIGHT`, `$$CONTROL_IMAGE`, `$$CN_STRENGTH`, `$$CN_START_PERCENT`, `$$SEED`, `$$OUTPUT_PREFIX`.

- [ ] **Step 6.2: Smoke test on shard via curl**

```bash
# upload a placeholder control image first
python -c "
import cv2, requests, io, numpy as np
img = (np.random.rand(1024,1024,3)*255).astype('uint8')
ok,buf = cv2.imencode('.png', img)
r = requests.post('http://192.168.87.25:8188/upload/image',
  files={'image': ('smoke.png', io.BytesIO(buf), 'image/png')},
  data={'overwrite':'true'})
print(r.json())
"
```

Then submit a minimal workflow with substitutions filled and confirm a doll comes out. Expected: 200 + a `prompt_id`, then within 30 s an image at `output/photobooth_smoke_00001_.png`. Investigate workflow JSON syntax if any node errors.

- [ ] **Step 6.3: Commit**

```bash
git add comfyui/workflows/photobooth_zimage_cn.api.json
git commit -m "feat(photobooth): parametric Z-Image Turbo + Fun-CN workflow"
```

---

## Task 7: Photobooth refine workflow JSON + refine.py

**Files:**
- Create: `comfyui/workflows/photobooth_zimage_refine.api.json`
- Create: `scripts/photobooth_sweep/refine.py`

- [ ] **Step 7.1: Write the refine workflow**

The pattern: VAEEncode(image) → SetLatentNoiseMask → KSampler at `$$DENOISE` → VAEDecode → SaveImage.

```json
{
  "1":  {"class_type": "UNETLoader", "inputs": {"unet_name": "z_image_turbo_bf16.safetensors", "weight_dtype": "default"}},
  "2":  {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "lumina2", "device": "default"}},
  "3":  {"class_type": "VAELoader", "inputs": {"vae_name": "z_image_ae.safetensors"}},
  "4":  {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": "$$POSITIVE_PROMPT"}},
  "5":  {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
  "7":  {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 3.0}},
  "30": {"class_type": "LoadImage", "inputs": {"image": "$$IMAGE"}},
  "31": {"class_type": "LoadImage", "inputs": {"image": "$$MASK"}},
  "32": {"class_type": "ImageToMask", "inputs": {"image": ["31", 0], "channel": "red"}},
  "33": {"class_type": "VAEEncode", "inputs": {"pixels": ["30", 0], "vae": ["3", 0]}},
  "34": {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["33", 0], "mask": ["32", 0]}},
  "8":  {"class_type": "KSampler", "inputs": {"model": ["7", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["34", 0], "seed": "$$SEED", "steps": 6, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": "$$DENOISE"}},
  "9":  {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
  "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "$$OUTPUT_PREFIX"}}
}
```

- [ ] **Step 7.2: Implement `refine.py`**

```python
"""Masked-img2img refine pass via remote ComfyUI."""
from __future__ import annotations

import io
import json
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import requests


def upload(url: str, bgr: np.ndarray, name: str) -> str:
    ok, buf = cv2.imencode(".png", bgr)
    r = requests.post(f"{url}/upload/image",
                      files={"image": (name, io.BytesIO(buf.tobytes()),
                                       "image/png")},
                      data={"overwrite": "true"}, timeout=30)
    r.raise_for_status()
    j = r.json()
    return f"{j['subfolder']}/{j['name']}" if j.get("subfolder") else j["name"]


def run_workflow(url: str, wf: dict, timeout_s: int = 120) -> dict | None:
    cid = str(uuid.uuid4())
    r = requests.post(f"{url}/prompt", json={"prompt": wf, "client_id": cid},
                      timeout=30)
    if r.status_code != 200:
        return None
    pid = r.json()["prompt_id"]
    for _ in range(timeout_s):
        h = requests.get(f"{url}/history/{pid}", timeout=10).json()
        if pid in h:
            if h[pid]["status"].get("status_str") != "success":
                return None
            for o in h[pid].get("outputs", {}).values():
                for im in o.get("images", []):
                    return im
            return None
        time.sleep(1)
    return None


def fetch(url: str, im: dict) -> np.ndarray:
    r = requests.get(f"{url}/view", params={
        "filename": im["filename"], "subfolder": im.get("subfolder", ""),
        "type": "output"}, timeout=60)
    r.raise_for_status()
    return cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)


def refine(*, comfy_url: str, image_bgr: np.ndarray, mask_u8: np.ndarray,
           denoise: float, prompt: str, seed: int,
           prefix: str = "photobooth/refine") -> np.ndarray | None:
    wf = json.load(open("comfyui/workflows/photobooth_zimage_refine.api.json"))
    img_name = upload(comfy_url, image_bgr, f"{prefix.replace('/', '_')}_img.png")
    mask_3 = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
    msk_name = upload(comfy_url, mask_3, f"{prefix.replace('/', '_')}_mask.png")

    subs = {"$$POSITIVE_PROMPT": prompt, "$$IMAGE": img_name,
            "$$MASK": msk_name, "$$DENOISE": float(denoise),
            "$$SEED": int(seed), "$$OUTPUT_PREFIX": prefix}
    for n in wf.values():
        for k, v in n.get("inputs", {}).items():
            if isinstance(v, str) and v in subs:
                n["inputs"][k] = subs[v]
    im = run_workflow(comfy_url, wf)
    if im is None:
        return None
    return fetch(comfy_url, im)
```

- [ ] **Step 7.3: Smoke test refine end-to-end**

```bash
python -c "
import cv2
from scripts.photobooth_sweep.features import build_skin_mask
from scripts.photobooth_sweep.refine import refine
img = cv2.imread('exp_output/cn_grid/swaps/id_01_str90_st6.png')
mask = build_skin_mask(img)
out = refine(comfy_url='http://192.168.87.25:8188',
             image_bgr=img, mask_u8=mask,
             denoise=0.30, prompt='matryoshka doll face',
             seed=42, prefix='photobooth/smoke_refine')
print('out:', None if out is None else out.shape)
"
```

Expected: returns an HxWx3 array; saved on the shard.

- [ ] **Step 7.4: Commit**

```bash
git add comfyui/workflows/photobooth_zimage_refine.api.json scripts/photobooth_sweep/refine.py
git commit -m "feat(photobooth): masked-img2img refine workflow + driver helper"
```

---

## Task 8: Driver loop

**Files:**
- Create: `scripts/photobooth_sweep/driver.py`
- Create: `scripts/photobooth_sweep/README.md`

- [ ] **Step 8.1: Implement `driver.py`**

```python
"""Photobooth Phase 1 sweep driver.

Resumable: every (photo_id, cfg_id) cell appends one row to scores.parquet.
On startup, completed rows are skipped. Artifacts go under
exp_output/photobooth_phase1/.
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests

os.chdir("/home/newub/w/vamp-interface")
sys.path.insert(0, "scripts")
from swap_core import (DEFAULT_SWAPPER, detect_source, load_swapper,
                       make_face_app, swap_identity)

from photobooth_sweep import color_match, refine as refine_mod
from photobooth_sweep.axes import lhs_sample, write_manifest
from photobooth_sweep.features import (PHASE1_PHOTOS, build_skin_mask,
                                       extract_phase1_features)
from photobooth_sweep.preprocess import (FACE_PIXEL_BUDGETS,
                                         build_control_image)
from photobooth_sweep.scorer import score_cell

URL = os.environ.get("COMFY_URL", "http://192.168.87.25:8188")
OUT = Path("exp_output/photobooth_phase1")
RENDERS = OUT / "renders"
SOURCES = OUT / "sources"
LOGS = OUT / "logs"
MANIFEST = OUT / "manifest.parquet"
SCORES = OUT / "scores.parquet"
ANCHOR = "exp_output/cn_grid/renders/id_03_str90_st6.png"

BASE_PROMPT = ("a vibrant traditional Russian matryoshka nesting doll, "
               "glossy red and gold lacquer, ornate floral painting, "
               "with a realistic photographic human face, soft three-"
               "dimensional shading, correct facial proportions, defined "
               "nose and lips, natural-sized eyes, centered frontal face, "
               "wooden doll, plain background")


def _prompt(cfg: dict, feats: dict) -> str:
    prefix = ""
    if cfg["demo_inject"]:
        prefix = (f"a {feats['age_bin']}-year-old {feats['race']} "
                  f"{feats['gender']} face, ")
    return prefix + BASE_PROMPT


def _upload(bgr, name):
    ok, buf = cv2.imencode(".png", bgr)
    r = requests.post(f"{URL}/upload/image",
                      files={"image": (name, io.BytesIO(buf.tobytes()),
                                       "image/png")},
                      data={"overwrite": "true"}, timeout=30)
    r.raise_for_status()
    j = r.json()
    return f"{j['subfolder']}/{j['name']}" if j.get("subfolder") else j["name"]


def _submit_cn(*, photo_id, cfg, prompt, control_name, seed):
    wf = json.load(open("comfyui/workflows/photobooth_zimage_cn.api.json"))
    canvas_w, canvas_h, _ = FACE_PIXEL_BUDGETS[cfg["face_pixel_budget"]]
    subs = {"$$POSITIVE_PROMPT": prompt, "$$WIDTH": canvas_w,
            "$$HEIGHT": canvas_h, "$$CONTROL_IMAGE": control_name,
            "$$CN_STRENGTH": float(cfg["cn_strength"]),
            "$$CN_START_PERCENT": float(cfg["cn_start_percent"]),
            "$$SEED": int(seed),
            "$$OUTPUT_PREFIX": f"photobooth_phase1/{photo_id}/{cfg['cfg_id']}"}
    for n in wf.values():
        for k, v in n.get("inputs", {}).items():
            if isinstance(v, str) and v in subs:
                n["inputs"][k] = subs[v]
    im = refine_mod.run_workflow(URL, wf, timeout_s=120)
    return None if im is None else refine_mod.fetch(URL, im)


def _done_keys() -> set[tuple[str, str]]:
    if not SCORES.exists():
        return set()
    tbl = pq.read_table(SCORES)
    return {(r["photo_id"], r["cfg_id"]) for r in tbl.to_pylist()
            if r.get("ok")}


def _append_score(row: dict) -> None:
    tbl = pa.Table.from_pylist([row])
    if SCORES.exists():
        existing = pq.read_table(SCORES)
        tbl = pa.concat_tables([existing, tbl], promote_options="default")
    pq.write_table(tbl, str(SCORES))


def main(n_configs: int = 40, seed: int = 0):
    for p in (OUT, RENDERS, SOURCES, LOGS):
        p.mkdir(parents=True, exist_ok=True)

    feats = extract_phase1_features(SOURCES / "features.json")
    for pid in PHASE1_PHOTOS:
        src = SOURCES / f"{pid}.png"
        if not src.exists():
            src.symlink_to(Path("../../../data/importer/identities") /
                           f"{pid}.png")

    cfgs = lhs_sample(n_configs=n_configs, seed=seed)
    write_manifest(MANIFEST, photo_ids=PHASE1_PHOTOS, configs=cfgs)

    app = make_face_app()
    swapper = load_swapper(DEFAULT_SWAPPER)
    anchor = cv2.imread(ANCHOR)

    done = _done_keys()
    cells = [(pid, c) for pid in PHASE1_PHOTOS for c in cfgs]
    todo = [(pid, c) for pid, c in cells if (pid, c["cfg_id"]) not in done]
    t_start = time.time()
    print(f"phase1: {len(cells)} cells, {len(done)} done, {len(todo)} to run",
          flush=True)

    for i, (pid, cfg) in enumerate(todo):
        out_dir = RENDERS / pid / cfg["cfg_id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        src_bgr = cv2.imread(str(SOURCES / f"{pid}.png"))
        src_face = detect_source(app, src_bgr)
        src_emb = src_face.normed_embedding
        src_mask = build_skin_mask(src_bgr)

        try:
            control = build_control_image(
                src_bgr, condition=cfg["cn_condition"],
                canny_preset=cfg["canny_preset"],
                face_pixel_budget=cfg["face_pixel_budget"])
            cv2.imwrite(str(out_dir / "control.png"), control)
            ctrl_name = _upload(control, f"phase1_{pid}_{cfg['cfg_id']}.png")

            prompt = _prompt(cfg, feats[pid])
            seed_v = 91_000_000 + int(pid.split("_")[1])
            t0 = time.time()
            doll = _submit_cn(photo_id=pid, cfg=cfg, prompt=prompt,
                              control_name=ctrl_name, seed=seed_v)
            render_s = time.time() - t0
            if doll is None:
                raise RuntimeError("cn workflow returned None")
            cv2.imwrite(str(out_dir / "doll.png"), doll)

            t1 = time.time()
            swap, _mode, _det = swap_identity(app, swapper, doll, src_face)
            swap_s = time.time() - t1
            swap = color_match.apply(cfg["color_match"], swap, src_bgr,
                                     build_skin_mask(swap))
            cv2.imwrite(str(out_dir / "swap.png"), swap)

            final = swap
            refine_s = None
            if cfg["refine_denoise"] != "off":
                t2 = time.time()
                refined = refine_mod.refine(
                    comfy_url=URL, image_bgr=swap,
                    mask_u8=build_skin_mask(swap),
                    denoise=float(cfg["refine_denoise"]),
                    prompt=BASE_PROMPT, seed=seed_v + 1,
                    prefix=f"photobooth_phase1/{pid}/{cfg['cfg_id']}_refine")
                refine_s = time.time() - t2
                if refined is not None:
                    final = refined
                    cv2.imwrite(str(out_dir / "refine.png"), refined)
            cv2.imwrite(str(out_dir / "final.png"), final)

            metrics = score_cell(
                app=app, source_bgr=src_bgr, source_mask=src_mask,
                source_emb=src_emb, doll_bgr=doll, final_bgr=final,
                anchor_bgr=anchor)
            row = {"photo_id": pid, "cfg_id": cfg["cfg_id"], "ok": True,
                   "render_s": round(render_s, 2),
                   "swap_s": round(swap_s, 2),
                   "refine_s": None if refine_s is None else round(refine_s, 2),
                   "fail_reason": None,
                   **metrics}
            _append_score(row)
            eta = (time.time() - t_start) / (i + 1) * (len(todo) - i - 1)
            print(f"  [{i+1}/{len(todo)}] {pid}/{cfg['cfg_id']} "
                  f"cos={metrics['id_cos']} det={metrics['det_mode']} "
                  f"eta={eta/60:.1f}min", flush=True)
        except Exception as e:
            _append_score({"photo_id": pid, "cfg_id": cfg["cfg_id"],
                           "ok": False, "fail_reason": str(e)[:200],
                           "id_cos": None, "det_mode": "failed",
                           "det_score": None, "lab_delta_ab": None,
                           "clip_style": None, "face_frac": None,
                           "render_s": None, "swap_s": None, "refine_s": None})
            (LOGS / "failures.jsonl").open("a").write(
                json.dumps({"photo_id": pid, "cfg_id": cfg["cfg_id"],
                            "error": str(e)}) + "\n")

    print(f"done in {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 8.2: README**

`scripts/photobooth_sweep/README.md`:

```markdown
# photobooth_sweep — Phase 1 LHS pilot

Run Phase 1 (4 photos × 40 LHS configs):

```bash
COMFY_URL=http://192.168.87.25:8188 \
  uv run python -m scripts.photobooth_sweep.driver
```

Resume after interruption: the same command. Completed cells are skipped.

Outputs land under `exp_output/photobooth_phase1/`:
- `manifest.parquet` — one row per cell with axis values
- `scores.parquet` — one row per cell with metrics
- `renders/{photo_id}/{cfg_id}/{control,doll,swap,refine,final}.png`
- `logs/failures.jsonl` — per-cell exception trail
- `sources/features.json` — per-photo features extracted once

To analyse:

```python
import duckdb
db = duckdb.connect()
df = db.execute("""
  SELECT * FROM 'exp_output/photobooth_phase1/manifest.parquet' m
  JOIN 'exp_output/photobooth_phase1/scores.parquet' s
    USING (photo_id, cfg_id)
""").df()
```
```

- [ ] **Step 8.3: Commit**

```bash
git add scripts/photobooth_sweep/driver.py scripts/photobooth_sweep/README.md
git commit -m "feat(photobooth): Phase 1 driver loop + README"
```

---

## Task 9: Smoke run (1 photo × 5 configs)

- [ ] **Step 9.1: Run a sliver**

```bash
COMFY_URL=http://192.168.87.25:8188 \
  uv run python -c "
from scripts.photobooth_sweep.driver import main
main(n_configs=5, seed=0)
" 2>&1 | tee /tmp/photobooth_smoke.log
```

Expected: ~5 cells × 4 photos = 20 cells finish in ~3 min; `scores.parquet` exists with 20 rows; `det_mode == 'default'` on ≥80% of rows.

- [ ] **Step 9.2: Inspect outputs**

```bash
duckdb -c "SELECT photo_id, det_mode, id_cos, lab_delta_ab, face_frac FROM 'exp_output/photobooth_phase1/scores.parquet'"
```

Sanity checks: id_cos values 0.5–0.9, lab_delta_ab finite, face_frac > 0.02 on most cells.

- [ ] **Step 9.3: Decision gate**

If any of:
- ≥30% cells `ok=False` → debug before full run
- All id_cos null → swap/scorer bug, debug
- All face_frac == 0 → control image / canvas-size bug, debug

Otherwise wipe `exp_output/photobooth_phase1/` and proceed to Task 10.

```bash
rm -rf exp_output/photobooth_phase1/  # only if smoke passed
```

---

## Task 10: Full Phase 1 (4 × 40)

- [ ] **Step 10.1: Launch**

```bash
COMFY_URL=http://192.168.87.25:8188 \
  uv run python -m scripts.photobooth_sweep.driver 2>&1 \
  | tee exp_output/photobooth_phase1/logs/run.log
```

Expected: ~40 min wall.

- [ ] **Step 10.2: Verify completeness**

```bash
duckdb -c "
  SELECT COUNT(*) total,
         COUNT(*) FILTER (WHERE ok) ok,
         COUNT(*) FILTER (WHERE NOT ok) failed,
         AVG(render_s) avg_render
    FROM 'exp_output/photobooth_phase1/scores.parquet'
"
```

Expected: total=160, ok≥150, avg_render ~5–9 s.

- [ ] **Step 10.3: Save findings stub**

Create `docs/research/2026-05-19-photobooth-phase1-findings.md` with the metric summary table grouped by each axis. Commit so the analysis has a home for Phase 2 design.

- [ ] **Step 10.4: Final commit**

```bash
git add exp_output/photobooth_phase1/manifest.parquet \
        exp_output/photobooth_phase1/scores.parquet \
        exp_output/photobooth_phase1/sources/features.json \
        docs/research/2026-05-19-photobooth-phase1-findings.md
git commit -m "exp(photobooth): Phase 1 results — 4 photos x 40 LHS configs"
```

---

## Self-review checklist

- **Spec coverage:** all 8 axes are encoded in axes.py (Task 1). Skin mask defined once (Task 2). Color match (Task 3). Preprocess incl. depth + composite (Task 4). All 5 scoring metrics (Task 5). Both workflow JSONs (Tasks 6+7). Resumable driver (Task 8). Smoke + full run (Tasks 9+10). Open question on `canny+depth` flagged in Task 4.

- **Placeholders:** none. Every code block is complete.

- **Type consistency:** `cfg["color_match"]`, `cfg["cn_condition"]`, `cfg["canny_preset"]`, `cfg["face_pixel_budget"]` all use the levels declared in `AXES`. `build_skin_mask` returns uint8 0/255 — consumed identically in color_match, scorer, and driver.
