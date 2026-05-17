# Chibi Geometry Redesign — Staged Re-priming Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `ChibiField` (scale-and-slide) with a staged pipeline of *re-priming* operators that produce a genuine block-headed chibi from a v3 FLAME-topology textured mesh.

**Architecture:** Four ordered `verts → verts` stages — HeadBlock, ProportionRemap, ReliefFlatten, FeaturePrimitives — composed by a `ChibiPipeline`. Each stage blends the mesh toward an explicit chibi target shape (a superellipsoid block, a smoothed surface, feature primitives) and is gated by its own geometric metric. Global structure first, local detail last.

**Tech Stack:** Python 3.10 (the `lam` conda env), PyTorch, pytorch3d, pytest. FLAME 5023-vert topology. Spec: `docs/superpowers/specs/2026-05-17-chibi-geometry-redesign-design.md`.

---

## Conventions (apply to every task)

- **Env preamble** for every test/run command:
  `export PATH=/home/newub/miniconda3/envs/lam/bin:$PATH PYTHONPATH=/home/newub/w/LAM XFORMERS_DISABLED=1`
- All repo commands run from `/home/newub/w/vamp-interface`.
- Tests: `python -m pytest <path> -v`. New test files go in `tests/`.
- Commit messages use conventional commits and end with:
  `Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>`
- FLAME assets:
  - template OBJ: `/home/newub/w/LAM/model_zoo/human_parametric_models/flame_assets/flame/head_template_mesh.obj`
  - masks: `/home/newub/w/LAM/model_zoo/human_parametric_models/flame_assets/flame/FLAME_masks.pkl`
    (regions: `scalp forehead face nose lips eye_region left/right_eyeball left/right_eye_region neck boundary left/right_ear`)
- A v3 textured anchor mesh for integration runs:
  `exp_output/lam_chibi/renders/bake_v3/asian_m/asian_m_textured.obj`
- Existing helpers to reuse: `chibi.mesh_extract.load_textured_mesh`, `chibi.mesh.TexturedMesh`,
  `chibi.landmarks.landmark_positions / landmark_lines / GROUPS / region_falloff_weights`,
  `chibi.field.ChibiField` (its `remap` math only).

---

## File structure

| File | Responsibility |
|------|----------------|
| `src/chibi/primitives.py` | Geometry primitives: `Box`, `fit_box`, `project_superellipsoid`, `laplacian_smoothed`, `mask_weights`. No chibi logic. |
| `src/chibi/chibi_metrics.py` | Verification metrics: `box_residual`, `relief_energy`, `bridge_height`, `eye_aspect`, `foldover_count`. |
| `src/chibi/stages/__init__.py` | Marks `stages` a package. |
| `src/chibi/stages/head_block.py` | Stage 1 — `head_block(verts, faces, masks_path, params)`. |
| `src/chibi/stages/proportion_remap.py` | Stage 2 — `proportion_remap(verts, params)` (reuses `ChibiField.remap`). |
| `src/chibi/stages/relief_flatten.py` | Stage 3 — `relief_flatten(verts, faces, masks_path, params)`. |
| `src/chibi/stages/feature_primitives.py` | Stage 4 — `feature_primitives(verts, faces, masks_path, params)`. |
| `src/chibi/pipeline.py` | `ChibiPipeline` — composes the stages, JSON params, `run(mesh, through=None)`. |
| `src/chibi/mesh_deform.py` (modify) | `apply_chibi` runs `ChibiPipeline` instead of `ChibiField`. |
| `scripts/chibi_mesh_render.py` (modify) | `--through STAGE` flag for per-stage inspection renders. |
| `tests/test_primitives.py`, `tests/test_chibi_metrics.py`, `tests/test_chibi_stages.py`, `tests/test_chibi_pipeline.py` | Tests. |

All stage operators share one signature shape: they take `verts` (V,3 float64) plus whatever they need, and **return new verts** (V,3) — never mutate, never change vert count, never touch faces.

---

## Task 1: `primitives.py` — geometry primitives

**Files:**
- Create: `src/chibi/primitives.py`
- Test: `tests/test_primitives.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_primitives.py`:

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import torch
from chibi.primitives import (Box, fit_box, project_superellipsoid,
                              laplacian_smoothed)


def test_fit_box_recovers_known_extent():
    # points filling a [-2,2]x[-1,1]x[-3,3] box
    g = torch.linspace(-1, 1, 8)
    pts = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    pts = pts * torch.tensor([2.0, 1.0, 3.0])
    box = fit_box(pts)
    assert torch.allclose(box.center, torch.zeros(3), atol=1e-4)
    assert torch.allclose(box.half, torch.tensor([2.0, 1.0, 3.0]), atol=1e-4)


def test_project_superellipsoid_high_exponent_is_boxy():
    box = Box(center=torch.zeros(3), half=torch.ones(3))
    # a point on the +x face diagonal; n large -> projects near the box face
    p = torch.tensor([[0.5, 0.5, 0.0]])
    out = project_superellipsoid(p, box, exponent=16.0)
    # boxy: the dominant axis lands on the face (coord ~= +-1)
    assert out.abs().max() > 0.95


def test_project_superellipsoid_n2_is_sphere():
    box = Box(center=torch.zeros(3), half=torch.ones(3))
    p = torch.tensor([[3.0, 4.0, 0.0]])          # radius 5
    out = project_superellipsoid(p, box, exponent=2.0)
    assert torch.allclose(out.norm(), torch.tensor(1.0), atol=1e-4)


def test_laplacian_smoothed_reduces_a_bump():
    # flat grid with one raised vertex -> smoothing lowers the bump
    g = torch.linspace(-1, 1, 7)
    xy = torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    verts = torch.cat([xy, torch.zeros(xy.shape[0], 1)], 1).double()
    bump = verts.shape[0] // 2
    verts[bump, 2] = 1.0
    # build faces from the grid
    faces = []
    n = 7
    for i in range(n - 1):
        for j in range(n - 1):
            a, b = i * n + j, i * n + j + 1
            c, d = a + n, b + n
            faces += [[a, b, c], [b, d, c]]
    faces = torch.tensor(faces, dtype=torch.int64)
    sm = laplacian_smoothed(verts, faces, iters=10, lam=0.5)
    assert sm[bump, 2] < 0.5            # bump pulled down
    assert verts[bump, 2] == 1.0        # input not mutated
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `export PATH=/home/newub/miniconda3/envs/lam/bin:$PATH PYTHONPATH=/home/newub/w/LAM XFORMERS_DISABLED=1 && python -m pytest tests/test_primitives.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chibi.primitives'`

- [ ] **Step 3: Implement `primitives.py`**

Create `src/chibi/primitives.py`:

```python
"""Geometry primitives for the chibi staged pipeline — no chibi logic here.

A `Box` is an axis-aligned bounding box. `project_superellipsoid` maps verts
radially (from the box centre) onto a superellipsoid inscribed in the box —
exponent 2 is an ellipsoid, large exponent approaches the box. That single
operator is the chibi "block" target. `laplacian_smoothed` is the relief-free
reference surface. `mask_weights` turns FLAME mask regions into smooth
per-vertex blend weights.
"""
from __future__ import annotations
from dataclasses import dataclass
import pickle
import numpy as np
import torch


@dataclass
class Box:
    """Axis-aligned box. center (3,), half (3,) — both float64."""
    center: torch.Tensor
    half: torch.Tensor


def fit_box(points: torch.Tensor, pct: float = 1.0) -> Box:
    """Axis-aligned box from a point cloud. `pct` is the percentile trim per
    axis (1.0 = use the 1st/99th percentile, robust to a few outlier verts;
    0.0 = exact min/max)."""
    p = points.to(torch.float64)
    lo = torch.quantile(p, pct / 100.0, dim=0)
    hi = torch.quantile(p, 1.0 - pct / 100.0, dim=0)
    return Box(center=(lo + hi) * 0.5, half=(hi - lo).clamp_min(1e-6) * 0.5)


def project_superellipsoid(verts: torch.Tensor, box: Box,
                           exponent: float) -> torch.Tensor:
    """Project each vert radially (from box.center) onto the superellipsoid
    inscribed in `box`. exponent=2 -> ellipsoid; large -> box. Returns (V,3)."""
    d = verts.to(torch.float64) - box.center
    # superellipsoid implicit: sum((|d_i|/h_i)^n) = 1 on the surface.
    norm = ((d.abs() / box.half).clamp_min(1e-12) ** exponent).sum(1)
    t = norm.clamp_min(1e-12) ** (-1.0 / exponent)        # scale onto surface
    return box.center + d * t.unsqueeze(1)


def _adjacency(faces: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (nbr_index, nbr_count) for a uniform-Laplacian: directed edges
    both ways, deduplicated is unnecessary (double-counting cancels in a
    mean)."""
    e = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], 0)
    e = torch.cat([e, e.flip(1)], 0)
    return e[:, 0], e[:, 1]


def laplacian_smoothed(verts: torch.Tensor, faces: torch.Tensor,
                       iters: int = 10, lam: float = 0.5) -> torch.Tensor:
    """Uniform-Laplacian smoothing: v += lam * (mean(1-ring nbrs) - v),
    `iters` times. Returns a new tensor; does not mutate `verts`."""
    v = verts.to(torch.float64).clone()
    src, dst = _adjacency(faces.to(torch.int64), v.shape[0])
    n = v.shape[0]
    for _ in range(iters):
        nbr_sum = torch.zeros_like(v).index_add_(0, src, v[dst])
        deg = torch.zeros(n, dtype=torch.float64).index_add_(
            0, src, torch.ones(src.shape[0], dtype=torch.float64))
        nbr_mean = nbr_sum / deg.clamp_min(1.0).unsqueeze(1)
        v = v + lam * (nbr_mean - v)
    return v


def mask_weights(verts: torch.Tensor, masks_path: str, names: list[str],
                 falloff: float = 0.015) -> torch.Tensor:
    """Smooth per-vertex weight in [0,1]: 1.0 on any vertex in any of the
    named FLAME mask regions, decaying as exp(-(d/falloff)^2) with Euclidean
    distance to the nearest masked vertex. falloff is in FLAME mesh units."""
    with open(masks_path, "rb") as fh:
        masks = pickle.load(fh, encoding="latin1")
    idx = np.unique(np.concatenate([np.asarray(masks[n]) for n in names]))
    v = verts.detach().to(torch.float64)
    core = v[torch.as_tensor(idx, dtype=torch.long)]
    d = torch.cdist(v, core).min(dim=1).values
    return torch.exp(-(d / falloff) ** 2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `export PATH=/home/newub/miniconda3/envs/lam/bin:$PATH PYTHONPATH=/home/newub/w/LAM XFORMERS_DISABLED=1 && python -m pytest tests/test_primitives.py -v`
Expected: PASS — 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/chibi/primitives.py tests/test_primitives.py
git commit -m "feat(chibi): primitives — Box fit, superellipsoid projection, Laplacian smoothing

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 2: `chibi_metrics.py` — per-stage verification metrics

**Files:**
- Create: `src/chibi/chibi_metrics.py`
- Test: `tests/test_chibi_metrics.py`

These metrics are what each later stage's gate asserts — they can *see* chibi-ness (block residual, relief energy, nose-bridge height, eye roundness, mesh fold-over), unlike the landmark targets the old fit used.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_chibi_metrics.py`:

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import torch
from chibi.primitives import Box, project_superellipsoid
from chibi.chibi_metrics import box_residual, relief_energy, foldover_count


def _grid_mesh(n=7):
    g = torch.linspace(-1, 1, n)
    xy = torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    verts = torch.cat([xy, torch.zeros(xy.shape[0], 1)], 1).double()
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b = i * n + j, i * n + j + 1
            c, d = a + n, b + n
            faces += [[a, b, c], [b, d, c]]
    return verts, torch.tensor(faces, dtype=torch.int64)


def test_box_residual_zero_on_surface():
    box = Box(center=torch.zeros(3), half=torch.ones(3))
    pts = torch.randn(50, 3).double()
    on = project_superellipsoid(pts, box, exponent=8.0)
    assert box_residual(on, box, exponent=8.0) < 1e-5
    off = on * 1.5
    assert box_residual(off, box, exponent=8.0) > 0.1


def test_relief_energy_drops_when_flat():
    verts, faces = _grid_mesh()
    w = torch.ones(verts.shape[0], dtype=torch.float64)
    flat = relief_energy(verts, faces, w)
    verts[verts.shape[0] // 2, 2] = 1.0
    bumpy = relief_energy(verts, faces, w)
    assert bumpy > flat


def test_foldover_count_zero_for_identity():
    verts, faces = _grid_mesh()
    assert foldover_count(verts, verts, faces) == 0
    flipped = verts.clone()
    flipped[:, 2] = -flipped[:, 2] - 0.0   # mirror -> normals flip
    assert foldover_count(verts, flipped, faces) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `export PATH=/home/newub/miniconda3/envs/lam/bin:$PATH PYTHONPATH=/home/newub/w/LAM XFORMERS_DISABLED=1 && python -m pytest tests/test_chibi_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chibi.chibi_metrics'`

- [ ] **Step 3: Implement `chibi_metrics.py`**

Create `src/chibi/chibi_metrics.py`:

```python
"""Geometric verification metrics for the chibi pipeline stages.

Each stage's gate asserts one of these. They measure the chibi properties
directly — block-ness, flatness, button-ness, eye roundness, mesh integrity —
so a stage that hits its target metric has actually done its job, unlike the
landmark-position targets the superseded ChibiField fit was blind through.
"""
from __future__ import annotations
import torch

from chibi.primitives import Box, project_superellipsoid, laplacian_smoothed
from chibi.landmarks import landmark_positions, GROUPS


def box_residual(verts: torch.Tensor, box: Box, exponent: float) -> float:
    """RMS distance of verts to the superellipsoid surface. Falls toward 0 as
    HeadBlock blends the head onto the block."""
    proj = project_superellipsoid(verts, box, exponent)
    return float(((verts.to(torch.float64) - proj) ** 2).sum(1).mean().sqrt())


def relief_energy(verts: torch.Tensor, faces: torch.Tensor,
                  weights: torch.Tensor, iters: int = 10) -> float:
    """Weighted RMS deviation of verts from their Laplacian-smoothed surface —
    the amount of 3-D micro-relief. ReliefFlatten drives it down."""
    sm = laplacian_smoothed(verts, faces, iters=iters)
    dev = ((verts.to(torch.float64) - sm) ** 2).sum(1)         # (V,)
    w = weights.to(torch.float64)
    return float(((dev * w).sum() / w.sum().clamp_min(1e-9)).sqrt())


def bridge_height(verts: torch.Tensor) -> float:
    """z-extent of the nose-bridge landmarks (68-pt indices 27-30). A button
    nose has a near-zero bridge ridge."""
    lm = landmark_positions(verts.to(torch.float64))
    z = lm[27:31, 2]
    return float(z.max() - z.min())


def eye_aspect(verts: torch.Tensor) -> float:
    """Width/height ratio of the eye landmark group (indices 36-47). A round
    chibi eye trends toward ~1; a realistic almond is ~2-3."""
    lm = landmark_positions(verts.to(torch.float64))
    eye = lm[GROUPS["eye"]]
    w = eye[:, 0].max() - eye[:, 0].min()
    h = (eye[:, 1].max() - eye[:, 1].min()).clamp_min(1e-9)
    return float(w / h)


def foldover_count(base_verts: torch.Tensor, verts: torch.Tensor,
                   faces: torch.Tensor) -> int:
    """Number of faces whose normal flipped relative to `base_verts` — the
    triangle self-fold metric (the wasp-waist class of failure)."""
    def fn(v):
        f = v.to(torch.float64)[faces.to(torch.int64)]
        n = torch.linalg.cross(f[:, 1] - f[:, 0], f[:, 2] - f[:, 0])
        return n / n.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return int(((fn(base_verts) * fn(verts)).sum(1) < 0).sum())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `export PATH=/home/newub/miniconda3/envs/lam/bin:$PATH PYTHONPATH=/home/newub/w/LAM XFORMERS_DISABLED=1 && python -m pytest tests/test_chibi_metrics.py -v`
Expected: PASS — 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/chibi/chibi_metrics.py tests/test_chibi_metrics.py
git commit -m "feat(chibi): chibi_metrics — block/relief/bridge/eye/foldover metrics

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 3: Stage 1 — `head_block.py`

**Files:**
- Create: `src/chibi/stages/__init__.py` (empty)
- Create: `src/chibi/stages/head_block.py`
- Test: `tests/test_chibi_stages.py`

`head_block` fits a `Box` to the head verts (cranium + face, excluding the `neck` region), then blends each vert toward its superellipsoid projection — strongly on the cranium (`scalp`+`forehead`), mildly on the `face` panel, zero on the neck. This is the structural block-head move.

- [ ] **Step 1: Write the failing test**

Create `tests/test_chibi_stages.py`:

```python
import sys, pathlib, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import torch, pytest
from chibi.fit import _load_obj_verts
from chibi.landmarks import FLAME_TEMPLATE, _template_faces

MASKS = "/home/newub/w/LAM/model_zoo/human_parametric_models/flame_assets/flame/FLAME_masks.pkl"
needs_flame = pytest.mark.skipif(not os.path.exists(MASKS),
                                 reason="FLAME assets not present")


def _flame_mesh():
    v = _load_obj_verts(FLAME_TEMPLATE).double()
    f = torch.as_tensor(_template_faces(), dtype=torch.int64)
    return v, f


@needs_flame
def test_head_block_identity_strength_is_noop():
    from chibi.stages.head_block import head_block, HeadBlockParams
    v, f = _flame_mesh()
    out = head_block(v, f, MASKS,
                     HeadBlockParams(exponent=8.0, strength_cranium=0.0,
                                     strength_face=0.0))
    assert torch.allclose(out, v, atol=1e-9)


@needs_flame
def test_head_block_lowers_box_residual():
    from chibi.stages.head_block import head_block, HeadBlockParams
    from chibi.primitives import fit_box
    from chibi.chibi_metrics import box_residual
    v, f = _flame_mesh()
    p = HeadBlockParams(exponent=8.0, strength_cranium=0.9, strength_face=0.3)
    out = head_block(v, f, MASKS, p)
    box = fit_box(v)
    assert box_residual(out, box, 8.0) < box_residual(v, box, 8.0)
    assert out.shape == v.shape
```

- [ ] **Step 2: Run test to verify it fails**

Run: `export PATH=/home/newub/miniconda3/envs/lam/bin:$PATH PYTHONPATH=/home/newub/w/LAM XFORMERS_DISABLED=1 && python -m pytest tests/test_chibi_stages.py -k head_block -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chibi.stages'`

- [ ] **Step 3: Implement the stage**

Create empty `src/chibi/stages/__init__.py`.

Create `src/chibi/stages/head_block.py`:

```python
"""Stage 1 — reshape the head toward a rounded block.

Fit an axis-aligned box to the head verts (everything but the neck), then
blend each vert toward its superellipsoid projection on that box. The cranium
(scalp+forehead) blends strongly — that is the block silhouette; the face
panel blends mildly — it flattens toward the box front but keeps enough shape
for the later stages to re-prime features on. The neck is left untouched.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch

from chibi.primitives import Box, fit_box, project_superellipsoid, mask_weights

CRANIUM = ["scalp", "forehead"]
FACE = ["face", "nose", "lips", "eye_region"]
NECK = ["neck"]


@dataclass
class HeadBlockParams:
    exponent: float = 8.0          # superellipsoid: 2 ellipsoid .. large box
    strength_cranium: float = 0.9  # blend weight on scalp+forehead
    strength_face: float = 0.3     # blend weight on the face panel


def head_block(verts: torch.Tensor, faces: torch.Tensor, masks_path: str,
               params: HeadBlockParams) -> torch.Tensor:
    """Return verts reshaped toward a superellipsoid block. Vert count and
    faces are unchanged."""
    v = verts.to(torch.float64)
    w_neck = mask_weights(v, masks_path, NECK, falloff=0.02)
    head = v[w_neck < 0.5]                       # box fit excludes the neck
    box = fit_box(head, pct=1.0)
    target = project_superellipsoid(v, box, params.exponent)

    w_cran = mask_weights(v, masks_path, CRANIUM, falloff=0.03)
    w_face = mask_weights(v, masks_path, FACE, falloff=0.03)
    # per-vertex blend strength; neck verts forced to 0.
    strength = (params.strength_cranium * w_cran
                + params.strength_face * w_face).clamp(0.0, 1.0)
    strength = strength * (1.0 - w_neck).clamp(0.0, 1.0)
    return v + strength.unsqueeze(1) * (target - v)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `export PATH=/home/newub/miniconda3/envs/lam/bin:$PATH PYTHONPATH=/home/newub/w/LAM XFORMERS_DISABLED=1 && python -m pytest tests/test_chibi_stages.py -k head_block -v`
Expected: PASS — 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/chibi/stages/__init__.py src/chibi/stages/head_block.py tests/test_chibi_stages.py
git commit -m "feat(chibi): stage 1 head_block — reshape head toward a superellipsoid

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 4: Stage 2 — `proportion_remap.py`

**Files:**
- Create: `src/chibi/stages/proportion_remap.py`
- Test: `tests/test_chibi_stages.py` (append)

Reuses `ChibiField`'s monotone vertical remap — the one operator from the old field that worked. This stage is a thin wrapper: build a `ChibiField`, frame it to the mesh, apply only its `remap` to the y-coordinate.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_chibi_stages.py`:

```python
@needs_flame
def test_proportion_remap_moves_eye_line_toward_half():
    from chibi.stages.proportion_remap import proportion_remap, RemapParams
    from chibi.landmarks import landmark_lines
    v, f = _flame_mesh()
    out = proportion_remap(v, RemapParams())
    base_eye = float(landmark_lines(v)["eye"])
    chibi_eye = float(landmark_lines(out)["eye"])
    # realistic eye ~0.48; chibi target 0.50 -> moves toward 0.5
    assert abs(chibi_eye - 0.50) < abs(base_eye - 0.50)
    assert out.shape == v.shape


@needs_flame
def test_proportion_remap_is_monotone_in_y():
    from chibi.stages.proportion_remap import proportion_remap, RemapParams
    v, f = _flame_mesh()
    out = proportion_remap(v, RemapParams())
    order_in = torch.argsort(v[:, 1])
    y_out_sorted = out[order_in, 1]
    # monotone remap: sorting by input y keeps output y non-decreasing
    assert (y_out_sorted[1:] - y_out_sorted[:-1] >= -1e-6).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `export PATH=/home/newub/miniconda3/envs/lam/bin:$PATH PYTHONPATH=/home/newub/w/LAM XFORMERS_DISABLED=1 && python -m pytest tests/test_chibi_stages.py -k proportion_remap -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chibi.stages.proportion_remap'`

- [ ] **Step 3: Implement the stage**

Create `src/chibi/stages/proportion_remap.py`:

```python
"""Stage 2 — quarter-grid vertical remap.

The one operator kept from the superseded ChibiField: a monotone piecewise-
linear remap of the normalized head fraction u, placing the feature bands on
the chibi quarter grid (big forehead, eye band on the 1/2 line, mouth on 3/4,
generous rounded chin). Reuses ChibiField.remap so the remap math has a single
source of truth.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import torch

from chibi.field import ChibiField
from chibi.landmarks import landmark_positions


@dataclass
class RemapParams:
    # 5 softplus increments for the 6-knot monotone remap; zeros = identity.
    # Defaults reproduce the fitted chibi quarter grid from the spec.
    remap_incr: list = field(
        default_factory=lambda: [0.34, -0.30, 0.27, -0.36, 0.0])


def proportion_remap(verts: torch.Tensor, params: RemapParams) -> torch.Tensor:
    """Return verts with the y-coordinate remapped to the chibi quarter grid;
    x and z unchanged. Vert count unchanged."""
    v = verts.to(torch.float64)
    y_crown = float(v[:, 1].max())
    y_chin = float(landmark_positions(v[:5023])[8, 1])
    fieldm = ChibiField(y_crown=y_crown, y_chin=y_chin,
                        z_center=float(v[:, 2].mean())).double()
    with torch.no_grad():
        fieldm.remap_incr.copy_(torch.tensor(params.remap_incr,
                                             dtype=torch.float64))
    u = fieldm.u_of(v)
    u_chibi = fieldm.remap(u)
    new_y = fieldm.y_crown - u_chibi * (fieldm.y_crown - fieldm.y_chin)
    out = v.clone()
    out[:, 1] = new_y
    return out
```

Note: `ChibiField.u_of`/`remap` need the mesh to be FLAME-topology (first 5023 verts) for the chin landmark. The pipeline (Task 5) only ever passes FLAME-topology meshes.

- [ ] **Step 4: Run test to verify it passes**

Run: `export PATH=/home/newub/miniconda3/envs/lam/bin:$PATH PYTHONPATH=/home/newub/w/LAM XFORMERS_DISABLED=1 && python -m pytest tests/test_chibi_stages.py -k proportion_remap -v`
Expected: PASS — 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/chibi/stages/proportion_remap.py tests/test_chibi_stages.py
git commit -m "feat(chibi): stage 2 proportion_remap — quarter-grid vertical remap

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 5: `ChibiPipeline` + rewire `apply_chibi` + per-stage render — **MILESTONE: block head**

**Files:**
- Create: `src/chibi/pipeline.py`
- Modify: `src/chibi/mesh_deform.py`
- Modify: `scripts/chibi_mesh_render.py`
- Test: `tests/test_chibi_pipeline.py`

After this task the pipeline runs HeadBlock + ProportionRemap end-to-end and renders. This is the milestone gate for the user's first priority — the block-like head — *before* the relief and feature stages are added.

- [ ] **Step 1: Write the failing test**

Create `tests/test_chibi_pipeline.py`:

```python
import sys, pathlib, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import torch, pytest
from chibi.fit import _load_obj_verts
from chibi.landmarks import FLAME_TEMPLATE, _template_faces

MASKS = "/home/newub/w/LAM/model_zoo/human_parametric_models/flame_assets/flame/FLAME_masks.pkl"
needs_flame = pytest.mark.skipif(not os.path.exists(MASKS),
                                 reason="FLAME assets not present")


@needs_flame
def test_pipeline_through_stops_after_named_stage():
    from chibi.pipeline import ChibiPipeline
    v = _load_obj_verts(FLAME_TEMPLATE).double()
    f = torch.as_tensor(_template_faces(), dtype=torch.int64)
    pipe = ChibiPipeline(MASKS)
    only_block = pipe.run(v, f, through="head_block")
    full = pipe.run(v, f)
    assert only_block.shape == v.shape
    # stopping early differs from the full run
    assert not torch.allclose(only_block, full)


@needs_flame
def test_pipeline_keeps_mesh_coherent():
    from chibi.pipeline import ChibiPipeline
    from chibi.chibi_metrics import foldover_count
    v = _load_obj_verts(FLAME_TEMPLATE).double()
    f = torch.as_tensor(_template_faces(), dtype=torch.int64)
    out = ChibiPipeline(MASKS).run(v, f)
    # no catastrophic self-fold (the wasp-waist class of failure)
    assert foldover_count(v, out, f) < 0.02 * f.shape[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `export PATH=/home/newub/miniconda3/envs/lam/bin:$PATH PYTHONPATH=/home/newub/w/LAM XFORMERS_DISABLED=1 && python -m pytest tests/test_chibi_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chibi.pipeline'`

- [ ] **Step 3: Implement `ChibiPipeline`**

Create `src/chibi/pipeline.py`:

```python
"""ChibiPipeline — compose the chibi re-priming stages.

Stages run global-structure-first: head_block, proportion_remap, then (added
in later tasks) relief_flatten and feature_primitives. `run(verts, faces,
through=)` applies them in order, optionally stopping after a named stage for
inspection. Parameters load from one JSON; absent keys use the dataclass
defaults.
"""
from __future__ import annotations
import json
from pathlib import Path
import torch

from chibi.stages.head_block import head_block, HeadBlockParams
from chibi.stages.proportion_remap import proportion_remap, RemapParams

# stage name -> (callable, params-dataclass). Extended in Tasks 6 and 7.
STAGE_ORDER = ["head_block", "proportion_remap"]


class ChibiPipeline:
    def __init__(self, masks_path: str, params_path: str | None = None):
        self.masks_path = masks_path
        cfg = {}
        if params_path and Path(params_path).exists():
            cfg = json.loads(Path(params_path).read_text())
        self.head_block = HeadBlockParams(**cfg.get("head_block", {}))
        self.remap = RemapParams(**cfg.get("proportion_remap", {}))

    def run(self, verts: torch.Tensor, faces: torch.Tensor,
            through: str | None = None) -> torch.Tensor:
        """Apply stages in order. `through` stops after that stage (inclusive).
        Returns deformed verts (V,3); faces are never modified."""
        if through is not None and through not in STAGE_ORDER:
            raise ValueError(f"unknown stage {through!r}; "
                             f"known: {STAGE_ORDER}")
        v = verts.to(torch.float64)
        v = head_block(v, faces, self.masks_path, self.head_block)
        if through == "head_block":
            return v
        v = proportion_remap(v, self.remap)
        return v
```

- [ ] **Step 4: Rewire `apply_chibi`**

In `src/chibi/mesh_deform.py`, replace the `_deform_verts` body so it runs the pipeline. Replace the existing `_deform_verts` function with:

```python
def _deform_verts(verts: torch.Tensor, faces: torch.Tensor,
                  masks_path: str, params_path: str | None) -> torch.Tensor:
    """Deform (V,3) verts with the staged ChibiPipeline."""
    from chibi.pipeline import ChibiPipeline
    pipe = ChibiPipeline(masks_path, params_path)
    return pipe.run(verts.to(torch.float64), faces)
```

And update `apply_chibi` to pass `faces` and treat `field_params_path` as the optional pipeline-params JSON:

```python
def apply_chibi(mesh, field_params_path: str | None, masks_path: str):
    """Apply the staged ChibiPipeline to `mesh` (ChibiMesh or TexturedMesh).
    `field_params_path` is the optional pipeline-params JSON (None -> stage
    defaults). Returns the same mesh type with deformed verts; faces and
    appearance pass through unchanged."""
    assert mesh.verts.shape[0] in (5023, 20018), (
        f"apply_chibi expects FLAME topology (5023 or 20018 verts); "
        f"got {mesh.verts.shape[0]}")
    deformed = _deform_verts(mesh.verts, mesh.faces, masks_path,
                             field_params_path)
    if isinstance(mesh, TexturedMesh):
        return TexturedMesh(verts=deformed, faces=mesh.faces, uv=mesh.uv,
                            uv_faces=mesh.uv_faces, texture=mesh.texture)
    return ChibiMesh(verts=deformed, faces=mesh.faces, rgb=mesh.rgb)
```

Delete the now-unused imports (`load_field_params`, `landmark_positions`, `region_falloff_weights`) from `mesh_deform.py` and update the module docstring's first line to: `"""Deform a mesh's vertices with the staged ChibiPipeline."""`. Leave the rest of the docstring.

- [ ] **Step 5: Update the existing `apply_chibi` tests**

In `tests/test_chibi_mesh.py`, the two `apply_chibi` tests built a `ChibiField` JSON via `save_field_params`. Replace both `@needs_flame` `test_apply_chibi_*` tests with pipeline-aware versions:

```python
@needs_flame
def test_apply_chibi_deforms_and_preserves_faces_rgb(tmp_path):
    mesh = _flame_template_chibi_mesh()
    out = apply_chibi(mesh, None, MASKS)          # None -> pipeline defaults
    assert torch.equal(out.faces, mesh.faces)
    assert torch.equal(out.rgb, mesh.rgb)
    assert out.verts.shape == mesh.verts.shape
    assert not torch.allclose(out.verts, mesh.verts, atol=1e-3)


@needs_flame
def test_apply_chibi_on_textured_mesh_returns_textured_mesh(tmp_path):
    from chibi.uv_template import load_flame_uv
    from chibi.fit import _load_obj_verts
    from chibi.landmarks import FLAME_TEMPLATE, _template_faces
    v = _load_obj_verts(FLAME_TEMPLATE).double()
    faces = torch.as_tensor(_template_faces(), dtype=torch.int64)
    fuv = load_flame_uv(FLAME_TEMPLATE)
    tex = torch.full((16, 16, 3), 0.5, dtype=torch.float32)
    mesh = TexturedMesh(verts=v, faces=faces, uv=fuv.uv,
                        uv_faces=fuv.uv_faces, texture=tex)
    out = apply_chibi(mesh, None, MASKS)
    assert isinstance(out, TexturedMesh)
    assert torch.equal(out.texture, mesh.texture)
    assert not torch.allclose(out.verts, mesh.verts, atol=1e-3)
```

Remove the now-unused `_flame_template_chibi_mesh` import line only if it is unused; it is still used by the first test, so keep it.

- [ ] **Step 6: Add `--through` to the render driver**

In `scripts/chibi_mesh_render.py`, add an argument and pass it down. After the existing `ap.add_argument` lines add:

```python
    ap.add_argument("--through", default=None,
                    help="stop the pipeline after this stage (inspection)")
```

The script currently calls `apply_chibi(mesh, args.field, MASKS)`. The pipeline needs `through`; `apply_chibi` does not take it. Replace that call with a direct pipeline run for the chibi mesh:

```python
    from chibi.pipeline import ChibiPipeline
    pipe = ChibiPipeline(MASKS, args.field if Path(args.field).exists() else None)
    deformed = pipe.run(mesh.verts, mesh.faces, through=args.through)
    chibi = type(mesh)(verts=deformed, faces=mesh.faces, uv=mesh.uv,
                       uv_faces=mesh.uv_faces, texture=mesh.texture)
```

(`mesh` is always a `TexturedMesh` in this script.) Tag the output filenames with the stage when `--through` is set: change the chibi/side-by-side output stems to `f"{stem}_{args.through}" if args.through else stem`.

- [ ] **Step 7: Run tests**

Run: `export PATH=/home/newub/miniconda3/envs/lam/bin:$PATH PYTHONPATH=/home/newub/w/LAM XFORMERS_DISABLED=1 && python -m pytest tests/test_chibi_pipeline.py tests/test_chibi_mesh.py -v`
Expected: PASS — all pass (pipeline 2, mesh tests incl. the rewritten apply_chibi 2).

- [ ] **Step 8: MILESTONE — render the block head and inspect**

```bash
bash scripts/chibi_mesh_render.sh asian_m
```
Then extract a frame and view:
```bash
ffmpeg -y -loglevel error -i exp_output/lam_chibi/renders/mesh_v1/sidebyside_asian_m.mp4 \
    -vf "select=eq(n\,0)" -vframes 1 /tmp/milestone_block.png
```
Expected: the chibi side is a coherent **block-ish head** with the quarter-grid proportions — no wasp-waist fold, no cracks. The face features are still realistic (relief + feature stages come next). If the head does not read as meaningfully more block-like than the baseline, tune `HeadBlockParams` (raise `exponent` toward 12-16 for blockier, raise `strength_cranium`) in `scripts/chibi_mesh_render.py`'s default path or via a params JSON, and re-render before proceeding.

- [ ] **Step 9: Commit**

```bash
git add src/chibi/pipeline.py src/chibi/mesh_deform.py scripts/chibi_mesh_render.py tests/test_chibi_pipeline.py tests/test_chibi_mesh.py
git commit -m "feat(chibi): ChibiPipeline + rewire apply_chibi — block-head milestone

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 6: Stage 3 — `relief_flatten.py`

**Files:**
- Create: `src/chibi/stages/relief_flatten.py`
- Modify: `src/chibi/pipeline.py`
- Test: `tests/test_chibi_stages.py` (append)

`relief_flatten` blends the face-panel verts toward their Laplacian-smoothed surface — erasing cheekbone volume, nasolabial folds, brow ridge, nose-bridge ridge (rule 8a).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_chibi_stages.py`:

```python
@needs_flame
def test_relief_flatten_lowers_relief_energy():
    from chibi.stages.relief_flatten import relief_flatten, ReliefParams
    from chibi.chibi_metrics import relief_energy
    from chibi.primitives import mask_weights
    v, f = _flame_mesh()
    w = mask_weights(v, MASKS, ["face", "nose"], falloff=0.02)
    out = relief_flatten(v, f, MASKS, ReliefParams(strength=0.8))
    assert relief_energy(out, f, w) < relief_energy(v, f, w)
    assert out.shape == v.shape


@needs_flame
def test_relief_flatten_identity_strength_is_noop():
    from chibi.stages.relief_flatten import relief_flatten, ReliefParams
    v, f = _flame_mesh()
    out = relief_flatten(v, f, MASKS, ReliefParams(strength=0.0))
    assert torch.allclose(out, v, atol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `export PATH=/home/newub/miniconda3/envs/lam/bin:$PATH PYTHONPATH=/home/newub/w/LAM XFORMERS_DISABLED=1 && python -m pytest tests/test_chibi_stages.py -k relief_flatten -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chibi.stages.relief_flatten'`

- [ ] **Step 3: Implement the stage**

Create `src/chibi/stages/relief_flatten.py`:

```python
"""Stage 3 — flatten facial relief.

Blend the face-panel verts toward their Laplacian-smoothed surface, erasing
the 3-D micro-anatomy (cheekbone volume, nasolabial folds, brow ridge, nose-
bridge ridge) that makes a chibi-proportioned realistic face read as uncanny
(painter rule 8a). The eyeballs are excluded — they are re-primed in stage 4,
and smoothing a sphere into the lid is not wanted here.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch

from chibi.primitives import laplacian_smoothed, mask_weights

PANEL = ["face", "nose", "forehead", "eye_region", "lips"]


@dataclass
class ReliefParams:
    strength: float = 0.8     # blend weight toward the smoothed surface
    iters: int = 12           # Laplacian smoothing iterations
    falloff: float = 0.02     # mask falloff (FLAME units)


def relief_flatten(verts: torch.Tensor, faces: torch.Tensor, masks_path: str,
                   params: ReliefParams) -> torch.Tensor:
    """Return verts with the face panel blended toward its smoothed surface.
    Vert count and faces unchanged."""
    v = verts.to(torch.float64)
    sm = laplacian_smoothed(v, faces, iters=params.iters)
    w = mask_weights(v, masks_path, PANEL, falloff=params.falloff)
    blend = (params.strength * w).clamp(0.0, 1.0).unsqueeze(1)
    return v + blend * (sm - v)
```

- [ ] **Step 4: Wire the stage into the pipeline**

In `src/chibi/pipeline.py`:
- add `from chibi.stages.relief_flatten import relief_flatten, ReliefParams`;
- change `STAGE_ORDER` to `["head_block", "proportion_remap", "relief_flatten"]`;
- in `__init__` add `self.relief = ReliefParams(**cfg.get("relief_flatten", {}))`;
- in `run`, after the `proportion_remap` line and before `return v`, add:

```python
        v = relief_flatten(v, faces, self.masks_path, self.relief)
        if through == "relief_flatten":
            return v
```

Also move the existing `if through == "proportion_remap": return v` check in — i.e. after the `proportion_remap` call insert `if through == "proportion_remap": return v` before the relief line.

- [ ] **Step 5: Run tests**

Run: `export PATH=/home/newub/miniconda3/envs/lam/bin:$PATH PYTHONPATH=/home/newub/w/LAM XFORMERS_DISABLED=1 && python -m pytest tests/test_chibi_stages.py tests/test_chibi_pipeline.py -v`
Expected: PASS — all pass.

- [ ] **Step 6: Inspect**

```bash
bash scripts/chibi_mesh_render.sh asian_m
ffmpeg -y -loglevel error -i exp_output/lam_chibi/renders/mesh_v1/sidebyside_asian_m.mp4 -vf "select=eq(n\,0)" -vframes 1 /tmp/relief.png
```
Expected: the face panel is visibly smoother — cheekbone and nasolabial relief reduced. If identity is lost (face unrecognisable / featureless), lower `ReliefParams.strength`; if relief survives, raise it. Tune via the defaults and re-render.

- [ ] **Step 7: Commit**

```bash
git add src/chibi/stages/relief_flatten.py src/chibi/pipeline.py tests/test_chibi_stages.py
git commit -m "feat(chibi): stage 3 relief_flatten — smooth away facial micro-relief

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 7: Stage 4 — `feature_primitives.py`

**Files:**
- Create: `src/chibi/stages/feature_primitives.py`
- Modify: `src/chibi/pipeline.py`
- Test: `tests/test_chibi_stages.py` (append)

Re-prime the three features as chibi primitives on the now-flat panel: eye → big round lens (per-eye centroid — this is the fix for the eye-pod ballooning), nose → button (collapse the bridge, shrink the tip), mouth → strip (vertical compression).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_chibi_stages.py`:

```python
@needs_flame
def test_feature_primitives_round_the_eye_and_shrink_the_bridge():
    from chibi.stages.feature_primitives import (feature_primitives,
                                                 FeatureParams)
    from chibi.chibi_metrics import eye_aspect, bridge_height
    v, f = _flame_mesh()
    out = feature_primitives(v, f, MASKS, FeatureParams())
    assert eye_aspect(out) < eye_aspect(v)          # rounder
    assert bridge_height(out) < bridge_height(v)    # flatter bridge
    assert out.shape == v.shape


@needs_flame
def test_feature_primitives_identity_params_is_noop():
    from chibi.stages.feature_primitives import (feature_primitives,
                                                 FeatureParams)
    v, f = _flame_mesh()
    out = feature_primitives(v, f, MASKS,
                             FeatureParams(eye_round=0.0, bridge_collapse=0.0,
                                           mouth_compress=0.0))
    assert torch.allclose(out, v, atol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `export PATH=/home/newub/miniconda3/envs/lam/bin:$PATH PYTHONPATH=/home/newub/w/LAM XFORMERS_DISABLED=1 && python -m pytest tests/test_chibi_stages.py -k feature_primitives -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement the stage**

Create `src/chibi/stages/feature_primitives.py`:

```python
"""Stage 4 — re-prime eye, nose, mouth as chibi primitives.

On the flattened panel, rebuild each feature toward a chibi target:
  - eye: round the lid aperture toward a circle, per-eye centroid (per-eye is
    the fix for the shared-centroid ballooning of the old field);
  - nose: collapse the bridge toward the local face plane -> a button;
  - mouth: vertical compression toward a strip.
Each is a blend-toward-target with its own strength knob.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch

from chibi.primitives import mask_weights


@dataclass
class FeatureParams:
    eye_round: float = 0.7        # 0..1 blend toward a round aperture
    eye_enlarge: float = 1.6      # in-plane aperture scale
    bridge_collapse: float = 0.8  # 0..1 collapse of nose-bridge depth
    mouth_compress: float = 0.5   # vertical mouth compression (0..1)


def _round_one_eye(v: torch.Tensor, w: torch.Tensor, params: FeatureParams
                   ) -> torch.Tensor:
    """Round + enlarge one eye's verts (weighted by w) about their own
    centroid, in the x-y (face-tangent) plane only — z is left alone so the
    eye does not balloon in depth."""
    wsum = w.sum().clamp_min(1e-6)
    c = (v * w[:, None]).sum(0) / wsum
    d = v - c
    # current aperture half-extents
    hx = (d[:, 0].abs() * w).sum() / wsum
    hy = (d[:, 1].abs() * w).sum() / wsum
    r = (hx + hy) * 0.5                                  # round target radius
    # blend each axis' scale toward the common radius, then enlarge
    sx = (1.0 + params.eye_round * (r / hx.clamp_min(1e-6) - 1.0))
    sy = (1.0 + params.eye_round * (r / hy.clamp_min(1e-6) - 1.0))
    scale = torch.tensor([sx * params.eye_enlarge,
                          sy * params.eye_enlarge, 1.0], dtype=v.dtype)
    target = c + d * scale
    return v + w[:, None] * (target - v)


def feature_primitives(verts: torch.Tensor, faces: torch.Tensor,
                       masks_path: str, params: FeatureParams) -> torch.Tensor:
    """Return verts with eye/nose/mouth re-primed. Vert count unchanged."""
    v = verts.to(torch.float64)
    # --- eyes: per-eye, so the centroid is each eye's own ---
    for region in (["left_eye_region", "left_eyeball"],
                   ["right_eye_region", "right_eyeball"]):
        w = mask_weights(v, masks_path, region, falloff=0.012)
        v = _round_one_eye(v, w, params)
    # --- nose: collapse the bridge toward the local panel plane (z) ---
    w_nose = mask_weights(v, masks_path, ["nose"], falloff=0.012)
    z_panel = (v[:, 2] * w_nose).sum() / w_nose.sum().clamp_min(1e-6)
    v = v.clone()
    v[:, 2] = v[:, 2] + params.bridge_collapse * w_nose * (z_panel - v[:, 2])
    # --- mouth: vertical compression toward the lip-region centroid ---
    w_mouth = mask_weights(v, masks_path, ["lips"], falloff=0.010)
    y_c = (v[:, 1] * w_mouth).sum() / w_mouth.sum().clamp_min(1e-6)
    v[:, 1] = v[:, 1] + params.mouth_compress * w_mouth * (y_c - v[:, 1])
    return v
```

- [ ] **Step 4: Wire the stage into the pipeline**

In `src/chibi/pipeline.py`:
- add `from chibi.stages.feature_primitives import feature_primitives, FeatureParams`;
- change `STAGE_ORDER` to `["head_block", "proportion_remap", "relief_flatten", "feature_primitives"]`;
- in `__init__` add `self.features = FeatureParams(**cfg.get("feature_primitives", {}))`;
- in `run`, after the `relief_flatten` block and before `return v`, add:

```python
        v = feature_primitives(v, faces, self.masks_path, self.features)
        return v
```

- [ ] **Step 5: Run tests**

Run: `export PATH=/home/newub/miniconda3/envs/lam/bin:$PATH PYTHONPATH=/home/newub/w/LAM XFORMERS_DISABLED=1 && python -m pytest tests/test_chibi_stages.py tests/test_chibi_pipeline.py -v`
Expected: PASS — all pass.

- [ ] **Step 6: Inspect**

```bash
bash scripts/chibi_mesh_render.sh asian_m
ffmpeg -y -loglevel error -i exp_output/lam_chibi/renders/mesh_v1/sidebyside_asian_m.mp4 -vf "select=eq(n\,0)" -vframes 1 /tmp/features.png
```
Expected: rounder, larger eyes (no depth pods); a button nose with no bridge ridge; a strip mouth. Tune `FeatureParams` strengths against the render and the painter-rules doc.

- [ ] **Step 7: Commit**

```bash
git add src/chibi/stages/feature_primitives.py src/chibi/pipeline.py tests/test_chibi_stages.py
git commit -m "feat(chibi): stage 4 feature_primitives — round eyes, button nose, strip mouth

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 8: Full-pipeline verdict + per-stage inspection renders

**Files:**
- Modify: `scripts/chibi_mesh_render.py` (only if the verdict step exposes a bug)

- [ ] **Step 1: Render every stage for both anchors**

```bash
for s in asian_m me_512; do
  for stage in head_block proportion_remap relief_flatten feature_primitives; do
    python scripts/chibi_mesh_render.py \
      --obj exp_output/lam_chibi/renders/bake_v3/$s/${s}_textured.obj \
      --field exp_output/lam_chibi/diff_geometry/chibi_pipeline_params.json \
      --out exp_output/lam_chibi/renders/chibi_v2 --through $stage || true
  done
  python scripts/chibi_mesh_render.py \
    --obj exp_output/lam_chibi/renders/bake_v3/$s/${s}_textured.obj \
    --field exp_output/lam_chibi/diff_geometry/chibi_pipeline_params.json \
    --out exp_output/lam_chibi/renders/chibi_v2
done
```
(The `--field` JSON need not exist — `ChibiPipeline` falls back to stage defaults; `|| true` guards the inspection runs.)

- [ ] **Step 2: Build the per-stage contact sheet**

```bash
python - <<'PY'
import imageio.v2 as imageio, numpy as np, glob
frames=[]
for stage in ["head_block","proportion_remap","relief_flatten","feature_primitives"]:
    fn=f"exp_output/lam_chibi/renders/chibi_v2/mesh_chibi_asian_m_{stage}.mp4"
    g=glob.glob(fn)
    if g:
        v=imageio.get_reader(g[0]); frames.append(v.get_data(0)); v.close()
if frames:
    imageio.imwrite("/tmp/chibi_v2_stages.png",np.concatenate(frames,axis=1))
    print("wrote /tmp/chibi_v2_stages.png")
PY
```

- [ ] **Step 3: Verdict — eyeball against the painter-rules doc**

Open `/tmp/chibi_v2_stages.png` and the final `sidebyside_*.mp4`. Check, against `docs/research/2026-05-15-chibi-painter-proportion-rules.md`:
- head reads as a rounded **block**, not a warped realistic skull;
- features clustered low, big forehead, rounded chin sweep;
- eyes large and **round**, not almonds or depth-pods;
- nose a **button** — no bridge, no defined tip;
- no wasp-waist fold, mesh coherent from all turntable angles.

Record the verdict (which rules pass, which need knob tuning) in a `TaskUpdate` comment on task #32. The flat-skin appearance rule (8b) is explicitly out of scope — it is the separate appearance track.

- [ ] **Step 4: Run the full test suite**

Run: `export PATH=/home/newub/miniconda3/envs/lam/bin:$PATH PYTHONPATH=/home/newub/w/LAM XFORMERS_DISABLED=1 && python -m pytest tests/test_primitives.py tests/test_chibi_metrics.py tests/test_chibi_stages.py tests/test_chibi_pipeline.py tests/test_chibi_mesh.py -v`
Expected: PASS — all pass.

- [ ] **Step 5: Commit any verdict-driven fixes**

```bash
git add -A
git commit -m "chore(chibi): full-pipeline verdict — staged chibi v2

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Self-review notes

- **Spec coverage.** HeadBlock (rule 1) → Task 3; ProportionRemap (rules 2,6,7) → Task 4; ReliefFlatten (rule 8a) → Task 6; FeaturePrimitives (rules 3,4,5) → Task 7; per-stage metrics → Task 2 + each stage's gate; `ChibiPipeline` + ARKit-agnostic rewire → Task 5; `--through` inspection → Task 5. Flat skin (8b) and hair are spec'd out of scope. ARKit-basis secant rescaling is named in the spec but not exercised here — the chibi turntable verdict is static; driven animation is a later task, so this plan does not build `deformation_secant`. That is a deliberate scope cut, noted here so it is not mistaken for a gap.
- **No placeholders.** Every stage step ships complete operator code; "tune the strength" steps are explicit render-and-adjust loops on named dataclass fields, not vague directives.
- **Type consistency.** Every stage is `(verts, [faces], [masks_path], params-dataclass) → verts`; `ChibiPipeline.run(verts, faces, through=None)`; params dataclasses `HeadBlockParams / RemapParams / ReliefParams / FeatureParams` each map 1:1 to a JSON key in `STAGE_ORDER`.
