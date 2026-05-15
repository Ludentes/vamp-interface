# Differentiable Chibi-Geometry Recipe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hand-tuned chibi deformation knots with a ~13-parameter differentiable field fitted to the painter quarter-grid via a FLAME-landmark proportion loss.

**Architecture:** A small `torch.nn.Module` (`ChibiField`) parameterizes the deformation as a monotone vertical remap + per-height radial skull scale + per-feature region transforms. `chibi/landmarks.py` computes differentiable FLAME landmark positions and holds the quarter-grid targets. `chibi/fit.py` Adam-optimizes the field params against landmark + smoothness + minimal-deformation losses (CPU, no renderer). `scripts/chibi_make_assets.py` gains a `--field_params` path that applies the fitted field and uses its exact autograd Jacobian to rescale the ARKit basis. Output is the unchanged 4-asset bundle.

**Tech Stack:** Python 3.12, PyTorch, numpy, pytest, `uv`. The fit code (`src/chibi/`) runs in the vamp `uv` env (torch + numpy only). The `chibi_make_assets.py` integration runs in the `lam` conda env because it needs `pytorch3d` for mesh subdivision.

**Spec:** `docs/superpowers/specs/2026-05-15-chibi-differentiable-geometry-recipe-design.md`

## Key paths

- Vamp repo: `/home/newub/w/vamp-interface` (`$VAMP`).
- FLAME assets (LAM repo): `/home/newub/w/LAM/model_zoo/human_parametric_models/flame_assets/flame/`
  - `head_template_mesh.obj` — 5023-vert FLAME template.
  - `flame_arkit_bs.npy` — `(52, 5023, 3)` ARKit basis.
  - `FLAME_masks.pkl` — region masks (`eye_region` 751, `left_eyeball`/`right_eyeball` 546 each, `nose` 379, `lips` 254). `pickle.load(..., encoding='latin1')`.
  - `landmark_embedding_with_eyes.npy` — `np.load(..., allow_pickle=True).item()`; dict with `full_lmk_faces_idx` `(1,70)` int64 and `full_lmk_bary_coords` `(1,70,3)` float64.
- Existing deformation script: `scripts/chibi_make_assets.py` (544 lines; hand-knob path is `T_KNOTS`/`SY_KNOTS`/`SR_KNOTS` + `deform_verts` + `rescale_arkit_bs`).

## File Structure

- **Create `src/chibi/__init__.py`** — empty package marker.
- **Create `src/chibi/field.py`** — `ChibiField` nn.Module: the deformation field + its autograd Jacobian. One responsibility: given vertices, produce deformed vertices and the per-vertex 3×3 Jacobian.
- **Create `src/chibi/landmarks.py`** — load FLAME landmark embedding; differentiable landmark positions from deformed template verts; the `QUARTER_GRID_TARGETS` constant.
- **Create `src/chibi/fit.py`** — `fit_chibi_field(...)`: the Adam optimization loop + loss terms; CLI entry that writes `chibi_field_params.json` + `loss_curve.png`.
- **Modify `scripts/chibi_make_assets.py`** — add `--field_params PATH`; when given, deform via `ChibiField` and rescale the basis via its Jacobian instead of the hand knobs.
- **Create `tests/test_chibi_field.py`**, **`tests/test_chibi_landmarks.py`**, **`tests/test_chibi_fit.py`** — pytest unit tests.

## Conventions for every task

- Run tests from `$VAMP` with `uv run pytest <path> -v` (vamp env) unless the step says otherwise.
- The FLAME landmark scheme inside the 70 `full_lmk` points follows the standard 68-point layout: jaw 0–16 (chin = 8), brows 17–26, nose 27–35 (nose tip = 30), eyes 36–47, mouth 48–67. Points 68–69 are eyeball centres — unused here.
- Normalized head axis `u`: `u = (y_crown - y) / (y_crown - y_chin)`, clamped to `[0, 1]`. `u=0` at the crown, `u=1` at the chin. `y_crown` = max y of the template; `y_chin` = y of landmark 8.
- All scales are stored as **log-scales** (`exp(param)`, identity at `param=0`) so they stay positive and identity-initialise cleanly.

---

### Task 1: ChibiField — vertical remap + radial scale

**Files:**
- Create: `src/chibi/__init__.py` (empty)
- Create: `src/chibi/field.py`
- Test: `tests/test_chibi_field.py`

- [ ] **Step 1: Write failing tests for the remap**

```python
# tests/test_chibi_field.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import torch
from chibi.field import ChibiField, REALISTIC_KNOTS


def _field():
    return ChibiField(y_crown=1.0, y_chin=0.0, z_center=0.0)


def test_identity_field_leaves_verts_unchanged():
    f = _field()  # default params = 0 → identity
    v = torch.tensor([[0.1, 0.5, 0.2], [-0.3, 0.9, 0.0]])
    out = f(v)
    assert torch.allclose(out, v, atol=1e-5)


def test_remap_is_monotone_for_random_params():
    f = _field()
    with torch.no_grad():
        f.remap_incr.copy_(torch.randn(5))
    u = torch.linspace(0, 1, 50)
    uc = f.remap(u)
    assert torch.all(uc[1:] - uc[:-1] >= -1e-6)
    assert torch.allclose(uc[0], torch.tensor(0.0), atol=1e-5)
    assert torch.allclose(uc[-1], torch.tensor(1.0), atol=1e-5)


def test_realistic_knots_span_unit_interval():
    assert REALISTIC_KNOTS[0] == 0.0 and REALISTIC_KNOTS[-1] == 1.0
    assert len(REALISTIC_KNOTS) == 6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd $VAMP && uv run pytest tests/test_chibi_field.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chibi'`.

- [ ] **Step 3: Implement the remap half of `field.py`**

```python
# src/chibi/field.py
"""Differentiable chibi deformation field.

ChibiField is a tiny nn.Module (~13 params) that bends a FLAME head into
chibi proportions. It is the learnable replacement for the hand-tuned
T/SY/SR knots in scripts/chibi_make_assets.py.

Normalized head axis u: u = (y_crown - y) / (y_crown - y_chin), clamped
to [0,1]. u=0 crown, u=1 chin. The field is the composition of:
  1. a monotone vertical remap  u -> u_chibi  (6 knots, 5 free increments)
  2. a per-height radial xz scale  r(u)        (6 log-scale knots)
  3. per-feature region similarity transforms  (applied in Task 2's caller)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

# Realistic u-positions of the 6 remap knots: crown, brow, eye, nose, mouth, chin.
REALISTIC_KNOTS = (0.0, 0.33, 0.42, 0.67, 0.80, 1.0)


def _interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """Differentiable 1-D linear interpolation (torch has no torch.interp)."""
    idx = torch.searchsorted(xp, x.clamp(xp[0], xp[-1]), right=True)
    idx = idx.clamp(1, len(xp) - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    f0, f1 = fp[idx - 1], fp[idx]
    w = (x - x0) / (x1 - x0).clamp_min(1e-8)
    return f0 + w * (f1 - f0)


class ChibiField(nn.Module):
    def __init__(self, y_crown: float, y_chin: float, z_center: float):
        super().__init__()
        self.register_buffer("y_crown", torch.tensor(float(y_crown)))
        self.register_buffer("y_chin", torch.tensor(float(y_chin)))
        self.register_buffer("z_center", torch.tensor(float(z_center)))
        self.register_buffer("realistic_knots", torch.tensor(REALISTIC_KNOTS))
        # 5 increments between the 6 knots; softplus -> positive; normalized to 1.
        self.remap_incr = nn.Parameter(torch.zeros(5))
        # per-knot radial log-scale; exp(0)=1 -> identity.
        self.radial_log = nn.Parameter(torch.zeros(6))

    def chibi_knots(self) -> torch.Tensor:
        """The 6 chibi u-positions. Monotone by construction."""
        seg = F.softplus(self.remap_incr) + 1e-4
        cum = torch.cat([torch.zeros(1), torch.cumsum(seg, 0)])
        return cum / cum[-1]

    def remap(self, u: torch.Tensor) -> torch.Tensor:
        """Map realistic u -> chibi u via the piecewise-linear monotone remap."""
        return _interp(u, self.realistic_knots, self.chibi_knots())

    def u_of(self, verts: torch.Tensor) -> torch.Tensor:
        span = (self.y_crown - self.y_chin).clamp_min(1e-6)
        return ((self.y_crown - verts[:, 1]) / span).clamp(0.0, 1.0)

    def forward(self, verts: torch.Tensor) -> torch.Tensor:
        """Apply remap + radial scale. verts: (N,3). Returns (N,3)."""
        u = self.u_of(verts)
        u_chibi = self.remap(u)
        new_y = self.y_crown - u_chibi * (self.y_crown - self.y_chin)
        r = torch.exp(_interp(u, self.realistic_knots, self.radial_log))
        new_x = verts[:, 0] * r
        new_z = self.z_center + (verts[:, 2] - self.z_center) * r
        return torch.stack([new_x, new_y, new_z], dim=1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd $VAMP && uv run pytest tests/test_chibi_field.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
cd $VAMP && git add src/chibi/__init__.py src/chibi/field.py tests/test_chibi_field.py
git commit -m "feat(chibi): ChibiField vertical remap + radial scale"
```

---

### Task 2: ChibiField — region transforms + Jacobian

**Files:**
- Modify: `src/chibi/field.py`
- Test: `tests/test_chibi_field.py`

- [ ] **Step 1: Write failing tests for region transforms and the Jacobian**

```python
# append to tests/test_chibi_field.py

def test_region_transform_scales_only_masked_verts():
    f = _field()
    with torch.no_grad():
        f.s_eye_log.copy_(torch.log(torch.tensor(1.5)))
    n = 200
    v = torch.rand(n, 3)
    # region_weights: 1.0 for the first 20 verts (the "eye"), 0 elsewhere.
    w = torch.zeros(n)
    w[:20] = 1.0
    out = f(v, region_weights={"eye": w})
    # verts fully outside the region are unchanged by the region transform.
    base = f(v)
    assert torch.allclose(out[50:], base[50:], atol=1e-5)
    # masked verts moved away from the region centroid (scale 1.5 > 1).
    assert not torch.allclose(out[:20], base[:20], atol=1e-4)


def test_local_jacobian_matches_finite_difference():
    f = _field()
    with torch.no_grad():
        f.remap_incr.copy_(torch.randn(5) * 0.3)
        f.radial_log.copy_(torch.randn(6) * 0.2)
    v = torch.tensor([[0.12, 0.55, 0.08], [-0.2, 0.3, -0.05]])
    J = f.local_jacobian(v)                       # (2,3,3)
    eps = 1e-4
    for k in range(2):
        for j in range(3):
            dv = v.clone(); dv[k, j] += eps
            num = (f(dv)[k] - f(v)[k]) / eps
            assert torch.allclose(J[k, :, j], num, atol=1e-3)


def test_local_jacobian_includes_region_transforms():
    f = _field()
    with torch.no_grad():
        f.s_eye_log.copy_(torch.log(torch.tensor(1.8)))
    n = 120
    v = torch.rand(n, 3)
    w = torch.zeros(n); w[:30] = 1.0
    rw = {"eye": w}
    J_full = f.local_jacobian(v, region_weights=rw)   # (n,3,3)
    J_glob = f.local_jacobian(v)
    # Outside the region the two agree; inside they differ (region scaling
    # is now in the Jacobian, so the eye-enlarge shows up in J).
    assert torch.allclose(J_full[60:], J_glob[60:], atol=1e-5)
    assert not torch.allclose(J_full[:30], J_glob[:30], atol=1e-3)
    # FD-check the region-inclusive Jacobian against the same fixed-centroid
    # function jacrev differentiates: global field then eye transform about a
    # detached centroid.
    _, cen = f.forward(v, rw, return_centroids=True)
    cen = cen["eye"].detach()
    s = torch.exp(f.s_eye_log.detach())               # (1,) -> isotropic eye
    eps = 1e-4
    for k in (5, 10):                                 # two in-region verts
        for j in range(3):
            dv = v.clone(); dv[k, j] += eps
            g0 = f.forward(v[k:k+1])[0]
            g1 = f.forward(dv[k:k+1])[0]
            g0 = g0 + w[k] * (g0 - cen) * (s - 1.0)
            g1 = g1 + w[k] * (g1 - cen) * (s - 1.0)
            num = (g1 - g0) / eps
            assert torch.allclose(J_full[k, :, j], num, atol=1e-3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd $VAMP && uv run pytest tests/test_chibi_field.py -v`
Expected: FAIL — `AttributeError: 'ChibiField' object has no attribute 's_eye_log'` / `local_jacobian`.

- [ ] **Step 3: Add region params, region transforms, and the Jacobian**

In `src/chibi/field.py`, add to `__init__` after `self.radial_log`:

```python
        # per-feature region log-scales (identity at 0).
        self.s_eye_log = nn.Parameter(torch.zeros(1))      # uniform eye enlarge
        self.s_nose_xy_log = nn.Parameter(torch.zeros(1))  # nose width
        self.s_nose_z_log = nn.Parameter(torch.zeros(1))   # nose depth/bridge
        self.s_mouth_y_log = nn.Parameter(torch.zeros(1))  # mouth vertical
```

Replace the `forward` signature/body to accept `region_weights` and apply region transforms after the global field:

```python
    def _region_scale_vecs(self) -> dict:
        """Per-region diagonal scale vector (sx,sy,sz)."""
        e = torch.exp(self.s_eye_log)
        nxy, nz = torch.exp(self.s_nose_xy_log), torch.exp(self.s_nose_z_log)
        my = torch.exp(self.s_mouth_y_log)
        return {
            "eye": torch.cat([e, e, e]),
            "nose": torch.cat([nxy, nxy, nz]),
            "mouth": torch.cat([torch.ones(1), my, torch.ones(1)]),
        }

    def forward(self, verts: torch.Tensor, region_weights: dict | None = None,
                return_centroids: bool = False):
        """Apply remap + radial scale, then blended region transforms.

        region_weights: {region_name: (N,) tensor in [0,1]} smooth falloff
        masks. region_name in {"eye","nose","mouth"}. None -> global field only.
        return_centroids: also return {name: (3,) centroid} actually used, so
        local_jacobian can replay the region transforms with fixed centroids.
        """
        u = self.u_of(verts)
        u_chibi = self.remap(u)
        new_y = self.y_crown - u_chibi * (self.y_crown - self.y_chin)
        r = torch.exp(_interp(u, self.realistic_knots, self.radial_log))
        new_x = verts[:, 0] * r
        new_z = self.z_center + (verts[:, 2] - self.z_center) * r
        out = torch.stack([new_x, new_y, new_z], dim=1)
        centroids = {}
        if region_weights:
            scales = self._region_scale_vecs()
            for name, w in region_weights.items():
                wsum = w.sum().clamp_min(1e-6)
                centroid = (out * w[:, None]).sum(0) / wsum
                centroids[name] = centroid
                delta = (out - centroid) * (scales[name] - 1.0)
                out = out + w[:, None] * delta
        return (out, centroids) if return_centroids else out

    def local_jacobian(self, verts: torch.Tensor,
                       region_weights: dict | None = None) -> torch.Tensor:
        """Per-vertex 3x3 Jacobian of the COMPLETE field — remap + radial +
        the per-feature region transforms — via autograd. This is the exact
        pushforward that rescales the ARKit basis: bs_chibi[v] = J[v] @ bs[v].

        Including the region transforms is load-bearing: the eye region
        enlarges the eye ~2x, so the blink/squint basis rows must be pushed
        through the eye Jacobian or the lid cannot close the enlarged eye
        (the iris-through-lid leak, rebuilt by geometry). Region centroids are
        held fixed (detached) so each vertex's J is its local linear map; the
        residual centroid coupling is O(1/M) per region and negligible.
        Returns (N,3,3)."""
        if not region_weights:
            def single(v):                   # v: (3,)
                return self.forward(v[None, :])[0]
            return torch.vmap(torch.func.jacrev(single))(verts)
        _, centroids = self.forward(verts, region_weights, return_centroids=True)
        centroids = {k: c.detach() for k, c in centroids.items()}
        scales = {k: s.detach() for k, s in self._region_scale_vecs().items()}
        names = list(region_weights.keys())
        w_stack = torch.stack([region_weights[n] for n in names], dim=1)  # (N,R)

        def single(v, wi):                   # v:(3,) wi:(R,)
            g = self.forward(v[None, :])[0]
            for j, name in enumerate(names):
                g = g + wi[j] * (g - centroids[name]) * (scales[name] - 1.0)
            return g
        return torch.vmap(torch.func.jacrev(single))(verts, w_stack)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd $VAMP && uv run pytest tests/test_chibi_field.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
cd $VAMP && git add src/chibi/field.py tests/test_chibi_field.py
git commit -m "feat(chibi): ChibiField region transforms + autograd Jacobian"
```

---

### Task 3: Landmarks and quarter-grid targets

**Files:**
- Create: `src/chibi/landmarks.py`
- Test: `tests/test_chibi_landmarks.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_chibi_landmarks.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import torch
from chibi.landmarks import (load_landmark_embedding, landmark_lines,
                             QUARTER_GRID_TARGETS, FLAME_TEMPLATE)

EMB = "/home/newub/w/LAM/model_zoo/human_parametric_models/flame_assets/flame/landmark_embedding_with_eyes.npy"


def _template_verts():
    import numpy as np
    verts = []
    for L in pathlib.Path(FLAME_TEMPLATE).read_text().splitlines():
        if L.startswith("v "):
            p = L.split()
            verts.append([float(p[1]), float(p[2]), float(p[3])])
    return torch.tensor(verts, dtype=torch.float32)


def test_embedding_loads_with_expected_shapes():
    faces_idx, bary = load_landmark_embedding(EMB)
    assert faces_idx.shape == (70,)
    assert bary.shape == (70, 3)


def test_realistic_template_landmark_lines_match_known_u():
    v = _template_verts()
    lines = landmark_lines(v)  # uses faces from FLAME template internally
    # On the undeformed FLAME head, eye line sits above the nose, nose above
    # mouth, mouth above chin (u increases downward).
    assert lines["brow"] < lines["eye"] < lines["nose"] < lines["mouth"] < lines["chin"]
    assert 0.9 < lines["chin"] <= 1.01


def test_quarter_grid_targets_present():
    for k in ("eye", "nose", "mouth"):
        assert k in QUARTER_GRID_TARGETS["lines"]
    assert abs(QUARTER_GRID_TARGETS["lines"]["eye"] - 0.50) < 1e-9
    assert abs(QUARTER_GRID_TARGETS["lines"]["mouth"] - 0.75) < 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd $VAMP && uv run pytest tests/test_chibi_landmarks.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chibi.landmarks'`.

- [ ] **Step 3: Implement `landmarks.py`**

```python
# src/chibi/landmarks.py
"""FLAME landmark positions and the chibi quarter-grid targets.
# NOTE: the disk loaders below are wrapped in functools.lru_cache so the
# fit loop (which calls landmark_positions every step) parses the template
# .obj and embedding .npy exactly once.

A landmark is a fixed barycentric point on a template triangle, so its
position is a differentiable function of the (possibly deformed) template
vertices. We use the 70-point `full_lmk` embedding; indices follow the
standard 68-point layout (chin=8, nose tip=30, eyes 36-47, brows 17-26,
mouth 48-67).
"""
from __future__ import annotations
from functools import lru_cache
import numpy as np
import torch

FLAME_TEMPLATE = ("/home/newub/w/LAM/model_zoo/human_parametric_models/"
                  "flame_assets/flame/head_template_mesh.obj")

# Landmark index groups within the 70-point full_lmk layout.
GROUPS = {"brow": list(range(17, 27)), "eye": list(range(36, 48)),
          "nose": [30], "mouth": list(range(48, 68)), "chin": [8]}

# Chibi quarter-grid targets (research doc 2026-05-15-chibi-painter-proportion-rules).
QUARTER_GRID_TARGETS = {
    "lines": {"eye": 0.50, "nose": 0.625, "mouth": 0.75},
    # feature size targets, as fractions / multipliers (see fit.py for use):
    "eye_height_u": 0.25,   # eye bbox height ~ 1/4 head
    "nose_depth_mul": 0.45, # collapse nose z-depth to 45% of original
    "mouth_height_mul": 0.55,  # compress mouth height to a strip
}


@lru_cache(maxsize=4)
def load_landmark_embedding(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (faces_idx (70,), bary (70,3))."""
    d = np.load(path, allow_pickle=True).item()
    faces_idx = np.asarray(d["full_lmk_faces_idx"]).reshape(-1).astype(np.int64)
    bary = np.asarray(d["full_lmk_bary_coords"]).reshape(-1, 3).astype(np.float64)
    return faces_idx, bary


@lru_cache(maxsize=1)
def _template_faces() -> np.ndarray:
    faces = []
    for L in open(FLAME_TEMPLATE):
        if L.startswith("f "):
            idx = [int(p.split("/")[0]) - 1 for p in L.split()[1:]]
            if len(idx) >= 3:
                faces.append(idx[:3])
    return np.asarray(faces, dtype=np.int64)


def landmark_positions(verts: torch.Tensor) -> torch.Tensor:
    """Differentiable (70,3) landmark positions from template verts (5023,3)."""
    faces_idx, bary = load_landmark_embedding(
        FLAME_TEMPLATE.replace("head_template_mesh.obj",
                               "landmark_embedding_with_eyes.npy"))
    faces = _template_faces()
    tri = torch.as_tensor(faces[faces_idx], dtype=torch.long)   # (70,3)
    b = torch.as_tensor(bary, dtype=verts.dtype)                # (70,3)
    corners = verts[tri]                                        # (70,3,3)
    return (corners * b[:, :, None]).sum(1)                     # (70,3)


def _u(y: torch.Tensor, y_crown: torch.Tensor, y_chin: torch.Tensor) -> torch.Tensor:
    return ((y_crown - y) / (y_crown - y_chin).clamp_min(1e-6))


def landmark_lines(verts: torch.Tensor) -> dict:
    """Mean u-position of each landmark group. u from this mesh's own
    crown (max y) and chin (landmark 8)."""
    lm = landmark_positions(verts)
    y_crown = verts[:, 1].max()
    y_chin = lm[8, 1]
    return {name: _u(lm[idx, 1], y_crown, y_chin).mean()
            for name, idx in GROUPS.items()}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd $VAMP && uv run pytest tests/test_chibi_landmarks.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
cd $VAMP && git add src/chibi/landmarks.py tests/test_chibi_landmarks.py
git commit -m "feat(chibi): FLAME landmark positions + quarter-grid targets"
```

---

### Task 4: Region masks and smooth falloff weights

**Files:**
- Modify: `src/chibi/landmarks.py` (add mask loading)
- Test: `tests/test_chibi_landmarks.py`

- [ ] **Step 1: Write failing tests**

```python
# append to tests/test_chibi_landmarks.py
from chibi.landmarks import region_falloff_weights

MASKS = "/home/newub/w/LAM/model_zoo/human_parametric_models/flame_assets/flame/FLAME_masks.pkl"


def test_region_falloff_weights_shape_and_range():
    v = _template_verts()
    w = region_falloff_weights(v, MASKS)
    for name in ("eye", "nose", "mouth"):
        assert w[name].shape == (5023,)
        assert float(w[name].min()) >= 0.0 and float(w[name].max()) <= 1.0 + 1e-6
        assert float(w[name].max()) > 0.9       # core of the region is ~1
        assert float(w[name].sum()) > 1.0       # region is non-empty
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd $VAMP && uv run pytest tests/test_chibi_landmarks.py::test_region_falloff_weights_shape_and_range -v`
Expected: FAIL — `ImportError: cannot import name 'region_falloff_weights'`.

- [ ] **Step 3: Implement `region_falloff_weights`**

Append to `src/chibi/landmarks.py`:

```python
import pickle

# FLAME_masks region names that compose each chibi feature region.
REGION_MASKS = {
    "eye": ["eye_region", "left_eyeball", "right_eyeball"],
    "nose": ["nose"],
    "mouth": ["lips"],
}


def region_falloff_weights(verts: torch.Tensor, masks_path: str,
                           falloff: float = 0.015) -> dict:
    """Smooth per-vertex weight in [0,1] per feature region.

    1.0 on masked verts, decaying as exp(-(d/falloff)^2) with Euclidean
    distance d to the nearest masked vert. `falloff` is in FLAME mesh units
    (the head spans ~0.3 units); 0.015 gives a ~2-ring blend band so region
    transforms fade out without tearing.
    """
    masks = pickle.load(open(masks_path, "rb"), encoding="latin1")
    v = verts.detach()
    out = {}
    for region, names in REGION_MASKS.items():
        idx = np.unique(np.concatenate([np.asarray(masks[n]) for n in names]))
        core = v[torch.as_tensor(idx, dtype=torch.long)]      # (M,3)
        d2 = torch.cdist(v, core).min(dim=1).values            # (N,)
        out[region] = torch.exp(-(d2 / falloff) ** 2)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd $VAMP && uv run pytest tests/test_chibi_landmarks.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
cd $VAMP && git add src/chibi/landmarks.py tests/test_chibi_landmarks.py
git commit -m "feat(chibi): region falloff weights from FLAME masks"
```

---

### Task 5: The fit loop

**Files:**
- Create: `src/chibi/fit.py`
- Test: `tests/test_chibi_fit.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_chibi_fit.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import torch
from chibi.fit import fit_chibi_field, chibi_loss
from chibi.field import ChibiField
from chibi.landmarks import landmark_lines, region_falloff_weights, FLAME_TEMPLATE

MASKS = "/home/newub/w/LAM/model_zoo/human_parametric_models/flame_assets/flame/FLAME_masks.pkl"


def _template_verts():
    verts = []
    for L in pathlib.Path(FLAME_TEMPLATE).read_text().splitlines():
        if L.startswith("v "):
            p = L.split(); verts.append([float(p[1]), float(p[2]), float(p[3])])
    return torch.tensor(verts, dtype=torch.float32)


def test_loss_is_lower_after_fit():
    v = _template_verts()
    field = ChibiField(y_crown=float(v[:,1].max()), y_chin=0.0, z_center=0.0)
    rw = region_falloff_weights(v, MASKS)
    before = chibi_loss(field, v, rw)["total"].item()
    fitted = fit_chibi_field(v, MASKS, n_steps=120, lr=0.05, verbose=False)
    after = chibi_loss(fitted, v, rw)["total"].item()
    assert after < before * 0.5


def test_fit_lands_eye_and_mouth_near_quarter_grid():
    v = _template_verts()
    fitted = fit_chibi_field(v, MASKS, n_steps=300, lr=0.05, verbose=False)
    rw = region_falloff_weights(v, MASKS)
    deformed = fitted(v, region_weights=rw)
    lines = landmark_lines(deformed)
    assert abs(float(lines["eye"]) - 0.50) < 0.03
    assert abs(float(lines["mouth"]) - 0.75) < 0.03
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd $VAMP && uv run pytest tests/test_chibi_fit.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chibi.fit'`.

- [ ] **Step 3: Implement `fit.py`**

```python
# src/chibi/fit.py
"""Adam fit of a ChibiField to the painter quarter-grid.

Loss = landmark proportion error + Laplacian smoothness of the displacement
field + a minimal-deformation regularizer (the identity guard). CPU, no
renderer, no detector — the targets are vertex-space landmark positions.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch

from chibi.field import ChibiField
from chibi.landmarks import (landmark_lines, landmark_positions,
                             region_falloff_weights, _template_faces,
                             QUARTER_GRID_TARGETS, GROUPS)


def _mesh_laplacian_penalty(disp: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Uniform-Laplacian smoothness of a per-vertex displacement field.

    For each vertex, penalize ||disp_i - mean(disp of 1-ring neighbours)||^2.
    """
    n = disp.shape[0]
    e = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], 0)
    e = torch.cat([e, e.flip(1)], 0)
    nbr_sum = torch.zeros_like(disp).index_add_(0, e[:, 0], disp[e[:, 1]])
    deg = torch.zeros(n).index_add_(0, e[:, 0], torch.ones(e.shape[0]))
    nbr_mean = nbr_sum / deg.clamp_min(1.0)[:, None]
    return ((disp - nbr_mean) ** 2).sum(1).mean()


def chibi_loss(field: ChibiField, verts: torch.Tensor,
               region_weights: dict, faces: torch.Tensor | None = None,
               lam_smooth: float = 1.0, lam_reg: float = 0.05) -> dict:
    """Return {'total','landmark','smooth','reg'} loss tensors."""
    if faces is None:
        faces = torch.as_tensor(_template_faces(), dtype=torch.long)
    deformed = field(verts, region_weights=region_weights)
    lines = landmark_lines(deformed)
    tgt = QUARTER_GRID_TARGETS["lines"]
    l_lm = sum((lines[k] - tgt[k]) ** 2 for k in tgt)

    # feature-size targets
    lm = landmark_positions(deformed)
    y_crown = deformed[:, 1].max()
    y_chin = lm[8, 1]
    span = (y_crown - y_chin).clamp_min(1e-6)
    eye_y = lm[GROUPS["eye"], 1]
    eye_h = (eye_y.max() - eye_y.min()) / span
    l_lm = l_lm + (eye_h - QUARTER_GRID_TARGETS["eye_height_u"]) ** 2

    l_smooth = _mesh_laplacian_penalty(deformed - verts, faces)
    l_reg = (field.remap_incr ** 2).mean() + sum(
        (p ** 2).mean() for p in [field.radial_log, field.s_eye_log,
                                  field.s_nose_xy_log, field.s_nose_z_log,
                                  field.s_mouth_y_log])
    total = l_lm + lam_smooth * l_smooth + lam_reg * l_reg
    return {"total": total, "landmark": l_lm, "smooth": l_smooth, "reg": l_reg}


def fit_chibi_field(verts: torch.Tensor, masks_path: str, *,
                    n_steps: int = 300, lr: float = 0.05,
                    verbose: bool = True) -> ChibiField:
    field = ChibiField(y_crown=float(verts[:, 1].max()),
                       y_chin=float(landmark_positions(verts)[8, 1]),
                       z_center=float(verts[:, 2].mean()))
    rw = region_falloff_weights(verts, masks_path)
    faces = torch.as_tensor(_template_faces(), dtype=torch.long)
    opt = torch.optim.Adam(field.parameters(), lr=lr)
    history = []
    for step in range(n_steps):
        opt.zero_grad()
        loss = chibi_loss(field, verts, rw, faces)
        loss["total"].backward()
        opt.step()
        history.append(loss["total"].item())
        if verbose and step % 50 == 0:
            print(f"step {step:4d}  total={loss['total'].item():.5f}  "
                  f"lm={loss['landmark'].item():.5f}")
    field._loss_history = history
    return field


def save_field_params(field: ChibiField, path: str) -> None:
    """Write fitted params + frame buffers to JSON."""
    d = {"remap_incr": field.remap_incr.detach().tolist(),
         "radial_log": field.radial_log.detach().tolist(),
         "s_eye_log": field.s_eye_log.detach().tolist(),
         "s_nose_xy_log": field.s_nose_xy_log.detach().tolist(),
         "s_nose_z_log": field.s_nose_z_log.detach().tolist(),
         "s_mouth_y_log": field.s_mouth_y_log.detach().tolist(),
         "y_crown": float(field.y_crown), "y_chin": float(field.y_chin),
         "z_center": float(field.z_center)}
    Path(path).write_text(json.dumps(d, indent=2))


def load_field_params(path: str) -> ChibiField:
    d = json.loads(Path(path).read_text())
    field = ChibiField(d["y_crown"], d["y_chin"], d["z_center"])
    with torch.no_grad():
        field.remap_incr.copy_(torch.tensor(d["remap_incr"]))
        field.radial_log.copy_(torch.tensor(d["radial_log"]))
        field.s_eye_log.copy_(torch.tensor(d["s_eye_log"]))
        field.s_nose_xy_log.copy_(torch.tensor(d["s_nose_xy_log"]))
        field.s_nose_z_log.copy_(torch.tensor(d["s_nose_z_log"]))
        field.s_mouth_y_log.copy_(torch.tensor(d["s_mouth_y_log"]))
    return field
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd $VAMP && uv run pytest tests/test_chibi_fit.py -v`
Expected: 2 passed. (Each fit takes a few seconds on CPU.)

- [ ] **Step 5: Commit**

```bash
cd $VAMP && git add src/chibi/fit.py tests/test_chibi_fit.py
git commit -m "feat(chibi): differentiable fit loop + param save/load"
```

---

### Task 6: CLI entry for the fit

**Files:**
- Modify: `src/chibi/fit.py` (add `__main__`)

- [ ] **Step 1: Add a CLI block to `fit.py`**

Append to `src/chibi/fit.py`:

```python
def _load_obj_verts(path: str) -> torch.Tensor:
    verts = []
    for L in Path(path).read_text().splitlines():
        if L.startswith("v "):
            p = L.split()
            verts.append([float(p[1]), float(p[2]), float(p[3])])
    return torch.tensor(verts, dtype=torch.float32)


if __name__ == "__main__":
    import argparse
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description="Fit a ChibiField to the quarter-grid.")
    ap.add_argument("--template", required=True, help="5023-vert FLAME template .obj")
    ap.add_argument("--masks", required=True, help="FLAME_masks.pkl")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n_steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    verts = _load_obj_verts(args.template)
    field = fit_chibi_field(verts, args.masks, n_steps=args.n_steps, lr=args.lr)
    save_field_params(field, str(out / "chibi_field_params.json"))

    plt.figure()
    plt.plot(field._loss_history)
    plt.xlabel("step"); plt.ylabel("total loss"); plt.yscale("log")
    plt.title("ChibiField fit")
    plt.savefig(out / "loss_curve.png", dpi=110, bbox_inches="tight")

    rw = region_falloff_weights(verts, args.masks)
    lines = landmark_lines(field(verts, region_weights=rw))
    print("fitted landmark lines:",
          {k: round(float(v), 4) for k, v in lines.items()})
    print(f"wrote {out/'chibi_field_params.json'} and {out/'loss_curve.png'}")
```

- [ ] **Step 2: Run the CLI to verify it produces artifacts**

Run:
```bash
cd $VAMP && PYTHONPATH=src uv run --with matplotlib python -m chibi.fit \
  --template /home/newub/w/LAM/model_zoo/human_parametric_models/flame_assets/flame/head_template_mesh.obj \
  --masks /home/newub/w/LAM/model_zoo/human_parametric_models/flame_assets/flame/FLAME_masks.pkl \
  --outdir exp_output/lam_chibi/diff_geometry
```
Expected: prints fitted landmark lines with `eye≈0.50`, `mouth≈0.75`, `nose≈0.62`; writes `chibi_field_params.json` + `loss_curve.png` under `exp_output/lam_chibi/diff_geometry/`. (`PYTHONPATH=src` puts the `chibi` package on the path for `-m`; `--with matplotlib` supplies the loss-curve plot dependency, which is not in `pyproject.toml`.)

- [ ] **Step 3: Commit**

```bash
cd $VAMP && git add src/chibi/fit.py
git commit -m "feat(chibi): CLI entry for the chibi field fit"
```

---

### Task 7: Wire the fitted field into `chibi_make_assets.py`

**Files:**
- Modify: `scripts/chibi_make_assets.py`

**Context:** `chibi_make_assets.py` currently deforms verts with `deform_verts(...)` (hand knobs) and rescales the basis with `rescale_arkit_bs(...)` (hand `diag()`). This task adds a `--field_params` branch that uses `ChibiField` for both. The legacy path stays the default. This script runs in the **lam conda env** (`pytorch3d`); `ChibiField` only needs torch, which the lam env has.

- [ ] **Step 1: Add the `--field_params` argument**

In `scripts/chibi_make_assets.py`, in `main()` after the `--radial_strength` argument block, add:

```python
    ap.add_argument("--field_params", type=str, default=None,
                    help="Path to a chibi_field_params.json fitted by "
                         "chibi.fit. When given, deformation + basis rescaling "
                         "use the fitted ChibiField instead of the hand-tuned "
                         "T/SY/SR knots. The legacy knob path is the default.")
```

- [ ] **Step 2: Add a field-based deformation helper**

Near the top of `scripts/chibi_make_assets.py`, after the imports, add:

```python
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def deform_with_field(verts: np.ndarray, masks_path: str, field_params: str
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Deform verts with a fitted ChibiField. Returns (new_verts, jacobian).

    new_verts: (N,3) float64. jacobian: (N,3,3) float64 — the full-field
    Jacobian (remap + radial + region transforms), used to rescale the ARKit
    basis (replaces diag(s_r,s_y,s_r)). Including the region transforms is
    load-bearing: the enlarged-eye blink must be pushed through the eye
    Jacobian or the lid cannot close the chibi eye.
    The xyz columns (cols 0:3) are deformed; any rgb columns 3:6 pass through.
    """
    import torch
    from chibi.fit import load_field_params
    from chibi.landmarks import region_falloff_weights

    field = load_field_params(field_params)
    xyz = torch.as_tensor(verts[:, :3], dtype=torch.float32)
    rw = region_falloff_weights(xyz, masks_path)
    with torch.no_grad():
        deformed = field(xyz, region_weights=rw).double().numpy()
    jac = field.local_jacobian(xyz, region_weights=rw).detach().double().numpy()
    return deformed, jac
```

**Note on the 20018 baked mesh:** `deform_with_field` calls `region_falloff_weights`, which addresses the FLAME masks (5023-indexed) into the passed vert array. This is exact for the 5023 template. For the 20018 baked mesh it is correct **only if LAM's mesh subdivision preserves the original vertex indices** in the first 5023 slots (then `masks[n]` still points at eye/nose/lip verts and the distance-based falloff covers the new subdivision verts). Loop/midpoint subdivision usually does preserve original indices — verify this in Step 6 by checking the deformed baked `.obj` eye region visually matches the deformed template. If it does not, lift the region weights from 5023→20018 by the same `u`-fraction + nearest-template-vert mapping the script already uses for the field, rather than re-indexing the masks directly.

- [ ] **Step 3: Add a Jacobian-based basis rescaler**

After `rescale_arkit_bs` in `scripts/chibi_make_assets.py`, add:

```python
def rescale_arkit_bs_jacobian(arkit_bs: np.ndarray, jac: np.ndarray) -> np.ndarray:
    """Rescale each blendshape delta by the field's per-vertex Jacobian.

    arkit_bs: (52,5023,3). jac: (5023,3,3). For vert i the deformed delta is
    J_i @ delta_i — the exact linearization of the field, replacing the
    hand-derived diagonal scale. Returns (52,5023,3) float64.
    """
    bs = arkit_bs.astype(np.float64)
    # einsum: for each (blendshape b, vert v): J[v] @ bs[b,v]
    return np.einsum("vij,bvj->bvi", jac, bs)
```

- [ ] **Step 4: Branch the deformation in `main()`**

In `chibi_make_assets.py` `main()`, find where the template and baked meshes are deformed via `deform_verts(...)` and where `rescale_arkit_bs(...)` is called. Wrap both in a branch. The fitted-field branch:

```python
    if args.field_params:
        # Fitted-field path. Template (5023) drives the basis Jacobian;
        # the baked mesh (20018) is deformed by the same field.
        tpl_new, tpl_jac = deform_with_field(
            tpl_verts, args.flame_masks, args.field_params)
        baked_new_xyz, _ = deform_with_field(
            baked_verts, args.flame_masks, args.field_params)
        baked_new = baked_verts.copy()
        baked_new[:, :3] = baked_new_xyz
        arkit_rescaled = rescale_arkit_bs_jacobian(arkit_bs, tpl_jac)
        t_per_vert = None  # not used on this path
    else:
        # ... existing legacy knob path unchanged ...
```

Keep every existing legacy statement inside the `else:` branch. (Variable names `tpl_verts`, `baked_verts`, `arkit_bs`, `tpl_new`, `baked_new`, `arkit_rescaled` must match those already in `main()`; if the existing code uses different names, rename within this branch to match — do not rename the legacy variables.)

- [ ] **Step 5: Verify the legacy path is unchanged**

Run (lam conda env, legacy path — no `--field_params`):
```bash
cd $VAMP && PATH=/home/newub/miniconda3/envs/lam/bin:$PATH PYTHONPATH=src \
  python scripts/chibi_make_assets.py \
  --template /home/newub/w/LAM/model_zoo/human_parametric_models/flame_assets/flame/head_template_mesh.obj \
  --arkit_bs /home/newub/w/LAM/model_zoo/human_parametric_models/flame_assets/flame_arkit_bs.npy \
  --baked exp_output/lam_blender_handoff/splats/asian_m_textured_mesh.obj \
  --outdir /tmp/chibi_legacy_check --chibi_strength 1.0
```
Expected: emits the 4 assets under `/tmp/chibi_legacy_check/` exactly as before (no crash; same filenames as a pre-change run).

- [ ] **Step 6: Verify the fitted-field path emits the 4 assets**

Run (lam conda env, fitted path):
```bash
cd $VAMP && PATH=/home/newub/miniconda3/envs/lam/bin:$PATH PYTHONPATH=src \
  python scripts/chibi_make_assets.py \
  --template /home/newub/w/LAM/model_zoo/human_parametric_models/flame_assets/flame/head_template_mesh.obj \
  --arkit_bs /home/newub/w/LAM/model_zoo/human_parametric_models/flame_assets/flame_arkit_bs.npy \
  --baked exp_output/lam_blender_handoff/splats/asian_m_textured_mesh.obj \
  --outdir /tmp/chibi_field_check \
  --field_params exp_output/lam_chibi/diff_geometry/chibi_field_params.json
```
Expected: emits the deformed template `.obj`, deformed baked `.obj`, rescaled `arkit_bs` `.npy` `(52,5023,3)`, and `chibi_scale_ratio.npy` under `/tmp/chibi_field_check/`.

- [ ] **Step 7: Commit**

```bash
cd $VAMP && git add scripts/chibi_make_assets.py
git commit -m "feat(chibi): chibi_make_assets --field_params fitted-field path"
```

---

### Task 8: Verdict render + docs

**Files:**
- Create: `docs/research/2026-05-15-chibi-diff-geometry-spike.md`
- Modify: `docs/research/_topics/lam-chibi-recipe.md`

- [ ] **Step 1: Render the fitted chibi head**

Use the existing render harness on the fitted assets. Run:
```bash
cd $VAMP && PATH=/home/newub/miniconda3/envs/lam/bin:$PATH PYTHONPATH=src \
  bash scripts/chibi_render_with_assets.sh /tmp/chibi_field_check
```
(If `chibi_render_with_assets.sh` takes its asset dir differently, read its first 30 lines and pass the dir as that script expects.)
Expected: a render PNG/MP4 under `/tmp/chibi_field_check/renders/` showing the fitted-geometry chibi.

- [ ] **Step 2: Build a side-by-side vs the hand-knob chibi**

Run:
```bash
cd $VAMP && uv run python -c "
from PIL import Image
import sys
a = Image.open('exp_output/lam_chibi/renders/v2_three_moment_gate.png')
b = Image.open(sorted(__import__('glob').glob('/tmp/chibi_field_check/renders/*.png'))[0])
h = min(a.height, b.height)
a = a.resize((int(a.width*h/a.height), h)); b = b.resize((int(b.width*h/b.height), h))
c = Image.new('RGB', (a.width+b.width, h), 'white')
c.paste(a,(0,0)); c.paste(b,(a.width,0))
c.save('exp_output/lam_chibi/diff_geometry/sidebyside_handknob_vs_fitted.png')
print('wrote sidebyside')
"
```
Expected: writes `sidebyside_handknob_vs_fitted.png` (hand-knob left, fitted right).

- [ ] **Step 3: Write the spike write-up**

Create `docs/research/2026-05-15-chibi-diff-geometry-spike.md`:

```markdown
---
status: live
topic: lam-chibi-recipe
---

# Chibi Differentiable-Geometry Spike — Result

**Date:** 2026-05-15

## What shipped

A ~13-parameter `ChibiField` (`src/chibi/`) fitted by Adam to the painter
quarter-grid (`docs/research/2026-05-15-chibi-painter-proportion-rules.md`).
Replaces the hand-tuned T/SY/SR knots in `chibi_make_assets.py` via the new
`--field_params` path. No renderer or face detector in the fit — targets are
FLAME landmark vertex positions.

## Fitted result

[Fill in from the Task 6 CLI print: the fitted landmark lines dict, and
whether eye/mouth/nose landed within ±0.03 of 0.50 / 0.75 / 0.625.]

## Verdict render

`exp_output/lam_chibi/diff_geometry/sidebyside_handknob_vs_fitted.png` —
hand-knob chibi (left) vs fitted-field chibi (right). [Fill in: which reads
more chibi-correct, and whether the iris-through-lid leak is better/worse/
unchanged before Stage 2.]

## Next

Stage 2 — the render-space iris-leak fix (`2026-05-15-chibi-differentiable-
leak-fix-plan.md`) runs unchanged on the fitted assets.
```

Replace the two `[Fill in ...]` blocks with the actual observed values and a one-paragraph eyeball verdict.

- [ ] **Step 4: Update the topic index**

In `docs/research/_topics/lam-chibi-recipe.md`, under "Load-bearing dated docs", add a bullet:

```markdown
- [`2026-05-15-chibi-diff-geometry-spike.md`](../2026-05-15-chibi-diff-geometry-spike.md) — **Most recent.** Differentiable ChibiField (`src/chibi/`) fitted to the painter quarter-grid replaces the hand-tuned T/SY/SR knots; `chibi_make_assets.py --field_params`. Stage 1 (geometry); the diff-leak spike is Stage 2.
```

And in the "Asset surface" section, add a row:

```markdown
| `--field_params` | `chibi_field_params.json` | Fitted ChibiField params (Stage-1 geometry) | `chibi.fit` |
```

- [ ] **Step 5: Commit**

```bash
cd $VAMP && git add docs/research/2026-05-15-chibi-diff-geometry-spike.md docs/research/_topics/lam-chibi-recipe.md
git commit -m "docs(chibi): diff-geometry spike write-up + topic index"
```

---

## Final verification

After all tasks: from `$VAMP`, run `uv run pytest tests/test_chibi_field.py tests/test_chibi_landmarks.py tests/test_chibi_fit.py -v` → all pass. Confirm `chibi_make_assets.py` emits 4 assets on both the legacy and `--field_params` paths. Confirm the side-by-side render exists and the spike doc's `[Fill in]` blocks are replaced with real observations.
