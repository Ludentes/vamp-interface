# Chibi Identity-in-Texture Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render any job anchor as a drivable chibi by baking a Flux chibi portrait onto the UV of a fixed ARKit-rigged chibi mesh — identity in the texture, geometry/rig constant.

**Architecture:** A fixed canonical chibi mesh (Koban Chibi Base Mesh, ships native ARKit-52). Per anchor: Flux generates one flat-lit frontal chibi portrait; it is landmark-registered and TPS-warped onto the canonical mesh's frontal projection; a single-view projective bake (existing `src/chibi/` code) writes it into the UV atlas; mirror + dilate + skin-fallback fill non-frontal texels. The textured mesh is driven by its native ARKit-52 shape keys.

**Tech Stack:** Python 3.12 / uv; Flux + PuLID (existing project pipeline); insightface (landmarks); nvdiffrast (UV rasterise); pytorch3d `TexturesUV` (render); existing `src/chibi/` bake modules; headless Blender 4.0.2 for mesh export.

**Spec:** `docs/superpowers/specs/2026-05-18-chibi-identity-texture-design.md`

---

## Phase ordering and gates

- **Task 0 is a hard gate** — no rendering/bake work until Koban licensing (R0) is confirmed by the user.
- **Tasks 2, 3 (spikes S2, S1)** decide feasibility. If S1 (Task 3) shows TPS registration is insufficient, **Task 8** (inverse-render refinement) is pulled in; otherwise Task 8 is skipped.
- Local evaluation (running the spikes for knowledge) does not require R0 — R0 gates *shipping/product use*, not research inspection. But order Task 0 first anyway so the licensing answer is on record before substantial code lands.

---

### Task 0: Licensing gate (R0)

**Files:** none (decision gate).

- [ ] **Step 1: Confirm Koban usage rights**

Ask the user to confirm permitted use of `exp_output/chibi_meshes/koban/Koban Chibi Base Mesh 1.0.blend` (paid Gumroad asset, `READ ME.txt` states no explicit license). Record the answer in the chibi topic index (`docs/research/_topics/lam-chibi-recipe.md`) under a new "Licensing" line.

Expected: an explicit "OK for research/internal visualisation" or "swap to a different ARKit-rigged chibi mesh". If swap: the plan is unchanged except the asset path in Task 1.

- [ ] **Step 2: Commit the recorded decision**

```bash
git add docs/research/_topics/lam-chibi-recipe.md
git commit -m "docs(chibi): record Koban licensing decision"
```

---

### Task 1: Canonical-mesh asset prep

**Files:**
- Create: `scripts/koban_prep.py` (headless Blender export script)
- Create: `src/chibi/koban_asset.py`
- Test: `tests/test_koban_asset.py`

The Koban `.blend` cannot be parsed by Python directly; export to OBJ via headless Blender. `scripts/koban_prep.py` runs *inside* Blender (`blender -b --python`). `koban_asset.py` loads the exported artifacts for the rest of the pipeline.

**Canonical source = `Koban Chibi Base Mesh VRM export.blend`** (verified 2026-05-18). The base `Koban Chibi Base Mesh 1.0.blend` mesh has a degenerate UV layer (only 53 UV coords, the body unmapped) — useless for texture baking. The VRM-export `.blend` has the same `Chibi Base Mesh` object (5662 verts, 10980 tris) with a proper active UV unwrap (`UVMap`, 32940 loops, spanning x∈[0,0.5] y∈[0,0.672] — left-half, symmetry-mirrored) AND all 61 shape keys including the full camelCase ARKit-52. Export from this file.

- [ ] **Step 1: Write the Blender export script**

`scripts/koban_prep.py` — note the conda site-packages append (Blender omits it; see `/tmp/inspect_mesh.py` precedent).

The Koban mesh ships **no usable UV** (verified: 26 unique coords mesh-wide). This script *generates* a frontal-projection unwrap directly in Python — no Blender viewport operator (`project_from_view`/`smart_project` need a 3D region, unavailable headless). Algorithm: the *face region* is every polygon touching a vertex displaced (> ε) by any ARKit-52 shape key vs `Basis` (the ARKit basis only moves face geometry — a robust rig-derived mask). Face polygons get UV = frontal orthographic projection (`(x, z)` normalised by the face bbox into `[0, 0.95]²`); every non-face loop collapses to one reserved skin texel near `(0.98, 0.98)`.

```python
"""Run inside Blender: blender -b <koban.blend> --python scripts/koban_prep.py -- <outdir>
Generates a frontal-projection UV unwrap for the Koban chibi mesh (it ships
none), exports it to OBJ, and dumps the shape-key names to arkit_keys.json."""
import sys, os, json
sys.path.append("/home/newub/miniconda3/lib/python3.12/site-packages")
import bpy

# The 52 ARKit blendshape names — used to derive the face-region mask.
ARKIT_52 = (
    "eyeBlinkLeft", "eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft",
    "eyeLookUpLeft", "eyeSquintLeft", "eyeWideLeft", "eyeBlinkRight",
    "eyeLookDownRight", "eyeLookInRight", "eyeLookOutRight", "eyeLookUpRight",
    "eyeSquintRight", "eyeWideRight", "jawForward", "jawLeft", "jawRight",
    "jawOpen", "mouthClose", "mouthFunnel", "mouthPucker", "mouthLeft",
    "mouthRight", "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft",
    "mouthFrownRight", "mouthDimpleLeft", "mouthDimpleRight", "mouthStretchLeft",
    "mouthStretchRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower",
    "mouthShrugUpper", "mouthPressLeft", "mouthPressRight", "mouthLowerDownLeft",
    "mouthLowerDownRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "noseSneerLeft", "noseSneerRight", "tongueOut",
)
EPS = 1e-4          # vertex-displacement threshold for "is face vertex"
FACE_RECT = 0.95    # face UV island occupies [0, FACE_RECT]^2
SKIN_TEXEL = 0.98   # non-face loops collapse to this UV

argv = sys.argv[sys.argv.index("--") + 1:]
outdir = argv[0]
os.makedirs(outdir, exist_ok=True)

obj = bpy.data.objects["Chibi Base Mesh"]
me = obj.data
kb = me.shape_keys.key_blocks

with open(os.path.join(outdir, "arkit_keys.json"), "w") as f:
    json.dump([k.name for k in kb], f, indent=2)

# --- face mask: verts displaced by any ARKit-52 key relative to Basis ---
basis = kb["Basis"].data
nv = len(me.vertices)
is_face_vert = [False] * nv
for name in ARKIT_52:
    if name not in kb:
        continue
    skd = kb[name].data
    for i in range(nv):
        if (skd[i].co - basis[i].co).length > EPS:
            is_face_vert[i] = True
face_poly = {p.index for p in me.polygons
             if any(is_face_vert[v] for v in p.vertices)}

# --- frontal bbox of face verts (x = left-right, z = up; y = depth) ---
fvi = [i for i in range(nv) if is_face_vert[i]]
xs = [me.vertices[i].co.x for i in fvi]
zs = [me.vertices[i].co.z for i in fvi]
xmin, zmin = min(xs), min(zs)
xr = (max(xs) - xmin) or 1.0
zr = (max(zs) - zmin) or 1.0

# --- write the UV layer ---
me.uv_layers.clear()
uvl = me.uv_layers.new(name="UVMap")
for p in me.polygons:
    is_face = p.index in face_poly
    for li in range(p.loop_start, p.loop_start + p.loop_total):
        if is_face:
            co = me.vertices[me.loops[li].vertex_index].co
            uvl.data[li].uv = ((co.x - xmin) / xr * FACE_RECT,
                               (co.z - zmin) / zr * FACE_RECT)
        else:
            uvl.data[li].uv = (SKIN_TEXEL, SKIN_TEXEL)
me.update()

# --- export ---
bpy.ops.object.select_all(action="DESELECT")
obj.select_set(True)
bpy.context.view_layer.objects.active = obj
bpy.ops.wm.obj_export(filepath=os.path.join(outdir, "koban.obj"),
                      export_selected_objects=True, export_uv=True,
                      export_normals=True, export_materials=False,
                      apply_modifiers=False)
print(f"[koban_prep] face polys {len(face_poly)}/{len(me.polygons)}; "
      f"exported koban.obj + arkit_keys.json to {outdir}")
```

The frontal projection uses Blender axes `x` (left-right) and `z` (up), dropping `y` (depth). If the chibi faces `-y`, the face UV will be left-right mirrored — harmless for the bake (TPS registration absorbs it) but note it; the implementer should render once and confirm orientation.

- [ ] **Step 2: Run the export**

```bash
blender -b "exp_output/chibi_meshes/koban/Koban Chibi Base Mesh VRM export.blend" \
  --python scripts/koban_prep.py -- exp_output/chibi_meshes/koban_canonical
```
Expected: `koban_canonical/koban.obj` and `arkit_keys.json` created; stdout `[koban_prep] exported ...`. The OBJ exporter writes the *active* UV layer (`UVMap`) — confirm it is active before export.

- [ ] **Step 3: Write the failing test for `koban_asset.py`**

`tests/test_koban_asset.py`:

```python
import json, pathlib
from chibi.koban_asset import load_koban, ARKIT_52

CANON = pathlib.Path("exp_output/chibi_meshes/koban_canonical")

def test_koban_obj_parses_with_uv():
    k = load_koban(CANON)
    assert k.verts.shape[1] == 3 and k.faces.shape[1] == 3
    assert k.uv.shape[1] == 2 and k.uv_faces.shape == k.faces.shape
    assert 0.0 <= float(k.uv.min()) and float(k.uv.max()) <= 1.0001

def test_arkit_52_present():
    keys = set(json.loads((CANON / "arkit_keys.json").read_text()))
    missing = [n for n in ARKIT_52 if n not in keys]
    assert not missing, f"missing ARKit keys: {missing}"
```

- [ ] **Step 4: Run test to verify it fails**

Run: `uv run pytest tests/test_koban_asset.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chibi.koban_asset'`.

- [ ] **Step 5: Implement `koban_asset.py`**

```python
"""Load the exported Koban canonical mesh: a UV-textured triangle mesh plus
the ARKit-52 verification list. UV layout reuses FlameUV — it is topology-
agnostic (vt + f v/vt)."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import torch
from chibi.uv_template import load_flame_uv, FlameUV

# The 52 ARKit blendshape names (the canonical mesh must expose all of them).
ARKIT_52 = (
    "eyeBlinkLeft", "eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft",
    "eyeLookUpLeft", "eyeSquintLeft", "eyeWideLeft", "eyeBlinkRight",
    "eyeLookDownRight", "eyeLookInRight", "eyeLookOutRight", "eyeLookUpRight",
    "eyeSquintRight", "eyeWideRight", "jawForward", "jawLeft", "jawRight",
    "jawOpen", "mouthClose", "mouthFunnel", "mouthPucker", "mouthLeft",
    "mouthRight", "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft",
    "mouthFrownRight", "mouthDimpleLeft", "mouthDimpleRight", "mouthStretchLeft",
    "mouthStretchRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower",
    "mouthShrugUpper", "mouthPressLeft", "mouthPressRight", "mouthLowerDownLeft",
    "mouthLowerDownRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "noseSneerLeft", "noseSneerRight", "tongueOut",
)


@dataclass
class KobanMesh:
    verts: torch.Tensor      # (V,3) float32
    faces: torch.Tensor      # (F,3) int64
    uv: torch.Tensor         # (Nvt,2) float32
    uv_faces: torch.Tensor   # (F,3) int64


def _count_obj_faces(obj_path: Path) -> int:
    return sum(1 for ln in obj_path.read_text().splitlines()
               if ln.startswith("f "))


def load_koban(canon_dir: str | Path) -> KobanMesh:
    """Load koban.obj from the canonical dir. Parses verts itself; reuses
    load_flame_uv for the UV layout (pass the mesh's own face count)."""
    canon = Path(canon_dir)
    obj = canon / "koban.obj"
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    for ln in obj.read_text().splitlines():
        p = ln.split()
        if not p:
            continue
        if p[0] == "v":
            verts.append([float(p[1]), float(p[2]), float(p[3])])
        elif p[0] == "f":
            faces.append([int(t.split("/")[0]) - 1 for t in p[1:4]])
    fuv: FlameUV = load_flame_uv(str(obj), n_faces=_count_obj_faces(obj))
    return KobanMesh(
        verts=torch.tensor(verts, dtype=torch.float32),
        faces=torch.tensor(faces, dtype=torch.int64),
        uv=fuv.uv, uv_faces=fuv.uv_faces,
    )
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_koban_asset.py -v`
Expected: PASS. If `test_koban_obj_parses_with_uv` fails on the UV-range assert, Koban UVs tile outside [0,1] — handle by `uv = uv % 1.0` in `load_koban` and note it; if `test_arkit_52_present` fails, the OBJ export dropped shape keys (expected — OBJ has no shape keys; the test only checks `arkit_keys.json`, which is dumped pre-export, so this should pass).

- [ ] **Step 7: Define the frontal View + landmark vertices**

Create `exp_output/chibi_meshes/koban_canonical/view.json` (frontal `View`: azim 0, elev 0, a `dist` that frames the head — derive from mesh bbox) and `landmarks.json` (~15 vertex indices for eye corners, brows, nose tip, mouth corners, chin). Pick the landmark indices by rendering the mesh with `chibi.mesh_render.render` and visually identifying vertices, or by nearest-vertex to known 3D anatomical points. Document the picking method inline in `koban_asset.py` and add `load_koban_view(canon_dir)` and `load_koban_landmarks(canon_dir)` loaders.

- [ ] **Step 8: Commit**

```bash
git add scripts/koban_prep.py src/chibi/koban_asset.py tests/test_koban_asset.py \
  exp_output/chibi_meshes/koban_canonical/view.json \
  exp_output/chibi_meshes/koban_canonical/landmarks.json
git commit -m "feat(chibi): Koban canonical-mesh asset prep + loader"
```
(Do **not** commit `koban.obj` / the `.blend` — large/asset files; add `koban.obj` to `.gitignore` if not already covered by `exp_output/`.)

---

### Task 2: Spike S2 — canonical rig drives cleanly

**Files:** Create `scripts/koban_rig_test.py` (Blender script).

Investigation task — confirms the native ARKit-52 shape keys deform without broken geometry. No new repo code beyond the throwaway script.

- [ ] **Step 1: Write the rig-test Blender script**

`scripts/koban_rig_test.py` — opens the `.blend`, sets selected shape keys to 1.0, renders front views:

```python
"""blender -b <koban.blend> --python scripts/koban_rig_test.py -- <outdir>"""
import sys, os
sys.path.append("/home/newub/miniconda3/lib/python3.12/site-packages")
import bpy

outdir = sys.argv[sys.argv.index("--") + 1:][0]
os.makedirs(outdir, exist_ok=True)
obj = bpy.data.objects["Chibi Base Mesh"]
kb = obj.data.shape_keys.key_blocks

tests = {"neutral": [], "blink": ["eyeBlinkLeft", "eyeBlinkRight"],
         "jawOpen": ["jawOpen"],
         "smile": ["mouthSmileLeft", "mouthSmileRight"]}
scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
for label, keys in tests.items():
    for k in kb:
        k.value = 0.0
    for name in keys:
        kb[name].value = 1.0
    scene.render.filepath = os.path.join(outdir, f"rig_{label}.png")
    bpy.ops.render.render(write_still=True)
    print(f"[rig_test] rendered {label}")
```

- [ ] **Step 2: Run it**

```bash
blender -b "exp_output/chibi_meshes/koban/Koban Chibi Base Mesh VRM export.blend" \
  --python scripts/koban_rig_test.py -- exp_output/chibi_meshes/renders/rig_test
```
Expected: 4 PNGs. **Inspect them.** Acceptance: blink closes both eyes, jawOpen drops the jaw, smile raises mouth corners, no torn/exploded geometry.

- [ ] **Step 3: Record the verdict**

Append the S2 result to the chibi topic index. If the rig is broken, **stop the plan** — escalate to the user; the whole approach depends on a working native rig.

- [ ] **Step 4: Commit**

```bash
git add scripts/koban_rig_test.py docs/research/_topics/lam-chibi-recipe.md
git commit -m "spike(chibi): S2 — Koban ARKit-52 rig-drive verdict"
```

---

### Task 3: `register.py` — portrait → canonical registration (spike S1 core)

**Files:**
- Create: `src/chibi/register.py`
- Test: `tests/test_register.py`

- [ ] **Step 1: Write the failing test**

`tests/test_register.py` — a synthetic affine must be recovered by the TPS fit:

```python
import torch
from chibi.register import fit_tps, warp_image

def test_tps_recovers_affine():
    # 12 source points; target = known affine of source.
    torch.manual_seed(0)
    src = torch.rand(12, 2) * 200 + 28
    A = torch.tensor([[1.1, 0.05], [-0.03, 0.95]])
    b = torch.tensor([10.0, -6.0])
    dst = src @ A.T + b
    tps = fit_tps(src, dst)
    pred = tps(src)
    assert torch.allclose(pred, dst, atol=1e-3), (pred - dst).abs().max()

def test_warp_image_shape_preserved():
    img = torch.rand(256, 256, 3)
    src = torch.rand(8, 2) * 200 + 28
    dst = src + torch.randn(8, 2) * 3.0
    tps = fit_tps(src, dst)
    out = warp_image(img, tps)
    assert out.shape == img.shape
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_register.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chibi.register'`.

- [ ] **Step 3: Implement `register.py`**

```python
"""Register a Flux portrait to the canonical mesh's frontal projection.

Detects 2D face landmarks on the portrait (insightface), projects the canonical
mesh's annotated landmark vertices through the frontal View to 2D, fits a
thin-plate-spline warp portrait->canonical, and applies it. The warped portrait
is then ready for the single-view UV bake.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F


def _tps_kernel(r2: torch.Tensor) -> torch.Tensor:
    """U(r) = r^2 log r, evaluated stably from squared distance r2."""
    return 0.5 * r2 * torch.log(r2.clamp_min(1e-12))


@dataclass
class TPS:
    ctrl: torch.Tensor    # (K,2) source control points
    w: torch.Tensor       # (K,2) non-affine weights
    a: torch.Tensor       # (3,2) affine part

    def __call__(self, pts: torch.Tensor) -> torch.Tensor:
        """Map (N,2) source points -> (N,2) target points."""
        d2 = ((pts[:, None, :] - self.ctrl[None, :, :]) ** 2).sum(-1)
        U = _tps_kernel(d2)                       # (N,K)
        ones = torch.ones(pts.shape[0], 1)
        P = torch.cat([ones, pts], dim=1)         # (N,3)
        return U @ self.w + P @ self.a


def fit_tps(src: torch.Tensor, dst: torch.Tensor,
            reg: float = 0.0) -> TPS:
    """Fit a TPS mapping src (K,2) -> dst (K,2). `reg` relaxes exact
    interpolation (0 = exact)."""
    src = src.to(torch.float64)
    dst = dst.to(torch.float64)
    K = src.shape[0]
    d2 = ((src[:, None, :] - src[None, :, :]) ** 2).sum(-1)
    Kmat = _tps_kernel(d2) + reg * torch.eye(K, dtype=torch.float64)
    P = torch.cat([torch.ones(K, 1, dtype=torch.float64), src], dim=1)  # (K,3)
    # Solve [[K, P],[P^T, 0]] [w;a] = [dst;0]
    top = torch.cat([Kmat, P], dim=1)
    bot = torch.cat([P.T, torch.zeros(3, 3, dtype=torch.float64)], dim=1)
    L = torch.cat([top, bot], dim=0)
    rhs = torch.cat([dst, torch.zeros(3, 2, dtype=torch.float64)], dim=0)
    sol = torch.linalg.solve(L, rhs)
    return TPS(ctrl=src.to(torch.float32),
               w=sol[:K].to(torch.float32),
               a=sol[K:].to(torch.float32))


def warp_image(img: torch.Tensor, tps: TPS) -> torch.Tensor:
    """Warp (H,W,3) img in [0,1] by `tps`. The TPS maps portrait->canonical;
    to resample we need canonical->portrait, so the sampling grid pushes each
    output pixel back through tps (tps is near-affine-invertible at the scales
    used; for the small non-rigid part we invert by fixed-point iteration)."""
    H, W, _ = img.shape
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                            torch.arange(W, dtype=torch.float32),
                            indexing="ij")
    grid_pts = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1)  # (HW,2)
    # invert tps by fixed-point: p_{n+1} = p_n - (tps(p_n) - target)
    p = grid_pts.clone()
    for _ in range(20):
        p = p - (tps(p) - grid_pts)
    nx = p[:, 0] / (W - 1) * 2 - 1
    ny = p[:, 1] / (H - 1) * 2 - 1
    samp = torch.stack([nx, ny], dim=1).reshape(1, H, W, 2)
    src = img.permute(2, 0, 1)[None]
    out = F.grid_sample(src, samp, mode="bilinear", align_corners=True,
                        padding_mode="border")
    return out[0].permute(1, 2, 0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_register.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/chibi/register.py tests/test_register.py
git commit -m "feat(chibi): TPS portrait->canonical registration"
```

---

### Task 4: `chibi_portrait.py` — Flux flat-lit chibi portrait

**Files:**
- Create: `src/chibi/chibi_portrait.py`

Reuses the project's existing Flux pipeline. This task wraps it with a chibi-style, flat-lighting, front-facing prompt and a resumable per-anchor cache. Follow the matryoshka thread's Flux+PuLID call pattern (see `scripts/` matryoshka render drivers for the exact pipeline entry point).

- [ ] **Step 1: Implement `chibi_portrait.py`**

```python
"""Generate one flat-lit, front-facing, chibi-styled face portrait per anchor,
via the project Flux + PuLID pipeline. Resumable: skip if the PNG exists."""
from __future__ import annotations
from pathlib import Path

PROMPT = ("chibi character face portrait, front view, big round eyes, "
          "soft flat even studio lighting, no harsh shadows, matte skin, "
          "centered, neutral expression, plain background")


def generate_portrait(anchor_id: str, embedding, out_dir: str | Path,
                       seed: int) -> Path:
    """Generate <out_dir>/<anchor_id>.png if absent. `embedding` and `seed`
    are threaded into the project Flux+PuLID call exactly as the matryoshka
    render drivers do. Returns the PNG path."""
    out = Path(out_dir) / f"{anchor_id}.png"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    from <project flux module> import run_flux_pulid  # match matryoshka driver
    img = run_flux_pulid(prompt=PROMPT, embedding=embedding, seed=seed,
                         width=1024, height=1024)
    img.save(out)
    return out
```

Replace `<project flux module>` / `run_flux_pulid` with the actual entry point used by the matryoshka drivers — grep `scripts/` for the Flux call before writing this. If the project's Flux generation is ComfyUI-API-based, call that API instead, matching the existing driver.

- [ ] **Step 2: Smoke-test generation on one anchor**

Run a one-anchor generation (pick any anchor from `build_anchors_parquet.py` output). Expected: a 1024² PNG, recognisably a flat-lit chibi face. Inspect it.

- [ ] **Step 3: Commit**

```bash
git add src/chibi/chibi_portrait.py
git commit -m "feat(chibi): Flux flat-lit chibi portrait generation"
```

---

### Task 5: `koban_bake.py` — single-view UV bake

**Files:**
- Create: `src/chibi/koban_bake.py`
- Test: `tests/test_koban_bake.py`

Reuses `rasterize_uv_attrs` (UV rasterise → texel 3D positions) and `bake_points` (project + sample) from `texture_bake.py` / `bake.py`. New code: single-view wiring, UV-symmetric mirror-fill, skin-fallback.

- [ ] **Step 1: Write the failing test**

`tests/test_koban_bake.py`:

```python
import torch
from chibi.koban_bake import mirror_fill, skin_fallback

def test_mirror_fill_copies_seen_partner():
    tex = torch.zeros(8, 8, 3)
    seen = torch.zeros(8, 8, dtype=torch.bool)
    # left half seen with value 0.7, right half unseen
    tex[:, :4] = 0.7
    seen[:, :4] = True
    out, out_seen = mirror_fill(tex, seen)   # mirror across the W axis
    assert torch.allclose(out[:, 4:], torch.full((8, 4, 3), 0.7))
    assert out_seen.all()

def test_skin_fallback_fills_unseen_with_constant():
    tex = torch.zeros(8, 8, 3)
    seen = torch.zeros(8, 8, dtype=torch.bool)
    tex[0, 0] = torch.tensor([0.6, 0.4, 0.3])
    seen[0, 0] = True
    out = skin_fallback(tex, seen)
    assert torch.allclose(out[4, 4], torch.tensor([0.6, 0.4, 0.3]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_koban_bake.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chibi.koban_bake'`.

- [ ] **Step 3: Implement `koban_bake.py`**

```python
"""Single-view projective bake: a warped chibi portrait -> canonical UV atlas.

rasterize_uv_attrs gives per-texel 3D position + normal; bake_points projects
those through the frontal View and samples the (already-registered) portrait.
mirror_fill + dilate_texture + skin_fallback complete the non-frontal texels.
"""
from __future__ import annotations
import torch
from chibi.bake import bake_points
from chibi.camera_rig import View
from chibi.texture_bake import rasterize_uv_attrs, dilate_texture
from chibi.uv_template import FlameUV
from chibi.koban_asset import KobanMesh


def mirror_fill(texture: torch.Tensor, seen: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """Fill unseen texels from their left-right mirror partner. The canonical
    UV is laid out left-right-symmetric across the W axis, so a texel (h,w)
    mirrors to (h, W-1-w). Returns (texture, seen) with mirror-filled texels
    now marked seen."""
    out = texture.clone()
    out_seen = seen.clone()
    mir = torch.flip(texture, dims=[1])
    mir_seen = torch.flip(seen, dims=[1])
    fillable = (~seen) & mir_seen
    out[fillable] = mir[fillable]
    out_seen[fillable] = True
    return out, out_seen


def skin_fallback(texture: torch.Tensor, seen: torch.Tensor) -> torch.Tensor:
    """Fill any still-unseen texel with the median colour of seen texels
    (a flat skin tone). Back-of-head texels are covered by hair geometry, so a
    constant is acceptable."""
    out = texture.clone()
    if seen.any():
        med = texture[seen].median(dim=0).values
    else:
        med = torch.tensor([0.5, 0.5, 0.5])
    out[~seen] = med
    return out


def bake_portrait_to_uv(koban: KobanMesh, portrait: torch.Tensor,
                        view: View, tex_size: int = 1024,
                        depth: torch.Tensor | None = None) -> torch.Tensor:
    """portrait: (H,W,3) float[0,1], already TPS-registered to `view`.
    Returns the (tex_size,tex_size,3) float[0,1] UV atlas, top-origin."""
    fuv = FlameUV(uv=koban.uv, uv_faces=koban.uv_faces,
                  vt2v=torch.arange(koban.uv.shape[0]))  # 1:1 for an obj export
    pos_map, nrm_map, mask = rasterize_uv_attrs(
        koban.verts, koban.faces, fuv, tex_size)
    pts = pos_map[mask]                                  # (N,3)
    nrm = nrm_map[mask]
    img_u8 = (portrait.clamp(0, 1) * 255).to(torch.uint8)[None]   # (1,H,W,3)
    if depth is None:                                    # no occlusion test
        H, W, _ = portrait.shape
        depth = torch.full((1, H, W), 1e9)
    rgb, seen = bake_points(pts, nrm, img_u8, depth, [view],
                            mode="best", min_facing=0.1)
    tex = torch.full((tex_size, tex_size, 3), 0.5)
    seen_map = torch.zeros(tex_size, tex_size, dtype=torch.bool)
    tex[mask] = rgb
    seen_map[mask] = seen
    tex, seen_map = mirror_fill(tex, seen_map)
    tex = dilate_texture(tex, seen_map)
    tex = skin_fallback(tex, seen_map)
    return tex
```

Note: `bake_points` occlusion-tests against a `depth` map; with a single 2D portrait there is no real depth, so a far-constant disables the test (self-occlusion on a near-frontal chibi face is minor; `min_facing=0.1` already discards grazing texels). If S1 shows back-of-head bleed-through, render a canonical-mesh depth map for `view` and pass it.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_koban_bake.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/chibi/koban_bake.py tests/test_koban_bake.py
git commit -m "feat(chibi): single-view portrait->UV bake with mirror+fallback"
```

---

### Task 6: Spike S1 + S3 — end-to-end registration & shading check

**Files:** Create `scripts/chibi_identity_render.py` (driver).

- [ ] **Step 1: Write the driver**

`scripts/chibi_identity_render.py` — wires Tasks 1/3/4/5 + a verify render:

```python
"""End-to-end: anchor -> portrait -> register -> bake -> textured render.
  uv run python scripts/chibi_identity_render.py --anchor <id> --out <dir>
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch, imageio.v2 as imageio
from chibi.koban_asset import load_koban, load_koban_view, load_koban_landmarks
from chibi.chibi_portrait import generate_portrait
from chibi.register import fit_tps, warp_image
from chibi.koban_bake import bake_portrait_to_uv
from chibi.mesh import TexturedMesh
from chibi.mesh_render import render_textured

CANON = "exp_output/chibi_meshes/koban_canonical"


def detect_portrait_landmarks(png_path):
    """insightface 2D landmarks on the portrait, ordered to match
    load_koban_landmarks' canonical order."""
    import insightface, numpy as np
    app = insightface.app.FaceAnalysis(allowed_modules=["detection", "landmark_2d_106"])
    app.prepare(ctx_id=0)
    faces = app.get(np.asarray(imageio.imread(png_path))[..., ::-1])
    if not faces:
        raise RuntimeError(f"no face detected in {png_path}")
    return torch.tensor(faces[0].landmark_2d_106, dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    koban = load_koban(CANON)
    view = load_koban_view(CANON)
    canon_lmk = load_koban_landmarks(CANON)   # (K,2) in frontal-view pixels

    png = generate_portrait(args.anchor, embedding=None, out_dir=out, seed=0)
    port_lmk_all = detect_portrait_landmarks(png)
    # subset insightface's 106 pts to the K canonical landmark slots
    port_lmk = port_lmk_all[load_koban_landmarks(CANON, return_idx=True)]

    tps = fit_tps(port_lmk, canon_lmk)
    portrait = torch.tensor(imageio.imread(png) / 255.0, dtype=torch.float32)
    warped = warp_image(portrait, tps)
    imageio.imwrite(out / "warped.png", (warped * 255).byte().numpy())

    tex = bake_portrait_to_uv(koban, warped, view)
    imageio.imwrite(out / "texture.png", (tex * 255).byte().numpy())

    tmesh = TexturedMesh(verts=koban.verts, faces=koban.faces,
                         uv=koban.uv, uv_faces=koban.uv_faces, texture=tex)
    frames = render_textured(tmesh, azims=[-30, 0, 30])
    for a, fr in zip([-30, 0, 30], frames):
        imageio.imwrite(out / f"render_{a}.png", fr.numpy())
    print(f"[chibi_identity] wrote portrait/warped/texture/render to {out}")


if __name__ == "__main__":
    main()
```

The insightface-106 → canonical-K index map is part of `load_koban_landmarks` (Task 1 Step 7): annotate each canonical landmark with the insightface-106 index it corresponds to.

- [ ] **Step 2: Run S1 on one anchor**

```bash
uv run python scripts/chibi_identity_render.py --anchor <pick-one> \
  --out exp_output/chibi_meshes/renders/s1_spike
```
Expected: `portrait.png`, `warped.png`, `texture.png`, `render_{-30,0,30}.png`.

- [ ] **Step 3: Inspect — S1 verdict**

Acceptance S1: in `render_0.png`, eyes/nose/mouth land on the right canonical features; the face reads as a coherent chibi, not a smeared projection. Acceptance S3: no baked-in hard shadows / relief from the portrait's lighting.

- If **S1 passes**: skip Task 8, proceed to Task 7.
- If **S1 fails** (registration smears, features misplaced): proceed to Task 8 (inverse-render refinement).
- If **S3 fails** (double-shading): tighten the `chibi_portrait.PROMPT` toward flatter lighting and re-run; if still bad, add an albedo-delighting pass before the bake.

- [ ] **Step 4: Record verdict + commit**

Append S1/S3 verdicts to the chibi topic index.
```bash
git add scripts/chibi_identity_render.py docs/research/_topics/lam-chibi-recipe.md
git commit -m "spike(chibi): S1 registration + S3 shading verdict"
```

---

### Task 7: ARKit-driven verify render

**Files:**
- Modify: `scripts/chibi_identity_render.py` (add a `--drive` mode)
- Create: `scripts/koban_drive_render.py` (Blender script)

The textured mesh must be shown *driven*. The texture is baked in pytorch3d space; driving uses the mesh's native shape keys in Blender. This task applies the baked texture as the Koban material's base-colour map and renders an ARKit clip.

- [ ] **Step 1: Write the drive-render Blender script**

`scripts/koban_drive_render.py` — opens the `.blend`, assigns `texture.png` as the base-colour image of the `Chibi Base Mesh` material, animates a neutral→blink→jawOpen→smile clip, renders to a PNG sequence:

```python
"""blender -b <koban.blend> --python scripts/koban_drive_render.py -- <texture.png> <outdir>"""
import sys, os
sys.path.append("/home/newub/miniconda3/lib/python3.12/site-packages")
import bpy

tex_path, outdir = sys.argv[sys.argv.index("--") + 1:][:2]
os.makedirs(outdir, exist_ok=True)
obj = bpy.data.objects["Chibi Base Mesh"]

# assign texture.png as the base-colour map
mat = obj.data.materials[0]
mat.use_nodes = True
bsdf = mat.node_tree.nodes.get("Principled BSDF")
img = mat.node_tree.nodes.new("ShaderNodeTexImage")
img.image = bpy.data.images.load(tex_path)
mat.node_tree.links.new(img.outputs["Color"], bsdf.inputs["Base Color"])

kb = obj.data.shape_keys.key_blocks
clip = [("neutral", []), ("blink", ["eyeBlinkLeft", "eyeBlinkRight"]),
        ("jawOpen", ["jawOpen"]), ("smile", ["mouthSmileLeft", "mouthSmileRight"])]
scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
for label, keys in clip:
    for k in kb:
        k.value = 0.0
    for n in keys:
        kb[n].value = 1.0
    scene.render.filepath = os.path.join(outdir, f"drive_{label}.png")
    bpy.ops.render.render(write_still=True)
    print(f"[drive] {label}")
```

- [ ] **Step 2: Run the verify render**

```bash
blender -b "exp_output/chibi_meshes/koban/Koban Chibi Base Mesh VRM export.blend" \
  --python scripts/koban_drive_render.py -- \
  exp_output/chibi_meshes/renders/s1_spike/texture.png \
  exp_output/chibi_meshes/renders/drive_verify
```
Expected: 4 PNGs showing the *textured* chibi driven through the clip.

- [ ] **Step 3: Inspect**

Acceptance: the baked identity texture stays correctly placed as the rig drives; eyes/mouth move under the texture without the texture sliding off features.

- [ ] **Step 4: Commit**

```bash
git add scripts/koban_drive_render.py scripts/chibi_identity_render.py
git commit -m "feat(chibi): ARKit-driven verify render of textured canonical mesh"
```

---

### Task 8: (CONDITIONAL) Inverse-render texture refinement

**Trigger:** Only if Task 6 Step 3 records S1 as **failed** — TPS registration alone does not place features correctly.

**Files:**
- Create: `src/chibi/koban_bake_refine.py`
- Test: `tests/test_koban_bake_refine.py`

Approach B from the spec: initialise the UV texture with the Task-5 bake, then optimise it (and a small residual 2D warp) so that `render_textured` of the canonical mesh from the frontal view matches the portrait under a photometric + LPIPS loss.

- [ ] **Step 1: Write the failing test**

`tests/test_koban_bake_refine.py`:

```python
import torch
from chibi.koban_bake_refine import refine_texture

def test_refine_reduces_photometric_loss():
    # toy: a flat target image, refine should drive a uniform texture toward it
    target = torch.full((64, 64, 3), 0.8)
    init_tex = torch.full((128, 128, 3), 0.2)
    refined, final_loss, init_loss = refine_texture(
        init_tex, target, steps=50, _toy=True)
    assert final_loss < init_loss * 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_koban_bake_refine.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chibi.koban_bake_refine'`.

- [ ] **Step 3: Implement `koban_bake_refine.py`**

Implement `refine_texture(init_tex, target, *, steps, view=None, koban=None, _toy=False)`: an Adam loop on the texture tensor. In `_toy` mode the "render" is an identity resize of the texture to the target resolution (so the test exercises the optimiser without pytorch3d). In real mode it renders the textured canonical mesh through `view` with `render_textured` and uses MSE + LPIPS against the portrait. Return `(refined_tex, final_loss, init_loss)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_koban_bake_refine.py -v`
Expected: PASS.

- [ ] **Step 5: Wire into the driver + re-run S1**

Add a `--refine` flag to `scripts/chibi_identity_render.py` that calls `refine_texture` after `bake_portrait_to_uv`. Re-run the S1 anchor; inspect.

- [ ] **Step 6: Commit**

```bash
git add src/chibi/koban_bake_refine.py tests/test_koban_bake_refine.py \
  scripts/chibi_identity_render.py
git commit -m "feat(chibi): inverse-render UV texture refinement (Approach B)"
```

---

## Self-Review

**Spec coverage:**
- R0 licensing → Task 0. R1 immutable geometry → Task 1 exports a fixed mesh; no deform task exists. R2 per-anchor texture → Task 5/6 output. R3 ARKit-52 drive → Task 2 (rig works) + Task 7 (driven render). R4 resumable batch → `generate_portrait` skip-if-exists (Task 4); driver is per-anchor. R5 no double-shading → Task 6 S3 + flat-light prompt (Task 4). R6 chibi-styled texture → Task 4 `PROMPT`. R7 continuity re-measurement → explicitly out of scope (spec + here).
- Spikes S1/S2/S3 → Tasks 3+6 / 2 / 6.
- Approaches B/C → B is conditional Task 8; C is out of scope (spec).

**Placeholder scan:** One deliberate unknown — `<project flux module>` in Task 4, with an explicit instruction to grep the matryoshka drivers for the real entry point before writing. This is genuine codebase-specific wiring the implementer must resolve; flagged, not hidden.

**Type consistency:** `KobanMesh` (verts/faces/uv/uv_faces) consistent Tasks 1/5/6. `TPS` / `fit_tps` / `warp_image` consistent Tasks 3/6. `bake_portrait_to_uv` / `mirror_fill` / `skin_fallback` consistent Tasks 5/6. `TexturedMesh` matches the existing `src/chibi/mesh.py` field names (verts/faces/uv/uv_faces/texture).

**Known follow-ups documented in-task:** Koban UVs possibly outside [0,1] (Task 1 Step 6); single-view depth/occlusion (Task 5 Step 3 note); insightface-106→canonical-K index map (Task 1 Step 7 + Task 6 Step 1).
