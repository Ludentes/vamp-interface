# ARKit landmark-control spike — design

## Goal

Answer one question with zero training: **does a MediaPipe face-mesh control
image, fed into InfiniteYou's InfuseNet spatial-control slot, steer FLUX's
output expression while ArcFace identity holds?**

This is "Path 1" from `_topics/arkit-controlnet.md`, run *before* any CFM
training. It follows the falsified zero-train FluxSpace spike
(`2026-05-18-arkit-controlnet-spike-verdict.md`): identity injection via
InfuseNet was solid there; only the expression channel failed. This spike
swaps the failed channel (FluxSpace attention editing) for InfuseNet's own
spatial-control slot driven by a landmark mesh.

## The central risk

The released InfuseNet weights were trained with a **5-keypoint** control image
(eyes / nose / mouth corners) — enough to lock head *position*, not expression.
A 478-vertex MediaPipe tessellation is out-of-distribution for those weights.
The spike's real job is to find out whether the slot has any expression
bandwidth at all. A clean negative is a valid, useful result — it routes the
thread to the fallback (stock depth ControlNet, stacked) or to CFM training.

## Approach

No blendshape-to-geometry synthesis. We **select real expression exemplars**
from `reverse_index.parquet` by their stored ARKit coefficients, extract each
exemplar's MediaPipe 478-landmark mesh, normalize it to a canonical portrait
framing, and render it as the InfuseNet control image. Blendshapes drive
*exemplar selection*, not geometry synthesis — this sidesteps the missing
blendshape→vertex basis (the same gap that made FLAME look necessary).

Reuses the three axes already in `src/arkit_controlnet/axes.py`
(`smile`, `pucker`, `surprise`) — each axis names the ARKit `target_channels`
used both to *select* exemplars and to *score* outputs.

## Architecture

Four units. Three new files plus one new workflow; `axes.py` and
`eval_spike.py` are reused unchanged.

### `src/arkit_controlnet/landmark_control.py` — exemplar selection + mesh render

- `select_exemplars(axis_name, k) -> list[Path]` — reads `reverse_index.parquet`,
  filters to `source == "ffhq"`, `bs_detected == True`, and FFHQ PNGs present on
  disk; ranks by the sum of the axis's `target_channels` (`bs_<channel>`
  columns); returns the top-`k` image paths. Plus `select_neutral(k)` — the `k`
  rows with the smallest total expression energy (sum of all non-`_neutral`
  `bs_*` columns) for the baseline control.
- `render_landmark_mesh(image_path, size=(864,1152)) -> np.ndarray` — runs the
  shared MediaPipe `FaceLandmarker` (imported from `eval_spike`), takes the 478
  normalized landmarks, **recenters and isotropically scales** the face so its
  bounding box occupies ~45% of canvas height centred on the canvas, and draws
  the `FACEMESH_TESSELATION` edges as thin white lines on black. Returns an
  RGB `uint8` array. No detection → raises `ValueError` (caller skips exemplar).
  Normalization is what fixes the previous spike's framing artifact: the
  control image now also pins head size and position.

### `comfyui/workflows/arkit_landmark_spike.json` — the graph

Clone of `arkit_controlnet_spike.json` with two changes:

- Node 8 `EmptyImage` (blank control) → `LoadImage` reading `$$CONTROL_FILENAME`
  (the rendered mesh, uploaded per case).
- FluxSpace nodes 13/14/15 removed; `KSampler.model` wires straight from
  `UNETLoader` (`["1", 0]`). This spike isolates the InfuseNet control slot —
  no attention editing in the graph.

`InfuseNetApply.strength` / `start_percent` / `end_percent` stay as
`$$`-placeholders so the runner can sweep them.

### `src/arkit_controlnet/run_landmark_spike.py` — the driver

- Identities: reuse `run_spike.select_identities(3)` (3 is enough for a spike).
- For each axis in `smile`/`pucker`/`surprise`: pick 1 exemplar via
  `select_exemplars(axis, k=3)` (first that yields a mesh), render its mesh,
  save to `exp_output/arkit_landmark_spike/control_<axis>.png`.
- One neutral control from `select_neutral` as the per-identity baseline.
- Sweep `strength ∈ {0.6, 1.0}` (the slot's effective range; 2 values keeps the
  run ~24 renders / ~25 min). `end_percent` fixed at 1.0 for the first run.
- Per case: upload identity image + control mesh via `ComfyClient.upload_image`,
  `$$`-substitute the workflow, `client.generate`. Resumable (skip fresh PNG,
  ≥1 KB) and per-case `try/except`, matching `run_spike.py`.
- Metrics row per case: `arcface_cos(output, identity)` and `expr_cos` — the
  cosine (`eval_spike.expr_cos`) between the output's 52-d blendshape vector and
  the *exemplar's* 52-d vector (does the output wear the target expression?).
  Written to `metrics.parquet`; a `collage.png` is built identical in layout to
  the last spike (identities × axes×strength, source + control + outputs).

### `src/arkit_controlnet/eval_spike.py` — one addition

Add `bs_vector(image_path) -> np.ndarray` (52-d, order = `ARKIT_BLENDSHAPE_NAMES`)
and `expr_cos(path_a, path_b) -> float` (cosine of the two 52-d vectors,
excluding `_neutral`). `bs_read` already does the per-name extraction; these
wrap it. Existing `arcface_cos` / `bs_delta` untouched.

## Data flow

```
parquet bs_* columns ──select_exemplars──▶ exemplar image
exemplar image ──MediaPipe FaceLandmarker──▶ 478 landmarks
478 landmarks ──normalize + draw tessellation──▶ control mesh PNG
control mesh PNG ─┐
identity PNG ─────┼─▶ InfuseNet (identity tokens + control slot) ─▶ FLUX ─▶ output
                  │
output ──ArcFace──▶ arcface_cos      output ──MediaPipe──▶ expr_cos vs exemplar
```

## Success criteria

The spike *succeeds at answering* regardless of outcome. The substantive
positive result — "InfuseNet's slot can carry expression" — holds if, on a
majority of identity×axis cells at some swept strength:

- identity holds: `arcface_cos ≥ 0.55` (last spike's usable-band floor), **and**
- expression moves toward target: `expr_cos(output, exemplar)` exceeds
  `expr_cos(neutral_output, exemplar)` by ≥ 0.05, **and**
- the move is visible in `collage.png` (eyeball gate — metrics can be fooled by
  occlusion, as the last spike showed).

A negative (mesh ignored, or identity destroyed when the control bites) is
recorded as such and routes to the fallback below.

## Fallback (documented, not built)

If the dense mesh is ignored: the verdict doc recommends **approach C** — render
a depth map from the same 478 landmarks and drive a *stock* FLUX Depth
ControlNet stacked alongside identity-only InfuseNet (InfiniteYou documents
plug-and-play ControlNet stacking). A depth ControlNet has the dense-spatial
bandwidth InfuseNet's 5-kp-trained slot lacks. Still no FLAME, still no
training. Sparse-contour rendering (lips/eyes/oval polylines only — closer to
the 5-kp training distribution) is a cheaper intermediate variant worth one run
before committing to C.

## Error handling

- Exemplar with no MediaPipe detection → `render_landmark_mesh` raises
  `ValueError`; `select_exemplars` returns `k=3` candidates so the runner falls
  through to the next.
- Output with no detectable face → `arcface_cos` returns `-1.0`,
  `expr_cos` returns `-1.0` (both already fail loudly on bad reads).
- Fewer than 3 FFHQ images on disk → `select_identities` raises (existing
  behaviour).

## Testing

- `tests/arkit_controlnet/test_landmark_control.py`:
  - `select_exemplars("smile", 3)` returns 3 paths; their summed smile
    coefficients are all above the corpus median (selection actually ranks).
  - `render_landmark_mesh` on the local fixture face returns an `(1152,864,3)`
    `uint8` array whose white-pixel count is non-trivial (mesh actually drawn).
  - meshes rendered from a high-`smile` exemplar and a high-`jawOpen` exemplar
    differ (`np.array_equal` is `False`) — different expressions → different
    control.
- `tests/arkit_controlnet/test_eval_spike.py`: extend with `expr_cos(face,face)
  ≈ 1.0` and `bs_vector` length 52.
- Tests `skipif` the parquet / fixture is absent, matching existing suite.
- Non-trivial code reviewed by `superpowers:code-reviewer` before the run
  (standing rule).

## Out of scope (YAGNI)

No training, no FLAME, no depth rendering (fallback only), no per-channel
strength tuning, no expression *interpolation* — exemplars are discrete. One
modality (dense tessellation) in the first run; sparse-contour only if it fails.
