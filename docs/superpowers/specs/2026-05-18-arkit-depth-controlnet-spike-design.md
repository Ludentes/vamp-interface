---
status: live
topic: arkit-controlnet
---

# ARKit depth-ControlNet stacked spike — design

## Goal

Test the last cheap zero-train route for the expression channel: drive a **stock
FLUX Depth ControlNet** with a depth map rasterized from MediaPipe landmarks,
stacked alongside an **identity-only InfuseNet**. The depth ControlNet has its
own weights trained to reproduce dense geometry — unlike the InfuseNet
5-keypoint slot, which the dense-mesh spike
(`2026-05-18-arkit-landmark-control-spike-verdict.md`) falsified for expression.

The question: does a stock depth ControlNet steer FLUX output expression toward
a target while InfuseNet holds ArcFace identity — with zero training?

## Why this is the only fallback worth running

The sparse-contour fallback was dropped: a thinner polyline drawing is still an
out-of-distribution image to the InfuseNet keypoint slot, so it shares the
falsified failure mode. The depth ControlNet is different in kind — a separate
model whose weights *were* trained on dense depth. It has a genuine mechanism;
the open risks (landmark-derived depth too crude, ControlNet-vs-InfuseNet
contention) are real and testable.

## Architecture

One ComfyUI graph stacking two control mechanisms onto shared FLUX conditioning:

- **Identity** — `IDEmbeddingModelLoader` → `ExtractIDEmbedding` (from the ID
  photo) → `InfuseNetApply`. InfuseNet runs **identity-only**: its spatial-control
  `image` input is fed an all-black image, so only the 8 ArcFace tokens steer
  identity, no keypoint geometry constraint.
- **Expression** — `ControlNetLoader` (InstantX FLUX.1-dev ControlNet-Union,
  already installed at `controlnet/FLUX.1/instantx-union/`) →
  `SetUnionControlNetType` with `type="depth"` → `ControlNetApplyAdvanced`,
  driven by the rasterized depth map.

The two compose through ComfyUI's standard conditioning mechanism:
`ControlNetApplyAdvanced` writes `d['control']` on the conditioning;
`InfuseNetApply` reads `prev_cnet = d.get('control')` and chains it via
`set_previous_controlnet`. **Order:** depth ControlNet applied first, then
`InfuseNetApply`, so both controls are active during sampling.

### Data flow

```
ID photo ──► ExtractIDEmbedding ──┐
                                  ▼
CLIP text ──► (pos,neg) ──► ControlNetApplyAdvanced ──► InfuseNetApply ──► KSampler ──► VAEDecode
                                  ▲                          ▲
exemplar ──► render_depth_map ────┘          black image ─────┘
```

## Components

### `face_landmarks_xyz` — `src/arkit_controlnet/eval_spike.py`

New function alongside the existing `face_landmarks_xy`. Returns the MediaPipe
FaceLandmarker landmarks as an `(478, 3)` float array — normalized `x, y` in
`[0,1]` plus the raw `z` (head-centred relative depth; more negative = closer to
camera). Raises `ValueError` if no face is detected. `face_landmarks_xy` stays
unchanged (the mesh renderer still uses it).

### `render_depth_map` — `src/arkit_controlnet/landmark_control.py`

`render_depth_map(image_path: Path) -> np.ndarray` — rasterizes the exemplar's
face mesh as a grayscale depth control image, `(1152, 864, 3)` uint8.

- Get `(478,3)` landmarks via `face_landmarks_xyz`.
- Recenter + isotropically scale `x,y` to the same framing constants the mesh
  renderer uses (`_FACE_FRAC = 0.45`, `_CENTER_Y = 0.42`, `864×1152` canvas) so
  the depth map pins head size/position identically.
- Map `z` to grayscale: `g = 255 * (z_max - z) / (z_max - z_min)` — nearest
  point (nose tip) → white, farthest → black. Background stays 0 (far).
- **Painter's algorithm:** sort the `FACEMESH_TESSELATION` triangles by mean `z`
  descending (farthest first); `cv2.fillConvexPoly` each triangle flat-shaded
  with its mean-`z` grayscale. Nearer triangles paint last, over farther ones —
  a coarse but correct face-depth blob. Replicated across 3 channels.

### `comfyui/workflows/arkit_depth_spike.json`

New API-format workflow, modelled on `arkit_landmark_spike.json`. Adds three
nodes — `ControlNetLoader`, `SetUnionControlNetType`, `ControlNetApplyAdvanced`
— and a third `LoadImage` for the black InfuseNet control image. `$$`-placeholders:
`$$IDENTITY_FILENAME`, `$$DEPTH_FILENAME`, `$$BLACK_FILENAME`,
`$$DEPTH_STRENGTH`, `$$SEED`, `$$OUTPUT_PREFIX`. InfuseNet strength is **fixed at
0.6** in the template (str 1.0 collapsed identity in the prior spike).

### `src/arkit_controlnet/run_depth_spike.py`

Modelled on `run_landmark_spike.py`. Reuses `select_exemplars` / `select_neutral`
/ `select_identities` unchanged. Per axis: render one depth map from the chosen
exemplar; also write one all-black 864×1152 PNG once. Sweep: 3 identities × (3
axes + neutral) × depth strength `{0.5, 0.8}` = 24 cases. Fixed seed 2026, 25-step
euler. Resumable (skip fresh PNG). Metric calls wrapped in try/except → `-1.0`.
Writes `exp_output/arkit_depth_spike/metrics.parquet` + per-case PNGs.

## Success criteria

Same bar as the falsified spikes, so results are directly comparable. On a
**majority** of the 9 identity×axis cells, at some swept depth strength:

1. `arcface_cos(output, id_photo) ≥ 0.55` — identity holds.
2. `expr_cos(output, exemplar)` beats the **same-identity, same-depth-strength
   neutral-control** output by **≥ 0.05** — expression moved toward target.
3. Eyeball gate — the axis columns are visibly distinct from neutral in the
   collage.

The baseline-relative delta (criterion 2) is computed in a post-run analysis
step, not stored directly in `metrics.parquet` (which stores raw
`expr_cos(output, own-exemplar)`), exactly as in the prior spike's verdict.

## Error handling

- **MediaPipe re-detection failure** on an exemplar → fall through the `k=3`
  exemplar candidates (existing `_prepare_controls` pattern).
- **InfuseNet black-image collapse** — an all-black spatial control is a
  plausible failure point for InfuseNet. Mitigation: the runner renders the
  three **neutral**-axis cases first; if neutral outputs have no detectable face
  (`arcface_cos = -1.0`) the run is reported inconclusive on depth — the failure
  is InfuseNet's, not the depth ControlNet's — and the fallback is to feed
  InfuseNet the depth map instead of black (face-shaped hint, weaker isolation).
  This is a documented checkpoint, not auto-retry.
- **Metric crash** on an unreadable PNG → caught, recorded as `-1.0`, sweep
  continues.

## Testing

- `face_landmarks_xyz` returns `(478,3)`; `x,y` columns equal
  `face_landmarks_xy` for the same image; `z` column is non-constant.
- `render_depth_map` returns `(1152,864,3)` uint8; contains both near-white
  (`>200`) and black pixels; the depth blob is centred near `_CENTER_Y`.
- Selection tests unchanged (reused from `test_landmark_control.py`).

## Out of scope

FLAME rendering, any training, sparse-contour control, depth-strength values
outside `{0.5, 0.8}`, InfuseNet strengths other than 0.6. If this spike also
fails, the thread proceeds directly to the CFM training run — no further
zero-train fallback.
