---
status: live
topic: arkit-controlnet
---

# CFM conditioning render — design

**Date:** 2026-05-18
**Goal:** Produce the spatial conditioning signal the trained expression
ControlNet (the CFM run) is supervised against: for each training row, a render
of a FLAME head deformed by that row's 52 ARKit blendshapes and posed to match
the row's photo. Decide what to precompute and what to render live.

## Context

All three zero-train expression routes are falsified
(`2026-05-18-arkit-depth-controlnet-spike-verdict.md`). The expression channel
must be trained: conditional flow matching against real `(photo, FLAME-render)`
pairs, base DiT + identity path frozen, only the InfuseNet-style control
encoder trains. This spec covers *only* the FLAME-render side of that pair —
the render function and its one expensive precompute. The CFM training run
itself is a separate, larger spec.

## What the render must be

The control image must be **reproducible at inference from 52 ARKit
coefficients alone** — at inference time there is no photo and no MediaPipe
mesh, only a target expression vector. That rules out conditioning on a
MediaPipe 478-mesh (not a function of the 52 coefficients) and forces a FLAME
render: `verts = v_template + flame_arkit_bs · arkit52`.

The render carries **expression and head pose, not identity**. Identity is the
InfuseNet/ArcFace path's job. So the mesh is the canonical FLAME template — no
per-identity shape coefficients. The control image stays identity-free; the CFM
model cannot leak face shape from it.

The render must be **spatially aligned with the photo**. A ControlNet is a
spatial conditioner — InfiniteYou's own control slot is a *posed* 5-keypoint
image ("control the generation position"). A frontal render paired with a
3/4-view photo is contradictory supervision. So the FLAME head is rotated to
the photo's head pose and placed at the photo's face bbox.

## Approaches

**Approach A — render 79k PNGs to a static cache.** One pass, store a PNG per
`image_sha256`. Rejected: commits to one render modality (normals vs depth vs
flat-shaded is an *open question* — see topic index) before the CFM run can A/B
it, and a re-cache costs the whole pass again. The geometry step it would amend
(`flame_arkit_bs · arkit52`) is a trivial matmul — there is nothing expensive to
amortize on the FLAME side.

**Approach B — cache only the pose, render live (recommended).** The single
genuinely expensive, image-dependent step is recovering each photo's head pose,
which means re-running MediaPipe FaceLandmarker over the source images. Cache
*that*. The render itself — deform, rotate, rasterize — is cheap CPU work
(painter's-algorithm flat-shade, the same `cv2.fillConvexPoly` path already
proven in `landmark_control.render_depth_map`, ~10k FLAME faces) and runs live
in the CFM dataloader. Modality stays a render-function argument, so the CFM
run picks normals/depth/flat with no re-cache.

**Approach C — skip pose, render frontal.** Rejected above: contradictory
spatial supervision.

**Selected: B.** Precompute a pose cache; render on the fly.

## Architecture

Two units, plus a one-time asset-prep step.

### FLAME asset prep (one-time)

`flame2023.pkl` (LAM) stores `v_template` and faces as `chumpy` arrays —
unpicklable without the legacy `chumpy` dep. Extract once (in a chumpy-equipped
env, or via LAM's own loader) into a plain `output/flame_assets/flame_base.npz`
holding `v_template` (5023, 3) float32 and `faces` (n_faces, 3) int32. The
ARKit basis `flame_arkit_bs.npy` (52, 5023, 3) is already plain numpy and is
copied alongside.

The basis's 52 channels must be in the same order as the `bs_*` columns /
`ARKIT_BLENDSHAPE_NAMES`. This is asserted, not assumed — see Testing.

### `src/arkit_controlnet/flame_render.py`

Pure geometry + rasterization. No I/O of photos, no MediaPipe. Testable in
isolation.

- `load_flame_assets() -> FlameAssets` — loads the npz + basis, caches at
  module level. `FlameAssets` holds `v_template`, `faces`, `arkit_basis`.
- `deform(arkit52: np.ndarray) -> np.ndarray` — `v_template + einsum('k,kij->ij',
  arkit52, basis)`, returns (5023, 3). `arkit52` ordered per
  `ARKIT_BLENDSHAPE_NAMES`.
- `render(verts, rotation, bbox, modality, H, W) -> np.ndarray` —
  rotates verts by `rotation` (3×3), orthographically projects (drop z),
  similarity-maps the projected face bbox onto `bbox` (the photo's face box:
  `cx, cy, w, h`, normalized), flat-shades faces, painter's-algorithm composite
  by per-face mean camera-z. Returns (H, W, 3) uint8. `modality` ∈
  {`normals`, `depth`, `flat`}; default `normals` (per-face normal → RGB — the
  richest geometry signal and standard for face ControlNets).

Only the MediaPipe matrix's **rotation** is reused; its translation/scale live
in MediaPipe's metric space and do not transfer. In-plane placement and scale
come entirely from the 2D bbox fit — robust across coordinate systems.

### `src/arkit_controlnet/build_ffhq_index.py`

Builds `output/ffhq_index/ffhq_sha_index.parquet` (see *Corpus* above): the
`image_sha256 → (shard_idx, row_idx)` join key. Resumable per shard. Run once
before the pose cache.

### `src/arkit_controlnet/build_pose_cache.py`

Batch MediaPipe FaceLandmarker over the FFHQ images — read from the parquet
shards via `ffhq_sha_index` — → `pose_cache.parquet` at
`output/flame_pose_cache/`. One row per image:

- `image_sha256`
- `rotation` — 9 floats, the 3×3 of `facial_transformation_matrix`
- `bbox_cx, bbox_cy, bbox_w, bbox_h` — face landmark extent, normalized to the
  image, from the same MediaPipe result
- `pose_detected` — bool; `False` rows are excluded from the CFM corpus

Resumable: skip any `image_sha256` already present in the parquet (per
`feedback_resumable_generation`). Runs under the miniconda python (mediapipe
`solutions`), like the spike runners.

## Data flow

```
FFHQ parquet shards ─sha256─▶ ffhq_sha_index.parquet
        │                            │
        └─image bytes────────────────┤
                                     ▼
                         MediaPipe ─▶ rotation + bbox  ─────────┐
                                       (pose_cache.parquet)     │
reverse_index.bs_* ─▶ arkit52 ─deform─▶ verts ──────────────────┼─▶ render() ─▶ control image
flame_base.npz + flame_arkit_bs.npy ────────────────────────────┘   (live, in CFM dataloader)
```

The CFM dataloader (future spec) joins `reverse_index` → `pose_cache` →
`ffhq_sha_index` on `image_sha256`, drops `pose_detected == False`, reads the
photo from the shard, and calls `render()` per item.

## Corpus — full FFHQ-70k, read from the parquet shards

`reverse_index.parquet` has 79,116 rows: 70,000 `source == ffhq` (69,928 with
`bs_detected`), plus ~9k synthetic rows (`flux_corpus_v3`,
`flux_solver_a_grid_squint`) that the CFM run does not use. Only 2,725 FFHQ
PNGs are materialized under `output/ffhq_images/`, but the **full FFHQ-70000**
dataset is on the Seagate drive at
`/media/newub/Seagate Hub/arc_distill/ffhq_parquet/` — 190 HuggingFace parquet
shards, 90 GB, images embedded as an `image` column.

So the CFM corpus is ~69,928 `(photo, render)` pairs. The photos are **read
from the parquet shards** — decoding 70k images to PNG would cost ~100 GB of
disk for no benefit, and both the pose-cache build and the future CFM dataloader
can iterate the shards directly.

`reverse_index` keys rows on `image_sha256`; the FFHQ parquet has no sha. One
prerequisite, built once:

### `output/ffhq_index/ffhq_sha_index.parquet`

Iterate the 190 shards, decode each image, compute `image_sha256`, record
`(image_sha256, shard_idx, row_idx)`. This is the join key from `reverse_index`
to the actual image bytes. The build also **verifies the sha convention**: the
2,725 already-materialized PNGs are named by sha — confirm that hashing a shard
image reproduces a sha that exists in `reverse_index` (i.e. the sha is of the
image bytes the project standardized on). If the conventions differ — e.g. the
PNGs were re-encoded before hashing — the index build resolves it (hash the
re-encoded form) before any pose-cache work proceeds. Resumable per shard.

## Error handling

- MediaPipe no-detect → `pose_detected = False`; row excluded downstream. No
  raise — one bad image must not abort a 2,725-image batch.
- Missing FLAME asset (`flame_base.npz` not yet extracted) → `load_flame_assets`
  raises with the prep command in the message.
- `deform` with non-finite `arkit52` → raise `ValueError` (corpus corruption,
  must surface, not silently render garbage).
- `render` with a degenerate `bbox` (w or h ≈ 0) → raise `ValueError`.

## Testing

- `deform(zeros(52))` is exactly `v_template` (neutral → template).
- Channel-order guard: load a high-`bs_jawOpen` row's vector, `deform` it,
  assert the lower-lip vertex group moves down relative to template. A wrong
  basis/column permutation fails this. (One assertion per a few well-separated
  channels — jawOpen, mouthSmileLeft, browInnerUp.)
- `render` of a neutral mesh is non-empty and the face occupies the bbox to
  within a few percent.
- `pose_detected` rate on the 2,725 FFHQ images is sanity-checked against
  `reverse_index.bs_detected` — the two MediaPipe detections should largely
  agree.
- Manual verification (per the standing post-task rule): render ~8 FFHQ rows,
  alpha-overlay each on its photo, save a collage to
  `exp_output/flame_render_check/`, eyeball pose + expression alignment.

## Out of scope

The CFM training run, the InfuseNet control-encoder modification, the dataloader
itself, and SPMS pair mining. This spec ends at: the FFHQ sha-index and the pose
cache exist, and `render()` produces an aligned control image from a row's
blendshapes.
