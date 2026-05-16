---
status: live
topic: matryoshka-portrait
---

# Matryoshka Identity-Swap Sweep — Design

## Goal

Run a 252-cell parameter sweep on the Windows ComfyUI box that, for every
cell, generates a generic matryoshka doll with Flux-Krea + PuLID and then
transfers a real identity onto its painted face with `inswapper_128`. The
sweep maps how identity-swap quality varies across anchors, prompt finish,
ControlNet structure, and PuLID schedule entry — a ~3.3 h unattended job.

## Why this shape

The PuLID-weight ladder (`/tmp/matpeek/pulid_ladder_montage.png`) proved
PuLID cannot transfer identity through the flat-painted matryoshka style: at
coherent weights (≤2.0) every doll is a generic rosy-cheeked woman, and
overcranking (≥3.0) collapses the image. But the follow-up swap probe
(`swap_ladder_facezoom.png`) showed the complementary result: a PuLID doll is
a *coherent doll with a detectable face*, and `inswapper` swapped a
recognisable identity onto it — SCRFD detected the painted face at 0.78–0.83
confidence, no detector circumvention needed.

So the roles split cleanly: **PuLID is the doll generator, inswapper is the
identity vehicle.** This sweep is the systematic exploration of that pipeline.

## The recipe (fixed, not swept)

Locked from the `pw150 default` probe result:

- **PuLID weight = 1.5**, `start ∈ {0.0, 0.1}` (swept), `end = 1.0`
- **Swap = SCRFD-default**: `insightface` `buffalo_l` detects the doll face,
  `inswapper_128` swaps the anchor identity in. MediaPipe synthetic-kps swap
  is the *fallback* only when SCRFD finds no face.
- **No repaint pass.** The swapped face stays photoreal; re-stylising it as
  matryoshka paint is a separate, later experiment.

## The grid (252 cells, 1 deterministic seed each)

| Axis | Values | Count |
|---|---|---|
| Anchor | `id_user` + `id_00`…`id_19` | 21 |
| Prompt finish | glossy / satin / matte | 3 |
| CN strength | 0.0 (prompt-only) / 0.5 (doll-form structure) | 2 |
| PuLID start | 0.0 / 0.1 | 2 |

`21 × 3 × 2 × 2 = 252`. Seed is deterministic per cell:
`70_000_000 + cell_index × 7919`. No seed resampling — variety comes from the
252 parameter combinations. At ~47 s/cell (PuLID gen ≈42 s + swap ≈4 s) the
run is ≈3.3 h.

Prompt finish substitutes the lacquer token in the base prompt (addresses the
"too glossy" feedback):

- glossy → `glossy lacquer finish`
- satin  → `satin lacquer finish`
- matte  → `matte painted finish`

## Architecture

Two-process design, one runner. ComfyUI generates the doll; a standalone
Python stage in the *same runner process* does detection + swap. Chosen over
a ComfyUI-only graph (no ReActor node on the box, and our SCRFD-default +
fallback logic is easier to control in Python) and over two separate passes
(needless second artifact and resume state).

### Components

**`scripts/swap_core.py`** — reusable detect-and-swap module, no sweep
knowledge. The pipeline's testable unit.

- `make_face_app() -> FaceAnalysis` — `buffalo_l`, detection + recognition,
  **CPU providers only** (`ctx_id=-1`) so the swap never contends with
  ComfyUI for the 3090's VRAM. `det_size=(640, 640)`.
- `load_swapper(path) -> swapper` — `inswapper_128.onnx`.
- `detect_source(app, img_bgr) -> Face | None` — highest-`det_score` face.
- `mediapipe_kps_bbox(img_bgr) -> (kps, bbox) | (None, None)` — FaceMesh
  fallback landmarker (iris-ring eye centres, lm 1 nose, 61/291 mouth; bbox
  from the 0–467 mesh extent). Import of `mediapipe.solutions` is guarded — if
  unavailable the fallback degrades to "unavailable", it does not crash.
- `swap_identity(app, swapper, doll_bgr, source_face) -> (result, mode, score)`
  — try SCRFD detection on the doll; on hit, swap with `mode="default"`. On
  miss, try the MediaPipe synthetic-`Face` path with `mode="forced"`. If both
  fail, return `(doll_bgr, "failed", 0.0)` — the un-swapped doll is kept.

**`scripts/matryoshka_swap_sweep.py`** — the sweep runner. Reuses
`matryoshka_sweep.py` helpers (`queue`, `wait`, `download`, `build_workflow`,
`_retry`, the node-id assertions) and imports `swap_core`. Per cell:

1. Build the PuLID workflow (anchor PNG, finish-substituted prompt, CN
   strength, PuLID start; weight 1.5 and end 1.0 fixed).
2. `queue → wait → download` the doll PNG to `…/dolls/<stem>.png`.
3. `swap_core.swap_identity(...)` with the anchor's own photo as source →
   write `…/swapped/<stem>.png` (atomic).
4. Append a manifest row.

Pre-flight before any queuing: assert all 21 anchor PNGs exist, ComfyUI
reachable, `inswapper_128.onnx` present, workflow node ids match. Fail loud.

### Data flow

```
anchor.png ──┬─> PuLID identity input ─┐
             │                         ├─> ComfyUI ─> doll.png ─┐
template_canny.png ─> Canny CN ────────┘                        │
             │                                                  v
             └─> inswapper SOURCE ──────> swap_core ──> swapped.png
```

### Outputs

```
refs_matryoshka/swap_sweep/
  dolls/<stem>.png          generic PuLID doll (pre-swap)
  swapped/<stem>.png        identity-swapped doll
manifest_swap_sweep.parquet
```

`<stem>` = `<anchor>_<finish>_cn<NNN>_ps<NNN>_seed<N>`. Manifest columns:
`cell, anchor, finish, cn_strength, pulid_start, pulid_weight, seed,
doll_png, swapped_png, swap_mode, swap_det_score, workflow_version`.

## Error handling

- **ComfyUI HTTP blips** — `_retry` (3 tries, backoff), already in the
  helpers. Terminal failure: log, mark the cell failed, continue.
- **SCRFD miss on the doll** — MediaPipe forced fallback; both miss → keep
  un-swapped doll, `swap_mode="failed"`. The sweep never aborts on a swap
  miss.
- **Resumable** — skip-if-exists on `swapped/<stem>.png`. Atomic temp-rename
  writes so a killed run leaves no half-PNG masquerading as done.
- **Crash recovery** — the runner is launched as a Windows Scheduled Task
  (survives logout / SSH disconnect), same pattern as the InfiniteYou
  download. Per-cell progress is the presence of the output PNG. The manifest
  carries swap results (`swap_mode`, `swap_det_score`), so it is written
  incrementally — appended after each cell and on exit — not up front.

## Testing

`tests/test_swap_core.py`, runnable on the Windows box (needs the onnx
models, no GPU):

- `swap_identity` on the existing `swap_pw150` doll + `id_03` source returns
  `mode == "default"`, `det_score > 0.5`, and an output of the doll's shape.
- `detect_source` on a real photo returns a `Face`; on a blank image returns
  `None`.
- `mediapipe_kps_bbox` returns a `(5, 2)` kps array on a doll image when
  MediaPipe is available; the import guard is exercised.

Pre-flight asserts in the runner are the integration-level test: a missing
anchor or unreachable ComfyUI fails before any compute is spent.

Post-sweep, a small `matryoshka_swap_montage.py` builds per-anchor contact
sheets (doll vs swapped, across finishes) for eyeball review — separate from
the runner, run once at the end.

## Out of scope

Low-denoise repaint pass; the swap-then-warp (option C) path; InfiniteYou as
an identity vehicle; the nested-doll v2. Each is its own later experiment.
