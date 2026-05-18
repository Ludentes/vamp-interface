---
status: live
topic: arkit-controlnet
---

# Matryoshka identity-swap rebuild — design

## Why

The matryoshka pipeline is swap-only: a generic doll is generated, then a real
identity is swapped onto its painted face by `inswapper_128`. The swap test
(`docs/research/2026-05-18-matryoshka-bakeoff-verdict.md`) confirmed the swap
fires on both the Z-Image and SDXL bake-off arms — but identity barely reads.

Root cause: the doll face is a small patch of an 864×1152 image. `inswapper_128`
aligns the target to a 128 px template; from a small face it has almost no
pixels to work with, SCRFD cannot even detect the face (every cell falls back
to synthetic MediaPipe keypoints), and the 128 px swap output pasted back onto
a small region is low-detail mush. The 2026 tooling survey
(`general-purpose` research, this session) found no swapper that materially
beats `inswapper_128` on identity — the fix is a **pipeline restructure**, not
a new model.

## Goal

Real identity reads on the swapped doll face, for both the Z-Image Turbo
(photoreal doll) and SDXL-Lightning (flat-illustration doll) bake-off renders.
Lightweight, no training. The downstream swap is the *only* identity path for
the Z-Image arm — no identity adapter exists for the Lumina2/NextDiT family —
so this rebuild serves both arms.

## Approaches considered

**A. Restructure `swap_identity` in-process (chosen).** Crop the doll face,
upscale the crop to 512 px, run detection + swap on the isolated high-res crop,
restore it with a face-restoration net, downscale, and paste back under a
feathered mask. Keeps `swap_core.py` as the self-contained detect-and-swap
module its consumers already call; keeps the swap on CPU (no ComfyUI GPU
contention); restoration loads lazily on GPU. The crop-upscale is what lets
SCRFD detect the face (a 512 px isolated face is far more in-distribution than
a small face in a scene) — that is the "help insightface" win — and gives the
restorer a real 512 px canvas to hallucinate skin micro-detail onto.

**B. Move swap+restore into a ComfyUI ReActor node.** ReActor bundles
crop/restore/paste-back. Rejected: it moves the swap onto the GPU where it
contends with generation, adds a heavy custom-node dependency, and forces the
Python sweep harness to round-trip through ComfyUI workflows for a stage that
is currently clean Python.

**C. Bolt a restore pass onto the unchanged `swap_identity` output.** Rejected:
without crop-upscale the swap still aligns from a handful of pixels; restoring
a tiny mushy face only sharpens the mush. The crop-upscale must happen *before*
the swap for the restore to have signal to work with.

## Architecture

Two modules. `swap_core.py` is extended; `face_restore.py` is new and small.

### `scripts/face_restore.py` (new)

One responsibility: turn a swapped face crop into a sharp, realistic one.

- `restore_face(crop_bgr) -> np.ndarray` — lazy-loads the restoration net on
  first call, caches it, returns a restored BGR image the same size as the
  input. If the restorer is unavailable (import or weights failure) it logs
  once and returns the input unchanged — restoration is an enhancement, never
  a hard dependency.
- Primary restorer: **GFPGAN v1.4** (`gfpgan` 1.3.8 on PyPI). Chosen over
  CodeFormer because the goal is *identity preservation* — the 2026 survey
  rates GFPGAN as the most conservative restorer (least likely to drift to a
  different person), where CodeFormer's `w` blend can beautify identity away.
  The restorer is isolated behind `restore_face`, so swapping in CodeFormer
  later is a one-function change.
- **Install hazard (must be handled in the plan):** `gfpgan` pulls `basicsr`,
  which imports `torchvision.transforms.functional_tensor` — removed in modern
  torchvision. The plan must pin/patch this (the standard one-line shim) and
  verify the import in a smoke step before any pipeline work.

### `scripts/swap_core.py` (extended)

Unchanged and reused: `make_face_app`, `load_swapper`, `detect_source`,
`mediapipe_kps_bbox`, `collapse_eyes`.

New private helpers:

- `_crop_region(bbox, img_shape, margin_frac=0.45) -> (x0,y0,x1,y1)` — a
  square region around the MediaPipe face bbox, expanded by `margin_frac` of
  the larger bbox side, clamped to the image.
- `_feathered_mask(h, w, feather_frac=0.12) -> float mask` — a rounded mask,
  1.0 in the interior, Gaussian-feathered to 0.0 at the edges, for seam-free
  paste-back.

Rewritten `swap_identity(app, swapper, doll_bgr, source_face, collapse=True,
restore=True) -> (result_bgr, mode, det_score)` — the signature its two
consumers (`matryoshka_swap_sweep.py`, `matryoshka_bakeoff_swap_test.py`)
already use, plus a `restore` flag defaulting True. New data flow:

1. `mediapipe_kps_bbox(doll_bgr)` → `kps_full, bbox_full`. If `None`, return
   `(doll_bgr, "failed", 0.0)` — unchanged behaviour.
2. `_crop_region(bbox_full, doll.shape)` → crop the doll face; record region
   and the upscale factor `s` to reach 512 px.
3. Upscale the crop with Lanczos so its **longer side is 512 px**, preserving
   aspect ratio (the region is square unless clamped at an image border).
4. Re-run `mediapipe_kps_bbox` on the 512 crop → crop-space `kps`. If
   `collapse`, run `collapse_eyes` on the crop with crop-space `kps`.
5. `app.get(crop)` — SCRFD on the isolated 512 crop. If it finds a face,
   `mode="default"`, use that `Face`. Else build a forced `Face` from the
   crop-space MediaPipe `kps`/`bbox`, `mode="forced"`.
6. `swapper.get(crop, target, source_face, paste_back=True)` → swapped crop.
7. If `restore`, `restore_face(swapped_crop)` → restored crop.
8. Downscale the restored crop back to the region size.
9. Composite into a copy of `doll_bgr` over the region using
   `_feathered_mask`.
10. Return `(result, mode, det_score)`.

### img2img glue — documented option, not built in

A low-denoise (~0.3) img2img pass over the pasted result re-melts the swap seam
into the doll's material. It needs ComfyUI (a Z-Image / SDXL img2img graph),
so it does **not** belong in `swap_core.py`. It is recorded here as the
designated next lever if the feathered paste-back still shows a seam, and will
be a documented optional stage in the eval — not implemented in this rebuild.

## Evaluation

`matryoshka_bakeoff_swap_test.py` is extended to be the quantitative harness:

- **Identity metric.** After the swap, re-detect the face on the *output* with
  `buffalo_l` and take its 512-d ArcFace embedding; report
  `cos(source_embedding, output_embedding)`. The pre-rebuild pipeline scores
  near-random; the rebuild must show a clear, consistent lift. Print a table
  (arm × step × identity → cosine, swap mode, det_score) and keep the
  doll-vs-swap montage (`swap_test.png`, `swap_zoom.png`).
- **Coverage.** Representative renders from both arms (Z-Image 6/8/12-step,
  SDXL 4/8-step) × ≥3 identities, so the metric is not read off one face.
- **Regression guard.** A render with no MediaPipe-detectable face must still
  return `(doll, "failed", 0.0)` and not raise.

Success: median identity cosine rises materially over the current pipeline on
both arms, and the montage shows a face that visibly reads as the source
identity rather than a generic blurred face.

## Error handling

- No detectable doll face → return the doll unchanged, `mode="failed"`.
- Crop region clamped at an image border → `_crop_region` clamps; a non-square
  clamped region is upscaled anyway (aspect handled per-axis).
- Restorer import/weights failure → `restore_face` logs once, returns input
  unchanged; the swap still completes.
- Restorer finds no face in the swapped crop → return the crop unchanged.

## Out of scope

InstantID at generation for the SDXL arm (additive, separate spec); any new
swapper model; training; the img2img glue stage.
