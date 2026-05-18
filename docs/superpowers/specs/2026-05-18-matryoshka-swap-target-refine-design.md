---
status: live
topic: arkit-controlnet
---

# Matryoshka swap-target refine pass — design

## Why

The matryoshka pipeline is swap-only: a fast model generates a generic doll,
`inswapper_128` swaps a real identity onto its face downstream. The swap
rebuild (`2026-05-18-matryoshka-swap-rebuild-design.md`) restructured the swap
to crop→upscale→swap→paste-back and lifted median ArcFace cosine to ~0.72 —
but identity still does not read convincingly.

The cause is the *target* the swap is given. A fast model renders a matryoshka
with a **flat folk-art painted face**: oversized eyes, no nose modeling, a flat
mouth, no shading. SCRFD cannot detect it (every Z-Image cell falls back to
`forced` MediaPipe keypoints), and even when keypoints land, the face crop is
far outside the ArcFace template distribution, so `inswapper` aligns and blends
poorly. Generation-time identity injection is *not* the fix — the PuLID weight
ladder already falsified that (identity dies in the flat paint even at weight
4.0), and InfiniteYou/InfuseNet runs only on Flux, not the fast arms.

The fix is to give `inswapper` a **better target**: re-render only the doll's
face region into an actual face-like structure — proper proportions, modeled
nose and lips, natural-sized eyes, soft shading — so SCRFD detects it natively
and the swap aligns from an in-distribution face. Identity still comes entirely
from the downstream swap; this pass is identity-blind.

## Goal

A pre-swap **face-region inpaint refinement** stage: take a generated doll,
mask its face region, re-diffuse only that region with a generic
"realistic-face-structure" prompt, and emit a swap-ready doll. Must serve both
fast arms — Z-Image Turbo (prompt-only) and SDXL-Lightning. Success: SCRFD
detects the refined face (`mode=default` instead of `forced`) on a clear
majority of cells, and post-swap median identity cosine rises materially over
the 0.72 baseline.

## Approaches considered

**A. Face-region inpaint pass in a ComfyUI workflow (chosen).** After the
first-pass doll is generated, detect its face region with MediaPipe, build a
feathered mask, and run a masked img2img (latent noise mask, `denoise` < 1.0)
that re-diffuses only that region against a realistic-face-structure prompt.
Output is a swap-ready doll PNG; the existing `swap_core` pipeline then swaps
unchanged. The masked region is exactly the user's "specify a small part of
the image for the model to work on" — focused compute that pushes the face
hard without disturbing the doll body or lacquer. `denoise` is the
flat-paint↔face-structure dial. Model-agnostic at the ComfyUI level (a latent
noise mask works for any latent diffusion model), so one workflow shape covers
both Z-Image and SDXL. Keeps the clean split: ComfyUI produces swap-ready
dolls, `swap_core` stays a pure CPU detect-and-swap module.

**B. Crop→img2img→swap inline in `swap_core`.** `swap_core` crops the face,
round-trips the crop through a ComfyUI img2img call, then swaps. Rejected: it
couples the CPU-only swap module to a GPU service, and the swap-rebuild design
already rejected putting img2img inside `swap_core` for this reason.

**C. Full-image low-denoise img2img.** Re-diffuse the whole doll at low
`denoise` with a face-emphasis prompt. Rejected: a full-image pass either
disturbs the doll body and lacquer or is too weak to restructure the face —
the user explicitly wants the *regional* lever. Masking is the point.

## Architecture

Three units. Generation-side mask building is kept separate from `swap_core`
(swap-side); the inpaint runs in ComfyUI; a driver orchestrates.

### `scripts/face_region.py` (new)

One responsibility: turn a doll image into a face-region inpaint mask.

- `build_face_mask(doll_bgr, feather_frac=0.10) -> np.ndarray | None` —
  MediaPipe-detects the doll face (reuses `swap_core.mediapipe_kps_bbox`),
  expands the bbox with `swap_core._crop_region`, and returns a single-channel
  `uint8` mask the size of the full image: white over the (feathered) face
  region, black elsewhere. Returns `None` if no face is detected — the caller
  then skips refinement for that doll and passes it through unchanged.
- The feather avoids a hard inpaint seam at the mask border.

### ComfyUI inpaint workflow

`comfyui/workflows/matryoshka_inpaint.api.json` (new) — a masked img2img
graph: load the doll image + the mask, VAE-encode with the mask as a latent
noise mask, sample at a parameterized `denoise` against a fixed
realistic-face-structure positive prompt, decode, save. The driver patches the
checkpoint/sampler nodes per arm (Z-Image Turbo vs SDXL-Lightning), the
`denoise` value, and the seed. One graph, arm-specific node values — the same
pattern as the bake-off sweep workflows.

Inpaint prompt (fixed, identity-blind): a realistic painted human face, soft
three-dimensional shading, correct facial proportions, defined nose and lips,
natural-sized eyes — pushing *structure*, not photorealism, so the result
still belongs on a wooden doll.

### `scripts/matryoshka_refine.py` (new)

Driver. Input: a directory of first-pass doll PNGs (e.g. the bake-off
`renders/`). For each doll: `build_face_mask`; if `None`, copy through; else
upload doll + mask to the ComfyUI input dir, run `matryoshka_inpaint` for the
doll's arm at the configured `denoise`, save the refined doll to an output
directory. Resumable (skip-if-exists), atomic writes, manifest up front —
matching the existing sweep harnesses.

## Data flow

```
first-pass doll PNG
  → build_face_mask  (MediaPipe detect → _crop_region → feathered mask)
  → ComfyUI matryoshka_inpaint  (masked img2img, denoise dial)
  → refined swap-ready doll PNG
  → [existing] swap_core.swap_identity  → swapped result
```

## Evaluation

Extend `matryoshka_bakeoff_swap_test.py` to run on a refined-doll directory and
report, against the un-refined baseline:

- **SCRFD detection rate.** Fraction of cells where `swap_identity` returns
  `mode=default` (SCRFD saw the face) vs `forced`. This is the direct
  "good-enough-for-insightface" metric — the refine pass must move cells from
  `forced` to `default`.
- **Identity cosine.** Median ArcFace `id_cos` (the metric added in the swap
  rebuild), before vs after refinement, per arm. Must rise over ~0.72.
- **`denoise` sweep.** A small ladder (e.g. 0.4 / 0.55 / 0.7) to locate where
  the face becomes a good swap target without the region drifting off the
  doll. One identity, both arms.

Success: detection rate and median `id_cos` both rise materially on both arms;
the montage shows a face that reads as a modeled face rather than flat paint.

## Error handling

- No MediaPipe-detectable face on the first-pass doll → `build_face_mask`
  returns `None`; the driver copies the doll through unrefined (the swap then
  behaves exactly as today — no regression).
- ComfyUI unreachable or the inpaint job fails → the driver logs and skips
  that cell, leaving the un-refined doll; resumable rerun picks it up.
- Refined face still undetected by SCRFD → the existing swap `forced` fallback
  still fires; the pass degraded gracefully, just with no detection win.

## Out of scope

Generation-time identity injection (PuLID/InfuseNet — falsified or
Flux-only); a post-swap img2img blend pass (the user noted it as an option to
remember — recorded here, not built); retraining; a full new generation
sweep — this refines existing bake-off dolls.
