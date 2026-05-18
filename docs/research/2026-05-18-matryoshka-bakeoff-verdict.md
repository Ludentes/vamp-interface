---
status: live
topic: arkit-controlnet
---

# Matryoshka fast-model bake-off — verdict

Bake-off design: `docs/superpowers/specs/2026-05-18-matryoshka-fast-model-bakeoff-design.md`.
Ran on the Windows 3090, 143 ok cells (140-cell grid + a few smoke cells),
0 failed. Harness `scripts/matryoshka_bakeoff_sweep.py`, montage
`scripts/matryoshka_bakeoff_montage.py`. Artifacts in
`exp_output/matryoshka_bakeoff/` (`manifest.parquet`, `renders/`, `montage.png`).

## Timing

Warm server-side generation (full graph, ComfyUI `/history` timestamps):

| arm            | steps | warm gen | cold load |
|----------------|-------|----------|-----------|
| flux_krea      | 8     | 12.8 s   | ~78 s     |
| flux_krea      | 15    | 22.5 s   |           |
| flux_krea      | 20    | 29.5 s   |           |
| flux_schnell   | 2     | 4.6 s    | ~86 s     |
| flux_schnell   | 4     | 7.2 s    |           |
| flux_schnell   | 8     | 12.4 s   |           |
| sdxl_lightning | 4     | 3.2 s    | ~31 s     |
| sdxl_lightning | 8     | 4.3 s    |           |
| zimage_turbo   | 6     | 7.2 s    | ~65 s     |
| zimage_turbo   | 8     | 9.2 s    |           |
| zimage_turbo   | 12    | 13.2 s   |           |

SDXL-Lightning is the speed floor: ~3 s/render warm, ~31 s cold load (smallest
checkpoint). Everything else pays a 65–86 s one-time cold load.

## Quality

- **flux_krea (control)** — best doll: photoreal lacquered surface, coherent
  single doll, clean khokhloma florals. Slow (22 s median). Large realistic
  eyes, acceptable.
- **flux_schnell** — unstable at low steps. 2-step output is dark/noisy garbage;
  4-step renders *multiple nested dolls* (Canny silhouette read as a stack).
  Only 8-step gives a clean single doll — at which point it is no faster than
  flux_krea 8-step (~12 s) for visibly lower fidelity.
- **sdxl_lightning** — fastest, but renders a flat 2D *illustration*, not a
  photoreal object, and still paints big black doll-eyes (confirms the
  eye-problem is **not** FLUX-specific — it is a matryoshka-prior artifact).
- **zimage_turbo** — clean, coherent, photoreal-ish single doll at every step
  count tested (6/8/12), even though it is **prompt-only** (no Canny ControlNet
  exists for Z-Image). 6-step / 7.2 s is the sweet spot.

## Swap test (2026-05-18, follow-up)

The first verdict rejected sdxl_lightning on the assumption that flat
illustration output is "unsuitable for face-swap input". That was tested
directly — `scripts/matryoshka_bakeoff_swap_test.py` runs the full swap_core
pipeline (MediaPipe kps → eye-collapse → inswapper_128) on representative
zimage / sdxl dolls; see `swap_test.png` and `swap_zoom.png`.

**The swap works on both arms.** SCRFD never sees the painted doll face on
either arm (every cell falls back to MediaPipe `forced` mode — same as the
PuLID pipeline), the eye-collapse fires, and inswapper plants a real identity
face in the doll's face region. So "unsuitable for swap" was wrong.

What remains is an *aesthetic* difference, not a mechanical failure:
- **zimage_turbo** — photoreal doll body, so the swapped photoreal face blends
  into a consistent material; the result reads as one object.
- **sdxl_lightning** — flat-illustration doll body, so a photoreal swapped face
  sits inside a 2D-illustration figure: a visible style seam. Acceptable if the
  product wants an illustrated doll with a photo face; jarring if it wants a
  coherent photoreal object.

## Verdict

For the swap-only pipeline (generation makes a generic doll, inswapper adds
identity downstream):

**Z-Image Turbo at 6 steps is the recommended fast arm.** 7.2 s warm —
**3.1× faster than the flux_krea baseline** (22.5 s) — with a coherent
photoreal single doll. It loses Canny silhouette control, but a *generic* doll
does not need pose locking; the prompt alone holds the matryoshka form.

flux_schnell is rejected: only usable at 8 steps, where it is no faster than
the baseline and lower quality. sdxl_lightning is **viable** — the swap works
and at 3 s it is the speed floor — but it produces an illustrated doll with a
photo face rather than a coherent photoreal object. Keep it as the fast/stylized
option; Z-Image is the default for a photoreal result.

Open follow-up: confirm inswapper identity transfer succeeds on a Z-Image
6-step doll (the swap stage was assumed, not yet tested end-to-end on this arm).

## Swap rebuild (2026-05-18)

The swap test above confirmed the swap *fires* but identity barely *read* —
the doll face is a small patch of an 864×1152 image, so SCRFD never detects it
and inswapper aligns from a handful of pixels. `swap_core.swap_identity` was
rebuilt (design: `docs/superpowers/specs/2026-05-18-matryoshka-swap-rebuild-design.md`,
plan: `docs/superpowers/plans/2026-05-18-matryoshka-swap-rebuild.md`): crop the
doll face → upscale the crop to 512 px → detect/collapse/swap on the isolated
high-res crop → feathered paste-back. `matryoshka_bakeoff_swap_test.py` now
reports an ArcFace identity cosine (`swap_test_scores.csv`).

**Result — the restructure works.** Median identity cosine **0.719** over 8
measured cells across both arms, vs near-random for the pre-rebuild pipeline.
SDXL-Lightning 4-step now detects the crop face with SCRFD (`mode=default`,
det 0.55) instead of always falling back to forced kps — the crop-upscale is
exactly the "help insightface" lever. Per-arm:

| arm            | step | id_03 | id_11 | mode    |
|----------------|------|-------|-------|---------|
| zimage_turbo   | 6    | 0.813 | 0.776 | forced  |
| zimage_turbo   | 8    | 0.665 | 0.625 | forced  |
| zimage_turbo   | 12   | 0.620 | 0.392 | forced  |
| sdxl_lightning | 4    | 0.823 | 0.773 | default |
| sdxl_lightning | 8    | nan   | nan   | failed  |

`nan` = the swap itself failed: sdxl 8-step renders no MediaPipe-detectable
doll face at all, so `swap_identity` returns the doll unchanged
(`mode=failed`) — the regression guard fired cleanly, no crash. The
identity-cosine metric mirrors the swap's detection path (shared
`crop_and_upscale`) and falls back to a forced MediaPipe Face through the
recognition model when SCRFD misses the output face, so a measured `nan` now
means only a genuinely failed swap, not a SCRFD-weak one.

**GFPGAN restoration falsified as an identity step.** The design proposed a
GFPGAN restore pass after the swap. A/B measured it *lowering* median identity
cosine **0.773 → 0.525** — GFPGAN regularizes the swapped face toward a generic
restoration prior, beautifying the identity away. `restore` now defaults
`False`; the crop→upscale restructure alone carries the lift. GFPGAN stays
behind the `restore` flag (CPU-pinned, lazy) but is off by default.
