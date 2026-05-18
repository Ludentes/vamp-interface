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
  eye-problem is **not** FLUX-specific — it is a matryoshka-prior artifact). The
  illustration look is wrong for a swap pipeline: inswapper expects a
  photoreal-ish face region.
- **zimage_turbo** — clean, coherent, photoreal-ish single doll at every step
  count tested (6/8/12), even though it is **prompt-only** (no Canny ControlNet
  exists for Z-Image). 6-step / 7.2 s is the sweet spot.

## Verdict

For the swap-only pipeline (generation makes a generic doll, inswapper adds
identity downstream):

**Z-Image Turbo at 6 steps is the recommended fast arm.** 7.2 s warm —
**3.1× faster than the flux_krea baseline** (22.5 s) — with a coherent
photoreal single doll. It loses Canny silhouette control, but a *generic* doll
does not need pose locking; the prompt alone holds the matryoshka form.

flux_schnell is rejected: only usable at 8 steps, where it is no faster than
the baseline and lower quality. sdxl_lightning is rejected on quality despite
the 3 s speed: flat illustration output is unsuitable for face-swap input.

Open follow-up: confirm inswapper identity transfer succeeds on a Z-Image
6-step doll (the swap stage was assumed, not yet tested end-to-end on this arm).
