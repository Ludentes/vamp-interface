---
status: live
topic: arkit-controlnet
---

# InfiniteYou matryoshka workflow design

**Date:** 2026-05-17
**Goal:** Render an identity-preserving single Russian matryoshka doll from a
person's photo, using InfiniteYou (InfuseNet) for identity instead of PuLID.
PuLID failed to transfer identity through the flat-painted matryoshka style;
InfuseNet's ArcFace-residual injection is the next attempt.

## Context

The matryoshka v1 sweep (`scripts/matryoshka_sweep.py`) used Flux-Krea + PuLID
+ a fixed Canny doll-silhouette template. PuLID could not push identity through
the flat-painted-face style — the doll looked generic regardless of
`pulid_weight`. The fallback was a CPU `inswapper_128` face-swap on top of the
generic doll (`scripts/swap_core.py`, `matryoshka_swap_sweep.py`), which works
but is a post-hoc paste, not native generation.

InfiniteYou is the native-identity alternative. All weights and the
`ComfyUI_InfiniteYou` node are installed locally (see
`docs/research/2026-05-16-infiniteyou-asset-report.md`). ComfyUI is running on
the local RTX 5090 at `127.0.0.1:8188`.

## Approach selection

**Chosen: new API-format workflow JSON + a new sweep script** cloned from
`matryoshka_sweep.py`. The existing sweep harness (manifest, resumable
skip-if-exists, atomic PNG writes, retry/timeout, `SCHEDULE_NODES` assertion)
is exactly what we want; only the node graph and the swept axes change.

Rejected alternatives:
- *Manual single renders in the ComfyUI UI* — not reproducible, no manifest,
  no seed determinism. Fine for one smoke test, not for the sweep.
- *Build the workflow dict inline in Python* — more code, the graph is no
  longer inspectable as a standalone artifact, and it diverges from the
  established `*.api.json` template pattern in this repo.

## The node graph

InfuseNet is a ControlNet generalization. The official `InfuseNetApply` node
reads `d.get('control')` off the incoming conditioning and calls
`set_previous_controlnet(prev)` — so a **normal Canny ControlNet applied first
chains natively into InfuseNet**. That is the whole feasibility argument: the
doll *structure* path (Canny) and the *identity* path (InfuseNet) compose
without a custom node.

Graph (API-format node IDs are assigned when the JSON is authored):

```
UNETLoader(flux1-dev-fp8)                         ─┐
DualCLIPLoader(t5xxl_fp8, clip_l, flux)           ─┤
VAELoader(ae.safetensors)                         ─┤
                                                   │
CLIPTextEncodeFlux(positive matryoshka prompt) ──┐ │
CLIPTextEncode(negative)                       ──┤ │
                                                 │ │
ControlNetLoader(InstantX Union)  ──┐            │ │
LoadImage(template_canny.png)     ──┤            │ │
ControlNetApplyAdvanced(pos, neg, canny_cn,      │ │
        canny_image, str, start, end) ──> (pos', neg')
                                                 │ │
IDEmbeddingModelLoader(sim_stage1/image_proj) ──┐│ │
LoadImage(identity id_NN.png)                  ─┤│ │
ExtractIDEmbedding(detector,arcface,proj,img) ──> id_cond
                                                 ││ │
InfuseNetLoader(sim_stage1/infusenet_sim_fp8) ──┐││ │
EmptyImage(black, W×H)  ── control image        │││ │
InfuseNetApply(pos', neg', id_cond, infusenet_cn,│││ │
        black_image, vae, str, start, end) ──> (pos'', neg'')
                                                     │
KSampler(unet, pos'', neg'', empty_latent, seed) ──> latent
VAEDecode ──> SaveImage
```

Key decisions:
- **Base model: FLUX.1-dev fp8** (`flux1-dev-fp8.safetensors`). InfuseNet
  residuals are dev-aligned. The ByteDance ComfyUI example ships a schnell
  graph for speed; we override to dev for identity fidelity. Sampler: 20 steps,
  `euler`/`beta`, CFG 1.0, CLIPTextEncodeFlux guidance 3.5.
- **InfuseNet variant: `sim_stage1`** (identity-favoring, not `aes_stage2`).
- **Control image for InfuseNet: a black `EmptyImage`** (no 5-keypoint pose
  constraint). We do *not* want the real face's pose imposed on a doll; the
  doll form comes entirely from the Canny path. `ExtractFacePoseImage` is not
  used.
- **Canny path = the existing fixed doll-silhouette template**
  (`data/importer/refs/matryoshka/template_canny.png`), strength fixed at 0.5,
  `start/end = 0.0/0.5` — the structure-then-release schedule proven in the
  PuLID sweep.
- **Resolution 864×1152** (portrait, matches the InfiniteYou example latent
  and the doll's tall aspect).

## The sweep

Reuse the `matryoshka_sweep.py` harness. New script
`scripts/matryoshka_infiniteyou_sweep.py`. Swept axes — InfuseNet strength is
the identity dial (the analogue of `pulid_weight`):

| Axis | Values | Rationale |
|---|---|---|
| identity | id_03, id_08, id_14, id_15 | same 4 as the PuLID sweep; id_14 = hand-picked reference |
| infusenet_strength | 0.6, 0.8, 1.0 | identity-injection strength |
| infusenet_end | 0.5, 0.8 | how late in the schedule identity keeps pulling |
| seeds | 2 per cell | deterministic: `seed_base + render_index * 7919` |

Fixed: Canny strength 0.5, Canny start/end 0.0/0.5, InfuseNet start 0.0.
Grid = 4 × 3 × 2 × 2 = **48 renders**. At ~15 s/render on the 5090, ~12 min.

Manifest written up front (parquet). Resumable. Output PNGs to
`data/importer/refs_matryoshka_infu/`.

## Identity-image precondition

`ExtractIDEmbedding` runs insightface antelopev2 ArcFace on the identity PNG
and **raises if no face is detected**. The 4 chosen identities are real
portraits with detectable faces (the swap sweep's no-face failures were
id_19-class anchors, not these). The sweep script will pre-flight each
`id_NN.png` through a one-line `FaceAnalysis.get()` before queuing and skip +
log any that fail, so a bad identity doesn't abort the run.

## Error handling

- Workflow node-ID assertion at startup (`SCHEDULE_NODES`-style) so a
  re-exported/renumbered workflow fails loud instead of injecting into the
  wrong node.
- Per-render try/except with retry on HTTP blips (inherited from the harness).
- Atomic PNG writes; skip-if-exists for resume.
- `arcface` recognition weight auto-downloads via facexlib on first run; the
  first render will be slow — expected, not an error.

## Testing

- **Smoke test:** one render (id_14, strength 0.8, end 0.5, one seed) before
  the full sweep — confirms the graph loads, all weights resolve, a face is
  detected, and a PNG comes back.
- **Sweep:** the 48-cell grid; eyeball the montage for identity transfer
  through the flat-paint style (the thing PuLID failed).
- **Comparison:** montage InfiniteYou results next to the PuLID-sweep and the
  inswapper-swap results for the same 4 identities.

## Out of scope (YAGNI)

- The nested doll set (v2).
- `aes_stage2` variant, bf16 precision, multi-ID.
- Windows-box execution — kit is staged there but node install + testing is a
  later step.
