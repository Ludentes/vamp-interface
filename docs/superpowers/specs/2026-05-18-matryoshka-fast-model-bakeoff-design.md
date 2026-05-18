---
status: live
topic: arkit-controlnet
---

# Matryoshka fast-model bake-off — design

## Why

Matryoshka doll generation on FLUX.1-Krea-dev costs ~30 s/render on a 3090.
The Windows 3090 is free over the weekend. Test whether a faster 2026-era
base model can generate the *generic doll* at materially lower latency
without losing matryoshka folk-art quality.

## Frame: generation does not need identity

The matryoshka pipeline is two stages:

1. **Generation** — make a generic painted doll. Canny locks the doll
   silhouette; the prompt supplies khokhloma folk-art style.
2. **Swap** — `inswapper_128` paints a real identity onto the doll face.

Identity lives entirely in stage 2. So "faster model" only has to mean a
faster *generic-doll* generator. Identity-at-generation (PuLID / InfiniteYou)
is explicitly **out of scope** for this bake-off — confirmed with the user.

This makes the bake-off cheap: every arm needs only a Canny ControlNet,
which ComfyUI drives natively (no custom nodes). The swap stage is
model-agnostic and already works on any RGB doll.

## Arms

All weights already present on the Windows box — no downloads.

| Arm | Base | Steps | Canny | Notes |
|-----|------|-------|-------|-------|
| `flux_krea` *(control)* | `FLUX1/flux1-krea-dev_fp8_scaled` | 20 | FLUX Canny CN | current baseline |
| `flux_schnell` | `flux1-schnell-fp8` | 4 | FLUX Canny CN | distilled, same family |
| `sdxl_lightning` | `sd_xl_base_1.0` + `sdxl_lightning_8step_lora` | 8 | SDXL Canny CN | cross-family, fast |
| `zimage_turbo` | `z_image_turbo_bf16` | ~8 | none — **prompt-only** | 2026 model; structure unconstrained |

`flux-2-klein-4b` is also on the box; held as an optional 5th arm.

## Held constant

- Same Canny doll template (`matryoshka_template_canny.png`) for the three
  Canny arms. Z-Image runs prompt-only — flagged in the verdict, not a bug.
- Prompt: the matryoshka base + khokhloma style suffix, adapted per family
  (FLUX/Z-Image take long natural prompts; SDXL takes comma-tag style).
- 2 seeds per arm.
- 4 downstream identities for the swap stage.

Grid: 4 arms x 2 seeds = 8 doll renders -> swap 4 identities -> 32 swapped.

## Metrics

- **Time** — wall-clock seconds per render, logged to the manifest.
- **Quality** — eyeball montage scored on:
  - matryoshka / khokhloma style fidelity
  - Canny structure adherence (n/a for Z-Image)
  - **natural eye size** — does the model paint small swap-friendly eyes, or
    the FLUX big-black-eye blobs that need `collapse_eyes`?
  - post-swap identity read

## Hypothesis worth flagging

The big-black-eye problem may be a FLUX-family artifact. SDXL and Z-Image
may paint smaller, more naive folk-art eyes and sidestep the `collapse_eyes`
pre-step entirely. If so, a faster base also simplifies the swap stage.

## Out of scope

- Identity-at-generation (PuLID / InfiniteYou) — separate thread.
- SDXL IP-Adapter FaceID — would be new custom-node infra.
- Z-Image ControlNet — does not exist yet; revisit if released.

## Deliverables

- Per-arm ComfyUI API workflows under `comfyui/workflows/`.
- `scripts/matryoshka_bakeoff_sweep.py` — Windows-runnable, resumable,
  logs render time per cell.
- `scripts/matryoshka_bakeoff_montage.py` — time-vs-quality comparison grid.
- Verdict appended to `docs/research/`.
