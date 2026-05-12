# liveportrait-stylized

**Status:** live, active. Generation phase Q2 2026.

The thread to build a LivePortrait-fast renderer that tolerates stylized
anchors. Photoreal LP is fast and works; PersonaLive on heavy stylization
collapses, and the painting-mode workarounds are too slow for consumer GPUs.
The product gap is "LP latency, painting/anime quality."

## Current belief

The photoreal-prior in LP is localized to **G's SPADE γ/β projection
layers** (channel-affine, fine-layer texture statistics). F is partially
robust, M is style-agnostic, W's failures on stylized inputs are *anatomical*
(non-human morphology like muzzles, horns, long beards) not stylistic.

The right surgical target is therefore a **G-LoRA** trained on paired
(photoreal, styled) stills of the same identity. Identity coupling is
the keystone constraint — without it the LoRA conflates style transfer with
identity drift.

Identity coupling is supplied by **shared StyleGAN2 W vectors** across
FFHQ-StyleGAN2-transfer-trained heads (MetFaces, Ukiyoe, DualStyleGAN's 10,
NeuralKuvshinov, etc.). The Civitai Flux/SDXL LoRA ecosystem is reached
*indirectly* via the importer bridge — Flux+PuLID+CN generates curated
reference stills, JoJoGAN or DualStyleGAN fits a new W-compatible StyleGAN2
style head from those references, downstream cascade resumes on the clean
W-pair channel.

Anthro/furry stylization breaks the bridge (FFHQ alignment requires human
face landmarks). That remains a separate, harder problem on the LP-animals
branch, post-v1.

## Load-bearing docs

| Date | Doc | Role |
|---|---|---|
| 2026-05-08 | `2026-05-08-stylized-liveportrait-renderer.md` | Initial scoping. Three-path proposal A/B/C. Animals-mode as existence proof. |
| 2026-05-08 | `2026-05-08-face-image-vocabulary-latent.md` | Foundations: image/face/mascot, latent topology, FLAME/ARKit as engineered conventions not natural axes. |
| 2026-05-08 | `2026-05-08-stylization-locus-and-transfer.md` | **Supersedes initial scoping.** Mechanics: SPADE γ/β locus, transfer recipes, layered surgical targets. |
| 2026-05-08 | `2026-05-08-stylized-renderer-session-handoff.md` | End-of-session handoff. Available models, keystone pipeline, Spike 0.5 specification, open A/B/C decision. |
| 2026-05-12 | `2026-05-12-flux-to-stylegan2-importer-bridge.md` | **Adds the importer bridge.** Flux LoRA → JoJoGAN/DualStyleGAN → W-compatible head. One-week office-PC generation plan. |
| 2026-05-12 | `2026-05-12-flame-for-stylized-anchors.md` | Adjacent: LAM (SIGGRAPH 2025) verdict for FLAME-morphology stylized humanoids (orc/demon pass; anime/non-human out). Complementary pillar to the LP-G LoRA path. |

Sub-research outputs cited by the 2026-05-08 synthesis: `docs/research/_parts/`
(latent-topology, flame-3dmm, arkit-facs, overlap-question, style-locus,
face-domain-adaptation, peft-image-gen).

## Open decisions

- **Product shape A vs B vs C.** A=per-anchor 30-min fit. B=pre-shipped style packs. C=hybrid auto-classify + optional fine-tune. The importer bridge (2026-05-12) materially strengthens B by lifting its ceiling from ~5 to ~unbounded style packs. Decision still pending user input.
- **JoJoGAN vs DualStyleGAN as production importer.** JoJoGAN is the first-spike cheap variant. DualStyleGAN is the architectural production target if we ship ≥3 packs.

## Spike ladder (current)

| # | Spike | Status | Notes |
|---|---|---|---|
| 0a | Model-soup α between LP-human and LP-animals G | not run | Cheapest possible probe. Likely negative but free. |
| 0b | Inverse Pinkney in LP-G (coarse human + fine animals) | not run | |
| 0c | Forward Pinkney (coarse animals + fine human) | not run | |
| 0.5 | **Paired-synthetic + LoRA-G via MetFaces (W-pair, no importer)** | not run | **Keystone.** Tests G-LoRA mechanism on cleanest possible data. |
| **0.6** | **First importer spike: oil-painting Flux LoRA → JoJoGAN → 100K pairs → G-LoRA** | not run | Tests the bridge on mild style. Runs after 0.5 passes. |
| Refs | One-week office-PC reference-set generation (5–7 packs) | **in flight (2026-05-12 plan)** | Inputs to 0.6 and follow-ons. |
| 1 | Single-anchor JoJoGAN-style 30-min LoRA fit | not run | Per-anchor path A fallback. |

## Falsified / dead-end (do not re-explore)

- **Direct FLAME/ARKit conditioning of a learned face generator.** GIF 2020 falsified. Generators want pixel-space hints, not abstract parameter vectors. Our shipped ARKit→bridge→implicit-keypoints architecture is on the correct side of this.
- **Diffusion-based stylized renderers as speed competitors** (X-Portrait, AniPortrait). Inference-cost-disqualified; remain useful as quality references and synthetic-data generators.
- **No public stylized LP variant exists** (confirmed via search). LP-animals is the only LP-family fine-tune available.
- **Direct Track 2 (Flux+CN+PuLID+LoRA paired generation for LP-G training)** — not falsified, but downgraded to "expansion track" by the 2026-05-12 importer bridge, which dominates on identity quality at lower per-usable-pair compute.
