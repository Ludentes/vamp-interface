---
status: live
topic: neural-deformation-control
---

# Loss zoo comparison: LivePortrait vs our slider stack

**Date:** 2026-05-03
**Purpose:** put the LivePortrait Stage 1 + Stage 2 loss soup side-by-side with the loss terms our slider training thread has accumulated. Identify what each system uses, what's analogous, what's load-bearing on one side and missing on the other, and which terms are worth porting in either direction.

## LivePortrait loss zoo

LivePortrait trains in two stages; the loss soup is largest in Stage 1.

### Stage 1 — appearance + motion + warp + decoder, end-to-end

Eight loss terms, all summed with hand-tuned weights. The model in scope: appearance extractor 𝓕, motion extractor 𝓜 (ConvNeXt-V2-Tiny producing 21 implicit 3D keypoints + per-kp expression deltas + head rotation + scale + translation + eye/lip scalars), warping module 𝓦, SPADE decoder 𝓖.

| # | Term | What it pulls toward | Why it's there |
|---|---|---|---|
| 1 | **Keypoint equivariance** | Apply a random 2D affine to the source image, the recovered keypoints should follow the same affine | Prevents kp collapse, forces 𝓜 to be a real geometric estimator, not a content cheat |
| 2 | **Keypoint prior** | Keypoints should spread across the face (variance term + minimum pairwise distance) | Without this, all 21 kps collapse to the nose |
| 3 | **Head-pose loss** | Predicted rotation matches a pretrained head-pose estimator | Decouples global pose from expression in the kp representation |
| 4 | **Deformation prior** | L2 on the per-kp expression delta vector | Keeps expression small and centred; lets warp do most of the work |
| 5 | **Cascaded perceptual (×3)** | LPIPS-style VGG features, computed at global crop + face crop + lip crop, summed | Multi-scale supervision; lip crop ensures fine articulation isn't washed out by the global term |
| 6 | **Cascaded GAN (×3)** | Three discriminators (global, face, lip) each running a hinge GAN loss | Sharpens texture; the face/lip discriminators specifically prevent identity blur |
| 7 | **ArcFace face-id** | Cosine of ArcFace embedding(generated) vs ArcFace embedding(source) | Hard identity anchor. Without this LivePortrait would drift like every GAN before it |
| 8 | **Landmark-guided Wing** | Wing loss on detected landmarks (predicted vs ground-truth driving frame) | Robust localisation of mouth/eye landmarks; Wing handles small errors better than L2 |

### Stage 2 — retargeting MLPs

Trained with Stage 1 frozen. Three small MLPs (stitching, eyes retargeting, lip retargeting). Loss terms are simpler:

- **Closure regression** for eyes and lips — given a target eye/lip closure scalar, the MLP should produce kp deltas that achieve that closure.
- **Cross-identity stitching loss** — when source and driver are different people, the stitched composite should not show seams between the warped face region and the source background.

### What this soup is doing as a whole

Stage 1 is a multi-objective optimisation over four coupled goals: **(a) recover correct geometry** (#1, #2, #3, #4, #8), **(b) reconstruct pixels well** (#5), **(c) reconstruct them sharply** (#6), **(d) keep the same person** (#7). No single term carries the load — drop any one and the model degrades in a specific failure mode (drop #1 → collapse, drop #6 → blur, drop #7 → identity drift).

Stage 2 is small and surgical: hand-engineered control points (eye closure, lip closure, stitching) get their own dedicated regression heads.

## Our slider loss zoo

Our setup is Concept-Slider LoRA training on Flux Krea, with the slider authoring framework formalised in `2026-05-03-slider-operational-handbook.md`. Loss terms have accumulated over the v1 → v4 iterations.

### Direction term (always present)

The Concept Slider core: encourage the model under +slider to predict noise consistent with prompt B (target), under −slider with prompt A (anchor). Roughly:

```
L_direction = || ε_θ(z_t, +α; prompt_B) − ε_target_+ ||² + || ε_θ(z_t, −α; prompt_A) − ε_target_− ||²
```

Where `ε_target_±` come from the frozen base model's predictions at prompts A and B respectively. This is the "tell the model the target directly" piece per `project_editing_framework_principle`.

### Auxiliary objectives we layer on top

These are the "loss soup" we've built up. Most are evaluated by **distilled student networks operating directly on Flux's noisy latent** (the `*_distill` models) — no VAE decode in the training loop.

| # | Term | What it pulls toward | Status |
|---|---|---|---|
| D | **Direction loss** | Target prompt vs anchor prompt direction (above) | Always on |
| B | **bs_loss** (blendshape regression) | Predicted blendshape vector matches target (e.g. b_squint = 1.0, others = baseline) | Primary signal for bs_only sliders (v1k recipe). Computed by `mediapipe_distill` student on noisy latent. |
| A | **arc_loss** (identity) | ArcFace cosine to anchor stays high | Computed by `arc_distill latent_a2_*` student on noisy latent. Direct analogue to LivePortrait term #7. |
| S | **siglip_loss** (semantic) | SigLIP embedding stays close to anchor's | Computed by `siglip_distill` student. Soft semantic preservation; analogue to perceptual #5 but at semantic-not-pixel level. |
| P | **PGD adversarial inner loop** | Critic robustness — student perturbed by adversarial noise during training to prevent classifier-fooling exploits | New as of v4_pgd (mediapipe). Defends against the "wrong-but-sharp" failure mode. |
| G | **Step-gating** | Loss only computed in selected diffusion timesteps (e.g. t ≤ 0.5 for structural sliders, capped t_max ≈ 0.6 for v3_t after v1d falsification) | Not a loss term per se, but a critical scheduling lever. See `project_bs_lora_step_gating`. |
| M | **Per-channel R² mask** | Drop training contribution from channels where the critic's R²(t) is too low at the current timestep | Added in v3_t fix to `project_v1d_slider_falsified`. Roughly a confidence-weighted bs_loss. |

### What this soup is doing as a whole

The slider loop is solving a different optimisation than LivePortrait. It is a **conditioned editing problem on a pre-trained generator**, not a from-scratch training problem. The base model already knows how to produce sharp identities (Flux is a strong prior); we don't need GAN losses or perceptual losses to keep image quality up. We need three things: (a) **push the edit in the right direction** (D, B), (b) **don't break identity** (A), (c) **don't break unrelated semantics** (S), plus engineering layers to defend against critic-fooling (P) and channel-specific noise tolerance (G, M).

## Side-by-side mapping

| Goal | LivePortrait term | Our slider term | Notes |
|---|---|---|---|
| **Identity preservation** | #7 ArcFace face-id | A — arc_loss via arc_distill | Direct analogue. Both use ArcFace embeddings. We compute ours on noisy latent via distilled student; theirs operates on decoded RGB. |
| **Semantic / perceptual preservation** | #5 cascaded perceptual (LPIPS at 3 crops) | S — siglip_loss via siglip_distill | Loose analogue. LivePortrait's perceptual is local pixel-feature, ours is global semantic. We have *no* per-region (face/lip) variant. |
| **Sharpness / texture quality** | #6 cascaded GAN (×3) | (none) | We don't need it — Flux base is already sharp. But adversarial *training data* terms might be relevant; see Implications below. |
| **Geometry correctness** | #1, #2, #3, #4, #8 (kp equivariance, prior, head-pose, deformation prior, Wing landmark) | B — bs_loss via mediapipe_distill | We collapse all "did the geometry change correctly" into one bs_loss against MediaPipe ARKit-52. LivePortrait splits this into 5 separate terms because it has 21 kps + 6 DoF pose, not 52 named channels. |
| **Robustness / no exploits** | (none explicit; the GAN discriminators do this implicitly) | P — PGD adversarial inner loop | We have an *explicit* defence against critic-fooling. LivePortrait gets it for free from the GAN setup (any discriminator-fooling artefact gets penalised by the next discriminator update). |
| **Channel-specific weighting** | (none — uniform across kps) | M — per-channel R² mask, G — step-gating | We have channel-aware weighting because our targets are 52 named blendshapes with very different noise tolerances. LivePortrait's 21 kps are anonymous; no per-axis structure to exploit. |
| **Direction / target specification** | (none — driven by paired (source, driver) frames; the driver IS the target) | D — direction loss (Concept Slider core) | Fundamentally different mechanism. LivePortrait gets its target from a real frame; we synthesise the target via prompt A vs B. |
| **Drift in fine articulation** | #5 lip-crop perceptual + #6 lip discriminator + Stage 2 lip MLP | (partial — bs_loss covers mouth channels but no per-region crop loss) | LivePortrait specifically calls out the lip region as fragile and dedicates 3 separate terms to it. Our bs_loss treats `mouthSmileLeft` the same as `eyeSquintLeft`. |

## What's load-bearing on each side

### What LivePortrait can't drop

- **#7 ArcFace face-id.** Their own ablation (Section 4.4) shows identity collapses immediately without it. Same conclusion as our v1d failure mode (`project_v1d_slider_falsified`).
- **#1 keypoint equivariance.** Without it the motion extractor is a content cheat.
- **#6 GAN discriminators.** Removing → predictable LPIPS-trained blur. The pixel-level perceptual (#5) is not a substitute.

### What our slider can't drop

- **D direction term.** It's the entire mechanism of edit specification. Drop it and there's no slider.
- **A arc_loss.** Confirmed by the squint v0 → v1 sequence — without identity anchor, smile/squint sliders drift face age and ethnicity.
- **G step-gating.** Without it, late-timestep training corrupts texture (the same domain where Flux base does fine detail rendering); structural sliders trained without t-gating produce mushy faces.

### What looks dropable on each side

- **LivePortrait #4 deformation prior** — is a regularisation choice, not a hard constraint. Could be replaced with implicit regularisation from the GAN.
- **Our M per-channel R² mask** — useful but circumstantial. If critics improve uniformly we wouldn't need it.

## What's missing on our side that LivePortrait has

1. **Per-region perceptual / discriminator losses.** LivePortrait's lip-region GAN + lip-crop perceptual is what keeps fine mouth articulation from collapsing into a smear. Our bs_loss measures whether the channel value moved, not whether the *texture* of the mouth region looks right. **This is the most actionable port.** A "lip-region SSIM-to-base" or "face-region LPIPS-to-base" auxiliary loss could harden the mouth/eye regions of slider edits.
2. **Equivariance under input perturbation.** LivePortrait's #1 says "if I rotate the source 10°, the kps should rotate 10°." Our slider has no analogous self-consistency check. We *could* require: the slider direction at z_t should equal the slider direction at a perturbed z'_t. This is roughly what the noise-conditional-distill thread is reaching for.
3. **A geometric prior on the edit space.** LivePortrait's #2 (kp spread) and #4 (small deltas) are priors over what valid motion looks like. Our slider has no prior on what a valid edit *direction* looks like — we let the optimiser find any direction that satisfies D + B + A + S. Adding "the edit should be in the span of measured demographic PCs" or "the edit should be sparse in the NMF basis" would be the analogue.

## What's missing on LivePortrait's side that we have

1. **Explicit adversarial robustness (PGD).** LivePortrait's discriminators harden the *generator* against discriminator gradients. Our PGD inner loop hardens the *critic* against generator gradients. Different direction; both useful. PersonaLive could in principle benefit from a PGD-trained motion extractor that's robust to driving-frame perturbations.
2. **Latent-space evaluation.** All LivePortrait losses operate on decoded RGB. Ours mostly operate on Flux's noisy latent via distilled students. For their pure-deformation pipeline this is moot (no diffusion), but for PersonaLive's diffusion path, putting their identity / perceptual losses inside the latent loop would speed training significantly. **This is the most actionable port in the other direction.**
3. **Channel-aware loss weighting.** LivePortrait treats all 21 kps identically. If they exposed semantic kp groups (eye region, mouth region, brow region), they could weight per-region. Mostly a no-op for them since their training data is plentiful, but relevant if anyone tries to fine-tune LivePortrait on a small custom corpus.

## Implications for the unified iPhone-pipeline plan

This comparison sharpens two of the bridge-and-beyond decisions in `2026-05-03-iphone-pipeline-unified-plan.md`:

- **Phase 3 bridge MLP loss design.** Default was MSE on `(b_arkit, m_personalive)` pairs. Borrow from LivePortrait: also include a self-consistency term (the same b_arkit on a slightly perturbed driver frame should produce a similar m). And weight the mouth/eye channels explicitly given LivePortrait's recognition that those regions are perceptually load-bearing.
- **Phase 5 stabilizer.** LivePortrait's identity persistence comes from one strong loss (#7) at training time, not from runtime intervention. PersonaLive inherits this but only partially (the diffusion student is distilled from a stronger model that had it). Adding our `arc_distill` student as an *inference-time guidance* signal — periodically nudge the rendered frame back toward source-identity — is a clean port of our infrastructure into their pipeline.

## Implications for the slider thread

The reverse direction. Two ports worth a real experiment:

- **Region-conditioned auxiliary loss.** Add `lip_region_lpips_to_base` and `eye_region_lpips_to_base` terms (computed via VAE-decoded crops or via a region-aware distilled student). Hypothesis: this fixes a class of v3/v4 failures where the channel value is achieved but the local texture looks slightly off — the "wrong-but-sharp" failure mode that PGD partially addresses but doesn't eliminate.
- **Equivariance / self-consistency term.** `slider_direction(z_t) ≈ slider_direction(z_t + small_noise)` as a regulariser. Should reduce the per-channel R²(t) instability we keep finding (`project_blendshape_temporal_availability`).

## References

- LivePortrait paper: Guo et al., 2024, arXiv 2407.03168. Loss section is §3.3.
- Our slider operational handbook: `2026-05-03-slider-operational-handbook.md` (procedures, dataset, critic training).
- LivePortrait + X-NeMo + PersonaLive analysis: `2026-05-03-liveportrait-x-nemo-analysis.md`.
- v1d falsification: `project_v1d_slider_falsified_by_clean_student.md`.
- Step-gating empirical findings: `project_bs_lora_step_gating.md`.
- Per-channel R² heterogeneity: `feedback_blendshape_per_channel.md`.
- Editing framework principle (direction loss as core): `project_editing_framework_principle.md`.
- Topic index: `_topics/neural-deformation-control.md`.
