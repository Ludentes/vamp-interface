---
status: live
topic: neural-deformation-control
---

# Interesting ideas — open threads, gaps, things to come back to

**Date:** 2026-05-03
**Purpose:** A single place to park *interesting* ideas surfaced during the late-April / early-May neural-deformation + slider research sprint that we are not pursuing right now but should not lose. Each entry has a one-line hook, a "why interesting" paragraph, a concrete first experiment, and pointers. Re-read this doc when picking up a new thread; when you act on an entry, move it into a dated research doc and link from here.

## Cross-pollinating our slider stack into PersonaLive / LivePortrait

These are techniques we developed on the LoRA-slider side that have no analogue in the LivePortrait or PersonaLive training recipes, and that look like they would port cleanly. If any of them help, that is a small contribution back to the open-source talking-head ecosystem.

### PGD-adversarial critic for warp-based training

The bs_v4_pgd thread (see `models/mediapipe_distill/bs_v4_pgd/`) trains the BlendshapeStudent critic with a PGD inner loop so the slider trainer cannot classifier-fool the critic. LivePortrait's Stage-1 face-id loss uses a frozen ArcFace as the critic with no robustness training; nothing prevents the warp-and-decode generator from finding adversarial pixel patterns that satisfy the cosine objective without preserving identity. Running the same PGD recipe on the ArcFace-id loss inside LivePortrait Stage 1 (or the cascaded perceptual loss) is a small architectural change with a potentially large quality bump. **First test:** retrain just the face-id loss with PGD on a 1k-sample LivePortrait fine-tune, measure cross-identity reenactment ArcFace cosine vs. baseline.

### Latent-space distilled critics inside Stage-1 perceptual loss

LivePortrait's Stage-1 cascaded perceptual loss runs VGG / GAN forward on the rendered RGB output. For a warp-based pipeline this is unavoidable, but for diffusion variants like X-NeMo and PersonaLive the perceptual losses could in principle run directly on the U-Net's latent output without VAE-decoding to RGB first. Our `arc_distill` and `mediapipe_distill` heads (`models/arc_distill/latent_a2_full_native_shallow/`, `models/mediapipe_distill/bs_v4_pgd/`) are exactly this — pixel-space teachers distilled to operate on Flux VAE latents. **First test:** try one of our distilled critics as an auxiliary loss inside PersonaLive Stage-2 distillation training; measure wall-clock speedup vs. RGB-decode-then-VGG.

### Step-gating at inference for ALP sliders

Our `project_bs_lora_step_gating` finding — that bs_only LoRAs trained at t≤0.5 should be applied only over the early-mid portion of the diffusion schedule (~5–80 %) — has an obvious analogue for PersonaLive's 4-step distilled student. Rather than applying a slider equally to all 4 steps, gate the slider to specific steps based on what blendshape it controls. This requires the per-channel R²(t) emergence curves we already built (`project_blendshape_temporal_availability`). **First test:** ALP smile slider applied at PersonaLive step 1 only vs. step 4 only vs. all steps — does the qualitative output change? If so, there is a per-slider optimal step, and the same per-channel R²(t) framework that works for Flux applies to PersonaLive's much shorter schedule.

### NMF / sparse atom basis on the 1D motion latent

The 1D motion bottleneck in X-NeMo and PersonaLive is a learned vector with no per-axis interpretability. The same NMF decomposition recipe that gave us interpretable atoms over the 52-d ARKit blendshape vector (`models/blendshape_nmf/au_library.npz`) should apply: extract motion latents from a corpus of driver clips, NMF the resulting matrix, get a sparse interpretable basis. Each atom would correspond to a learned "expression direction" inside the diffusion model's motion space. **First test:** NMF k=8 over PersonaLive motion latents from 1k driver clips; visualise each atom by reconstructing a portrait with only one atom active.

### Solver C (inverse pair selection) for PersonaLive's training data

Solver C — selecting training pairs from existing data that already maximally exhibit the target axis, rather than synthesising or generating them — is the only reason our v1k squint slider works at all. Same idea applies to PersonaLive's pretraining data: rather than uniformly sampling driver-source pairs, select pairs that maximise the motion-latent's Δ along a target axis. Should give cleaner per-axis sliders for free at training time. **First test:** ablation against uniform sampling on a 10k-pair subset.

### Reference Feature Masking applied to our LoRA training

X-NeMo's RFM trick — randomly masking regions of the reference image's feature map and forcing the cross-attention path to recover them from the motion signal — is exactly the kind of identity-motion disentanglement we struggle with in slider training (squint LoRA leaks crow's-feet, smile LoRA leaks age). Could we apply RFM-style masking inside the Flux DiT during slider LoRA training? Mask a region of the reference attention cache and force the slider direction to "recover" it. **First test:** add a feature-mask augmentation to the v1l squint training loop, measure whether age-leak (cosine to known-age direction) drops while squint fidelity holds.

### Historical Keyframe Mechanism for long-form Flux portrait sequences

We have no story for long-form portrait sequences in vamp-interface; each face is generated independently. PersonaLive's historical keyframe bank — attending to a small set of past keyframes during each new chunk — is exactly the trick for keeping a sequence of portrait variations identity-consistent over time. **First test:** apply to a `vamp interface` sequence where the same job_id renders at different sus_levels; do faces look more "the same person" with cross-frame attention than independent renders?

## Bridging neural deformation to ARKit / FLAME / our LoRAs

These are the unbuilt or under-built bridges that surfaced repeatedly during the late-April research sprint and that gate the "use both paths together" production architecture.

### ARKit-52 → AdvancedLivePortrait 12-slider mapping

Nobody publishes one. The AdvancedLivePortrait `calc_fe` function has 12 hand-tuned sliders touching specific keypoint indices of LivePortrait's 21-keypoint motion tensor; ARKit-52 has 52 named channels. Roughly 20–25 of the 52 channels map cleanly (smile_L/R → smile, blink_L/R → blink, brow_innerUp/outerUp → eyebrow, gaze axes → pupil_x/y, etc.); the rest (squint, sneer, jaw-side, lip-funnel, lip-roll, dimple, cheek-puff) need either least-squares fits to multiple sliders or a separate path. **Build:** ~50-line mapping table + composition function. **First validation:** drive ALP from a real ARKit capture (iPhone TrueDepth) through the bridge, compare against driving ALP from the same person's video directly.

### MLP from b_arkit (52-d) → PersonaLive 1D motion latent

The cleanest approach for "drive PersonaLive from blendshapes" is a small MLP trained on (b_arkit, motion_latent) pairs extracted from a corpus where both are known. Phase 4b Approach 3 in `2026-05-03-personalive-experiment-plan.md` describes the protocol; nobody has run it. **First test:** train MLP on 5k driver frames where MediaPipe gives b_arkit and PersonaLive's motion_extractor gives the latent; gate at cosine ≥ 0.7 between predicted and ground-truth latent on held-out frames.

### Compose-LoRA-edit + warp test

Apply v1k squint LoRA to a Flux portrait at low strength, then run ALP smile on the result. Open from `_topics/neural-deformation-control.md`. The two control mechanisms are designed to coexist; but do they actually compose, or do they fight? Failure modes are opposite (LoRA: classifier fooling / wrong-but-sharp; warp: identity-drift / right-but-degraded), so naively they should be additive — but until tested it is hypothesis. **First test:** 5 anchors × {LoRA only, warp only, both, neither} × measured ArcFace cos + MediaPipe blendshape readout.

### Identity-drift micro-benchmark across portrait animators

We have no shared benchmark for "does this portrait animator preserve identity at moderate driver intensity". The 20-Flux-portraits × 5-slider-settings × ArcFace-cos-0.7 protocol is in the open-questions list of `_topics/neural-deformation-control.md` and has not been run. Should be a half-day. Once it exists, we can compare LivePortrait, PersonaLive, AdvancedLivePortrait sliders, our v1k LoRA, ridge directions, and FluxSpace prompt-pair on the same axis. **First run:** the 20 portraits already exist; the 5 slider settings exist; running the four pipelines and computing the matrix is just glue.

### Squint = f(blink, eyebrow, smile) empirical fit

Open question in `_topics/neural-deformation-control.md`. If squint can be expressed as a learned linear combination of three ALP sliders that already exist, we get squint via ALP for free without training a v1l squint LoRA. **First test:** sweep blink × eyebrow × smile on a single ALP source, measure MediaPipe `eyeSquint_L/R` on each output, fit linear regression. R² ≥ 0.7 means we have it; R² < 0.5 means squint genuinely needs its own channel.

## Phenomena without theory

These are empirical regularities we observed but cannot explain, and where a small theoretical model would unlock follow-on work.

### The α ≈ 0.45 cliff in FluxSpace mix_b

`project_fluxspace_injection_threshold` finds a sharp non-monotonic cliff in FluxSpace edits at mix_b ≈ 0.45 across multiple axes. The Lobashev Hessian Geometry paper (`docs/papers/hessian-geometry-2506.10632.pdf`) gives a toy model of exactly this — diverging Lyapunov exponent at a bimodal phase boundary — but is validated only on SD 1.5 / Ising / TASEP, not on Flux. **First test:** compute the Lobashev Fisher metric on Flux's attention cache directly and look for the bimodality at α ≈ 0.45. If it shows up, we have a principled explanation; if not, the cliff is something else.

### Non-monotonic smile likelihood along linear strings

`docs/papers/diffusion-string-method-2602.22122.pdf` Fig. 4 directly matches our non-monotonic smile phenomenology — log-likelihood is non-monotonic along linear-initialised strings and flattens under a Principal-Curve regime. The paper operates on state space / VAE latent and needs both score and velocity (Flux gives only velocity), so we cannot port directly, but the *reparametrisation* trick — re-spacing string points to flatten log-likelihood — is portable as a cheap scheduling heuristic on `mix_b`. **First test:** apply log-likelihood-flattening reparametrisation to the FluxSpace smile sweep; does the visible non-monotonicity disappear?

### Per-channel R²(t) is a temporal guidance map

`project_blendshape_temporal_availability` finds that different blendshape channels emerge at different points in the Flux schedule (small features earlier, large features later). This is stronger than just an empirical observation — it suggests a principled scheduler that allocates *more inference steps* where the relevant channel is emerging and *fewer steps* elsewhere. The arcnet-informed-scheduler design (`docs/research/2026-04-29-arcnet-informed-scheduler-design.md`) is the identity version; the same idea applies per-blendshape. **First test:** redistribute Flux's 28 default steps based on the eyeSquint R²(t) curve and measure whether v1l squint training produces sharper edits with the same step budget.

## Things that became less important but worth keeping in view

These ideas were live earlier in the project and have been demoted, but a future shift in the landscape could revive them.

### Diffusion Classifier as training objective

The Diffusion Classifier paper (Li et al. 2023) proposes using the diffusion model itself as a per-class likelihood; this could in principle replace our distilled-critic stack as a slider training objective. Demoted because it requires N forward passes per training sample (one per candidate class) and is wall-clock-prohibitive. Worth revisiting if Flux gets a much cheaper inference path or if a distilled-classifier variant emerges. Note: closely related to the latent-space classifier research question we ran on 2026-05-03; topic-research found this is not a beaten path for Flux specifically.

### DPS-style noisy-latent classifiers as guidance

Our noise-conditional distillation thread (Path 2: FiLM/AdaLN-conditioned `(z_t, t)` student) is exactly DPS-flavoured. The arc Path 2 (5-stage FiLM) is running with modest gains; siglip sg_c (Path 2) was trained to epoch 10 with a schedule curve nearly flat to t = 0.75. Demoted because Path 1 (cheap retrain on noisy latents) saturates at the frozen-backbone ceiling and Path 2 returns are modest. Revisit if a better noise-conditioning architecture (perhaps cross-attention conditioning rather than FiLM) shows up in literature.

### Audio ↔ blendshape bridge

The PersonaLive recipe is largely audio-agnostic; an audio-conditioned variant is a natural and likely-imminent extension (Hallo-Live and RAP both already do this for diffusion-based portrait animators). vamp-interface has no audio surface and no immediate need for this. Worth tracking because if the talking-heads ecosystem standardises on a particular (audio, motion-latent) representation, our blendshape stack becomes a useful bridge.

### VTuber software as headless data factory

Discussed and parked as Phase 3 Option D in `2026-05-03-personalive-experiment-plan.md`. VSeeFace, VTube Studio, Warudo, Animaze speak ARKit-52 natively and can render prescribed-strength sweeps headlessly — collapses the "deform a face from neutral to smiling at controlled strength" problem into an off-the-shelf rendering pipeline. Trade-off is the OOD source domain (toon shader / MetaHuman). 30-min spike before Phase 3 commits to a path.

## Gaps in the public landscape worth tracking

These are the missing artefacts in the open-source talking-head ecosystem as of 2026-05-03. Each one is something a competent project could build and contribute, and each one would unlock a research direction we currently cannot pursue.

- **No public ARKit-52 → ALP slider mapping.** Bespoke 50 LOC; nobody has shipped it. We can.
- **No code release for FG-Portrait** (CVPR 2026, FLAME-driven portrait animation). The closest published method to "drive a portrait animator from parameters", and we cannot use it. Track via the project page; check monthly.
- **No code release for IM-Animation** (arXiv 2602.07498). Compact implicit motion encoder with cleaner identity-motion split than LivePortrait. Track.
- **No code release for OmniHuman** (ByteDance, Dreamina production model). Closed-source SOTA ceiling on multi-modal human animation. Will likely never release; track for what it implies about the achievable quality envelope.
- **No code or weights for VASA-1** (Microsoft, NeurIPS 2024). 40 FPS audio-driven; a known reference ceiling we cannot run. Likely permanent.
- **No standardised identity-drift benchmark across portrait animators.** Each paper measures differently. A community benchmark (20 portraits × 5 driver intensities × ArcFace cos) would resolve a lot of conflicting claims; nobody has built one.
- **PersonaLive's CVPR 2026 acceptance is unconfirmed** against the official program. Authors claim it; secondary sources echo. Verify against the official proceedings when published.
- **No public study composing image-side LoRA edits with warp-based animation.** I.e. "apply LoRA to source, then warp to drive". The compose-LoRA-edit-with-warp test below is the first such study we know of.
- **No published method takes ARKit-52 as input directly.** FG-Portrait takes FLAME (a richer parametric representation); no portrait animator we know of takes ARKit blendshape coefficients straight to motion latent. The b_arkit → motion-latent MLP would be the first.
- **No principled per-channel scheduler for Flux** that uses blendshape R²(t) emergence curves to allocate step density. arcnet-informed-scheduler design exists; per-channel generalisation does not.

## Process improvements

Not technical research, but ideas about how we work that surfaced during the sprint.

### Two-strikes frontmatter rule has paid off

The `frontmatter-tagger` Haiku agent + the rule about adding frontmatter on the second substantial edit has produced a clean topic graph in `_topics/`. Worth keeping; consider whether to extend to `docs/blog/` posts (currently only docs/research/ is in scope).

### Problem-indexed paper README beats feature-indexed

The 2026-05-03 rewrite of `docs/papers/README.md` from "what each paper does" to "read this if you need to ..." has already paid off in this very session. Worth applying the same pattern to other reference indexes (e.g. external-tool readmes, the `_topics/` files themselves).

### Operational handbooks beat scattered dated docs for procedural knowledge

The 2026-05-03 slider operational handbook (`2026-05-03-slider-operational-handbook.md`) is the first time we synthesised the entire April thread into a single procedural document with phased gates. Worth doing again for the next thread that produces 10+ dated docs (probably the neural-deformation thread itself, once Phase 1–4 of the experiment plan complete).

## How to use this doc

When picking up a new chunk of free time and looking for something interesting to work on, scan this doc top-to-bottom and pick the entry whose "first test" is cheapest given current context. When an idea moves into active work, create a dated research doc, link it from the matching `_topics/` file, and update the entry here with a status. When an idea is genuinely killed (we tried it, it does not work, we understand why), move it to a Falsified section at the bottom rather than deleting — the falsification trail is load-bearing for future decisions. When new ideas surface during other work, add them here within the same session rather than relying on memory or notes.
