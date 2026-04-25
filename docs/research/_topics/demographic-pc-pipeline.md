## Demographic-PC pipeline — current belief

**Status:** live but paused at Stage 4.5; cached-δ injection falsified,
Stage 5 path now runs through live prompt-pair composition with Concept
Sliders distillation as the production target. Last updated 2026-04-24.

### 2026-04-24 — distil pixel ArcFace into our cached-feature space (next-week plan)

See [2026-04-24-latent-arcface-distillation-plan.md](../2026-04-24-latent-arcface-distillation-plan.md).

Design doc for a custom identity loss that sidesteps PuLID-style VAE+rollout
cost. ArcFace pretrained weights are reused as a *labelling oracle*: for each
face image, target = φ_pixel(image); train a small head φ_ours over our
existing cached attention features (Option A), Flux latents (B fallback), or
noised latents (C eventual production form) to reproduce pair-cosine geometry.
Validation gate: pair-cosine correlation r ≥ 0.9 on a 1000-pair holdout.
Pilot uses data we already own (classifier_scores.parquet ArcFace embeddings
× cached attn pkls). Integration: add `L_total = L_diffusion + λ_id (1 − cos)`
to slider trainer, A/B against v1.1 on eye_squint α=1.0 pass rate (currently
67% corpus ceiling). Picked up after eye_squint v1.1 lands.

### 2026-04-24 — non-standard losses on LoRAs (PuLID mechanism deep-dive)

See [2026-04-24-non-standard-losses-lora-math.md](../2026-04-24-non-standard-losses-lora-math.md).

Follow-up to the identity-preserving LoRA survey. Goes deep on two
mechanisms: (1) PuLID's actual training math — Lightning T2I 4-step
rollout with truncated backprop (DRaFT-K style), accurate ID loss
`L_id = 1 − cos(φ(x̂0_L), φ(I_id))` on ArcFace-50, contrastive
alignment loss on with-ID vs without-ID cross-attention queries
sharing prompt and noise, IP-Adapter-style parallel cross-attention
(SDXL) / Flamingo-style inserted blocks (FLUX), 1.5M images on
8×A100, 41–63 GB training VRAM with Lightning trick. (2) Why an
ArcFace term can't just be added to a kohya LoRA loop — diffusion
training never produces images, x̂0 is garbage at high t,
backprop-through-sampling needs gradient checkpointing or truncation,
ArcFace can collapse into memorisation without an alignment / prior
regulariser, VAE in the gradient graph for latent diffusion, and a
review of how DDPO / DRaFT / ReFL solve the same three problems.
Conclusion sharpens survey recommendation: stack frozen PuLID-Flux
with our sliders rather than trying to retrofit an ArcFace term into
the slider training loop.

### 2026-04-24 — identity-preserving LoRA survey (for high-α slider sweeps)

See [2026-04-24-identity-preserving-lora-survey.md](../2026-04-24-identity-preserving-lora-survey.md).

Mapped the literature on identity preservation under attribute editing:
DreamBooth/Custom Diffusion/Celeb-Basis/Concept-Sliders carry no
ArcFace term in training; ID-Booth (FG 2025) is the cleanest
LoRA-shaped recipe with an explicit ArcFace triplet loss; encoder
methods (IP-Adapter-FaceID, PhotoMaker, InstantID, PuLID, Arc2Face)
inject ArcFace embeddings via cross-attention adapters. PuLID
(NeurIPS 2024) trains its adapter with a contrastive-alignment +
Lightning-branch ID loss specifically to leave non-ID model
behaviour untouched, making it the best theoretical companion for
our slider LoRAs. Recommended next step: stack PuLID-Flux on the
v3 anchor + existing slider LoRAs and re-measure ArcFace ≥ 0.75
retention at α=1.0 (currently 67%).

### 2026-04-24 — repositioning: pipeline = training-data generator for Concept Sliders

See memory [`project_editing_framework_positioning.md`](../../../.claude/projects/-home-newub-w-vamp-interface/memory/project_editing_framework_positioning.md),
[2026-04-24-overnight-axis-screening.md](../2026-04-24-overnight-axis-screening.md),
and [2026-04-24-flux-slider-training.md](../2026-04-24-flux-slider-training.md)
(training recipe + cost envelope for the downstream distillation).

The honest competitive-audit conclusion: our
`FluxSpaceEditPair(Multi)` pipeline is interpretability-first upstream
tooling, not a production inference mechanism. For serving at scale,
Concept Sliders (ECCV 2024) trained from our characterized prompt-pair
corpora win on inference cost (1× vs our 2N+1×) while preserving our
axis dictionary.

**What this means for the pipeline:**
- Our per-axis characterization corpora (9 existing crossdemo + ~8
  additional from the 2026-04-24 overnight screening) are the
  Concept-Sliders *training data*, not the final edit mechanism.
- Composition validation lives in our pipeline (cheap) before
  committing to training N sliders (expensive).
- Measurement-side of the pipeline (MediaPipe blendshapes, ArcFace
  drift, SigLIP probes) serves as the slider eval metric.
- Production direction: once a validated axis survives both a
  single-axis metric check AND a multi-axis composition check, it
  enters the slider-training queue.

### 2026-04-23 — cached-δ replay falsified (terminates multiple threads)

See [2026-04-23-cached-delta-replay-falsified.md](../2026-04-23-cached-delta-replay-falsified.md).

Four experiments (channel-mean δ, full-(L,D) δ at K=20 sites, full-(L,D) δ + per-token
renorm to scale=300, and on-latent sanity at the exact capture seeds)
all show cached direction injection produces zero visible edit while
live `FluxSpaceEditPair` produces dramatic edits at same (base, seed).
**Editing is a stateful residual-stream cascade, not a static direction**
— each block's edit contaminates the next block's input.

**Threads this terminates:**
- AU atom-injection pipeline (C1–C7 NMF atoms as editors) — atoms are
  measurement readouts, never injectors. `au_inject_*.npz` files are
  moot as edit mechanisms.
- Anchor-cache architecture for N-variations-from-one-anchor speedup —
  same mechanism; same reason it fails.
- Option C (learn `g(attn_base) → δ`) — `attn_base` isn't a static
  input under editing; it evolves through the cascade.

**Threads still alive:**
- Live `FluxSpaceEditPair` / `FluxSpaceEditPairMulti` edit primitives.
- Channel-mean attention caches as *measurement* artefacts (ridge
  prediction of blendshape responses at CV R² 0.82–0.97).
- Prompt-pair axis dictionary (the 2026-04-24 overnight screening
  shows this path is productive).

### 2026-04-23 — atom-injection status: scoped failure, broader claim unverified

Visual test on 2026-04-23 showed **`directions_resid_causal.npz`
atom_16** (one of three related artefacts) fails to produce visible
edits at any tested scale — see
[2026-04-23-atom-inject-visual-failure.md](../2026-04-23-atom-inject-visual-failure.md).
This conclusion was reached in a prior conversation ~20 hours earlier
and lost across compaction, leading to a re-run of the same failing
path on the same file.

Not yet tested: `directions_k11.npz` (Phase-3-proper's explicit
output, norm ~0.02 — needs scales ~10³ different from what we used on
the causal file) and `directions_resid.npz` (non-causal, norm ~0.86).
The Phase-3-proper paper's constructive claim is only partially
falsified so far.

Prompt-pair FluxSpace edits (a *different* mechanism) continue to
work. Prompt-pair iteration
([2026-04-23-promptpair-iterate-plan.md](../2026-04-23-promptpair-iterate-plan.md))
is the active edit-mechanism thread. **Before any more atom-ridge
injection work: run the Option-C gating pilot on `directions_k11.npz`
at file-appropriate scales first.**

### TL;DR

- **What it is:** a staged pipeline to extract demographic-attribute
  axes (gender, age, glasses, smile, race, etc.) from Flux
  conditioning space and use them for controlled edits.
- **Where we are:** Stage 4.5 complete. Ridge direction beats
  prompt-pair contrast on manifold adherence; linearity R² 0.90 vs
  0.22; identity preserved at matched extremes where prompt-pair
  destroys it. Stage 5 was unblocked in principle.
- **Why Stage 5 is paused:** FluxSpace's own pair-averaged attention
  edits (discovered 2026-04-21) beat our ridge direction visually at
  scale 0.5–1.0. The `_topics/manifold-geometry.md` and
  `_topics/metrics-and-direction-quality.md` threads are the active
  continuation; Stage 5 (large-scale attribute extraction + downstream
  scam-hunting use) is queued behind a FluxSpace-vs-ridge combined
  approach.

### Stage map

| Stage | Purpose | Status |
|-------|---------|--------|
| 0 | Unified classifier API + prompt grid | done |
| 1 | 50-sample Flux-transfer sanity check | done, r(sus) strong |
| 2 | Full-scale generation + conditioning capture | done |
| 2b | Conditioning capture on black/glasses attribute-targeted prompts | done |
| 3 | Demographic classification on generated corpus | done |
| 4 | Direction extraction via ridge on 1785 conditionings | done |
| 4.5 | Ridge vs FluxSpace-coarse prompt-pair head-to-head | done; ridge wins |
| 5 | Large-scale axis extraction + downstream app | paused, see above |

### Load-bearing dated docs (in priority order)

1. [2026-04-20 Stage 4.5 comparison](../2026-04-20-demographic-pc-stage4_5-comparison.md)
   — the head-to-head numbers; current canonical result for this
   pipeline's Stage-4 output.
2. [2026-04-20 Stage 4.5 adversarial review](../2026-04-20-demographic-pc-stage4_5-adversarial-review.md)
   — B1/B2/B3 critiques that reshaped how we report Stage 4.5.
3. [2026-04-20 extraction plan](../2026-04-20-demographic-pc-extraction-plan.md)
   — the original pipeline plan. Superseded in parts by FluxSpace
   integration; still the canonical staging reference.
4. [2026-04-20 Stage 2-4 report](../2026-04-20-demographic-pc-stage2-4-report.md)
   — conditioning capture, classification, direction extraction
   mechanics.
5. [2026-04-20 Stage 1 report](../2026-04-20-demographic-pc-stage1-report.md)
   — sanity-check rejection of pre-Stage-1 assumptions.
6. [2026-04-20 demographic classifiers](../2026-04-20-demographic-classifiers.md)
   — classifier choices driving Stage 3.
7. [2026-04-20 Stage 1 report](../2026-04-20-demographic-pc-stage1-report.md)
   and [install log](../2026-04-20-demographic-pc-install-log.md)
   — provenance; read only if reproducing.

### Cross-thread links

- Direction-quality metrics that evaluate this pipeline's output:
  [metrics-and-direction-quality](metrics-and-direction-quality.md).
- Why FluxSpace superseded Stage 5 execution:
  [manifold-geometry](manifold-geometry.md).

### Open questions

- Can ridge (Stage 4) and FluxSpace (2026-04-21+) be combined into a
  single edit pipeline? The memory file
  `feedback_fluxspace_fallback.md` suggests yes — use FluxSpace for
  coarse on-manifold motion, ridge for the semantic projection. Not
  yet tried.
- Does Stage 5's downstream scam-hunting use case survive the
  manifold-geometry pivot? Still planned but paused; see
  `project_perception_curriculum_pivot.md` in memory.
