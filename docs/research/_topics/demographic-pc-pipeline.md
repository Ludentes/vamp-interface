## Demographic-PC pipeline — current belief

**Status:** live but paused at Stage 4.5; cached-δ injection falsified,
Stage 5 path now runs through live prompt-pair composition with Concept
Sliders distillation as the production target. Last updated 2026-04-24.

### 2026-04-29 — arc_distill step-2 ArcFace-pixel baseline: gate FAILED at full-image input

ResNet-18 (ImageNet pretrained) student trained on FFHQ 224² → InsightFace `buffalo_l` ArcFace 512-d teacher; 24,182 train + 1,514 val rows (SHA-prefix split, 'f' held out). Result: **val cosine mean = 0.377**, far below the 0.9 gate. Train cos loss kept dropping to 0.42 while val plateaued at 0.377 by epoch 5/8 — real overfit gap, but both sides too weak for the gate. Dominant suspect: input format mismatch — ArcFace was trained on aligned 112² face crops, we hand the student whole-image 224² portraits. Recommended next: rebuild compact dataset with SCRFD-bbox-aligned 112² crops (bboxes already in `output/reverse_index/reverse_index.parquet`), re-run gate. Cheap (~15 min). Defer arc_latent (step 3) until alignment question resolved — latent path inherits the same no-alignment handicap on top of less identity-aware input. Doc: [2026-04-29-arcface-pixel-baseline.md](../2026-04-29-arcface-pixel-baseline.md). Plan: [2026-04-28-arcface-pixel-baseline.md](../../superpowers/plans/2026-04-28-arcface-pixel-baseline.md).

### 2026-04-28 — FFHQ rescues Solver C on squint

Reran v1.1 on 26,040 ArcFace-detected FFHQ rows (cell key = ff_race × ff_gender × ff_age_bin; KNN top-50 ArcFace neighbors per row substitutes for "same identity"). All three pass criteria PASS: 76k pairs above p90, eff. rank **16/21** (vs Flux's 3/31), σ_θ² Spearman 1.0. PC0 of selected Δη captures only **16% of variance** (vs Flux's 59%), distributed mostly across blink + small smile residual; the Duchenne bundle is gone. Median |Δθ| selected = 1.166 ≈ 3σ. Read: Flux failure was a corpus-distribution problem, not a metric or coverage problem — FFHQ's broader photographic variance contains the squint-isolated pairs Flux's text-prompted distribution does not. Unlocks v9-squint Path B (FFHQ pairs through Flux VAE; ai-toolkit `image_reference_slider_trainer` port to flow-matching, ~30-50 lines per [2026-04-26-measurement-grounded-slider-plan.md](../2026-04-26-measurement-grounded-slider-plan.md) Path B). Path C demoted from "forced by squint" to "future thread for axes where even FFHQ is bundled." Composition route (Solver A) still tested in parallel as the cheapest possible answer. Doc: [2026-04-28-solver-c-ffhq-rescue.md](../2026-04-28-solver-c-ffhq-rescue.md).

### 2026-04-28 — Solver C feasibility on squint: Duchenne-bundled corpus (productive failure)

Solver C formalized as a finite-difference proxy for partial Fisher information `I_{θ|η}`; pair scoring `J = (Δθ)² / (σ² + Δη^T W Δη)` aligned to the Concept Sliders image-pair loss algebra. v1.1 (post adversarial review) ran on 8,115 Flux corpus rows with full Σ_η^{−1} Mahalanobis, ArcFace PC-8 included in η, σ_θ² sweep, Δη eigenvalue diversity check. **Result: PASS on count (487k pairs above p90) and σ_θ² stability (Spearman 1.0 across {1%, 5%, 25%}); FAIL on Δη diversity (eff. rank 3/31).** Top-3 selected-Δη eigenvectors decompose as PC0 = smile+blink+arc+cheek+gaze (59%, the Duchenne bundle), PC1 = blink+brow+cheek (11%), PC2 = brow (8%). Read: corpus has ample squint variance but **zero squint-isolated pairs** within demographic cells — every Δsquint pair also moves the Duchenne complement. Avoided ~1 GPU-hour training + 50 min eval that would have produced a "squint" LoRA encoding Duchenne. Three routes ahead: (1) composition `+squint − smile − cheek` via `compose_iterative` (zero training, try first); (2) corpus rebalance with eigenvector-informed prompts (mouth-neutral squint anchors, eyes-open-mouth-smiling controls) then rerun Solver C; (3) Path C latent φ surrogate (per-input gradient supervision, sidesteps pair distribution). Spec: [2026-04-28-solver-c-partial-fisher.md](../2026-04-28-solver-c-partial-fisher.md). Result + decomposition: [2026-04-28-solver-c-squint-feasibility-result.md](../2026-04-28-solver-c-squint-feasibility-result.md).

### 2026-04-28 — FFHQ reverse-index extension

All five reverse-index metric families (mivolo + fairface + insightface attrs + siglip-2 attribute probes + ARKit blendshapes + NMF atom projection from `au_library.npz`) now extracted over the 70k FFHQ corpus at 512² to match Flux dimensions. Single-pass extractor at [extract_ffhq_metrics.py](../../../src/demographic_pc/extract_ffhq_metrics.py); merger at [build_unified_reverse_index.py](../../../src/demographic_pc/build_unified_reverse_index.py). Output: `output/reverse_index/reverse_index.parquet` (sources: `flux_corpus_v3`, `ffhq`; join key: `image_sha256`). FFHQ detection rates: ArcFace/SCRFD ~60% (det_thresh=0.5; per [2026-04-27-arcface-detection-threshold.md](../2026-04-27-arcface-detection-threshold.md)); MediaPipe FaceLandmarker has different priors and recovers a different (typically wider) subset. MiVOLO predicts on every row — interpret accordingly. See [2026-04-28-ffhq-reverse-index-extension.md](../2026-04-28-ffhq-reverse-index-extension.md).

### 2026-04-27 — `arc_latent` distillation pre-flight: teacher mislabel fixed, FFHQ detection rate measured

While preparing pre-flight #1 of the [arc_latent distillation plan](../2026-04-27-arc-latent-distillation-plan.md):

- **Teacher identity corrected.** The plan and `project_vamp_measured_baseline` memory called the production face encoder "ArcFace IR101." It is not. ONNX inspection (130 nodes, input `(N,3,112,112)`, output `(1,512)`) confirms it is **ArcFace R50** as packaged in insightface's `buffalo_l` (recognition head `w600k_r50.onnx`, ResNet-50 backbone, ArcFace loss, WebFace600K training). The "IR101" recommended in [2026-04-07-face-recognition-embeddings.md](../2026-04-07-face-recognition-embeddings.md) (`minchul/cvlface_arcface_ir101_webface4m`) was never installed. Decision: ratify R50 as the actual teacher (every cached embedding and every ridge classifier is in R50 space; switching now would invalidate downstream for ~1pp IJB-C gain we don't need).
- **Two preprocessing paths in repo, only one canonical.** [classifiers.py:157](../../../src/demographic_pc/classifiers.py#L157) wraps `FaceAnalysis(name="buffalo_l")` with full SCRFD detect → 5-pt similarity align → 112×112 BGR → `(x-127.5)/127.5` → R50 → L2-normed pipeline. [fluxspace_primary_metrics.py:44-48](../../../src/demographic_pc/fluxspace_primary_metrics.py#L44-L48) takes a shortcut (`Resize(112) + Normalize([0.5]*3,[0.5]*3)` without alignment, RGB instead of BGR). Embeddings from the shortcut are **not** comparable to the canonical pipeline; the distillation corpus must use canonical only.
- **FFHQ detection rate at `det_thresh=0.5` is ~60-70%, not >99%.** See [2026-04-27-arcface-detection-threshold.md](../2026-04-27-arcface-detection-threshold.md). On 10 first images of `bitmind/ffhq` shard 0, 6/10 detected at canonical 0.5; 10/10 at 0.1. Misses are clearly visible frontal smiling faces — the detector finds them but at lower confidence, dropped by the threshold. Hypothesis: FFHQ's tight dlib alignment puts faces at face-to-frame ratios outside SCRFD's WIDER-FACE training distribution. Threshold stays at 0.5 (consistency with cached embeddings); expected effective FFHQ-distillation corpus is ~42k of 70k. Plenty for ResNet-18 distillation.
- **Reference fixture exists** at `tests/fixtures/arc_reference.npz` — 10 FFHQ images, SHA-256-keyed, 6 valid 512-d L2-normed embeddings + 4 explicitly recorded as `detected=False`. Round-trip check for any future corpus builder.

### 2026-04-27 — v5 prep: expanded glasses corpus + acetate variant + canonical dataset

For the v5 slider training run:
- **FluxSpace corpus expanded** for glasses axis: 6 missing demos rendered (latin_f, middle_f, latin_m, asian_f_elderly, young_black_m, young_european_f) at 4 seeds × 4 alphas via `expand_corpus_v3_multipair.py --axis glasses`. Index now 10 demos × 4 seeds × 4 alphas = 160 rows in `v3_1_glasses` source.
- **Anti-zebra `glasses_acetate` variant** added to `expand_corpus_v3_multipair.AXIS_PAIRS`: thick acetate / matte tortoiseshell prompts + `start_pct=0.30` (vs 0.15) to let Flux commit to base before glasses inject. Mitigates the thin-rim moiré Flux's VAE+patch tokenisation produces. 24 renders (6 demos × 4 seeds × α=0.6) at `crossdemo_v3_1/glasses_acetate/`.
- **`ai_toolkit_glasses_v2` dataset built** at `datasets/ai_toolkit_glasses_v2/` via `build_glasses_v2_dataset.py`: 63 pairs (39 main α=0/0.8 + 24 acetate α=0/0.6 reusing main α=0 anchor as no-edit baseline) with demographic-conditioned captions. v1 had 7 pairs × 1 caption.
- **`glasses_slider_v5.yaml`** at `~/w/ai-toolkit/config/`: η=4.0→2.5, lr cosine to 0 (1.25e-4 base), 1500→800 steps, save_every=100→50, LoRA alpha 1→16 (principled = rank), xattn scope kept, EMA off kept, adamw8bit kept.
- v6 will be Lion at lr ≈3e-5 (1/4 of AdamW, sign-based may push less hard into bundle directions). Lion preferred over Prodigy because we already know the right LR — Prodigy's auto-LR doesn't apply.
- **Stereotype-bundle hypothesis update from v4-step-600 SigLIP probes**: glasses Δ=+0.054 dominates over earrings (+0.022), beard (+0.012), formal_clothing (-0.006). The visual cluster shift in collages is real but doesn't show as a measurable competing axis on bipolar SigLIP probes — bundle is a visual gestalt (background, framing, posture), not per-channel attribute drift.

### 2026-04-27 — Canonical-schema LoRA eval (fusable with sample_index)

`measure_slider.py` now produces the canonical FluxSpace-corpus schema (92/92 columns matching `models/blendshape_nmf/sample_index.parquet`) plus 22 LoRA-specific columns (`slider_name`, `checkpoint`, `prompt_pool`, observed-* classifier predictions). Each row carries the 20-d NMF atom projection (atom_source="pinv_lora"), 52 ARKit blendshapes, demographic intent (parsed from prompt → ethnicity/gender/age/closest-base), and demographic observations from MiVOLO + InsightFace + FairFace. **Fusion rule**: `pd.concat([sample_index, lora_eval], join="outer")` → unified corpus for Path B (forward checkpoint mix solver) and Path C (inverse training-pair selection). Schema is now the load-bearing decision; column ordering is not. Both v4 step 600 and step 1400 evals re-scored in this schema.

### 2026-04-27 — Glasses v4 step 600 vs 1400: ship 600

See [step 1400 eval](../2026-04-27-glasses-v4-step1400-eval.md). 1400 has worse held-out coverage despite 800 more training steps (held-out @ +1.0 = 50% for 1400 vs 67% for 600; generalization gap 39 pts vs 22 pts). Identity improves slightly (mean ArcFace 0.69 → 0.71). Classic overfit: late η=4 training tightens in-distribution at the cost of held-out generalization. **Step 600 is the candidate**. Validates v5 plan (cosine LR, lower η, fewer steps).

### 2026-04-27 — Glasses v4 step 600 eval (first slider through the battery)

See [2026-04-27-glasses-v4-step600-eval.md](../2026-04-27-glasses-v4-step600-eval.md).

First slider evaluated under `measure_slider.py` (243 cells = 9 prompts × 9 strengths × 3 seeds). Glasses fails the original bidirectional pass criteria (separation @ ±1.5 = 0.06 vs ≥ 0.3) because the negative half is semantically empty — adding glasses is a concept, removing them from a baseline that doesn't have them isn't. Procedure now has a one-sided code path; glasses joined `ONE_SIDED_AXES`. Under one-sided metrics: engagement strength ≈ +1.0 (89% in-dist / 67% held-out cells gain glasses), saturation ≈ +1.5 (100% / 100%), identity collapses past +1.5 (mean ArcFace 0.42 → 0.30). Operating point ~+1.0; usable range +0.5 → +1.5. **Stereotype-bundle hypothesis falsified by these probes**: at s=+1.5 vs s=0, glasses Δ=+0.054 dominates over earrings (+0.022), hair_curly (+0.021), beard (+0.012), formal_clothing (-0.006). The visual cluster shift in the collage (bare → polo, hijab on `ho_middleeast_neutral`) is real but doesn't show up as a measurable competing axis on these particular SigLIP probes; bundle is a visual gestalt (background, framing, posture), not per-channel attribute drift. Step 1400 eval pending (currently rendering).

### 2026-04-26 — Three solvers, only one built; v4→v5 resume notes

See [2026-04-26-solvers-taxonomy-and-next-steps.md](../2026-04-26-solvers-taxonomy-and-next-steps.md).

Disambiguates "the solver" into three: **A** = FluxSpace composition (built; produced sample_index.parquet + cached attn pkls + ~7652 measured renders); **B** = forward checkpoint-mixing (idea, not built; min-norm LSQ over a slider's checkpoint ladder, inputs come from quality-measurement parquet, build only if single-checkpoint fails the battery); **C** = inverse training-pair selection (idea, not built; queries the measured corpus for off-stereotype pairs that break the cluster correlation, prerequisite to a meaningful Path B). Captures Path C four-gate kill plan (φ R²≥0.85 → ∇φ visual structure → no-LoRA gradient-ascent produces real glasses → 200-step LoRA pilot; total time-to-kill ~1h before any meaningful compute). Documents the **stereotype-bundle observation** from v4 step 600 (eyewear is a feature of a cluster shift "casual↔studio-professional", not an axis; bundle lives in T5-XXL + image priors, upstream of LoRA scope; explains why Path A is structurally weak on bundled axes — not a tuning problem). Resume-here checklist for next session: confirm v4 wraps, write `measure_slider.py`, run quality battery on best v4 checkpoint, decide v5 + v6 (v6 runs regardless of v5 outcome per user instruction), discuss solver C while v5 trains.

### 2026-04-26 — Slider quality measurement procedure

See [2026-04-26-slider-quality-measurement.md](../2026-04-26-slider-quality-measurement.md).

Per-slider eval battery: render grid (in-distribution training prompts + ≥6 held-out prompts × strengths {-2.5..+2.5} × ≥3 seeds), measurement set (intended-axis metric + ArcFace identity + SigLIP hair/accessories/clothing + MediaPipe blendshapes + lighting stats), parquet schema keyed by `(slider, checkpoint, prompt, strength, seed)`, pass criteria (monotonicity Spearman ≥ 0.9, identity ArcFace ≥ 0.4, usable range ≥ 1.0, in-dist↔held-out gap ≤ 0.15). A LoRA is "a slider" only after passing all four; eyeballing the training collage is what got us fooled across v0-v4. Multi-checkpoint solver and cross-slider comparison deliberately deferred until single-checkpoint pass exists. Implementation script (`measure_slider.py`) not built yet.

### 2026-04-26 — Measurement-grounded slider training plan

See [2026-04-26-measurement-grounded-slider-plan.md](../2026-04-26-measurement-grounded-slider-plan.md).

Path A (prompt-pair Concept Sliders, ai-toolkit on Flux Krea) struggled across v0-v4 with guidance distillation magnitude / scope / EMA tuning. We have measurement infrastructure (SigLIP margins, MediaPipe blendshapes, ArcFace, classifier scores) for every axis we care about — using a general image method on a problem with quantitative ground truth is the wrong tool. Plan documents three measurement-grounded paths in math-level detail: **D** = ridge regression on cached attention features for direction-extraction validation (1d, no LoRA); **B** = image-pair Concept Sliders fed by pair-filtered renders (2-3d, smallest move, supervision becomes `(z_neg − z_pos)/(2w)` direct latent-space difference instead of small prompt-pair velocity diff); **C** = latent-space surrogate `φ` trained to predict measured score, used as gradient supervision for LoRA (5-7d, cleanest math, one φ per axis works for all sliders). Per-axis: same machinery, different measurement column. Blendshape axes (smile, jaw, squint) are better-conditioned than glasses and should fall out once B works.

### 2026-04-26 — Slider experiments journal (append-only)

See [2026-04-26-slider-experiments-journal.md](../2026-04-26-slider-experiments-journal.md).

Append-only log of slider training runs: hypothesis → config delta → data → result → verdict → next. Source of truth for configs is the YAML; this doc captures *why* and *what happened*. Documented so far: `glasses_slider_v0_overshoot_lr2e-3` (FAIL — three compounding errors), `glasses_slider_v0` (in progress, partial-fail expected — degenerate direction), `glasses_slider_v1` (planned — xattn-only scope, no EMA). Backlog includes per-axis scope hypothesis, anchor class, image-pair Path B port.

### 2026-04-26 — Flux MMDiT attention decomposition for slider scope

See [2026-04-26-flux-attention-slider-scope.md](../2026-04-26-flux-attention-slider-scope.md).

Logical decomposition of Flux's joint MMDiT attention (Q/K/V across img and txt streams) to argue which weights actually carry a concept-slider direction vs which are bleed surface. Surgical → permissive ladder: `add_k_proj`+`add_v_proj` (38 mods, most direction-pure) → +`to_q` (57) → notebook xattn (~152) → ai-toolkit default (~494, current run, blew up at lr=2e-3). Right scope likely differs by axis (smile = semantic relabel, glasses = new visual content). Triggered by `glasses_slider_v0` audit identifying 6× module inflation as one of three compounding failure modes.

### 2026-04-24 — distil ArcFace AND MediaPipe into latent space; trainer rebuild plan

See [2026-04-24-latent-arcface-distillation-plan.md](../2026-04-24-latent-arcface-distillation-plan.md).

Doc now covers two companion oracles:

- **Latent ArcFace** (φ_id_latent) — identity-preservation regulariser, distilled
  from pixel ArcFace via (latent, arcface_embedding) pairs; pair-cosine
  correlation r ≥ 0.9 on holdout as the validation gate.
- **Latent MediaPipe** (φ_bs_latent) — *primary* axis supervisor, distilled
  from MediaPipe FaceLandmarker via (latent, 52-d blendshape) pairs;
  per-axis MSE ≤ 0.05 on holdout as validation gate. Added after v1.1
  image-pair training surfaced structural failure: identity contamination
  at all α (training cells labelled α=0.15 had 9% median id drift, which
  the LoRA reproduced as "scale 0.15 = shift toward training-mean face")
  plus FluxSpace sublinearity (target edit grows 2× while α grows 3.3×)
  that a linear LoRA can't fit.

Composed loss: `L_total = λ_bs·L_bs + λ_id·L_id + λ_diff·L_diff_anchor`.
Blendshape target is authored synthetically (anchor_bs + α·unit_delta),
making α↔effect linear by construction and ARKit-compatible by
construction — generalises to the 12-atom atlas without new rendering.

Distillation corpora already exist: classifier_scores.parquet has ArcFace
embeddings, sample_index.parquet has 52 bs_* columns, 7000+ PNGs for
fresh VAE-encoding. Pilot-train both heads in parallel (one GPU-day),
validate cheaply, integrate into trainer. Picked up when sliders ship.

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
