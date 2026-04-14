# vamp-interface Rebuild Plan

**Date:** 2026-04-14
**Status:** Design, verification pass complete, ready to execute (pending one pre-flight test)
**Depends on:** Face Research review (github.com/Ludentes/Face-Research), three verification agents dispatched 2026-04-14

## TL;DR

The original vamp-interface stack (two-pass SDXL img2img, fixed PNG anchor, denoising strength as the sus dial) predates three 2024-2026 primitives that collapse its engineering surface substantially: Arc2Face for identity-conditioned generation, Concept Sliders / h-space for parametric drift, and — the biggest finding from the verification pass — **Vox2Face (MDPI *Information* 17(2):200, 2026)**, which is a direct template for projecting arbitrary non-face embeddings into ArcFace's identity geometry and driving Arc2Face as a frozen generator.

The rebuild is not a research project. It is adapting Vox2Face's speech-to-face pattern to qwen-text-to-face, plus one empirical continuity test (half day of compute), plus a choice of drift mechanism (Concept Slider trained against Arc2Face directly, or h-space direction finding, or Arc2Face's expression adapter). Novelty is retained on the visualization primitive itself — no published work proposes face-per-datum as a continuous visualization encoding; Chernoff faces (1973) is the only conceptual ancestor.

## Background — why rebuild

The original vamp-interface architecture (see [diffusion-approach.md](diffusion-approach.md), [scenarios.md](scenarios.md)) is:

1. Fixed anchor face (PNG)
2. Two-pass SDXL img2img: identity pass + expression pass
3. Denoising strength as a scalar mapped from `sus_level` (0.05 → 0.55)
4. LoRA-based uncanny encoding (v8c curve)
5. Seeded per job_id, pre-generated and cached

This works well enough for P1/P2 to demonstrate the core idea, but has known limitations:

- **Denoising strength is a crude joint dial.** It mutates identity and expression together; there is no clean way to hold identity fixed while varying the drift axis.
- **No principled two-channel separation.** Identity (from text embedding) and drift (from sus_level) share a single mutation budget.
- **v8c LoRA curve overshoots** into zombie / skin-lesion territory at high strengths (see feedback_lora_uncanny_tuning.md in memory). The mechanism is fragile.
- **Gendered LoRA response** — LoRAs appear to hit female faces differently than male faces (user observation). Unfair by construction if not audited.
- **Not generalizable.** The current pipeline is hand-tuned to the fraud use case. To become a library, the drift axis needs to be learnable from data, not baked into a LoRA.

The [Face Research review](https://github.com/Ludentes/Face-Research) (13 chapters, published 2026-04-14) identified the three primitives that address these limitations:

- **Arc2Face + expression adapter** (ECCV 2024 Oral + ICCVW 2025): dedicated identity channel (ArcFace 512-d) + dedicated expression channel (FLAME blendshape coefficients) via independent cross-attention paths.
- **Concept Sliders** (ECCV 2024): LoRA weight pairs for learned semantic directions; compositional.
- **h-space direction finding** (Asyrp, ICLR 2023): semantic directions in the UNet bottleneck, no fine-tuning, apply at inference.
- **RigFace** (arXiv:2502.02465): full SD 1.5 fine-tune, sets quality ceiling, overkill for prototype.

## The rebuild plan

### Step 1 — Replace the anchor paradigm

**Old:** fixed PNG image, img2img drift.
**New:** fixed identity embedding as the anchor. All generated faces are conditioned on identity vectors near the anchor in the conditioning space Arc2Face uses.

Continuity in the conditioning space → continuity in face space, *assuming* Arc2Face is locally Lipschitz in that space. See Step 5 for the pre-flight test of this assumption.

### Step 2 — Projection from qwen embedding to Arc2Face conditioning space

**This is the hard step of the rebuild.** Goal: `qwen_1024 → conditioning_space` such that nearby qwen embeddings produce nearby faces and the mapping is learnable from a small amount of data.

**Template to follow: Vox2Face (2026).** Vox2Face distills a pretrained *speaker encoder*'s embeddings into ArcFace's hyperspherical identity space via metric alignment, then uses Arc2Face as a frozen generator, with a diffusion self-consistency loss updating only the projection + LoRA adapters. Our use case differs only in the source encoder (qwen text embedding instead of speech). The geometry (hypersphere), the loss structure (metric distillation + self-consistency), and the frozen-Arc2Face backend all transfer one-for-one.

**Projection target — the only viable option.** Previous drafts of this plan listed "project to Arc2Face's 5-token CLIP space" as an alternative. **That option is dead** based on reading the Arc2Face paper in depth:

- Arc2Face uses the prompt template `"photo of a ⟨id⟩ person"` tokenized into 5 tokens `{e1, e2, e3, ŵ, e5}`.
- Only **one** of those tokens carries identity (`ŵ`). The other four are frozen pseudo-prompt context.
- `ŵ ∈ ℝ^768` is produced by **zero-padding the ArcFace 512-d vector**. No MLP, no learned projection — direct substitution. Arc2Face ablates against a 4-layer MLP and explicitly picks the fine-tuned-CLIP + zero-padded-substitution design.
- The UNet's actual conditioning is the post-CLIP-transformer output over these 5 tokens, but identity lives in a single 768-d slot derived from a zero-padded 512-d ArcFace vector.

**There is no CLIP token space to project into.** There is only the ArcFace 512-d input slot.

**So the projection target is unambiguous: `qwen_1024 → ArcFace_512`.** Project into the ArcFace hyperspherical identity space, zero-pad to 768, feed through Arc2Face's frozen (fine-tuned) CLIP text encoder. This matches Vox2Face exactly.

**The pairing problem — a real issue Vox2Face does not solve for us.** Vox2Face has (speech, ground-truth face) pairs from HQ-VoxCeleb: a speaker *has* a face. We have (job posting, ???) — **no ground-truth face per job posting exists**. Stage I of the Vox2Face recipe (supervised alignment of source embedding to ArcFace features of the target face) needs a target we do not have. Three sub-options:

- **(a) Synthetic anchor assignment.** Assign each cluster in qwen embedding space to a randomly sampled ArcFace vector from a reference pool. Train the MLP to respect cluster structure via a cluster-preservation loss. Faces are arbitrary but consistent. Loses the "face as encoding" semantics but preserves continuity at the cluster level. Closest to current vamp-interface's fixed-anchor pattern, just generalized from one anchor to many.
- **(b) Contrastive-only training.** Skip Stage I entirely. Train the MLP end-to-end with only the Stage II SDS self-consistency loss plus a pairwise distance-preservation regularizer: `|d_ArcFace(f(e_A), f(e_B)) − C · d_qwen(e_A, e_B)| → 0`. No ground-truth faces needed. The MLP learns *some* projection that preserves relative structure. Risky — SDS without a stable target is prone to collapse.
- **(c) Two-channel decoupling — RECOMMENDED.** Treat identity and drift as separable problems. Identity channel uses option (b) — contrastive-only MLP `qwen → ArcFace` with distance-preservation. Drift/sus axis is learned separately via Step 4A (h-space direction finding on Arc2Face activations from paired low-sus and high-sus generations). This matches the original vamp-interface thesis of clean two-channel encoding and sidesteps the pairing problem entirely because the drift direction is learned from internal generations, not from external (job, face) pairs.

### Step 3 — Frozen Arc2Face as generator

Same as Vox2Face. No retraining of Arc2Face. Public code at `github.com/foivospar/Arc2Face`, public weights at `huggingface.co/FoivosPar/Arc2Face`, runs on 6-8 GB consumer GPU, 1-2 s per 512×512 image.

### Step 4 — Drift axis (sus_level encoding)

Three alternatives, ranked by experimental cost:

**REVISED 2026-04-14 after Tier 1 deep reads of Boundary Diffusion, RigFace, NoiseCLR.** The earlier "training-free, no labels, drop-in" framing of 4A was based on a subagent paraphrase of Boundary Diffusion that turned out to be wrong on multiple load-bearing points. Reading the actual paper (arXiv:2302.08357, NeurIPS 2023, not CVPR 2023): the method is (1) a **linear SVM hyperplane normal**, not class-mean difference — mean-diff is just a cruder heuristic with no paper endorsement; (2) **symmetric single-step shift** at one searched mixing timestep `t_m`, not Asyrp-style asymmetric per-step `P_t` modification — these are architecturally different; (3) validated **only on unconditional DDPM/iDDPM** (CelebA, LSUN, AFHQ) — never on Stable Diffusion, never with CFG, never on identity-locked fine-tunes. Applying Boundary Diffusion's recipe to CFG'd Arc2Face with an ArcFace ID lock is a research bet, not a drop-in.

The revised ranking — Asyrp's learned `f_θ` (the thing Boundary Diffusion is compared *against*, not a replacement for) is now the primary, because it is the only method in this family actually designed for per-timestep SD-family bottleneck edits with text conditioning.

- **4A — Asyrp learned f_θ on Arc2Face (PRIMARY, ~20 min training on 3× consumer GPU).** Train Asyrp's 1×1 conv with timestep conditioning using a CLIP directional loss (`y_source = "a photo of a person"`, `y_ref = "a photo of a zombie"` — "zombie" is literally in Asyrp's validated showcase on CelebA-HQ). Cost per Asyrp Table: 1000 samples, S=40 timesteps, 1 epoch, ~20 minutes on 3× RTX 3090. Apply asymmetrically at inference (Theorem 1: modify only `P_t`, leave `D_t`; `[T, t_edit]` where `LPIPS(x, P_t_edit) = 0.33`). `α` linearly scales drift magnitude. Switch Arc2Face from DPM-Solver to DDIM deterministic (η=0). No CFG changes. No Arc2Face-specific replication published — we'd be early-mover on a method with the right architectural preconditions (SD 1.5 UNet, DDIM, CFG compatible via Asyrp's formulation).
- **4B — Paired mean-difference heuristic in h-space (cheap spike, no published pedigree).** The crude version of 4A: generate N=500 low-sus / N=500 high-sus Arc2Face samples, DDIM-invert to a handful of timesteps, compute `Δh = mean(h_high) − mean(h_low)`, apply asymmetrically per Asyrp's formulation. **This is not Boundary Diffusion** — Boundary Diffusion uses SVM hyperplane normals and symmetric single-step shifts, and it has never been run on a CFG'd SD fine-tune. Mean-difference is just a zero-training heuristic worth a 2-hour spike before committing to 4A. If it gives visible controllable drift on Arc2Face, use it and skip the `f_θ` training. If not, fall through to 4A.
- **4C — Boundary Diffusion faithful port (research bet, deprioritized).** Fit a linear SVM on h_tm activations from labeled (low-sus, high-sus) Arc2Face generations at a searched `t_m`, use the hyperplane normal as a one-shot shift at `t_m`. Needs DDIM inversion + `t_m` search + 500-2000 labeled samples. Major unknowns: interaction with CFG, interaction with ArcFace ID lock. Only consider this if both 4A and 4B fail and we want to burn a week on a replication.
- **4D — Concept Slider trained against Arc2Face directly.** Point `pretrained_model.name_or_path` at Arc2Face checkpoint, supply ArcFace ID embeddings as conditioning (not text), train a "suspicious" direction with paired anchor/drifted images. Critical: **do not port a slider from vanilla SD 1.5** — V4 verification found LoRAs do not transfer reliably from base to fine-tuned checkpoints (sliders#95, LoRAtorio).
- **4E — Arc2Face expression adapter with FLAME coefficients.** Zero training, public code. Useful only as a smoke test of whether FLAME expression dimensions proxy our drift axis. Probably not — uncanny is not a FLAME dimension.

**Order:** 4B (2-hour spike) → 4A (20-min training if 4B is weak) → 4D (if both h-space paths fail) → 4E (sanity check regardless) → 4C (only as a last resort research bet).

**NoiseCLR ruled out.** Evaluated in T1.3 as a possible unsupervised alternative. Wrong slot (learns pseudo-tokens in CLIP text-conditioning space, which Arc2Face has semantically repurposed for ArcFace identity), wrong geometry (operates at ε-output via CFG, not h-space), vanilla SD only, and unsupervised discovery gives no guarantee "uncanny" is among the K directions it finds. Do not spend cycles on it.

### Step 5 — Continuity validation (MANDATORY pre-flight)

**This is the most important experiment of the rebuild.** Everything downstream assumes Arc2Face is locally Lipschitz in the conditioning space. The Arc2Face paper shows one qualitative interpolation demo (Section 4.4 "Averaging ID Features") with no smoothness metrics; Arc2Morph only studies the midpoint α=0.5; nobody has run the continuous sweep.

**Protocol:**

1. Sample 20-50 pairs of real ArcFace embeddings from a public face dataset (e.g., FFHQ aligned, or the Vox2Face test set).
2. For each pair, sweep α ∈ [0, 1] in 20 steps.
3. Generate ~20 frames per sweep via frozen Arc2Face.
4. Measure pairwise LPIPS (or DreamSim) between consecutive frames.
5. Distribution analysis:
   - **Smooth, short-tailed** distribution → ArcFace space is usable as-is. Proceed with Option 2a.
   - **Bimodal or heavy-tailed** → some pairs have discontinuities. Switch to Option 2b (CLIP token space) and re-run the test.
   - **Both spaces heavy-tailed** → need a continuity regularizer in Step 2 training.

Budget: half a day of compute, zero new training.

### Step 6 — Pipeline and caching

Same pattern as current vamp-interface: for each job, (a) compute qwen embedding, (b) project to conditioning space, (c) generate via frozen Arc2Face, (d) apply drift direction with strength = f(sus_level), (e) cache 256×256 PNG keyed by job_id. Fully offline, static serving.

## Verification findings (pre-rebuild)

Three parallel agents verified specific questions the Face Research review did not answer:

| Item | Result | Confidence | Impact |
|---|---|---|---|
| **V1** Arc2Face operational status (repo active, weights public, VRAM, forks) | Green. 6-8 GB VRAM, clone-and-run today. 791 stars, active October 2025 commits. | High | No blocker |
| **V2** Arc2Face expression adapter code release | Green. Same repo, October 2025 commit. Not paper-only. | High | Option 4C unblocked |
| **V3** Cross-domain projection prior art | **Vox2Face (MDPI 2026) is a direct template.** Speech encoder → ArcFace hypersphere via metric distillation, Arc2Face frozen, self-consistency loss. | High | **Step 2 reduces from research to adaptation** |
| **V4** Concept Sliders on fine-tuned Arc2Face | Unknown, evidence leans unreliable. sliders#95 reports "no effect" on custom checkpoints. LoRAtorio confirms cross-base LoRA degradation. | Medium | Train slider *against* Arc2Face directly, not port from vanilla SD |
| **V5** Arc2Face continuity | Not characterized in literature. Arc2Morph shows weak empirical case for CLIP space over ArcFace space at midpoint only. No Lipschitz analysis exists. | Medium | **Pre-flight continuity test is mandatory (Step 5)** |
| **V6** h-space on fine-tuned diffusion | Unknown. No published Arc2Face-specific work. HAAD (arXiv:2507.17554) implicitly confirms h-space survives few-shot fine-tuning. | Medium | Viable fallback (Step 4A), half-day probe |
| **V7** Face-as-visualization prior art | **Empty.** No 2023-2026 work proposes face-per-datum as a continuous visualization encoding. Chernoff faces (1973) is the only ancestor. | High | **Novelty claim defensible** |

## Tier 1 deep-read findings (2026-04-14 post-compaction)

The deeper-research queue at [2026-04-14-deeper-research-queue.md](2026-04-14-deeper-research-queue.md) identified three Tier 1 papers whose claims were load-bearing for this plan but had never been read against primary sources. All three were dispatched as parallel deep-read agents; all three produced material corrections.

| Item | Finding | Impact |
|---|---|---|
| **T1.1 Boundary Diffusion** (arXiv:2302.08357, NeurIPS 2023) | Method is linear SVM hyperplane normal, not class-mean difference. Symmetric single-step shift at searched `t_m`, not Asyrp-style asymmetric per-step P_t modification. Validated only on unconditional DDPM/iDDPM — never SD, never CFG, never identity fine-tunes. Still requires attribute-labeled pairs for the SVM fit. | Step 4A reframed: Asyrp learned `f_θ` promoted to primary; mean-difference demoted to a 2-hour spike heuristic; Boundary Diffusion faithful port deprioritized to 4C research bet. |
| **T1.2 RigFace** (arXiv:2502.02465) | Identity Encoder is a full second SD 1.5 UNet with per-layer spatial features fused via FaceFusion concat-attention. **No embedding input port — requires a pixel reference image.** Per-layer feature conditioning is worst-case for Lipschitz behavior. No code, no weights released. | **Removed from fallback list.** If Arc2Face fails Step 5 continuity, candidates are PhotoMaker, InstantID IdentityNet in isolation, or StyleGAN3 w+ — not RigFace. |
| **T1.3 NoiseCLR** (arXiv:2312.05390, CVPR 2024) | Learns pseudo-tokens in CLIP text-conditioning space (the slot Arc2Face repurposes for ArcFace). Operates at ε-output via CFG, not h-space. Vanilla SD 1.5 only — zero fine-tune validation. Unsupervised discovery — no guarantee the K directions include "uncanny". | Ruled out entirely. Wrong slot, wrong geometry, wrong backbone regime, wrong supervision mode for a targeted axis. |

The pattern from these three confirms the `feedback_shallow_research_risk.md` lesson: summary-level descriptions systematically compress away the architectural details that matter for implementation. Every one of the three subagent paraphrases we previously trusted turned out to be wrong in a way that would have cost days of implementation work.

## Reading notes

### Arc2Morph (arXiv:2602.16569v1, Feb 2026)

Di Domenico, Franco, Ferrara, Maltoni, University of Bologna. Published in the morphing-attack-detection / biometric security literature, not visualization.

**Architecture (inference-only, zero training):**
```
I_A ─► ArcFace ─► e_A ─┐
                       ├─► Arc2Face CLIP text encoder ─► p_A ─┐
I_B ─► ArcFace ─► e_B ─┘                                      │
                                                               ├─► slerp(α=0.5) ─► p_M
EMOCAv2 ─► 3D normal map ─► ControlNet ──────────────────────────────────────────┤
                                                                                  ▼
                                                                     Arc2Face UNet
                                                                             ▼
                                                                       BEN2 matting
                                                                             ▼
                                                                     final morphed face
```

**Pivotal detail** (not in the Arc2Face README, only in Arc2Morph's description): Arc2Face's text encoder is a finetuned CLIP-ViT-L/14 that maps ArcFace 512-d vectors into a **5-token CLIP text embedding** via the template `"photo of a ⟨id⟩ person"` with the ArcFace vector padded into the `⟨id⟩` slot. The UNet never sees raw ArcFace — it sees these 5 tokens. This is the *actual* conditioning space of Arc2Face and the space where continuity matters.

**Their ablation (Table V):**

| Dataset | Interpolation location | Method | MAP_Avg |
|---|---|---|---|
| FEI | Identity (ArcFace) | lerp | 0.9778 |
| FEI | Identity (ArcFace) | slerp | 0.9679 |
| FEI | CLIP tokens | lerp | 0.9747 |
| **FEI** | **CLIP tokens** | **slerp** | **0.9835** |
| MONOT | Identity | lerp | 0.8539 |
| **MONOT** | **CLIP tokens** | **slerp** | **0.8858** |

Slerp in CLIP token space wins both datasets by +0.6 to +3 percentage points of Morphing Attack Potential. Real but modest.

**What they do NOT study:** continuity along the α sweep, any α ≠ 0.5, LPIPS/FID/smoothness metrics, why CLIP beats ArcFace (only hand-waves about "higher dimensionality and richer semantic structure"). Code not released at time of v1.

**Reusable primitives:** Arc2Face as generator, slerp over lerp as default, ControlNet + normal map for pose control, BEN2 for background cleanup.

**Not reusable:** MAP metric (biometric security, not visualization), α=0.5 midpoint-only focus, the Vox2Face-grade projection training (Arc2Morph assumes you start with two real ArcFace embeddings; we start with qwen embeddings and have to learn the projection).

### Arc2Face paper (arXiv:2403.11641, ECCV 2024 Oral) — Section 4.4 and text encoder details

**What the interpolation experiment actually shows.** Section 4.4 ("Averaging ID Features") is purely qualitative: "we provide transitions between pairs of subjects by linearly blending their ArcFace vectors" and "Arc2Face generates plausible faces along the trajectory connecting their ArcFace vectors." Figure 10 shows a small qualitative grid; the number of pairs is not stated. **No quantitative trajectory analysis, no smoothness metric, no identity-distance plot, no failure-mode discussion.** For our continuity hypothesis this is thin — they assert plausible interpolation but never measure it.

**The text encoder architecture — the load-bearing detail.** Arc2Face's conditioning is a fine-tuned CLIP-ViT-L/14 text encoder processing the prompt template `"photo of a ⟨id⟩ person"` tokenized into 5 tokens. The `⟨id⟩` slot receives the ArcFace 512-d vector **zero-padded to 768-d**:

> "ŵ ∈ ℝ^768 corresponds to w ∈ ℝ^512 after zero-padding to match the dimension of eᵢ ∈ ℝ^768"

**Zero-padding, not a learned projection.** They ablate against a 4-layer MLP (Fig. 9) and find the fine-tuned-CLIP + direct-substitution design wins on identity similarity. Only one of the 5 tokens carries identity; the others are frozen pseudo-prompt context. A PCA analysis (Fig. 8) finds ArcFace space needs 300-400 components to preserve facial fidelity — high intrinsic dimensionality.

**Text channel is dead.** After fine-tuning, Arc2Face "exclusively adheres to ID-embeddings, disregarding its initial language guidance." Any downstream controllability must come from ControlNet, LoRAs, or h-space interventions, not from prompts.

**Training recipe.** Stage 1: ~21M images from WebFace42M (1M identities), super-resolved to 448×448 with GFPGAN v1.4, 5 epochs. Stage 2: FFHQ + CelebA-HQ at 512×512, 15 epochs. 8× A100, batch size 4/GPU, AdamW, LR 1e-6. ArcFace extractor is frozen IR-100 (WebFace42M-trained), unit-normalized.

**Inference.** DPM-Solver, 25 steps, CFG=3.0. Per-image time not reported in the paper; community reports ~1-2 s on a single consumer GPU.

**Non-memorization check.** Generated-to-training ArcFace similarity = 0.37 vs input-similarity = 0.74 — evidence of generalization, not retrieval.

### Vox2Face (MDPI Information 17(2):200, 2026)

**Note: the agent could only reach the MDPI abstract page, not the full PDF.** Exact hyperparameters, loss weights, LoRA rank, ablation details are not recovered. What follows is what's confirmed by the abstract and search-indexed excerpts. The full PDF will need a second pass when we need implementation details.

**Bibliographic:** "Vox2Face: Speech-Driven Face Generation via Identity-Space Alignment and Diffusion Self-Consistency", MDPI *Information*, Vol. 17 Issue 2 Article 200. Received 2025-12-25, published 2026-02-14. Correct DOI pattern: `10.3390/info17020200`. Correct URL: `https://www.mdpi.com/2078-2489/17/2/200` (the earlier draft had an ISSN typo). Authors and affiliations not recovered. License likely CC-BY 4.0 per MDPI default but not confirmed.

**Architecture (confirmed):**

- **Speech encoder**: pretrained AST (Audio Spectrogram Transformer, Gong et al., Interspeech 2021). Variant unconfirmed. Frozen.
- **Mapping network**: MLP distilling AST features into ArcFace 512-d hyperspherical space. Exact depth/widths unrecovered.
- **Diffusion backbone**: Arc2Face, frozen UNet weights, with LoRA adapters on attention projections (rank unrecovered).
- **Target space**: ArcFace 512-d, *not* CLIP token space. **Confirms our architectural choice.**

**Trainable**: MLP + LoRA. **Frozen**: AST, Arc2Face UNet base, VAE, CLIP text encoder, ArcFace network (used only as supervision signal).

**Losses (confirmed):**

1. **Stage I — metric distillation / identity alignment**: cosine similarity / contrastive loss on the ArcFace hypersphere, supervised by ArcFace features of the target face image. Exact form (plain cosine regression, InfoNCE, or ArcFace-style angular margin) not disambiguated from the abstract.
2. **Stage II — diffusion self-consistency**: SDS-style (Score Distillation Sampling, DreamFusion lineage). Sample t, add noise to VAE-encoded target, denoise with frozen Arc2Face conditioned on predicted identity vector ĉ = MLP(speech), backprop `ε_pred − ε_true` through the conditioning path only. Gradients flow into MLP weights and LoRA adapters. UNet base weights receive no gradient. **Discriminator-free. No perceptual loss. SDS + cosine alignment is the entire loss stack.**

**Training:**

- **Dataset**: HQ-VoxCeleb (curated high-quality subset of VoxCeleb with paired speaker audio and face images).
- **Two-stage regime**: Stage I pretrains the MLP with the alignment loss alone, Stage II unfreezes the LoRAs and switches to SDS self-consistency. **This two-stage structure is load-bearing** — pure SDS from a random MLP init would likely collapse.
- **Hardware, epochs, batch size, LR**: unrecovered.

**Results (from abstract):**

| Metric | Baseline | Vox2Face |
|---|---|---|
| ArcFace cosine similarity (↑) | 0.295 | 0.322 |
| R@10 speech→face retrieval (↑) | 29.8% | 32.1% |
| VGGFace Score (↑) | 18.82 | 23.21 |

Modest but non-trivial improvements over an unnamed strong baseline. **No FID, no LPIPS, no continuity metrics, no identity-geometry analysis.**

**Code/weights**: not visible in search results. **Assume unreleased** until the PDF confirms otherwise. MDPI has a Data Availability Statement section that would settle this.

**The pairing problem they sidestep and we cannot.** Vox2Face has ground-truth (speech, face) pairs from VoxCeleb. We have (job posting, ???). The Stage I alignment loss requires a target face we do not have. This is the gap that forces Step 2's option (c) (two-channel decoupling, contrastive-only MLP, drift via h-space).

### Asyrp (arXiv:2210.10960, ICLR 2023 Oral) — h-space direction finding

**What h-space is.** The UNet bottleneck at layer 8 — "the bridge of the U-Net, not influenced by any skip connection, has the smallest spatial dimension with compressed information." For SD 1.5 this is approximately 8×8×1280 ≈ 80k dimensions. Arc2Face inherits the SD 1.5 UNet structure unchanged, so the layer choice and bottleneck shape port directly.

**Direction-finding procedure.** Neither paired-image supervised nor PCA. Asyrp trains **a small 1×1 conv network f_θ with timestep conditioning** using a **CLIP directional loss** (Gal et al., StyleGAN-NADA):

> L^(t) = λ_CLIP · L_direction(P_t^edit, y_ref; P_t^source, y_source) + λ_recon · |P_t^edit − P_t^source|

You give it text prompts `y_source` and `y_ref` and the method finds a Δh per sample and per timestep. Training cost: **1000 samples, S=40 timesteps, 1 epoch, ~20 minutes on 3× RTX 3090**. Extremely cheap.

**The asymmetric reverse trick.** Directly adding Δh to both the predicted-x0 term `P_t` and the noise term `D_t` causes "destructive interference" (Theorem 1 in the paper) — the edit gets cancelled. The fix: modify only `P_t`, leave `D_t` unchanged:

> x_{t−1} = √α_{t−1} · P_t(ε^θ(x_t | Δh_t)) + D_t(ε^θ(x_t))

**Editing interval.** Apply Δh only over `[T, t_edit]` — early (noisy) timesteps only. `t_edit` is chosen by an LPIPS criterion: pick the latest timestep such that `LPIPS(x, P_t_edit) = 0.33`. After `t_edit`, the unmodified reverse process runs, preserving fine detail.

**Claimed properties.** Homogeneity ("same Δh leads to same effect on different samples"), linearity ("linearly scaling Δh controls magnitude"), robustness ("preserves quality without degradation"), timestep consistency ("Δh_t is *roughly* consistent across timesteps" — note the hedge).

**Validated edits on face models.** Smiling, sad, angry, tanned, disgusted, makeup, curly hair, bald, young, **zombie**, identity swap (Nicolas Cage, Zuckerberg), style transfer (Pixar, Frida, Modigliani). **"Zombie" is in the showcase** — Asyrp already demonstrates a direct drift into uncanny territory purely via CLIP text prompt. **Glasses are absent** — the 8×8 bottleneck is too coarse for small localized features, which bleed into global style shifts. For us this is a positive — we want global drift, not localized attribute manipulation.

**Applicability to Arc2Face:** straightforward. The UNet bottleneck is structurally identical. Asyrp requires DDIM deterministic sampling (η=0) for inversion, so we need to switch Arc2Face from DPM-Solver to DDIM during both training and inference. No CFG changes required. No Arc2Face-specific replication has been published — we would be early-mover.

**Boundary Diffusion as an "easy drop-in" — corrected.** Earlier drafts of this document claimed Boundary Diffusion (Zhu et al., NeurIPS 2023 — not CVPR 2023 as previously written; arXiv:2302.08357) showed "class-mean difference in h_t is a drop-in for Asyrp's f_θ". **That is wrong on three points.** Reading the paper: (1) the method is a *linear SVM hyperplane normal*, not class-mean difference; (2) the edit is a *symmetric one-shot shift* at a single searched mixing timestep `t_m`, not an asymmetric per-step `P_t` modification; (3) the paper validates only on unconditional DDPM/iDDPM (CelebA-HQ, LSUN, AFHQ) and never touches Stable Diffusion, CFG, or identity-fine-tuned backbones. The paper even notes: "the ability for unseen domain image editing is relatively limited compared to other learning-based methods." A faithful port to Arc2Face is Option 4C in the drift ranking — a research bet, not a drop-in. The honest zero-training heuristic for us is **paired mean-difference in h-space applied via Asyrp's asymmetric formulation** (Option 4B): it has no Boundary-Diffusion pedigree, it's just the cheapest thing worth trying before paying Asyrp's 20-minute training cost.

**Known community observation**: h-space directions trained on a base checkpoint transfer to fine-tunes but lose sharpness; retraining on the target checkpoint is cheap (20 min) and recommended. For any h-space direction this maps to: extract h_t from *Arc2Face* generations specifically, not from vanilla SD 1.5.

## License constraint — Arc2Face is CC-BY-NC

Arc2Face is released under **Creative Commons Attribution-NonCommercial**. This has real implications for the rebuild:

- **Any derivative work built on Arc2Face inherits the non-commercial restriction.** The facemap library, Scam Guessr, any hosted tool — all must be non-commercial in distribution.
- **This is compatible with** the Face Research review's CC-BY-4.0 license, an academic publication, an open-source library released under CC-BY-NC (not MIT/Apache), a free Scam Guessr game with no revenue, donations, or sponsorship that doesn't trigger commercial-use clauses.
- **This is incompatible with** a paid Scam Guessr app, ads on a Scam Guessr web property, a commercial API of the facemap library, enterprise licensing.

**Decision:** accept the CC-BY-NC constraint for the initial build. The immediate deliverables are a preregistered user study, an open-source library, and a free game — all non-commercial. If any of those take off and commercial licensing becomes valuable, options at that point:

1. Contact the Arc2Face authors at MPI/Imperial for commercial licensing terms.
2. Swap the backbone to a permissive-license alternative: SDXL, SD 3, Flux. Loses Arc2Face's identity conditioning — would need to retrain something equivalent. Major architectural cost.
3. Train an identity-conditioned backbone from scratch under a permissive license. Impractical for a solo developer with Claude Code. A paper-level effort on its own.

**Flag in the README when we ship:** "This library is non-commercial due to Arc2Face's CC-BY-NC license. Commercial use requires separate licensing from the Arc2Face authors."

## Go/no-go recommendation

**Proceed with the rebuild.** The verification pass turned up one major gift (Vox2Face), one warning (continuity not characterized, must test), and one architectural insight (the CLIP token space as the actual Arc2Face conditioning space). Nothing is a blocker. The novelty claim on the visualization primitive is defensible. The engineering surface is smaller than the initial plan suggested — Vox2Face collapses the "hard Step 2" from weeks of custom work into adapting a published pipeline.

**Critical path:**

1. Read Vox2Face in depth (2 hours)
2. Run the Step 5 continuity test (half day of compute, no new training)
3. Run the Step 4A h-space probe (half day)
4. Implement Step 2 following Vox2Face (1-2 weeks depending on data curation)
5. Wire Step 6 pipeline (1-2 days)
6. Run the original study plan (H1a/b/c via in-game metrics, see below)

**If any critical path step fails, the fallback order is:**

- Step 5 fails in ArcFace space → switch to CLIP token space
- Step 5 fails in both → add continuity regularizer to Step 2 training
- Step 4B mean-diff spike weak → try Step 4A (Asyrp learned f_θ, 20 min)
- 4A fails → 4D (Concept Slider trained directly against Arc2Face)
- 4D fails → 4E (FLAME expression adapter smoke test)
- 4E inadequate → 4C (Boundary Diffusion faithful SVM port, research-grade effort)
- All drift options fail → keep the v8c LoRA from the current pipeline as a stopgap and ship the visualization without parametric drift control

**Backbone fallback if Step 5 continuity test kills Arc2Face.** RigFace is ruled out as a fallback (Tier 1 T1.2 deep read): its Identity Encoder is a full second SD 1.5 UNet injecting per-layer spatial feature maps via FaceFusion concat-attention — no embedding input port, per-layer features are worst-case for local Lipschitz behavior, and the paper releases no code/weights. Candidate fallbacks that accept a continuous identity vector are PhotoMaker, InstantID's IdentityNet used in isolation, or StyleGAN3 FFHQ w+ space. Evaluate only if Step 5 forces the swap.

## Open questions

1. **Does Vox2Face release code?** The MDPI paper was published in *Information*, which typically requires code release under the open-science mandate, but not always. Need to check the paper itself.
2. **How did Vox2Face handle the "speech is not face" alignment problem?** Their data must contain speech-identity pairs (the speaker's face is known); we have job-posting-identity pairs only if we generate them. This may force us to a different loss structure.
3. **What qwen embedding statistics look like.** Before training the projection, we should characterize the qwen embedding distribution over the telejobs corpus: PCA spectrum, intrinsic dimensionality, clustering structure. This informs the projection architecture.
4. **Continuity regularizer design, if needed.** If Step 5 fails, the regularizer is probably a pairwise distance-preservation loss in the projection: `|d_face(f(e_A), f(e_B)) − C · d_embed(e_A, e_B)| → 0`. This is a known technique from metric learning; we'd just apply it here.

## Related documents

- **Face Research review** (parent review, published repo): [github.com/Ludentes/Face-Research](https://github.com/Ludentes/Face-Research)
- **Original vamp architecture**: [diffusion-approach.md](diffusion-approach.md), [scenarios.md](scenarios.md)
- **v3 anchor spec**: see recent git log entries for `spec/anchors-v3` (current implementation target before this rebuild plan supersedes it)
- **Research notes referenced**: [2026-04-12-embedding-to-face-latent-arithmetic-2026.md](../research/2026-04-12-embedding-to-face-latent-arithmetic-2026.md), [2026-04-12-rigface-technical-deep-dive.md](../research/2026-04-12-rigface-technical-deep-dive.md), [2026-04-09-embedding-space-and-face-conditioning.md](../research/2026-04-09-embedding-space-and-face-conditioning.md)
- **Game-study design** (the user-study protocol that will validate H1a/b/c once the rebuild ships): needs its own document

## Changelog

- **2026-04-14 (initial)** — Written after Face Research review published, three V1-V7 verification agents completed, Arc2Morph reviewed in depth.
- **2026-04-14 (Tier 1 deep reads)** — Three parallel deep-reads on Boundary Diffusion, RigFace, and NoiseCLR completed. Corrections applied: Step 4A reranked (Asyrp learned f_θ is now primary; mean-diff demoted to 2-hour spike; Boundary Diffusion faithful port is 4C research bet); RigFace removed from fallback list (per-layer features, no embedding port, no code); NoiseCLR ruled out. Boundary Diffusion venue corrected (NeurIPS 2023, not CVPR 2023). New Tier 1 findings table added.
- **2026-04-14 (update)** — Three parallel deep-dives on Vox2Face, Arc2Face, and Asyrp completed. Major corrections applied:
  - Killed the "project to 5-token CLIP space" option. Arc2Face has one 768-d identity slot receiving a zero-padded 512-d ArcFace vector; there is no CLIP token space to project into. The only viable projection target is ArcFace 512-d.
  - Added the pairing problem to Step 2 (Vox2Face has ground-truth (speech, face) pairs; we have no ground-truth face per job). Three sub-options (a/b/c) with (c) — two-channel decoupling — recommended.
  - Step 4A (h-space) upgraded from "probe" to "primary drift mechanism". Boundary-Diffusion-style mean-difference recipe is the concrete path: N=500 low-sus vs N=500 high-sus generations, class-mean Δh, asymmetric application per Asyrp. Zero training. "Zombie" is in Asyrp's validated showcase — the exact direction we want is already demonstrated to work.
  - Added license section — Arc2Face is CC-BY-NC, project stays non-commercial.
  - Added detailed reading notes for Arc2Face, Vox2Face, and Asyrp.
