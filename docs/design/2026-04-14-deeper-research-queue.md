# Deeper Research Queue — Post-Shallow-Research-Lesson

**Date:** 2026-04-14
**Status:** Prioritized queue, ready to execute
**Context:** After fact-checking the Face Research review ([github.com/Ludentes/Face-Research](https://github.com/Ludentes/Face-Research)) and deep-reading Arc2Face, Arc2Morph, Vox2Face, and Asyrp, we discovered three load-bearing claims in the review were materially wrong. See [2026-04-14-vamp-rebuild-plan.md](2026-04-14-vamp-rebuild-plan.md) for the corrected plan and the feedback memory `feedback_shallow_research_risk.md` for the lesson.

This document queues the remaining areas of the review and adjacent literature that have the same risk profile — claims we paraphrased from summaries or fact-checks but have not verified against the actual paper methods. Prioritized by impact on the vamp-interface rebuild and the review's public credibility.

## Why this queue exists

During the 2026-04-14 rebuild-planning session, three claims we had relied on turned out to be wrong when we finally read the primary sources:

1. We believed Arc2Face had a "5-token CLIP conditioning space" we could project into as an alternative target. Reading the Arc2Face paper revealed this was a misreading: there is **one** 768-d identity token slot receiving a zero-padded 512-d ArcFace vector via direct substitution, not a 5-token conditioning space. An entire rebuild option had to be killed.
2. We believed Arc2Morph demonstrated that ArcFace space is "highly non-smooth." It does not. It only shows that slerp in CLIP-output space beats lerp in ArcFace space for midpoint morphing by ~2-3 percentage points on biometric attack metrics, with **no smoothness analysis**.
3. We believed Vox2Face was a drop-in template. It is — *if you have ground-truth (source, face) pairs*. Our use case does not, and this forced a new Step 2 sub-option.

All three errors came from relying on the review's summary-level language and subagent paraphrases rather than reading the papers themselves. The queue below covers what could go the same way.

## Tier 1 — Directly load-bearing for the rebuild (read before building)

These are the items that, if wrong, would invalidate specific steps in the rebuild plan. They should be read before committing to the implementation.

### T1.1 — Boundary Diffusion (Zhu et al., CVPR 2023)

**Why it matters:** Step 4A of the rebuild (h-space mean-difference drift axis) is based on the Asyrp-analyzing agent's claim that Boundary Diffusion showed "class-mean difference in h_t activations works as a drop-in replacement for Asyrp's learned f_θ, via SVM hyperplane normals." This claim is a paraphrase of a paraphrase — we have never actually read the Boundary Diffusion paper. Step 4A's "no training required" framing hinges on whether Boundary Diffusion actually does class-mean difference, or whether it uses a different method we mistook for mean-difference, or whether it has restrictions (certain timesteps only, certain diffusion backbones only) we do not know about.

**What to extract:**
- Exact direction-finding procedure: SVM hyperplane on h_t? PCA? Class-mean difference? Something else?
- What supervision is needed: labeled image pairs? Labeled text prompts? Unsupervised?
- Which diffusion backbone(s) is the method validated on? DDPM, iDDPM, Stable Diffusion?
- How is the direction applied at inference? Asymmetrically (Asyrp-style) or symmetrically?
- Any failure modes or prerequisites (specific samplers, specific timestep ranges, classifier-free guidance requirements)?
- Explicit code availability and license

**Priority:** Highest. This is the paper the rebuild is about to build on and we have not read it.

**Rough read prompt to dispatch:**

> Deep read of Boundary Diffusion (Zhu et al., CVPR 2023). Find the arXiv / OpenReview ID. Report: the exact direction-finding procedure in h-space (hyperplane SVM, class-mean, PCA, or other — quote the method section), supervision requirements, diffusion backbones validated, inference-time application, failure modes, prerequisites, code availability. Our use case is finding a "suspicious/uncanny" direction in Arc2Face's (fine-tuned SD 1.5) UNet bottleneck from paired low-sus / high-sus generations. Tell me whether Boundary Diffusion's method transfers directly or whether we're adapting it. Under 700 words, quote the method section directly.

### T1.2 — RigFace full paper (Wei et al., arXiv:2502.02465, Feb 2025)

**Why it matters:** RigFace is the backup plan if Arc2Face's continuity fails the Step 5 pre-flight test. The rebuild plan assumes RigFace "fully fine-tunes SD 1.5 and adds a second UNet as an identity encoder" — a characterization from internal research notes (`2026-04-12-rigface-technical-deep-dive.md`) that was not verified against the primary source. If RigFace has the same continuity issue as Arc2Face (i.e., its identity conditioning is also not locally Lipschitz), then Arc2Face failing Step 5 means the entire rebuild fails without a fallback. If RigFace has a fundamentally different conditioning mechanism that is smooth, we have a viable Plan B.

**What to extract:**
- The exact architecture of the Identity Encoder — is it really a full second UNet, and how does it feed into the main UNet? (Concatenation, cross-attention, FaceFusion — verify against the paper)
- The exact training regime (already verified: 30k Aff-Wild pairs, 100k steps, 24 GPU hours on 2× MI250X) — no need to re-verify
- How identity is represented internally — is it a continuous embedding like Arc2Face, or a set of per-layer features, and does this affect continuity behavior?
- Any interpolation or continuity experiments in the paper (probably none, but check)
- Whether the method supports projecting an arbitrary input embedding or requires a specific reference image

**Priority:** High. The fallback plan must be characterized before we build on the primary plan.

**Rough read prompt to dispatch:**

> Deep read of RigFace (arXiv:2502.02465, "Towards Consistent and Controllable Image Synthesis for Face Editing", Wei et al., Feb 2025). Focus on: (1) the exact architecture of the Identity Encoder — is it a full second copy of the SD 1.5 UNet, and how do its features feed into the main UNet? (2) whether identity is represented as a continuous vector (like Arc2Face) or as per-layer features (like IP-Adapter), and the implications for continuity, (3) any interpolation / continuity / smoothness experiments in the paper, (4) whether the method accepts an arbitrary input embedding or requires a specific reference image as input. Our context: we are deciding whether RigFace is a viable fallback to Arc2Face for a data-visualization use case that needs locally-Lipschitz identity conditioning. Under 900 words, quote the method section.

### T1.3 — NoiseCLR (Dalva & Yanardag, CVPR 2024)

**Why it matters:** Mentioned by the Asyrp-analyzing agent as an unsupervised alternative to h-space direction finding, using contrastive learning in SD's h-space. If it works on fine-tuned models and finds semantically meaningful directions without labeled supervision, it could replace Step 4A's paired-data requirement entirely — we wouldn't need to generate 500 low-sus / 500 high-sus samples, the directions would emerge from the embedding distribution alone.

**What to extract:**
- The contrastive objective formulation
- Whether it has been applied to fine-tuned SD checkpoints or only vanilla SD
- What kinds of directions it finds (global style, fine attributes, both?)
- Training cost and hardware requirements
- Code availability

**Priority:** Medium. Not required for the rebuild, but could simplify Step 4A further if it works.

**Rough read prompt to dispatch:**

> Deep read of NoiseCLR (Dalva & Yanardag, CVPR 2024) — find the arXiv ID. Extract: the contrastive objective used for unsupervised direction discovery in Stable Diffusion's h-space, whether the method has been validated on fine-tuned SD checkpoints (Dreambooth, Arc2Face, any identity-fine-tuned face model) or only on vanilla SD, the kinds of directions it discovers (global style vs localized attributes), training cost, code availability. Our context: we want an unsupervised alternative to paired-supervised h-space direction finding on Arc2Face; can NoiseCLR fill that role? Under 600 words.

## Tier 2 — High-impact for review credibility (read before publishing for venues)

These items are not blockers for the rebuild but affect the credibility of the Face Research review if/when we push it to arXiv, HN, or a conference.

### T2.1 — LivePortrait implicit keypoint mechanism (Ch05)

The review's central claim about LivePortrait is that it extracts "K canonical keypoints + rotation + expression δ + translation + scale + eye/lip scalars" from a source image and uses these to warp a driving frame. This was softened from "K=21" during the 2026-04-14 fact-check pass. The actual extraction network, the motion transfer mechanism (warping field generation? direct feature transfer?), and the compositing pipeline have not been read. LivePortrait is one of the most-cited methods in the review and production builders will want to understand the mechanism when choosing it.

### T2.2 — GaussianAvatars / SplattingAvatar / 3D Gaussian Blendshapes FLAME binding details (Ch07)

The review says Gaussians are "attached to FLAME vertices or triangles and move with mesh deformation." The precise binding — per-vertex offsets, per-triangle tangent-space coordinates, learned rotation propagation — differs across the three canonical methods and determines what you can drive them with and how they fail under extreme pose. The 2026-04-14 fact-check verified the papers exist and their FPS numbers are correct; it did not verify the binding mechanisms.

### T2.3 — EMO / Hallo / FLOAT architectural differences (Ch06)

The review treats these as a family of "expressive portrait audio-to-video" methods. The actual differences — EMO's reference-audio attention, Hallo's hierarchical architecture, FLOAT's flow-matching ODE — are the interesting part and were not characterized at the architectural level during the fact-check. Flow matching specifically is a real technical distinction from diffusion and deserves its own explanation in the chapter.

### T2.4 — ARKit ↔ FLAME solver "1 ms" latency claim (Ch03, Ch10)

This number appears in multiple chapters as a load-bearing latency budget for the "full capture-to-render loop runs at 60 FPS" claim in Ch07. It was likely copied from a blog post or README and never verified. If it is actually 10 ms or 50 ms, several downstream latency claims in the review break. Easy to verify empirically by running the solver on a test input.

### T2.5 — SMIRK vs DECA vs EMOCA output differences (Ch04)

The review treats these as drop-in alternatives. Their outputs are subtly different and the choice affects downstream generation quality. Worth a brief comparison section in Ch04.

## Tier 3 — Worth doing eventually

These are the claims that might be wrong but are low-impact. Batch them into a future pass.

- **HeadStudio "~2 hours" training time** (Ch07) — flagged uncertain during fact-check, language softened, never actually verified.
- **Concept Sliders rank-4 and "50+ sliders compose without degradation"** (Ch08) — cited as a key property, never verified against the paper.
- **MorphFace context blending mechanism** (Ch08) — LFW number corrected during fact-check, the mechanism itself never read.
- **PSAvatar and GaussianHead architectural details** (Ch07) — verified they exist, never verified what they do.

## Tier 4 — Market claims (verify only if challenged)

- **Live2D market sizing ~$2.54B** and **~6000 VTuber channels** (Ch11) — from secondary sources.
- **VTuber rig commission pricing $500-3000** (Ch11) — anecdotal from community chatter.
- **Which open-source projects are "dead" vs "active"** — based on GitHub star counts and last-commit dates, not actually running the code.

## Tier 1b — New items surfaced by the Tier 1 deep reads (2026-04-14 post-Tier-1)

The Tier 1 results themselves introduced new dependencies that have the same shallow-research risk as the original queue. Adding them here so they do not get skipped.

### T1b.1 — PhotoMaker, InstantID (IdentityNet in isolation), StyleGAN3 FFHQ w+ as Arc2Face fallbacks

**Why it matters:** The RigFace deep read (T1.2) killed RigFace as the primary fallback if Arc2Face fails the Step 5 continuity pre-flight. The rebuild plan now lists PhotoMaker, InstantID's IdentityNet in isolation, and StyleGAN3 FFHQ w+ as replacement fallback candidates. **I promoted these to fallback candidates without verifying any of their architectural claims against primary sources.** This is exactly the shallow-research failure mode the Tier 1 pass was supposed to demonstrate. If any of these do not in fact accept "a continuous identity vector as input" the way I claimed, the fallback list has another dead entry and we won't learn that until it matters.

**What to extract:**
- **PhotoMaker** (TencentARC, arXiv:2312.04461): exact conditioning input — single embedding or stacked tokens? Is the embedding a continuous vector or a set of per-layer features? Is the backbone fine-tuned (like Arc2Face) or adapter-only? Does it accept an arbitrary 512-d vector you could project into from qwen, or does it require reference image(s)?
- **InstantID** (arXiv:2401.07519): can IdentityNet be used in isolation from the full InstantID stack? What is its exact input — a single ArcFace vector, a composed ID+image set, something else? Is it locally Lipschitz in the continuous vector?
- **StyleGAN3 FFHQ w+ space**: this is well-established ground — the continuity behavior of w+ is known, but I have never characterized the quality degradation when a w+ vector is produced by projection (e4e, pSp, restyle) rather than by optimization, and quality at projection is what matters for our use case.

**Priority:** Medium (only matters if Step 5 fails for Arc2Face). **High** if we commit to any of these in a design document before verification, because that's how blind alleys get into the plan.

### T1b.2 — Asyrp's "homogeneity" and "linearity" claims on the actual h-space we'd use

**Why it matters:** The revised Step 4A primary path is Asyrp's learned `f_θ` trained against Arc2Face activations. The plan cites Asyrp's claims that `Δh` is "homogeneous across samples" and "linearly scalable in magnitude" as properties that let us use `α` as a drift dial. These claims come from the Asyrp paper's face-model experiments (CelebA-HQ on unconditional DDPM); we have not verified they hold on a CFG'd SD 1.5 fine-tune like Arc2Face. If linearity breaks, the "α controls magnitude" story breaks, and the sus-level → drift-strength mapping collapses.

**What to extract:**
- The exact claim in Asyrp's paper vs. the hedged wording ("roughly consistent" language)
- Any published replication on Stable Diffusion specifically (not just DDPM)
- HAAD (arXiv:2507.17554), cited in the original V6 verification as "implicitly confirming h-space survives few-shot fine-tuning" — never actually read first-hand. Read it.

**Priority:** Medium. Only matters if Step 4A reaches implementation without a quick empirical linearity check; a half-day probe on Arc2Face could answer this directly without needing the paper read at all.

### T1b.3 — Vox2Face full PDF (we only read the abstract)

**Why it matters:** The rebuild plan's Step 2 architecture is modeled on Vox2Face. The original deep-read session could only reach the MDPI abstract page, not the full PDF — all exact hyperparameters, loss weights, LoRA rank, Stage I alignment loss form (plain cosine vs. InfoNCE vs. ArcFace angular margin), and the data availability / code-release statement are still unrecovered. This is the same "summary-level trust" risk the feedback memory warns about — only narrower, because we know we don't have the full text.

**What to extract:** full PDF, all loss definitions, all hyperparameters, data availability statement, license.

**Priority:** Medium-high. Before implementing Step 2 option (c) specifically, we need the Stage II SDS loss formulation in enough detail to transfer it to a discriminator-free contrastive setting.

### T1b.4 — Arc2Morph's ControlNet + BEN2 pipeline claims

**Why it matters:** The rebuild plan's "reading notes" section for Arc2Morph describes an architecture with EMOCAv2 → 3D normal map → ControlNet for pose and BEN2 for background matting. That description came from a single pass by one subagent. We have not verified (a) that EMOCAv2 is actually the extractor Arc2Morph uses, (b) that the ControlNet in question is normal-conditioned, (c) that BEN2 is what Arc2Morph calls out by name for matting, (d) that any of this matters for our use case. Arc2Morph itself is not on our critical path, but if we cite its pipeline in the review or as inspiration for the v3 anchor work, cited details need to be right.

**Priority:** Low. Only matters if we build on Arc2Morph's pipeline as opposed to just its ablation result.

### T1b.5 — "Arc2Face inference at ~1-2 s per image on a consumer GPU" claim

**Why it matters:** This number appears in multiple places in the rebuild plan and informs the caching strategy ("pre-generate offline, serve static"). It is a community-reported figure, not from the paper. If the real number is 5-10 s per image at the settings we actually use (CFG=3.0, 25 DPM-Solver steps, 512×512), the offline generation budget for tens of thousands of jobs scales accordingly.

**What to extract:** direct benchmark on our actual target GPU (not a paper claim). Half-hour task, no research needed. Queue as an implementation ticket, not a paper read.

**Priority:** Medium. Empirical, not paper-based. Run before committing to the caching plan.

## Execution recommendation

After compaction:

1. Dispatch three parallel deep-read agents for **T1.1, T1.2, T1.3** using the rough prompts in each section. Budget ~20 min per agent, ~4 hours of total work (mostly waiting).
2. Apply any architectural corrections to the rebuild plan ([2026-04-14-vamp-rebuild-plan.md](2026-04-14-vamp-rebuild-plan.md)).
3. If any of the corrections affect the Face Research review, apply them to the chapters and commit with the same pattern as the 2026-04-14 Ch08 correction (`ca49381` on `github.com:Ludentes/Face-Research`).
4. Decide whether to proceed with the rebuild Step 5 (continuity test) based on T1.1 and T1.2 findings. If Boundary Diffusion is confirmed as a clean mean-difference method and RigFace is confirmed as a viable fallback, proceed. If either introduces new unknowns, stop and queue more deep-reads.
5. Defer Tier 2 to a dedicated "pre-arXiv reading sprint" before submitting the review to any venue.

## Related documents

- **Rebuild plan:** [2026-04-14-vamp-rebuild-plan.md](2026-04-14-vamp-rebuild-plan.md)
- **Face Research review:** [github.com/Ludentes/Face-Research](https://github.com/Ludentes/Face-Research)
- **Lesson memory:** `~/.claude/projects/-home-newub-w-vamp-interface/memory/feedback_shallow_research_risk.md`
- **Primitives memory:** `~/.claude/projects/-home-newub-w-vamp-interface/memory/project_vamp_rebuild_primitives.md`

## Changelog

- **2026-04-14** — Initial version. Written after Vox2Face / Arc2Face / Asyrp deep reads corrected three rebuild-plan claims, before user-initiated compaction. Captures the remaining read queue so it survives the context reset.
- **2026-04-14 (post-Tier-1)** — All three Tier 1 items (T1.1 Boundary Diffusion, T1.2 RigFace, T1.3 NoiseCLR) completed. New Tier 1b section added with items the Tier 1 results themselves surfaced: T1b.1 (PhotoMaker / InstantID / StyleGAN3 w+ as unverified Arc2Face fallbacks), T1b.2 (Asyrp linearity on SD fine-tunes), T1b.3 (Vox2Face full PDF), T1b.4 (Arc2Morph pipeline details), T1b.5 (Arc2Face inference latency benchmark).
