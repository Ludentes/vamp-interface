# vamp-interface Rebuild — Blind Alleys

**Date:** 2026-04-14
**Status:** Living document. Append new dead ends as they are discovered.
**Purpose:** Record research paths that looked promising from summaries / review-level descriptions but turned out to be wrong when verified against primary sources. Each entry says what we thought, what we actually found, why it is dead, and the minimum bar to reopen.

This document exists because during the 2026-04-14 rebuild-planning session we repeatedly invested design effort in options that a single primary-source read killed. The `feedback_shallow_research_risk.md` memory captures the lesson; this document captures the specific dead paths so they do not get re-explored in future sessions.

## How to use this document

- **Before** committing to a design option, search here by name (Arc2Face, Boundary Diffusion, RigFace, etc.) to check if it has already been evaluated and killed.
- **When** an option is killed after investigation, append it here with the four fields below. Do not delete entries — killed paths staying killed is the point.
- **To reopen** a path, a new primary-source reading or new published result must change at least one of the "why dead" bullets. State which one, in the commit that reopens it.

## Entry template

    ### <N>. <Name> — <one-line summary of the dead path>
    **Claim we originally believed:** …
    **What the primary source actually says:** …
    **Why it is dead for our use case:** …
    **Minimum bar to reopen:** …

## Entries

### 1. Arc2Face has a "5-token CLIP conditioning space" to project into

**Claim we originally believed:** Arc2Face's conditioning is a 5-token CLIP text embedding, so `qwen_1024 → 5×768 CLIP token space` is a viable projection target alternative to `qwen_1024 → ArcFace_512`. We wrote this into the rebuild plan as an escape hatch if ArcFace space failed the continuity pre-flight.

**What the primary source actually says:** Arc2Face paper (arXiv:2403.11641, ECCV 2024 Oral). The template `"photo of a ⟨id⟩ person"` tokenizes to 5 tokens `{e1, e2, e3, ŵ, e5}`. Only **one** token carries identity: `ŵ ∈ ℝ^768` is produced by **zero-padding** the 512-d ArcFace vector, not by a learned projection. The paper ablates against a 4-layer MLP (Fig. 9) and picks fine-tuned-CLIP + zero-padded direct substitution. After fine-tuning, the model "exclusively adheres to ID-embeddings, disregarding its initial language guidance" — the text channel is dead. The four non-identity tokens are frozen pseudo-prompt context.

**Why it is dead for our use case:** There is no CLIP token space to project into. The only projection target is ArcFace 512-d. Projecting `qwen_1024 → 5×768` would produce a 5-token conditioning vector, but Arc2Face's fine-tuned UNet only listens to the `⟨id⟩` slot and ignores the others, so the extra dimensions would do nothing — and we would not get the ID-lock Arc2Face provides because we would bypass the ArcFace extractor entirely.

**Minimum bar to reopen:** A new published result showing that Arc2Face (or a similar fine-tuned identity backbone) can be driven by multi-token text-space conditioning without retraining. None exists as of 2026-04-14.

---

### 2. Arc2Morph proves "ArcFace space is highly non-smooth"

**Claim we originally believed:** Arc2Morph (arXiv:2602.16569, Feb 2026) demonstrated that ArcFace embedding space is not locally smooth, and that slerp in Arc2Face's CLIP-output space is the correct continuity fix. This was a load-bearing justification for adding the "project to CLIP space" escape hatch.

**What the primary source actually says:** Arc2Morph makes **no smoothness claim at all**. Its only ablation (Table V) reports a +0.6 to +3 percentage-point Morphing Attack Potential improvement for slerp-in-CLIP-tokens over lerp-in-ArcFace at the **α=0.5 midpoint only**. No continuity metric, no LPIPS along a sweep, no Lipschitz analysis, no failure-mode discussion. The only verbal justification is hand-waving about "higher dimensionality and richer semantic structure." It is a morphing-attack-detection paper, not a visualization-continuity paper.

**Why it is dead for our use case:** A +3% MAP gain at one midpoint value does not establish non-smoothness and does not endorse a projection-space change. The continuity question Arc2Morph would need to answer is not the one it asked.

**Minimum bar to reopen:** A published α-sweep LPIPS study (or equivalent) on Arc2Face, either confirming ArcFace-space discontinuity or characterizing CLIP-space continuity. Our Step 5 pre-flight test will produce this data directly — at that point we will know for ourselves, and this memory becomes obsolete.

---

### 3. Vox2Face is a drop-in template for `qwen → ArcFace → Arc2Face`

**Claim we originally believed:** Vox2Face (MDPI *Information* 17(2):200, 2026) solves the cross-domain projection problem and we can clone its recipe by swapping the speech encoder for qwen.

**What the primary source actually says:** Vox2Face's Stage I alignment loss is a cosine/contrastive regression of the projected speech embedding against ArcFace features of the *ground-truth target face*. Stage II is SDS self-consistency. Dataset is HQ-VoxCeleb (curated speaker audio paired with face images). The training signal relies on the ground-truth (speech, face) pair.

**Why it is dead for our use case as a pure clone:** We have `(job posting, ???)` — no ground-truth face per job. Stage I has no supervision target. A pure clone collapses.

**Why it is not fully dead:** Vox2Face's *architecture* (AST → MLP → ArcFace-hypersphere → frozen Arc2Face with LoRA adapters) is still valid as a template. What must change is the loss: we substitute Stage I's supervised alignment with contrastive-only distance preservation (Step 2 option b) or split identity and drift into decoupled channels (option c, recommended). The architecture transfers; the loss does not.

**Minimum bar to fully reopen as a clone:** A published variant of Vox2Face trained without ground-truth pairs on a cross-domain embedding (e.g., text → face). None known.

---

### 4. Boundary Diffusion gives us a "class-mean drop-in" for Asyrp's `f_θ`

**Claim we originally believed:** Boundary Diffusion (Zhu et al., cited as CVPR 2023) showed that class-mean difference in `h_t` activations is a drop-in replacement for Asyrp's learned `f_θ`, applied asymmetrically per Asyrp's Theorem 1. This was Step 4A's "training-free, no labels" framing in the first version of the rebuild plan.

**What the primary source actually says:** Boundary Diffusion is arXiv:2302.08357 and was published at **NeurIPS 2023, not CVPR 2023** (first dead sub-claim). The method fits a **linear SVM hyperplane** on `h_t` activations at a searched mixing timestep `t_m` and uses the **unit normal vector** of the hyperplane as the edit direction — not class-mean difference (second dead sub-claim). The edit is a **symmetric one-shot shift** `x'_tm = x_tm + ζ · d(n, x_tm)` applied once at `t_m` — not an asymmetric per-step `P_t` modification (third dead sub-claim). Validated **only on unconditional DDPM/iDDPM** backbones (CelebA-HQ, LSUN, AFHQ) — never on Stable Diffusion, never with CFG, never on identity-fine-tuned models (fourth dead sub-claim). The paper itself notes: "the ability for unseen domain image editing is relatively limited compared to other learning-based methods." The SVM still needs labeled binary pairs — "training-free" means no network training, not no supervision (fifth dead sub-claim).

**Why it is dead for our use case as a drop-in:** Five independent things in the original framing were wrong. A faithful port to Arc2Face requires (a) labeled `(low-sus, high-sus)` pairs for the SVM fit, (b) DDIM inversion, (c) a search over `t_m`, and (d) trust that a one-shot symmetric shift does not fight Arc2Face's CFG + ArcFace ID lock — a regime the method has never been tested in. This is Step 4C in the revised plan — a research bet, not a drop-in.

**What is still alive:** Paired mean-difference in `h_t` applied via Asyrp's asymmetric formulation (Step 4B) — as a zero-training heuristic with no Boundary Diffusion pedigree, just the cheapest thing worth a 2-hour spike before paying Asyrp's 20-min `f_θ` training cost.

**Minimum bar to reopen Boundary Diffusion as a drop-in:** A published validation of the method on a CFG'd, text-conditioned, fine-tuned SD-family model. None known. Until then, Asyrp's learned `f_θ` is the correct primary because it is the only h-space method in this lineage actually designed for SD-family UNets.

---

### 5. RigFace is a viable backbone fallback if Arc2Face fails the continuity pre-flight

**Claim we originally believed:** RigFace (arXiv:2502.02465, Feb 2025) fully fine-tunes SD 1.5 with a second UNet as an identity encoder, so if Arc2Face fails the Step 5 Lipschitz test we can swap to RigFace as Plan B.

**What the primary source actually says:** The Identity Encoder is indeed a full second SD 1.5 UNet, initialized from SD weights and **fully fine-tuned**. But identity is not a vector — it is a set of **per-layer spatial feature maps** fused into the denoising UNet via "FaceFusion" (concat along width → joint self-attention → slice off denoising half → continue). This happens at every transformer block. The only input port is a **pixel-space reference image** encoded by a frozen VAE and passed through the Identity UNet. There is no embedding slot. The paper does no interpolation, no continuity, no smoothness experiments. **No code, no weights, no license** are released as of v3.

**Why it is dead as a fallback:** Three independent blockers. (1) No embedding input port — to drive RigFace from a qwen embedding we would need a separate model that turns embeddings into face images first, i.e. exactly the Arc2Face-shaped problem we are trying to solve. (2) Per-layer spatial feature conditioning is the worst case for local Lipschitz behavior — there is no single continuous identity manifold to walk along; interpolating "between" two RigFace identities requires per-layer feature-map interpolation across every transformer block of a full UNet, and there is zero empirical or theoretical evidence this is smooth. ReferenceNet-style conditioning is known brittle to such interpolation (AnimateAnyone, MagicAnimate community experience). (3) No code or weights — even if the architecture were suitable, reproducing 100k steps of full SD 1.5 UNet fine-tuning on Aff-Wild pairs is a real training project, not a swap-in fallback.

**Minimum bar to reopen:** Release of code + weights by the RigFace authors, combined with a published interpolation / continuity study showing per-layer feature conditioning behaves smoothly under identity interpolation. Neither exists.

**Replacement fallbacks to consider instead:** Models that accept a continuous identity vector as input — PhotoMaker, InstantID's IdentityNet used in isolation, or StyleGAN3 FFHQ w+ space.

---

### 6. NoiseCLR is an unsupervised alternative to paired h-space direction finding

**Claim we originally believed:** NoiseCLR (CVPR 2024) does unsupervised contrastive learning in Stable Diffusion's h-space, so it could replace Step 4A's need for 500 low-sus / 500 high-sus labeled pairs. Directions would emerge from the embedding distribution alone.

**What the primary source actually says:** NoiseCLR is arXiv:2312.05390. The contrastive InfoNCE loss operates at the **UNet predicted-noise output** (ε-space), not in h-space. Directions are learned as **pseudo-tokens in the cross-attention text-conditioning slot** — plugged into classifier-free guidance at inference via `ε_θ(x_t, d_e) − ε_θ(x_t, φ)`. Experiments run **only on vanilla SD 1.5** — no DreamBooth, no LoRA, no Arc2Face, no IP-Adapter, no identity fine-tunes. Apache-2.0 license, code at gemlab-vt/NoiseCLR.

**Why it is dead for our use case:** Four independent blockers. (1) Wrong slot — the pseudo-tokens occupy CLIP text-conditioning space, which Arc2Face has semantically repurposed for ArcFace identity; zero evidence NoiseCLR's InfoNCE objective converges or stays disentangled when `c` no longer spans a CLIP manifold. (2) Wrong geometry — NoiseCLR operates at ε-output via CFG, while our plan targets the UNet bottleneck h-space; different injection points, different semantics, not interchangeable. (3) No fine-tuned backbone validation — the paper never touches identity-fine-tuned models. (4) Unsupervised — we have labels (`sus_level`), we know which axis we want; an unsupervised method that finds K=100 FFHQ-salient axes (age, race, glasses, lipstick, hair) gives no guarantee "uncanny" is among them, and post-hoc labeling 100 directions to find the one we need is more work than the supervised paired approach.

**Minimum bar to reopen:** A published validation of NoiseCLR on a fine-tuned identity-conditioned diffusion model, *or* a reason to believe the axes it discovers on vanilla SD would transfer unchanged to Arc2Face. Neither exists. Could still be useful as an exploratory tool on vanilla SD 1.5 to see what axes FFHQ naturally exposes — low priority, not on the critical path.

---

## Meta-pattern across these entries

Every single one of the six dead paths above came from trusting a summary (review-level, subagent paraphrase, or abstract) over a primary-source read. In three of the six (Arc2Face 5-token, Arc2Morph non-smoothness, Boundary Diffusion drop-in) the summary was materially wrong in ways that would have cost days of implementation work. In the other three (Vox2Face clone, RigFace fallback, NoiseCLR unsupervised alternative) the summary described a real paper accurately at a high level but elided the preconditions that made it inapplicable to our use case.

**The operational lesson:** for every option that affects the expensive critical path, read the primary source's method section before committing. Summaries are fine for orientation ("which methods exist, in which lineage") but not for implementation ("what exactly does this method do"). See `~/.claude/projects/-home-newub-w-vamp-interface/memory/feedback_shallow_research_risk.md` for the full lesson.

## Related documents

- **Rebuild plan:** [2026-04-14-vamp-rebuild-plan.md](../design/2026-04-14-vamp-rebuild-plan.md)
- **Deeper research queue:** [2026-04-14-deeper-research-queue.md](../design/2026-04-14-deeper-research-queue.md)
- **Feedback memory:** `~/.claude/projects/-home-newub-w-vamp-interface/memory/feedback_shallow_research_risk.md`
- **Primitives memory:** `~/.claude/projects/-home-newub-w-vamp-interface/memory/project_vamp_rebuild_primitives.md`

## Changelog

- **2026-04-14** — Initial version. Entries 1-3 from the first deep-read round (Arc2Face, Arc2Morph, Vox2Face). Entries 4-6 from the Tier 1 deep-read round (Boundary Diffusion, RigFace, NoiseCLR).
