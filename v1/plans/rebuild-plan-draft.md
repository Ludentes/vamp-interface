# vamp-interface Rebuild Plan

**Date:** 2026-04-14 (rewritten late session after first-hand reads of HyperFace and the Vox2Face full PDF)
**Status:** Design, ready to execute pending the continuity pre-flight test.
**Supersedes:** All earlier drafts of this document in this session. The evolution is captured in git history; the architectural conclusions below are the ones to build on.

## TL;DR

Build vamp-interface's continuous embedding-to-face mapping on top of **StyleGAN3 FFHQ**, with a learned MLP `qwen_1024 → W` trained against randomly-but-deterministically-assigned W targets and a pairwise distance-preservation regularizer. Drift axis comes from Step 4 on the chosen backbone (h-space or LoRA-weight-space direction). Fall through to an Arc2Face-based plan only if StyleGAN3 photorealism is inadequate for the visualization task.

The pivot away from Arc2Face as the primary Step 2 target is driven by three hard-won findings from primary-source reads:

1. **Every published identity-conditioning recipe is optimized for separation, not smoothness.** Arc2Face's backbone is fine on this front (it's just a generator), but ArcFace space itself is separation-maximizing by construction, Vox2Face's Stage I loss is AM-Softmax + InfoNCE (both class-discriminative), and HyperFace is a literal max-min-angle packing. None of these give us local Lipschitz behavior for free.
2. **Vox2Face's Stage I is not usable without ground-truth pairs**, and the paper runs no ablation of SDS alone without Stage I — we have no evidence Stage II trains stably from a random gϕ init. The earlier "Stage II is the transferable piece" framing was unsupported.
3. **StyleGAN3 FFHQ W-space is the only candidate in our option set that is locally Lipschitz by construction**, and our use case (learned projection into W, not real-image inversion) dodges the e4e/pSp/ReStyle reconstruction penalty that has historically made StyleGAN inversion hard.

This is a visualization problem, not a face-recognition problem. The tools built for FR have the wrong objective for us, and once we accept that, the cleanest path is the one that never puts us on the FR track.

## Why rebuild at all

Original vamp-interface (see [diffusion-approach.md](diffusion-approach.md), [scenarios.md](scenarios.md)):

1. Fixed anchor face (PNG)
2. Two-pass SDXL img2img: identity pass + expression pass
3. Denoising strength as a scalar mapped from `sus_level`
4. LoRA-based uncanny encoding (v8c curve)
5. Seeded per job_id, pre-generated and cached

Known problems: denoising strength mutates identity and expression together, no principled two-channel separation, v8c LoRA overshoots into zombie / skin-lesion territory, gendered LoRA response, not generalizable beyond fraud.

The [Face Research review](https://github.com/Ludentes/Face-Research) identified a newer primitive family (Arc2Face + expression adapter, h-space direction finding, Concept Sliders, FLAME-conditioned diffusion) that collapses the engineering surface of the rebuild. The question this plan answers: given what we *actually* found in those primary sources — including the dead paths that surfaced during the deep reads — what's the cleanest path to a continuous embedding-to-face visualization?

## Core design principle

**vamp-interface is a visualization problem, not a face-recognition problem.** Close embeddings should produce close faces. This is a *local Lipschitz* requirement on the mapping `qwen → face_pixels`. It is NOT a requirement for the generated faces to be "recognizable as the same person" or for them to have "high identity similarity to any reference". Those are FR metrics and they are what every published tool in this space is optimizing against. We are not building for those metrics.

The implication: we want a conditioning space that is smooth by construction, not one that is merely discriminative.

## Step 1 — Replace the anchor paradigm

**Old:** fixed PNG image, img2img drift.
**New:** fixed identity vector (W for StyleGAN3, ArcFace-512 for Arc2Face path) as the anchor. All generated faces are conditioned on vectors near the anchor in the chosen space.

## Step 2 — Projection `qwen → face_generation_input_space`

**Recommended path: `qwen_1024 → StyleGAN3 native W` (Option A).**

### Option A — StyleGAN3 FFHQ W-space direct projection (primary)

Train a plain MLP `g: ℝ^1024 → ℝ^512` (native W dim) with two losses:

1. **Target-regression loss.** Pick N deterministic-but-arbitrary W vectors (e.g., sample via StyleGAN3's mapping network from random Z seeds, or use real-face-inversion centroids from FFHQ). Assign one to each qwen cluster. Train `g` to regress the cluster's qwen embedding to its assigned W vector: `L_reg = ||g(e) − W_target(cluster(e))||²`.

2. **Pairwise distance-preservation regularizer.** For mini-batch pairs `(e_A, e_B)`:
   ```
   L_dist = |d_W(g(e_A), g(e_B)) − C · d_qwen(e_A, e_B)|
   ```
   `C` is a learned or fixed scale. This is the Lipschitz-preserving term.

**Total:** `L = L_reg + λ · L_dist`.

**Why this is the primary:**
- W-space is **locally Lipschitz by construction** — this is the well-known property that made StyleGAN the default face-editing latent for 2020-2022.
- The e4e/pSp/ReStyle reconstruction penalty literature does NOT apply — that penalty is a symptom of real-image inversion being hard, and we're not inverting anything. We're learning a fresh projection into native W.
- Single deterministic forward pass at inference: qwen → W → StyleGAN3 synthesis → PNG. No sampler stochasticity, no CFG, no SDS loop, no LoRA training.
- No pairing problem (targets are deterministically assigned, not supervised from ground-truth faces).
- No dependency on ArcFace, which was designed for separation not smoothness.
- No CC-BY-NC license constraint (StyleGAN3 is Nvidia Source Code License — verify commercial terms before productizing, but non-commercial research is permitted).

**Caveats:**
- Dated photorealism vs SDXL/FLUX for hair, backgrounds, non-portrait content. For a fixed square portrait crop this is mostly fine; for anything else it's a limitation.
- Fixed square portrait crop. Not a limitation for the scam-hunter / analyst / student use cases, which all render thumbnails.
- The assignment of W vectors to qwen clusters is arbitrary. This is a feature, not a bug — we want consistent-within-cluster, distinct-between-cluster, and smooth-within-embedding-variation. Random assignment gives us that.

**Training cost estimate:** MLP is tiny; full training on a consumer GPU runs in hours, not days. Orders of magnitude cheaper than any SDS-based plan.

### Option B — Arc2Face path with contrastive distance-preservation (fallback if A's photorealism is inadequate)

Train `g: ℝ^1024 → ℝ^512 (ArcFace)` with **only** the pairwise distance-preservation regularizer:
```
L = |d_ArcFace(g(e_A), g(e_B)) − C · d_qwen(e_A, e_B)|
```
No ground-truth targets. No AM-Softmax. No InfoNCE. This is deliberately the weakest possible Stage I — just enough to keep the projected vectors on a roughly unit-norm sphere while preserving relative distances.

Then optionally apply **Vox2Face's Stage II SDS self-consistency** (verified 2026-04-14 from the full PDF) to pull projected vectors onto the high-probability Arc2Face manifold:
- LoRA r=16, scale 16, on Q/K/V/O of both the Arc2Face CLIP text encoder and the UNet.
- Equation 14: `L_self = E[w(t) · ||ε_θ(z_t, c_gen, t) − ε_θ(z_t, c_self, t)||²]` with stop-gradient on `c_self`.
- 20k steps at AdamW LR 1e-6, batch 16, A800-class GPU. ~34 hours wall-clock per Vox2Face Table.

**Risk:** Vox2Face does not run a "Stage II alone" ablation. The `w/o ArcFace` ablation (Cosine 0.242 with shallow direct regression, vs 0.322 full) shows what happens without *any* ArcFace distillation, and the `w/CosReg` ablation (Cosine 0.317, R@10 22.5 — retrieval tanks) shows what cosine-regression alone gives. Neither of these tells us what SDS + distance-preservation does without AM-Softmax + InfoNCE. **This is a research bet, not a proven pipeline.** Run a small-scale convergence probe (1k steps on 100 identities) before committing to full training.

### Option C — HyperFace as anchor codebook (last resort)

Use HyperFace's packing to generate N well-spread ArcFace anchors. Assign each qwen cluster to its nearest anchor. Generate per-job faces as interpolations among k-nearest anchors in ArcFace space, weighted by qwen distance, then renormalized. This uses HyperFace as a *codebook*, not as a Stage I loss — HyperFace's separation objective is inherited only at the anchor scale, not at the job scale. Local smoothness comes from the interpolation step.

**Caveats:**
- HyperFace packing is explicitly separation-maximizing. Two adjacent qwen clusters may end up assigned to far-apart packed anchors, so cluster-level adjacency is lost. Only within-anchor local smoothness survives.
- Adds the assignment problem (nearest-unused, Hungarian, or OT).
- N must be chosen up-front; adding new identities later requires re-packing.

### Killed Step 2 paths (from blind-alleys doc)

- **Vox2Face clone with AM-Softmax + InfoNCE**: needs ground-truth (source, face) pairs we don't have. Entry 3 in blind alleys.
- **PhotoMaker fallback**: input is a stack of CLIP-image tokens from real faces, not a continuous vector. Entry 7.
- **InstantID IdentityNet in isolation**: needs ArcFace + landmark spatial map + IP-Adapter branch; cannot isolate ID channel. Entry 8.
- **RigFace fallback**: per-layer FaceFusion feature conditioning (worst case for Lipschitz), no embedding input port, GitHub 404, weights-only on HF. Entry 5.
- **HyperFace-as-Stage-I**: separation-maximizing objective breaks continuity hypothesis. See HyperFace entry in paper findings log; the intended composition does not preserve local qwen structure.

## Step 3 — Generator

**For Option A:** frozen StyleGAN3 FFHQ. Public code, public weights, deterministic synthesis.

**For Option B:** frozen Arc2Face. Public code at `github.com/foivospar/Arc2Face`, weights at `huggingface.co/FoivosPar/Arc2Face`. CC-BY-NC. Inference ~4.3 s per 512×512 image on A800 (Vox2Face Table measurement) — the community "1-2 s" number appears to have been optimistic. Recompute the caching budget against 4.3 s for 10k+ jobs.

**For Option C:** frozen Arc2Face (same as B).

**Upgrade path if SD 1.5 ceiling becomes limiting:** InfiniteYou (arXiv:2503.16418, ByteDance, Mar 2025, FLUX DiT backbone, InfuseNet residual injection of ArcFace features, code + weights released). **PROVISIONAL** — not first-hand read yet. Queued as T1b.10. Do not commit before reading.

## Step 4 — Drift axis (encoding sus_level)

Regardless of which Step 2 option wins, the drift axis lives in the backbone's editing latent. For StyleGAN3 (Option A), there are well-established W-space edit directions that don't need anything Asyrp-related. For Arc2Face (Options B, C), the following ranking holds.

**Mandatory α-sweep continuity gate before any drift-axis training.** Asyrp's homogeneity/linearity/timestep-consistency claims are NOT established on any latent-diffusion or CFG backbone per the paper's Section 5.3 (the "roughly consistent" hedge is load-bearing) and the paper explicitly lists "h-space in latent diffusion models" as future work. HAAD (arXiv:2507.17554) is an adversarial-attack paper, not a transfer study — my earlier citation of it as supporting evidence was wrong. Before investing in `f_θ` training, run the half-day gate: fixed seed + fixed identity token, single cheap learned Δh_mid, sweep α ∈ [−3, +3] in 0.5 steps, measure ArcFace-ID drift and LPIPS curves across 20 random IDs. Non-monotone or asymmetric collapse kills the option.

**Option 4S (StyleGAN3 path).** Train a W-space direction classifier on paired low-sus / high-sus latents, or use an established W-space edit (InterFaceGAN, StyleFlow-style) for a "disgusted" or "unhealthy" axis. Cheap, well-trodden territory.

**Option 4B (mean-difference heuristic, Arc2Face path, 2-hour spike).** Generate N=500 low-sus + N=500 high-sus Arc2Face samples, DDIM invert, `Δh = mean(h_high) − mean(h_low)`, apply asymmetrically per Asyrp Theorem 1. This is not Boundary Diffusion — Boundary Diffusion uses linear SVM hyperplane normals with symmetric one-shot shifts at a searched `t_m` on unconditional DDPM backbones, not mean-difference on SD. Mean-diff has no published pedigree on SD; it's just the cheapest thing worth a spike.

**Option 4A (Asyrp learned f_θ, Arc2Face path, 20 min training).** CLIP directional loss with `y_source="photo of a person", y_ref="photo of a zombie"`. "Zombie" is explicitly in Asyrp's validated CelebA-HQ edits. 1000 samples, 40 timesteps, 1 epoch, ~20 min on 3× RTX 3090. Asymmetric reverse, editing interval `[T, t_edit]`, DDIM η=0. Gate on the α-sweep first.

**Option 4F (LoRA weight-space direction finding, arXiv:2406.09413).** Fit PCA / linear classifiers in LoRA weight space. Natural fit for already-fine-tuned backbones like Arc2Face. **PROVISIONAL** — queued as T1b.9, not first-hand read yet. Promising in principle but don't commit before reading.

**Option 4D (Concept Slider trained directly against Arc2Face).** Do not port sliders from vanilla SD 1.5 (V4 verification: LoRAs don't transfer cleanly to fine-tuned backbones). Train fresh against Arc2Face with paired anchor / drifted images.

**Option 4E (Arc2Face expression adapter, FLAME coefficients).** Smoke test only. Uncanny is not a FLAME dimension.

**Option 4C (Boundary Diffusion faithful port, research bet).** Linear SVM hyperplane normal in h-space at a searched `t_m`, symmetric one-shot shift. Needs labeled pairs, DDIM inversion, `t_m` search, untested CFG interaction. Deprioritized — don't invest here until 4A/4B/4F all fail.

## Step 5 — Continuity pre-flight (mandatory)

Before any Step 2 training, verify the chosen backbone is locally Lipschitz in its conditioning space on a held-out test:

1. Sample 20-50 pairs of real conditioning vectors (FFHQ W for Option A, real ArcFace for Options B/C).
2. Sweep α ∈ [0, 1] in 20 steps.
3. Generate ~20 frames per sweep via the frozen generator.
4. Measure pairwise LPIPS between consecutive frames.
5. Distribution analysis:
   - Smooth, short-tailed → proceed with Step 2 training.
   - Bimodal or heavy-tailed → backbone fails continuity, switch options.

For Option A, the pre-flight is likely a formality since StyleGAN W-space is famously smooth, but it's still cheap insurance and doubles as a sanity check on our measurement protocol.

For Option B, this is where Arc2Face's ArcFace-512 behavior gets characterized for the first time on a real metric — no published work has run this specific experiment, so we'd be filling a research gap.

Budget: half a day of compute, zero training.

## Step 6 — Pipeline and caching

For each job: (a) compute qwen embedding, (b) project to conditioning space via `g`, (c) generate via frozen generator, (d) apply Step 4 drift direction with strength `f(sus_level)`, (e) cache 256×256 PNG keyed by job_id. Fully offline, static serving. Same pattern as current vamp-interface.

**Caching budget update:** Arc2Face on A800 runs ~4.3 s per 512×512 image (Vox2Face Table). For 10k jobs → ~12 hours of generation per run. StyleGAN3 is faster (sub-second per image on the same class of GPU). This is another reason to prefer Option A: orders of magnitude cheaper offline generation.

## Go/no-go recommendation

**Proceed with Option A (StyleGAN3).** Run the Step 5 pre-flight first as a cheap sanity check, then build Step 2 Option A directly. Budget ~1 week from the pre-flight to a working offline pipeline.

If Option A's photorealism is inadequate for the scam-hunter / analyst / student evaluations, fall through to Option B with a small-scale convergence probe first (1k steps on 100 identities, check that SDS + distance-preservation regularizer actually trains stably without AM-Softmax + InfoNCE). Only commit to Option B's full ~34-hour Stage II run after the probe confirms training stability.

**Critical path (Option A, optimistic):**

1. Step 5 continuity pre-flight on StyleGAN3 W — half day of compute
2. Step 2 Option A training — 1-2 days (MLP + regularizer, small dataset)
3. Step 4S W-space drift direction — half day
4. Step 5 pre-flight for the drift direction (α-sweep on the chosen axis) — half day
5. Step 6 pipeline wiring — 1-2 days
6. Offline generation of per-job faces — hours to a day depending on N jobs
7. Launch user study on the new version

**If we end up on Option B, add ~1 week for the SDS convergence probe + full Stage II training.**

## Residual reads before implementation

These are the remaining unverified items from the Tier 1b session. Addressing them is not strictly blocking for Option A but becomes blocking for Options B/C or for backbone upgrades.

- **T1b.8** — FEM (arXiv:2602.13168) — 20 min KAN-vs-MLP check. Only matters if Option A's MLP underperforms and we want to try KAN.
- **T1b.9** — LoRA weight-space editing (arXiv:2406.09413) — read before committing to Option 4F on the Arc2Face path.
- **T1b.10** — InfiniteYou (arXiv:2503.16418) — read before treating it as a backbone upgrade candidate.

## Related documents

- **Paper findings log** (all verified reads): [docs/research/papers/2026-04-14-paper-findings.md](../research/papers/2026-04-14-paper-findings.md)
- **Blind alleys** (8 killed paths): [docs/research/2026-04-14-rebuild-blind-alleys.md](../research/2026-04-14-rebuild-blind-alleys.md)
- **Deeper research queue**: [2026-04-14-deeper-research-queue.md](2026-04-14-deeper-research-queue.md)
- **Face Research review**: [github.com/Ludentes/Face-Research](https://github.com/Ludentes/Face-Research)
- **Shallow research lesson memory**: `~/.claude/projects/-home-newub-w-vamp-interface/memory/feedback_shallow_research_risk.md`
- **Primitives memory**: `~/.claude/projects/-home-newub-w-vamp-interface/memory/project_vamp_rebuild_primitives.md`

## Changelog

- **2026-04-14 (initial)** — Written after Face Research review published, V1-V7 verification, Arc2Morph deep read.
- **2026-04-14 (Tier 1 update)** — Vox2Face / Arc2Face / Asyrp deep reads corrected three earlier claims: killed the "5-token CLIP space" projection option, added the pairing problem, promoted h-space to primary drift.
- **2026-04-14 (Tier 1 deep reads)** — Boundary Diffusion / RigFace / NoiseCLR deep reads. Boundary Diffusion venue and method corrected. RigFace removed from backbone list. NoiseCLR ruled out.
- **2026-04-14 (Tier 1b deep reads — earlier)** — Fallback verification (PhotoMaker and InstantID killed, StyleGAN3 promoted, InfiniteYou added), Asyrp linearity claims corrected to "not established on SD", LoRA weight-space added as Option 4F, HyperFace promoted to Step 2 Option (d) RECOMMENDED (based on agent summary, not first-hand read), 2026 successor scan confirms Arc2Face is still the reference.
- **2026-04-14 (full rewrite — late session)** — First-hand reads of HyperFace and Vox2Face full PDF completed. **Major pivot**: HyperFace is separation-maximizing and incompatible with continuity, and Vox2Face's Stage II has no "without Stage I" ablation, so both are downgraded. **StyleGAN3 FFHQ W-space is now the primary Step 2 path** (Option A) because it's the only candidate that is locally Lipschitz by construction. Arc2Face path becomes Option B (fallback). This supersedes all earlier Step 2 recommendations in this document.
