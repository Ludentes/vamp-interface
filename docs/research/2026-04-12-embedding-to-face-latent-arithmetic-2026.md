# Research: Embedding → Face via Latent Arithmetic — 2025-2026 State of the Art

**Date:** 2026-04-12
**Question:** What is the cleanest 2025-2026 path for `embedding → face` where attribute edits are vector additions in a learned latent space — i.e. the StyleGAN W-space paradigm at modern quality?
**Sources:** 20+ papers (CVPR/ECCV/NeurIPS/ICCV 2024-2025, arXiv pre-2026)

---

## Executive Summary

The user's framing is correct. The StyleGAN paradigm — `L = f(embedding)`, `G(L + α·v) = edited face` — is the right mental model, and it is alive and well in 2025-2026. It just doesn't live in StyleGAN anymore. The same property (semantic linear directions in a learned latent space) exists in three distinct modern spaces: **diffusion h-space** (bottleneck activations, no retraining required), **3DMM parameter space** (FLAME blendshapes, physically grounded), and **disentangled flow-matching latent spaces** (2025 frontier). The key architectural gap is the mapping layer from an arbitrary job-posting embedding to the target face latent; none of the off-the-shelf models provide this out of the box, but it is a small trained projection (or even a linear map) in every candidate architecture. Our current Flux + ConditioningAverage approach goes *through text* as an intermediate; the alternatives below treat the face latent as a first-class vector space.

---

## The Mathematical Claim Being Evaluated

Current architecture (v3 directional anchors):
```
job_embedding → cosine similarity → softmax weights → compose text sentences
    → CLIP/T5 encode → ConditioningAverage → Flux denoising → face
```

The attribute edits (sus_band, expression phrases) live in **natural language**, not in a vector space. The "blending" is a weighted average of text conditioning vectors, which works but is indirect: the semantic structure is editorial, not geometric.

The StyleGAN-inspired alternative:
```
job_embedding → learned_projection → face_latent L
face = G(L)
edited_face = G(L + Σ αᵢ · vᵢ)  where vᵢ are learned semantic direction vectors
```

`vᵢ` might be: `v_sus` (uncanny direction), `v_age`, `v_gender`, `v_archetype` — found by InterFaceGAN-style classifiers or PCA on the latent space. The key property: **edits compose linearly, without going through text**.

The research question is: does this cleaner paradigm exist at modern quality?

---

## Key Findings

### 1. Diffusion h-Space: StyleGAN W-Space Inside a Frozen Flux/SDXL Model

The most immediately applicable finding: diffusion models already have a semantic latent space equivalent to StyleGAN's W-space, called **h-space** (the bottleneck activations of the UNet/transformer at each denoising timestep).

**Asyrp** (Kwon et al., ICLR 2023) established the foundational properties: h-space is homogeneous (edits transfer across images), linear (vector arithmetic works), robust (small edits produce small changes), and timestep-consistent [1]. **Haas et al.** (IEEE FG 2024) applied this directly to face editing — unsupervised PCA on h-space finds directions for pose, gender, and age without any classifier training [2]. The arithmetic is exactly `h + α · direction → modified face`.

**Self-Discovering Interpretable Directions** (Li et al., CVPR 2024) extends this: given a target concept (e.g., "sus expression", "evasive gaze"), you can *learn* a direction vector `v_concept` through the denoising process itself, with no data collection and no text involved after the initial concept is defined [3].

**DiffFERV** (IJCAI 2025) applies this to facial expression video editing, using both linear and geodesic traversal in h-space [4].

**What this means for this project:** You could, right now without retraining Flux:
1. Use DDIM inversion to map the anchor face to an h-space trajectory
2. Apply PCA or classifier-guided direction finding to identify `v_sus`, `v_courier`, etc.
3. At generation time: project job_embedding → h-space perturbation, add `α · v_sus`
4. Run Flux denoising with the modified h-space activation

This is vector arithmetic in the face latent space of the *exact model we're already using*.

### 2. Arc2Face: Arbitrary Embedding → Diffusion Latent (ECCV 2024 Oral)

**Arc2Face** [5] is the closest published architecture to the target pipeline. It takes a 512-d ArcFace face recognition embedding as input, projects it through an adapted encoder, and uses it as the conditioning signal for Stable Diffusion. No text prompts. The embedding IS the face identity.

The 2025 extension ([6], blendshape-guided expression adapter) adds FLAME blendshape parameters as a second conditioning channel — separate from identity, orthogonal, controlling expression independently via cross-attention. This is exactly the two-channel identity/expression architecture we have in v3, but with geometric/parametric channels instead of text.

**The gap for this project:** Arc2Face's input is an ArcFace embedding (face recognition features from a real face image). Our input is a job-posting qwen3-embedding. These are different spaces. However: Arc2Face's architecture is a projection layer (ArcFace 512-d → CLIP 768-d via learned MLP) on top of frozen SD. Training a new projection `qwen3 1024-d → Arc2Face_conditioning_space` is a small supervised problem if you have paired data, or can be learned self-supervised from the job corpus distribution.

Arc2Face has no inherent expression range limitation — it inherits SD/FLUX's expression flexibility, unlike StyleGAN.

### 3. 3DMM + Diffusion: Explicit Parametric Geometry as the Vector Space

The 3D Morphable Model (FLAME [7]) provides a **physically grounded vector space** for face attributes:
- Identity: 50-d shape coefficient vector `β_id`
- Expression: 12-d blendshape coefficient vector `β_exp`
- Pose: 6-d rotation/translation
- All three are **independent by construction** (they're separate PCA spaces from different training data)

**RigFace** (2025) [8] and **MorphFace** (2025) [9] use FLAME parameters as explicit conditioning for a fine-tuned diffusion backbone. At inference: `β_id + α · δβ_exp = new_face`. The expression delta `δβ_exp` is a 12-d vector, completely orthogonal to identity. Users rated 59.3% of RigFace expression edits as more realistic than ground truth.

For this project: `job_embedding → small MLP → β_id (50-d)` gives face identity. Separately: `sus_level → β_exp_sus (12-d)` defines the sus expression. Blend at generation time. This pipeline has stronger mathematical guarantees than any latent interpolation approach because FLAME's parameter spaces are orthogonal by construction.

**EMOCA** [10] is a monocular 3DMM reconstruction network — it extracts FLAME parameters from a single face image. This could be used to map a chosen anchor face into FLAME parameter space as a starting point, then apply deltas.

### 4. StyleGAN3 Fine-Tuned on Expressive Data: Expression Range Solved

The original StyleGAN concern was that FFHQ training gives almost no expression variation. This is now fixable. **GANmut** (2024) [11] fine-tunes a GAN on AffectNet (1M images, 7 emotion categories + valence/arousal continuous axes). The expression range expands dramatically. **Emotion amplification via fine-tuned StyleGAN** (SciOpen 2025) [12] demonstrates this specifically for expression intensity on video.

**Implication:** the FFHQ expression limit is a training data problem, not an architectural one. If you retrain or fine-tune StyleGAN3 on AffectNet + FFHQ + expression-labeled video frames, you get W-space with both identity AND expression variation. The direction-finding methods (SeFa, InterFaceGAN) then find `v_smile`, `v_sus`, `v_evasive_gaze` in that expanded W-space. The expression blendshape axis from v3 (sus_band phrases) becomes a learned latent direction instead of text.

StyleGAN3 is ~30M parameters (the generator). Training or fine-tuning it is hours on a single GPU, not the week-long project it would have been in 2019.

### 5. Disentangled Flow Matching (2025 Frontier)

**Disentangled Representation Learning via Flow Matching** (arXiv 2025) [13] directly attacks the problem: learn factor-conditioned flows in a compact latent space, with a non-overlap regularizer that forces orthogonality between factors. The result is a flow-matching model where each factor (identity, expression, pose) has its own independent flow. Vector arithmetic: `factor_1_value · e_1 + factor_2_value · e_2 → face`. Strong experimental results, but face-specific applications are not yet published — this is a general framework.

**Flow Matching in Latent Space** (Dao et al., NeurIPS 2023 → ongoing) [14] trains flow matching in a pretrained VAE's latent space, giving cheap training on top of existing compression. The latent space is the same one you'd do arithmetic in. This is the architecture that makes "small flow-matching model on top of existing VAE" viable.

### 6. What Is Actually Dead

The only genuinely stagnant area is the original StyleGAN3 architecture frozen at FFHQ. NVlabs has not published a major successor since 2021. The direction-finding methods (InterFaceGAN, SeFa, StyleCLIP) work fine but are applied to the same FFHQ-trained W-space from 2021. No one has released an FFHQ-replacement StyleGAN3 checkpoint trained on a richer expression corpus, though the tooling to do so is entirely available.

The GAN vs diffusion "war" is settled: diffusion wins on photorealism in all 2024-2025 benchmarks. GANs retain speed and explicit latent structure advantages. The field has moved toward hybrid architectures rather than pure GAN or pure diffusion.

---

## Comparison: Three Architectures for `Embedding → Editable Face`

| Property | h-space directions (Flux) | Arc2Face + blendshapes | StyleGAN3 + AffectNet |
|---|---|---|---|
| **Vector arithmetic** | ✓ (linear in h-space) | ✓ (FLAME + learned projection) | ✓ (W-space by definition) |
| **Expression range** | Full (Flux) | Full (SD inherits it) | Must fine-tune on AffectNet |
| **Input: arbitrary embedding** | Via DDIM inversion + adapter | Via projection MLP (trainable) | Via W-projection MLP (trainable) |
| **Training required** | Direction finding only (hours) | Small projection MLP | Fine-tune StyleGAN3 (1-2 days) |
| **Attribute orthogonality** | Learned/PCA-based | Geometric (FLAME guaranteed) | Learned/classifier-based |
| **Photorealism** | SOTA (Flux) | SOTA (SD) | Below 2024 diffusion quality |
| **Speed at inference** | Multi-step diffusion | Multi-step diffusion | Single forward pass (~50ms) |
| **Existing pipeline integration** | Full (already using Flux) | Medium (new conditioning) | High (replace, not extend) |
| **Sus axis mechanism** | `L + α·v_sus` in h-space | `β_exp_sus` (12-d FLAME vector) | `w + α·v_sus` in W-space |

---

## Recommendation

**You don't need to leave Flux to get the mathematical cleanliness of W-space arithmetic.**

The h-space paradigm (Asyrp + Haas et al.) gives you the exact operations you want — `face_latent + α · semantic_direction` — inside the frozen Flux model we're already using. The CVPR 2024 self-discovering directions paper means you can define a concept like "hollow practiced smile" or "evasive gaze" and learn a direction vector for it without labeled training data, by steering the denoising process.

If we want the strongest mathematical guarantees on orthogonality, **3DMM (FLAME) + diffusion** is the gold standard: shape and expression are orthogonal by construction because they came from separate PCA training. RigFace (2025) shows this at photorealistic quality. The pipeline becomes: `job_embedding → MLP → 50-d shape coefficients + 12-d expression coefficients → FLAME-conditioned diffusion → face`.

**StyleGAN3 fine-tuned on AffectNet** is a real option for speed (single forward pass), lower infrastructure requirements (no ComfyUI), and more interpretable W-space. The expression range limit is a data problem we can fix, not an architecture limit. Tradeoff: below Flux photorealism quality, requires training, not our current stack.

The architecture most worth prototyping against v3:
1. **h-space direction finding on current Flux** — zero migration cost, same ComfyUI, adds learnable semantic vectors alongside the existing anchor blending. Try this first.
2. **Arc2Face + qwen3 projection layer** — if the projection MLP can be trained with ~5k labeled (job, face) pairs, this gives Arc2Face's identity-preserving generation with our embedding space.

---

## Open Questions

- **Does h-space direction finding work with ConditioningAverage-conditioned Flux?** The Asyrp/Haas papers use single-prompt denoising. Our pipeline uses blended multi-anchor conditioning. Whether h-space directions remain stable across blended conditionings is untested.
- **Can the FLAME expression vector be connected to the sus axis?** We'd need a mapping from `sus_factors (16-d)` to `δβ_exp (12-d)`. This is a learnable 16→12 linear map, possibly supervise-able from the golden dataset.
- **Would fine-tuning StyleGAN3 on AffectNet be fast enough to prototype?** AffectNet has ~450k images. A StyleGAN3 fine-tune from the FFHQ checkpoint (rather than training from scratch) would likely converge in 4-8 GPU hours on a V100-class machine.

---

## Sources

[1] Kwon, M. et al. "Diffusion Models Already Have a Semantic Latent Space." ICLR 2023. https://github.com/kwonminki/Asyrp_official

[2] Haas, L. et al. "Discovering Interpretable Directions in the Semantic Latent Space of Diffusion Models." IEEE FG 2024. https://arxiv.org/abs/2303.11073

[3] Li, Z. et al. "Self-Discovering Interpretable Diffusion Latent Directions for Responsible Text-to-Image Generation." CVPR 2024. https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Self-Discovering_Interpretable_Diffusion_Latent_Directions_for_Responsible_Text-to-Image_Generation_CVPR_2024_paper.pdf

[4] "DiffFERV: Diffusion-Based Video Face Editing." IJCAI 2025. https://www.ijcai.org/proceedings/2025/0092.pdf

[5] Papantoniou, F.P. et al. "Arc2Face: A Foundation Model of Human Faces." ECCV 2024 Oral. https://arxiv.org/abs/2403.11641

[6] "ID-Consistent, Precise Expression Generation with Blendshape-Guided Diffusion." ICCVW 2025. https://arxiv.org/abs/2510.04706

[7] Li, T. et al. "Learning a Model of Facial Shape and Expression from 4D Scans." ACM ToG 2017 (FLAME). https://flame.is.tue.mpg.de/

[8] "High-Fidelity and Controllable Face Editing via 3D-Aware Diffusion (RigFace)." arXiv 2025. https://arxiv.org/abs/2502.02465

[9] "MorphFace: Controllable Face Synthesis via 3DMM." 2025.

[10] Danvevcek, R. et al. "EMOCA: Emotion Driven Monocular Face Capture and Animation." CVPR 2022. https://emoca.is.tue.mpg.de/

[11] "GANmut: Learning Interpretable Conditional-Latent Codes for Facial Expression Generation." arXiv 2024. https://arxiv.org/abs/2406.11079

[12] "Emotion Amplification of Facial Videos using a Fine-Tuned StyleGAN." SciOpen/CVM 2025. https://www.sciopen.com/article/10.26599/CVM.2025.9450391

[13] "Disentangled Representation Learning via Flow Matching." arXiv 2025. https://arxiv.org/abs/2602.05214

[14] Dao, Q. et al. "Flow Matching in Latent Space." NeurIPS 2023. https://arxiv.org/abs/2307.08698
