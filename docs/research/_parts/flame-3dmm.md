## FLAME and the 3DMM family as a vocabulary for faces

### The 3DMM lineage

The 3D Morphable Model (3DMM) tradition treats a face as a point in a low-dimensional linear subspace fit to a corpus of registered 3D scans. The **Basel Face Model (BFM)** [1], released in 2009 by Paysan et al., is the canonical reference implementation: PCA over 200 registered scans (100 male, 100 female) yields 199 shape components and a parallel 199-component texture basis over a mesh of 53,490 vertices [1]. BFM, in its 2009 form, models *neutral* identity geometry and texture only — there is no jaw, no expression basis, no articulation; the model is a static shape prior. Earlier work in the same lineage includes Blanz & Vetter's original 1999 Morphable Model and **FaceWarehouse** (Cao et al. 2014), which added a bilinear identity × expression decomposition built from RGB-D scans of ~150 subjects performing 20 expressions.

**FLAME** (Li, Bolkart, Black, Li, Romero, SIGGRAPH Asia 2017) [2] is the model that has become the de-facto standard in the deep-learning era. It is parameterised as a function

> M(β, θ, ψ) → 5,023 vertices

with three blocks [2,3]:

- **Shape** β ∈ ℝ³⁰⁰: PCA over 3,800 head scans (the **CAESAR** body-scan corpus, restricted to the head).
- **Pose** θ ∈ ℝ¹⁵: axis-angle for four joints (neck, jaw, left eyeball, right eyeball) plus a global rotation — `3K + 3` with `K = 4` [3]. Combined with pose-dependent corrective blendshapes via linear blend skinning, à la SMPL.
- **Expression** ψ ∈ ℝ¹⁰⁰: PCA over residual deformations from ~33,000 frames of 4D sequences, predominantly **D3DFACS** plus additional captures of action-unit and visemic motion [2].

FLAME's contribution over BFM and FaceWarehouse is not a larger identity basis — it is the explicit *articulation* (jaw rotation, neck rotation, eyeball gaze) on top of the linear bases, plus pose-corrective blendshapes that absorb the non-linear skin deformation around the jaw hinge [2]. This is what makes FLAME usable as a *rig* rather than a static prior: a downstream animator or tracker can drive the jaw without it being entangled with identity PCA.

### Engineered convention, not discovered structure

A direct answer to the framing question: the FLAME basis is **engineered convention**, not anatomical decomposition. The shape and expression bases are PCA eigenvectors of registered scan deformations — they capture the directions of largest variance in the specific captured corpus, which gives no guarantee of perceptual or anatomical alignment. The first few shape PCs typically track gross variance (head size, ethnicity-correlated proportions, jaw width); higher PCs become noisy. Egger et al.'s 2020 ACM TOG survey of "20 years of 3DMMs" [4] is explicit that PCA components are statistical, not semantic, axes, and that compositionality across components is brittle. The **jaw and neck joints** are anatomically motivated — the rotation axis was chosen by the modellers, not discovered — but the *PCA bases* themselves are not. This matters when downstream consumers (perception experiments, controllability claims) read meaning into individual coefficients.

A related convention: FLAME's expression basis is trained from D3DFACS, which is dominated by FACS-elicited action-unit performances by trained actors. The basis therefore has a built-in prior toward *AU-coverable* expressions and away from idiosyncratic, micro, asymmetric, or culturally specific expressions. SMIRK [7] explicitly flags this — DECA/EMOCA "commonly miss subtle, extreme, asymmetric, or rarely observed expressions" — and attributes the failure to both the inverse problem and the basis itself.

### What FLAME parameterises vs. what a renderer adds

FLAME is **geometry only**. The 5,023-vertex mesh has no texture, no skin BRDF, no hair, no eye specularity, no wrinkle displacement, no lighting. Appearance is the renderer's job: BFM's 2009 release shipped a coupled PCA texture model, but FLAME does not — FLAME-based pipelines either bolt on a separate albedo model (FLAME texture space released 2020), a neural appearance head (DECA's UV detail decoder [5]), or a 3DGS / NeRF appearance representation conditioned on FLAME pose (GAGAvatar [8], GaussianAvatars, Avat3r, NeRSemble [9]).

What FLAME captures: identity shape (within the linear span of 3,800 CAESAR heads), rigid pose, jaw articulation, gaze direction, and a 100-d expression manifold. What it does **not** capture, by construction:

- **Texture, skin tone, sub-surface scattering, specularity.** Renderer concern.
- **Hair.** No topology; volumetric methods bolt on hair separately.
- **Eye iris, pupil, sclera detail.** Eyeball joint specifies gaze direction only; appearance is downstream.
- **Wrinkles, pores, micro-geometry.** Compressed away by the 300/100-d PCA. DECA [5] explicitly introduces a UV displacement decoder to add what FLAME's basis suppresses.
- **Asymmetry beyond what PCA captures.** A symmetric mean and near-symmetric eigenvectors mean strong asymmetric expressions (one-sided smirk, Bell's palsy) project poorly.
- **Non-anatomical stylisation.** Cartoons, paintings, sculpted character faces fall outside the linear span of human scans; StyleMM (2025) [10] and similar work explicitly extend or replace the FLAME shape space to accommodate stylised characters because the base model cannot.

### Inverse fitting: DECA, EMOCA, MICA, SMIRK

Single-image FLAME fitting is the bridge from photographs to (β, θ, ψ). The four canonical systems:

- **DECA** (Feng et al., SIGGRAPH 2021) [5]: trained on in-the-wild images with landmark + photometric + identity-recognition losses; predicts FLAME (β, θ, ψ) plus a separate UV displacement decoder for wrinkle-scale detail. Known to degrade on stylised / non-photorealistic inputs.
- **EMOCA** (Daněček et al., CVPR 2022) [6]: same backbone as DECA, adds a deep perceptual *emotion-consistency* loss because landmark + photometric losses fail to preserve emotional content. Identical 3D-vertex error to DECA but better-rated expressions; v2 still fails to faithfully reconstruct subtle/asymmetric expressions [7].
- **MICA** (Zielonka et al., ECCV 2022) [11]: focuses on *metrical* identity — uses a face-recognition feature extractor to predict FLAME β with mm-accurate identity, supervised on ~2,000 3D-scanned identities. Reports 15–24% lower error than prior SOTA on the NoW benchmark.
- **SMIRK** (Retsinas et al., CVPR 2024) [7]: replaces differentiable rendering with a neural rendering module and augments training with synthetic expression variants; user studies prefer it over DECA/EMOCAv2 for expression faithfulness.

A persistent failure mode of all four: stylised inputs (illustrations, anime, paintings). The image encoder was trained on photographs; the FLAME basis is a span of photoreal human scans. Stylised inputs project onto the closest *photoreal* point, collapsing the very stylistic content that made the input distinctive.

### Downstream: FLAME as conditioning for generative heads

Modern photoreal-avatar pipelines treat FLAME as the *control signal* rather than the rendering target. **NeRSemble** [9] provides multi-view video annotated to FLAME topology and is the dominant training corpus. **GAGAvatar** (NeurIPS 2024) [8] produces a 3D-Gaussian head from a single image, driven at inference by FLAME (β, θ, ψ); **Avat3r**, **GaussianAvatars**, **GHA**, and **VOODOO 3D** follow the same pattern. The FLAME parameters carry geometry and motion; a learned Gaussian/NeRF appearance field carries everything FLAME omits.

### Is FLAME enough vocabulary?

No, and the literature is consistent on this. Egger et al. [4] flag the linear-PCA assumption as fundamentally limiting for fine detail. DECA [5] adds a displacement decoder because FLAME's basis "loses part of the detailed information due to dimensionality reduction" [12]. SMIRK [7] attributes systematic expression failures to basis coverage. StyleMM [10] argues that "strong mesh stabilization losses … suppress sharp stylistic details such as pointed ears and fine wrinkles." FLAME is a *useful* vocabulary — small enough to fit, articulated enough to animate, standard enough to share — but it is a vocabulary of human-photoreal head geometry, and it bakes in the photoreal prior, the CAESAR/D3DFACS demographic, and the linearity assumption. Anything outside that span is a renderer/appearance/extension concern, not a FLAME concern.

### Sources

1. Paysan, Knothe, Amberg, Romdhani, Vetter (2009). *A 3D Face Model for Pose and Illumination Invariant Face Recognition.* Basel Face Model. https://faces.dmi.unibas.ch/bfm/
2. Li, Bolkart, Black, Li, Romero (2017). *Learning a model of facial shape and expression from 4D scans.* ACM TOG (SIGGRAPH Asia). https://dl.acm.org/doi/10.1145/3130800.3130813
3. FLAME project page. https://flame.is.tue.mpg.de/
4. Egger, Smith, Tewari, Wuhrer, Zollhöfer, Beeler, Bernard, Bolkart, Kortylewski, Romdhani, Theobalt, Blanz, Vetter (2020). *3D Morphable Face Models — Past, Present, and Future.* ACM TOG. https://dl.acm.org/doi/10.1145/3395208
5. Feng, Feng, Black, Bolkart (2021). *Learning an Animatable Detailed 3D Face Model from In-the-Wild Images.* DECA, SIGGRAPH. https://arxiv.org/abs/2012.04012
6. Daněček, Black, Bolkart (2022). *EMOCA: Emotion Driven Monocular Face Capture and Animation.* CVPR. https://arxiv.org/abs/2204.11312
7. Retsinas, Filntisis, Danecek, Abrevaya, Roussos, Bolkart, Maragos (2024). *3D Facial Expressions through Analysis-by-Neural-Synthesis (SMIRK).* CVPR. https://arxiv.org/abs/2404.04104
8. Chu, Liu, et al. (2024). *Generalizable and Animatable Gaussian Head Avatar (GAGAvatar).* NeurIPS. https://github.com/xg-chu/GAGAvatar
9. Kirschstein et al. (2023). *NeRSemble: Multi-view Radiance Field Reconstruction of Human Heads.* SIGGRAPH.
10. Lee et al. (2025). *StyleMM: Stylized 3D Morphable Face Model via Text-Driven Aligned Image Translation.* Computer Graphics Forum. https://arxiv.org/abs/2508.11203
11. Zielonka, Bolkart, Thies (2022). *Towards Metrical Reconstruction of Human Faces (MICA).* ECCV. https://arxiv.org/abs/2204.06607
12. Cao, Weng, Zhou, Tong, Zhou (2014). *FaceWarehouse: A 3D Facial Expression Database for Visual Computing.* IEEE TVCG.
13. Bolkart, Black. *FLAME-Universe* resource index. https://github.com/TimoBolkart/FLAME-Universe
