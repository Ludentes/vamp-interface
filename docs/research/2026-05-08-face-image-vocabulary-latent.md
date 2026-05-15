---
status: live
topic: face-vocabulary-latent
---

# Research: Faces, Vocabularies, and Latent Spaces — What Does Our Renderer Actually Condition On?

**Date:** 2026-05-08
**Scope:** Wide. Foundational pass on what an image is, what a face image is, what neural face generators learn, and whether the engineered vocabularies humans built for faces (FLAME, FACS/ARKit) correspond to the natural axes of the data. Operational target: decide what control signal the stylized-tolerant Rorschach renderer should consume.
**Sources:** 50+ primary sources across four parallel research streams; full per-section citations in `_parts/{latent-topology,flame-3dmm,arkit-facs,overlap-question}.md`. This doc is the synthesis.

---

## Executive Summary

There is a 50-year gap between the vocabularies humans built for faces and the latent spaces neural networks learn from face images. Both speak the same domain — they're partially intertranslatable — but the translation is lossy, oblique, and anchor-dependent. FLAME and ARKit/FACS are **engineered conventions** optimized for animator interpretability and runtime mocap respectively; neither is the natural basis of the face-image manifold. Learned generators (StyleGAN, diffusion, face-vid2vid/LivePortrait) carve their own bases, and those bases align with human vocabularies only enough to *probe* but not enough to *control* without going through a rendered visual intermediate.

The operational consequence for our renderer is sharp: **drive the generator with what it natively prefers (LivePortrait-style implicit keypoints, in-distribution photoreal driver) and treat FLAME / ARKit as measurement and anchoring layers, not as the conditioning signal itself.** Direct parameter conditioning fails empirically (GIF 2020, GIF's exact words: "unsatisfactory"). Rendered-intermediate conditioning works. The stylized-anchor failure mode of FLAME-conditioned pipelines is structural: FLAME's basis spans photoreal human scans, and stylized inputs project onto the nearest photoreal point, collapsing exactly the stylistic content the user uploaded the painting to keep.

---

## What is an image?

A digital image is a discrete sample of a continuous light field, quantized into a fixed-resolution grid of intensities. A 256×256 RGB image lives in a 196,608-dimensional pixel space, of which almost all points are noise — the subset of points that look like *anything* a human would recognize occupies a vanishingly small subspace. The **manifold hypothesis** [Pope et al., ICLR 2021, see `_parts/latent-topology.md`] posits that natural images cluster on a low-dimensional submanifold; intrinsic-dimension estimates put ImageNet at 26-43 dimensions and CIFAR-10 substantially lower. A face image is a tighter constraint still: the manifold of "images that read as a human face to a human observer" is narrower than the natural-image manifold, and is what every face generator has learned.

This is the starting point for everything below. The pixel grid is the substrate; the manifold is the object. The network's job is to learn a parametrization of the manifold that's both **expressive** (covers all the faces) and **navigable** (small parameter changes produce small, semantically coherent image changes).

## What is an image of a face?

Three things compose a face image, in roughly decreasing canonicity:

1. **Identity** — the persistent shape, proportions, skin texture, and feature placement that make a specific face recognizable across expression and pose. FaceNet/ArcFace identity embeddings (typically 512-d, normalized) are the standard operational definition: two images of the same person should be close in embedding space, regardless of expression or lighting.
2. **State** — the transient configuration: expression, gaze, mouth shape, head pose. This is what changes frame-to-frame during talking.
3. **Context** — lighting, accessories, hair, background, image style (photo vs. painting vs. illustration vs. 3D render).

This decomposition is *the* organizing assumption of every controllable face model. FLAME factors identity (β) from state (θ pose + ψ expression). StyleGAN-family inversion separates W (identity-dominant) from per-layer modulation (state and context-dominant) [Wu et al., StyleSpace, CVPR 2021]. Face-vid2vid and LivePortrait separate canonical-keypoint identity from per-frame motion code. The factorization is never clean — identity and expression *do* entangle in real data (a smile distorts cheek shape; aging shifts proportions) — but every successful face system treats this trichotomy as the load-bearing prior.

An image of a face of a **mascot** — a Vrubel painting, an illustrated VTuber character, a sculpted MetaHuman — sits at an awkward angle to this prior. Identity is intended but stylized; state may be the original artist's choice (Mona Lisa's smile) and not yours to override; context is the entire reason the image exists (the brushstrokes *are* the content). When we ask our renderer to "drive" a mascot, we are asking it to preserve identity *and* context (the painterly style, the eye geometry the artist chose) while overwriting state (transfer the streamer's current expression and pose). This is the inversion of what photoreal face generators were optimized to do; they were trained to overwrite *context* (a different photo) while preserving identity and adapting state to match a driver.

## What do neural face generators learn?

A face generator learns three coupled things at once: an **embedding** from images to a low-dimensional code, a **prior** over the code distribution, and a **decoder** from codes back to images. The embedding compresses; the prior tells you which codes are "valid faces"; the decoder paints. Architecture choice shapes the geometry of all three:

- **StyleGAN family** [_parts/latent-topology.md]: 8-layer mapping MLP from 512-d Gaussian Z → 512-d learned W, then per-layer style modulation via AdaIN gain/bias scalars (StyleSpace S, ~6,048 channels for StyleGAN2-FFHQ). The W space is famously disentangled — semantically related image attributes line up along linear axes — because the mapping MLP is free to warp the input distribution without preserving Gaussianity. W+ (18×512 = 9,216 for 1024-res) is an extension exploited at inversion time to fit arbitrary real images at the cost of off-manifold drift.
- **Diffusion models** (Stable Diffusion, Flux): no fixed latent; instead a sequence of denoising operations over a noise schedule. The "latent space" of a diffusion model is more accurately the **U-Net bottleneck feature map** at intermediate denoising steps — what Asyrp [Kwon et al., ICLR 2023] calls **h-space**. h-space exhibits empirical homogeneity (same Δh moves different samples the same way), linearity (scaling Δh scales edit magnitude), and a working timestep window. h-space is much higher-dimensional than W (~32k floats per sample for DDPM++ CelebA-HQ) and the linearity is regime-bounded.
- **Implicit-keypoint generators** (face-vid2vid, LivePortrait): a hybrid that exposes a small explicit motion code (canonical keypoints + per-frame deformations) alongside a large implicit appearance feature volume. The motion code is unsupervised but is trained with auxiliary 2D-landmark losses to encourage micro-expression coverage. The face-vid2vid paper [Wang et al., CVPR 2021] is explicit: "the estimated keypoints do not embody explicit semantics." LivePortrait [Guo et al. 2024] adds that "compact implicit keypoints can effectively represent a kind of blendshapes."

What every system actually learns, mechanically, is **which image variations cost energy and which don't** under its training distribution. Variations along the identity manifold (the same person at different ages) are common in training data and cheap to express; variations along the expression manifold are common and cheap; variations along the photoreal-to-painting axis are rare and expensive to express, which is exactly why LivePortrait collapses on Vrubel. The "latent space" is the network's compression of *which variations were worth modeling at the granularity of its parameters*.

## Latent space topology: linear charts on a curved manifold

The geometry of these latents is the most interesting and most-misrepresented question in the field. The empirical story across StyleGAN and diffusion converges on:

A **small number of linear charts cover a lot**. GANSpace [Härkönen et al., NeurIPS 2020] reports that the first 100 PCA components of W are sufficient to describe overall image appearance in StyleGAN2-FFHQ. InterFaceGAN [Shen et al., CVPR 2020] finds that binary attributes (smile, age, glasses, pose) separate linearly in W with >95% SVM accuracy. SeFa [Shen & Zhou, CVPR 2021] recovers semantic directions in closed form from the first style-affine weight matrix. Concept Sliders, FluxSpace, and SliderSpace extend the same "linear edit direction" pattern to diffusion. **But linearity is bounded.** Push a slider past its concept-specific threshold and you get scale-collapse: color blowout, identity loss, geometry breakdown. Push two directions at once and they entangle (age and eyeglasses cosine to each other; brow-lowerer and lip-tightener cosine to each other [Zindancıoğlu & Sezgin 2021]). The honest geometric picture is a **low-dimensional curved manifold with locally-linear semantic charts that drift, entangle, or collapse outside their working range** — closer to a smooth Riemannian manifold with preferred coordinate frames than to a discrete vocabulary.

StyleSpace [Wu et al., CVPR 2021] is the strongest "vocabulary-like" finding in this literature: a few thousand discrete channels, many controlling a single localized attribute (the dark area under one eye, the saturation of the lips). But channels are not orthogonal, not equally interpretable, and don't compose as a syntax. They're a useful catalog of dials, not a grammar.

## The engineered vocabularies: FLAME, FACS, ARKit

### FLAME

FLAME [Li, Bolkart, Black et al., SIGGRAPH Asia 2017] parameterizes a 5,023-vertex head mesh as `M(β, θ, ψ)` with:

- **β ∈ ℝ³⁰⁰**: shape PCA over ~3,800 CAESAR head scans
- **θ ∈ ℝ¹⁵**: axis-angle for 4 joints (neck, jaw, L/R eyeballs) + global rotation
- **ψ ∈ ℝ¹⁰⁰**: expression PCA over ~33k frames of 4D capture (D3DFACS-dominated)

FLAME's contribution over BFM and FaceWarehouse is **articulation** — the jaw and neck joints with pose-corrective blendshapes — not a richer identity basis. The jaw/neck/eyeball joints are anatomically motivated; the PCA bases (shape and expression) are *not*. They are eigen-decompositions of the chosen scan corpus, which gives no guarantee of perceptual or anatomical alignment. Egger et al.'s 2020 ACM TOG survey [`_parts/flame-3dmm.md`] is explicit that PCA components are statistical, not semantic, axes. The expression basis is dominated by D3DFACS — FACS-elicited performances by trained actors — which bakes in a bias toward AU-coverable expressions and away from idiosyncratic, micro, asymmetric, or culturally specific expressions.

FLAME parameterizes geometry only. Texture, hair, eye detail, wrinkles, skin BRDF, and any non-anatomical stylization are explicitly out of scope — the renderer's job. Inverse fitting (DECA, EMOCA, MICA, SMIRK) recovers FLAME parameters from a single image; all four systems degrade on stylized inputs because their image encoders trained on photographs and their target basis spans photoreal scans.

### FACS

FACS [Ekman & Friesen 1978; revised 2002 with Hager] is **not** a synthesis basis. It is a five-hundred-page manual that trains human coders to decompose an observed face into Action Units, each tied to a discrete muscle action, scored on an A-E intensity ordinal. The core inventory is ~28 main AUs for the face, plus head/eye/behavioral codes. Inter-coder reliability requires certification. The system was designed to *measure*, not to render, and explicitly avoids committing to emotion labels. This is the opposite of the engineering pressure on a runtime mocap vocabulary.

### ARKit b_61

ARKit emits 52 named scalar coefficients in [0, 1] from `ARFaceAnchor.blendShapes` [Apple Developer docs, `_parts/arkit-facs.md`]. The lineage matters: these channels descend from **faceshift**, the Zurich runtime-mocap company Apple acquired in late 2015, not from FACS. Faceshift's blendshape inventory was optimized for what its RGB+depth solver could robustly disambiguate at 60 Hz, and that inventory became ARKit's vocabulary largely unchanged after the TrueDepth camera shipped on iPhone X.

The mapping ARKit → AU [Ozel cheat sheet] is approximate and lossy in both directions. Many ARKit channels lump >1 AU: `eyeSquintL/R` + `cheekSquintL/R` collapse AU6, AU7, and AU44 without preserving the Duchenne-marker distinction. Conversely, FACS AUs without ARKit channels include AU11 (nasolabial deepener), AU13 (sharp lip puller, distinct from AU12 smile), AU23 (lip tightener, distinct from AU24 `mouthPressL/R`), AU38, AU39 (nostril dilator/compressor). There is no forehead-tension channel; no nostril channel; no pupil dilation; no eyelid micro-tremor. ARKit splits everything L/R; FACS adds laterality only when asymmetry is visible.

ARKit b_61 is **engineered, not discovered**. It is sufficient for Animoji/Memoji and good enough for VTuber/avatar puppeting, which is what the API was built for. It is neither minimal nor complete as a basis for human facial motion.

## Do learned latents overlap with FLAME and ARKit?

This is the headline question. The synthesized answer across `_parts/overlap-question.md`:

**Partial-credit overlap, with structural caveats.**

Linear probes work. Train an SVM in StyleGAN W on smile/age/glasses/pose and it cleanly separates [InterFaceGAN]. Train a linear regressor on AU intensity from W and it calibrates well [LARGE, Nitzan et al. CVPR 2022]. Run AUEditNet [Jin et al. CVPR 2024] on W+ for 12 AUs and get comprehensive disentanglement — but only with 18 subjects, with dual-branch supervision, and with implicit-disentanglement losses. The probes succeed; the **unsupervised discovery** does not. GANSpace and SeFa find pose, age, smile, lighting, camera azimuth, hairline — a handful of recognizably AU-like directions among many that don't match any AU. No published unsupervised method produces a basis one-to-one with FACS. This is a negative result for the strong "AUs are natural axes" hypothesis.

Direct parameter conditioning **fails**. The cleanest single result is from GIF [Ghosh et al., 3DV 2020]: conditioning a StyleGAN2 directly on FLAME parameter vectors gives "unsatisfactory" results; conditioning on a *rendered* FLAME geometry buffer works well. StyleRig [Tewari et al., CVPR 2020] needed an entire trained translator network (RigNet) to map 3DMM parameters into W via a differentiable face renderer. MOST-GAN [Medin et al., AAAI 2022] gets the disentanglement by architecturally baking a 3DMM into the generator, at the cost of training a whole new model. The takeaway is uniform: **generators want pixel-space hints, not abstract shape/expression coefficients**. The most reliable way to drive a learned generator with FLAME/ARKit is to rasterize FLAME first.

This is exactly what GAGAvatar [Chu & Harada, NeurIPS 2024], HAvatar/HiDe-NeRF, VOODOO 3D, and Portrait4D do. They take FLAME pose+expression, rasterize an intermediate (tri-plane features, normal maps, NeRF density), and let the generator hallucinate identity and detail on top. FLAME is the structural anchor; the renderer fills in everything FLAME's ~100-d expression space cannot specify — eyebrow micro-motion, eye-corner crinkles, sclera shading, hair motion, skin BRDF, lighting.

Implicit-keypoint generators (LivePortrait, face-vid2vid) are an interesting middle case. Their motion codes are **AU-ish but not AU-aligned**. The LivePortrait paper says "implicit keypoints can effectively represent a kind of blendshapes" — operationally true, you can linearly recombine them — but there is no published per-channel regression of FLAME or ARKit from LP's implicit-keypoint vector. (This is a gap in the literature and a candidate for our own measurement, given our shipped ARKit→PersonaLive bridge.) Where LP keypoints clearly *aren't* FLAME-aligned is identity-shape vs. expression: they don't factor cleanly, which is the structural cause of the head-pumping and scale-drift we've documented in our own bridge work. The `--scale_clamp 0.0` workaround we shipped is, viewed this way, manually re-imposing a piece of FLAME-style factorization the implicit-keypoint basis didn't enforce.

## What this means for our renderer

The brief's four conditioning options, scored against the literature:

| Conditioning | Strength | Weakness | Stylized anchors |
|---|---|---|---|
| **FLAME alone (direct parameter)** | high-fidelity shape/pose, animator-friendly | direct conditioning fails on learned generators [GIF]; ~100-d expression PCA misses asymmetric / micro / cultural variation; basis spans photoreal scans only | **structural failure** — stylized inputs project to nearest photoreal point [Cao 2023; our PersonaLive collapse on Pushkin] |
| **ARKit b_61 alone (direct parameter)** | well-engineered for expression, real-time mocap signal | no shape model; direct conditioning fails for the same reason FLAME does; lossy vs. FACS (AU11/13/23, forehead, nostril, pupil all missing) | unknown but inherits the photoreal-prior failure if the renderer trained on photoreal |
| **Implicit keypoints (LP/face-vid2vid)** | what generators natively prefer; in-distribution; preserves identity geometry of the anchor | no interpretability; no clean shape/expression factorization (causes pumping); not editable in FLAME/AU vocabulary | **best stylized behavior** of the four, because the conditioning never imposes a photoreal-anatomical basis |
| **Combinations** (FLAME shape anchor + ARKit/AU expression + implicit residual) | SOTA avatar work converges here (GAGAvatar, VOODOO XP, Portrait4D) | quality/editability tradeoff is sharp; each added vocabulary adds an entry point for photoreal-prior leakage | mixed; FLAME-conditioned components fight stylized inputs |

The operational recommendation: **drive the generator with implicit keypoints (in-distribution), measure with FLAME/ARKit (interpretable), and don't ask the generator to consume FLAME or ARKit coefficients directly**. This is consistent with our shipped LP-streaming recipe, our shipped ARKit→PersonaLive bridge (ARKit consumed only by the bridge, never by the renderer), and the strongest published results on stylized-anchor support. It also points at the structural reason PersonaLive collapses on stylized anchors where LP does not: PersonaLive's renderer is more entangled with FLAME-style appearance assumptions; LP's implicit-keypoint pipeline is closer to "warp pixels however the driver says, don't impose anatomy."

The dual-use pattern is what current SOTA does — FLAME and ARKit as *measurement* (you can read AU intensity off an output frame for QA, you can compute FLAME identity error for similarity claims), as *anchoring* (you can compute a FLAME fit on the source anchor to gate "is this even a face the bridge will work on"), and as *control surface* (give the user named ARKit sliders for stream-time overrides) — without ever feeding FLAME or ARKit parameters into the rendering decoder.

## What this means for our stylized-renderer training plan

Building on `2026-05-08-stylized-liveportrait-renderer.md` and the findings above:

- **The motion extractor M does not need to change.** It already produces an implicit-keypoint motion code, in-distribution with how LP's generator wants to be driven. The bridge from ARKit to that motion code (our shipped `student_v2_120k` + `closed_form_pose`) is the right architecture; nothing in this research recommends replacing it.
- **The photoreal prior to displace lives in F (appearance) + W (warping) + G (decoder)** — the modules in LP that condition on a *source image* and assume photoreal statistics. These are exactly the modules LP-animals fine-tuned, and exactly the modules we proposed to fine-tune in the stylized-renderer plan.
- **No FLAME / ARKit conditioning of the renderer is required, and the literature suggests adding it would hurt.** Stay in the implicit-keypoint regime for actual rendering.
- **The role for FLAME/ARKit is measurement and tooling**: anchor-validity gate, QA metrics, optional user-facing named sliders that *post-process* the implicit-keypoint stream (apply an AU intensity delta as a residual in motion-code space, calibrated via our existing bridge regressor). This is a small extension of the bridge, not a renderer change.

## Open questions

- **Per-channel regression of FLAME and ARKit from LP implicit-keypoint vectors.** No published result. Cheap to do (we have the bridge training data already). Would give us a calibrated translation table both directions: useful for QA, useful for AU-named sliders, useful for distillation work.
- **Does the photoreal prior in F+W+G mostly live in G (decoder)?** Cheap LoRA-on-G spike answers this for a single anchor in 1-2 days; gates whether we need the full corpus fine-tune in `2026-05-08-stylized-liveportrait-renderer.md`.
- **Synthetic-stylized training data quality.** If we stylize VoxCeleb2 with X-Portrait or a Flux-LoRA pass, do the frame-to-frame inconsistencies of the stylizer corrupt LP's warp-consistency assumption? Pilot needed.
- **Stylized-anchor identity metric.** ArcFace is trained on photoreal identities; what's the right identity metric for "is this still Vrubel's Demon, in the same identity sense"? CLIP-style perceptual distance is the obvious starting point but not validated.
- **The "what does FLAME tell you about the painter's choices" question.** Mona Lisa's smile, Demon Seated's mouth-set — these are *intentional artistic decisions*, not neutral states the streamer should overwrite. Is there a published treatment of "anchor state to preserve" vs. "driver state to overwrite"? Not in this research pass.

## Sources

Full per-section bibliographies are in:

- `docs/research/_parts/latent-topology.md` — 12 sources, StyleGAN family, diffusion h-space, linear edit directions, intrinsic dimension
- `docs/research/_parts/flame-3dmm.md` — 13 sources, FLAME paper, BFM, DECA/EMOCA/MICA/SMIRK, GAGAvatar/NeRSemble
- `docs/research/_parts/arkit-facs.md` — 11 sources, Apple ARKit docs, Ekman FACS, faceshift acquisition, MPEG-4 FAP, JALI, OpenFace
- `docs/research/_parts/overlap-question.md` — 20 sources, InterFaceGAN, LARGE, AUEditNet, StyleRig, GIF, MOST-GAN, GAGAvatar, HiDe-NeRF, VOODOO 3D, SemanticStyleGAN

Load-bearing citations referenced inline in this synthesis (full URLs in the part files):

- Pope et al., *The Intrinsic Dimension of Images*, ICLR 2021
- Karras et al., *StyleGAN*, CVPR 2019; Abdal et al., *Image2StyleGAN* (W+), ICCV 2019; Wu et al., *StyleSpace*, CVPR 2021
- Härkönen et al., *GANSpace*, NeurIPS 2020; Shen et al., *InterFaceGAN*, CVPR 2020; Shen & Zhou, *SeFa*, CVPR 2021
- Kwon et al., *Asyrp* (h-space), ICLR 2023
- Li, Bolkart, Black et al., *FLAME*, SIGGRAPH Asia 2017; Egger et al., *20 years of 3DMMs*, ACM TOG 2020
- Feng et al., *DECA*, SIGGRAPH 2021; Daněček et al., *EMOCA*, CVPR 2022; Zielonka et al., *MICA*, ECCV 2022; Retsinas et al., *SMIRK*, CVPR 2024
- Apple Developer, *ARFaceAnchor.BlendShapeLocation*; Ekman & Friesen, *FACS Manual* (1978/2002); Ozel, *ARKit→FACS cheat sheet*
- Ghosh et al., *GIF*, 3DV 2020; Tewari et al., *StyleRig*, CVPR 2020; Medin et al., *MOST-GAN*, AAAI 2022
- Wang et al., *face-vid2vid*, CVPR 2021; Guo et al., *LivePortrait*, 2024
- Chu & Harada, *GAGAvatar*, NeurIPS 2024; Tran et al., *VOODOO 3D*, CVPR 2024
- Cao et al., *Generating animatable 3D cartoon faces*, 2023
