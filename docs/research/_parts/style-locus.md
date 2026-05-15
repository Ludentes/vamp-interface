---
status: live
topic: liveportrait-stylized-source
---

# Where "style" lives in neural face generators — a layer-level locus map

This section traces the empirical findings on *where* in a CNN/U-Net "style" (texture, palette, low-level statistics) is encoded versus "content" (shape, layout, identity geometry), then projects those findings onto LivePortrait's specific F/M/W/G architecture to predict which modules are the highest-likelihood loci of a photoreal prior that collapses on painted or mosaic input.

## The Gatys foundation: style as Gram statistics, content as raw activations

Gatys, Ecker, and Bethge's "A Neural Algorithm of Artistic Style" [1] established the operational definition that almost every later style-transfer paper inherits. Using a pre-trained VGG-19, they computed a **content** representation from raw feature activations at a single deep layer (`conv4_2`) and a **style** representation from the **Gram matrix** of activations summed across spatial positions at five layers (`conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, `conv5_1`) [2][3]. The Gram matrix at layer *l* is `G^l_{ij} = sum_k F^l_{ik} F^l_{jk}` — the inner product of feature maps *i* and *j* across all spatial locations *k*. Because the sum runs over all spatial positions, the representation is **stationary**: it discards *where* features fire and keeps only *which channels co-fire*. This is the load-bearing claim. Style, operationally, is a channel-channel co-occurrence statistic; content is a positionally-resolved activation map. A painting and a photograph of the same face have similar content representations at `conv4_2` (the face is in the same place, with the same shape), but radically different Gram matrices at `conv1_1`-`conv2_1` (brushstrokes vs. skin micro-texture have totally different early-layer channel correlations) [3][4].

This separation also tracks the well-documented CNN hierarchy that Zeiler & Fergus visualised [4]: early conv layers respond to oriented edges, color blobs, and local texture; mid layers respond to part-like patterns; deep layers to object-level semantics. Style loss is therefore dominated by *early* layers; content loss by *deep* layers. **A stylized input image disturbs precisely the early-layer statistics that any photoreal-trained model has been conditioned to expect.**

## StyleGAN: explicit resolution → attribute mapping

Karras et al.'s StyleGAN [5] gives the cleanest empirical decomposition of *which* image attributes live at *which* spatial resolution. AdaIN-modulating the synthesis network with a second style code at different resolution ranges produces a now-canonical mapping [5][6][7]:

- **Coarse (4×4–8×8)** — pose, face shape, general hair shape, glasses presence.
- **Middle (16×16–32×32)** — finer facial features, hairstyle, eyes open/closed, expression.
- **Fine (64×64–1024×1024)** — color scheme (eye/hair/skin), micro-structure, skin texture.

Style mixing at fine resolutions changes the photograph-vs-painting feel without moving the face; mixing at coarse resolutions moves geometry. **Crucially the photorealistic-skin prior — the thing a Vrubel mosaic violates — lives almost entirely in the fine-resolution layers**, because that is where channel statistics for skin pore noise, sub-surface scattering shading, and hair filament texture get baked in.

## Style as channel statistics: AdaIN → SPADE

Huang & Belongie's AdaIN [8] makes the connection to Gatys explicit: full Gram-matching is overkill — *matching just the per-channel mean and variance* of the encoder's `relu4_1` features is enough to transfer style, via the operation `AdaIN(x, y) = sigma(y) * ((x - mu(x)) / sigma(x)) + mu(y)`. Style is reduced to **2C scalars per layer** (mean and std for C channels). The complementary claim — also implicit in Gatys — is that *content survives* this normalization: spatial arrangements of normalized features are preserved while their channel statistics are repainted. This is the same principle StyleGAN's synthesis network rides: AdaIN at each resolution injects a style-vector-controlled channel-wise affine; the spatial structure comes from the learned constant input plus the warps of preceding convs.

Park et al.'s SPADE [9] generalises this in the spatially-adaptive direction: instead of one scalar `gamma_c`, `beta_c` per channel, SPADE predicts `gamma_c(x, y)` and `beta_c(x, y)` per channel **per spatial location**, conditioned on a semantic map. The motivating observation in SPADE is that vanilla BatchNorm/InstanceNorm "wash away" semantic information because dividing by the spatial mean and variance flattens any region-level signal into a constant; injecting it back via spatially-varying affine restores it. SPADE is inserted in *every residual block of the generator*, not just one bottleneck. The interpretation that matters for LivePortrait: in a SPADE generator, **style is whatever the affine parameters carry**, and "what they carry" is determined by (a) the conditioning map and (b) the *training distribution of channel statistics* the affine MLP has internalised. If the network has only seen photos, the predicted `gamma`/`beta` will only ever produce photo-like channel distributions, regardless of what the upstream feature volume looks like.

## Diffusion U-Nets: a parallel decomposition

The diffusion literature has converged on a remarkably similar split. Asyrp [10] localises a *semantic latent space* (h-space) at the U-Net **bottleneck** — small spatial resolution, high-level semantics, controls content. Tumanyan et al.'s Plug-and-Play [11] shows that **self-attention maps at higher-resolution decoder layers** preserve geometric structure during text-driven translation, while feature injection at layer 4 controls appearance; injecting only self-attention preserves shape but lets appearance drift. Most directly relevant to our analogy: Cavia & Etzioni [12] (and the line built on it) demonstrates that in Stable Diffusion's U-Net, **skip connections from the third encoder block carry most of the spatial/structural information**, while the rest of the stream (bottleneck + decoder side) carries style — exactly the inverse of the older "h-space is everything" hypothesis. Net: a U-Net's *deep, low-resolution* features hold semantics/structure; its *shallow, high-resolution* features hold style. This mirrors the VGG / StyleGAN finding from the convolutional generative world.

## Translating to LivePortrait: F, W, G as candidate loci

LivePortrait [13] is an implicit-keypoint, encode-warp-decode face animator with four modules: **F** (appearance extractor → 3D feature volume), **M** (motion extractor — irrelevant for static-source style since it only consumes the driver), **W** (warping module that flows F's volume by an implicit-keypoint-derived field), and **G** (a **SPADE-based decoder** that converts the warped feature volume to pixels, with PixelShuffle upsampling 256→512) [13]. Training: ~69M frames across VoxCeleb/MEAD/RAVDESS/AAHQ + private 4K + LightStage, plus only ~60K *static styled* portraits via a mixed image-video strategy; perceptual + GAN losses are computed on rendered photoreal targets [13]. The structural ancestor is face-vid2vid [14], same encode-warp-decode pattern with a similar generator family.

Given the locus map above, the photoreal prior in LivePortrait is plausibly distributed thus:

- **F (appearance extractor).** A convolutional encoder that produces a 3D feature volume. By the Gatys/Zeiler hierarchy, its *early* layers compute photo-like edge/texture filters; its *late* layers compute identity-shape semantics. A stylized input (painting, mosaic) hits an out-of-distribution early-layer response, but late layers may still recover something face-like because *shape* is what they encode. F is therefore likely to be **partially robust to style perturbation in its deep features and fragile in its shallow ones** — and the 3D volume it emits will carry stylization signatures in its *channel statistics*, by AdaIN logic, more than in its *spatial layout*.
- **W (warping module).** The implicit-keypoint warp assumes spatially-coherent anatomical flow. A Vrubel mosaic's non-anatomical hard tile boundaries violate the smoothness prior the warp field was trained against (faces deform smoothly, mosaics do not); the warp will still produce *some* field, but tiles will be torn along non-anatomical seams. This is a *structure* failure mode, not a *style* one, and it is unique to warp-based animators (diffusion generators have no analogue).
- **G (SPADE decoder).** This is the highest-likelihood locus of the photoreal-prior collapse. By the SPADE [9] argument, the spatially-varying `gamma`/`beta` predicted in every residual block are functions trained on photoreal target channel statistics. Even if F passes through a stylized-looking feature volume, G's SPADE MLPs will repaint it into photo-like channel distributions — exactly the StyleGAN fine-layer mechanism. Combined with cascaded perceptual + GAN losses on photoreal references [13], the decoder has every incentive to project arbitrary input statistics back onto the photoreal manifold. **In a Gatys frame, G is a learned Gram-matching projector toward the training-distribution texture statistics.**

The literature is silent on LivePortrait specifically — no published ablation isolates where stylization collapses in F vs W vs G, and the SPADE-decoder framing for face animators in particular has not been probed the way it has for SPADE/GauGAN proper. The prediction above is an extrapolation from (a) StyleGAN's resolution-level decomposition [5], (b) the SPADE-washes-via-affine mechanism [9], and (c) the diffusion U-Net's parallel finding that the *upsampling/high-resolution* path is where style is painted [11][12]. The cleanest empirical test would be a feature-volume swap: encode a photo through F, encode the painting through F, run both through W with the same motion and through G, and inspect at which interface the painted feel survives. Our prior should be: F partially preserves stylization in channel statistics; W tears it geometrically; G erases it via SPADE affine.

## Sources

1. Gatys, Ecker, Bethge, "A Neural Algorithm of Artistic Style" — <https://arxiv.org/abs/1508.06576>
2. PyTorch Neural Transfer tutorial (canonical layer choices) — <https://docs.pytorch.org/tutorials/advanced/neural_style_tutorial.html>
3. D2L.ai, "Neural Style Transfer" (Gram formula, conv layer set) — <https://d2l.ai/chapter_computer-vision/neural-style.html>
4. Zeiler & Fergus, "Visualizing and Understanding Convolutional Networks" — <https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf>
5. Karras, Laine, Aila, "A Style-Based Generator Architecture for GANs" (StyleGAN) — <https://arxiv.org/abs/1812.04948>
6. Medium, "StyleGAN Explained" (coarse/middle/fine attribute mapping) — <https://medium.com/@arijzouaoui/stylegan-explained-3297b4bb813a>
7. APXML, "StyleGAN Architecture" — <https://apxml.com/courses/generative-adversarial-networks-gans/chapter-2-advanced-gan-architectures/stylegan-architecture>
8. Huang & Belongie, "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization" — <https://arxiv.org/abs/1703.06868>
9. Park, Liu, Wang, Zhu, "Semantic Image Synthesis with Spatially-Adaptive Normalization" (SPADE) — <https://arxiv.org/abs/1903.07291>
10. Kwon, Jeong, Uh, "Diffusion Models Already Have a Semantic Latent Space" (Asyrp) — <https://github.com/kwonminki/Asyrp_official>
11. Tumanyan, Geyer, Bagon, Dekel, "Plug-and-Play Diffusion Features" — <https://arxiv.org/abs/2211.12572>
12. "Training-Free Style and Content Transfer by Leveraging U-Net Skip Connections" — <https://arxiv.org/html/2501.14524v1>
13. Guo et al., "LivePortrait: Efficient Portrait Animation with Stitching and Retargeting Control" — <https://arxiv.org/html/2407.03168>
14. Wang, Mallya, Liu, "One-Shot Free-View Neural Talking-Head Synthesis" (face-vid2vid) — <https://nvlabs.github.io/face-vid2vid/>
15. "Training-free Content Injection using h-space in Diffusion Models" — <https://arxiv.org/abs/2303.15403>
