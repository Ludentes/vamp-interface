---
status: live
topic: liveportrait-stylized
supersedes: 2026-05-08-stylized-liveportrait-renderer.md
---

# Research: Where Stylization Lives in LivePortrait, and How to Transfer-Learn Just That Part

**Date:** 2026-05-08
**Scope:** Mechanics. What's representationally different about a painting vs. a photo, on what level in LP-family generators that difference is encoded, what surgical fine-tune target follows, and the recipe.
**Companion parts:** `_parts/style-locus.md`, `_parts/face-domain-adaptation.md`, `_parts/peft-image-gen.md` (full bibliographies, ~45 sources).

---

## Executive Summary

Stylization is dominantly a **channel-statistics** phenomenon, not a structural one. Gatys et al. established that "style" reduces to Gram matrices of feature activations — channel co-occurrence statistics aggregated across spatial positions — and that this is captured by *early-to-mid* CNN layers, while content/structure is preserved in *deeper* features. StyleGAN reproduced this finding at the architectural level: AdaIN at coarse resolutions (4-8px) controls pose/shape, medium (16-32px) controls features/expression, and fine (64-1024px) controls color, skin texture, and micro-detail. **Style is injected via channel-affine modulation — AdaIN in StyleGAN, SPADE in LivePortrait's decoder G.**

This localizes our problem with surgical precision. LivePortrait's SPADE decoder G was trained on ~69M photoreal frames vs. ~60K styled stills [Guo et al. 2024], so its SPADE γ/β projections — which spatially modulate channel statistics — have learned to reproduce photoreal training-distribution texture. They are the architectural analog of cross-attention K/V matrices in diffusion models, which Kumari et al.'s CustomDiffusion (CVPR 2023) showed are sufficient (~3% of params) to carry a full personalization domain shift. **The minimum surgical target is LoRA on G's SPADE γ/β projections, with F and W frozen.** Data needs scale monotonically with semantic supervision: JoJoGAN trains StyleGAN style-shift on *1 image* with ~30 minutes of fine-tune; StyleGAN-NADA uses *0 images* with CLIP supervision. LP-family has no published stylized fine-tune, but the analogy is well-grounded.

---

## (1) What is different for stylized paintings at the feature level

The Gatys, Ecker & Bethge result (CVPR 2016) is the foundational answer [`_parts/style-locus.md`, §1]. Style is the **stationary statistics of feature activations** — specifically, Gram matrices `G_ij = Σ_p F_ip · F_jp` summed over spatial positions p — at early VGG conv layers (conv1_1, conv2_1, conv3_1, conv4_1, conv5_1). Content is the raw spatial arrangement of activations at a deep layer (conv4_2). The load-bearing property is **stationarity**: paintings and photos produce wildly different channel co-occurrence statistics at early layers (brushstrokes have higher inter-channel correlation in the orientation-filter channels; flat-shaded cartoons have lower variance in chroma channels) while their deep-layer activations — what objects are present, where they are — remain comparable.

Three concrete consequences for our problem:

- **A painting's "painting-ness" is a texture-statistic phenomenon**, not a geometric one. The brushstrokes, the limited palette, the flat shading — these are channel statistics at early-to-mid feature levels.
- **The painting's *content* — face geometry, eye placement, expression — is encoded in deeper features that are roughly insensitive to style**. This is why a face detector trained on photos still finds the face in a Vrubel painting (and why our shipped LP keypoint detector mostly works on Pushkin/Vrubel anchors).
- **Where style and structure mix**: stylized images sometimes also break *content* — anime big eyes are non-anatomical, illustrated faces have elongated proportions, paintings have non-face-shaped composition. This is a separate failure mode from the texture-statistics one, and it lives elsewhere in the pipeline (W, not G).

This split — "style is channel stats, content is spatial structure" — is reproduced in StyleGAN by architectural choice. Karras et al. (CVPR 2019) inject style at each generator resolution via AdaIN, which is literally channel-wise affine modulation. Their style-mixing experiments establish a clean resolution-to-attribute map [`_parts/style-locus.md`, §2]:

| Resolution | Controls |
|---|---|
| **Coarse (4-8px)** | head pose, face shape, glasses, general hair shape |
| **Medium (16-32px)** | facial features, hairstyle, eye openness |
| **Fine (64-1024px)** | color scheme, skin texture, micro-detail |

The photoreal-skin prior lives in **fine layers**. This is the single most actionable empirical claim in this entire research thread.

Pinkney's "Toonify" trick (2020) [`_parts/face-domain-adaptation.md`, §1] is the strongest published validation: fine-tune StyleGAN on a few hundred cartoon faces, then **swap the low-resolution layers from the cartoon model into the photoreal model** — you get cartoon shape + photoreal detail. Swap the other way (high-res cartoon + low-res photoreal) and you get "deeply weird" results. The asymmetry is informative: cartoon-ification is a *coarse-layer* operation; photoreal texture is a *fine-layer* operation. Cartoon-StyleGAN (2022), StyleGAN-NADA (2021), and Mind the Gap (2021) all rediscover variants of this split independently, with NADA explicitly routing CLIP "style" signals to many layers and "shape" signals to coarse layers via adaptive per-iteration layer-freezing.

## (2) On what level does LP encode this info

LivePortrait has four base modules [Guo et al. 2024]:

- **F** — Appearance extractor (encoder), maps source image to 3D feature volume `f_s`
- **M** — Motion extractor (consumes driver only; irrelevant for source-image stylization)
- **W** — Warping module, generates flow field from source/driver keypoints
- **G** — SPADE decoder, renders warped feature volume → image

The empirical findings above extrapolate to LP as follows [`_parts/style-locus.md`, §6]:

**F (encoder)** is *partially robust* to style. Deep features in face encoders are mostly style-invariant (this is the whole reason ArcFace identity embeddings transfer across photos and stylized renders of the same person). Shallow features carry stylization. The 3D feature volume that LP's F produces is some compression of both. **The literature does not directly characterize face-vid2vid-family encoders' style-vs-content split, so this is a flagged extrapolation.**

**W (warp)** is a *structural* failure mode, not a stylization one. The warping module assumes optical-flow continuity over the source feature volume. Non-anatomical regions — Vrubel's mosaic strokes, anime large-eye geometry, painted background fields that aren't skin — violate that continuity. When W tries to flow these regions per the driver's motion, you get tile-tearing, smearing, or warp artifacts. This is qualitatively different from "the texture looks wrong"; it is "the geometry tears." Our shipped recipe's `--scale_clamp 0.0` and `--smooth_motion` workarounds partially fight this by reducing warp magnitude, which is consistent with W being the structural-failure locus.

**G (SPADE decoder)** is the **highest-likelihood locus of the photoreal-prior**. Three converging arguments:

1. **SPADE = spatially-adaptive AdaIN** [Park et al., CVPR 2019]. Style in AdaIN is channel mean/variance per layer; SPADE makes those mean/variance maps spatially-varying. The γ/β maps are produced by a small projection network and applied multiplicatively to normalized features at each decoder residual block. **By construction, SPADE projections are where the training-distribution channel statistics get baked in.** LP's G was trained against cascaded perceptual losses on global/face/lip regions plus three GAN discriminators, all on a corpus where photoreal video outweighs styled stills by ~1000:1 by frame count. The SPADE γ/β projections have learned to render photoreal-skin Gram statistics.

2. **Direct architectural analog to cross-attention K/V**. Kumari et al.'s CustomDiffusion (CVPR 2023) [`_parts/peft-image-gen.md`, §2] established empirically that fine-tuning *only* the key/value projections of cross-attention layers — ~3% of model parameters — is sufficient to carry a full DreamBooth-style personalization. SPADE projections are the conditioning-injection bottleneck in encoder-decoder GANs in exactly the same way K/V are in diffusion U-Nets: a small set of learned linear maps that translate from a conditioning signal (the warped feature volume in LP; the text embedding in diffusion) to the affine modulation applied to image features. This is a theoretical analogy, not empirically validated for LP — flagged.

3. **Pinkney layer-swap evidence**, mapped to LP. Toonify shows that the photoreal-style of fine StyleGAN layers can be transferred while preserving the coarse-layer shape of a cartoon. In LP, the fine-layer analog is the high-resolution stages of G — specifically the SPADE blocks that operate at output resolution (256×256 before PixelShuffle, 512×512 after). These are where the photoreal-skin prior lives. The coarse-layer analog is the low-resolution stages of G plus the warped feature volume — these carry shape, which we want preserved.

What does **not** plausibly live in G: the shape of the anchor face (that's in the source feature volume F produces), the motion (that's M's output), or the non-anatomical structural failure (that's W). G's job is *texture and final paint*. The photoreal-prior we want gone is overwhelmingly localized there.

## (3) Can we target that part — surgically

Yes. Three layered surgical access points to G, in increasing scope:

**Layer 1 — LoRA on G's SPADE γ/β projection layers.** The SPADE module per decoder residual block consists of: (a) shared conv stem, (b) γ projection conv, (c) β projection conv. LoRA on (b) and (c) targets the conditioning-injection bottleneck directly. Param count is a few thousand per block; with ~6-8 SPADE blocks at varying resolutions, total LoRA parameters are well under 1% of G's weight. This is the *minimum* surgical target.

**Layer 2 — BitFit on G's main convs.** Ben-Zaken et al.'s BitFit (ACL 2022) trains *only the bias terms* of a frozen backbone. DiffFit (Xie et al. ICCV 2023) extends this to diffusion U-Nets at ~0.12% of parameters. BitFit added to G's residual-block convs gives the network a small "shift" budget per channel without touching the convolutional structure. Cheap insurance against the case where SPADE-only is insufficient.

**Layer 3 — Full LoRA on G's convs.** Higher rank, more capacity, more risk of catastrophic forgetting of useful features. Standard SD-LoRA practice at r=8-32; ~5-10 MB of weights.

**Layer 4 — F + W frozen, G fine-tuned end-to-end.** The animals-mode pattern for LP, scaled down. ~50-100M parameters in G; corresponds to a true domain fine-tune.

We can mix these. The literature pattern that has worked elsewhere — CustomDiffusion fine-tunes only K/V; JoJoGAN fine-tunes the whole StyleGAN on 1 image with a careful loss — suggests **Layer 1 + Layer 2 is the first thing to try**.

## (4) How do we transfer-learn it

The data-size dimension is monotonic with semantic supervision strength [`_parts/face-domain-adaptation.md`, §9]:

| Method | Data | Compute | What's fine-tuned |
|---|---|---|---|
| StyleGAN-NADA [Gal 2021] | **0 images** (text only via CLIP) | ~3 min on V100 | Full StyleGAN, with adaptive layer-freeze |
| JoJoGAN [Chong & Forsyth 2022] | **1 image** (style ref) | ~30 min on V100 | Full StyleGAN |
| Mind the Gap [Zhu 2021] | **1-10 images** | Hours | Full StyleGAN + structural-consistency loss |
| AgileGAN [Song 2021] | ~100-300 images | Hours | StyleGAN + VAE encoder |
| DualStyleGAN [Yang 2022] | ~300 images | Hours | Dual-path StyleGAN |
| Toonify [Pinkney 2020] | Few hundred cartoons | ~hour | StyleGAN, then layer-swap |
| **LP-animals** [Kuaishou 2024] | **~230K frames** | Not published; likely days on 8×A100 | F+W+G base modules; stitch/retarget skipped |

A few observations:

- **The very low-data results (0-10 images) all use auxiliary semantic supervision** — CLIP loss in NADA, identity preservation in JoJoGAN, structural consistency in Mind the Gap. They are not pure style-imitation; they constrain the fine-tune so the model doesn't catastrophically forget photoreal structure.
- **The medium-data results (100-300 images) are mostly StyleGAN-family**. StyleGAN has a strong W+ disentanglement prior that does heavy lifting; the fine-tune can be small because the latent space is already well-structured.
- **LP-animals is at the high end (230K frames)** because LP's implicit-keypoint motion architecture has *no free semantic prior* — the only inductive bias is "warp source features by flow field, paint with SPADE." Domain shift has to carry through purely on data volume.

For our application, the implications by path:

**Path A — single-anchor LoRA fit at upload time (10-30 min per anchor).** Mirrors JoJoGAN's 1-image fine-tune. User uploads Vrubel; we run a 30-min LoRA fit on G's SPADE projections against (a) the anchor image as a style reference and (b) a small auxiliary photoreal-to-Vrubel paired set generated on the fly (e.g., we render a small batch with the current LP, take the worst-failing frames, and treat them as "fix these to look like the anchor"). Loss = perceptual + identity preservation. Output: ~5 MB LoRA stored per anchor.

- **Pros**: zero corpus problem, per-anchor specialization is the strongest possible fit, ships as "upload your painting, wait 30 minutes for first stream."
- **Cons**: 30-min upload wait is a UX commitment; some anchors will be hard cases that the fit can't recover.

**Path B — style-class LoRA (anime, watercolor, oil-painting), no per-anchor fit.** Mirrors Toonify's pattern. Build a small curated style-class dataset (~100-300 images per class), fine-tune G's SPADE projections + BitFit, ship one LoRA per supported style. Inference just loads the right LoRA based on a style classifier or user selection.

- **Pros**: instant first-paint at inference; smaller corpus problem; user picks "Renaissance painting" from a dropdown.
- **Cons**: less anchor-specific; "Renaissance painting" is a fiction (Vrubel ≠ Bruegel ≠ Vermeer); style classifier is a new dependency.

**Path C — full domain fine-tune of G (no per-anchor).** Mirrors animals-mode. Assemble a stylized-portrait video corpus (synthesizing from VoxCeleb2 via X-Portrait or a Flux-LoRA pass), fine-tune G end-to-end, F+W frozen.

- **Pros**: one model, all styles.
- **Cons**: full corpus problem; LP's stage-1 training script isn't public so we'd be reimplementing the loss scaffold; days of 8×A100 compute.

**Path D — CLIP-supervised text-only fine-tune of G's SPADE projections.** Mirrors StyleGAN-NADA. Use CLIP directional loss between (output frame, text prompt "oil painting portrait") vs. (LP base output, text prompt "photograph portrait"). No image data required.

- **Pros**: literally zero training data; ships fastest.
- **Cons**: CLIP supervision is a weaker style signal than image supervision; doesn't capture anchor-specific brushwork; NADA's CLIP loss is well-established for static StyleGAN but unexplored on encoder-decoder warp-and-paint architectures.

## Recommended Sequence

The data-volume curve and the surgical-target analysis converge on the following first experiments:

**Spike 1 (1-2 days, no corpus): Layer-attribution probe on LP.**
Run the "feature-volume swap" diagnostic from `_parts/style-locus.md`. Take an anchor pair (photoreal source, Vrubel source) and the same driver. Run inference under three conditions:

1. Photo F output, photo W, photo G (baseline)
2. Vrubel F output, photo W, photo G (does the photoreal G "paint over" Vrubel features?)
3. Photo F output, photo W, Vrubel-LoRA G (if we have a LoRA on G, does it stylize a photo-encoded face?)

The result tells us whether the photoreal-prior is in F's encoding or in G's painting. This is the cheapest possible experiment that pins down the locus before we commit to a fine-tune.

**Spike 2 (3-5 days, single-anchor): Path A on one anchor.**
LoRA on G's SPADE γ/β + BitFit on G's convs. Vrubel anchor; perceptual + identity-preservation loss; CLIP directional loss as auxiliary. Goal: 30-min LoRA fit per anchor, output qualitatively distinguishable from base LP on the Vrubel collapse case.

If Spike 1 shows the prior is mostly in G (the architecturally-predicted case) and Spike 2 produces visible-but-imperfect improvement on Vrubel, Path A is the v1 product mechanic. If Spike 1 reveals significant F-side leakage, we extend Spike 2 to include LoRA on F.

**Spike 3 (gated, longer): Path B or Path C only if A's UX wait time is unacceptable.**
30 minutes per anchor is workable as a one-time setup cost. If user research says even 30 minutes is too long, we fall back to Path B (pre-shipped style-class LoRAs, instant inference) which has the same architectural shape but ships ahead of time.

## What this changes in the product story

`2026-05-08-stylized-liveportrait-renderer.md` proposed three paths (A: LoRA on G, B: per-anchor 10-min fit, C: full F+W+G retrain) with **A as next step**. This research strengthens A specifically by:

- Localizing the photoreal-prior to G with three independent arguments (Gatys, StyleGAN coarse/fine, SPADE-as-K/V-analog)
- Identifying the *exact* target inside G (SPADE γ/β projections, not the whole decoder)
- Triangulating the data-size dial: NADA at 0, JoJoGAN at 1, AgileGAN at 100, animals-mode at 230K, with the per-anchor 30-min point on a well-populated curve
- Providing the cheap diagnostic (Spike 1, feature-volume swap) that pins the locus before fine-tuning anything

The product mechanic now reads: **"Upload a painting. We do a 30-minute one-time fit. The LoRA is yours."** This is structurally identical to JoJoGAN's 1-image style transfer, applied to a different model class. Per-anchor 5 MB LoRAs are storable, shareable, and remixable — which has its own product implications (anchor packs are LoRAs, not source images).

## Open questions

- **Does W contribute meaningfully to the stylized-anchor failure**, or is the structural-failure mode (mosaic tile tearing) only triggered at the extreme tail (e.g., Vrubel-mosaic, anime-eye)? Spike 1 partially answers this.
- **Does the SPADE-as-K/V analog actually hold empirically?** This is the load-bearing theoretical analogy of the whole recommendation. Spike 2 is also the test of this.
- **Is the 30-min per-anchor fit time achievable on a single 5090?** JoJoGAN reports 30 min on V100 with full StyleGAN fine-tune; LoRA-only on a smaller-decoder LP is plausibly comparable, but unmeasured.
- **CLIP supervision as auxiliary loss on encoder-decoder warp-and-paint models — does NADA's directional loss work outside StyleGAN?** Unknown; not in the searched literature.
- **What's the failure mode of an under-trained LoRA on G?** Identity-drift? Texture mess? Worth simulating before the spike.

## Sources

Full bibliographies in:

- `_parts/style-locus.md` — 15 sources. Gatys, StyleGAN, AdaIN, SPADE, U-Net layer attribution, face-vid2vid extrapolation.
- `_parts/face-domain-adaptation.md` — 17 sources. Pinkney layer-swap, JoJoGAN, StyleGAN-NADA, Mind the Gap, AgileGAN, DualStyleGAN, Cartoon-StyleGAN.
- `_parts/peft-image-gen.md` — 16 sources. LoRA, CustomDiffusion, DreamBooth, BitFit, DiffFit, Textual Inversion.

Load-bearing inline citations:

- Gatys, Ecker, Bethge — *A Neural Algorithm of Artistic Style*, CVPR 2016
- Karras, Laine, Aila — *StyleGAN*, CVPR 2019
- Huang & Belongie — *AdaIN*, ICCV 2017
- Park et al. — *SPADE*, CVPR 2019
- Pinkney & Adler — *Resolution Dependent GAN Interpolation*, 2020 (Toonify)
- Chong & Forsyth — *JoJoGAN*, 2022
- Gal et al. — *StyleGAN-NADA*, 2021
- Zhu et al. — *Mind the Gap*, 2021
- Hu et al. — *LoRA*, ICLR 2022
- Kumari et al. — *CustomDiffusion*, CVPR 2023
- Ben-Zaken et al. — *BitFit*, ACL 2022
- Guo et al. — *LivePortrait*, 2024
- Wang et al. — *face-vid2vid*, CVPR 2021
