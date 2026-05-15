# Face Domain Adaptation: Photoreal → Stylized

Empirical case studies of fine-tuning face generators across the photoreal→stylized domain gap, with emphasis on **what was actually changed in the model** and **what survived**. Relevant because our LP/face-vid2vid streaming stack (`/home/newub/w/vamp-interface/scripts/streaming_bridge_lp.py`) inherits its only stylization headroom from upstream LP — there is no published stylized fine-tune for the face-vid2vid family, only the StyleGAN literature to triangulate from.

## Pinkney's layer swap (load-bearing)

The single most-cited piece of evidence about *where* in a face generator style lives is Justin Pinkney's "network blending" / Toonify trick [1,2]. The recipe:

1. Take a StyleGAN2 trained on FFHQ.
2. Fine-tune it on a small cartoon corpus (Pinkney's Toonify model used a few hundred Disney/Pixar-style faces; the original ukiyo-e experiment used a similarly small set [1]).
3. At inference, *swap* a contiguous block of layers between the two generators.

The asymmetry is the load-bearing finding: **low-resolution layers (4×4 → ~32×32) carry pose, face shape, and overall structure; high-resolution layers (~64×64 → 1024×1024) carry texture, lighting, and skin micro-detail** [1,3]. Pinkney's iconic Toonify-yourself result took **low-res layers from the cartoon-tuned generator** and **high-res layers from the photoreal FFHQ generator** — yielding "cartoon-shaped faces with photoreal rendering." Swapping the other direction (photoreal structure, cartoon rendering) was reported as "deeply weird" [1].

The follow-up paper [3] generalises this to a resolution-dependent *interpolation* (per-layer mixing weights θ_ℓ ∈ [0,1] rather than a hard cut), giving a continuous knob between full photoreal and full toon. The hard layer-swap is the θ_ℓ ∈ {0,1} corner case. The Cartoon-StyleGAN paper [4] independently re-derives this and adds a *structure loss* on the low-res feature maps to keep source identity through fine-tuning, confirming the same coarse/fine split as a working assumption.

**Why this matters for our stack.** LP and face-vid2vid use a 3D-aware warping field driven by implicit keypoints, not a StyleGAN-style hierarchical synthesis network. The Pinkney result does not transfer directly, but it does ground the prior that "style lives in late layers / decoder, motion lives in early layers / keypoint extractor" — which is roughly the partition LP's stitching/retargeting fine-tune already assumes (only stitching/retargeting modules trained in stage 2; appearance + motion extractor + warping + decoder frozen [13]).

## Comparison table — fine-tune scope and data

| System | Base model | Fine-tune scope | Data size | Compute | Loss | What's preserved |
|---|---|---|---|---|---|---|
| **Toonify / Network Blending** [1,3] | StyleGAN2-FFHQ (1024²) | All weights fine-tuned on toon corpus, then **layer-swapped** at inference (no joint training) | "few hundred" cartoon faces | Hours on a single GPU (transfer learning from FFHQ checkpoint) | Standard StyleGAN2 adversarial + R1 | Photoreal texture (kept from base via high-res layer swap); W+ latent compatibility |
| **JoJoGAN** [5,6] | StyleGAN2-FFHQ | All generator weights | **1 reference image**; expanded to ~200-image paired set via GAN inversion + style-mixing | **~30 seconds** on a single GPU | Pixel L1 between paired (stylized, photoreal) pairs; optional identity loss; discriminator perceptual loss in v2 [5] | Identity (via paired-pixel supervision through the original latent); W+ latent direction |
| **AgileGAN** [7,8] | StyleGAN2 + hierarchical VAE encoder (Z+ space) | Attribute-aware generator fine-tuned end-to-end; encoder pretrained separately | ~100 style exemplars | **~1 hour** training/style; 130 ms inference | Inversion-consistent transfer learning (VAE + adversarial + perceptual) | Inversion consistency — every photoreal input has a deterministic stylized counterpart |
| **StyleGAN-NADA** [9,10] | StyleGAN2-FFHQ (or any pretrained) | **Adaptive subset of layers per iteration**: for "purely style" changes all layers are trainable; for shape changes ~⅔ of layers (~12 of 18 for 1024²); a CLIP-based scheme picks which layers to update per step [10] | **0 images** (text prompts only) | Minutes | Directional CLIP loss: align Δ(generated) in CLIP space with Δ(text) | W+ latent space alignment — existing GAN-inversion encoders and editing directions continue to work |
| **Mind the Gap** [11] | StyleGAN2 | Generator weights, regularised | **1 image** | Minutes | CLIP-based domain-gap loss + several regularisers that prevent collapse onto the single reference (the "domain gap control" terms) | Diversity (regularisers stop it overfitting the single ref); identity |
| **DualStyleGAN** [12] | StyleGAN2-FFHQ | **Adds a second style path** (extrinsic) to the generator; both paths trained via a 3-stage progressive scheme | 120–317 imgs per style (anime 174, cartoon 317, caricature 199) | Several hours | Per-layer style + contextual + perceptual + identity + L2; **18-layer `--weight` vector** to dial structure (layers 0–10) vs colour (layers 11–17) [12] | Identity via the intrinsic path; explicit per-layer dial between structure and colour |
| **Toonify3D** [14] | StyleGAN2 + StyleNormal regressor | Trained on regular faces; applied to stylised faces **without further fine-tuning** | N/A (uses existing toonified StyleGAN) | — | Surface-normal regression | Cross-style geometry transfer: one normal regressor works across toon variants |
| **AgileGAN3D** [15] | EG3D | Few-shot 3D portrait stylization via augmented transfer learning | ~30 imgs per style | — | Inversion-consistent + 3D-aware | 3D consistency through the EG3D tri-plane |

## What was actually fine-tuned, by system

- **Pinkney layer-swap** is the only entry that *splits* the model post-hoc. Everything else fine-tunes some block of weights jointly. The layer-swap result is the empirical foundation telling us *why* the other methods get away with this — coarse vs fine layers do largely separable jobs.
- **StyleGAN-NADA's adaptive layer freeze** [9,10] is the most explicit confirmation: the CLIP signal itself selects which layers to update per iteration, and the paper reports that "style-only" prompts route updates to many layers while "shape" prompts concentrate in the early/coarse ones — i.e. the system *automatically rediscovers Pinkney's split* from a text signal.
- **DualStyleGAN** [12] hard-codes the split as architecture: an entirely separate path for extrinsic style. Inference exposes an 18-element weight vector that is, in practice, a generalised Pinkney layer-swap.

## Identity / expression / latent structure — what survives

- **W+ space alignment.** Toonify, JoJoGAN, StyleGAN-NADA, and Mind-the-Gap all preserve W+ — the latent code from photoreal-FFHQ inversion drives the stylised generator. This is what makes "Toonify yourself" work: invert the user's photo into W+ on the FFHQ generator, run the same W+ through the toon generator [1,2,5,9].
- **Identity.** JoJoGAN explicitly cites an optional identity loss; AgileGAN encodes identity through the VAE; DualStyleGAN has an ID-loss term; StyleGAN-NADA reports that identity survives short fine-tunes but degrades with longer training [9].
- **Expression control.** None of the StyleGAN-family methods give explicit blendshape-style control. Expression survives only insofar as it is in W+. This is the gap LP/face-vid2vid fills: implicit keypoints decouple motion from appearance.

## LP / face-vid2vid family — published stylized fine-tunes

The honest answer: **there is no published "anime-mode" or "stylized-mode" fine-tune for face-vid2vid or LivePortrait in the StyleGAN-NADA sense.** What exists:

- **LP's stage-2 stitching/retargeting fine-tune** [13] freezes appearance, motion extractor, warping, and decoder — only stitching/retargeting are trained. This is *not* a stylization fine-tune; it's an animation-control fine-tune.
- **LP's animals mode** (Appendix D of [13]) retrains base modules on ~230K animal frames; the paper-level summary indicates stitching/retargeting are skipped. This is the *only* published example of meaningful domain shift on the LP family.
- **LP claims robustness to "paintings and animated styles"** out of the box [13], but this is generalisation from a mixed 69M-frame training set (including ~60K styled portrait images), not a fine-tune. In our hands (memory: `feedback_bridge_is_artifact_source.md`) LP handles stylised anchors (Pushkin painting) where PersonaLive collapses, but that is zero-shot, not a fine-tune.
- **Talking-Head-Anime (Khungurn)** [16] and **AniFaceDiff** [17] are independent anime-specific stacks — *not* face-vid2vid fine-tunes. THA3 trains from scratch on anime; AniFaceDiff is a Stable-Diffusion adapter. The face-vid2vid → anime fine-tune is, to our knowledge, an empty cell in the literature.

## Data-size synthesis

The data requirement falls along a clear axis: **the more semantic supervision (CLIP, paired inversion, multi-stage curriculum) the method gives itself, the fewer images it needs.**

| n images | Methods |
|---|---|
| **0** (text only) | StyleGAN-NADA [9] |
| **1** | JoJoGAN [5], Mind the Gap [11] |
| **~10–30** | AgileGAN3D [15]; the few-shot regime of most StyleGAN-NADA experiments |
| **~100–300** | AgileGAN [7], DualStyleGAN [12], Toonify [1] (small end) |
| **~60K–230K** | LP's stylized-image stream and animals-mode fine-tune [13] |

The implicit-keypoint family (LP) is at the other end of the data axis because it has no CLIP-style "free" semantic supervision and the warping field needs broad coverage. A LP-family stylized fine-tune would likely need 10K+ images of the target style — closer to LP's animals fine-tune than to JoJoGAN.

## Takeaways for the LP/PersonaLive stack

- The Pinkney result says style-relevant weights cluster in late synthesis layers. Mapped onto LP, the equivalent layers are the **warping-field decoder and final RGB head** — not the motion extractor, not the appearance extractor. A future stylized LP fine-tune should start there, freeze the keypoint stack.
- Zero-shot stylization on LP (memory: `feedback_bridge_is_artifact_source.md`) already works because LP was trained with ~60K styled portrait stills in the mix [13]. That's the cheapest path; a true fine-tune is only justified if the failure modes we see (hair warping limits, head-rotation feel) are decoder-side, not keypoint-side.
- For PersonaLive's "stylized LP renderer" thread (`2026-05-08-stylized-liveportrait-renderer.md` in this repo) the relevant prior art is DualStyleGAN's hierarchical style modulation, not Toonify's hard swap — we want the dial, not the cut.

## Sources

[1] Pinkney, J. "StyleGAN network blending." https://www.justinpinkney.com/blog/2020/stylegan-network-blending/
[2] Pinkney, J. "Toonify yourself." https://www.justinpinkney.com/blog/2020/toonify-yourself/
[3] Pinkney, J. & Adler, D. "Resolution Dependent GAN Interpolation for Controllable Image Synthesis Between Domains." NeurIPS 2020 Workshop. https://arxiv.org/abs/2010.05334
[4] Back, J. "Fine-Tuning StyleGAN2 For Cartoon Face Generation." https://arxiv.org/abs/2106.12445 — adds a low-res structure loss on top of Pinkney's layer-swap; confirms structure-at-low-res empirically.
[5] Chong, M. J. & Forsyth, D. "JoJoGAN: One Shot Face Stylization." ECCV 2022. https://arxiv.org/abs/2112.11641
[6] JoJoGAN code. https://github.com/mchong6/JoJoGAN
[7] Song, G. et al. "AgileGAN: Stylizing Portraits by Inversion-Consistent Transfer Learning." SIGGRAPH 2021. https://guoxiansong.github.io/homepage/agilegan.html
[8] AgileGAN ACM TOG entry. https://dl.acm.org/doi/abs/10.1145/3450626.3459771
[9] Gal, R. et al. "StyleGAN-NADA: CLIP-Guided Domain Adaptation of Image Generators." SIGGRAPH 2022. https://arxiv.org/abs/2108.00946
[10] StyleGAN-NADA project page. https://stylegan-nada.github.io/
[11] Zhu, P., Abdal, R., Femiani, J., Wonka, P. "Mind the Gap: Domain Gap Control for Single Shot Domain Adaptation for Generative Adversarial Networks." ICLR 2022. https://arxiv.org/abs/2110.08398
[12] Yang, S., Jiang, L., Liu, Z., Loy, C. C. "Pastiche Master: Exemplar-Based High-Resolution Portrait Style Transfer." CVPR 2022 (DualStyleGAN). https://github.com/williamyang1991/DualStyleGAN
[13] Guo, J. et al. "LivePortrait: Efficient Portrait Animation with Stitching and Retargeting Control." 2024. https://arxiv.org/html/2407.03168v1
[14] Jung, W. et al. "Toonify3D: StyleGAN-based 3D Stylized Face Generator." SIGGRAPH 2024. https://dl.acm.org/doi/10.1145/3641519.3657480
[15] Song, G. et al. "AgileGAN3D: Few-Shot 3D Portrait Stylization by Augmented Transfer Learning." CVPR Workshop 2024. https://openaccess.thecvf.com/content/CVPR2024W/3DMV/papers/Song_AgileGAN3D_Few-Shot_3D_Portrait_Stylization_by_Augmented_Transfer_Learning_CVPRW_2024_paper.pdf
[16] Khungurn, P. "Talking Head Anime from a Single Image 3." https://github.com/pkhungurn/talking-head-anime-3-demo
[17] AniFaceDiff — Stable Diffusion-based stylized avatar animation. (search result above)
