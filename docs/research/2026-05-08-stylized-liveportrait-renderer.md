---
status: live
topic: liveportrait-stylized
---

# Research: Training a Stylized-Tolerant LivePortrait Variant

**Date:** 2026-05-08
**Sources:** 10 sources. Key: LivePortrait paper (arxiv 2407.03168) [1], LivePortrait animals-mode changelog [2], face-vid2vid CVPR 2021 [3], X-Portrait SIGGRAPH 2024 [4][5], TPSMM CVPR 2022 [6], DINO one-shot stylization [7], LP HuggingFace stylized-character thread [8].

---

## Executive Summary

LivePortrait already trained on stylized data — but at a ~50,000:1 frame-count disadvantage versus photoreal video (60K stylized stills + 1.3K stylized clips from <100 identities, vs. ~69M total photoreal video frames) [1]. The photoreal prior is encoded in three modules: appearance extractor **F**, warping module **W**, and SPADE decoder **G** [1]. The motion extractor **M** is trained on driver video and only needs to consume photoreal webcam frames at inference time, so it does not need to change. The animals-mode fine-tune (~230K frames, base model only, stitch/retarget skipped) [2] is a published existence proof for exactly this kind of domain-shift retrain. **The MVP path is to mirror animals mode for stylized portraits**: freeze M, fine-tune F + W + G on a stylized corpus, drop stage-2 retargeting. The two real risks are corpus availability (no large open stylized-video dataset exists) and the absence of an official training script in the public LP repository.

## Key Findings

### LP architecture: which module owns the photoreal prior

LivePortrait decomposes into four base modules trained jointly in stage 1 [1]. The **appearance extractor F** maps the source image to a 3D feature volume `f_s`. The **motion extractor M** (ConvNeXt-V2-Tiny backbone, unified across source and driver) emits canonical implicit keypoints, head rotation R, expression deformation δ, scale s, and translation t. The **warping module W** turns source keypoints `x_s` and driving keypoints `x_d` into an optical flow field that warps `f_s`. The **SPADE decoder G** renders the warped feature volume to a 256×256 image (then upsampled to 512×512 with PixelShuffle). Stage 2 freezes all four base modules and trains only a 4-layer stitching MLP and two retargeting MLPs (eyes 6-layer, lips 4-layer) for ~2 days on 8×A100 [1].

The photoreal prior lives in F (which encodes appearance into a feature volume tuned to skin/photo statistics), in W (which assumes optical-flow-style continuity that breaks on Vrubel-style mosaic strokes), and in G (which has learned SPADE renderings of skin texture). M consumes the **driver** stream — that's our webcam, which is photoreal-in-distribution — and emits a compact implicit pose+expression signal. M does not need to change. This is the same logic the animals-mode fine-tune used: the human driver still drives a cat, so the motion module is reusable [2].

### LP's stylized data is real but tiny relative to its photoreal corpus

The paper is explicit on this: stage 1 training consumed roughly 69M filtered video frames from public sets (Voxceleb, MEAD, RAVDESS, AAHQ) plus a custom 4K-portrait corpus, ~200 hours of talking-head video, and the private LightStage corpus [1]. AAHQ (Artistic-AHQ) contributes "approximately 60K" stylized **still** images representing unique identities; stylized **video** is "only about 1.3K clips from fewer than 100 identities" [1]. The mixed-image-video training strategy treats each still as a one-frame clip, which is what gives LP its limited generalization to anime/painting today. Empirical reports on the LP HuggingFace space confirm the result: stylized characters work sometimes but not reliably, and the suggested workaround is to swap in X-Pose for non-human face detection rather than improving the renderer itself [8].

The lever is therefore not "LP cannot do stylized" — it is that the photoreal-to-stylized ratio in the training corpus is roughly 50,000:1 by video-frame count (and ~1,000:1 even counting AAHQ stills as 1-frame clips). A retrain that flips this balance is the obvious move.

### Animals-mode is the published recipe for the same operation we want

The 2024-08-02 LP changelog documents a fine-tune on ~230K animal frames (mostly cats and dogs) [2]. Three details matter for our purposes. First, the stitching and retargeting modules from stage 2 were **not** trained for animals due to "several technical issues" — users are told to run with `--no_flag_stitching`, and paste-back is discouraged [2]. Second, animal mode swaps in X-Pose for keypoint detection because the native landmark detector assumes a human face anatomy [2]. Third, no training cost is published, but the relative dataset size (230K frames vs. 69M for stage 1) suggests this is closer in magnitude to a stage-2 retrain than a from-scratch run. This matters: it implies the F/W/G base modules are *fine-tunable on a small target-domain corpus* without catastrophic loss of the motion-warp scaffolding learned on the photoreal corpus.

For stylized portraits, the analogous recipe is: same base modules, ~10⁵-frame stylized corpus, M frozen, no stage 2. The biggest unknown is whether F + W + G need full re-training or whether LoRA-style low-rank adapters on the SPADE decoder G would carry most of the appearance shift while leaving F and W intact. None of the sources surveyed report on LoRA fine-tunes of LP specifically.

### What's actually open-source vs. inference-only

The KwaiVGI/LivePortrait GitHub repository ships inference code, model weights, and stitch/retarget training paths, but the searched sources do not surface a public stage-1 training script for the base modules [1][2]. Animals-mode weights are released as a fine-tuned checkpoint (`liveportrait_animals` on HuggingFace), not as a reproduction recipe. **This is the largest practical risk for our path**: replicating stage-1-style training from scratch requires reimplementing the loss scaffold (cascaded perceptual loss across global/face/lip regions, three GAN discriminators, equivariance and prior losses on implicit keypoints, Wing-loss landmark guidance) [1] without an official reference. The "fine-tune base modules from a released checkpoint" path is feasible but undocumented.

### Diffusion alternatives are quality references, not speed references

X-Portrait (SIGGRAPH 2024, ByteDance) [4][5] is the strongest published baseline for stylized portrait animation: a conditional diffusion model with ControlNet-style hierarchical motion attention, with the explicit claim that "once trained, the model is able to generalize to out-of-domain appearances through its learned latent space, as exemplified by stylized portraits." X-Portrait 2 / X-NeMo is the follow-up [5]. Source examples include Pexels, Midjourney, and DeviantArt portraits, confirming stylized coverage. **But neither paper publishes per-frame inference cost, and the architecture is fundamentally many-step diffusion** — empirically, diffusion-based portrait animation runs at ~hundreds of milliseconds to seconds per frame even with aggressive distillation (cf. our PersonaLive shipping at ~3s glass-to-OBS [project memory]). X-Portrait is therefore a quality target ("this is what a stylized-tolerant renderer should look like") not a speed competitor to LP's 12.8 ms/frame on RTX 4090 [8].

The DINO-guided one-shot stylization paper [7] is interesting as a per-anchor adaptation pattern: it fine-tunes a deformation-aware StyleGAN+STN against a single paired real-style example in ~10 minutes on an RTX 3090. But the output is **still images only** — there is no animation pipeline. It cannot replace LP. It is potentially useful as a **per-anchor preprocessing step** that emits a "neutralized" representation our renderer can consume, but that's a downstream design question.

TPSMM (CVPR 2022) [6] is the architectural predecessor concept and does not improve on LP for stylized inputs.

### The corpus is the actual bottleneck

LP-animals required 230K frames [2]. A stylized-portrait fine-tune needs a corpus of comparable order. Available stylized data falls into three buckets, none of which is large-scale curated stylized **video**:

- **AAHQ** (~60K stylized stills, used by LP at stage 1 [1]) — a starting point for image-only mixed training but only ~1× per identity, so weak for video training.
- **Bespoke synthetic data** generated by running an existing diffusion stylization on a photoreal driver dataset (CelebV-Text, VoxCeleb2, TalkingHead-1KH [3]). This produces paired (photoreal driver → stylized output) frames at scale and is the most likely path for ≥100K-frame stylized video.
- **Curated stylized video clips** from animation-keyframe datasets (Sakuga, anime-portraits-corpora). Small and license-fragmented.

The bespoke-synthetic option is the most promising: stylize a photoreal video corpus once with a heavy diffusion model (X-Portrait, AniPortrait, or even a still-frame Flux-LoRA pass), then fine-tune LP's F+W+G against the resulting stylized frames using the same photoreal driver as the motion source. This is structurally identical to what LP did at stage 1 (mixed image+video on AAHQ), just at a corpus scale that flips the photoreal:stylized ratio.

### Per-anchor test-time fit may be more pragmatic than full retrain

A second, smaller-effort path: skip the corpus fine-tune entirely and fit a small per-anchor adapter at first-paint time. The DINO paper's 10-min/RTX-3090 fit [7] for still-image stylization is the existence proof for "anchor-specific neural fit at upload time." For our pipeline, this would mean: when a user uploads a Vrubel painting, run a 5-30 minute fit that produces a small LoRA on the SPADE decoder weights specific to that anchor. Inference then loads the LoRA on top of the base LP weights. This trades upload latency for renderer generality and avoids the need to assemble a stylized video corpus. Risk: no published evidence that LoRA on SPADE-decoder weights captures the full domain shift; would need to be spiked.

## Comparison

| System | Architecture | Stylized? | Speed (per frame) | Open? | Trainable on custom data? |
|---|---|---|---|---|---|
| LivePortrait base [1] | F+M+W+SPADE-G + stitch/retarget | partial (60K stills + 1.3K clips in training) | ~13 ms / RTX 4090 | inference only | base train script not public; stitch/retarget script public |
| LP-animals [2] | base modules fine-tuned, no stage 2 | n/a (animals, not stylized) | ~13 ms / RTX 4090 | inference + weights | recipe described but script not published |
| **LP-stylized (proposed)** | **mirror animals recipe; F+W+G fine-tune on stylized corpus, M frozen, no stage 2** | **target** | **~13 ms / RTX 4090 (unchanged)** | **n/a (us)** | **needs stage-1 reimpl OR LoRA on G** |
| X-Portrait [4] | Diffusion + ControlNet motion attention | yes (claimed via diffusion prior) | not published; many-step diffusion ⇒ ≫LP | inference released | not described |
| TPSMM [6] | Thin-plate-spline warping | weak | ~LP-class | full code | yes |
| DINO one-shot stylization [7] | StyleGAN+STN per-anchor fit | yes (still only) | 10 min fit then fast inference | full code | yes (per anchor) |

## Open Questions

- **Is there a public LP stage-1 training script?** The repo search did not surface one. Confirming this requires direct repo inspection. If absent, base-module fine-tuning requires reimplementing 8 loss terms from the paper [1], which is non-trivial but bounded.
- **Does LoRA on SPADE-decoder weights alone capture the photoreal→stylized shift?** No published evidence either way. This is a cheap spike (~1-2 days, no corpus needed if test against a single anchor).
- **Does W (warping module) need to change?** Optical-flow-style warping arguably *should* break on non-anatomical regions like Vrubel mosaic strokes — but the failure mode could equally live in G (decoder texture priors) without W needing retraining. Inspecting LP-animals weights for which modules drift most from base would answer this empirically.
- **What's the smallest viable stylized video corpus?** Animals worked at 230K frames with anatomically-similar-to-human keypoints; stylized faces have more domain-gap variance. 100K is a guess; could be more.
- **Synthetic stylized data quality**: if we generate stylized video by passing photoreal video through a diffusion stylizer, do the frame-to-frame inconsistencies of the stylizer poison LP's learned warp consistency? Unknown; would need a small pilot.

## Sources

[1] Guo, Zhang, Liu et al. "LivePortrait: Efficient Portrait Animation with Stitching and Retargeting Control." arXiv:2407.03168, 2024. https://arxiv.org/html/2407.03168v1 (Retrieved 2026-05-08)
[2] LivePortrait Animals Mode Changelog. KlingAIResearch/LivePortrait. https://github.com/KlingAIResearch/LivePortrait/blob/main/assets/docs/changelog/2024-08-02.md (Retrieved 2026-05-08)
[3] Wang, Mallya, Liu. "One-Shot Free-View Neural Talking-Head Synthesis for Video Conferencing." CVPR 2021. https://nvlabs.github.io/face-vid2vid/ (Retrieved 2026-05-08)
[4] Xie, Tang, Yang et al. "X-Portrait: Expressive Portrait Animation with Hierarchical Motion Attention." SIGGRAPH 2024. https://byteaigc.github.io/x-portrait/ (Retrieved 2026-05-08)
[5] X-Portrait 2 / X-NeMo project page. https://byteaigc.github.io/X-Portrait2/ (Retrieved 2026-05-08)
[6] Zhao, Zhang. "Thin-Plate Spline Motion Model for Image Animation." CVPR 2022. arXiv:2203.14367. https://arxiv.org/abs/2203.14367 (Retrieved 2026-05-08)
[7] Bao et al. "Deformable One-shot Face Stylization via DINO Semantic Guidance." arXiv:2403.00459, 2024. https://arxiv.org/html/2403.00459v2 (Retrieved 2026-05-08)
[8] LivePortrait HuggingFace Space — "A cartoony face (how to improve)" discussion. https://huggingface.co/spaces/KwaiVGI/LivePortrait/discussions/20 (Retrieved 2026-05-08)
[9] LivePortrait official GitHub. https://github.com/KwaiVGI/LivePortrait (Retrieved 2026-05-08)
[10] AAHQ dataset (Artistic-AHQ), referenced as styled training data in [1].
