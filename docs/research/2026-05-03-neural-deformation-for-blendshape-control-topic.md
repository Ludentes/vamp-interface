---
status: live
topic: neural-deformation-control
---

# Neural deformation as an alternative to LoRA sliders for blendshape control

Practical viability check, going beyond the chapter overview in `~/w/Face-Research/05-neural-deformation.md`.

## Executive summary

The PowerHouseMan ComfyUI-AdvancedLivePortrait node implements a hand-coded, *additive* mapping from named sliders ("smile", "blink", "aaa", "eee", "woo", "wink", "pupil_x", "pupil_y", "eyebrow") onto a small set of LivePortrait's 21 implicit-keypoint indices via a function `calc_fe` — there is no public ARKit-52 mapping, no learned regressor, and the mapping is engineering shorthand rather than an orthogonal basis [1][2]. Driving LivePortrait directly from a target ARKit/MediaPipe blendshape vector is **not** an established public path: MediaPipe's 52 blendshapes are well-documented and easily extracted [3][4], but no published code maps `b ∈ R^52 → 21-dim implicit motion tensor`; you would build it yourself by curating slider-pair correspondences. IM-Animation (arXiv:2602.07498) defines disentanglement *implicitly* via a 1D motion-token bottleneck plus joint-heatmap supervision and a three-stage curriculum — it has **no explicit identity loss and no released checkpoint** [5]; FG-Portrait (arXiv:2603.23381) is the more useful target because it accepts FLAME expression coefficients directly as drivers (CSIM 0.462, FID 87.0 cross-reenactment on VFHQ) but also has no public code yet [6]. The actively releasable diffusion-conditioning path is HunyuanPortrait (CVPR 2025, code at Tencent-Hunyuan), which uses an Intensity-Aware Motion Encoder injected via cross-attention into stable video diffusion and reports ArcFace-similarity 8.87 vs LivePortrait 8.71 on the same eval [7][8] — but it conditions on a *driving video*, not blendshape vectors. **Bottom line for vamp-interface:** the cheapest credible non-LoRA path is "Flux render → AdvancedLivePortrait expression edit with hand-tuned slider deltas," accepting that (a) the slider basis is not orthogonal, (b) stylization and large rotations break it [9][10], and (c) blendshape-vector-driven control requires you to author the `b → 21-dim` mapping yourself.

## Q1 — How AdvancedLivePortrait actually maps sliders onto LivePortrait's motion tensor

LivePortrait's motion representation is **21 implicit 3D keypoints + head pose + expression deformation**, with the keypoints functioning as "implicit blendshapes" learned end-to-end (not aligned to ARKit) [11][12]. Per the paper: "compact implicit keypoints effectively represent a kind of blendshapes… expressions cannot be explicitly controlled but rather require a combination of these implicit blendshapes" [11].

PowerHouseMan's node implements *named* sliders by hand-tuned linear deltas on specific keypoint indices inside a method `calc_fe` (nodes.py, ~lines 760–830) [1]. Examples extracted from the source:

- `smile` modifies indices 3, 7, 13, 14, 16, 17, 20, e.g. `x_d_new[0, 20, 1] += smile * -0.01`, `x_d_new[0, 14, 1] += smile * -0.02` [1]
- `aaa` (mouth open) targets 3, 7, 17, 19, 20 with a coupled pitch-rotation term
- `eee` modifies 14, 20 (negative y)
- `woo` modifies 3, 7, 14, 17 with smaller magnitudes
- `eyes`/`wink` adjust 1, 2, 3, 7, 11, 13, 15, 16, 17 plus roll/yaw
- `pupil_x` / `pupil_y` → keypoints 11, 15
- `eyebrow` → 1, 2 with sign-conditional branching

**Composition**: all sliders accumulate additively into `x_d_new`; the rotation matrix from pose sliders is then composed via `x_d_new @ new_rotate` in `ExpressionEditor.run()` [1]. So pairwise additivity within the set is by construction, but **orthogonality is not** — `smile` and `aaa` both touch keypoints 3, 7, 17, 20 and will interact non-linearly once they pass through LivePortrait's warping module + decoder. There is **no public mapping document or paper** describing a principled derivation of these constants; they are engineering tuned, and ARKit-52 alignment is not provided [1][2]. MediaPipe's own 52-blendshape model is well-documented and known to under-fire on `eyeWidenLeft/Right`, `noseSneer*`, `cheekPuff` and to lack expressive capacity vs ARKit/Maxine [3][4].

## Q2 — Closed-loop "target b ∈ R^52 → motion tensor → warp"

No published code or paper does this end-to-end for LivePortrait. The available building blocks:

- MediaPipe Face Landmarker emits 52 ARKit-named blendshape coefficients per frame [3][4]; this is the *measurement* side.
- AdvancedLivePortrait's `calc_fe` accepts ~10 named sliders, not the 52-d ARKit vector [1].
- The mapping `R^52 → R^{21×3}` is unknown publicly and would need to be authored — either by:
  1. Hand-routing the ~10 ARKit channels that semantically match existing sliders (`mouthSmile* → smile`, `eyeBlink* → eyes`, `jawOpen → aaa`, `mouthFunnel/Pucker → woo`), accepting that ~40 ARKit channels have no slider counterpart;
  2. Or learning a regressor: render LivePortrait outputs at known slider settings, run MediaPipe on the result, and least-squares-fit `21×3 = M · b` (per-channel R² will be poor for channels the slider basis doesn't span).

Closest published precedent is **FG-Portrait**, which accepts FLAME `ψ` (expression coefficients) and `θ` (pose) as inference inputs rather than only a driving video [6]. FLAME ψ is a 50–100-d coefficient vector closer to ARKit semantics than LivePortrait's implicit tensor; conversions FLAME↔ARKit exist in the avatar community.

## Q3 — Identity preservation: IM-Animation and FG-Portrait, in detail

**IM-Animation (arXiv:2602.07498) [5].** Disentanglement is **architectural, not loss-based**. The mechanism: per-frame motion is compressed into compact 1D motion tokens (rather than 2D spatial tokens), which removes the spatial-alignment leak that lets identity bleed through in implicit methods. A retargeting module uses learnable mask tokens concatenated with image-latent + temporal-motion tokens as a bottleneck. Training is a three-stage curriculum: (1) regress joint heatmaps from the motion video (motion-encoder pretrain), (2) train retargeting with the mask-token bottleneck, (3) finetune the video diffusion model. The paper does **not** specify an explicit identity-preservation loss term. **No code or checkpoint is publicly released as of fetch date** [5]. Driving is video-only; no blendshape input.

**FG-Portrait (arXiv:2603.23381) [6].** This is the more interesting candidate for our use case because it explicitly supports parameter-driven editing: "the model can be driven not only by a driving image, but also by user-specified expression and pose parameters" — i.e. you can hand it a FLAME ψ and get the corresponding warp. The 3D-flow representation computes per-pixel backward correspondence from a target FLAME mesh (assembled from source shape + driving pose/expression) back to source pixel locations, which is a learning-free, geometry-driven motion field. Reported metrics on cross-reenactment: VFHQ CSIM 0.462 (ArcFace cosine), FFHQ FID 87.0; self-reenactment LPIPS 0.158. Authors flag two failure modes: limited mesh resolution caps fine expression detail (relevant to subtle blendshapes like `cheekSquint*`), and cartoon portraits need style-specific finetuning. **No public code or checkpoint as of fetch date** [6].

## Q4 — Diffusion-internal conditioning on motion tokens (ControlNet-style)

This is now an active research direction. Concrete examples:

- **HunyuanPortrait** (CVPR 2025, code released at Tencent-Hunyuan) [7][8]. Uses Stable Video Diffusion as backbone; an Intensity-Aware Motion Encoder produces implicit motion descriptors which are injected into the denoising U-Net via cross-attention plus AdaLN intensity modulation — explicitly an IPAdapter/cross-attn pattern, not a ControlNet sidecar. Identity-Sim 8.87, FID-VID 75.81, beats LivePortrait (8.71) on identity preservation [7][8]. **Critically: video diffusion, not single-image; conditions on driving video, not blendshape vector.**
- **X-NeMo** (ICLR 2025, code at github.com/bytedance/x-nemo-inference) [13]. Distills a 1D identity-agnostic motion latent from driving images and injects via cross-attention. Trained with a dual GAN decoder for disentanglement. Same cross-attention pattern as Hunyuan; same video-driven limitation.
- **RealPortrait** (AAAI) and **X-Portrait** [14][15] use parallel ControlNet branches for motion guidance — closer to the ControlNet pattern the question asks about. Still video-driven.
- **KDTalker** [16] is the closest "implicit-keypoint inside diffusion" example: predicts implicit 3D deformation keypoints + transformation params from audio via a spatiotemporal diffusion model.

No public work I found takes the **specific** path "Flux/SDXL conditioned on a 21-dim LivePortrait motion tensor or an ARKit-52 vector" as a single forward generation. That would be a novel contribution.

## Q5 — Practical fidelity loss of "Flux portrait → LivePortrait warp" cascade

I could not find a published independent benchmark of the specific Flux-then-LivePortrait cascade. What's available:

- LivePortrait reports its own self-reenactment numbers on real driving in the paper [11] and a community compression study reports usable quality at 22 kbit/s [17] — both on real source images, not synthetic Flux outputs. Synthetic source images often have subtly different statistics (Flux renders are smoother, lack high-frequency skin micro-detail), and LivePortrait's appearance-feature extractor was trained on real video; expect slightly degraded warp coherence and a higher chance of decoder hallucination on the warped patch.
- HunyuanPortrait reports ArcFace 8.87 vs LivePortrait 8.71 vs AniPortrait 7.95 on the same eval [7]; FG-Portrait reports CSIM 0.462 cross-reenactment [6]. These are absolute numbers under the authors' eval, not a head-to-head on synthetic source.

For our pipeline you would need to measure this yourself: render N Flux portraits, apply known slider settings, run ArcFace + a MediaPipe-blendshape regressor, report `(ArcFace_cos_drift, blendshape_target_error)` per slider amplitude. This is the same eval shape we already use for sliders; reusing the harness is straightforward.

## Q6 — Failure modes vs LoRA sliders

Documented and likely failure modes for the warp path that LoRA sliders avoid by construction:

- **Stylized portraits.** LivePortrait is reported to fail on stylized images and animal faces [9][10] — its appearance encoder was trained on real human video; cartoon/painterly Flux outputs (e.g. anything with a non-photoreal style LoRA stacked) warp poorly. LoRA sliders trained on Flux-distribution images don't have this problem.
- **Large head rotations.** LivePortrait struggles with large pose deltas between source and driver [9]; the warp produces temporal artifacts and visible occlusion holes. LoRA sliders don't move the head, so the failure doesn't apply.
- **Occlusions** (hair-over-eye, hand-near-face, glasses frame edges) [9]. The warp operates on a single appearance feature map and inpaints crudely behind occluders; LoRA sliders re-render and avoid this.
- **Eye gaze and pupil control.** AdvancedLivePortrait exposes `pupil_x`/`pupil_y` on keypoints 11 and 15 [1] but the keypoint geometry is coarse — fine gaze targets are approximate. LoRA sliders trained on a gaze-labeled corpus can be more precise.
- **Compound expressions (squint + jaw at once).** Because slider deltas in `calc_fe` overlap on shared keypoints (e.g. smile and aaa both touch indices 3, 7, 17, 20 [1]), the warping module's nonlinearity will entangle them — squint+jaw will not behave as the sum of squint-alone and jaw-alone. LoRA sliders are also not perfectly orthogonal in latent space, but they're at least trained against pixel targets where the compound exists.
- **Identity drift on extreme expression**. Implicit-keypoint methods leak some identity through the appearance feature when expression is far from neutral; this is the explicit motivation for IM-Animation, FG-Portrait, X-NeMo, HunyuanPortrait [5][6][7][13]. Our LoRA sliders fail differently — they fool the bs critic and drift identity in latent space rather than warping geometry.
- **High-frequency texture loss.** Any warp + decode pass softens skin texture compared to the original Flux render. LoRA sliders preserve original texture quality (or break it differently — via classifier fooling, not blur).

The two failure classes compose: the LoRA path fails toward "wrong-but-sharp" (classifier-fooling, identity drift); the warp path fails toward "right-but-degraded" (correct geometry, softer/blotchier texture, occlusion artifacts).

## Recommendation for vamp-interface (not asked, but load-bearing)

If the goal is a quick non-LoRA blendshape-control path to compare against, the minimum-effort credible build is: Flux portrait → AdvancedLivePortrait `calc_fe` with slider deltas you tune to match a target MediaPipe blendshape readout. Treat the existing 10 sliders as the basis (don't try to extend to 52). Eval with the same `(ArcFace_cos, bs_critic_R²)` harness already in use. If subtle-channel control matters, FG-Portrait is the right paper to track for code release; HunyuanPortrait is the only currently-runnable strong baseline with explicit identity-preservation gains, but it's video-driven only.

## Sources

- [1] [PowerHouseMan/ComfyUI-AdvancedLivePortrait nodes.py](https://github.com/PowerHouseMan/ComfyUI-AdvancedLivePortrait/blob/main/nodes.py) — `calc_fe` slider→keypoint index mapping, additive composition, `ExpressionEditor.run()`
- [2] [PowerHouseMan/ComfyUI-AdvancedLivePortrait README](https://github.com/PowerHouseMan/ComfyUI-AdvancedLivePortrait/blob/main/README.md)
- [3] [MediaPipe ARKit 52 blendshapes support — issue #3421](https://github.com/google/mediapipe/issues/3421)
- [4] [MediaPipe Blendshapes recording and filtering — Samer Attrah, Medium](https://medium.com/@samiratra95/mediapipe-blendshapes-recording-and-filtering-29bd6243924e)
- [5] [IM-Animation: An Implicit Motion Representation for Identity-decoupled Character Animation, arXiv:2602.07498](https://arxiv.org/html/2602.07498)
- [6] [FG-Portrait: 3D Flow Guided Editable Portrait Animation, arXiv:2603.23381](https://arxiv.org/html/2603.23381)
- [7] [HunyuanPortrait: Implicit Condition Control for Enhanced Portrait Animation, arXiv:2503.18860](https://arxiv.org/html/2503.18860v1)
- [8] [Tencent-Hunyuan/HunyuanPortrait — official code](https://github.com/Tencent-Hunyuan/HunyuanPortrait)
- [9] [LivePortrait paper (limitations) — arXiv:2407.03168](https://arxiv.org/html/2407.03168v1)
- [10] [LivePortrait — animal/stylized issue #119](https://github.com/KwaiVGI/LivePortrait/issues/119)
- [11] [LivePortrait paper (motion representation) — arXiv:2407.03168](https://arxiv.org/abs/2407.03168)
- [12] [LivePortrait project page](https://liveportrait.github.io/)
- [13] [bytedance/x-nemo-inference — ICLR 2025](https://github.com/bytedance/x-nemo-inference)
- [14] [RealPortrait: Realistic Portrait Animation with Diffusion Transformers, AAAI](https://ojs.aaai.org/index.php/AAAI/article/download/33012/35167)
- [15] [X-Portrait: Hierarchical Motion Attention, arXiv:2403.15931](https://arxiv.org/html/2403.15931v3)
- [16] [KDTalker: implicit keypoint spatiotemporal diffusion, arXiv:2503.12963](https://arxiv.org/html/2503.12963v1)
- [17] [Perceptually lossless talking-head compression with LivePortrait — Lumiste](https://mlumiste.com/technical/liveportrait-compression/)
