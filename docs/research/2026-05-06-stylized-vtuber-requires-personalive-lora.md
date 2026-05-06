---
status: live
topic: personalive-acceleration
---

# Stylized vtubing requires PersonaLive + LoRAs; LivePortrait has no analogous surface

Date: 2026-05-06

## Headline

After running both backbones across photoreal, painting, stylized-humanoid, and non-human reference sets against the MySlate_5 yaw-stress driver, the empirical conclusion is:

- **Photoreal frontal references**: both work; PersonaLive marginally cleaner.
- **Photoreal atypical pose** (Tikhonov on pillow): LivePortrait wins; PersonaLive collapses to its FFHQ prior.
- **Painting** (Pushkin): PersonaLive cleaner; LivePortrait usable.
- **Stylized humanoid** (orc/zombie/demon/anime): both fail or are crop-confounded. Open pending crop ablation.
- **Non-human** (cartoon ducks, with animal model): structurally incoherent under yaw on both. **No vtuber path on either backbone today.**

The unique question this doc answers: *if neither backbone handles stylized references out of the box, which one has a knob we can turn?*

## Answer: PersonaLive has a LoRA surface; LivePortrait doesn't

PersonaLive's failure mode on stylized references is **generative collapse** — the SD1.5 RefNet+UNet was trained on FFHQ-distilled identity, so off-distribution refs get pulled toward photoreal humans. This is the mechanism Concept Sliders, IP-Adapter, PuLID, and standard SDXL-style LoRAs are designed to modulate.

LivePortrait's failure mode is **warp infeasibility** — the implicit-keypoint regressor was trained on VoxCeleb humans; given a duck or stylized humanoid reference its 21 implicit keypoints don't anchor sensibly, and the warping module produces structural breakage rather than identity drift. There is no prior to retune. There is no attention surface to inject a δ. The W+G modules are not diffusion; LoRA techniques don't apply.

| Backbone | Failure mechanism on stylized | Tunable surface? |
|---|---|---|
| PersonaLive | Generative collapse to FFHQ prior | **Yes** — RefNet/UNet LoRA, low-α style LoRA, prompt conditioning |
| LivePortrait | Warp/keypoint infeasibility | **No** — would require retraining the implicit-keypoint regressor on stylized data |

Retraining the keypoint regressor on stylized data is a 12-month research project (need dataset, need motion-disentangled keypoint definition that survives stylization). LoRA training on PersonaLive is a 1–3 day exercise per character class.

## What this means for the four-product framework

From `2026-05-06-vtuber-pipeline-priorities.md`:

- **RGB-PersonaLive (Regime A)**: stays the default. LoRA is the fix for stylized refs.
- **ARKit-PersonaLive (Regime A)**: same backbone, same LoRA path applies.
- **RGB-LivePortrait (Regime A)**: viable for photoreal refs (incl. atypical poses), useful as a *router branch*, not a primary product. Depends on FasterLivePortrait+TRT for realtime FPS.
- **Slider authoring (Regime B)**: composes cleanly upstream — Regime B authors the canonical stylized portraits that Regime A drives. No conflict.

## Performance numbers (vanilla LivePortrait)

Measured 2026-05-06 on RTX 5090:

- 600-frame yaw clip, 512², `--flag_crop_driving_video --no-flag_pasteback`
- **71.13s wall time = ~8.4 FPS end-to-end**, ~10 FPS subtracting model load + warmup.

Vanilla LivePortrait is **not realtime**. The realtime path is [FasterLivePortrait](https://github.com/warmshao/FasterLivePortrait) with ONNX/TensorRT engines, reported at 30+ FPS on RTX 3090 (so ~40+ FPS expected on 5090). For any vtuber product the LivePortrait branch must use the FasterLivePortrait fork; vanilla is for sanity tests only.

## LoRA experiment proposals (PersonaLive)

In rough order of cost/risk:

- **Low-α RefNet style LoRA** on a small stylized-portrait set (zombie/orc/painting characters): targets the identity prior without retraining MotEncoder. Cheapest. Retrain Stage 2 only.
- **Per-character RefNet LoRA**: train one LoRA per stylized class (orc, demon, zombie, anime). Use Concept-Slider-style paired training. Identity-preserving by construction.
- **Joint UNet+RefNet LoRA**: broader distribution shift. Higher risk of motion-extractor confusion (MotEncoder still expects FFHQ-like inputs in cropping space).

Pre-conditions: standardized cropper (the ongoing crop-ablation thread) and frame-0 baseline-subtraction control. Both must be locked before LoRA training so we can attribute quality changes to the LoRA, not to upstream confounds.

## What's explicitly out of scope

- LivePortrait LoRA / fine-tuning: no surface for it short of retraining the keypoint regressor.
- Animal-model retraining for cartoons: animal v1.1 weights handle real animals; cartoon ducks are off-distribution for both XPose and the keypoint head. Same retraining cost as above.

## Companions

- `2026-05-06-vtuber-pipeline-priorities.md` — four-product framework, regime split.
- `2026-05-06-pivot-decision-personalive-vs-liveportrait.md` — UNet-surgery analysis.
- `2026-05-06-liveportrait-architecture-review.md` — five-module decomposition.
- `_topics/personalive-acceleration.md` — topic index (updated in same commit).
