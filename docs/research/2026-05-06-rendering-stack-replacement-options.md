---
status: live
topic: personalive
---

# Rendering stack replacement options for PersonaLive

## Summary

PersonaLive (CVPR 2026, [arXiv 2512.11253](https://arxiv.org/abs/2512.11253), [GVCLab/PersonaLive](https://github.com/GVCLab/PersonaLive)) hits 15-20 FPS at 512² on H100 via SD1.5 + 4-step distillation + StyleGAN2-FFHQ adversarial loss. Two empirical pain points drive this study: (a) we want 30+ FPS / lower-VRAM / stylized-anchor robustness, and (b) on non-human refs (anime, cartoon zombie, orc, gray demon) the output collapses to a generic FFHQ blonde inside the silhouette — the paper itself shows this in §5/Fig. 8.

The space splits cleanly. **Speed** is a near-solved problem with several drop-in or low-touch paths (Hyper-SD/DMD2 LoRA on the existing UNet, TensorRT-FP8, FlashPortrait's adaptive-skip), and one orthogonal escape hatch (LivePortrait — the GAN baseline runs ~78 FPS at 512² on a 4090 with no diffusion at all). **Quality on stylized refs** is the harder, more-open problem; the cleanest fixes are (i) replace the FFHQ-StyleGAN2 discriminator and re-run Stage-2 adversarial training on a mixed corpus, (ii) train per-style LoRAs on the denoising UNet, or (iii) hop to a bigger backbone with a more diverse prior (FLUX-class). The expensive path — full retrain on FLUX-Schnell or Z-Image-Turbo — is feasible on 5090 only at small batch + LoRA-style adaptation, not full fine-tune.

Recommended next experiments at the bottom; the highest-leverage cheap one is **the LivePortrait stylized-anchor sanity check** (1 hour) — if LivePortrait handles our 5 stylized anchors at 78 FPS we have a "fast path for non-human, PersonaLive for photoreal" branch and most of the rendering problem dissolves.

---

## Axis 1 — Speed

### Step distillation on the existing SD1.5 UNet

PersonaLive already runs 4 steps via the appearance-distillation loss in their Stage 3. Going lower (1-2 steps) without retraining the conditioning stack is plausible because the conditioning (ReferenceNet + motion encoder + pose guider) is upstream of the denoising UNet and is largely indifferent to step schedule. Candidates:

- **Hyper-SD** ([arXiv 2404.13686](https://arxiv.org/html/2404.13686v2)) — Trajectory Segmented Consistency. **SOTA on SD1.5 low-step**: in the LoRA-distill setting on SD1.5 it achieved 20% lower FID than DMD2 with 4% the training time. Drops to 1-2 steps cleanly. **MIT license**, LoRA available on HF. Best first try.
- **DMD2** ([github.com/tianweiy/DMD2](https://github.com/tianweiy/DMD2)) — Distribution Matching Distillation v2; needs teacher access for retrain. Stronger when retrained, weaker as a drop-in LoRA. CC-BY-NC (research-only).
- **PCM** (Phased Consistency Models) — sweet-spot at 4-8 steps, less mature SD1.5 LoRA story than Hyper-SD.
- **LCM / LCM-LoRA** — 2024-vintage, superseded for quality at the same step count by Hyper-SD; still the easiest 4-step LoRA available. Apache-2.0.
- **SD-Turbo / SDXL-Turbo / ADD** ([arXiv 2311.17042](https://arxiv.org/abs/2311.17042)) — 1 step at 512² for SD-Turbo (~67 ms UNet on A100). Requires full Stage-2 retrain because the discriminator is baked into ADD itself. Stability AI non-commercial license.
- **Flash-DMD** ([arXiv 2511.20549](https://arxiv.org/html/2511.20549v1)) — Nov 2025; outperforms DMD2 with 2.1% the training cost. Worth tracking; flow-based, not directly SD1.5.

**Z-image-turbo** is real and is from **Alibaba Tongyi Lab, not Tencent** ([Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo), base paper [arXiv 2511.22699](https://arxiv.org/abs/2511.22699)). 6B-param single-stream DiT distilled with Decoupled-DMD + DMDR (DMD + RL). 8 NFEs, ~sub-second on H800, fits 16 GB VRAM. **Apache-2.0 (free for commercial)**. Not a portrait-animation pipeline — it's text-to-image — so adopting it means rebuilding the entire ReferenceNet/motion-encoder stack on a new DiT backbone. High cost, but the license + speed make it the most attractive non-FLUX big-backbone candidate.

### Architecture replacement at base

- **SDXL-Turbo backbone** (1024² native, 1 step) — full retrain of conditioning stack required. The ReferenceNet pattern (deep feature injection at every UNet block) does port to SDXL — that's effectively how AnimateDiff-XL and Hallo work — but you lose the existing 4-step distill, the motion encoder weights, and the pose guider. ~3-6 weeks of GPU work on top of a working SD1.5 baseline, plus the StyleGAN2 disc no longer matches the resolution.
- **SD3 / SD3.5** — flow-matching DiT, ReferenceNet pattern needs rework (no UNet skip-symmetry to inject into); SenseFlow ([arXiv 2506.00523](https://arxiv.org/html/2506.00523v1)) is the closest analogue and explicitly distills SD3.5-Large + FLUX-dev. Not a portrait-anim adaptation yet.
- **FLUX-dev / FLUX-Schnell** — 12 B params. **Schnell uses 4-step latent ADD, Apache-2.0 license**. Portrait-identity work exists ([PuLID-FLUX](https://github.com/ToTheBeginning/PuLID), [InfiniteYou-FLUX](https://github.com/bytedance/InfiniteYou) ICCV 2025 Highlight) but neither is animation. We'd need to graft a temporal/motion stack onto FLUX, which nobody has published yet. 24-48 GB VRAM, 5090 has 32 GB so it's marginal at fp8. See Quality section.
- **Z-Image / Z-Image-Turbo** — covered above. 6B DiT, Apache-2.0, no portrait-anim port exists.
- **Video DiTs** (HunyuanVideo, LTX-Video, Mochi-1, CogVideoX, Wan2.2) — heavyweight, latent-video native. **Wan2.2-Animate-14B** ([Wan-AI/Wan2.2-Animate-14B](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B)) is the closest production talking-head model; not real-time, but decoupled body/face skeleton conditioning + Relighting LoRA for character swap. Large-VRAM, batch-mode, not streaming. Useful as an offline quality reference, not a PersonaLive replacement.

### GAN paths (no diffusion)

- **LivePortrait** ([arXiv 2407.03168](https://arxiv.org/html/2407.03168v1), [project page](https://liveportrait.github.io/)) — implicit-keypoint warp-and-decode, no diffusion. **12.8 ms/frame on RTX 4090 (~78 FPS @ 512²)**. **MIT license**. Already supports cartoon/anime portraits with retargeting. Authors note it lacks fine detail (PersonaLive paper §4 quotes this). **This is the lowest-risk fast path for stylized refs.** On a 5090 we should expect ≥100 FPS.
- **X-Portrait / X-NeMo** ([byteaigc.github.io/X-Portrait2](https://byteaigc.github.io/X-Portrait2/)) — diffusion, not GAN; PersonaLive credits X-NeMo as a code ancestor. Not faster than PersonaLive.
- **FaceVid2Vid / MegaPortraits** — older (2021-2022), warp-based GAN, fast but lower quality than LivePortrait, no live maintenance.
- **Wav2Lip / SadTalker** — audio-driven lip-sync only, not full face animation. Wrong shape.
- **StyleHEAT / PIRender** — pre-LivePortrait warp-based, dead/stale. Skip.
- **NeRF / 3DGS talking heads** (NHA, INSTA, GaussianHeadAvatar, [GaussianAvatars CVPR 2024](https://github.com/ShenhanQian/GaussianAvatars), [Avat3r](https://tobias-kirschstein.github.io/avat3r/)) — per-subject training, photoreal quality, real-time once trained. Bad fit for our anchor model (we don't have multi-view capture per anchor). Useful only if we add a "register a new anchor" workflow.

### Hardware / decoder optimisations (orthogonal to model choice)

- **TinyVAE** — already known, +25% FPS in PersonaLive paper. Free win.
- **TensorRT FP8 / INT8** — NVIDIA reports 1.72× (INT8) and 1.95× (FP8) over fp16 torch.compile on RTX 6000 Ada for diffusion ([NVIDIA blog](https://developer.nvidia.com/blog/tensorrt-accelerates-stable-diffusion-nearly-2x-faster-with-8-bit-post-training-quantization/)). FP8 is the better target on 5090 (Blackwell has native fp8). Pairs cleanly with our existing `2026-05-04-personalive-tensorrt-plan.md`. Also: NVIDIA Model-Optimizer has a portrait-friendly diffusion PTQ recipe.
- **FlashPortrait** ([arXiv 2512.16900](https://arxiv.org/abs/2512.16900), CVPR 2026, [Francis-Rings/FlashPortrait](https://github.com/Francis-Rings/FlashPortrait)) — adaptive-step skipping via higher-order latent derivatives + sliding window; **6× speedup, ID-preserving, infinite-length**. Designed as a *general accelerator* on top of an existing video-diffusion portrait pipeline. **Most exciting drop-in candidate** for PersonaLive — same authors' lineage as the streaming/sliding-window pattern PersonaLive uses. License: see repo (Apache-2.0 likely).

---

## Axis 2 — Quality on stylized / non-human refs

The diagnosis is FFHQ-prior pull from two sources: (i) the StyleGAN2-FFHQ discriminator in Stage 2 reshapes the UNet's output distribution toward FFHQ, (ii) the SD1.5 base model already has a strong "natural human face" prior. A non-human ReferenceNet feature lands OOD, gets pulled to the nearest mode in the discriminator's manifold = generic FFHQ blonde.

### Stylized portrait-animation specialists

- **RAIN** ([arXiv 2412.19489](https://arxiv.org/html/2412.19489v1)) — *Real-time Animation of Infinite Video Stream*. 1D temporal-attention adapters on top of fine-tuned SD; trained on character-animation data including anime/illustration. The mechanism PersonaLive cites for streaming, not for stylization per se — RAIN's anime ability comes from its training corpus, not from its architecture. Does not solve our problem cleanly because it still needs a corpus to inherit from. Code/license: check repo (link in paper). **Useful as: the streaming-architecture reference**; less useful as a stylized-quality fix.
- **Knot Forcing** ([arXiv 2512.21734](https://arxiv.org/abs/2512.21734)) — autoregressive video diffusion with knot-and-context tricks; PersonaLive-class infinite real-time portrait animation, 2025-12. Also targeted at human refs.
- **Hallo / Hallo2 / Hallo3** ([fudan-generative-vision/hallo2](https://github.com/fudan-generative-vision/hallo2), [arXiv 2412.00733](https://arxiv.org/html/2412.00733v1)) — audio-driven, ICLR 2025. Hallo3 = video-DiT backbone, identity-reference network, best quality in the family. Not real-time. Photoreal-trained.
- **AniPortrait / V-Express / EchoMimic** — audio-driven, 2024 cohort. AniPortrait has the cleanest landmark-driven conditioning. None trained on heavy anime/cartoon data.
- **EMO** (Alibaba, 2024) — closed source, audio-driven, no public code.
- **CAP4D** ([arXiv 2412.12093](https://arxiv.org/html/2412.12093v1)) — animatable 4D portrait via morphable multi-view diffusion. Heavy, closer to avatar than animation.
- **FantasyPortrait** ([arXiv 2507.12956](https://arxiv.org/html/2507.12956v1)) — multi-character, expression-augmented DiT. Worth glancing at for stylized expression handling but not faster.
- **DualStyleGAN / StyleGAN-NADA / Cartoon-StyleGAN** — *image* style transfer, not animation, but **directly relevant as discriminator-replacement candidates**: StyleGAN2 weights fine-tuned on Cartoon/Anime/Pixar/Sketch domains exist as drop-ins. Replace the FFHQ-pretrained discriminator with a Cartoon-StyleGAN2 disc and re-run Stage-2.

### No-train / light-train fixes for the existing PersonaLive UNet

Ranked by expected leverage:

1. **Replace the discriminator + re-run Stage-2.** Use [Cartoon-StyleGAN](https://github.com/happy-jihye/Cartoon-StyleGAN) or [DualStyleGAN](https://github.com/williamyang1991/DualStyleGAN) weights — pretrained anime/cartoon StyleGAN2 — as the Stage-2 discriminator. Ideally **mix** discriminators (FFHQ + anime + cartoon, weighted) so the UNet doesn't lose photoreal. This is the most direct cure for the diagnosed root cause. Cost: Stage-2 alone, not Stage-3 distillation; ~1-3 days on 5090.
2. **CivitAI SD1.5 anime/cartoon LoRAs on the denoising UNet at inference.** Hundreds available (Counterfeit, MeinaMix, ToonYou, Disney Pixar Cartoon, AnythingV5, etc.). Risk: the temporal-attention modules and ReferenceNet were trained against FFHQ-styled features; injecting an anime LoRA may produce identity-bank confusion or temporal flicker. Realistic outcome: visible improvement on stylized anchors, some flicker, identity drift. **Cheap to test (an hour per LoRA).**
3. **Per-style LoRA training on the denoising UNet.** A few hundred stylized (image, driver-pose) pairs, ~hours on 5090. AnimateDiff-LoRA recipe applies. Better than (2) because trained against the actual conditioning signals.
4. **CFG hack: replace the zero-uncond CLIP embed.** Push the unconditional toward "generic human face" so the conditional has to *push away* from human prior. Free to try, ~30 min, low expected leverage but instructive — tells us how much of the collapse is base-model prior vs. discriminator pull.
5. **Discriminator-free fine-tune (Stage-2-skip).** Train Stage 1 only with reconstruction loss on a mixed photoreal+stylized corpus, accept softer outputs, see if stylized refs survive. Diagnostic.

### Bigger backbones (FLUX-class) for portrait animation

No public FLUX-based **video-driven** portrait animation exists as of 2026-05. PuLID-FLUX, [InfiniteYou-FLUX](https://github.com/bytedance/InfiniteYou) (ICCV 2025 Highlight; InfuseNet beats PuLID-FLUX 72.8% to 27.2% on identity preservation in user testing) and IP-Adapter-FLUX target single-image identity preservation. Extending these to driven animation = open research project, ~3-6 months. **OmniHuman-1 / 1.5** ([arXiv 2502.01061](https://arxiv.org/html/2502.01061v1)) is the closest public ByteDance equivalent; DiT-based, not FLUX, **closed weights** (only API access via Replicate). Not actionable.

A **FLUX-Schnell + ReferenceNet + 4-step distill** portrait-animation pipeline is plausible but unbuilt; this is the "heavy win" path. 5090 (32 GB) handles FLUX-Schnell fp8 inference; training even LoRA is tight. Likely needs A100/H100 rental.

### Distillation + transfer paths

- **Distill FLUX-portrait → PersonaLive-shape student**: theoretically attractive (use a big-backbone teacher to teach a small-backbone student about stylized refs), but requires the FLUX-portrait teacher to exist first.
- **Cross-domain transfer: keep PersonaLive motion/temporal stack, swap UNet only**: feasible but the ReferenceNet is the binding piece — you'd swap UNet + ReferenceNet together and re-run Stage 2-3.
- **DreamBooth / StyleAlign on the SD1.5 base** before PersonaLive Stage 1 — biases the prior toward stylized but loses photoreal generality. Not recommended unless stylized becomes the dominant use case.

### Avatar / 3D paths (orthogonal, ecosystem-aligned)

- **VRoid / Live2D / VTuber rig driven by ARKit blendshapes** — already mature, sub-millisecond render, solves the non-human case completely (it's all rig). Loses photoreal. **Aligns with our existing ARKit-bridge thread**: we already have ARKit blendshapes flowing to PersonaLive; teeing the same ARKit signal off to a Live2D/VRoid rig is essentially free given current infra. **This is the single best non-human-anchor escape hatch in terms of effort.**
- **GaussianAvatars / Avat3r / GPAvatar (CVPR 2025)** — per-subject neural avatars; high quality, real-time once trained, but require multi-view per anchor. Wrong shape for our flow.

---

## Decision matrix

Ranked by leverage × feasibility, given RTX 5090 + our `apply_bridge_to_personalive.py` + 5 stylized + 9 photoreal anchors.

### Quick wins — no/light retrain (this week)

| Option | Effort | Expected leverage | Risk |
|---|---|---|---|
| **LivePortrait stylized-anchor sanity check** | 1-2 h | High (≥78 FPS, MIT, anime-capable) | Quality below PersonaLive on photoreal |
| **TensorRT FP8** on existing PersonaLive | 2-3 d | Medium (~1.95× = 30-40 FPS) | Build pain on 5090/Blackwell stack |
| **TinyVAE** (if not already wired) | hours | Low-Medium (+25%) | None |
| **Hyper-SD LoRA → 1-2 step** on existing UNet | 1 d | Medium (2× speed, quality dip) | Quality regression on photoreal |
| **CivitAI anime LoRA at inference** | 1 h/LoRA | Medium (sometimes-fix on stylized) | Flicker, identity drift |
| **CFG zero-uncond → "generic human face"** diagnostic | 30 min | Low alone, high diagnostic value | None |
| **FlashPortrait integration** on PersonaLive | 3-5 d | High (claims 6×) | Newly-released code, integration debt |

### Medium-effort wins — LoRA / partial retrain (this month)

| Option | Effort | Expected leverage | Risk |
|---|---|---|---|
| **Replace Stage-2 discriminator** with Cartoon-StyleGAN2 (or mixed FFHQ+anime+cartoon) and re-run Stage 2 only | 3-5 d | **Highest for the stylized-anchor problem** | Photoreal quality may regress; mixing weights is empirical |
| **Per-style LoRA train on denoising UNet** (5 stylized anchors → 5 LoRAs) | 1-2 d each | High on the trained style, none on others | Per-style scaling cost |
| **ARKit → Live2D/VRoid rig** branch (parallel to PersonaLive) | 1-2 weeks | Total cure for non-human if we accept non-photoreal | Visual style change; UI to switch branches |

### Heavy wins — backbone swap (quarter+)

| Option | Effort | Expected leverage | Risk |
|---|---|---|---|
| **FLUX-Schnell + ReferenceNet + temporal + 4-step distill** portrait pipeline | 3-6 months | Highest quality ceiling, unblocks stylized via FLUX prior diversity | First-of-kind build; A100/H100 rental needed for training; speed will halve at best |
| **Z-Image-Turbo backbone** (Apache-2.0, 6B, 8-NFE) + portrait-anim grafting | 2-4 months | Mid-quality ceiling but cleaner license + size than FLUX | No prior portrait-anim work on this backbone; pioneering risk |
| **SDXL-Turbo backbone** with full Stage 1-3 retrain | 6-10 weeks | 1024² native, 1-step | Stability AI non-commercial license |

---

## Concrete next experiments (priority-ranked)

1. **LivePortrait sanity check (1 hour).** Run our 5 stylized + 9 photoreal anchors through stock LivePortrait at 512². Score with our existing `render_metrics.parquet` pipeline (mediapipe per-frame). Decision: if stylized scores ≥ "PersonaLive collapse mode," declare a dual-path branching (LivePortrait for non-human, PersonaLive for photoreal) and stop the stylized-fix work on PersonaLive itself.
2. **CFG zero-uncond diagnostic (30 min).** Replace zero-text uncond with CLIP embed of "generic human face." If collapse worsens, FFHQ pull is mostly discriminator-side (do experiment 4). If collapse persists unchanged, base-model prior dominates (do experiment 5).
3. **CivitAI anime LoRA inference (2 hours).** ToonYou + Counterfeit on the denoising UNet at inference; same anchor set; same metrics. Cheap and tells us how much the base SD1.5 prior alone is responsible for the collapse vs. the discriminator.
4. **Discriminator swap pilot (3-5 days).** Re-run Stage 2 with mixed discriminators: 0.5 × FFHQ + 0.25 × Cartoon-StyleGAN2 + 0.25 × Anime-StyleGAN2. Single training run, evaluate on both anchor sets; if photoreal degrades by <5% on FID and stylized improves materially, ship it.
5. **TensorRT FP8 build (2-3 days).** Continue per `2026-05-04-personalive-tensorrt-plan.md`. Should land 30-40 FPS at 512² on 5090, independent of stylized work.
6. **FlashPortrait wiring (3-5 days).** Adaptive-step skipping is the highest expected speed gain at the model level; the github [okdalto/ComfyUI-FlashPortrait](https://github.com/okdalto/ComfyUI-FlashPortrait) port suggests integration is tractable.
7. **Hyper-SD LoRA distillation to 2-step (1 day).** If Stage-3 distillation is held constant and we just add Hyper-SD LoRA on the denoising UNet, do we get to ~30 FPS at acceptable photoreal quality? Cheap test.
8. **(Stretch) ARKit → VRoid/Live2D parallel branch.** Reuses the existing ARKit blendshape extraction (zero new training). 1-2 weeks of integration. Gives the non-human anchors a *correct* render path independent of any diffusion stack.

---

## Sources

- PersonaLive: [arXiv 2512.11253](https://arxiv.org/abs/2512.11253), [GVCLab/PersonaLive](https://github.com/GVCLab/PersonaLive), [HF papers](https://huggingface.co/papers/2512.11253), [HF weights](https://huggingface.co/huaichang/PersonaLive)
- Step distillation: [Hyper-SD](https://arxiv.org/html/2404.13686v2), [DMD2](https://github.com/tianweiy/DMD2), [SDXL-Turbo / ADD](https://arxiv.org/abs/2311.17042), [SDXL-Lightning](https://arxiv.org/html/2402.13929v1), [Flash-DMD](https://arxiv.org/html/2511.20549v1), [SenseFlow](https://arxiv.org/html/2506.00523v1)
- Z-Image: [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo), [arXiv 2511.22699](https://arxiv.org/abs/2511.22699), [GitHub](https://github.com/Tongyi-MAI/Z-Image)
- FLUX: [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell), [PuLID-FLUX](https://github.com/ToTheBeginning/PuLID), [InfiniteYou ICCV 2025](https://github.com/bytedance/InfiniteYou)
- Portrait animation: [LivePortrait](https://liveportrait.github.io/) ([arXiv 2407.03168](https://arxiv.org/html/2407.03168v1)), [X-Portrait](https://byteaigc.github.io/x-portrait/) ([arXiv 2403.15931](https://arxiv.org/abs/2403.15931)), [RAIN](https://arxiv.org/html/2412.19489v1), [Knot Forcing](https://arxiv.org/abs/2512.21734), [FlashPortrait CVPR 2026](https://arxiv.org/abs/2512.16900) ([repo](https://github.com/Francis-Rings/FlashPortrait)), [Hallo2 ICLR 2025](https://github.com/fudan-generative-vision/hallo2), [Hallo3](https://arxiv.org/html/2412.00733v1), [OmniHuman-1](https://arxiv.org/html/2502.01061v1), [Wan2.2-Animate-14B](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B), [FantasyPortrait](https://arxiv.org/html/2507.12956v1), [CAP4D](https://arxiv.org/html/2412.12093v1)
- Stylized GANs: [DualStyleGAN CVPR 2022](https://github.com/williamyang1991/DualStyleGAN), [Cartoon-StyleGAN](https://github.com/happy-jihye/Cartoon-StyleGAN), [AniGAN](https://arxiv.org/abs/2102.12593)
- Avatars: [GaussianAvatars CVPR 2024](https://github.com/ShenhanQian/GaussianAvatars), [Avat3r](https://tobias-kirschstein.github.io/avat3r/), [GPAvatar CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Feng_GPAvatar_High-fidelity_Head_Avatars_by_Learning_Efficient_Gaussian_Projections_CVPR_2025_paper.pdf)
- Quantization / TRT: [NVIDIA TRT-INT8/FP8 blog](https://developer.nvidia.com/blog/tensorrt-accelerates-stable-diffusion-nearly-2x-faster-with-8-bit-post-training-quantization/), [NVIDIA Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer)
- AnimateDiff: [guoyww/AnimateDiff](https://github.com/guoyww/AnimateDiff), [ComfyUI-AnimateDiff-Evolved](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved)
