---
status: live
topic: neural-deformation-control
---

# Talking-head / portrait-animation SOTA — late 2025 through mid-2026

Companion to `2026-05-03-neural-deformation-for-blendshape-control-{topic,practical}.md` (those cover the video/landmark-driven branch — LivePortrait, AdvancedLivePortrait, HunyuanPortrait, X-NeMo). This doc adds the audio-driven branch, the streaming/infinite-length frontier, and a comparative read across both.

## Executive summary

The field has consolidated around **DiT (Diffusion Transformer) backbones distilled or autoregressively re-organised for streaming inference** [1][2][3][4]. The 2024 wave (EMO, V-Express, Hallo, LivePortrait) produced offline-quality demos; the 2025 wave (Sonic, Hallo2, Teller, OmniHuman, TalkingMachines) productionised quality and length; the 2025-late / 2026 wave (PersonaLive, RAP, Hallo-Live, FG-Portrait) is hitting **real-time on a single consumer GPU**. PersonaLive (CVPR 2026) is the current best-in-class openly-runnable streaming portrait animator: 12 GB VRAM, infinite length, Apache-2.0, ComfyUI node, 7–22× speedup over diffusion baselines [5]. Teller is the leader for *audio-driven* AR streaming at 25 FPS [1]. ByteDance's OmniHuman-1 / 1.5 (DiT, ~19 k h training data, omni-conditions text+audio+pose) defines the closed-source production frontier and seeds the design pattern of multi-modal weak-to-strong conditioning [6]. Identity preservation has effectively saturated as a metric (CSIM > 0.9 across the leaderboard) [7]; differentiation is now about (a) FPS at fixed quality, (b) conditioning richness (audio + pose + text + ARKit), and (c) license / runnable footprint. For the vamp pipeline the highest-leverage adoption is **PersonaLive** for live preview rendering and **FG-Portrait** as the closest existing template for ARKit-blendshape-driven editable portrait animation.

## Audio-driven branch

| Method | Year | Venue | Backbone | FPS | Hardware | Identity | License | Code |
|--------|------|-------|----------|-----|----------|----------|---------|------|
| EMO | 2024 | ECCV | UNet diffusion | offline | — | — | research-only | partial |
| Hallo | 2024 | arXiv | UNet diffusion | ~0.05× RT (20.93 s for 1 s) [1] | A100 | — | MIT | yes |
| Sonic | CVPR 2025 | CVPR | UNet diffusion + global audio attn | not reported | — | not reported in abstract | research | yes [8] |
| Hallo2 | ICLR 2025 | ICLR | UNet diffusion + super-res | offline | — | — | MIT | yes [9] |
| OmniHuman-1 / 1.5 | 2025 | arXiv | DiT, omni-conditions | not reported (cloud-served) | not disclosed | not reported | closed (ByteDance) | API only [6] |
| Teller | CVPR 2025 | CVPR | AR transformer (FMLG) + ETM | **25 FPS**; 0.92 s per 1 s | not specified | not reported | research | not yet released |
| TalkingMachines | 2025 | arXiv | 18 B DiT distilled to 2 steps | "real-time", number not in abstract | multi-GPU disagg | not reported | research | demo page only |
| AvatarSync | 2025 | arXiv | Autoregressive | not reported | — | new SOTA on HDTF (English) | research | tbd |
| RAP | 2026 | arXiv | Video-DiT + hybrid audio attn | "real-time" | not specified | not reported | research | tbd |
| Hallo-Live | 2026 | arXiv | Async dual-stream DiT + HP-DMD | **20.38 FPS, 0.94 s latency** | 2× H200 | not reported | research | tbd |

**Lineage.** EMO/Hallo (UNet-LDM, offline) → Sonic (better audio attention) → Hallo2 (length + 4K) → OmniHuman (DiT scale + omni-conditions) → Teller / TalkingMachines (distillation + AR for streaming) → Hallo-Live (joint A/V streaming with preference distillation). Two architectural splits matter: (1) **autoregressive vs diffusion-step-distilled** (Teller / AvatarSync vs TalkingMachines / Hallo-Live) — both end up at ~20–25 FPS but AR is simpler causally; (2) **single-stream lip-driven vs joint A/V generation** (most methods vs Hallo-Live) — joint generation reduces articulation lag via Future-Expanding Attention [4].

**Current frontier (audio-driven).** Teller is the best documented real-time AR system; Hallo-Live is the most complete joint streaming system but needs 2× H200 (not consumer); RAP claims real-time on a single GPU with a hybrid attention DiT but lacks numeric specifics in the abstract. OmniHuman remains the ceiling for *quality* but is API-only.

## Video / landmark-driven branch

Already covered in detail in `2026-05-03-neural-deformation-for-blendshape-control-topic.md`. Frontier rows for cross-reference:

| Method | Year | Venue | Driver | FPS | Notes |
|--------|------|-------|--------|-----|-------|
| LivePortrait | 2024 | arXiv | implicit kp | 12.8 ms/frame ≈ 78 FPS (RTX 4090) | KwaiVGI; baseline |
| AdvancedLivePortrait | 2024 | community | landmark + slider UI | same | ComfyUI-native |
| HunyuanPortrait | 2025 | arXiv | implicit motion latent | not separately reported, > LivePortrait quality | Tencent |
| X-NeMo | 2025 | arXiv | disentangled latent attn | not reported | best at extreme + nuanced motion transfer [10] |
| FG-Portrait | CVPR 2026 | CVPR | parametric 3D head + 3D flows | not reported | **first to expose user-editable expression/pose handles via 3D flow encoding** [11] |

**Frontier (video-driven).** LivePortrait still wins on raw FPS and adoption; HunyuanPortrait wins on motion expressiveness; X-NeMo wins on cross-identity extreme-motion transfer; FG-Portrait wins on *editability* — it lets you author the driving signal as 3D head parameters rather than capture a video, which is the bridge to ARKit / FLAME control.

## Streaming / infinite-length methods

PersonaLive (CVPR 2026) is the load-bearing entry [5]. Three primitives:

1. **Hybrid implicit signals** — implicit facial representations + 3D implicit keypoints for image-level expressive motion control.
2. **Fewer-step appearance distillation** — student model that retains identity quality at 2–4 sampling steps.
3. **Autoregressive micro-chunk streaming** — sliding training + historical keyframe mechanism to keep identity stable across unbounded-length AR rollout.

Reported: 7–22× speedup over diffusion-based portrait animators, runs on a single 12 GB GPU, TensorRT path adds ~2× more, RTX 50-series (Blackwell) supported with `xformers` disabled. Apache-2.0; 2.6 k stars on GitHub; community ComfyUI node by `okdalto` [5]. Hallo-Live is the audio-driven counterpart but needs 2× H200 — not consumer-streamable yet.

## Hybrid / parameter-driven (FLAME, ARKit, 3D flow)

Sparse but emerging.

- **FG-Portrait** computes 3D flows directly from a parametric 3D head model (not a learned encoder), then conditions a diffusion model via 3D flow encoding + depth-guided sampling [11]. This gives user-specified pose and expression without retraining and is the closest published template for plumbing ARKit/FLAME coefficients into a diffusion-quality renderer.
- **OmniHuman-1.5** accepts text + audio + pose jointly and is trained weak-to-strong across them, but pose here means body keypoints rather than ARKit blendshapes [6].
- **NVIDIA Audio2Face-3D** (arXiv 2508.16401) maps audio to FLAME / ARKit blendshape coefficients but is *not* a portrait renderer — it produces coefficients for downstream rigs (Metahuman etc.). Useful as the *control* end of a two-stage pipeline (audio → blendshape coefs → portrait animator).

No published method yet takes ARKit-52 directly as primary input to a real-time photorealistic portrait animator. FG-Portrait is the closest, by going through 3D-flow as an intermediate.

## Performance benchmarks

Only numbers actually reported in the source materials. Independent reproductions are scarce; treat arXiv-reported numbers as upper-bound author claims.

| Method | FPS | Latency | Hardware | Resolution | Source |
|--------|-----|---------|----------|------------|--------|
| Hallo (2024 baseline) | ~0.05× RT | 20.93 s / 1 s video | A100 | 512² | [1] |
| Teller | 25 | 0.92 s / 1 s video | not specified | not specified | [1] |
| Hallo-Live | 20.38 | 0.94 s | 2× H200 | not specified | [4] |
| LivePortrait | ~78 | 12.8 ms/frame | RTX 4090 | 512² | external benchmark |
| VASA-1 | 40 | not reported | not specified | 512² | [7] |
| PersonaLive | not numerically stated; "7–22× faster than diffusion baselines" | streaming | single 12 GB GPU (RTX 3060+) | not stated | [5] |
| Sonic / Hallo2 / RAP / AvatarSync / OmniHuman / TalkingMachines | **not reported in abstract** | — | — | — | — |

**Identity preservation (CSIM, ArcFace cosine).** VividTalk reports 0.916 [7]; VASA-1 and most CVPR 2025 entries report > 0.9 on HDTF. Above ~0.9 the metric stops differentiating — visible identity drift is now driven by long-rollout AR error accumulation, not single-frame fidelity. PersonaLive's streaming keyframe mechanism and Hallo-Live's HP-DMD distillation are both targeted at exactly that long-rollout failure mode.

## Adoption signals

| Signal | Methods leading |
|--------|-----------------|
| GitHub stars | LivePortrait (~13 k), Sonic (~2 k+), PersonaLive (2.6 k as of 2026-05) [5][8] |
| ComfyUI nodes | LivePortrait (Kijai), AdvancedLivePortrait, ComfyUI-PersonaLive (okdalto), HunyuanPortrait (community) |
| HuggingFace weights | Sonic, Hallo2, PersonaLive (huaichang/PersonaLive), OmniHuman (no weights — API only) |
| Production deployment | OmniHuman → Dreamina/Seedance (ByteDance), LivePortrait → many VTuber stacks |
| Open license | LivePortrait (MIT), Sonic (research), Hallo2 (MIT), **PersonaLive (Apache-2.0)** [5] |

## Recommended priority for vamp-interface pipeline

After LivePortrait sanity check (already planned), in order:

1. **PersonaLive** — replaces LivePortrait as the streaming preview backbone. Apache-2.0, runs on our hardware, ComfyUI node exists, infinite-length matters for any "watch the face react as you slide a knob" UX. Expected 1–2 day integration via the ComfyUI-PersonaLive node. Low risk: same input shape as LivePortrait (one source image + driving signal).
2. **FG-Portrait** — *the* paper to read end-to-end before designing our ARKit-driven editable face renderer. The 3D-flow trick converts FLAME parameters into a motion field the diffusion model can consume, sidestepping the need to retrain a face decoder on ARKit coefficients. Even if we don't adopt the codebase, the architecture is the most relevant published prior art for a parameter-driven editable portrait animator.
3. **Teller** (when code drops) — for the case where we want to drive faces from audio (e.g. read sus_factors as a TTS narration that animates the face) at 25 FPS without diffusion-step costs. AR formulation is also conceptually closer to our blendshape-distillation thread (`mediapipe_distill`) than diffusion.

Skip for now: OmniHuman (closed), Hallo-Live (2× H200), AvatarSync / RAP (no code yet), X-NeMo / HunyuanPortrait (already covered in the topic doc and overlapping with LivePortrait functionally).

## Caveats / unverified claims

- All FPS / CSIM numbers above are arXiv-author-reported. Independent benchmarks across methods on a common hardware configuration do not yet exist — the "Talking-Head Generation in Practice" review paper [7] is the closest but is non-exhaustive.
- PersonaLive's "7–22× speedup" is vs unspecified diffusion baselines; the 12 GB VRAM number is from the GitHub README and not separately verified.
- FG-Portrait was submitted March 2026 and is recent enough that no community reproductions exist.
- DeX-Portrait, MMFace-DiT exist in CVPR 2026 paper roundups but have no arXiv preprint surfaced as of 2026-05-03 — track via the proceedings when published.

## Sources

1. [Teller arXiv 2503.18429](https://arxiv.org/abs/2503.18429)
2. [TalkingMachines arXiv 2506.03099](https://arxiv.org/abs/2506.03099)
3. [RAP arXiv 2508.05115](https://arxiv.org/abs/2508.05115)
4. [Hallo-Live arXiv 2604.23632](https://arxiv.org/abs/2604.23632)
5. [PersonaLive arXiv 2512.11253](https://arxiv.org/abs/2512.11253) / [GitHub GVCLab/PersonaLive](https://github.com/GVCLab/PersonaLive) / [project page](https://personalive.app/) / [ComfyUI-PersonaLive](https://github.com/okdalto/ComfyUI-PersonaLive)
6. [OmniHuman-1 arXiv 2502.01061](https://arxiv.org/abs/2502.01061) / [omnihuman-lab.github.io](https://omnihuman-lab.github.io/)
7. [Talking-Head Generation in Practice (OpenReview)](https://openreview.net/pdf?id=ns3TgZYQTZ); [VASA-1 NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/014fe398da515cd552fa6e1f33e0565e-Paper-Conference.pdf)
8. [Sonic arXiv 2411.16331](https://arxiv.org/abs/2411.16331) / [GitHub jixiaozhong/Sonic](https://github.com/jixiaozhong/Sonic)
9. [Hallo2 arXiv 2410.07718](https://arxiv.org/abs/2410.07718)
10. [X-NeMo arXiv 2507.23143](https://arxiv.org/abs/2507.23143)
11. [FG-Portrait arXiv 2603.23381](https://arxiv.org/abs/2603.23381)
12. [AvatarSync arXiv 2509.12052](https://arxiv.org/abs/2509.12052)
13. [Awesome-Talking-Head-Synthesis (Kedreamix)](https://github.com/Kedreamix/Awesome-Talking-Head-Synthesis)
14. [talking-face-arxiv-daily (liutaocode)](https://github.com/liutaocode/talking-face-arxiv-daily)
15. [Audio2Face-3D arXiv 2508.16401](https://arxiv.org/pdf/2508.16401)
16. [CVPR 2026 papers](https://cvpr.thecvf.com/virtual/2026/papers.html)
