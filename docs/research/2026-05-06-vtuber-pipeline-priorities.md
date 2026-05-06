---
status: live
topic: personalive-acceleration
---

# vtuber pipeline priorities — two regimes, four products

Date: 2026-05-06

## Why this doc exists

The "should we pivot from PersonaLive?" framing collapsed two unrelated products into a single decision. Pulling them apart:

- **Regime A — realtime puppeteering.** Live performance, OBS streaming, interactive viz. Hard FPS budget (25+ on a 4080).
- **Regime B — static-portrait authoring.** Slider work, hero shots, corpus generation. Quality dominates; seconds-to-minutes per shot is fine.

These two regimes have different backbones, different acceptance criteria, and different failure modes. They should be evaluated independently.

## The four products

| Product | Regime | Driving signal | Backbone class | Status |
|---|---|---|---|---|
| RGB-driven vtuber (PersonaLive) | A | RGB camera + facemesh | SD1.5 + RefNet, 4-step distill | Bridge done; OBS pipeline in progress |
| ARKit-driven vtuber (PersonaLive) | A | ARKit blendshapes (iPhone) | Same as above + ARKit→motion bridge | Bridge trained (`student_v3_lam10`); OBS plumbing pending |
| RGB-driven vtuber (LivePortrait) | A | RGB camera | Warp-based GAN | Sanity test pending |
| Static-portrait slider authoring | B | Slider parameters | Flux + LoRA / FluxSpace | Active thread (FluxSpace ComfyUI nodes) |

## Where common confusions land

**"Is LivePortrait better than our LoRA work?"** Wrong axis. They serve different products. LoRA/FluxSpace = Regime B authoring. LivePortrait = Regime A puppeteering. The slider corpus is *upstream* of the vtuber: it produces the reference image the vtuber drives. There is no path that turns Flux+LoRA into a 25-FPS vtuber.

**"Should we replace PersonaLive's UNet with a DiT (FlashPortrait/Wan-Animate)?"** Only meaningful in Regime B. FlashPortrait benchmarks at ~0.83 FPS render speed on H100 cluster hardware (paper Table 1, 720s for a 20s video) — it lost the speed contest by a wide margin. It's a long-form quality proposal for offline rendering, not a realtime proposal. Do not consider it for Regime A.

**"Does PersonaLive's FFHQ-collapse-on-stylized-refs problem block everything?"** No — it blocks Regime A on stylized references. Regime B (static authoring) doesn't drive a reference image; it generates it. Regime A on photoreal references is unaffected. Decoupling these means the FFHQ-collapse problem is a Regime-A-stylized-only concern.

## The two questions actually live on the user's mind

1. **"Can we repeat our MediaPipe experiments with LivePortrait?"** — i.e. can we build an ARKit-bridge analog for LivePortrait? Open. LivePortrait drives via implicit keypoints, not blendshapes; an ARKit→implicit-keypoint bridge is a fresh research problem analogous to but distinct from the PersonaLive bridge work. Gated on LivePortrait sanity passing first.
2. **"Can we up the quality of PersonaLive somehow?"** — open thread. Levers: γ/α/β runtime knobs (no retrain), discriminator swap (kills FFHQ collapse, retrain Stage 2), 1-step distill via Hyper-SD/DMD2 (speed gain + new discriminator decision). See `2026-05-06-pivot-decision-personalive-vs-liveportrait.md` for the four UNet-surgery middle paths.

## Prioritization (confirmed 2026-05-06)

1. **Finish RGB realtime OBS pipeline for PersonaLive.** Current foundation. Everything else reuses this scaffolding.
2. **LivePortrait sanity + benchmarks.** Two gates: (a) FPS on 4080 ≥ 25, (b) stylized-OOD identity holds on the zombie/orc/duck/demon set + photoreal cells. Both must pass before building any LivePortrait product.
3. **If gates pass → RGB realtime OBS for LivePortrait** as a second product (Regime A alternative backbone).
4. **In parallel with item 1: ARKit OBS pipeline for PersonaLive.** Bridge already exists (`student_v3_lam10`); work is OBS plumbing reusing item 1's scaffolding. Cheapest parallel item.

Explicitly *not* on the priority list right now:

- ARKit→LivePortrait bridge — fresh research problem, gated on item 2 passing.
- DiT replacement (FlashPortrait/Wan-Animate) for Regime A — disqualified on speed.
- DiT replacement for Regime B — interesting but not the user's current question; FluxSpace thread continues.

## Acceptance gates per product

**RGB-PersonaLive OBS pipeline:** sustained ≥15 FPS at 512² on 4080 (matches the paper's claimed 4090 number scaled down); identity stable on photoreal refs over a 5-minute take; no observable rotation/crop bugs.

**ARKit-PersonaLive bridge — validation gate before OBS plumbing.** Earlier framing of "OBS in parallel" was overoptimistic. The bridge (`student_v3_lam10`) is trained but unaudited for realtime use; the calibration-blind-spot history (yaw L2 was partially yaw-symmetric and picked the wrong sign initially) means we can't trust unaudited axes. Required before OBS work starts:
  - Sign-agreement test on all three Euler axes (yaw/pitch/roll) against ground-truth ARKit recordings.
  - Head-attenuation-at-extremes diagnosis from `render_metrics.parquet` (3-amp decomposition).
  - Bridge inference latency benchmark on 4080 (target <5 ms/frame).
  - End-to-end live path: iPhone → ARKit stream → script → PersonaLive → OBS, validated on a short take.
  - Only after these pass: build the OBS plumbing. The plumbing itself is cheap; trusting the bridge isn't.

**LivePortrait sanity (gating its OBS build):** ≥25 FPS at 512² on 4080 using **FasterLivePortrait with TensorRT engines** (not vanilla repo; see "LivePortrait descendants" below); stylized refs maintain reference identity over the 600-frame yaw stress; photoreal refs match or beat PersonaLive teacher_full on the 3 curated cells.

## LivePortrait descendants — the 2-year community survey

Question: did the community produce a "next-gen LivePortrait"? **No single dominant successor.** The field forked into three non-overlapping branches:

**Branch 1 — engineering descendants (actual realtime successors).** Same warp-based core, productionized.

- [FasterLivePortrait](https://github.com/warmshao/FasterLivePortrait) — ONNX/TensorRT fork. **30+ FPS on RTX 3090 with TRT** (incl. pre/post). Adds animal, multi-face, region-driving. v2.0 Jan 2025. The de-facto realtime fork — what real users run.
- [JoyVASA](https://github.com/jdh-algo/JoyVasa) — audio-driven motion generator that plugs into LivePortrait's keypoint space. Diffusion for motion only, not for pixels.

This is *the* realtime successor — the realtime ecosystem is FasterLivePortrait + JoyVASA. 30 FPS on 3090 implies ~40+ FPS on 4080, comfortably above our Regime A bar. It is "same model, productionized," not "new architecture."

**Branch 2 — quality successors via diffusion (NOT realtime; these are PersonaLive's siblings, not LivePortrait's).** The community pivoted upmarket: X-NeMo / X-Portrait2 (ByteDance, ICLR 2025), SkyReels-A1 (Skywork), HunyuanPortrait (Tencent), FantasyPortrait (Alibaba, CVPR 2026), Hallo3 (Fudan, CVPR 2025), EchoMimic V2/V3 (Ant Group), DeX-Portrait (Dec 2025), FactorPortrait (Dec 2025), FlashPortrait. All diffusion or DiT-based. All non-realtime. All Regime B candidates if at all.

**Branch 3 — hybrids (unproven).** RAP (Aug 2025) and Teller (Mar 2025) claim "real-time" with Video DiTs but don't publish FPS in their abstracts; "realtime" in DiT papers usually means "approachable," not 30 FPS. Mixed code availability. Worth a follow-up read but not a candidate for a sanity test yet.

**Conclusion for our use case:** the LivePortrait sanity test should specifically run **FasterLivePortrait with TRT engines on the 4080**, not vanilla LivePortrait. Vanilla benchmarks worse and isn't where current FPS numbers come from.

## Streaming-infrastructure techniques inventory (revisit when building OBS pipeline)

The audio-driven papers (RAP, Teller, RAIN) and long-form DiT papers (FlashPortrait) are wrong-driver for our use case but solve **the streaming infrastructure problem we will inevitably hit** when running camera → render → OBS for minutes-to-hours of continuous output. Driver and streaming infrastructure are orthogonal — we can pick a warp-based driver (LivePortrait family) and still need the diffusion-side long-stream stability tricks once frame-1 vs frame-10000 identity drift starts showing.

Catalog to re-consult when building the OBS path:

| Paper | Technique | Why it matters at minutes-to-hours runtime |
|---|---|---|
| [RAIN](https://arxiv.org/abs/2412.19489) | 1D attention blocks; long-range token attention at low memory | Stream pixels without buffering hitches |
| [RAP](https://arxiv.org/abs/2508.05115) | Static-dynamic latent inheritance; hybrid full+window attention | Prevent identity drift / error accumulation across long streams |
| [Teller](https://arxiv.org/abs/2503.18429) | 200 ms chunk processing budget; AR transformer over discrete motion tokens | Sub-200 ms end-to-end latency design; quantized motion provides clean rejoin points |
| FlashPortrait | Sliding-window with weighted blending; Taylor-expansion latent prediction | Smooth window transitions; cheap step skipping |
| FasterLivePortrait | TRT engine fusion, multi-face, region-driving | Production realtime engineering reference |

Conclusion (May 2026): no published method solves "DiT-quality realtime portrait animation driven by performer's face." The unicorn doesn't exist yet. Audio-driven + diffusion is the closest thing the literature offers, but the driver mismatch disqualifies it for our product. When we inevitably hit long-stream stability problems on the warp-based path, harvest techniques from these papers rather than trying to swap backbones.

## Open architectural questions (parked)

- Stylized vtuber path. If LivePortrait fails OOD too (different mechanism but possible), the cheapest path is PersonaLive 1-step distill with a non-FFHQ discriminator. Do not build for this until both PersonaLive teacher_full and LivePortrait have failed the stylized set.
- Slider × LivePortrait. Warp-based pipelines have no attention surface to inject FluxSpace δs. Slider work stays Regime B regardless of what wins Regime A.
- DiT-class backbone for Regime B. Worth a separate doc if the slider thread runs out of room on Flux.

## Companions

- `2026-05-06-pivot-decision-personalive-vs-liveportrait.md` — UNet-surgery analysis, the four middle paths, sunk-cost audit.
- `2026-05-06-rendering-stack-replacement-options.md` — survey of speed/quality candidates.
- `2026-05-05-personalive-architecture-notes.md` — runtime control surface (γ/α/β), conditioning channels.
- `_topics/personalive-acceleration.md` — topic index (update with this doc in same commit).
