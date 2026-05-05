# Topic: neural-deformation-control

Living index for the neural-image-deformation (LivePortrait lineage) thread as an alternative / complement to LoRA-slider blendshape control on Flux portraits.

## Current belief (2026-05-04)

**PersonaLive is the default path for animated photoreal portraits.** Phase 2
3×6 grid (`2026-05-04-personalive-default-decision.md`) on Flux anchors
driven by ARKit captures showed identity preservation, expression tracking,
and skin texture all clearly above what we got from months of slider/LoRA/Flux
work. Quantitative confirmation: ArcFace cosine never below buffalo_l τ=0.40
across 900 frames; smile macro-correlation up to 0.81. Slider stack remains
canonical for **offline parameter authoring** (atom measurement, axis
isolation, demographic factor work) and for **static** Flux-portrait
generation; it is no longer the primary path for animated output.

**Earlier complementary-mechanism stance (2026-05-03, superseded but kept
for context):** neural deformation was a complementary mechanism to LoRA
sliders, with the bridge `ARKit-52 → 12 ALP sliders` providing partial
coverage and the LoRA path filling the rest (squint included). Phase 2
showed PersonaLive subsumes most of what that bridge was meant to do for
the animated-output use case. The ALP bridge work is no longer load-bearing;
the PersonaLive motion-API + the iPhone-pipeline plan
(`2026-05-03-iphone-pipeline-unified-plan.md`) supersede it.

## Key dated docs

- [2026-05-03-neural-deformation-synthesis.md](../2026-05-03-neural-deformation-synthesis.md) — combined decision doc. Read this first.
- [2026-05-03-neural-deformation-for-blendshape-control-topic.md](../2026-05-03-neural-deformation-for-blendshape-control-topic.md) — paper-side research (calc_fe internals, IM-Animation / FG-Portrait disentanglement, diffusion-internal motion conditioning).
- [2026-05-03-neural-deformation-for-blendshape-control-practical.md](../2026-05-03-neural-deformation-for-blendshape-control-practical.md) — install recipe, slider table, two-hour build plan, fallback escalation order.
- [2026-05-03-slider-thread-recipe.md](../2026-05-03-slider-thread-recipe.md) — companion parking note for the LoRA-slider thread. The two paths are designed to coexist.
- [2026-05-03-talking-heads-2025-2026-survey.md](../2026-05-03-talking-heads-2025-2026-survey.md) — broader audio-driven + streaming SOTA survey (Teller, PersonaLive, Hallo-Live, OmniHuman, RAP, FG-Portrait). Adoption recommendations for the vamp pipeline. 10 PDFs in `docs/papers/`.
- [2026-05-03-liveportrait-x-nemo-analysis.md](../2026-05-03-liveportrait-x-nemo-analysis.md) — architectural deep dive on LivePortrait (pure deformation, GAN-trained, 12.8 ms/frame) and X-NeMo (1D motion bottleneck + cross-attn, the recipe PersonaLive borrows). Includes side-by-side loss-zoo comparison vs our slider stack.
- [2026-05-03-xformers-blackwell-pivot.md](../2026-05-03-xformers-blackwell-pivot.md) — first gotcha doc. xformers source-build for sm_120 is a no-op (kernels structurally removed); use flash-attn or SDPA. Applies to PersonaLive AND ComfyUI on the 5090. Superseded in scope by the saga doc below but kept for the structural-restructuring detail.
- [2026-05-03-xformers-flashattn-saga.md](../2026-05-03-xformers-flashattn-saga.md) — **full saga.** Six-attempt chain ending with the verdict that xformers + flash-attn buys ≤10% over SDPA on Blackwell torch 2.11+cu128 (SDPA already dispatches to FA-2 internally for fp16). Captures three traps: structural kernel removal, broken local source-build masking the wheel, fp32 enable-probe in diffusers. Recommends moving to TRT.
- [2026-05-03-phase1-personalive-sanity-plan.md](../2026-05-03-phase1-personalive-sanity-plan.md) — **next action.** Concrete Phase 1 plan: smoke + Flux test + motion-API probe (the probe unblocks the Windows Phase 3 extraction script).
- [2026-05-03-iphone-pipeline-unified-plan.md](../2026-05-03-iphone-pipeline-unified-plan.md) — **active plan.** Reframes the work around building the missing pipeline (pre-flight + bridge + stabilizer + iPhone client) that makes neural deformation the photoreal counterpart to VTube Studio. Vamp-interface's offline parameter-driven deformation is a strict subset and ships at the bridge milestone (Phase 3).
- [2026-05-04-live-link-face-capture-format.md](../2026-05-04-live-link-face-capture-format.md) — driver-clip source decision for Phase 2. Live Link Face (Epic, free iOS) records ARKit-52 + 1080p MOV locally; take folder exports `frame_log.csv` (per-frame 52 blendshapes + head/eye rotation) + MOV sharing SMPTE timecode. Same captures double as Phase 3 bridge training data. Expo / custom webui / public dataset alternatives weighed and falsified for this use.
- [2026-05-05-metahuman-animator-depth-format.md](../2026-05-05-metahuman-animator-depth-format.md) — MHA-mode take format research. `depth_data.bin` is undocumented Oodle-Kraken-compressed AVDepthData; only public decoder is unmaintained. MHA solver emits ~130 proprietary `CTRL_expressions`, not ARKit-52. **Verdict: discard MHA takes, always record in ARKit mode.**
- [2026-05-04-personalive-default-decision.md](../2026-05-04-personalive-default-decision.md) — **strategic decision.** Phase 2 3×6 grid (3 Flux anchors × 6 ARKit-driven clips) → PersonaLive becomes the default path for animated photoreal portraits, replacing the slider/LoRA/Flux stack for that use case. Quantitative confirmation: ArcFace floor 0.43, never below τ=0.40 same-person; smile macro-corr up to 0.81. Slider stack stays canonical for offline parameter authoring + static Flux work.
- [2026-05-04-video-super-resolution-survey.md](../2026-05-04-video-super-resolution-survey.md) — survey of OSS video/face SR models in 2026 for upscaling PersonaLive's 512² output. **Recommendation: GFPGAN v1.4 (Apache-2.0) as first line, SeedVR2-3B (Apache-2.0, ICLR 2026) as quality-ceiling pass.** CodeFormer / KEEP / Upscale-A-Video all blocked by NTU S-Lab non-commercial license — earlier "try CodeFormer first" advice retracted.
- [2026-05-04-realtime-vtuber-pipeline-plan.md](../2026-05-04-realtime-vtuber-pipeline-plan.md) — **realtime VTuber-with-a-real-face plan** (iPhone → Linux 5090 → OBS). Two architectures: A) RGB-driving via PersonaLive's native webcam pipeline (~half a day, 130–200 ms glass-to-glass), B) ARKit-driving via Phase 3 bridge (~3–5 days, 80–130 ms glass-to-glass, smaller motion signal, no identity leak). Recommendation: ship A first.
- [2026-05-04-iphone-to-linux-webcam-transport.md](../2026-05-04-iphone-to-linux-webcam-transport.md) — companion research on iPhone-to-Linux RGB transport options. **Recommendation: NDI HX Camera (free iOS) → OBS DistroAV plugin → v4l2loopback** (~80–200 ms over 5 GHz Wi-Fi). Fallback: Larix Broadcaster → MediaMTX/go2rtc → ffmpeg. Reincubate Camo confirmed dead-end on Linux in 2026; iOS still doesn't expose UVC over USB. Notable side-finding: lived Linux VTuber practice routes ARKit blendshapes (iFacialMocap/Live Link Face), not RGB — parallels Architecture B.
- [2026-05-03-personalive-experiment-plan.md](../2026-05-03-personalive-experiment-plan.md) — superseded by the unified plan, but still authoritative for Phase 1+2 step-by-step procedure and the Phase 3/4 option taxonomy.
- [2026-05-03-loss-zoo-comparison.md](../2026-05-03-loss-zoo-comparison.md) — side-by-side of LivePortrait Stage 1+2 8-term loss soup vs our slider stack (D/B/A/S/P/G/M). Identifies what's load-bearing on each side, what we should port from them (region-conditioned perceptual, equivariance regulariser), what they could port from us (latent-space distilled critics, PGD-robust critic).
- [2026-05-03-interesting-ideas.md](../2026-05-03-interesting-ideas.md) — open threads, gaps in the public landscape, things to come back to. Cross-pollination between our slider stack and the LivePortrait/PersonaLive ecosystem; missing artefacts (ARKit-52→ALP map, FG-Portrait code, identity-drift benchmark); phenomena without theory.

## Background reading

- `~/w/Face-Research/05-neural-deformation.md` — chapter overview of the LivePortrait / FOMM lineage. Don't re-fetch; read locally.

## Open threads

- ARKit-52 → ALP slider bridge has not been written or benchmarked yet.
- Identity-drift micro-benchmark (20 Flux portraits × 5 slider settings, ArcFace cosine threshold 0.7) has not been run.
- Squint-channel empirical fit `squint = f(blink, eyebrow, smile)` has not been tried.
- Compose-LoRA-edit-with-warp test (apply v1k squint LoRA, then ALP smile) — open.
- FG-Portrait / IM-Animation code release status — periodic check-back.
