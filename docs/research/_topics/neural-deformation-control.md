# Topic: neural-deformation-control

Living index for the neural-image-deformation (LivePortrait lineage) thread as an alternative / complement to LoRA-slider blendshape control on Flux portraits.

## Current belief (2026-05-03)

Neural deformation is a **complementary** mechanism to LoRA sliders, not a replacement. Use it for the ~20–25 of ARKit-52 channels that map cleanly onto AdvancedLivePortrait's 12 named sliders (smile, blink, eyebrow, gaze, jaw, pucker, head pose). Keep the LoRA path (v1k bs_only + arc_distill anchor recipe) for the ~25–30 channels with no slider home — squint included, which is the immediate active channel. The two paths fail in opposite directions: LoRA fails "wrong-but-sharp" (classifier fooling, identity drift); warp fails "right-but-degraded" (correct geometry, softer texture). Build the bridge `ARKit-52 → 12 ALP sliders` ourselves (~50 LOC) and accept the partial coverage. **Do not wait** for FG-Portrait / IM-Animation / 2026 paper checkpoints — the architecturally-aligned methods all lack public code as of fetch date.

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
