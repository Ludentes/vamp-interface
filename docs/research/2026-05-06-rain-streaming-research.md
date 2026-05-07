---
status: live
topic: arkit-bridge
---

# RAIN-style streaming for PersonaLive — research notes

## Why this question

About to wire iPhone Live Link Face → PersonaLive → v4l2loopback → OBS. PersonaLive's `Pose2VideoPipeline_Stream.__call__` is window-batched (`temporal_window_size=4`, `temporal_adaptive_step`-cohort denoising over multiple windows in one Python call), not a true `step(per_frame_pose) -> frame` interface. On RTX 5090, per-frame inference floor is ~50 ms; the batched call shape pushes usable end-to-end latency to ~1.5 s. Question: is there a known technique that converts batched-window diffusion video into a streaming step() loop without retraining?

Short answer: no inference-only conversion exists. The streaming techniques in this space are training-time architectural choices baked into the model. PersonaLive is *itself* already built on RAIN + StreamDiffusion (confirmed in `PersonaLive/README.md` Acknowledgements line 256). The RAIN-style mechanism is already present inside `pipeline_pose2vid.py` — what's missing is exposure of the loop's internal state (`motion_bank`, `noise_latents`, scheduler buffers) as instance fields you can drive one window at a time.

## Findings

### RAIN — Real-time Animation of Infinite Video Stream

- **Title / authors / venue**: Pscgylotti et al., 2024. Preprint Dec 2024.
- **arXiv**: https://arxiv.org/abs/2412.19489 ; project page https://pscgylotti.github.io/pages/RAIN/ ; **code** https://github.com/Pscgylotti/RAIN
- **Mechanism**: pipeline-style staggered denoising. Latent buffer holds frames at *different* noise levels simultaneously (a "denoising cohort"). Each forward pass denoises the whole buffer one step; the front-most cohort exits as a clean frame, a fresh noisy frame enters at the back. Adds 1D temporal attention blocks that span the whole multi-noise-level buffer so the model sees long-range context within one batched forward.
- **Reported numbers**: 18 fps, ~1.5 s latency, 512×512, single RTX 4090. (This is identical to PersonaLive's window-batched latency on a stronger card — because PersonaLive inherits the recipe.)
- **Help convert PersonaLive's batched window into a true step()?** **Partial / already applied.** RAIN *is* the streaming recipe. The 1.5 s figure is intrinsic to the cohort design: latency = `temporal_adaptive_step × temporal_window_size / fps`. There's nothing in RAIN that gives you sub-cohort streaming; cohort granularity is the unit.
- **Training requirement**: fine-tune the SD UNet for a few epochs with the new 1D attention blocks. Not inference-only.

### StreamDiffusion — pipeline-level batched denoising for image streams

- **Title / authors**: Kodaira et al., arXiv 2312.12491 (Dec 2023).
- **arXiv**: https://arxiv.org/abs/2312.12491 ; **code**: https://github.com/cumulo-autumn/StreamDiffusion
- **Mechanism**: StreamBatch — denoise N frames at staggered noise levels in one UNet batch. RCFG (residual CFG), input/output queues, autoencoder fp16/TRT, pre-compute prompt embeds. Image-only; no temporal attention.
- **Helps?** This is the substrate PersonaLive's streaming mode also inherits. Same conclusion: already applied. The `temporal_window_size × temporal_adaptive_step` cohort *is* the StreamBatch.
- **Code**: yes, mature.

### Knot Forcing — autoregressive diffusion for real-time portrait animation

- **arXiv**: https://arxiv.org/html/2512.21734 (Dec 2025). Project: https://humanaigc.github.io/knot_forcing_demo_page/
- **Mechanism**: fine-tunes Wan2.1-T2V-1.3B with masked inpainting, then distills to a causal model via Self Forcing. Constant per-chunk latency via fixed-length sliding window + KV cache.
- **Numbers**: 17.5 fps, "consumer GPU"; explicit ms latency not given.
- **Helps?** **No** for our V1. Architecturally tied to Wan2.1; requires both fine-tune and Self-Forcing distill; no public code at time of search.

### REST — diffusion-based real-time end-to-end streaming talking head

- **arXiv**: https://arxiv.org/html/2512.11229 (Dec 2025, contemporaneous with PersonaLive).
- **Mechanism**: ID-Context Cache (ID-Sink reference embeds + Context-Cache concatenating prior chunk K/V) plus Asynchronous Streaming Distillation (non-streaming teacher with async noise schedule supervising a streaming student via contrastive + smoothness losses).
- **Helps?** **No.** Architecturally baked into REST's bespoke Streaming A2V-DiT (28 transformer blocks). Authors give no transfer recipe; no code.

### CausVid / Self-Forcing / MemRoPE — autoregressive video diffusion with KV cache

- CausVid (CVPR 2025, https://causvid.github.io, https://github.com/tianweiy/CausVid): distills bidirectional teacher into causal student; 9.4 fps streaming with KV cache.
- Self-Forcing (https://self-forcing.github.io/static/self_forcing.pdf): rolling KV cache, sub-second latency.
- MemRoPE (https://memrope.github.io): training-free fixed-size KV cache via memory-token EMAs.
- **Helps?** **No** for our V1. CausVid + Self-Forcing are training-time distill recipes on a different backbone (typically Wan/CogVideoX class). MemRoPE is the only training-free entry but targets DiT KV-cache, not the cross-window cohort issue PersonaLive has.

### Teller / RAP / READ / TalkingMachines — adjacent streaming talking-head DiTs

- Teller (CVPR 2025, arXiv 2503.18429), RAP (arXiv 2508.05115), READ (arXiv 2508.03457), TalkingMachines (arXiv 2506.03099). All audio-driven, all training-time architectural commits. None is a wrapper that retrofits an existing batched pipeline.

## What PersonaLive's `Pose2VideoPipeline_Stream` actually looks like

Inspected `/home/newub/w/PersonaLive/src/pipelines/pipeline_pose2vid.py`:

- `__call__` (line 440 and 748 — two variants) takes the *full* pose sequence and runs `windows + temporal_adaptive_step - 1` cohort iterations in one Python call.
- Internal state that would need to become instance fields for true streaming: `motion_bank` (line 589/840), `noise_latents` (line 838), the per-cohort timestep schedule (`init_timesteps`, line 528/836), and the running `mid_latents` from the previous window.
- `temporal_window_size = 4`, `temporal_adaptive_step` defaults to 4 in offline configs → cohort of 16 frames in flight, exit cadence of 4.
- This is **literally** the RAIN mechanism, plus motion_bank as a learned long-range memory.

## Verdict

**Don't search further for a RAIN-style retrofit — PersonaLive already is one.** The cohort latency floor (~temporal_window_size × temporal_adaptive_step × per-step time / parallelism) is intrinsic to the recipe. Published streaming-diffusion-portrait work that goes lower (Knot Forcing 17.5 fps; REST; Self-Forcing) does so by training a different architecture, not by wrapping an existing batched pipeline.

**V1 → V2 path is the only realistic one:**

1. **V1 (ship now)**: call the existing `__call__` on a sliding cohort of poses, accept the ~1.5 s latency as the streaming budget. This is what the RAIN paper itself reports on 4090; on 5090 we should beat that proportionally to UNet step time, possibly to ~0.8–1.0 s.
2. **V2 (manual refactor)**: split `Pose2VideoPipeline_Stream.__call__` into `init_stream(reference_image, first_pose_chunk)` and `step(pose_chunk_of_size_temporal_window_size) -> frames_of_size_temporal_window_size`, hoisting `motion_bank`, `noise_latents`, `mid_latents`, and scheduler index to instance state. The math is unchanged; you're just exposing the inner `for i in range(windows + ...)` loop as an external driver loop. Output cadence stays at one 4-frame block per `step()` call.
3. **Don't pursue V3** (sub-window streaming, finer than `temporal_window_size`) without a research investment — that requires either dropping the temporal attention spans or retraining with `temporal_window_size=1`, both of which break what PersonaLive learned.

The latency budget for V2 is exactly: `(temporal_adaptive_step) × temporal_window_size × frame_time` of "first-frame latency", then steady-state `temporal_window_size × frame_time` per `step()` output. With `frame_time≈50 ms` on 5090, that's ~800 ms first-frame and ~200 ms per 4-frame block (= 20 fps continuous), assuming `temporal_adaptive_step=4`. Drop `temporal_adaptive_step` to 2 if quality holds → ~400 ms first-frame.

## Sources

- RAIN — https://arxiv.org/abs/2412.19489 ; https://github.com/Pscgylotti/RAIN ; https://pscgylotti.github.io/pages/RAIN/
- StreamDiffusion — https://arxiv.org/abs/2312.12491 ; https://github.com/cumulo-autumn/StreamDiffusion
- PersonaLive — https://arxiv.org/abs/2512.11253 ; https://github.com/GVCLab/PersonaLive
- Knot Forcing — https://arxiv.org/html/2512.21734
- REST — https://arxiv.org/html/2512.11229
- CausVid — https://causvid.github.io ; https://github.com/tianweiy/CausVid
- Self-Forcing — https://self-forcing.github.io
- MemRoPE — https://memrope.github.io
- Teller — https://arxiv.org/abs/2503.18429
- RAP — https://arxiv.org/html/2508.05115
- READ — https://arxiv.org/pdf/2508.03457
- TalkingMachines — https://arxiv.org/html/2506.03099v1
