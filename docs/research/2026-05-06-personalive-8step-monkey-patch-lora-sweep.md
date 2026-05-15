---
status: live
topic: personalive-acceleration
---

# 8-step monkey-patch unlocks LoRA budget on PersonaLive; off-the-shelf SD1.5 LoRAs top out at Ghibli α=3 quality

Date: 2026-05-06

## Headline

PersonaLive's 4-step distilled inference is hardcoded in 6+ places (`pipeline_pose2vid.py:472,778`; `wrapper.py:121`; etc.) with `timesteps=[999,666,333,0]` and `set_step_length(333)`. Off-the-shelf SD1.5 LoRAs are budget-bound at this step count: only 4 forward passes integrate the LoRA's weight delta. We monkey-patched the schedule to vanilla 8-step DDIM (`[999, 874, 749, 624, 499, 374, 249, 124]` + `step_length=None` + `alphas_cumprod` cast to fp16) at runtime, without touching PersonaLive's tree, and ran the Ghibli + Demon Slayer LoRAs in combination with the decoupled-CLIP probe (`--reference_clip`).

**Outcome**: 8-step monkey-patch works mechanically and unlocks off-the-shelf LoRA influence. Ghibli at α=3 with anime CLIP decouple visibly flips the output toward Ghibli style ("almost there, minor artifacts"). Demon Slayer at the same config is dramatically weaker (peak Δ 14.51 vs Ghibli's 33.58) AND visibly more artifacty — magnitude does not track quality across LoRA trainings.

## Pixel-delta sweep (vs 4-step no-LoRA baseline, MySlate_5 yaw stress, frame indices 0/50/150/280)

| Config | f=0 | f=50 | f=150 | f=280 |
|---|---|---|---|---|
| 4-step LoRA α=2 den-only (prior) | — | — | ~15.8 | — |
| 4-step decoupled CLIP alone | 2.19 | 4.77 | 6.78 | 4.63 |
| 4-step compound (decouple + LoRA both α=1) | 5.96 | 10.33 | 12.66 | 10.79 |
| 8-step Ghibli α=2 | 8.61 | 12.19 | 13.55 | 11.46 |
| 8-step Ghibli α=3 (no decouple) | 12.87 | 20.71 | 25.91 | 21.28 |
| **8-step Ghibli α=3 + decoupled CLIP** | **16.84** | **29.50** | **33.58** | **30.80** |
| 8-step Demon Slayer α=3 + decoupled CLIP | 5.48 | 9.76 | 14.51 | 10.18 |

The α=2→α=3 jump on Ghibli at 8-step nearly doubled peak Δ (13.55 → 25.91), confirming the LoRA was unsaturated at α=2. Adding decoupled CLIP on top added another +7.67 at peak, with super-linear stacking compared to the same combo at 4-step (+6.80 at peak in the 4-step compound vs +17.47 at peak in the 8-step combo — 2.6× bigger CLIP-channel contribution). The 8 cross-attention integration passes amplify the CLIP-image embed proportionally.

## Mechanism notes

- The schedule and `step_length` are coupled by the magic number 333. The DDIM step formula does `prev_timestep = t - step_length`. With `step_length=333` and `[999,666,333,0]`, per-step jumps land exactly on the next anchor. Monkey-patching only the schedule (without `step_length=None`) creates a conditioning mismatch: the UNet is told "you're at t=833" while the latent is actually at noise level 666.
- `scheduler.step()` reads `self.alphas_cumprod` directly without dtype casting. `alphas_cumprod` is fp32 by default. In the 4-step ship config jump=1 so the inner loop runs once and the fp32 latents leaving step() get cast back to fp16 via `noise_latents = ... .to(dtype=self.dtype)` at the end of the outer iter. With 8-step `jump=2`, iter j=1 sees fp32 latents from j=0 → fp16 UNet conv_in → `Input type (float) and bias type (c10::Half) should be the same`. Fix: `pipe.scheduler.alphas_cumprod = pipe.scheduler.alphas_cumprod.to(dtype=fp16)` once.
- The pipeline never calls `scheduler.set_timesteps`. We avoid it deliberately because it rebuilds tensors at fp32 and breaks `add_noise`'s dtype contract. We just set `pipe.scheduler.num_inference_steps = N` directly so the `step_length=None` fallback computes `prev_timestep = t - 1000//N`.
- Schedule injection done via `torch.tensor` monkey-patch with strict pattern match (`data == [999, 666, 333, 0]`). Restored after the `pipe(...)` call. Scoped, surgical, no PersonaLive tree touch.

Implementation in `scripts/apply_bridge_to_personalive.py`, auto-engages when `--num_inference_steps != 4`. Cost in wall-time: ~31s for 300 frames at 8-step (vs ~31s at 4-step — they're identical because the inner loop runs `jump` UNet calls per window in both cases, and `windows × jump` is constant when `temporal_adaptive_step` is constant).

## LoRA design axis findings

Same offset family (rank 128 / alpha 128), same naming convention (Kohya-diffusers), same 192/264 attention pair match against PersonaLive's UNets — but Ghibli pulls 2.3× harder than Demon Slayer at α=3. **Per-LoRA training quality matters more than nominal rank/alpha for both magnitude and artifact behavior.** Magnitude does not track quality.

Probed but not tested: AnimeAnything SD1.5 (civitai 548391) is a LoCon with A1111 naming convention. Our `_kohya_to_diffusers` doesn't handle `lora_unet_input_blocks_*`; 0/282 match. Loader extension to handle A1111 + LoCon conv pairs is ~45 min; deferred.

## Status of LoRA-on-PersonaLive thread

- **Confirmed**: 8-step monkey-patch is a working tool. LoRA + decoupled CLIP can produce visible categorical style transfer. Ghibli α=3 + anime CLIP delivers "kinda Ghibli" with minor artifacts; not production-clean but *clearly the right direction*.
- **Confirmed (negative)**: pushing α higher will worsen artifacts (the α=2→α=3 jump already showed strain); shopping for higher-magnitude LoRAs by Δ doesn't help (Demon Slayer was bigger-rank-equal but worse-quality).
- **Inferred ceiling**: off-the-shelf SD1.5 LoRAs at 8-step monkey-patch top out around "kinda Ghibli α=3 quality." Production-clean output requires either (a) a substantially better-trained LoRA than what civitai surfaces by search rank, or (b) train a LoRA against PersonaLive's 4-step trajectory directly per the canonical answer in `2026-05-06-stylized-vtuber-requires-personalive-lora.md`.

## Photoreal-cleanup product implication (carried over from sibling doc)

For vamp-interface's uncanny-valley signal mechanism, the 4-step compound (decoupled CLIP + Ghibli LoRA both α=1) produces "cleaner-than-real photoreal" — useful as the legit-end calibration anchor for sus_level=0. The 8-step monkey-patch + α=3 combo overshoots this into Ghibli territory; not the same lever. Keep both knobs.

## Output artifacts

- `exp_output/personalive/hail_mary_8step/photoreal__ghibli_both_a2_8step.mp4` — Ghibli α=2, no decouple
- `exp_output/personalive/hail_mary_8step/photoreal__ghibli_both_a3_8step.mp4` — Ghibli α=3, no decouple
- `exp_output/personalive/hail_mary_8step/anime_clip__photoreal_spatial__ghibli_both_a3_8step.mp4` — Ghibli α=3 + anime CLIP **(best result)**
- `exp_output/personalive/hail_mary_8step/demon_slayer__anime_clip__a3_8step.mp4` — Demon Slayer α=3 + anime CLIP (control)
- `exp_output/personalive/hail_mary_8step/_frames/triple_a3_f150.png` — baseline | 8-step LoRA | + decouple
- `exp_output/personalive/hail_mary_8step/_frames/lora_compare_f150.png` — baseline | Ghibli | Demon Slayer
- `runs/hail_mary_*.log` — full stdout for each render

## Companions

- `2026-05-06-decoupled-clip-channel-falsified-cleanup-effect.md` — sibling doc, falsifies decoupled-CLIP-alone for stylization, frames cleanup-knob product use.
- `2026-05-06-stylized-vtuber-requires-personalive-lora.md` — sets up the LoRA-on-PersonaLive thesis; this doc partially confirms (working tool) and partially defers (off-the-shelf ceiling) the path forward.
- `_topics/personalive-acceleration.md` — topic index, update in same commit.

## Open / next probes (not pursued in this thread)

- A1111 naming + LoCon loader extension (~45 min) → AnimeAnything sweep at α=1, 2, 3 with 8-step + decouple. Tests whether broader weight-surface coverage (conv layers + attention) cleans up artifacts at the same magnitude.
- Custom LoRA training on PersonaLive's 4-step trajectory targeting RefNet specifically (canonical fix, 1–3 days per `2026-05-06-stylized-vtuber-requires-personalive-lora.md`).
- Step counts beyond 8 (12, 16) at the same α=3 — does the integration budget plateau, or keep climbing? Wall-time is per-window-jump-product, so cost stays constant at fixed temporal_adaptive_step=4.
