---
status: live
topic: liveportrait-stylized
---

# RAIN — streaming-vs-video refresher

**Date:** 2026-05-11
**Context:** Rorschach round-3 render pipeline shaping had been treating the filter slot (S0.10) as a *per-frame* operation. The product is a *stream*, not a video. This doc refreshes what "stream-native stylization" requires, anchored on the RAIN literature, and lands a small but load-bearing update to the shaping doc (R10.1 + S0.10 streaming contract).

---

## Two unrelated "RAIN"s

Both touch our pipeline; only one is stream-native. Naming up front to prevent later confusion.

### RAIN (Shu, Feng, Cao, Zha — arXiv:2412.19489, Dec 2024)
"Real-time Animation of Infinite Video Stream." Diffusion-based, causal-streaming portrait/character animation. **This is the one the streaming-vs-video framing points at.**

### RAIN (Ling, Zhang, Wang, Zhu — CVPR 2021)
"Region-aware Adaptive Instance Normalization for Image Harmonization." Image-only, harmonizes foreground over background. **Architecturally interesting for our compositor seam** (S0.9 paste-back), since image harmonization is exactly what we need to make a face-region LP output sit on top of a painted body without a visible seam. Not a streaming technique; treat as a candidate harmonization op inside S0.9 path rather than a renderer.

The rest of this doc is about the 2024 streaming RAIN.

---

## RAIN (2024) — what's actually new

| Property | Value |
|---|---|
| Base | SD-Image-Variations + AnimateDiff 1D temporal blocks |
| Driver | DWPose body keypoints (21pt) + face landmarks (68→26pt linear map); also tested with MiDaS depth for style transfer |
| Hardware | Single RTX 4090, TensorRT |
| **Throughput** | **18 fps at 512², ~55 ms/frame** (includes DWPose extraction) |
| Length | "Infinite" — no fixed clip horizon, only a sliding window |
| Window | K=16 frames, p=4 groups, 4 frames per noise level |
| Causality | Causal at the **stream level**; non-causal **inside the 16-frame window** |
| Training | Stage 1 30k steps image pairs (8×A100, batch 32); Stage 2 20k steps 16-frame clips; LCM distillation 1200 steps |

### The mechanism

Standard StreamDiffusion: batch size = denoising steps. One frame per noise level; small temporal context.

RAIN: batch size = (denoising steps) × p. K/p consecutive frames sit at the *same* noise level; adjacent groups offset by T/p ≈ 250 timesteps (T=1000). Temporal Adaptive Attention runs across these groups, including **cross-noise-level attention**, which prior stream methods had avoided.

This is what gives RAIN its temporal horizon. It's also why strict-causal masking fails — they tested it and report visible jitter "every 4 frames," exactly the group boundary. So **RAIN is causal across the stream but non-causal within its 16-frame window**. The window is the load-bearing design choice.

### Drift handling

Long-run identity drift is acknowledged but unresolved. Their words: "rarely, a little stir in previous frame may cause subsequent frames abnormal color." Recovery only happens "when the character appears in and out of the camera."

Mitigation: freeze one reference identity frame, fed back into the model. Same architectural pattern as our **frame-0 calibration** (R3.3) — RAIN independently arrives at it.

### Versus competitors on the same RTX 4090

| Method | FPS | Notes |
|---|---|---|
| StreamDiffusion | 37 | Cheaper, less temporally coherent, weak long-range |
| Live2Diff | 16 | Comparable arch to RAIN |
| **RAIN** | **18** | Longer effective temporal context |
| AnimateAnyone | offline | Quality reference, not real-time |

Quality (PSNR/SSIM/LPIPS/FVD on a 1-min held-out clip): RAIN trades ~5–10% on LPIPS/SSIM vs AnimateAnyone for 5–10× speedup.

---

## Implications for Rorschach

### RAIN itself is not v1, but is the v∞ ceiling

18 fps on a 4090 = ~55 ms just for the renderer, before DWPose extraction (which is bundled in their number), before v4l2 sink, before any post-pipeline filter or compositor. On a 3080 (our published min-spec), this slips below 30 fps comfortably and busts R4.1 (p50 ≤ 200 ms glass-to-OBS).

**LP stays the v1 renderer.** RAIN-class diffusion-streaming is the v∞ swap-in slot: same architectural position as LP, much higher quality ceiling, much higher compute cost. Comes back into play once a) consumer GPUs catch up or b) the LCM/distillation/turbo line gets RAIN-class methods under 30 ms/frame.

### Three RAIN design patterns transfer to our pipeline *today*

These do not require RAIN itself; they are streaming-architecture lessons.

#### Reference-frame anchoring against drift

RAIN freezes one identity frame as anti-drift guardrail. We already do this — frame-0 calibration on the source anchor, per `paced_sink.py` and our LP daemon's `prepare_source` path. **RAIN independently validates the architectural choice.**

Worth promoting: when AdaAttN/StyTr²/ReReVST is in S0.10, give it access to the frame-0 reference too, so the filter has a stable "what this stream should look like" anchor and can clamp drift. Concretely: pre-compute the filter's stylized output on frame-0 once at anchor-add time; on every live frame, EMA-blend toward that pre-stylized frame-0 in low-frequency LAB channels (sub-ms cost, kills slow color drift).

#### Sliding-window causal-with-lookahead, not strict-causal

This is the load-bearing one. **Strict per-frame stylization flickers visibly at 30 fps.** RAIN's strict-causal ablation shows "obvious jitter every 4 frames" — the exact pathology AdaAttN/StyTr²/AnimeGANv3 will produce on LP's output if we apply them frame-by-frame with no temporal coupling.

For our filter slot we need a small lookahead buffer — even 2–3 frames (~70–100 ms added latency) is acceptable if it kills flicker. We were not planning to do this; we should be.

Two implementations of the lookahead:

1. **In-filter (preferred)**: ReReVST is natively flow-consistent; its design already requires a few frames of context. Pick ReReVST for painterly and treat its inherent latency as a feature.
2. **Post-filter wrap (for non-flow-aware filters)**: wrap AdaAttN/StyTr²/AnimeGANv3 with a thin temporal-coherence pass — RAFT or even SDF flow on the LP output, warp prior stylized frame to current, blend with current-frame raw filter output at α ≈ 0.5 in mid-frequency bands, hard-clamp in low-frequency bands to the frame-0 stylized reference. Cost: ~5–8 ms on a 3080 with TRT.

Either way, the **filter pass must be a streaming pass, not a per-frame pass.**

#### Per-noise-level batching is a diffusion trick, doesn't transfer to LP — but earmarks v1.5+

RAIN's K/p grouping is specific to diffusion. Doesn't apply to LP-style feedforward warping. But it explains why if we ever want to insert a diffusion stage (e.g. SD-based painterly filter for highest-quality mode in v1.5+), we must think in terms of **small-window streaming-batches**, not one-image-at-a-time. StreamDiffusion's batching pattern is the entry point; RAIN's K/p extension is the polish.

This is the second part of the v∞ ceiling: diffusion-quality filtering at streaming speed is solvable, but it requires the streaming-batch architecture from the start. If we ever build it, we don't build a vanilla SD img2img pipeline and then optimize — we build it streaming-native, RAIN/StreamDiffusion-style, day one.

---

## Shaping doc consequences (landed)

Added to `rorschach/docs/shaping/shaping.md`:

- **R10.1** — Filter pass operates in streaming mode, not video mode. Per-frame stylization filters ship with a ≤3-frame lookahead buffer (≤100 ms added latency) and access to the frame-0 anchor reference, used for temporal-coherence clamping. Strict per-frame filtering is not acceptable — flickers visibly at 30 fps (cf. RAIN 2024 §5.3).
- **S0.10 streaming contract** — filter pass has access to (a) the 3 prior LP output frames, (b) the frame-0 reference (both raw and pre-stylized). ReReVST satisfies this natively. AdaAttN / StyTr² / AnimeGANv3 must be wrapped with a flow-consistency or EMA-anchored-to-frame-0 temporal coherence post-pass.

This is small but load-bearing: it forces the V0 spike (X4) to test filters under streaming conditions, not single-image conditions, so we discover the flicker problem at spike time rather than V1 integration time.

---

## What we don't know yet

- Exact ms cost on a 3080 for RAFT-flow temporal-coherence wrap. Spike X4 should measure.
- Whether 3 frames of lookahead is enough or whether we need 6–8 to kill flicker on AdaAttN. RAIN uses 16 inside its diffusion window, but their flicker is from group boundaries, not per-frame stylization noise — different mechanism.
- Whether the Shekhar 2023 "Interactive Control over Temporal Consistency while Stylizing Video Streams" approach (mentioned in the parallel WebSearch results) gives us a simpler streaming-coherence primitive than rolling our own RAFT wrap. Worth a follow-up read before X4 starts.

---

## Sources

- [RAIN: Real-time Animation of Infinite Video Stream (Shu, Feng, Cao, Zha — arXiv:2412.19489, Dec 2024)](https://arxiv.org/abs/2412.19489)
- [HTML version of RAIN paper](https://arxiv.org/html/2412.19489v1)
- [RAIN: Region-aware Adaptive Instance Normalization for Image Harmonization (Ling, Zhang, Wang, Zhu — CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/papers/Ling_Region-Aware_Adaptive_Instance_Normalization_for_Image_Harmonization_CVPR_2021_paper.pdf) — distinct paper, candidate for compositor-seam harmonization in S0.9
- [Interactive Control over Temporal Consistency while Stylizing Video Streams (Shekhar et al. — Computer Graphics Forum, 2023)](https://onlinelibrary.wiley.com/doi/10.1111/cgf.14891) — directly applicable streaming-coherence primitive, follow-up read

## Related local docs

- `rorschach/docs/shaping/shaping.md` — round-3 render pipeline shape (B + D + J)
- `docs/research/2026-05-08-stylization-locus-and-transfer.md` — parent thread, SPADE γ/β locus, LoRA-on-G path (v1.5+)
- `docs/research/2026-05-07-liveportrait-streaming-baseline.md` — current LP daemon performance reference
- `docs/research/2026-05-07-llf-streaming-pacing-and-sampling.md` — first streaming-pacing lessons, where the per-frame-vs-streaming distinction first bit us
