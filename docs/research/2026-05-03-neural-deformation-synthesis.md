---
status: live
topic: neural-deformation-control
---

# Neural deformation as the blendshape-control mechanism — synthesis

**Date:** 2026-05-03
**Combines:** [topic research (paper-side)](2026-05-03-neural-deformation-for-blendshape-control-topic.md) + [practical research (recipes / code / tools)](2026-05-03-neural-deformation-for-blendshape-control-practical.md)

## Decision

**Build the bridge: Flux render → AdvancedLivePortrait `Expression Editor` with a hand-authored ARKit-52 → 12-slider mapping.** This is the cheapest credible non-LoRA blendshape-control path (~2 h to first frame, ~2 GB VRAM beyond Flux). Treat it as a *second mechanism* alongside the LoRA sliders, not a replacement — the failure modes are complementary, and together they bracket the design space.

Do **not** wait for FG-Portrait, IM-Animation, or any 2026 paper checkpoint — none of the most architecturally-aligned methods (those that take a parameter vector rather than a driving video as input) have public code as of fetch date.

## What changed vs the slider thread

| Property | LoRA-slider path (current) | Neural-deformation path (this doc) |
|---|---|---|
| Per-axis training | ~1 h GPU per axis × 52 channels × tuning loops | none — install once, reuse |
| Channel coverage | each channel needs a curated pair corpus + critic gate | ~12 sliders cover ~20–25 of ARKit-52 well; ~25–30 channels have **no slider home** at all |
| Identity preservation | requires arc_distill anchor to avoid collapse (v1k recipe) | preserves source pixels by warping; degrades via texture softening / inpaint, not via identity drift |
| Stylization tolerance | trained on Flux distribution → robust on Flux outputs | fails on stylized / cartoon / animal Flux outputs (LivePortrait issue #119) |
| Compound expressions | LoRAs compose with measurable interaction; can be co-trained | additive in slider space but `calc_fe` keypoint indices overlap (smile and aaa share kp 3,7,17,20) → entanglement after warp |
| Sub-512 detail / texture | preserved (re-render at full quality) | softened by warp + decode pass |
| Failure direction | "wrong-but-sharp" (classifier fooling, identity drift) | "right-but-degraded" (correct geometry, blurred texture, occlusion artifacts) |
| Latency | sliders are free at inference once trained (~Flux baseline) | +1–2 s per render on 4090 (negligible for offline) |
| Channels with no path | 0 (anything we curate pairs for) | `cheekPuff`, `cheekSquint*`, `noseSneer*`, `mouthDimple*`, `mouthFrown*`, `mouthRoll/Shrug*`, `tongueOut`, fine eye-squint, fine gaze targets |

## Architectural reality check

LivePortrait's 21 implicit 3D keypoints are **not aligned to ARKit semantically** — they are the basis the network discovered as informative for motion description, full stop. PowerHouseMan's `calc_fe` (nodes.py) is a hand-tuned linear delta on specific keypoint indices; the 12 named sliders are *engineering shorthand*, not a derived semantic basis. There is no public ARKit-52 → 21-d motion-tensor mapper, in any repo, in any paper, at any quality level. We will write that bridge ourselves, accept the partial coverage, and tune the scale factors empirically against MediaPipe-readout exemplars.

The architecturally clean alternative is **FG-Portrait** (CVPR 2026, arXiv:2603.23381), which accepts FLAME ψ (~50–100-d expression coefficients, semantically close to ARKit) directly and computes a 3D-flow warp from the assembled mesh. FLAME↔ARKit converters exist in the avatar community. Reported metrics: VFHQ CSIM 0.462, FFHQ FID 87.0, self-reenactment LPIPS 0.158. **No code release.** Flag this for periodic check-back.

The actively-runnable diffusion-internal path is **HunyuanPortrait** (Tencent-Hunyuan, CVPR 2025) — Intensity-Aware Motion Encoder injected via cross-attention + AdaLN into Stable Video Diffusion, ArcFace 8.87 vs LivePortrait 8.71. **But it conditions on a driving video, not a blendshape vector** — wrong API for our offline `sus_factors → portrait` pipeline. Same shape for X-NeMo (ICLR 2025, code at bytedance/x-nemo-inference), RealPortrait, X-Portrait. None of them solve the specific problem we have.

**No published method exists that conditions a Flux/SDXL forward pass on a 21-d motion tensor or an ARKit-52 vector as a single forward generation.** This would be a novel contribution if we ever wanted to chase it.

## The bridge — what we actually have to build

```
sus_factors (16-d)
    ↓ [PCA / ridge / NMF — already in our pipeline]
arkit_blendshapes (52-d)
    ↓ [src/blendshape_bridge/arkit_to_alp.py — write this, ~50 LOC]
alp_sliders (12 named)
    ↓ [AdvancedLivePortrait Expression Editor]
warped Flux portrait
```

The middle link is the only piece of new engineering. Initial table (from the practical doc, verified against `nodes.py`):

| ARKit channel(s) | ALP slider | Range | Notes |
|---|---|---|---|
| `eyeBlinkLeft+Right`/2 | `blink` | 0..+5 | negative = wide-open (eye widen) |
| `eyeBlinkLeft − eyeBlinkRight` | `wink` | 0..25 | signed asymmetry |
| `browInnerUp − (browDown_L+R)/2` | `eyebrow` | -10..+15 | |
| `eyeLookOut/In*` combined | `pupil_x` | -15..+15 | |
| `eyeLookUp/Down*` combined | `pupil_y` | -15..+15 | |
| `jawOpen` × 100 | `aaa` | -30..+120 | |
| `mouthStretchLeft/Right` | `eee` | -20..+15 | |
| `mouthFunnel + mouthPucker` | `woo` | -20..+15 | |
| `(mouthSmileLeft + mouthSmileRight)/2` | `smile` | -0.3..+1.3 | |
| (head pose) | `rotate_pitch/yaw/roll` | -20..+20 deg | not in ARKit-52 |
| `eyeSquintLeft/Right` | **none directly** | — | empirical fit: small `+blink` + slight `-eyebrow` |

**Squint is the channel we are mid-thread on, and it has no dedicated ALP slider.** That's the sharpest concrete reason to keep the LoRA-slider path alive — at least until someone validates an empirical squint = f(blink, eyebrow, smile) fit.

## What this changes about the slider thread

Read `2026-05-03-slider-thread-recipe.md` first (the parking note). The neural-deformation path is *complementary*, not *replacement*:

- **Channels well-covered by ALP sliders** (jawOpen, smile, blink, brow, gaze, pucker/funnel) → use neural deformation, skip the LoRA training entirely. Saves ~1 h × N channels of GPU.
- **Channels with no slider home** (squint, cheekPuff, sneer, dimple, frown, lipRoll, shrug, tongue, fine gaze) → must stay on the LoRA path. The v1k recipe (bs_only + arc_distill anchor) is the right tool here. v4_pgd validation and v1l_squint_arc remain the next steps.
- **Critic role survives** — `bs_v3_t` / `bs_v4_pgd` becomes the *measurement and gate* for both paths. Use it to verify the warp produced the requested ARKit reading; use it as the LoRA training signal where the warp can't reach.

## Two-hour install plan

1. (15 min) `git clone` `kijai/ComfyUI-LivePortraitKJ` and `PowerHouseMan/ComfyUI-AdvancedLivePortrait` into `/home/newub/w/ComfyUI/custom_nodes/`. `pip install -r requirements.txt` for both. Restart ComfyUI. Models (~500 MB) auto-download from `Kijai/LivePortrait_safetensors` on first run.
2. (15 min) Open `Bacis_Liveportrait_with_expression_editing.json`, swap source for a Flux portrait, manually verify `smile=0.8` produces a visible smile.
3. (30 min) Write `src/blendshape_bridge/arkit_to_alp.py` from the table above. Sanity-test with hand-constructed vectors.
4. (30 min) Micro-benchmark identity drift on 20 Flux portraits × 5 slider settings using existing `insightface buffalo_l`. Threshold pass at ArcFace cosine ≥ 0.7. **If median < 0.7 the path is unrecoverable — pivot to PersonaLive.**
5. (30 min) Iterate scale factors against 3–4 ARKit reference exemplars (open mouth, smile, squint+brow-down, wink).

## Fallback escalation order

If LivePortrait is too coarse / too lossy after the micro-benchmark:
1. **PersonaLive** (CVPR 2026, Apache-2.0, 12 GB, code at GVCLab/PersonaLive, ComfyUI wrapper at `okdalto/ComfyUI-PersonaLive`) — landmark-driven, richer control surface than 12 sliders, designed for streaming.
2. **Follow-Your-Emoji** (SIGGRAPH Asia 2024) — landmark-driven diffusion animator, heavier (multi-second), better on stylized inputs. Good if our Flux LoRAs push us off-distribution for LivePortrait.
3. **HunyuanPortrait** (CVPR 2025, Tencent-Hunyuan) — only useful if we can synthesize a driving video from blendshape coefficients (e.g., render a FLAME mesh with the target ψ as the driver). Strongest identity preservation. Single-image API not available.
4. **Wait on FG-Portrait code** (CVPR 2026) — the architecturally cleanest fit (FLAME ψ → 3D flow → warped portrait). Track the arXiv abstract page for code release.

## Open questions

- Do the 12 sliders compose orthogonally enough on Flux outputs to encode a 16-d sus signal continuously? (Practical answer comes from running step 5 of the install plan with our existing per-portrait-pair seed-locked harness.)
- Does the warp preserve the sus signal we're trying to encode? Specifically: does a Flux portrait of "fraud rank 0.9" still feel uncanny after a smile=0.5 warp, or does the smile *override* the uncanny?
- Can we compose a LoRA-slider edit with a LivePortrait warp on the same portrait? (E.g., apply the LoRA squint at training scale, then ALP smile on top.) Composability check.
- For the squint channel specifically: empirical `squint = α·blink + β·eyebrow_neg + γ·smile_neg` fit — does it converge against MediaPipe readouts, or does it always entangle with one of the other channels?

## Cross-references

- Slider thread parking note: `2026-05-03-slider-thread-recipe.md`
- Source research (paper side): `2026-05-03-neural-deformation-for-blendshape-control-topic.md`
- Source research (practical): `2026-05-03-neural-deformation-for-blendshape-control-practical.md`
- Background reading: `~/w/Face-Research/05-neural-deformation.md`
- Relevant memory: `project_bs_lora_step_gating.md`, `project_blendshape_temporal_availability.md`
