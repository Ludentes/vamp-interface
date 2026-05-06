---
status: live
topic: neural-deformation-control
---

# LivePortrait — architecture review and extension points

Date: 2026-05-06
Companion: `2026-05-06-vtuber-pipeline-priorities.md` (the why)
Paper: `docs/papers/liveportrait-2407.03168.pdf` (Guo et al., Kuaishou, arXiv 2407.03168 v1)
Code (local): `/home/newub/w/LivePortrait` (KwaiVGI/LivePortrait, MIT-ish, depth-1 clone 2026-05-06)
Engineering fork (local): `/home/newub/w/FasterLivePortrait` (warmshao/FasterLivePortrait, ONNX/TRT productionization)
Weights (local):
- `/home/newub/w/LivePortrait/pretrained_weights/` — 1.9 GB, both human and animal (v1.1) `.pth`. Excluded ONNX (covered by FasterLivePortrait clone).
- `/home/newub/w/FasterLivePortrait/checkpoints/` — ~2.5 GB ONNX bundle (`warmshao/FasterLivePortrait` HF mirror).

## TL;DR

LivePortrait is **face-vid2vid scaled up and decoupled**, not a new architecture. Five pure-PyTorch CNN/MLP modules; 12.8 ms/frame model time on A100 (paper); FasterLivePortrait reports 30+ FPS end-to-end on 3090 with TensorRT (incl. crop/paste). No diffusion, no transformer, no temporal layer. The motion representation is a stack of **21 implicit 3-D keypoints + scale + Euler pose + expression deformation**, treated as "implicit blendshapes." Stitching and per-region (eyes, lip) retargeting are tiny MLPs sitting *outside* the trained base model — they edit keypoints, not pixels. This is what makes the system both fast and surgically controllable.

For our pipelines this matters in three places:

1. **The implicit-keypoint space is the natural target for an ARKit bridge** — same shape as our PersonaLive bridge problem (regress motion from blendshapes), different target space (21×3 implicit kps + R + t + s + 21×3 expression delta) and different driver math (Eqn. 7 in §3.4).
2. **Cross-id pose mismatch and the "expression-friendly vs pose-friendly" toggle** are not implementation details — they are the cross-id correctness story, and the source of half of the "looks weird in cross-id" artifacts.
3. **Every place where existing work injects custom motion** (eye/lip retargeting MLPs, the stitching MLP, JoyVASA's audio→motion-template plugin) is a documented extension point we can use *without retraining the warping or appearance network*.

## What LivePortrait actually is

### The five modules (numbers from `src/config/models.yaml`)

| Symbol | Module | Backbone | Input | Output | Notes |
|---|---|---|---|---|---|
| F | Appearance Feature Extractor | 6-resblock CNN with 2 down-blocks, reshape to volume | RGB 256² | feature volume **f_s ∈ ℝ^{32×16×64×64}** | Run once per source |
| M | Motion Extractor | ConvNeXtV2-Tiny + 7 FC heads | RGB 256² | `kp ∈ ℝ^{21×3}` (canonical), `R(pitch,yaw,roll)` (Euler), `δ ∈ ℝ^{21×3}` (expr deformation), `t ∈ ℝ^3`, `s ∈ ℝ` | Eqn. 2: `x = s·(x_c·R + δ) + t`. Run per frame on driver and once on source. |
| W | Warping Module | DenseMotionNetwork (5-block CNN + occlusion) + warp + 3D refine | f_s, x_s (21,3), x_d (21,3) | warped feature `Bx256x64x64` | DenseMotion produces `Bx16x64x64x3` deformation field via grid_sample 5-D (the kernel that needs the TRT custom op). |
| G | SPADE Generator | SPADE decoder + PixelShuffle | warped feature 256×64×64 | RGB 512² | Output upsamples 256→512 via PixelShuffle. |
| S, R_eyes, R_lip | Stitching + Retargeting | Three small MLPs | concatenated kps, eye/lip ratios | Δ ∈ ℝ^{21×3} (+ tx, ty for stitching) | Stage-2 only; trained with base frozen. |

Total trained-from-scratch params at Stage 1: F + M + W + G + Disc. Stage 2 freezes those four and trains only S, R_eyes, R_lip. This separation is load-bearing for our extension story (see below).

### Inference loop (per driver frame)

```
once:        f_s, x_c_s, R_s, δ_s, t_s, s_s = M(I_s);  x_s_canonical = M's kp head
per frame:   x_c_d, R_d, δ_d, t_d, s_d = M(I_d)
             x_s, x_d = paper Eqn. 7  (cross-id transform — see "the toggle" below)
             if stitching:    Δ_st = S([x_s, x_d]);   x_d ← x_d + Δ_st
             if eye retarg:   x_d  += α_eyes · R_eyes(x_s, c_eyes)
             if lip retarg:   x_d  += α_lip   · R_lip(x_s, c_lip)
             ret = W(f_s, kp_source=x_s, kp_driving=x_d)
             I_p = G(ret['out'])      # 512² RGB
             paste back to original frame using stitching tx,ty
```

`src/live_portrait_wrapper.py` exposes each step as its own method (`get_kp_info`, `transform_keypoint`, `retarget_eye`, `retarget_lip`, `stitching`, `warp_decode`). **Every one of these is a clean injection point.**

### The cross-id transformation that papers gloss over

Eqn. 7 in §3.4 is where same-id and cross-id reenactment diverge:

```
x_s   = s_s · (x_c,s · R_s + δ_s) + t_s
x_d,i = s_s · (s_d,i / s_d,0) · ( x_c,s · (R_d,i · R_d,0⁻¹ · R_s)
                                + (δ_s + δ_d,i − δ_d,0) )
        + (t_s + t_d,i − t_d,0)
```

Read this carefully. The driving keypoint is the **source's canonical shape** rotated by the driver's *delta* rotation since frame 0, expressed in the source's frame. `δ_d - δ_d,0` is added as an expression *delta*, not replaced. Scale is multiplicative ratio. Translation is additive delta. This is "pose-friendly" mode (the default in `inference_config.py`).

`expression-friendly` mode replaces the rotational composition with a different formulation that biases toward exact driver expression at the cost of pose stability. The toggle is `inf_cfg.driving_option`.

**This matters for our work** because the moment we plug an ARKit-driven bridge into LivePortrait, we have to pick a side: do we want absolute ARKit pose (replace `R_d`) or relative ARKit pose (delta from a calibration frame)? The bridge math is not "predict 21×3 kps from blendshapes" — it's "predict (R_delta, δ_delta, t_delta, s_ratio) from (ARKit head pose, ARKit blendshapes)." Smaller, cleaner, and more identity-preserving than predicting kps directly.

### Stage-1 vs Stage-2 separation (the unlock)

Stage 1 (base): F, M, W, G trained jointly on 69M video frames + 60K styled stills, 8×A100 ×10 days. Generalization comes from this stage. **We will never retrain Stage 1.**

Stage 2 (control): S, R_eyes, R_lip — three small MLPs (largest is the eye retargeter at 6 layers / max 256 hidden) trained for ~2 days on 8×A100. Trained with the base model frozen. **This is the cheap extension surface** — same setup as our PersonaLive bridge (frozen UNet, train a small adapter).

Eyes retargeting MLP: input 66 = 21×3 + 3 (eye-open ratio tuple), output 63 = 21×3 (kp delta). Lip: 65→63. Stitching: 126 = 2×21×3 → 65 = 21×3 + tx,ty.

### Output and resolution

- Output is **512²**, achieved by a final PixelShuffle in G upscaling from 256 to 512. The warped feature volume is at 64² spatial; the bottleneck is in `W` and the SPADE decoder.
- Source can be any size up to `source_max_dim=1280`. The cropped/aligned face is 256² for processing; pasteback returns the rest of the source frame untouched (modulo stitching).

## What's good vs what's brittle

### Good

- **Implicit blendshapes work.** The δ ∈ ℝ^{21×3} learned in Stage 1 actually composes additively (Eqn. 7's `δ_s + δ_d,i − δ_d,0`) and additively across retargeters (`α_eyes · Δ_eyes + α_lip · Δ_lip` in `Algorithm 1`). The retargeting ablations (Fig 8/9) show the MLPs do learn semi-disentangled controls.
- **Fast.** 12.8 ms/frame model time on 4090 with naive PyTorch. FasterLivePortrait engine-fused TRT lands 30+ FPS on 3090 with crop+paste. PersonaLive's TRT ceiling on 5090 was ~20-21 FPS; LivePortrait should clear 40+ on 4080 with the same TRT path. (Will be a sanity-check gate, not yet measured here.)
- **Stitching is a real fix, not cosmetic.** §4.3 + Fig 7 — when not stitched, paste-back has visible shoulder discontinuity. Trained as a tiny MLP, runs in negligible time. We get this for free.
- **Animal generalization** (paper §D, `pretrained_weights/liveportrait_animals/base_models_v1.1/`) — same architecture, same Stage 2 modules, fine-tuned base. Suggests OOD generalization is a fine-tuning question, not architectural.

### Brittle

- **Cross-reenactment with large pose change.** Stated explicitly in §5 Limitations. The `pose-friendly` Eqn. 7 helps but doesn't solve. Driver yaw beyond ±60° is still rough.
- **Shoulder/foreground jitter.** Same paragraph. Stitching helps; FasterLivePortrait adds extra smoothing, but this is the visible failure mode in long clips.
- **Implicit kps are unconstrained.** They're not anatomical landmarks. Two consequences: (a) they re-distribute themselves freely under fine-tuning and (b) attempts to inject "clean" external motion (like ARKit blendshapes) need to go through a learned bridge, not a hand-coded mapping. This is exactly the lesson from the PersonaLive ARKit-bridge work — applied here, it argues for *not* trying to land ARKit blendshapes directly into δ.
- **Warping module needs grid_sample 5-D**, which is not in stock TensorRT (`grid-sample3d-trt-plugin` is a third-party CUDA op needed for TRT). Adds friction to the engineering deploy path; FasterLivePortrait scripts handle it.
- **Eyes-open / lip-open ratios are 1-D scalars** (`c_d_eyes` is `Bx3` for eye-open tuple, `c_d_lip` is `Bx2`). The retargeter is conditioned on a low-D summary, not the full ARKit blendshape vector. If we want richer retargeting (cheek puff, brow flash, jaw side) we have to train new retargeters with new condition channels.

## Extension points, ranked by leverage

For each one: what it gives us, cost, risk.

### Tier 1 — high leverage, low cost (no Stage 1 retrain, often no training)

1. **Run-time control flags as tuning surface.** `inference_cfg` exposes ~20 flags: `driving_option` (pose vs expression friendly), `driving_multiplier` (overdrive expression intensity), `driving_smooth_observation_variance` (motion smoothing), `flag_normalize_lip` (closed-lip prior), `flag_relative_motion`, `animation_region` ∈ {exp, pose, lip, eyes, all}. **Almost all the "feels off" knobs exist already.** First pass on any quality complaint should be a sweep here.
2. **Animation-region masking.** `animation_region="exp"` keeps source pose. `"lip"` only animates lips. Concretely: on a low-quality driving video, `animation_region="exp"` + smoothing trades pose-tracking for stability. Useful for podcast-style talking-head where the speaker's head barely moves.
3. **Use FasterLivePortrait's TRT path.** Treat vanilla LivePortrait as the reference implementation, FasterLivePortrait as the deploy target. Engine cache is per-shape — first run is slow, subsequent are fast. **Sanity test should run this stack, not `python inference.py`.**
4. **Drop in JoyVASA-style audio→motion-template plugin.** Audio drives implicit-kp deltas; rest of the pipeline is unchanged. Gives us an audio-driven product almost for free *if* we want one. Same architectural shape as a hypothetical ARKit driver.

### Tier 2 — medium leverage, modest cost (Stage 2 retrain only, ~hours on a single 5090)

5. **ARKit-conditioned retargeter.** Replace `R_eyes(x_s, c_eyes:3)` with `R_arkit(x_s, b_arkit:52) → Δ_kp ∈ ℝ^{21×3}`. Same MLP shape, condition vector grows from 3 to 52. Trained with frozen base on a paired (ARKit-extracted-from-driver, kp-from-driver) corpus. Cleaner than retraining δ end-to-end. **This is the LivePortrait analog of the PersonaLive ARKit bridge.**
   - Variant A: condition on raw ARKit blendshapes.
   - Variant B: condition on (ARKit blendshapes, ARKit head pose) and replace the rotation entirely (driver-bypass rotation path).
   - Variant B is closer to "puppeteer mode" — performer's iPhone face fully replaces the driving video. No driver video required at inference. Big UX win.
6. **Per-axis stitching multipliers.** The current stitching MLP is one-shot. Splitting it into (head-stitch, shoulder-stitch) with separate α coefficients buys finer control over the visible shoulder jitter without retraining the warping module. Cheap Stage-2 ablation.
7. **Driver-side smoothing schedule learned from data.** `driving_smooth_observation_variance` is a single hyper. Replacing with a learned per-axis smoother (Kalman or exponential-decay-NN over kp deltas) addresses the long-clip drift problem. Same family as `2026-05-04-mediapipe-distill` work — a small temporal head.
8. **Auxiliary canonical-kp regularization for fine-tuning.** If we ever fine-tune the base on stylized OOD references (orc, demon, duck), the loss term that prevents canonical kps from drifting is `L_E` + the landmark-guided `L_guide`. Both are described in §3.2. We'd reuse them on a small stylized dataset. Risk: stylized refs lack landmarks.

### Tier 3 — high leverage, high cost (Stage 1 fine-tune)

9. **Fine-tune the base on stylized refs to fix OOD identity collapse.** Animal model proves this works (§D). Estimated cost: 1.3K stylized clips wasn't enough for them in Stage 1; at fine-tune, much smaller corpora work. Expected: 1–2 days on a single 5090. *Only* attempt if Stage-2 fixes (item 5–7) don't get us there.
10. **Replace ConvNeXtV2-Tiny with something stronger.** Motion Extractor M is a 28M-param backbone. The expression head's regression quality is the upstream cap on everything. Bigger backbone (ConvNeXtV2-Base, ViT-B) costs ~3× M's inference budget but might fix subtle expression underestimation. Marginal; only worth it after items 5–9.
11. **Replace SPADE generator with a diffusion-distilled one for quality regime.** This is "PersonaLive on top of LivePortrait keypoints." Out of scope for Regime A. Would re-introduce the PersonaLive speed problem. **Only mentioned for completeness — don't.**

### Tier 4 — research, only if everything else fails

12. **Train a 4096-d implicit-kp variant** to give M more capacity. Speculative; nobody has done this. Would forfeit pretraining.
13. **Replace warping with diffusion-based rendering** — i.e., become FADM/AniPortrait. Re-introduces the speed problem. Don't.

## Concrete things to try first (the work-order)

In order, with measurable gates.

1. **Reproduce paper FPS on 4080 with FasterLivePortrait + TRT engines.** Gate: ≥25 FPS at 512² on the canonical `s10.jpg` × `d14.mp4` test (their assets). Fail → engineering issue, not a model issue; debug TRT plugin.
2. **Run our 5 stylized + 3 photoreal anchors through vanilla LivePortrait** (no ARKit bridge yet). Measure: identity preservation (CSIM via insightface buffalo_l), expression match, pose match (APD-style angular error). Gate: photoreal stable, stylized at least *recognizable*. This sets the baseline.
3. **Tune Tier-1 flags on stylized refs.** `animation_region`, `driving_multiplier`, `driving_smooth`, `flag_normalize_lip`. Document which knobs help which artifact.
4. **Build the LivePortrait ARKit bridge (Tier-2 item 5, Variant A first).** Architecture mirror of the PersonaLive bridge: paired-frame distillation against a teacher that runs vanilla LivePortrait. Train the eye/lip retargeters, conditioned on full ARKit blendshape vector, against frozen base. Acceptance: the same 3-axis sign-agreement test applied to PersonaLive bridge, plus retargeter-output-magnitude sanity vs paper Fig 8/9 sweeps.
5. **Variant B (driver-bypass).** Train a kp-delta MLP conditioned on full ARKit (52 BS + 6 pose). Replaces the entire `M(I_d)` call at inference — the iPhone is the driver, no video needed.
6. **Wire FasterLivePortrait + ARKit bridge into our OBS pipeline.** Reuse the PersonaLive scaffolding.

## Open questions / things to verify by reading code (not paper)

- `flag_relative_motion=True` (default) does what exactly relative to `pose-friendly`? Both seem to encode "delta from frame 0" semantics; possible aliasing.
- The `motion_multiplier` is applied where in the chain? Skim of `live_portrait_pipeline.py` says it scales kp_driving deltas globally — confirm this is multiplicative on `(δ_d - δ_d,0)`, not on the absolute δ.
- Is `animation_region="exp"` masking implemented as kp-channel masking or as a different transform path? Affects whether we can compose region masks with our ARKit bridge cleanly.

## Mapping to existing memories and threads

- **PersonaLive ARKit bridge lessons apply 1:1 here.** Sign-agreement test ([feedback_calibration_blind_spots](memory pointer)), 90k-step floor, render-enriches-parquet rule — all carry over.
- **The `_topics/personalive-acceleration.md` strategy doc** still owns the realtime story; this doc is the parallel "what we'd do *with* LivePortrait" for the LivePortrait product gated on item 1 in `2026-05-06-vtuber-pipeline-priorities.md`.
- **Slider/FluxSpace work is unaffected** (Regime B). LivePortrait can consume slider-generated portraits as source images; that integration is an inference-time concatenation, not a coupling.

## Cited evidence

- Paper §3.1 (Eqn. 1, face-vid2vid recap), §3.2 (Stage-1 architecture & losses), §3.3 (Stage-2 stitching/retargeters, Fig 3, Eqn. 5–6), §3.4 (Eqn. 7, inference, `Algorithm 1`).
- `src/config/models.yaml` for sizes; `src/live_portrait_wrapper.py` for stage decomposition; `src/modules/convnextv2.py:fc_kp/fc_pitch/...` for the M head structure.
- §5 Limitations on cross-reenactment + shoulder jitter (the brittleness we target with Tier-2 item 7).
- FasterLivePortrait README.md for TRT path, JoyVASA integration, and the grid_sample 3-D plugin requirement.
