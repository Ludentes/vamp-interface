---
status: live
topic: neural-deformation-control
---

# PersonaLive architecture notes — components, extension points, perf, training, replacements

Working notes on the PersonaLive (CVPR 2026, arxiv 2512.11253) inference
stack as it lives at `~/w/PersonaLive`. Goal: a single document we can
return to when reasoning about extension, replacement, or optimization of
any component. Numbers labelled **(speculative)** are pending empirical
measurement on RTX 5090 — see *Perf budget* section.

For paper-side architecture context see also:
- `2026-05-04-personalive-tensorrt-plan.md` (acceleration thread)
- `2026-05-03-talking-heads-2025-2026-survey.md` (positioning vs peers)
- `2026-05-03-liveportrait-x-nemo-analysis.md` (lineage)

## Three-stage training overview (paper §3)

| Stage | What it trains | What loads | What's frozen during |
|---|---|---|---|
| 1. Image-level hybrid motion | Motion path: `pose_guider`, `motion_encoder`; pose conditioning route into D | `denoising_unet` (2D), `reference_unet`, VAE, image_encoder | Backbone diffusion (mostly), motion_extractor (off-the-shelf E_k from LivePortrait) |
| 2. Fewer-step appearance distill | Distill many-step → 4-step student via `L_mse + L_lpips + L_adv` (with StyleGAN2 discriminator) | Stage 1 weights + temporal_module init | Reference path; only D's weights move; trainable subset |
| 3. Micro-chunk streaming + sliding training + HKM | Temporal-module weights for chunk-wise generation; history bank / motion bank for long-term consistency | All Stage 2 outputs + temporal_module | Everything except the temporal stack |

Maintainer status: training code release pending (issue #17, no ETA).
Moore-AnimateAnyone covers Stage-1 reconstruction faithfully on a single
5090 (per `2026-05-05-moore-stage1-probe-handoff.md`); Stages 2-3 unprobed.

## Component table

(Sizes verified by `du -h` on `~/w/PersonaLive/pretrained_weights/personalive/`;
parameter counts inferred from architecture or `state_dict()` length where
known.)

| Component | Class | File:Line | Weights | Param/size | I/O shapes | Stateless? |
|---|---|---|---|---|---|---|
| Reference UNet | `UNet2DConditionModel` (diffusers) | external | `reference_unet.pth` 3.3 GB | ~860 M | `(B,4,64,64) z + (B,L,768) image_emb` → cached attn KV | per-call |
| Denoising UNet | `UNet3DConditionModel` (vendored) | `src/models/unet_3d.py` | `denoising_unet.pth` 4.6 GB + `temporal_module.pth` 1.7 GB | ~860 M (2D) + ~46 M (temporal) | `z (B,4,T,64,64), t, image+motion encoder hidden, pose_cond_fea` → noise | per-call (T-batched) |
| ReferenceNet→Denoising attn cache | (built-in via `BasicTransformerBlock`) | `src/models/transformer_*.py` | shared above | — | reference KV → cross-attn slots | per-session |
| Pose encoder (LivePortrait MotionExtractor) | `MotionExtractor` | `src/liveportrait/motion_extractor.py:19` | `motion_extractor.pth` 108 MB | ~30 M (ConvNeXtV2-tiny) | `(B,3,256,256)` → posed `(B,21,3)` (and raw `kp_info` dict) | per-frame |
| PoseGuider | `PoseGuider` | `src/models/pose_guider.py` | `pose_guider.pth` 4.2 MB | ~700 K | `(B,3,T,512,512)` keypoint canvas → `(B,320,T,64,64)` | per-frame (CNN) |
| Motion encoder (FAN-SA) | `MotEncoder` | `src/models/motion_encoder/encoder.py:17` | `motion_encoder.pth` 236 MB | ~50 M | `(B,3,T,224,224)` face crop stack → `(B,T,32,16)` | per-frame (verified: line 38 `rearrange "b c f h w -> (b f) c h w"`) |
| VAE | `AutoencoderKL` (sd-vae-ft-mse) | external | ~170 MB | ~83 M | `(B,3,512,512) ↔ (B,4,64,64)` | per-image |
| TinyVAE (optional) | `AutoencoderTiny` | external | ~5 MB | ~1 M | same as VAE | per-image |
| Image encoder | `CLIPVisionModelWithProjection` (sd-image-variations-diffusers) | external | ~230 MB | ~190 M | `(B,3,224,224)` → `(B,L,768)` | per-session |
| Scheduler | `DDIMScheduler` (vendored) | `src/scheduler/scheduler_ddim.py` | — | — | timestep ↔ noise | stateful only across denoising-loop steps |
| Pipeline (offline) | `Pose2VideoPipeline_Stream` | `src/pipelines/pipeline_pose2vid.py` | — | — | reference image + driving video → MP4 | sliding-chunk stateful |
| Online wrapper | `Wrapper` | `src/wrapper.py` | — | — | streaming wrapper around the pipeline | accumulates motion/pose deques |

Total deployment storage: ~10.4 GB weights + ~5 GB external base diffusion
checkpoint = ~15 GB. Inference VRAM under fp16: ~12-14 GB on a real driving
video (RTX 5090, per `personalive-acceleration.md`).

## Per-component notes (deeper)

### Pose encoder — `MotionExtractor`

ConvNeXtV2-tiny detector head emitting `{kp (B,63), pitch/yaw/roll (B,66
each), t (B,3), scale (B,1)}`. Pitch/yaw/roll are **66-bin classification
distributions** that get converted to degrees via `headpose_pred_to_degree`
(soft-argmax over uniform bins).

`get_kp(kp_info)` (motion_extractor.py:53) realizes equation (2) of the
paper:
```
rot_mat = get_rotation_matrix(pitch, yaw, roll)
kp_transformed = (kp.view(B, 21, 3) @ rot_mat) * scale[..., None]
kp_transformed[..., 0:2] += t[..., None, 0:2]
```

`interpolate_kps_online` (motion_extractor.py:177) is the deployment-path
helper: keeps `k_c` from reference (`kp1['kp']`), blends `t` via `t_scale=0.5`,
zeroes the driving's scale contribution (`s_scale=0`). This is a
key "expression vs identity" boundary: PersonaLive's own production code
treats canonical keypoints + scale as **identity-bound** and only allows
rotation + a fraction of translation through from the driving signal.

**Extension points:**
- Replace LivePortrait's E_k with a different keypoint detector (FLAME,
  MediaPipe FaceLandmarker, ARKit). Output must remain (B, 21, 3) post-pose;
  the 7 specifically-rasterized indices must correspond to roughly the same
  landmarks, otherwise PoseGuider sees a foreign distribution. Calibration
  is non-trivial.
- Bypass entirely (our v1 plan): cache `kp_ref/t_ref/s_ref` once per session;
  build R_d per frame from any rotation source.
- Keep but augment: feed both the real LivePortrait kp and the closed-form
  ARKit-built kp through draw_keypoints separately, weighted-blend the
  resulting canvases. Could improve robustness when LiveLink ARKit is
  noisy.

**Replacement options:**
- ARKit head Euler → closed form (chosen for v1 plan). Cost: 0 ms inference.
- Distill from RGB→keypoints with a smaller backbone (MobileNetV3-tiny,
  ~3 M). Cost: ~0.5 ms vs 1-2 ms (speculative). Useful if we ever want to
  drive PersonaLive from ARKit-less RGB on a phone.

### PoseGuider

`InflatedConv3d` stem + 3 stride-2 InflatedConv3d downsamples + zero-init
final conv → 320-channel feature map. Total ~700 K params; fp16 inference
~0.2 ms (speculative). Tiny — never the bottleneck.

The 7 colored dots at radius 4 are a deliberately **low-information
conditioning signal**. The denoising UNet learns to use this as global
position + rough head pose, not detail. This is why the dots-only canvas
is sufficient: any expression-bearing detail would be redundant with what
MotEncoder already provides via cross-attention.

**Extension points:**
- Train a thicker PoseGuider variant for use with denser pose canvases
  (FLAME mesh rasterization, ARKit blendshape-rendered face). Would shift
  load away from MotEncoder and might enable SR-style detail recovery.
  Requires a Stage-1 retrain.
- Add multi-modal input channels: depth, normals, ID embedding. Conv stem
  trivially accepts more channels; weights for new channels need init via
  zero-init or interpolation.

### Motion encoder — `MotEncoder` / `FAN_SA`

Frame-parallel feature extractor. `forward` (encoder.py:36-41):
```python
latent = self.model(rearrange(x, "b c f h w -> (b f) c h w"))
latent = self.final_proj(latent)
latent = rearrange(latent, "b (l c) -> b l c", c=self.out_ch) + self.pe
latent = rearrange(latent, "(b f) l c -> b f l c", f=x.shape[2])
```

The FAN-SA backbone is a **stack of hourglasses with spatial self-attention**
(`FAN_feature_extractor.FAN_SA`). 224² input is downsampled through the
hourglass stages to a 32×16 latent (32 spatial positions × 16 channels)
*per frame*. The 1-D sincos PE across the 32 positions is added inside
`forward`; downstream UNet consumes the full PE-augmented tensor.

Critically, **expression detail goes through here**, via cross-attention into
the denoising UNet (`encoder_hidden_states[1]`). The model has the entire
RGB face as input — eyes, mouth, brows, micro-deformations — so the 32×16
latent is a *learned compression* of that detail.

**Extension points:**
- Replace FAN_SA backbone with a smaller / faster / domain-specialized
  encoder (DINOv2-small, MediaPipe-FaceMesh-style geometry, ARKit
  blendshape-renderer). Output stays (B, T, 32, 16) PE-included. Requires
  Stage-1 retrain.
- Add an auxiliary expression-classification head on the 32×16 latent for
  diagnostic / interpretability work; cheap to bolt on, doesn't disturb
  inference path.
- Training-time KD from a stronger encoder (e.g., a self-supervised face
  expression model) to make m_f more robust to lighting/occlusion.

**Replacement options:**
- Distilled MLP from ARKit blendshapes (chosen for v1 plan). Cost: ~0.05 ms
  inference (speculative).
- Replace at the seam (different cross-attn dim) — would require denoising
  UNet retrain. Heavy.

### Denoising UNet (`UNet3DConditionModel`)

Vendored from AnimateAnyone-lineage; 2D Stable Diffusion UNet with 3D
temporal modules inflated. ~860 M params for the 2D stack + ~46 M for
temporal_module.

Inference call shape (offline):
```python
noise_pred = denoising_unet(
    z_t,                          # (B, 4, T, 64, 64)
    t_step,                        # scalar
    encoder_hidden_states=[
        image_prompt_embeds,       # (B, L_clip, 768) from CLIP+ReferenceNet
        motion_hidden_states,      # (B, T, 32, 16) from MotEncoder
    ],
    pose_cond_fea=pose_fea,        # (B, 320, T, 64, 64) from PoseGuider
    return_dict=False,
)[0]
```

Stage 2 distillation reduces denoising trajectory to N=4 steps. The
trajectory in Figure 3 of the paper shows Stage 2 establishes layout in
the first step (highest noise) and refines appearance in subsequent steps —
which is the empirical justification for the
fewer-step distill: only the early steps carry layout-changing
information, the rest is "polishing" and can be aggressively compressed.

**Extension points:**
- LoRA fine-tune for: identity (per-subject), style (cartoon, photoreal),
  domain (medical, low-light). Same recipe as Concept-Sliders for SDXL,
  applied to UNet3DConditionModel; should work cleanly given the model is
  fundamentally a SD1.5-derivative.
- Adapter modules in the cross-attn slots to route additional conditioning
  (audio, gesture) without retraining the backbone.

**Replacement options:**
- Plug in a different diffusion backbone (e.g., Wan2.5, LTX-Video) for the
  3D denoising. Heavy — shapes and conditioning conventions differ.
- Step-distillation downstream: turn the 4-step student into 2-step or
  1-step via Hyper-SD-style adversarial distillation. Listed as expensive
  but tractable in the acceleration thread.

### Reference UNet + image encoder

`reference_unet` runs once per session on the reference RGB and caches its
KV pairs across all transformer blocks. The image encoder (CLIP) provides
the image-prompt embedding inserted into every denoising-step
cross-attention. Together these form `R(I_R)` from paper equation (1).

**Extension points:**
- Replace CLIP with SigLIP / SigLIP-2 / DINOv2 for richer ID embedding.
  Stage-1 retrain required.
- Add an explicit ArcFace-style ID loss during Stage 1 to harden identity
  preservation. Already half-present in the LPIPS term but not directly
  ID-targeted.

### Scheduler / streaming wrapper

`Pose2VideoPipeline_Stream` (`src/pipelines/pipeline_pose2vid.py`)
implements the micro-chunk slide. Each denoising window is split into N
chunks with progressively higher noise levels (paper §3.3 eqns. 5-6).
After each denoising step the window slides forward by one chunk, emits
M=4 clean frames, and appends a fresh noisy chunk at the back.

`Wrapper` (`src/wrapper.py`) is the online (real-time) variant: maintains
`pose_pile`, `motion_pile`, `motion_bank`, `history_bank` deques, plus the
`temporal_window_size` and `temporal_adaptive_step` knobs from the config.

**Extension points:**
- Lower temporal_window_size to reduce VRAM at the cost of temporal
  coherence.
- Replace HKM threshold τ=17 with a learned criterion (small classifier
  on motion_hidden_state norm) for adaptive keyframe selection.
- Externalize the motion/history banks for cross-session consistency
  ("memory" of how the user's face has been deformed over time).

### Vendored utilities

- `draw_keypoints` (src/utils/util.py:337) — colored-dot rasterization
  (7 dots, fixed colors, radius=4 px, 512² canvas). Sometimes called
  per-chunk in offline, sometimes once per batch in online; both code
  paths converge on the same `(B, 3, T, 512, 512)` output.
- `get_boxes` (src/utils/util.py:374) — face crop bbox computed from the
  same 7 keypoints; used to extract the 224² face crop fed to MotEncoder.
  Tightly coupled to `draw_keypoints`'s keypoint selection — change one,
  must change the other.
- `crop_face` (src/utils/util.py) — used in `inference_offline.py` to
  loose-crop reference + driving via MediaPipe FaceMesh as a preprocessing
  step before everything else. **Note:** breaks under mediapipe ≥ 0.10.x
  due to `mp.solutions` API removal — see below.

## Perf budget (speculative; needs measurement)

Order-of-magnitude, fp16, RTX 5090, batch=1 (T_total=4 chunked frames):

| Path | Component | Speculative time/frame | Bottleneck? |
|---|---|---|---|
| Setup (1×/session) | CLIP image_encoder | 5 ms | no |
| Setup (1×/session) | reference_unet | 30 ms | no |
| Setup (1×/session) | motion_extractor on ref | 1 ms | no |
| Setup (1×/session) | motion_encoder on ref crop | 2 ms | no |
| Per-chunk | motion_extractor on driving (×T) | 1 ms × T | no |
| Per-chunk | draw_keypoints | <0.1 ms | no |
| Per-chunk | pose_guider | 0.5 ms | no |
| Per-chunk | motion_encoder on driving stack | 5 ms × T | minor |
| Per-chunk | denoising_unet × 4 steps × T frames | **40-50 ms** | **YES** |
| Per-chunk | VAE.decode × T | 5 ms × T | minor |

At T=4 (default chunk size), end-to-end ≈ 60-100 ms / chunk → 40-60 FPS
emission. PersonaLive's published 15.8 FPS on H100 is for a different
setup (8× H100 training; H100 inference benchmark with full pipeline
including I/O). 5090 vs H100 puts it in the 8-15 FPS ballpark on stock,
~20-25 FPS post-acceleration per the TRT thread.

**Replacing motion_encoder with a 0.7M-param MLP saves ~20 ms / chunk
at T=4** (5 ms × 4 → ~0.05 ms × 4) — the second-largest single-component
saving available, after step-count distillation. **(speculative; measure
in Task 8 readout.)**

**Replacing motion_extractor's driving-frame call with closed-form math
saves ~4 ms / chunk at T=4** (1 ms × 4 → 0 ms). Smaller but still real-time-
visible.

### Empirical measurement plan

Pending after the bridge v1 land. To execute:

```bash
# Component-level micro-benchmarks at T=4, fp16, RTX 5090.
# Wrap each module in a torch.cuda.synchronize-bracketed timer.
# Save to exp_output/perf/personalive-component-budget.json.
```

Sketch in `scripts/bench_personalive_components.py` (planned, not
written): time each named seam over 100 iterations after warmup, dump
JSON with min/median/p99 latency.

Numbers above are speculative until that lands.

## Training notes (what we know about Stage 1)

From `2026-05-05-moore-stage1-feasibility-probe.md`:

- **Single 5090 fits** Stage 1 at batch=1, 512², bf16, gradient checkpointing,
  **8-bit Adam (bnb)**, no xformers. ~1.05 s/it.
- 32-bit Adam OOMs (Adam state ~13.6 GB on the 1.7B trainable params).
- Diffusers 0.24 pin (Moore vendor); torch 2.11 + cu128.
- Dataset format: `HumanDanceDataset` expects `(video.mp4, pose.mp4)` pairs;
  the original training pipeline uses DWPose-rendered keypoint videos as
  the pose stream. PersonaLive deviates: the pose stream is a `draw_keypoints`
  rendering of LivePortrait keypoints, not DWPose.

What's not yet probed:
- Stage 2 (temporal module + adversarial distillation): pairs teacher +
  student forward passes; memory blowup likely.
- MotEncoder + MotionExtractor at training time: extra forward+backward
  per step over 224² and 256² crops respectively. Plausibly pushes us from
  "fits" to "doesn't fit" on a 5090.
- IPS at meaningful dataset size: 1.05 s/it is on a 30-frame synthetic
  cached video. Real dataset = disk reads + DWPose / MediaPipe / face crop
  per step → 2-5× slower per step expected.

**For our bridge v1 we don't need any of Stage 1/2/3 retraining.** The
distill is MSE-only on cached `(b_expr, m_f)` pairs; the closed-form
keypoint path needs no training at all. This is the v1's biggest win.

## Replacement opportunities (ranked by leverage × feasibility)

1. **MotEncoder student** (this plan, v1) — drives the entire ARKit
   bridge. ~1M params, MSE-only distill. **High leverage, high feasibility.**
2. **Motion-extractor closed-form** (this plan, v1) — eliminates the
   driving-frame keypoint detect entirely. **High leverage, high feasibility.**
3. **PoseGuider rebuild on richer canvas** (FLAME mesh, blendshape render).
   Requires Stage-1 retrain. **Medium leverage, medium feasibility.**
4. **TinyVAE swap** for `vae.decode` during inference (`vae_tiny_path` in
   PersonaLive's config). ~10× faster decoder, slight quality loss.
   **Low-medium leverage, high feasibility** — already a one-line config
   change in PersonaLive.
5. **2-step student** via Hyper-SD-style adversarial distill on top of
   the existing 4-step student. **High leverage, low feasibility** (week+
   of work, GAN-training instability).
6. **Per-subject ID LoRA** on denoising UNet for tight identity locking
   on a fixed reference. **Medium leverage, medium feasibility** (weekend
   of work per subject; needs ID-specific corpus).
7. **Replace base diffusion** (SD1.5 → Wan2.5 / LTX) — would require
   essentially full retrain. **High potential leverage, low feasibility.**

## Known gotchas (encountered or read about)

- **`mp.solutions.face_mesh` removed in mediapipe 0.10.x.** PersonaLive's
  inference scripts and `controlnet_aux.DWposeDetector` both break.
  Workaround in our pipeline: skip controlnet_aux, run motion_encoder
  directly on the LLF crop (face is roughly centered already by the
  selfie framing).
- **xformers / flash-attn falsified on Blackwell sm_120** for now (per
  `2026-05-03-xformers-flashattn-saga.md`). Stock SDPA is the floor; FA-2
  dispatches internally for fp16. Not blocking; just no easy attention
  acceleration win.
- **diffusers version pin:** PersonaLive's vendored UNet code is fragile
  against `diffusers>=0.30`. Our bridge tooling uses the vamp-interface
  `.venv` (diffusers 0.37); when calling PersonaLive code we sys.path-
  inject and hope no API drift bites. So far load_state_dict on
  pose_guider/motion_encoder/motion_extractor works clean. Denoising UNet
  through the offline pipeline likely needs PersonaLive's own venv.
- **Symlinks under `pretrained_weights/`:** PersonaLive expects
  `sd-image-variations-diffusers/` and `sd-vae-ft-mse/` at specific
  relative paths. Our bridge re-uses them via the existing PersonaLive
  install; don't move them.
- **`get_kp_info` / `get_kp` lives on the MotionExtractor class itself,
  not in `liveportrait/utils.py`.** Earlier draft of the bridge plan got
  this wrong. Correct call: `motion_extractor.detector(x)` returns the raw
  dict; `motion_extractor(x)` calls `get_kp(detector(x))` and returns the
  posed (B, 21, 3) tensor.

## Open questions / unknowns

- **PoseGuider on out-of-distribution k_d.** What happens if our closed-
  form k_d differs visibly from a real motion_extractor render? The dots
  could land on slightly different pixels; PoseGuider's CNN is mostly
  translation-equivariant in its first stages, so small shifts should be
  benign — but big shifts might shift the rendered face's positional bias
  in the latent. Sign-flip calibration (Task 9) handles the signed shifts;
  systematic offset would still be a calibration issue.
- **MotEncoder PE assumption.** Our student emits PE-included latents
  matching real MotEncoder's output. If MotEncoder's PE pattern is
  load-bearing for cross-attention (the UNet's queries learned to
  position-attend to specific PE channels), the student's emergent PE
  structure must mirror it. Provable by inspecting whether learned student
  outputs show sinusoidal structure across the L=32 axis.
- **Long-term identity drift under ARKit-only driving.** Stage 3's HKM
  uses real-frame motion banks for "have I seen this expression in my
  history." With ARKit-only driving the bank is populated with the
  student's outputs, not real MotEncoder outputs. Possible drift mode
  worth measuring.

## What we'd want to measure but haven't

- Actual per-component latency on RTX 5090, fp16, T=4, with vs without
  TRT (the *speculative* numbers above).
- Per-component VRAM footprint at inference (rough estimate ~12 GB total,
  but breakdown unknown).
- Sensitivity of generated quality (LPIPS, ArcFace ID) to MotEncoder
  feature noise (i.e., how tight does our student need to fit?).
- HKM bank cardinality vs identity drift on long ARKit-driven sessions.

Adding measurements is incremental: one bench script per row above. Most
useful to run after the bridge v1 distill lands, since then we can also
A/B "real MotEncoder driving frames" vs "student MotEncoder driving
frames" on identical b₆₁ inputs.

## Reading order for someone landing here cold

1. Paper §3 — `docs/papers/personalive-2512.11253.pdf` pages 3-5.
2. Bridge v1 design — [`2026-05-05-arkit-bridge-v1-design.md`](2026-05-05-arkit-bridge-v1-design.md).
3. This doc — components, perf, replacements.
4. Acceleration thread — [`_topics/personalive-acceleration.md`](_topics/personalive-acceleration.md).
5. Wrapper code — `~/w/PersonaLive/src/wrapper.py:280-365` for the online
   inference loop.
