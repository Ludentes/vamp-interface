---
status: live
topic: neural-deformation-control
supersedes: 2026-05-05-arkit-poseguider-distill-plan.md
---

# ARKit → PersonaLive bridge v1 — design (paper-faithful)

Companion plan: [`2026-05-05-arkit-bridge-v1-plan.md`](2026-05-05-arkit-bridge-v1-plan.md).
Supersedes [`2026-05-05-arkit-poseguider-distill-plan.md`](2026-05-05-arkit-poseguider-distill-plan.md).

The "v1" tag is deliberate: this design implements the paper's hybrid
conditioning (closed-form implicit keypoints + distilled facial motion
embedding) for the **deployment direction** (ARKit-only driving, no driving
RGB). Future versions may add translation tracking, learned keypoint
calibration, or replace the MSE distill with a diffusion-loss fine-tune.

## Why supersede the prior plan

The prior plan distilled a single ARKit-fed student against a frozen
**Moore-AnimateAnyone** PoseGuider on **DWPose** images. Two errors:

1. **PersonaLive's PoseGuider does not consume DWPose.** It consumes a
   `draw_keypoints` rendering of LivePortrait 3D implicit keypoints —
   7 colored dots at indices `[1, 2, 3, 4, 12, 15, 20]` from the 21-keypoint
   set (`src/utils/util.py:337`). A DWPose-trained student would produce
   features the denoising UNet was never trained to consume.
2. **PoseGuider carries head pose, not expression.** Per paper Section 3.1,
   "implicit facial representations focus solely on local facial dynamics
   [expression], we further introduce 3D implicit keypoints to capture
   global pose, position, and scale information." The two paths are
   architecturally distinct.

## Paper architecture (Section 3.1)

```
{k_{c,d}, R_d, t_d, s_d} = E_k(I_D)                 # driving
{k_{c,s}, R_s, t_s, s_s} = E_k(I_R)                 # source / reference

k_d = s_d · k_{c,s} · R_d + t_d                     # paper eqn. (3)

m_f = E_f(I_D)                                       # facial motion (FAN-based)

A = D( M=[m_f, k_d], R=ReferenceNet(I_R), z_t )     # denoising
```

`E_k` = LivePortrait MotionExtractor (108 MB, ConvNeXtV2-tiny). `E_f` = FAN-SA
MotEncoder (236 MB).

**Critical paper detail:** equation (3) uses `k_{c,s}` (source/reference's
canonical keypoints), not `k_{c,d}`. Identity geometry comes from the
reference; only **rotation, translation, scale** carry driving information.
Confirmed in code: `~/w/PersonaLive/src/liveportrait/motion_extractor.py`
line 123 holds `kp = kp1['kp']` (reference) for all frames, and
`interpolate_kps_online` (`s_scale=0`, `t_scale=0.5`) deliberately blends
driving's t/s back toward reference's. PoseGuider sees a head-pose-rotated
copy of the reference's identity geometry, not the driving's.

## Decomposition for an ARKit-only bridge

| Path | Conditioning | Driving signal we have | Implementation |
|---|---|---|---|
| Implicit keypoints `k_d` → PoseGuider | head pose, position, scale | ARKit head yaw/pitch/roll (3 floats) | **closed-form math** — no learning |
| Implicit facial representation `m_f` → cross-attn | expression | ARKit 52 blendshapes + 6 eye rotations (58 floats) | **distilled MLP student** |

### Closed-form keypoint path (no learning)

At session start, run real `E_k(I_R)` once and cache:
- `kp_ref` — canonical keypoints, shape `(1, 21, 3)`
- `t_ref` — translation, shape `(1, 3)`
- `s_ref` — scale, shape `(1, 1)`

Per driving frame i:
```python
R_d_i = euler_to_rotmat(b₆₁[i, 52], b₆₁[i, 53], b₆₁[i, 54])  # yaw, pitch, roll
k_d_i = (kp_ref @ R_d_i) * s_ref
k_d_i[..., 0:2] += t_ref[..., 0:2]                            # follow E_k convention
canvas_i = draw_keypoints(k_d_i, height=512, width=512)        # real PersonaLive function
pose_fea_i = pose_guider(canvas_i)                             # real PersonaLive module
```

Holding `s_d ≈ s_s` and `t_d ≈ t_s` matches `interpolate_kps_online`'s
defaults. For VTuber-style use (head approximately centered) this is
appropriate. If head-translation tracking matters later, ARKit's
4×4 facial transformation matrix's translation component can be plumbed
through Live Link's "Animation Data" stream — out of scope for v0.

This path is **algorithmically correct relative to the paper**. There is no
distillation loss to converge; the only knob is whether the radian
convention out of LLF (HeadYaw is yaw about Y-up in iPhone camera frame)
matches LivePortrait's pitch/yaw/roll convention. That's a calibration
question (sign flips) verifiable by rendering one neutral frame and
comparing canvases.

### Distilled expression path

Student `S_expr: ℝ⁵⁸ → ℝ¹⁶×ℝ³² → ℝ¹⁶×ℝ³²` (output PE-augmented to match
real MotEncoder convention). Architecture: 4-layer MLP, ~0.7–1M params.

Teacher: real `motion_encoder(face_crop_224.unsqueeze(2))` per frame.
Per-frame stateless (verified — encoder.py:38 rearranges
`b c f h w → (b f) c h w` before the FAN trunk; no temporal mixing).

Loss: per-frame MSE on the (1, 32, 16) output. The PE addition happens
*inside* MotEncoder; the student should output PE-included values to be
drop-in compatible.

Per-frame supervised triples `(b₆₁_i, m_f_target_i)` extracted from the
33K-frame Live Link Face corpus already on disk.

## Why this design and not "twin students"

An earlier draft (preserved in git as the brief uncommitted twin-student
sketch) proposed two students because it conflated PoseGuider's
**deployment input** (a rendered 7-dot canvas) with **deployment-time
necessary work** (rendering that canvas). The paper makes clear that the
canvas is computed deterministically from {kp_ref, R_d, t_ref, s_ref}; only
R_d varies per frame, and ARKit gives R_d directly. No mapping to learn.

Net effect: replacing the twin with single-student reduces:
- Training complexity (one loss, one student).
- Cached-data per pkl (drop T_pg, ~2.5 MB/frame; keep T_me, ~1 KB/frame).
  Real corpus pair cache shrinks from ~40 GB to ~30 MB.
- Failure modes (Student-A could have collapsed; closed-form can't).

## Inference seam in PersonaLive

Two adapters in `wrapper.py`:

```python
# Replace pose_guider's call site at wrapper.py:330
def patched_pose_guider(_unused_keypoints):
    return cached_pose_fea  # precomputed from kp_ref + ARKit head rotations

# Replace motion_encoder's driving-frame call at wrapper.py:309
def patched_motion_encoder(_unused_face_stack):
    # ref slot is m_f_ref (real, computed once); driving slots from student
    return torch.cat([m_f_ref_cached, student(b_expr_per_frame)], dim=1)
```

`m_f_ref_cached` is computed from the real reference RGB once at session
start. Same for `kp_ref/t_ref/s_ref` — both groups of cached values come
from real PersonaLive modules; we only swap *driving-frame* outputs.

## Training data (already on disk)

| Take | ARKit frames |
|---|---|
| MySlate_2 | 7440 |
| MySlate_3 | 3212 |
| MySlate_4 | 813 |
| MySlate_5 | 7340 |
| MySlate_6 | 5208 |
| MySlate_7 | 6434 |
| MySlate_8 | 2597 |
| **total** | **33,044** |

Stride=2 → ~16,500 pairs. Pair cache size ≈ 16,500 × (244 B + 1 KB) ≈ 20 MB.
The whole student corpus fits in RAM.

## Out of scope for this design

- Stage-2 / temporal modules of PersonaLive — handled by `temporal_module.pth`,
  separate concern; the bridge is per-frame.
- Real-time deployment plumbing — Phase 4 of the unified plan.
- Diffusion-loss "replace" fine-tune — gated on this MSE-only distill's
  per-channel readouts.
- Translation-tracking (`t_d ≠ t_s`) — Live Link only sends rotation in
  its 9-float wire format. Selfie-style head-roughly-centered behavior is
  the v0 contract.

## Risks

- **ARKit-to-LivePortrait Euler convention mismatch.** ARKit head Euler is
  iPhone camera frame (Y up, Z out of screen). LivePortrait's Euler comes
  from a face-detector head-pose head; sign conventions could differ.
  Mitigation: at first render, compute pose_fea via real motion_extractor on
  one frame, compare against pose_fea via closed-form path. If mismatched,
  it's a sign flip on one of {yaw, pitch, roll} — find by enumeration (8
  combinations, 1 minute).
- **Student collapse on long-tail blendshapes** (e.g., `tongueOut`, asymmetric
  smiles) if our 7 takes don't exercise them. Per-channel sensitivity sweep
  catches this. Fix: record more diverse takes (recipe in plan).
- **MotEncoder PE.** The student outputs (1, 32, 16) values that should
  already include the sincos PE that real MotEncoder adds in its forward.
  We can either (a) train the student against PE-included targets and trust
  it learns the additive structure, or (b) target PE-stripped values and
  add PE downstream. Picking (a) for simplicity — failure mode visible in
  output stats (PE has known sinusoidal structure).

## Self-review

- **Placeholder scan** — none load-bearing. Every shape, weight path, hook
  seam, loss, and corpus identified.
- **Internal consistency** — the closed-form path mirrors PersonaLive's own
  `interpolate_kps_online` defaults; the student's I/O matches MotEncoder's
  exact per-frame slice; no contradictions with paper or code.
- **Scope check** — focused on the bridge. The closed-form path is ~30 LoC;
  the student path is ~200 LoC plus dataset/eval boilerplate.
- **Ambiguity check** — Student input is 58 floats (52 blendshapes + 6 eye
  rotations). Eye roll is held at 0 from Live Link Face's wire format
  (Apple-side limitation). Rotation magnitudes from ARKit are radians.

## Reading order

1. This doc — `2026-05-05-arkit-bridge-v1-design.md`
2. Plan — `2026-05-05-arkit-bridge-v1-plan.md`
3. Paper section 3.1 — `docs/papers/personalive-2512.11253.pdf` page 4
4. Wrapper code — `~/w/PersonaLive/src/wrapper.py:280-360`
5. Earlier (superseded) plan — `2026-05-05-arkit-poseguider-distill-plan.md`
