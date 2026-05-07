---
status: live
topic: arkit-bridge
supersedes: 2026-05-06-yaw-sign-flip-fix.md
---

# ARKit-bridge: final pose config + lessons for the next distill

This is the wrap-up doc for the ARKit→PersonaLive bridge. The student
itself was settled days ago (`runs/student_v2_120k/student_best.pt`,
holdout_v3 ratio_mean = 0.0114, structural floor confirmed by LR/seed
sweep). This doc closes out the orthogonal axis-sign / pose-frame
problem — which turned out to live entirely in `closed_form_pose.py` —
and writes down the methodology lessons before we start the next
distill (LivePortrait teacher → smaller student / different driver).

## Final config

`src/arkit_bridge/closed_form_pose.py`:

```python
EULER_SIGNS = (+1.0, -1.0, +1.0)   # (yaw, pitch, roll)
F_KP_REF   = I                     # identity, no F-conjugation
```

The `compose_kd` path still computes `R_eff = F · R · F^T` for code
locality, but with `F = I` it reduces to `R_eff = R`. Renderer applies
`k_d = (kp_ref @ R) · s + t`.

This config was validated by 4-up side-by-side (`actor | teacher |
OLD bridge | FIXED bridge`) on three axis-isolated 3-second windows
selected by maximum dominant-axis range / quiet other axes:

| axis | take | ARKit frames | dominant range |
|---|---|---|---|
| yaw   | 5 | 1080–1260 | 76° |
| pitch | 2 | 1980–2160 | 52° |
| roll  | 3 | 1830–2010 | 47° |

User visual verdict on the FIXED column: all three axes match teacher.
Render artifacts under `exp_output/arkit_bridge/render/axis_clips_fixed/`.

## How we got here

The journey was longer than the final config suggests because three
separate bugs interacted, and we kept "fixing" downstream without
catching upstream confounders.

### Bug 1: training-data corruption from sideways frames

Earliest `data/arkit_bridge_pairs_v1_broken_sideways/` corpus was
built with cv2 returning landscape on raw .mov, so face_mesh saw
sideways faces. Caught and rebuilt; current `data/arkit_bridge_pairs/all/`
uses `cv2.CAP_PROP_ORIENTATION_AUTO=1` and is clean. v2_120k trained on
the clean corpus.

### Bug 2: rotation flag in `apply_bridge_to_personalive.py`

The script does **not** set `CAP_PROP_ORIENTATION_AUTO`. `--rotate_iphone`
defaults to True (applies `np.rot90(rgb, k=-1)`). The rule is
path-dependent:

- Raw .mov in `data/llf-takes/` → cv2 returns landscape → keep default
  rotate=True
- ffmpeg-transcoded mp4 in `data/llf-clips-auto/` or `data/llf-takes-small/`
  → ffmpeg already rotated → use `--no_rotate_iphone`

Memory file `feedback_iphone_rotation_default_wrong.md` was wrong with
"always pass `--no_rotate_iphone`" — that's true only for transcoded
mp4, not raw .mov.

The bug only bites `--mode teacher_full` and `--mode teacher_motion`,
because `--mode bridge` patches the seams that consume RGB
(`patch_pose=True` and `patch_motion=True` ignore the input frames and
use the closed-form pose path + student instead). All bridge-mode
renders are unaffected by the rotation flag.

This bug masqueraded as a take-3 source corruption because we hit it
on a teacher_full render of take 3 with `--no_rotate_iphone`. Once
diagnosed: only one corrupt artifact existed and was re-rendered. No
training contamination. Long-term fix is to put
`cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)` into
`apply_bridge_to_personalive.py` and drop the `--rotate_iphone` flag.

### Bug 3: F-conjugation from contaminated calibration

Calibration v3 (`exp_output/arkit_bridge/calibration_v3/`) ran a frame
search on a 600-frame yaw clip and reported winners
`EULER_SIGNS=(+1,-1,-1)` and `F* = diag(+1,-1,+1)` with
`F R F^T` "frame conjugation". Those numbers were
**confounded by Bug 2**: the calibration clip had been processed with
the wrong rotation, so the correlation surface the search optimised
over had geometry that wasn't representative of the actual frame.
Worse, **L2 over implicit keypoints is partially yaw-symmetric** (per
`feedback_calibration_blind_spots.md`) — the calibration metric can't
distinguish yaw mirrors that the renderer can; a ~5% gap on yaw fell
within search noise.

The OLD config `signs=(+1,-1,-1) F=diag(1,-1,1)` produces effective
applied rotation `(yaw, +pitch, +roll)` because F-conjugation negates
both X (pitch) and Z (roll) components of `Rz · Ry · Rx`. That is:
`F R_x(θ) F^T = R_x(-θ)`, `F R_z(θ) F^T = R_z(-θ)`, `F R_y F^T = R_y`.
Effective signs after conjugation are `(sy, -sp, -sr)`.

Symptoms with OLD config (visible in our axis-isolated mediapipe
re-extraction):
- yaw: matches actor (the pipeline-wide yaw mirror is downstream of
  closed_form_pose; teacher and bridge both have it)
- **pitch: `corr(MP_pitch, ARKit_pitch) = +0.93`, but MP_pitch range
  6.4° vs teacher's 46° — sign correct, amplitude crushed to 13%**
- **roll: `corr(MP_roll, ARKit_roll) = -1.00` — sign inverted vs
  teacher**

The pitch amplitude collapse with OLD never showed up in offline
ratio_mean. Only rendered-quality measurement caught it.

### The intermediate "FIXED" config that broke roll

We tried `signs=(+1,+1,+1) F=diag(1,-1,1)` first — effective
`(yaw, -pitch, -roll)`. That fixed pitch (full amplitude restored at
sign matching teacher) but roll then went the wrong way. F=diag(1,-1,1)
**couples pitch and roll signs**: any conjugation of that form negates
both X and Z components together, so you cannot fix one axis without
flipping the other.

### The final config

The bind broke when we accepted that F = I + per-axis EULER_SIGNS gives
the only configuration where every axis is independently controllable:

```
signs = (+1, -1, +1), F = I
→ effective rotation = R_z(+roll) · R_y(+yaw) · R_x(-pitch)
```

Per-axis target mapping derived from the OLD/intermediate-FIXED data:
- yaw effective +1 (both OLD and intermediate produced visually-correct
  yaw)
- pitch effective −1 (intermediate's pitch was correct, OLD's was sign
  +1 but amplitude-crushed)
- roll effective +1 (OLD's roll was correct; intermediate flipped it)

There may still be a pipeline-wide yaw mirror (mediapipe ZYX
decomposition reads `corr = -1.00` against ARKit on every variant
including teacher), but it cancels uniformly because everything
downstream is consistent with itself. The actor's visible motion in
the side-by-side matches all three downstream variants on yaw.

## Methodology lessons for the next distill

A separate distill run is queued for LivePortrait (likely a
motion-encoder distill or a smaller student of LivePortrait's
implicit-keypoint head). The lessons that earned themselves the right
to be standing rules:

### Verify training data before training

Before any non-trivial training run on a new corpus, render a 60-frame
slice through the teacher inference path and a 60-frame slice through
the student-target extraction path. Gate on
`cos(extracted_target, teacher_inference_target) > 0.95` per
`feedback_verify_training_data_first.md`. Without this, silent corpus
corruption (Bug 1's flavour) burns 1–2 days before a downstream metric
hints at it.

### Offline `ratio_mean` is not a quality oracle

The OLD config's pitch-amplitude collapse (13% of teacher's range)
never surfaced in `ratio_mean`. The student fits the implicit-keypoint
target distribution well; whatever the *renderer does with those
keypoints* is invisible to a per-channel L2/ratio loss. Always close
the loop with a rendered-quality measurement before declaring a
config done. Our methodology:

1. Render full takes (or axis-isolated 3-sec clips) through teacher,
   bridge, OLD-bridge.
2. Extract per-frame mediapipe blendshapes + facial transformation
   matrix → per-frame (yaw, pitch, roll) via ZYX Euler decomposition.
3. Append to `exp_output/arkit_bridge/parquet/render_metrics.parquet`
   via `scripts/build_render_metrics_parquet.py`.
4. For each (variant, take), compute `bs_cos_to_teacher`,
   `bs_amp_ratio`, `ypr_l1_err`, `region_energy_diff`,
   `bs_temporal_jitter` via `scripts/v4_bakeoff_downstream_metrics.py`.

### Calibration metrics must be able to distinguish what they're asked to

Calibration v3's L2-over-implicit-keypoints couldn't tell yaw mirrors
apart from the correct yaw because the implicit keypoint cloud is
near-symmetric under yaw flip up to ~5%. A grid search returned a
nominally-best F that produced visually mirrored output. The fix
applied here was to reframe calibration as a perceptual A/B against
visual ground truth: render axis-isolated sweeps and compare to actor
side-by-side. Per `feedback_calibration_blind_spots.md`: always
verify a calibration metric can distinguish what it's asked to
distinguish; move metric downstream of any symmetries; the grid you
enumerate must be a superset of the true ambiguity.

### Axis-isolated test stimuli reveal what mixed-motion takes hide

Three 3-second windows (each one axis dominant, ARKit frames selected
by max dominant-axis range and minimum range on the other two axes)
gave us ground-truth-vs-render correlations per axis that the
4-second-mixed-motion holdout couldn't deliver. The filter logic is
in scripts/find_axis_windows.py-style ad-hoc Python in this session;
worth promoting to a reusable script before the next distill.

### Sign-convention bugs travel in twos and threes

We had three independent sign / rotation issues all interacting:

1. cv2 rotation honoring (Bug 2)
2. ARKit's reported Euler frame vs the renderer's expected frame (the
   F-conjugation question)
3. Mediapipe's ZYX Euler decomposition vs ARKit's HeadYaw/Pitch/Roll
   axis convention (the `corr = -1.00` on yaw that exists across all
   variants including teacher)

When a render looks wrong, enumerate all three before patching any
one. Patching the rotation matrix to fix issue 3 will mis-correct
issue 1's symptom.

### Write the inventory before deleting

When destruction (delete take 3, retrain) was on the table, a 15-min
non-destructive verification path was available (read scripts to see
what corpus was used + check rotation handling). Doing the inventory
first turned a "delete + 24h retrain" into "fix one render, audit
done." Before any destructive action, list the artifacts that will be
re-derived and confirm they're actually contaminated rather than
suspected.

### Mode interaction matters more than flags

`apply_bridge_to_personalive.py` has three modes (`bridge`,
`teacher_motion`, `teacher_full`) with different seam-patching
behaviour. A flag like `--rotate_iphone` only matters in modes that
consume the unmasked input. Future distill scripts should print a
"effective inputs that will be consumed" summary at startup so the
operator sees which flags actually affect the run.

## What ships

- **Student weights**: `runs/student_v2_120k/student_best.pt`
- **Pose config**: `src/arkit_bridge/closed_form_pose.py` with
  `EULER_SIGNS=(+1,-1,+1)`, `F_KP_REF=I`
- **Render driver**: `scripts/apply_bridge_to_personalive.py`. For raw
  .mov input use default `--rotate_iphone=True`; for transcoded mp4
  pass `--no_rotate_iphone`. The flag only matters in
  `--mode teacher_full` / `--mode teacher_motion`; bridge mode is
  rotation-invariant.
- **Eval**: `scripts/build_render_metrics_parquet.py` +
  `scripts/v4_bakeoff_downstream_metrics.py` cover offline ratio_mean
  and rendered-quality scorecards.

## What didn't make it (deferred to LivePortrait distill)

- Auto-detect rotation in apply_bridge_to_personalive.py
  (`cv2.CAP_PROP_ORIENTATION_AUTO=1` everywhere) — drop the
  `--rotate_iphone` flag entirely.
- Promote axis-isolated window selector to a reusable script.
- Investigate the pipeline-wide yaw `corr = -1.00` between mediapipe
  ZYX decomposition and ARKit HeadYaw — likely a measurement-side
  convention mismatch, not a render-side bug, but worth confirming
  with a frontal-symmetry test.
- Consider whether F-conjugation should be removed from
  `compose_kd` entirely now that calibration confirmed F = I; the
  matmul is cheap but the code path invites future confusion.
