---
status: live
topic: arkit-bridge
---

# Yaw sign-flip fix in closed-form pose path (2026-05-06)

## TL;DR

`src/arkit_bridge/closed_form_pose.py:19` — `EULER_SIGNS[0]` was `+1.0`, should be `-1.0`. Set to `(-1.0, -1.0, -1.0)` after direct measurement on rendered output. Pitch and roll were correctly calibrated; only yaw fell through.

The student weights need no retraining: the closed-form pose path is a parallel non-learned shim that maps ARKit head Euler → LivePortrait keypoints, and is only active in `mode=bridge` (not in `teacher_full`). The fix lives entirely in inference.

## Diagnostic timeline

User watched the first v3_full bridge renders (takes 2/3/4 of the v3_lam10 1200-frame batch) and called out two things: "we might have flipped the yaw" and "head positioning a bit attenuated on the extremes". The amplitude question is downstream of the sign — the sign was the load-bearing claim and worth confirming first.

`render_metrics.parquet` already had teacher_full and bridge ypr per frame for takes 2/3/4 (and teacher_full for takes 5 and 8). Sign-agreement query (`scripts/diagnose_signs_and_amps.py`):

```
ch        n_active    agree   status
yaw            496    0.135   FLIP
pitch          507    0.797   OK
roll          1305    0.918   OK
```

Direct correlation across takes 2/3/4 (n=2924 matched frames):

```
Pearson r(yaw_b,  yaw_t) = -0.736
Pearson r(yaw_b, -yaw_t) = +0.736
median(yaw_b / yaw_t) on active frames = -0.799
std(yaw_t)=0.155, std(yaw_b)=0.156   (magnitudes match)
```

Per-take corroboration on take 5 (out-of-sample relative to the 2/3/4 finding):

```
yaw:   r=-0.874  sign_agree=0.000  median(b/t)=-0.72  n_active=77
pitch: r=+0.992  sign_agree=0.995  median(b/t)=+0.81  n_active=212
roll:  r=+0.962  sign_agree=0.989  median(b/t)=+1.06  n_active=853
```

**Sign agreement of exactly 0.000 on 77 frames is not noise. Pure noise = 0.50; pure negation = 0.00.** The bridge produces `yaw ≈ −teacher_yaw` deterministically across takes.

## Why calibration missed it

The original `scripts/calibrate_euler_signs.py` enumerated 8 sign combos and minimised `mean(L2(k_d_cf, k_d_real))` where `k_d_real` is PersonaLive `motion_extractor` output on the driver RGB. Looking at `exp_output/arkit_bridge/calibration/myslate_2.json`:

```
(+1, -1, -1): 0.0902   ← chosen as best
(-1, -1, -1): 0.0946   ← only 4.9% worse — within sample noise
```

Three failure modes stacked, in decreasing importance:

- The metric is **partially yaw-symmetric** on a near-symmetric kp_ref. LivePortrait's 21 implicit keypoints sit on a roughly L↔R-symmetric facial topology (paired eyes, mouth corners, jaw silhouette). `kp @ R(+yaw)` and `kp @ R(−yaw)` produce mirror-image clouds whose L2 to a near-symmetric target differs only by the kp_ref's residual asymmetry (~5%). Pitch and roll do not share this symmetry (face is not mirror-symmetric forehead↔chin or hair↔neck) and were calibrated correctly.
- Calibration metric was on the **wrong space** — internal motion-extractor keypoints rather than the rendered output's mediapipe-extracted yaw. The actual user-visible quantity is the latter.
- **Sample SNR was low.** 30 random frames per take, signal scales with `sin(yaw)`, many frames near-frontal. Take 5 already noted "ties within 0.005 RMSE" — the calibration was telling us it couldn't discriminate; we treated it as a passing margin.

A bonus structural reason: the 8-way grid is a Z₂³ subgroup of the full signed-permutation group (48 elements); if the true ARKit→LivePortrait mapping is a signed permutation outside the diagonal sign-flip family, the grid picks the closest projection, which can have the wrong yaw sign.

Saved as `feedback_calibration_blind_spots.md` so future calibration of any axis on near-symmetric data (gaze L/R, mouth-corner pull asymmetry) avoids the same trap.

## The fix

```diff
-EULER_SIGNS = (+1.0, -1.0, -1.0)  # (yaw, pitch, roll)
+EULER_SIGNS = (-1.0, -1.0, -1.0)  # (yaw, pitch, roll)
```

Smoke-tested by re-rendering take 2 with the new sign and re-running the same correlation query. Acceptance: `r(yaw_b, yaw_t) > +0.7` and `sign_agree > 0.85` on take 2's active-yaw frames.

If smoke passes, all 7 takes re-render in background; the existing `student_v3_lam10` checkpoint is unchanged.

## New calibration metric (replacement)

`scripts/calibrate_euler_signs.py` to be replaced by `scripts/calibrate_euler_signs_v2.py` whose loop is:

1. Pick a take with strong yaw motion (take 5 — `n_active≈77`, range ~±15°).
2. For each of 8 sign combos:
   - Render ~120 frames through PersonaLive in `bridge` mode with that sign.
   - mediapipe FaceLandmarker → per-frame ypr on the rendered output.
   - Score: `corr(rendered_yaw, input_yaw) + corr(rendered_pitch, input_pitch) + corr(rendered_roll, input_roll)`. Higher is better.
3. Pick combo with highest sum-correlation; report all 8 for transparency.
4. **Negative control:** synthetically negate one input axis and verify the new metric picks the negation. If it doesn't, the metric is still degenerate — escalate.

This metric breaks symmetry by going through the renderer (texture, hair, ears, garments are not mirror-symmetric) and aligns the optimised quantity with the user-visible quantity. The cost is ~8 × 30s = 4 min of GPU time per calibration run, vs ~30s for the L2 version.

## What this does not fix

- **Yaw amplitude attenuation** — bridge magnitude is ~0.7–0.9× teacher. Unrelated to sign; needs separate investigation. Possibly student-side, possibly PersonaLive's own motion-encoder attenuation visible only at extremes. Decompose with the 3-amp metric (`amp_personalive`, `amp_arkit_path`, `amp_student`) once render_metrics is full.
- Other potential sign issues on **eye-look** axes (`eyeLookInLeft / Right` etc.) — the bridge's b_61 channels [52:58] include LeftEye and RightEye yaw/pitch/roll which use the same `EULER_SIGNS` calibration in spirit but are routed differently through `apply_bridge_to_personalive`. Q1 sign-agreement showed those channels at 78–82% (above flip threshold), so probably fine, but worth a confirming look once full data is in.

## Acceptance gates

- Smoke (this doc): r(yaw_b, yaw_t) > 0.7 on take 2 after fix.
- Full re-render: r > 0.7 on all 7 takes; sign_agree > 0.85 on every take with n_active > 50.
- Negative control on new calibration script: synthetic sign-scramble produces a clearly different best.
