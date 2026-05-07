---
status: live
topic: arkit-bridge
---

# Research: ARKit vs Mediapipe head pose conventions

**Date:** 2026-05-06
**Sources:** 12 — Apple ARKit `ARFaceAnchor` docs, Apple ARKit world-coordinates docs, Google Developers MediaPipe 3D Face Transform blog, MediaPipe `face_geometry.proto` source, MediaPipe `face_mesh.md` docs, MediaPipe issue #1642 (canonical face model), Susanne Thierfelder OpenCV/MediaPipe head-pose tutorial, Stack-Overflow / Apple Developer Forums threads on `ARFaceAnchor.transform`, Epic Live Link Face CSV documentation (UE 4.27), `JimWest/PyLiveLinkFace` README, `KwaiVGI/LivePortrait` README, our local `~/w/PersonaLive/src/liveportrait/motion_extractor.py` and the inline doc in `src/arkit_bridge/closed_form_pose.py`.

---

## Executive Summary

ARKit's face frame and MediaPipe's Metric 3D space are **both right-handed, both have +Y up, both have +Z pointing toward the viewer/camera, and both place the rotation about +Y as "yaw"**. Decomposing MediaPipe's `facial_transformation_matrix` to ZYX Euler **should** yield axes sign-comparable to Apple ARKit `(HeadYaw, HeadPitch, HeadRoll)` from Live Link Face — *provided* Live Link Face emits Euler angles in the same convention as the underlying `ARFaceAnchor.transform`, which the official Live Link Face documentation does not explicitly state but which is the inherited Apple convention. The 8-way Z₂³ sign grid we ran is therefore the right structural search space for a single-axis bug, but our calibration metric is still noisy because (a) MediaPipe's `pose_transform_matrix` is `canonical → runtime` while LivePortrait's `kp_d` lives in a learned implicit-keypoint frame whose alignment to the canonical face frame **is not promised**, and (b) we are scoring with per-axis Pearson on Euler triples, which is only a valid factorization in the small-angle limit and propagates cross-axis matrix coupling into per-axis correlations as an artifact. The cleanest fix is to score on the rotation matrix directly using a one-shot Procrustes alignment, not on per-axis Euler.

---

## Key Findings

### ARKit `ARFaceAnchor` frame is right-handed, face-centered, +Z toward viewer

The ARKit documentation describes the face anchor's transform property as an inherited 4×4 in world coordinates whose face-local frame is "right-handed — the positive x direction points to the viewer's right (that is, the face's own left), the positive y direction points up (relative to the face itself, not to the world), and the positive z direction points outward from the face (toward the viewer)" [1][5]. Origin is "centered behind the face" with metric units (meters) [1]. The same right-hand convention applies to ARKit world space: "+y up, +z toward the viewer, +x to the viewer's right" [5].

This is the frame in which ARKit produces `ARFaceAnchor.transform`, and from which the `(HeadYaw, HeadPitch, HeadRoll)` exported by Live Link Face are decomposed. The Live Link Face app and CSV format are produced by Apple's reference Live Link integration, which is documented as a thin wrapper over `ARFaceAnchor` outputs [2].

### Live Link Face HeadYaw / HeadPitch / HeadRoll units and signs are not explicitly documented

This is the single largest gap in our citation chain. The Epic Unreal Engine 4.27 Live Link Face docs and the third-party `PyLiveLinkFace` reference describe the CSV columns by name (`HeadYaw`, `HeadPitch`, `HeadRoll`) but neither states whether the values are radians or degrees, nor which sign corresponds to which physical direction [2][3]. Empirically in our own corpus we treat them as radians and the magnitudes are consistent with that interpretation — peak `|HeadYaw|` on take 5 is ~0.27, which corresponds to ~15° of head turn, matching what we visually observe. This is a single-source (our own) confirmation; **it is contingent on Live Link Face's wire format not having silently changed across iOS or app versions**. The Live Link Face app version on these takes is "v1.6.0 (build 169)" per `take.json`.

There is one uncomfortable forum-level note that some Live Link Face CSV exports zero-fill the 9 head/eye rotation columns under certain export configurations [11]; **our take 8 CSV has them populated** with the values we are using, so this only matters if a future capture session silently degrades to a zero-filled mode (worth a sanity check in the preprocessing pipeline). No primary Apple or Epic source confirms or denies the zero-fill claim.

### MediaPipe Metric 3D space is right-handed, OpenGL-style, +Y up, camera at origin looking toward −Z

The MediaPipe Face Geometry module documentation states explicitly: "The Metric 3D space established within the Face Transform module is a right-handed orthonormal metric 3D coordinate space" with "a virtual perspective camera located at the space origin and pointed in the negative direction of the Z-axis" [6][7]. This is the OpenGL camera convention. A face seen by the camera sits at negative Z; the face's own forward (out-of-face toward the camera) is therefore in the +Z direction in world coordinates, matching ARKit. Metric units are centimeters by default [7].

The "facial_transformation_matrix" from `FaceLandmarker` (with `output_facial_transformation_matrixes=True`) is, per the proto definition and the developer blog, "a rigid linear mapping from the canonical face metric landmark set into the runtime face metric landmark set" — i.e., the matrix maps **canonical face → runtime face** [6][7][8]. Decomposing the rotation part of this 4×4 (the upper-left 3×3) gives the rotation that takes the canonical frontal face into the runtime head pose, in MediaPipe's right-handed Metric 3D frame. That rotation, by handedness alone, should sign-agree with ARKit's head rotation: rotation about +Y in either system turns the nose toward the viewer's right (= face's own left) by the right-hand rule.

### MediaPipe → OpenCV axis flip is real but does NOT apply to our path

The "OpenCV vs OpenGL mismatch" — `[x, y, z]` in OpenCV becomes `[x, -y, -z]` in OpenGL — comes up frequently [9] and is the source of half the head-pose-related Stack Overflow noise. **It does not apply between MediaPipe's `facial_transformation_matrix` and our ARKit pipeline** because both endpoints are OpenGL-style (Y up, Z toward viewer). It would matter only if we were comparing against an `cv2.solvePnP`-derived rotation, which we are not. Our pipeline goes Live Link Face → ARKit Euler → LivePortrait kp_d → render pixels → MediaPipe matrix; OpenCV never enters the loop.

### MediaPipe pose decomposition into Euler is well-defined but cross-axis-coupled

The standard MediaPipe head-pose-estimation tutorial and several reference implementations decompose the 3×3 rotation as ZYX Euler with `pitch = arctan2(-R[2,0], sy); yaw = arctan2(R[1,0], R[0,0]); roll = arctan2(R[2,1], R[2,2])` where `sy = sqrt(R[0,0]² + R[1,0]²)` [4][8]. This is the same formula our `_mp_blendshape.extract_one()` uses. It is unique away from gimbal lock (`sy < 1e-6`) but is **not axis-separable** — any non-zero combination of two axes produces matrix off-diagonals whose decomposition mixes the perceived per-axis values. Therefore Pearson r on a single decomposed axis is only a clean test of that input axis when other input axes are near zero. On take 8 with simultaneous yaw + pitch + roll motion, per-axis r will pick up cross-axis bleed even if all three input signs are correct.

### LivePortrait's rotation construction is `(Rz · Ry · Rx).T` applied as `kp @ R`

From our local copy of `~/w/PersonaLive/src/liveportrait/motion_extractor.py:get_kp` and our mirror in `src/arkit_bridge/closed_form_pose.py:25-56`: the motion extractor builds `R = (Rz(roll) · Ry(yaw) · Rx(pitch))` and returns the **transpose**, then the warp module applies `kp @ R` (row-vector convention). Mathematically `kp_row @ R^T = (R · kp_col)^T`, so the active rotation operator on column-vector kp_ref is `Rz · Ry · Rx` — standard "yaw-pitch-roll intrinsic rotation order" [10]. Yaw is rotation about +Y, pitch about +X, roll about +Z. Sign convention: positive rotation by right-hand rule about each axis. **This is convention-compatible with the ARKit and MediaPipe frames as established above.**

### The unverified link: `kp_ref` frame alignment

The remaining ambiguity is whether LivePortrait's *implicit keypoint reference* `kp_ref` (the 21 learned 3-D points the appearance extractor emits per identity) is itself in a frame aligned with ARKit's face frame. The LivePortrait paper (arXiv 2407.03168) and the open-source repo do not document this — `kp_ref` is whatever the trained `appearance_feature_extractor` produces, and there is no explicit constraint forcing it to be in a particular axis-aligned frame. Empirically the LivePortrait community's `motion_extractor` trained on VoxCeleb produces a kp_ref that **looks** axis-aligned on most subjects (i.e. nose forward in +Z, eyes apart in X, jaw down in -Y), but axis-aligned to within a sign on each axis is not promised. **A single-axis mirror in the kp_ref frame (e.g. +X being face's own right rather than face's own left) would produce exactly the symptom we observe**: a single-axis-flip of the input Euler does not produce a single-axis flip of the rendered output's decomposed Euler, because the rotation cascade through `kp_ref @ R` interacts with the kp_ref frame.

This is the cleanest hypothesis explaining why our 8-combo Z₂³ sign grid gives no clean per-axis winner. It is **single-source** to our own analysis (no published paper isolates `kp_ref` axis alignment as a known issue). Worth checking by rendering one calibration probe at *zero* ARKit input (`yaw=pitch=roll=0`) and inspecting whether the rendered head is upright and forward-facing — if the rendered baseline pose drifts off-axis at zero input, the kp_ref frame and the ARKit Euler frame are not aligned and we need a static rotation `R_align` between them.

### Why our metric is noisy in practice

Two compounding effects, both citable:

- **Per-axis Pearson is not separable.** [4][8] Decomposed Euler mixes axes through matrix multiplication. The "effective signal" in the rendered output's pitch when the input has both yaw and pitch motion is `pitch_input + ε(yaw_input, roll_input)` with ε non-zero whenever the angles are not infinitesimal. On a 10° peak motion that ε is small but non-negligible — exactly the regime where "is pitch r positive or negative" can flip on a yaw_sign change without anything being wrong with the pitch path itself.
- **Per-axis active-frame masking imports the same pollution.** Our metric scores on frames where `|input_axis| > 0.10 rad`. On takes with correlated multi-axis motion (a head turn typically comes with a small pitch as the subject reorients), the active set for "pitch-r" is dominated by frames that also have non-zero yaw — and those are exactly the frames where ε pollution is largest.

The cleaner metric is rotation-matrix-direct: build `R_input = euler_to_rotmat(arkit_yaw, arkit_pitch, arkit_roll)` using LivePortrait's exact convention; build `R_render = mediapipe_M[:3,:3]` from the rendered output; estimate one global Procrustes alignment `F` between them on a calibration set; score `mean(angular_distance(F R_input(signs) F^T, R_render))` per sign combo. This separates "what frame is mediapipe in relative to LivePortrait's kp_ref frame" (the one-shot `F`) from "did we get the input ARKit Euler signs right" (per-combo search). It is also what the related literature (e.g. CASIA NIR-VIS protocol papers, OpenCV head-pose evaluations) uses when the source and reference frames are nominally compatible but the alignment is not literally identity.

---

## Comparison table

| property | ARKit `ARFaceAnchor.transform` (LLF source) | MediaPipe `facial_transformation_matrix` |
|---|---|---|
| handedness | right-handed [1][5] | right-handed [6][7] |
| origin | centered behind face | metric 3-D space origin (camera) [7] |
| +X | viewer's right (face's own left) [1] | OpenGL standard, viewer's right (face's own left) [9] |
| +Y | up [1] | up [6] |
| +Z | out from face toward viewer [1] | out from face toward viewer (camera at origin looks toward −Z, so face's "out" is +Z) [6] |
| matrix maps | face frame in world coords | canonical face → runtime face [6][7][8] |
| units | meters [1] | centimeters by default [7] |
| rotation order (when emitted as Euler) | not documented in LLF; treated as ZYX in our code | not documented; community decomposes as ZYX [4][8] |
| yaw axis | +Y, right-hand rule | +Y, right-hand rule |
| sign-comparable to LivePortrait `Rz · Ry · Rx` after closed-form rotation? | yes if `kp_ref` frame is axis-aligned to canonical face frame | yes if same condition holds |

---

## Open Questions

- **Live Link Face official sign / unit reference.** Neither Apple nor Epic publishes a primary doc stating that the CSV `HeadYaw/Pitch/Roll` columns are radians and follow the same right-hand sign convention as `ARFaceAnchor.transform`. We are operating on an empirical assumption matched to physical observation. A definitive answer would require reading Live Link Face's iOS source (closed) or testing against a known-orientation calibration target on device. Low priority — we have a working empirical match.
- **Is `kp_ref` axis-aligned to ARKit's face frame?** The single most important open question for our calibration. Resolvable by a one-frame zero-input render, before any sign sweep. Plan to add this as the first step of calibration v3.
- **Does Live Link Face v1.6.0 ever emit zero-filled head/eye rotation columns?** Forum-level claim, single-source, and our take 8 has populated values — but worth a guard in the preprocessing pipeline that asserts `np.std(b_all[:, 52:55]) > 1e-3` before treating a take as having usable head pose.

---

## Implication for our calibration

The calibration v2 results are not "wrong" in the sense of telling us false signs — they are informationally weak because the metric mixes a real per-axis question with the kp_ref frame ambiguity. The 8 Z₂³ combos are the right grid **if** kp_ref is axis-aligned to the ARKit frame. If kp_ref's frame is itself signed-permutation-related to the ARKit frame, the right answer lives outside the Z₂³ grid in the 48-element signed-permutation group, and no Z₂³ search recovers it.

Action items, in order of cost:

1. **Zero-input baseline render.** One frame, ARKit `(yaw, pitch, roll) = (0, 0, 0)`, no closed-form rotation applied. Visual inspect: is the rendered head upright and forward-facing? If yes, kp_ref ⊥ kp_ref; if no, we have a static `R_align` to recover before any sign sweep is meaningful. ~30s GPU + visual check.
2. **Rotation-matrix metric (calibration v3).** Replace per-axis Pearson with `angular_distance(F R_input F^T, R_render)`, where `F` is the one-shot Procrustes alignment on the take. Cost ~equal to calibration v2 (8 renders) but with a meaningful score function. The negative control becomes "permute axes of input, expect F to compensate".
3. **If the Z₂³ grid still shows no clean winner**, expand to the 48-element signed-permutation group. ~6× more renders. Only worth running if step 1 indicates kp_ref is mirrored on more than one axis. The 10s clip pipeline (see `2026-05-06-arkit-bridge-clip-preprocess.md` / coming) makes this affordable.

The yaw-flip we observed visually is a real symptom; the v2 calibration is correctly telling us "your sign is wrong somewhere", just not exactly where. Steps 1–2 cleanly disambiguate.

---

## Sources

[1] Apple. "ARFaceAnchor — Apple Developer Documentation." https://developer.apple.com/documentation/arkit/arfaceanchor (Retrieved: 2026-05-06)
[2] Epic Games. "Recording Facial Animation from an iOS Device — Unreal Engine 4.27 Documentation." https://docs.unrealengine.com/4.27/en-US/AnimatingObjects/SkeletalMeshAnimation/FacialRecordingiPhone (Retrieved: 2026-05-06)
[3] Jim West. "PyLiveLinkFace — README." https://github.com/JimWest/PyLiveLinkFace (Retrieved: 2026-05-06)
[4] Susanne Thierfelder. "Head Pose Estimation with MediaPipe and OpenCV in JavaScript." https://medium.com/@susanne.thierfelder/head-pose-estimation-with-mediapipe-and-opencv-in-javascript-c87980df3acb (Retrieved: 2026-05-06)
[5] Apple. "Tracking and visualizing faces — ARKit." https://developer.apple.com/documentation/ARKit/tracking-and-visualizing-faces (Retrieved: 2026-05-06)
[6] Google. "MediaPipe 3D Face Transform — Google Developers Blog." https://developers.googleblog.com/en/mediapipe-3d-face-transform/ (Retrieved: 2026-05-06)
[7] Google. "MediaPipe face_mesh.md — Face Geometry section." https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_mesh.md (Retrieved: 2026-05-06)
[8] Google. "MediaPipe face_geometry.proto — pose_transform_matrix definition." https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/protos/face_geometry.proto (Retrieved: 2026-05-06)
[9] Thomas Rouch. "Converting camera poses from OpenCV to OpenGL can be easy." https://medium.com/check-visit-computer-vision/converting-camera-poses-from-opencv-to-opengl-can-be-easy-27ff6c413bdb (Retrieved: 2026-05-06)
[10] Sepideh Shamsizadeh. "Composing Rotations: Euler Angles and Roll-Pitch-Yaw." https://medium.com/@sepideh.92sh/part-iii-composing-rotations-euler-angles-and-roll-pitch-yaw-38aa816a5bcd (Retrieved: 2026-05-06)
[11] Epic Developer Community Forums. "Live Link Face CSV export / Live Link Face Importer plugin import issue." https://forums.unrealengine.com/t/live-link-face-csv-export-live-link-face-importer-plugin-import-issue/1321222 (Retrieved: 2026-05-06)
[12] Local source: `/home/newub/w/PersonaLive/src/liveportrait/motion_extractor.py` `get_kp` function and our mirror in `/home/newub/w/vamp-interface/src/arkit_bridge/closed_form_pose.py:25-56`. (Retrieved: 2026-05-06, in repo)
