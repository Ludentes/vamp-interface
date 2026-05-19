---
status: live
topic: arkit-controlnet
---

# CFM render-cache verification — FLAME render eyeball check

This doc records the manual verification of the CFM render-cache (Task 7 of the
render-cache plan). The goal: confirm the FLAME mesh, when posed from the cached
MediaPipe pose and deformed by a row's ARKit blendshapes, lands ON the face in
the photo and visibly tracks expression.

## Render-cache build stats

- **Pose detection rate (Task 6):** 69928 / 70000 FFHQ rows produced a usable
  head pose. 72 rows undetected — face too small / occluded / off-frame.
- **Pose / blendshape detection agreement:** 1.000. Every row where MediaPipe's
  face-landmarker returned blendshapes also returned a pose, and vice versa —
  the two detection passes never disagree, so `bs_detected` is a safe proxy for
  pose availability.

## Verification method

`src/arkit_controlnet/verify_flame_render.py` picks 8 FFHQ rows by expression
energy (`sum |bs|` over the 51 non-neutral ARKit channels): the 4 lowest-energy
(calm) and 4 highest-energy (expressive). For each row it deforms the FLAME
base mesh via `deform(mediapipe_to_basis_vector(bs))`, renders a normal-map
control image with `render(verts, rotation, bbox, modality="normals")` at the
photo's native resolution, and alpha-blends it (0.5/0.5) over the photo. The
result is a vertical 8-row collage, each row `photo | normal-map | overlay`.

## Per-row assessment

Rows top-to-bottom; rows 0–3 are calm, rows 4–7 are expressive.

- **Row 0 (calm, adult male, head tilted):** Expression correct — relaxed,
  closed mouth. Pose tracks the head's leftward yaw but the mesh sits a little
  high and large relative to the face; overlay covers forehead-to-chin with a
  modest vertical offset. Acceptable alignment, not pixel-perfect.
- **Row 1 (calm, boy, frontal):** Expression correct — neutral closed mouth.
  Pose frontal and matching; mesh slightly oversized so the chin extends below
  the photo chin, but eyes/nose/mouth land on the right features.
- **Row 2 (calm, toddler, near-frontal):** Expression correct — faint
  closed-mouth smile. Pose frontal; mesh sits slightly high, eyes land just
  above the photo eyes. Reads as aligned.
- **Row 3 (calm, child, frontal):** Expression correct — neutral. Best calm
  alignment of the four; eyes, nose and mouth overlay cleanly on the photo
  features.
- **Row 4 (expressive, man, tongue out + head turn):** Expression correct and
  striking — the mesh shows an open jaw with the tongue clearly protruding,
  matching the photo. Pose tracks the head's yaw and tilt well.
- **Row 5 (expressive, woman, laughing, eyes shut):** Expression correct —
  wide-open mouth and squeezed-shut eyes both visible on the mesh. Pose roughly
  matches; mesh slightly low so the open mouth overlays the photo's chin/lower
  lip rather than dead-center, but the expression signal is unambiguous.
- **Row 6 (expressive, man, open smile, head turn):** Expression correct —
  open smiling mouth. Pose tracks the head yaw cleanly; one of the better
  alignments in the expressive set.
- **Row 7 (expressive, woman, smile, hand on chin, head down+turn):**
  Expression correct — open smile. Pose follows the downward+yawed head;
  alignment good given the hand partially occludes the jaw in the photo.

## Verdict

**PASS with a known minor offset.** Across all 8 rows:

- **Pose:** reads correctly — yaw, tilt and downward pitch all track the photo
  head. No row has a grossly wrong rotation.
- **Expression:** the calm/expressive split is obvious. Calm rows (0–3) show
  closed, relaxed mouths; expressive rows (4–7) show open jaws (row 4 tongue
  protrusion, rows 5–7 open smiles) and, where present, squinted/shut eyes.
  Expression is clearly driven by the blendshapes.
- **Normal-map orientation:** NOT inside-out. The nose reads as a raised ridge,
  the face surface is convex with correct depth. No `order` flip is needed in
  `render`.

The one consistent imperfection is a small **vertical offset / scale mismatch**:
on several rows the mesh sits a touch high and slightly oversized, so the chin
overhangs the photo chin. This is a calibration nicety in the bbox→render
mapping, not a correctness failure — the mesh still lands on the right facial
features and tracks pose and expression. Fixable later by tightening the
bbox-to-camera fit; it does not block use of the render-cache as a CFM control
signal.

## Collage

![collage](../../exp_output/flame_render_check/collage.png)

(The PNG lives under `exp_output/` which is gitignored — regenerate locally with
`PYTHONPATH=src /home/newub/miniconda3/bin/python -m arkit_controlnet.verify_flame_render`.)
