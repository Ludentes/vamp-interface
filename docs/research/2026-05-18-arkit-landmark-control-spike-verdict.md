---
status: live
topic: arkit-controlnet
---

# ARKit landmark-control spike — verdict

Spike of "Path 1": feed a **MediaPipe 478-vertex face-mesh control image** into
InfiniteYou's InfuseNet spatial-control slot and see whether it steers FLUX's
output expression while ArcFace identity holds — zero training. This follows the
falsified zero-train FluxSpace spike
(`2026-05-18-arkit-controlnet-spike-verdict.md`), swapping the failed expression
channel (attention editing) for InfuseNet's own control slot.

3 FFHQ identities × 3 axes (`smile`/`pucker`/`surprise`) + a neutral baseline ×
strength `{0.6, 1.0}`, fixed seed 2026, 25-step euler. RTX 5090, 24/24 cases
rendered.

- Workflow: `comfyui/workflows/arkit_landmark_spike.json`
- Runner: `src/arkit_controlnet/run_landmark_spike.py`
- Selection + mesh render: `src/arkit_controlnet/landmark_control.py`
- Renders + metrics: `exp_output/arkit_landmark_spike/` (`collage.png`,
  `metrics.parquet`)

## Verdict: negative — the dense mesh control does not carry expression

The InfuseNet slot was trained on a 5-keypoint control image; a 478-vertex
tessellation is out-of-distribution for those weights, and the spike confirms
the **central risk** named in the design: the slot has essentially no
expression bandwidth for a dense mesh.

### Success criteria — 2 of 9 cells pass

Criterion: at some swept strength, identity holds (`arcface_cos ≥ 0.55`) **and**
expression moves toward target — `expr_cos(output, exemplar)` beats the
same-identity, same-strength neutral-control output by ≥ 0.05.

| identity | smile | pucker | surprise |
|---|---|---|---|
| d70eb952 | fail | fail | fail |
| 14b641b8 | fail | **pass** (+0.108) | **pass** (+0.052) |
| 9a0f66e6 | fail | fail | fail |

Only `14b641b8` produced any positive cell, and `surprise` only barely clears
the +0.05 floor. Several deltas are strongly **negative** (`9a0/smile` −0.404,
`d70/pucker@1.0` −0.237) — the mesh control actively perturbed expression
*away* from target, the signature of an OOD control image acting as noise
rather than signal.

### Identity collapse at strength 1.0

For 2 of 3 identities, `strength = 1.0` destroys the face entirely —
`arcface_cos = -1.0`, no detectable face (`9a0` on all axes, `14b` on
pucker/neutral). Only `d70` survived str 1.0, and `d70` independently fails the
identity floor even at str 0.6 and under the neutral control (`arcface ≈ 0.35`),
so it is a weak-identity case throughout. There is no strength that is both
strong enough to bite and weak enough to keep identity — the same
high-control-strength collapse the FluxSpace spike showed at scale ≥ 1.5.

### Eyeball gate — failed

`collage.png` (strength 0.6): within each identity row, the smile / pucker /
surprise / neutral columns are visually near-identical. The control mesh
produces no expression change a viewer can see. Identity *is* roughly preserved
across the row — the InfuseNet identity-token path still works, consistent with
the prior spike. It is specifically the **control slot as an expression
channel** that fails.

## Implication for the thread

Both zero-train routes are now falsified for expression: FluxSpace attention
editing (prior spike) and the InfuseNet control slot driven by a dense mesh
(this spike). Identity injection via InfuseNet remains solid in both. The
expression channel must be **trained** — CFM against `(photo, expression-render)`
pairs, as in `2026-05-16-arkit-controlnet-infiniteyou.md`.

## Fallback before committing to a training run

Per the design's documented fallback, one cheap experiment remains before CFM:

- **Sparse-contour modality** — render only lip / eye / face-oval polylines from
  the same 478 landmarks (closer to the 5-keypoint training distribution than a
  dense tessellation). One run; if it also fails, the slot has no expression
  bandwidth at all.
- **Approach C — stacked depth ControlNet** — render a depth map from the
  landmarks and drive a *stock* FLUX Depth ControlNet alongside identity-only
  InfuseNet (InfiniteYou documents plug-and-play ControlNet stacking). A depth
  ControlNet has the dense-spatial bandwidth the 5-kp-trained InfuseNet slot
  lacks. Still no FLAME, still no training.

If both fail, the thread proceeds directly to the CFM training run.
