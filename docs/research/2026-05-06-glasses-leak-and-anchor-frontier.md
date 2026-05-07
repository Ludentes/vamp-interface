---
status: live
topic: arkit-bridge
---

# Glasses leakage in `bridge` vs `personalive_rgb`, and the anchor OOD frontier (2026-05-06)

Two threads from the v3_lam10 first-look review, captured for follow-up after the yaw-sign fix lands.

## Glasses leakage on take 8

The driver wears glasses; the anchor (`asian_m__06_neutral.midframe.png`) does not. Glasses appear in the rendered output. `personalive_rgb` shows them prominently; `bridge` shows a faint trace.

Driver-RGB ingestion paths in PersonaLive + LivePortrait, and what each mode does with them:

| path | sees driver pixels? | active in `personalive_rgb` | active in `bridge` |
|---|---|---|---|
| LivePortrait `motion_extractor` → kp_d (21 implicit kp + s + t) | yes (CNN over RGB) | yes | **no — closed-form ARKit Euler → kp_d** |
| PersonaLive `motion_encoder` → m_f (1, 32, 16) for cross-attn | yes (CNN over RGB) | yes | **no — `MotEncoderStudent(b_61)` replaces it** |
| LivePortrait `appearance_extractor` over the **anchor** | no | yes | yes |

Predicts the observed delta. `bridge` should be glasses-free in principle because both driver-RGB paths are cut and the anchor has no frames.

### Why bridge still has *some* glasses

Two cheap diagnostics, both runnable in <10 min after calibration finishes:

- **(a) Anchor has subtle frame shadow.** Inspect `data/llf-phase2/asian_m__06_neutral.midframe.png` zoomed on the eye region. Faint dark crescents under the brow get amplified into frame-like outlines under extreme yaw warping. If yes, swap to a different `06_neutral` mid-frame; this is anchor-curation, not architectural.

- **(b) Student paints eye-region shadow.** Render `bridge` on take 8 with `b_expr[eyeSquintLeft/Right] = 0` and `eyeBlinkLeft/Right = 0`. If "glasses" disappear, m_f conditioning is the culprit (the student's eye-channel output drives the decoder to paint shadows that *resemble* frames). Otherwise the anchor is responsible.

Order them (a) → (b); (a) is a single eog inspection.

### Wider implication

For any glasses-wearing user, **`bridge` is structurally superior to `personalive_rgb`** — disentangled by construction, not by hope. LivePortrait's implicit-kp_d space is well known to leak appearance (multiple downstream papers note the residual-identity issue). ARKit-driven pose is closed-form; identity cannot enter.

This is publishable framing for the v4 writeup: the bridge is not just a latency win, it's a privacy/disentanglement win for any salient occluder (glasses, hats, masks, hair).

For RGB-only users (no ARKit available), the corresponding mitigation is a glasses pre-mask before `motion_extractor` (mediapipe eye-region landmarks → alpha-out frame regions). That's a PersonaLive-side patch, not a bridge-side patch — separate thread.

## Anchor frontier — measuring where the appearance encoder breaks

Current `anchors.parquet` has **1 row** (`asian_m`). Every diagnostic is conditioned on that one face. We need a measured frontier of where the LivePortrait `appearance_extractor` degrades, and we already have most of the ingredients.

### Anchor pool (~20 entries)

| category | source | n |
|---|---|---|
| real, demographic grid | `data/llf-phase2/*_neutral.midframe.png` (race × gender × age) | 6–8 |
| real, B&W | desaturate one of the above | 2 |
| stylized — Pixar/3D render | Flux v3 corpus, prompt "Pixar-style portrait of …" | 2 |
| stylized — oil painting | Flux corpus or one quick render | 2 |
| stylized — anime | Flux corpus | 2 |
| abstract — stick figure | hand-drawn 256×256 | 1 |
| non-human — VTuber/cat-girl/furry | Flux render or open VTuber asset | 2 |

### Schema additions to `anchors.parquet`

```
anchor_id            str
source_path          str
category             str    {real_photo, real_bw, pixar, painting, anime, stick, vtuber}
race                 str?   (real only)
gender               str?
age_bucket           str?
realism              str    {photo, painting, render, illustration, abstract}
has_glasses          bool
face_topology        str    {human, stylized_human, nonhuman}
notes                str
```

### Measurement protocol

One render per anchor, against a single high-range driver (take 5 or take 8), 60-second slice in `bridge` mode. All metrics flow into `render_metrics.parquet` automatically per the every-render-enriches-parquet rule.

Per-anchor scorecard:

- **head fidelity** — Pearson r(rendered_yaw, driver_yaw), pitch, roll. Real photos should hit r > 0.9; OOD anchors will degrade.
- **expression range preservation** — `std(bs_render) / std(bs_teacher_full)` per channel; mean across 51 channels for headline, kept per-channel to identify which expressions die first.
- **artifact rate** — fraction of frames where mediapipe `FaceLandmarker` fails on the rendered output. Catches catastrophic mode (anime → blob).
- **identity drift** — ArcFace `cos(anchor_crop, render_crop)` averaged over frames. Meaningful only for real-photo anchors. For stylized/non-human, replace with CLIP image-image cos against the anchor (proxy for "did the style survive").
- **visual collage** — 4×N grid, one row per anchor, columns at yaw=−15°, 0°, +15°, smile, blink. `docs/blog/`-ready.

### Predicted failure curve (to be measured)

In order from "works" → "broken":

1. real photo, no glasses
2. real photo, glasses (issue 1, residual ghost)
3. real B&W (slight desat drift)
4. Pixar render (eyes go subtly human, identity recognizable)
5. oil painting (texture-collapses toward photorealism — appearance_extractor's training prior wins)
6. anime (identity collapses to nearest-human topology; eye/mouth geometry mismatch)
7. stick figure (mediapipe finds no face; full breakdown)
8. VTuber non-human (same as 7 or unrecognizable hybrid)

This curve is itself a contribution — the CVPR PersonaLive paper does not quantify the OOD frontier of LivePortrait's `appearance_extractor`. Measuring it cleanly opens a credible v4 thread: replace the appearance_extractor with one trained on a richer style distribution (ID-Animator, Arc2Face-style ID conditioning, or a Flux-LoRA trained on stylized portraits).

### Cost

- 20 anchors × ~50 s render × 60 s slice ≈ 17 min GPU
- mediapipe extraction in parallel (CPU, free)
- collage assembly ~5 min
- writeup ~half day

### Run order

1. **Glasses experiments first** — both diagnostics ((a) anchor inspection, (b) zeroed eye blendshapes). 5 min after calibration finishes. Determines whether the residual glasses are an anchor problem or an m_f problem.
2. **Build anchor pool table** — commit `anchors.parquet` v2 with category column before any new render. The schema bump should be one shot, not amended later.
3. **Render the 20 anchors** — single batch overnight, side-by-side mp4s in `exp_output/arkit_bridge/render/anchor_sweep/`, register in `default_registry()` of `build_render_metrics_parquet.py`.
4. **Per-category scorecard** — one polars query, one collage, one `docs/blog/` post.

The anchor sweep also independently confirms or denies hypothesis (a) from the glasses thread, because asian_m's render against the same take will sit in the dataset alongside seven other clean-eye anchors. If asian_m is the only one with frame ghosts, it's the anchor.

## Acceptance gates

- Glasses threads: a one-line answer per diagnostic in this doc, plus a follow-on plan if neither (a) nor (b) explains the residual.
- Anchor sweep: a per-category scorecard CSV under `exp_output/arkit_bridge/anchor_sweep/scorecard.csv` and a 4×20 collage under `docs/blog/images/`.

## Dependencies (must land first)

- Yaw sign fix (`docs/research/2026-05-06-yaw-sign-flip-fix.md`) — anchor sweep should run with the empirically-correct EULER_SIGNS so that head-fidelity Pearson r is interpretable. If we run before the fix, all r values collapse together near zero on yaw and we lose the headline metric.
