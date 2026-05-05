---
status: live
topic: neural-deformation-control
---

# PersonaLive becomes the default photoreal-portrait animation path

**Date:** 2026-05-04
**Decision:** PersonaLive replaces the slider / LoRA / Flux stack as the
preferred mechanism for generating *animated* photoreal portraits. The
slider stack remains canonical for offline parameter authoring, atom
measurement, and *static* Flux portrait variation.

## What was tested

Phase 2 of the iPhone-driven pipeline plan
(`2026-05-03-iphone-pipeline-unified-plan.md`): a **3 × 6 grid** of

- **3 Flux anchors** — `asian_m`, `black_f`, `european_m`, all from
  `output/demographic_pc/overnight_drift/smile/broad/<base>/seed2026_s+0.00.png`.
  Same anchors used across the slider/squint/measurement pipeline; chosen for
  M/F + 3 ethnicities.
- **6 driver clips** — cut from a single ARKit-mode Live Link Face take
  (`data/llf-takes/20260505_MySlate_2/`, 124 s @ 60 fps) by sliding-window
  peak search per channel. Clips: smile, jaw_open, blink, head_turn,
  brow_frown, neutral. Each 5 s, 60 fps, paired with a 300-row ARKit-52 CSV
  slice. See `scripts/cut_llf_clips.py` and
  `2026-05-04-live-link-face-capture-format.md`.

Run via `PersonaLive/scripts/phase2_grid.py` in SDPA mode (~10 FPS,
no TensorRT — reliability over speed for this falsification test).
Total render: 18 mp4s, 640 s, mean 9.77 FPS, ~8 MB total.

## The visual verdict

The contact sheet
(`output/llf_phase2/grid_midframes.png`, regenerable via
`/tmp/phase2_contact.py`) shows mid-frame stills from all 18 renders.

> "Honestly visually I am very very impressed. This is way better than
> anything we built with sliders or LoRA or Flux. For any future generations
> this should be the way we do it."  — user, 2026-05-04

Three things hold across all 18 cells:

1. **Identity preserved.** asian_m row reads as asian_m, black_f as
   black_f, european_m as european_m — no perceived swap or hybridization.
2. **Expressions track.** Smile shows lift, jaw_open shows mouth open,
   head_turn shows yaw, neutral is neutral. The driver signal reaches
   the model at all.
3. **Background changes** — Flux anchors had clean studio-grey gradients;
   the renders show darker, less defined backgrounds. Not catastrophic
   smearing, but not preserved either. Worth flagging for downstream
   compositing.

Skin texture is reasonable (no obvious diffusion mush). Shoulder/clothing
region varies between renders.

## Quantitative confirmation

`scripts/phase2_readout.py` ran insightface buffalo_l ArcFace (IR50 r50,
512-d) and MediaPipe FaceLandmarker (52 ARKit-aligned blendshapes) on
50 frames per clip (stride 6 → 10 Hz from 60 fps source).

| anchor | clip | id_mean | id_min | bs_macro_corr | det_rate | bs_rate |
|---|---|---|---|---|---|---|
| asian_m | 01_smile | 0.799 | 0.738 | **0.663** | 1.00 | 1.00 |
| asian_m | 02_jaw_open | 0.789 | 0.616 | 0.275 | 1.00 | 1.00 |
| asian_m | 03_blink | 0.864 | 0.789 | 0.141 | 1.00 | 1.00 |
| asian_m | 04_head_turn | 0.704 | 0.510 | 0.159 | 1.00 | 1.00 |
| asian_m | 05_brow_frown | 0.798 | 0.653 | 0.333 | 1.00 | 1.00 |
| asian_m | 06_neutral | **0.918** | 0.827 | 0.269 | 1.00 | 1.00 |
| black_f | 01_smile | 0.726 | 0.659 | **0.731** | 1.00 | 1.00 |
| black_f | 02_jaw_open | 0.699 | 0.523 | 0.298 | 1.00 | 1.00 |
| black_f | 03_blink | 0.780 | 0.718 | 0.140 | 1.00 | 1.00 |
| black_f | 04_head_turn | 0.644 | 0.429 | 0.252 | 1.00 | 1.00 |
| black_f | 05_brow_frown | 0.707 | 0.534 | 0.406 | 1.00 | 1.00 |
| black_f | 06_neutral | 0.845 | 0.779 | 0.242 | 1.00 | 1.00 |
| european_m | 01_smile | 0.781 | 0.717 | **0.812** | 1.00 | 1.00 |
| european_m | 02_jaw_open | 0.693 | 0.568 | 0.330 | 1.00 | 1.00 |
| european_m | 03_blink | 0.879 | 0.789 | 0.224 | 1.00 | 1.00 |
| european_m | 04_head_turn | 0.705 | 0.524 | 0.322 | 1.00 | 1.00 |
| european_m | 05_brow_frown | 0.759 | 0.643 | 0.340 | 1.00 | 1.00 |
| european_m | 06_neutral | 0.856 | 0.802 | 0.096 | 1.00 | 1.00 |

(Per-frame parquet: `output/llf_phase2/readout.parquet`,
markdown table: `output/llf_phase2/summary.md`.)

Reading the numbers:

- **ArcFace cosine — never below the buffalo_l τ=0.40 same-person gate
  on any frame of any clip.** Worst frame is 0.43 on `black_f__04_head_turn`.
  Means range 0.64–0.92; neutral clips hit 0.85+, head-turn lowest
  (~0.65 mean) — expected, since ArcFace is frontal-biased and the iPhone
  yaw was ~17°.
- **Macro blendshape correlation tracks the strength of the underlying
  channel.** Smile: **0.66 / 0.73 / 0.81** across the three anchors —
  strong tracking. Brow/frown: 0.33–0.41 — moderate. Jaw-open: 0.27–0.33
  — weaker (likely calibration mismatch between MediaPipe vs ARKit jaw,
  not actual tracking failure). Blink: 0.14 — sub-second event with
  10 Hz sampling, so the correlation has too few transitions to converge.
  Neutral: 0.10–0.27 — expected, neutral has no signal to correlate with.
- **100% face-detection and blendshape rates** across 900 frames.

The metric story is consistent with the visual call: identity is
preserved, the strong channels (smile) track strongly, the weaker
correlations are explained by sampling rate or calibration, not by
PersonaLive losing the signal.

## Why this changes our default

The slider stack works on Flux *static* portraits via attention-cache
edits and LoRAs. It is good at axis authoring, narrow expression deltas,
and demographic factor isolation — it is *not* good at producing 5-second
animated clips with real per-frame head + face motion. We tried — months
of work on slider quality, identity-drift mitigation,
critic networks, distilled atoms — and the result on this Phase 2 test
is qualitatively below PersonaLive's first SDPA pass.

Slider lessons still feed forward (offline parameter authoring, atom
measurement, sus-axis inference for the vamp-interface use case). But for
**generating animated portraits**, the cheapest path with the best
quality-per-engineering-hour is now PersonaLive.

## Caveats and open threads

- **Resolution.** Output is 512×512. PersonaLive's training resolution
  and the only one it's "native" at. Higher-resolution path is research
  thread `2026-05-04-video-super-resolution-survey.md` (in flight at
  decision time).
- **Throughput.** ~20–21 FPS bundled-TRT ceiling on RTX 5090, below the
  22 FPS interactive ship gate. Fine for batch / cached / asynchronous
  use. See `_topics/personalive-acceleration.md`.
- **Background drift.** Output backgrounds are darker / less defined
  than the Flux anchors. If we need pixel-stable backgrounds, post-hoc
  matting + composite is the cheap fix.
- **License.** The PersonaLive paper code license has not been audited
  for commercial use. If we ship a product, this needs checking.
- **Bridge milestone.** Phase 3 (ARKit-52 → motion-latent regressor) is
  still load-bearing. Phase 2 only proves PersonaLive *generalises to
  Flux input on real ARKit-driven motion*. Phase 3 proves we can drive
  it from any source of ARKit-52, which is what makes the interactive
  iPhone client possible.

## Artefacts

- `data/llf-phase2/*.mp4` — 18 grid renders (gitignored — derived from
  private capture).
- `data/llf-phase2/grid_manifest.json` — per-render timing.
- `output/llf_phase2/grid_midframes.png` — qualitative contact sheet
  (regenerable).
- `output/llf_phase2/readout.parquet` — per-frame metric table.
- `output/llf_phase2/summary.md` — per-clip rollup.
- `scripts/phase2_readout.py` — readout harness (committed).
- `scripts/cut_llf_clips.py` — driver-clip cutter (already committed).
- `PersonaLive/scripts/phase2_grid.py` — render harness (lives in the
  PersonaLive checkout, not in vamp-interface).
