---
status: live
topic: metrics-and-direction-quality
---

# Overnight 11-axis screening — visual + MediaPipe metrics

## TL;DR

Rendered 990 prompt-pair characterization samples (11 axes × 6 bases × 3
seeds × 5 α) without attention capture to avoid filling disk. Visual
screening said 6 strong / 3 caveat / 2 falsified. MediaPipe blendshape
scorer on the 5 blendshape-measurable axes refines the picture: the
visual call was right on direction in all 5 cases but miscalibrated on
magnitude for 3 of them. Net keepers for the slider-training corpus:
roughly 8 of the 11 axes, though `brow_furrow`'s base-dependence is a
real caveat.

## Method

- Output: `output/demographic_pc/fluxspace_metrics/crossdemo/<axis>/<axis>_inphase/<base>/s<seed>_a<alpha>.png`
- Edit: `FluxSpaceEditPair` scale=1.0, start_percent=0.15, α∈{0, .25, .5, .75, 1}, mix_b=α
- 6 crossdemo bases from `BASES_FULL`, 3 seeds (2026/4242/1337)
- MediaPipe scorer: `score_blendshapes.py` on the full corpus
  (4428/5416 PNGs scored; failures are non-face renders at extreme
  scales in unrelated `verify/` subtree, not ours)
- Per-axis analysis: `analyze_overnight_metrics.py`

## Per-axis verdicts

### Blendshape-measurable (5)

**eye_squint** — STRONG
- Target `sum(eyeSquintLeft + eyeSquintRight)` went 0.08 → 0.98 across
  α, monotonic, effect +0.90, Δ ≥ 0.71 on all 6 bases.
- Visual read underclaimed as "decent with identity drift". Identity
  drift is real but the target AU is clean.

**gaze_horizontal** — STRONG
- Target `sum(eyeLookOutL/R + eyeLookInL/R)` went 0.43 → 1.48,
  monotonic, effect +1.05, Δ ≥ 0.82 on all 6 bases.
- Visual read underclaimed as "subtle, only some bases". Metric says
  otherwise — signal is large and consistent.

**brow_lift** — CONFIRMED with caveat
- Target `sum(browInnerUp + browOuterUpL/R)` went 0.64 → 1.05, effect
  +0.41 but **non-monotonic** (dip at α=0.25 below baseline).
- Base-dependent: young_european_f/southasian_f/asian_m/european_m
  Δ > 0.4; black_f and elderly_latin_m Δ ≈ 0.14–0.18 (likely ceiling
  effect from higher baseline brow_lift on those bases).

**brow_furrow** — OVERCLAIMED
- Target `sum(browDownL/R)` went 0.10 → 0.28, effect +0.18,
  non-monotonic (peak at α=0.75, then falls back).
- **Fails on 3/6 bases**: asian_m Δ = -0.001, black_f Δ = +0.004,
  southasian_f Δ = +0.011. elderly_latin_m carries the whole signal
  (Δ = +0.73).
- Visual screening said "strong across bases" — false. Training a
  slider on this corpus would fit an inconsistent signal.

**mouth_stretch** — FALSIFIED AS STRETCH, CONFIRMED AS JAW_OPEN
- Target `sum(mouthStretchL/R)` *decreased* 0.06 → 0.01 with α.
- Cross-check `jawOpen`: 0.003 → 0.54 — 180× increase.
- Prompt text "mouth stretched wide horizontally, lips pulled taut
  sideways" pulls the model toward shocked-open-mouth instead. Rename
  the axis to `mouth_open` or rewrite prompts for true horizontal
  stretch (`mouthDimpleL/R + mouthStretchL/R` target).

### SigLIP-2 probe scored (6)

Using `score_overnight_siglip.py` (piggybacks on `Siglip2Backend` from
`score_clip_probes.py` — the filename is a pre-migration artefact; the
backend is `google/siglip2-so400m-patch16-384` per the
2026-04-23 VLM survey recommendation).

**age** — CONFIRMED STRONG, CROSS-BASE CONSISTENT
- `elderly` margin −0.15 → +0.04, monotonic, effect +0.185
- Per-base Δ +0.15–0.21 (very consistent)

**gender** — CONFIRMED STRONG, MOST CONSISTENT
- `feminine` margin −0.13 → +0.07, monotonic, effect +0.205
- Per-base Δ +0.19–0.22 (tightest spread of all axes)

**hair_style** — CONFIRMED STRONG
- `long_hair` margin −0.05 → +0.10, monotonic, effect +0.148
- Per-base Δ +0.09–0.22 (moderate base variance)

**hair_color** — CONFIRMED MODERATE
- `black_hair` margin −0.05 → +0.05, monotonic, effect +0.094
- Per-base Δ +0.06–0.13. Half the effect size of hair_style,
  consistent with the visual observation that bald bases partially
  confound hair_color with hair_presence.

**skin_smoothness** — VISUAL CALL *REFUTED*, signal is weak-but-real
- `rough_skin` margin −0.01 → +0.06, monotonic, effect +0.068
- Per-base Δ +0.04–0.08 (small but consistent)
- Visual screening said "no visible difference" — thumbnail
  resolution hid the signal. Keep as marginal axis.

**nose_shape** — CONFIRMED BROKEN
- `aquiline_nose` margin +0.0004 → +0.0065, effect +0.006
  (10–30× smaller than every other axis)
- Per-base noisy, elderly_latin_m goes *negative* (Δ=−0.006)
- Noise-floor response. Drop.

## Re-prompt v2 results (2026-04-24 evening)

Re-rendered `mouth_stretch` and `brow_furrow` with rewritten prompt pairs
(`src/demographic_pc/reprompt_axes.py`, 180 renders). Scored with MediaPipe
and compared against v1 via `analyze_reprompt_v2.py`.

**mouth_stretch_v2** — FIXED directionality
- New prompts: A=`lips gently together in a calm closed mouth`,
  B=`lips pressed firmly together and pulled wide sideways in a tight
  grimace, mouth remains closed`
- Target `mouthStretchL/R`: v1 effect −0.053 (wrong direction) → v2 effect
  **+0.044, monotonic**.
- `jawOpen` leakage cut ~3×: v1 went 0.003 → 0.54, v2 0.016 → 0.19.
- Per-base: 5/6 bases now positive (vs 0/6 in v1). `black_f` still
  negative (Δ = −0.076).
- Verdict: keeper. Some shocked-open-mouth still bleeds in but the axis
  is now directionally correct.

**brow_furrow_v2** — PARTIALLY FIXED
- New prompts: A=`relaxed eyebrows at rest in a neutral expression`,
  B=`both eyebrows strongly pulled down and pressed together, brows knit
  tightly` (pure muscle phrasing, no age-coded creases/wrinkles).
- Target `browDownL/R` effect grew +0.18 → **+0.26** and is now monotonic.
- Previously-dead bases rescued: `black_f` +0.004 → +0.046 (10×),
  `southasian_f` +0.011 → +0.083 (7×). `european_m` doubled (+0.31 →
  +0.65).
- **`asian_m` still flat** (Δ = −0.0003 → −0.0003). The mechanical
  phrasing fixed 5/6 bases but left the East-Asian male base
  unresponsive.
- Verdict: keep with `asian_m` caveat. If the slider-training corpus
  needs even cross-base coverage, either drop `asian_m` samples from
  this axis or attempt an `asian_m`-specific prompt variant.

## Axis-keeper recommendations for slider-training corpus

Keep as-is (7, if we confirm the visual ones with classifier metrics):
- age, gender, hair_style, hair_color (pending classifier confirmation)
- eye_squint, gaze_horizontal (metric-confirmed strong)
- brow_lift (metric-confirmed with base-dependent ceiling)

Re-prompt-fixed (2, see v2 section above):
- mouth_stretch_v2 → directionally correct now; keep
- brow_furrow_v2 → 5/6 bases respond; keep with `asian_m` caveat

Drop (2):
- skin_smoothness, nose_shape — no signal from text pair. Defer to
  image-anchor slider training if we ever need these axes.

Combined with existing corpora (smile, jaw, anger, surprise, disgust,
pucker, lip_press, glasses — 8 axes) the prompt-pair dictionary reaches
**15 candidate axes** once mouth_stretch and brow_furrow are fixed.

## Cross-thread implications

- Confirms that prompt-pair axis characterization is cheap and produces
  useful training-data corpora (~90 renders per axis × 15 axes ≈ 1350
  renders total to characterize the whole dictionary).
- Recasts our pipeline's role after the 2026-04-23 cached-δ replay
  falsification: we're a **training-data generator + composition
  validator** for eventual Concept-Sliders distillation. See
  `project_editing_framework_positioning.md` in memory.
- Refutes the earlier "NMF k=8 is enough" intuition for the axis count
  question indirectly: several distinct AU axes (brow_lift, brow_furrow,
  eye_squint, gaze) respond cleanly to text pairs, suggesting the
  production slider dictionary should probably cover O(20) axes,
  not 8. NMF is a compressed measurement basis, not the edit dictionary.

## Artifacts

- Raw scores: `output/demographic_pc/overnight_blendshapes.json` (4428
  PNG → 52-d blendshape maps)
- Per-axis collages: `docs/research/images/2026-04-24-overnight-<axis>.png`
- Master overview: `docs/research/images/2026-04-24-overnight-master.png`
- Analysis script: `src/demographic_pc/analyze_overnight_metrics.py`
- Render script (resumable): `src/demographic_pc/overnight_new_axes.py`
