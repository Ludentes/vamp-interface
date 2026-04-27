---
status: live
topic: demographic-pc-pipeline
---

# Glasses slider v4 step 600 — quality eval

First slider evaluated under the
[measurement procedure](2026-04-26-slider-quality-measurement.md). v4 is
the η=4 hail-mary regime; step 600 is the first checkpoint where
glasses appeared on females during training-time sample renders.

## Setup

- Checkpoint: `glasses_slider_v4_000000600.safetensors`
- Trainer config: `ai-toolkit/config/glasses_slider_v4.yaml` (xattn scope,
  η=4, lr=1.25e-4, EMA off, 1500 steps)
- Render: 9 prompts (3 in-dist, 6 held-out) × 9 strengths
  (-2.5..+2.5) × 3 seeds = 243 cells, FluxPipeline diffusers bf16,
  guidance 3.5, 28 steps, 512×512
- Score: ArcFace IR50 via insightface buffalo_l; SigLIP-2-so400m on
  6 probes (glasses, hair_long, hair_curly, earrings, formal_clothing,
  beard); MediaPipe FaceLandmarker (52 blendshapes)
- Output: `models/sliders/glasses_v4/glasses_slider_v4_000000600/`
  (renders + eval.parquet + collage + summary)

## Decision: one-sided axis

Glasses fails the original bidirectional pass criteria
(separation @ ±1.5 = 0.06 vs ≥ 0.3 required). Adding glasses to a face
is a concept; "removing glasses" from a baseline that doesn't have them
isn't, so the negative half just drifts (identity collapses below
s=−1.5 with no semantic gain). The procedure now has a one-sided code
path; glasses lives in `ONE_SIDED_AXES`. Future axes default
bidirectional and only join after the data shows a flat negative half.

## Results (one-sided)

**Spearman ρ over s ∈ [0, +2.5]** on `siglip_glasses` margin:

- in-distribution: 0.77
- held-out: 0.82

Below the 0.9 bar because s=+0.5 is essentially indistinguishable from
s=0 — the slider doesn't engage until ~+1.0.

**Fraction of cells with glasses present** (`siglip_glasses > 0`) is
the cleaner human-aligned metric:

| s | in-dist | held-out |
|---|---|---|
| 0.0 | 0% | 0% |
| +0.5 | 22% | 0% |
| +1.0 | 89% | 67% |
| +1.5 | 100% | 100% |
| +2.5 | 100% | 100% |

Engagement strength ≈ +1.0. Coverage saturates by +1.5. Generalisation
gap at +1.0 (in-dist 89% vs held-out 67%) closes by +1.5.

**Identity (ArcFace cos vs same-prompt-same-seed s=0 baseline)**:

| s | mean | floor |
|---|---|---|
| +0.5 | 0.79 | OK |
| +1.0 | 0.54 | borderline |
| +1.5 | 0.42 | at threshold |
| +2.5 | 0.30 | lost |

17/108 positive cells in s∈[0,+1.5] fall below 0.4; almost all of those
sit at +1.5. Past +1.5, identity collapses (consistent with v4's
aggressive η=4 regime overshooting).

## Zoom: 0 → +1.5 at 0.25 step, 4 seeds (2026-04-27)

After the wide grid suggested operating point ~+1.0, a finer
zoom-render fills in the engagement curve. 144 new cells: 3 new
strengths (0.25, 0.75, 1.25) × 3 existing seeds + a 4th seed (7777)
across all 7 strengths × 9 prompts. Total parquet now 387 rows
(36 cells per (prompt × scale) × 7 zoom strengths + the wider negative/
+2.5 cells from the original render).

**Fraction with glasses present** (`siglip_glasses_margin > 0`):

| s | in-dist | held-out |
|---|---|---|
| 0.00 | 0% | 0% |
| 0.25 | 0% | 0% |
| 0.50 | 17% | 0% |
| **0.75** | **75%** | **38%** ← engagement |
| 1.00 | 92% | 67% |
| **1.25** | **100%** | **96%** ← saturation |
| 1.50 | 100% | 100% |

**Identity (mean ArcFace cos to baseline)**:

| s | mean | cells <0.4 |
|---|---|---|
| 0.25 | 0.87 | 0/36 |
| 0.50 | 0.79 | 1/36 |
| 0.75 | 0.66 | 2/36 |
| 1.00 | 0.56 | 6/36 |
| 1.25 | 0.48 | 9/36 |
| 1.50 | 0.42 | 16/36 |

**Two operating points emerge** depending on priority:

- **+1.0** — balanced default. 92% in-dist / 67% held-out coverage,
  identity 0.56 mean (6/36 cells failing the 0.4 floor).
- **+1.25** — held-out maximiser. 100% / 96% coverage, identity 0.48
  (9/36 cells failing).

The slider is **monotonic, not stepped**, between +0.50 and +1.50 —
the zoom resolves what the wide grid showed as a "doesn't fire below
+1.0" plateau into a smooth engagement ramp from 17% at +0.5 to 100%
at +1.25.

Score parquet now includes seed 7777 alongside 1337/2026/4242, giving
4 seeds for the zoomed strengths and 3 seeds for the wider strengths.
Disk discovery in `score_grid` picks up everything that exists; no
risk of missing zoom cells when re-running.

## Operating point

**+1.0 is the sweet spot**. Glasses on 89% of in-dist / 67% of
held-out cells, mean identity 0.54 (above the 0.4 floor). +1.5 reaches
full coverage but identity averages 0.42 with several cells failing.

**Usable range: +0.5 → +1.5**. Outside this window the slider either
doesn't fire (≤+0.5) or destroys identity (≥+2.0).

## Stereotype-bundle observation: smaller than feared

Side-channel deltas at s=+1.5 vs s=0 (in-dist mean):

| Channel | Δ |
|---|---|
| siglip_glasses | +0.054 |
| siglip_earrings | +0.022 |
| siglip_hair_curly | +0.021 |
| siglip_beard | +0.012 |
| siglip_hair_long | +0.005 |
| siglip_formal_clothing | -0.006 |

Glasses is the dominant SigLIP channel — bundle hypothesis from the
training-collage observation predicted formal_clothing/beard moving
*more* than glasses. They don't. The visual cluster shift (bare → polo
backgrounds, more makeup, hijab appearing on `ho_middleeast_neutral` at
+1.5) is real in the renders but doesn't show up as a measurable
competing axis on these particular probes. The bundle is a visual
gestalt (background, framing, posture) more than a per-channel
attribute drift.

Open: do we need a wider probe set (background-formality,
photographic-style, posture) before declaring the bundle gone? Or is
"the slider mostly moves glasses, with a stylistic side-show" the
honest summary?

## Verdict on step 600

Workable as a one-way glasses slider with operating point ~+1.0.
**Not perfect** — held-out coverage at +1.0 is 67%, identity at +1.5
is 0.42, the slider has zero useful range below +0.5. **Better than v0–v3**, which never reached glasses at all on females. Whether this is
the slider we ship depends on how step 1400 compares.

## Next

1. Eval step 1400 (already rendering) — same procedure, same parquet
   schema. Compare engagement strength, identity floor, generalisation
   gap. Hypothesis: 1400 either fully saturates earlier (engagement at
   +0.5 instead of +1.0) at the cost of identity, or plateaus.
2. If 1400 is no better, decide whether to ship 600 as `glasses_v4_step600`
   or wait for v5 (cosine LR, η=2.5, save_every=50).
3. The fraction-with-feature metric is per-axis. Smile/eye_squint
   should keep the bidirectional path; only one-sided axes need this
   binary proxy.

## Files

- `models/sliders/glasses_v4/glasses_slider_v4_000000600/eval.parquet`
- `models/sliders/glasses_v4/glasses_slider_v4_000000600/quick_collage_seed1337.png`
- `models/sliders/glasses_v4/glasses_slider_v4_000000600/summary_one_sided.txt`
- `models/sliders/glasses_v4/glasses_slider_v4_000000600/renders/<prompt>/seed*_str*.png`
