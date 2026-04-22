---
status: live
topic: manifold-geometry
summary: At fixed mix_b=0.5 scale is still non-linear — 59% of 78 scale-trajectories non-monotonic. Non-linearity is three stacked engineering curves (injection threshold, saturation sigmoid, per-base baseline), not manifold curvature.
---

# Phase-4 (redirected) — scale-sweep linearity at fixed mix_b

**Date:** 2026-04-22
**Follow-up to:** `2026-04-22-inphase-monotonicity.md`, `2026-04-22-blendshape-bridge-plan.md`
**Data:** `intensity_full/blendshapes.json` (340 scored of 504 rendered; 32% face-detection failures concentrated at scale ≥ 1.4).
**Script:** `src/demographic_pc/analyze_intensity_linearity.py`.

## TL;DR

At fixed `mix_b=0.5` (above the injection threshold found in Phase 1),
**`scale` is still a nonlinear control** — only 13% of 78 trajectories
are classified linear (Class A), 28% are saturating monotonic
(Class B), and **59% are non-monotonic** (Class C). The bridge plan's
implicit assumption — "above threshold, per-axis edits are linear in
scale" — is wrong.

## The three-parameter control surface

FluxSpace exposes at least three knobs with distinct mechanics:

1. **`mix_b`** — attention-cache blend ratio. Injection threshold at
   ≈0.45 (shown in Phase 1); below, nothing; above, fast onset.
2. **`scale`** — direction magnitude. Non-linear response inside the
   useful window; saturates fast; crashes at ~1.4.
3. **`start_percent`** — step when the edit turns on. Trades
   intensity for identity preservation.

Any "fine intensity" control needs to navigate all three, not just one.

## What the raw scale trajectories actually look like

Representative cells (scale ∈ {0.2, 0.4, 0.7, 1.0}):

**`01_faint` sp=0.25** (weak smile prompt):
| base | s=0.2 | s=0.4 | s=0.7 | s=1.0 |
|---|---|---|---|---|
| asian_m | 0.05 | 0.20 | 0.35 | 0.42 |
| european_m | 0.46 | 0.77 | 0.88 | 0.88 |
| young_european_f | 0.01 | 0.66 | 0.74 | 0.77 |

**`02_warm` sp=0.25** (standard smile prompt):
| base | s=0.2 | s=0.4 | s=0.7 | s=1.0 |
|---|---|---|---|---|
| asian_m | 0.18 | 0.82 | 0.88 | 0.87 |
| european_m | 0.69 | 0.96 | 0.96 | 0.96 |
| young_european_f | 0.42 | 0.93 | 0.95 | 0.95 |

**`04_manic` sp=0.25** (strong jaw prompt):
| base | s=0.2 | s=0.4 | s=0.7 | s=1.0 |
|---|---|---|---|---|
| asian_m | 0.01 | 0.65 | 0.66 | 0.63 |
| european_m | 0.01 | 0.07 | 0.16 | 0.18 |
| young_european_f | 0.00 | 0.06 | 0.15 | 0.16 |

Patterns:

- **Saturating sigmoid with narrow linear range.** `02_warm` jumps
  from near-zero to ~0.9 between s=0.2 and s=0.4, then plateaus.
  Only one scale step of linear regime.
- **Huge baseline variance per base.** `european_m` already scores
  0.47 mouthSmile at the weakest edit (`01_faint` s=0.2), while
  `asian_m` scores 0.07. Same FluxSpace direction, different faces,
  very different readout.
- **Strong per-base response variance on jaw.** Same `04_manic`
  direction at scale=1.0 gives jaw=0.63 on `asian_m` but jaw=0.18 on
  `european_m`. The direction is not base-invariant.
- **Collapse at scale ≥ 1.4.** 32% of renders failed face detection;
  all the misses are at scale 1.4 (sp=0.40 especially). Where face
  detection survives, blendshape scores often crash (e.g., a smile
  score drops to 0.05 at scale=1.4 after being 0.79 at scale=1.0 —
  the face is present but the expression became unrecognizable).

## Start_pct effect on response range

Mean (max − min) blendshape response across seeds and bases:

| ladder | sp=0.15 | sp=0.25 | sp=0.40 |
|--------|---------|---------|---------|
| 01_faint | 0.50 | 0.47 | 0.60 |
| 02_warm | 0.36 | 0.53 | 0.63 |
| 03_broad | 0.35 | 0.41 | 0.69 |
| 04_manic | 0.36 | 0.43 | **0.04** |
| 05_cackle | 0.42 | — | — |

Pattern: later start_pct **widens** response range for weak prompts
(01-03) because we get cleaner sweep coverage before collapse, but
**kills** strong prompts (04-05) entirely at sp=0.40 because the edit
fires too late in the DiT to propagate.

## Cross-axis leakage

Mean spillover onto secondary channels (max − min of non-primary
channel during primary-axis sweep):

| ladder | → jaw | → stretch | → funnel |
|--------|-------|-----------|----------|
| 01_faint | 0.007 | 0.031 | 0.03 |
| 02_warm | 0.041 | 0.137 | 0.07 |
| 03_broad | **0.089** | **0.194** | 0.09 |
| 04_manic | 0.000 | 0.104 | 0.15 |

`03_broad` leaks 9% onto jaw and 19% onto stretch — this prompt is
"grinning broadly with teeth showing," which is actually a *mix* of
AU12 + AU25 + AU26. Not a clean AU.

## What this means for the bridge plan

The blendshape bridge is still the right target, but **simple ridge
fits from attention cache to AU coefficients won't produce
axis-linear directions on their own.** The FluxSpace mechanism itself
has narrow linear dynamic range per direction.

Three things the bridge pipeline must add:

1. **Prompt selection for single-AU edits.** `03_broad` leaks 19%
   because the prompt wording bundles multiple AUs. AU-aligned
   directions need prompt pairs where both endpoints activate the
   same AU only. This is a prompt-engineering task guided by
   measured leakage, not just ICA-on-blendshapes.
2. **Per-base calibration.** The same direction produces
   very different responses per base. Ridge fits trained on one
   demographic mix will not transfer cleanly. Either fit per-base
   or normalise output by measured baseline.
3. **Work in the linear regime only.** scale < 0.5 for most axes;
   treat beyond-0.5 as saturation territory. Collapse at ≥1.4.

## What this falsifies

Another round of hypothesis revision:

- **Original:** α-interp phase cliff = cross-phase mixture of AUs.
  Phase 1 falsified this — cliff is a FluxSpace injection threshold.
- **Post-Phase-1:** above threshold, per-axis edits are linear in
  scale. **This writeup falsifies that** — scale is non-linear with
  narrow linear regime and heavy base-dependent variance.
- **Remaining claim:** some AU axes (01_faint-style weak prompts) at
  small scale produce approximately linear blendshape response within
  a narrow window, and that window varies per base. This is testable
  and much more modest.

## Implications for the Riemann thread

The dominant non-linearity is **not** manifold curvature in the
attention cache. It is:

- Injection threshold (sigmoid in mix_b at ~0.45)
- Saturation curve (sigmoid in scale)
- Base-specific baseline shift (additive bias per identity)
- Collapse cliff (scale ≥ 1.4)

None of these are Fisher-metric / geodesic problems. They are
engineering-parameter-curve problems. **Riemann thread is downgraded
further.** The local-metric experiment would only make sense after
we've characterised the three engineering curves and confirmed
residual non-linearity in the attention-cache structure itself.

## Revised Phase 4 (what to actually test)

**Step 4.1 — clean prompt pairs for AU isolation.** Using the
measured cross-axis leakage, pick (or re-engineer) prompt pairs per
target AU that keep leakage < 5%. Start from `01_faint`-style weak
prompts; avoid compound phrases.

**Step 4.2 — per-base baseline correction.** Before fitting ridge,
subtract the per-base baseline blendshape at scale=0 (the unedited
render). This makes targets comparable across demographics.

**Step 4.3 — low-scale ridge fit.** Ridge-fit attention-cache
features against baseline-corrected blendshape coefficients,
restricted to scale ∈ [0.2, 0.8] (pre-saturation linear regime).

**Step 4.4 — per-axis scale characterisation.** For each fitted
direction, render a scale-sweep at fixed mix_b=0.5, sp=0.25 over
s ∈ {0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0}. Fit a sigmoid. Record
per-axis (midpoint, slope, saturation-ceiling) triple — this is the
direction's *response profile*, consumer-facing control.

This drops the original Phase-5 Riemann leg entirely unless residual
non-linearity survives after baseline correction and sigmoid fitting.

## Injection-threshold diagnostic — do we still need it?

Short answer: **not as urgent.** We already know from Phase 1 that
mix_b=0.5 is safely above threshold and mix_b=0.4 is safely below.
The exact location (0.42? 0.44? 0.48?) and its start_pct dependence
would be a 30-render pilot if needed, but it doesn't change the
bridge plan — we fix mix_b=0.5 throughout and work with the other
two knobs.

**Defer unless needed.** If `Revised 4.1-4.4` produces clean
directions, we never need the threshold diagnostic.

## Pyright / tidy

- Two suppressions needed: `scipy.stats.kendalltau(...).statistic`
  — this is the documented attribute; Pyright stubs are stale. Add
  `# type: ignore[attr-defined]` on both call sites when next
  touching those files.

## Artefacts

- `output/demographic_pc/fluxspace_metrics/analysis/intensity_linearity.json`
- Plot TODO: scale-response curves faceted by (ladder, base, sp).
