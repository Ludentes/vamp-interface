# FFA Progression Curriculum — From Trivial Discrimination to Gestalt Pattern-Spotting

**Date:** 2026-04-20 (revised same day to pass `designing-perception-experiments` skill checklist)
**Status:** Design doc, pre-experiment
**Relation to other docs:** Complements `docs/blog/2026-04-16-part-2-math-of-pattern-preservation.md`. Where Part 2 asks "can the generator preserve pattern?", this asks "can a trained human reader recover pattern faster from faces than from a matched non-face channel — and at what task complexity does that face-advantage appear?"

---

## The Bet

Human visual cortex has specialized face-processing machinery — the fusiform face area (FFA) and associated circuits — that do three things generic vision does badly:

- **Holistic processing:** the whole face is perceived as a configuration, not a bag of features. Tiny relational changes register as "wrong" without being locatable.
- **Parallel glance-reading:** a grid of faces is read in roughly constant time, not O(n). Odd-one-out pops.
- **Fine identity discrimination:** ~5000 faces distinguishable across a lifetime, most of which share the same topology to five decimal places.

General visual channels (bars, scatterplots, colormaps) beat faces at **precise value readout**, especially at low dimensionality and unlimited exposure. This is Cleveland-McGill territory and it is not in dispute.

**The useful regime for faces is the opposite corner:** many items visible at once, brief exposure, holistic judgment (similarity, anomaly, cluster membership) rather than readout. The question this curriculum answers is *where exactly the crossover lies* — the D, n, task-type, exposure-time regime past which faces beat a matched non-face encoding. Everything downstream (scam-hunter UX, data-mining "AHA" demo) depends on operating confidently past that crossover.

## Why A Curriculum, Not A Single Experiment

Two reasons.

**One — the observer has to be trained.** FFA machinery is not automatically recruited by unfamiliar synthetic faces. The calibration "these dials mean these things" has to be learned before pattern-spotting is possible. A naive viewer shown 256 faces in a grid has no prior for what counts as similar. A trained viewer does.

**Two — we need to falsify at each step.** If D=1 readout fails on faces (it should, actually — bars win there), that's informative, not failure. The curriculum is a ladder of increasingly demanding claims; each level has a pass/fail criterion. We cannot skip to Level 5 and interpret the outcome, because we will not know *which* assumption broke.

The terminal level is an **AHA demonstration** — a trained viewer spots one or two anomalies in a grid of hundreds of faces in a single glance, on designed geometry where the ground truth is known exactly. If that works on synthetic D≤6 input, the Flux-pipeline claim (faces as pattern-preservers) is established under ideal conditions.

## How This Follows the Skill

Every level in this curriculum is itself a small perception experiment. Each level design is held to the `designing-perception-experiments` skill's 11-item checklist: paradigm, difficulty knob with units, exposure time, matched non-face comparator, training/test separation, controls (including inversion for face-specific claims), trial count + staircase/grid, d′ + psychometric-function analysis with bootstrap CIs per subject, sample-size rationale (Smith & Little 2018), preregistration, tooling.

Rather than restating all eleven items on every level, this doc lifts the shared pieces into **Global Invariants** below and then specifies per level only the parameters that change (paradigm, geometry, difficulty knob units, exposure time, pass criterion, which controls escalate at that level).

## Global Invariants

These hold across all levels. A level that deviates must say so explicitly.

**Source geometry is hand-designed.** Points in ℝ^D with D∈[1,6]. Clusters, rings, manifolds, grids — whatever the level requires. Ground-truth distances are known by construction. No qwen, no learned teachers.

**Face generator stays Flux + Flux conditioning space.** Part 2's pipeline unchanged. Only the input geometry swaps.

**Matched non-face comparator on every level.** Same D-dimensional vector, rendered as a non-face glyph (see *Comparator choice by D*). Comparator matched on luminance / contrast / spatial-frequency envelope within a tolerance; pre-session audit by a simple classifier confirms the two encodings carry the same discriminability for the task (if the classifier can distinguish outliers on one encoding but not the other, comparator is unfair — redesign before collecting human data). Required per skill §4, face-specific §"Matched non-face control".

**Demographic-axis confound handling (face-specific §"Demographic-axis confound").** For every level, the difficulty direction in ℝ^D is sampled from the set of directions **orthogonal to Flux's top demographic PCs** (age, apparent gender, apparent ethnicity — identified via a pretrained classifier on a sample of 500 generated faces before the curriculum starts). Trials whose projected direction correlates |r|>0.2 with any demographic PC are rejected and resampled. Failure to do this makes the task trivially easy and non-generalizable.

**Inversion control at Levels 2 and above.** Every level from Level 2 upward (i.e. every level where we bet on face-advantage) includes a matched inverted-face block, same stimuli rotated 180°. An effect that persists equally on inverted faces cannot be attributed to FFA / holistic processing — it's generic image discrimination. Required per skill face-specific §"Inversion control — non-negotiable for FFA claims"; Yin 1969; Rossion 2013.

**Training / calibration / test separation.** Every level runs (a) calibration: ~30 labeled examples, unlimited time, ground-truth displayed; (b) practice with feedback; (c) test without feedback. Only test-block trials are reported. Calibration/practice Δ schedule is disjoint from test Δ schedule — no stimulus leakage across phases.

**Trial uniqueness.** Every face shown to a given participant is drawn from a fresh input vector; no face reused within a session. Pool size per level ≥ 5× test trial count.

**Adaptive staircase + fixed anchor grid.** 3-down-1-up staircase per encoding per level (converges to ~79% correct) for threshold estimation, **plus** a fixed grid of ~5 Δ levels (each ~40 trials) for full psychometric-function fitting via `psignifit 4` or Palamedes. Staircase sets the prior; grid produces the PF. Required because single-Δ designs give one accuracy point with no curve, per skill common-mistakes §"Running only at one difficulty".

**Analysis.** Per-subject item-level d′ and psychometric-function fit with **bootstrap 95% CIs (Wichmann & Hill 2001)**. Group-level via mixed-effects logistic on item-level decisions, fixed effects = encoding × Δ, random intercepts + slopes per subject. **Never pool raw trials across subjects into a single PF.**

**Sample size.** N=3–5 participants, per-subject ~600 trials across conditions. Justified by Smith & Little (2018), *Small is beautiful: In defense of the small-N design*, Psych. Bull. Rev. 25(6): dense within-subjects psychophysics with many trials per subject supports strong inference at small N when (a) each subject contributes many trials, (b) within-subject effect is large relative to between-subject variance, (c) per-subject analyses are reported, (d) effect replicates across the small N. Ticked per level; any level that can't tick them escalates to larger N.

**Preregistration.** Every level registered on AsPredicted with: hypothesis (directional, e.g. "face threshold Δ₇₉ < comparator threshold Δ₇₉"), paradigm, Δ levels, encodings, exclusion rules, stopping rule (fixed N, no optional stopping). Report includes the "21-word solution" (Simmons, Nelson, Simonsohn 2012). Deviations disclosed.

**Tooling.**
- Presentation (timing-critical, <200ms exposure + mask): PsychoPy standalone runner.
- Presentation (unlimited-time levels, browser acceptable): jsPsych + jspsych-psychophysics plugin.
- Adaptive staircase: `aepsych` (Meta) for multi-dim; `PsychoPy.data.StairHandler` for simple.
- PF fitting + bootstrap CIs: `psignifit 4` (Python).
- Stimulus generation: Flux v3 (production pipeline for faces); matplotlib / custom SVG for non-face comparators.

**Honest reporting.** Every per-level writeup ends with: "We report how we determined our sample size, all data exclusions, all manipulations, and all measures." Per-subject plots alongside pooled contrasts.

## Comparator Choice by D

The matched non-face comparator must carry the same D-dimensional information.

| D | Comparator | Notes |
|---|---|---|
| 1 | Single vertical bar, length = value | Cleveland-McGill canonical; faces expected to lose |
| 2 | 2D scatter point on axes | Clean; faces at parity or behind |
| 3 | 3-bar glyph OR 3D-to-2D scatter (PCA) | 3-bar preserves information but isn't visually parallel; scatter is parallel but loses 1D. Report both as a within-level sub-comparison. |
| 4-6 | D-bar glyph OR 2D scatter with color/size encoding remaining dims | Same tradeoff amplifies. Levels 4-5 report two comparators and the face-advantage margin against each. |

At D>2 no single baseline is canonical. This is a design constraint, not a design flaw — it means Levels 4-5 naturally test "faces vs the best available chart encoding we know."

## The Levels

Each level's spec lists only per-level parameters. Global invariants above still apply.

### Level 0 — Single-Dial Calibration

| Decision | Value |
|---|---|
| Question | Is the D=1 axis perceivable in faces at all? |
| Geometry | D=1, 16 points linearly spaced in [0, 1] |
| Paradigm | 2-AFC, unlimited exposure |
| Difficulty knob | Δ in source units; log-spaced levels 0.05, 0.1, 0.2, 0.4, 0.8 |
| Exposure | Unlimited time, no mask (explicitly non-FFA level) |
| Comparator | Single bar, length proportional to value |
| Inversion control | **Skipped** (this level is not an FFA claim — it's a sanity floor) |
| Trial count (per encoding) | 40 practice + 100 test per subject |
| Pass criterion (face) | Trained subject d′ ≥ 1.5 (≈83% 2-AFC correct) at largest Δ; psychometric function monotone |
| Pass criterion (face vs bar) | **Faces lose to bars. That is the expected and acceptable outcome.** The purpose of Level 0 is to verify the face-encoding axis exists and is monotone, not to beat bars. |
| Failure means | Face generator isn't producing monotone variation along the axis. Abort curriculum and fix generator before climbing. |

### Level 1 — Triangle / Continuity (P1 perceptual test)

| Decision | Value |
|---|---|
| Question | Does source-space distance predict perceived face distance? |
| Geometry | D=2, triplets (A, B, C) sampled so \|d(A,B) − d(A,C)\| exceeds margin m, varying m |
| Paradigm | 3-AFC same/different triangle: "which pair is more similar?" Unlimited exposure |
| Difficulty knob | m in σ units of sampling prior; levels 0.3, 0.6, 1.0, 1.5, 2.5 |
| Exposure | Unlimited, no mask |
| Comparator | 2D scatter, same points labeled A/B/C on axes |
| Inversion control | **Skipped** (continuity is not uniquely an FFA claim — bars are also expected to track distance) |
| Trial count | 40 practice + 120 test per encoding |
| Pass criterion | Face threshold Δ₇₉ within 10pp of scatter threshold Δ₇₉; correlation ρ(source distance, face-space distance) ≥ 0.7 via Spearman |
| Secondary | Rank correlation ρ on same triplets as P1 framework metric — compare to offline ArcFace-embedding ρ to see whether perception agrees with the embedder |
| Failure means | Generator preserves geometry in ArcFace-space but not for human perception. Framework §2.7 disconnect — fix before Level 2+. |

### Level 2 — Glance Odd-One-Out (first FFA-advantage bet)

| Decision | Value |
|---|---|
| Question | Does face-specific / holistic processing give an advantage over matched chart under glance exposure? |
| Geometry | D=3, 4 items: 3 from cluster A, 1 from cluster B = A + Δ·u |
| Paradigm | 4-AFC oddity ("which is different?"). Secondary 2-AFC variant (pair per trial) for cleaner d′. |
| Difficulty knob | Δ in within-cluster σ units; levels 0.5, 1.0, 1.5, 2.5, 4.0 |
| Exposure | 200ms stimulus + 400ms dynamic mask (scrambled tiles from same encoding). Required per skill: glance is where FFA earns its keep. |
| Comparator | 3-bar glyph per item, matched luminance/contrast |
| Inversion control | **Required.** Third encoding: same faces, rotated 180°. Matched 200ms exposure + mask. |
| Trial count | 40 practice + 80 test per encoding × 3 encodings = 240 test trials |
| Pass criterion (primary) | Face Δ₇₉ < Comparator Δ₇₉ with 95% bootstrap CI of contrast excluding 0 on face-better side, in ≥4/5 subjects |
| Pass criterion (face-specific licensing) | Face Δ₇₉ < Inverted-face Δ₇₉ with CI excluding 0. This is what licenses "FFA / holistic" language. Without it, the skill forbids face-specific claims — report only as "face-encoding-specific" or downgrade. |
| Secondary | RT(face) vs RT(comparator) at matched accuracy |
| Failure means | If face beats comparator but inverted equals upright → generic image-statistics advantage, not holistic. Drop FFA framing. If face doesn't beat comparator → the attribute dials are local (eyes only / mouth only) rather than configural; fix projection / conditioning layout before climbing. |

### Level 3 — Small-Grid Anomaly

| Decision | Value |
|---|---|
| Question | Does the face advantage scale to a multi-item grid under time pressure? |
| Geometry | D=3, 16 points: 12 from cluster A, 4 outliers at varying Δ per trial |
| Paradigm | Bounded visual search: 5s exposure, click all suspected outliers. Enter submits. |
| Difficulty knob | Δ in σ units per outlier; fixed anchor levels 1.0, 1.5, 2.5 |
| Exposure | 5s — long enough for a few saccades, short enough to prevent full serial inspection |
| Comparator | 16-item grid of 3-bar glyphs at matched size |
| Inversion control | **Required.** 16-item grid of inverted faces, 5s exposure. |
| Trial count | 60 test per encoding × 3 encodings = 180 trials per subject |
| Pass criterion (primary) | Face F1 > Comparator F1 at matched Δ, per-subject bootstrap CI of contrast excluding 0, ≥4/5 subjects |
| Pass criterion (FFA licensing) | Face F1 > Inverted-face F1 with CI excluding 0 |
| Secondary | RT at first-click; false-alarm locations (if they cluster near in-set points nearest to B, confirms geometry-driven rather than random) |
| Failure means | Face channel doesn't scale with n under time pressure. Likely uncanny signal is too subtle under clutter; amplify mapping slope or accept Level 2 as ceiling. |

### Level 4 — Structured Manifold Recovery

| Decision | Value |
|---|---|
| Question | Does trained face-reading recover *structure* (rings, manifolds, clusters) not just pointwise differences? |
| Geometry | D=4-6, 64 points as one of: two tangent Gaussians; ring embedded in cloud; Swiss roll (2D manifold in 4D). Grid position randomized — structure must be recovered from face appearance, not spatial layout. |
| Paradigm | Two-stage per trial: (a) binary detection "is there structure?" (b) group assignment — click each face and assign it to group 1 or 2 (for clusters) or rank along the manifold (for ring/Swiss roll). |
| Difficulty knob | Manifold separability metric: Fisher ratio `tr(S_B)/tr(S_W)` of the designed geometry (same as framework P4). Levels 1.0, 2.0, 4.0, 8.0. |
| Exposure | 10s — structure recovery needs more time than pop-out |
| Comparator | (i) 2D scatter after random linear projection to 2D (structure partially hidden); (ii) D-bar glyphs per point. Report against both. |
| Inversion control | **Required.** 10s, inverted 64-face grid. |
| Trial count | 40 test per encoding × 4 encodings = 160 trials per subject |
| Pass criterion (primary) | Adjusted Rand Index (ARI) of participant clusterings > comparator ARI, CI excluding 0, in ≥4/5 subjects |
| Pass criterion (FFA licensing) | Face ARI > Inverted ARI, CI excluding 0 |
| Failure means | Pointwise discrimination (Levels 2-3) doesn't compose into structural perception. Would force a rethink of whether face channel supports the full Part 2 claim or only the cluster-membership sub-claim. |

### Level 5 — AHA at Scale

| Decision | Value |
|---|---|
| Question | Can a trained viewer glance-spot embedded anomalies in 256 faces? Existence proof for "faces as data-mining tool." |
| Geometry | D=4-6, 256 points from one dominant distribution + 1-2 embedded anomalies at small Δ (close-but-distinct, same neighborhood) |
| Paradigm | 2s exposure, no mask (visual search at rapid-glance). Click any anomalies; Enter submits. |
| Difficulty knob | Anomaly Δ in σ units of dominant distribution; fixed anchor levels 0.5, 1.0, 2.0, 4.0 |
| Exposure | 2s — short enough that comparators should be at chance |
| Comparator | (i) 2D scatter of 256 points at matched pixel budget; (ii) D-bar glyphs per point (probably illegible at 256 — that's the point) |
| Inversion control | **Required** but pragmatically secondary: Level 5 is an existence proof; FFA licensing is nice-to-have. Main report: upright-face vs comparator. Secondary: upright vs inverted. |
| Trial count | 40 test per encoding per subject |
| Pass criterion (primary — demo deliverable) | Face precision > 0.5 at 2s exposure for Δ ≤ 1σ; comparators at chance (precision ≈ 0.01 given 2/256 anomalies). This asymmetry is the demo. |
| Pass criterion (scientific) | Face Δ₇₉ lower than both comparators, CI excludes 0, ≥4/5 subjects. Level 5 comparators are expected to be near-floor; effect size is large and positive. |
| Failure means | Even on hand-designed geometry, the face channel doesn't beat charts at 2s × 256 items. The AHA demonstration doesn't work. Scam-hunting application of this pipeline is not supported; look at Levels 2-3 as the realistic operating regime. |

## Training Protocol — Concrete Numbers

For every level:

1. **Calibration:** ~30 labeled examples per encoding. Source coordinates displayed next to face / glyph. Unlimited time, free exploration. Participant signals when ready.
2. **Practice with feedback:** ~50 trials, per-trial correctness displayed. Advance to test only when 10-trial rolling accuracy crosses the level's pass bar on practice-Δ schedule.
3. **Test, no feedback:** level-specific trial count per encoding. Interleaved staircase + fixed-grid-anchor trials.
4. **Matched-baseline block:** same test against the non-face comparator(s). Counterbalanced order across participants (Latin square).
5. **Inversion block (Levels 2+):** same test against inverted-face encoding. Counterbalanced with other blocks.

Inter-block breaks: ≥2 min between blocks, forced 10 min between sessions, no single session > 45 min.

Calibration and practice Δ schedules are drawn from a held-out range, disjoint from the test grid, so participants don't see test stimuli during training.

Participants: 3–5 per level, within-subject across encodings. Different participants across levels are acceptable (each level is its own preregistered mini-study), but having 1-2 participants traverse the full curriculum provides the tightest within-subject curriculum-level evidence.

## Analysis — Concrete Pipeline

Per subject per encoding per level:

1. Fit psychometric function with `psignifit 4`: Weibull, γ fixed (chance level by paradigm), λ in [0, 0.05] with prior, α and β fit. Extract Δ₇₉ and slope.
2. 95% bootstrap CIs on Δ₇₉ (Wichmann & Hill 2001 part II).
3. Compute d′ at each Δ level; report accuracy *and* d′ side-by-side (accuracy alone mixes sensitivity with bias per Macmillan & Creelman 2005).

Across subjects per encoding per level:

4. Mixed-effects logistic regression on item-level decisions: `response ~ encoding * Δ + (1 + Δ | subject)` via `bambi` or `statsmodels`. Fixed-effect contrast encoding₁ vs encoding₂ gives the group-level face-advantage at each Δ.
5. Alternatively (small N, cleaner visualization): report per-subject Δ₇₉ with CI bar, plus group median and CI across subjects.

Across levels (curriculum-level synthesis):

6. **Face-advantage curve:** x = level index (roughly task complexity × n × 1/exposure), y = `Δ₇₉(comparator) − Δ₇₉(face)` in σ units. Zero-crossing is the face-vs-chart crossover; curve rising to the right confirms the FFA-regime hypothesis.
7. **FFA licensing curve:** x = level, y = `Δ₇₉(inverted-face) − Δ₇₉(upright-face)`. Positive margin at Levels 2+ licenses holistic-processing interpretation; flat margin would force a retreat to "face-encoding-specific" language without FFA claims.

All analyses preregistered before data collection.

## Preregistration Template (Per Level)

```
Level: __
Hypothesis: Δ₇₉(face) < Δ₇₉(comparator) in ≥4/5 participants, bootstrap CI of contrast excludes 0 on face-better side.
Secondary hypothesis (Levels 2+): Δ₇₉(face) < Δ₇₉(inverted-face), CI excludes 0.
Paradigm: __ (per level)
Encodings: face, comparator, (inverted-face for Levels 2+)
Difficulty levels (Δ): __ in σ units, sampled orthogonal to demographic PCs
Exposure: __
Trial counts: __ practice with feedback, __ test without, per encoding
Staircase: 3-down-1-up (79% target)
Fixed-grid anchors: Δ ∈ {__}, __ trials each
N: __ participants. Stopping rule: fixed N, no optional stopping.
Exclusion rules: subjects with catch-trial false-alarm rate > 30% excluded (preregistered, applied before analysis).
Analysis: psignifit 4 psychometric fit, bootstrap 95% CI on Δ₇₉. Mixed-effects logistic for pooled contrast.
Sample-size justification: within-subjects dense-trial design per Smith & Little (2018).
Statement: "We report how we determined our sample size, all data exclusions, all manipulations, and all measures."
```

Registered on AsPredicted (aspredicted.org) before each level's data collection.

## Metrics and Analysis — Cross-Level Synthesis

Beyond per-level psychometric analysis:

- **Face-advantage curve** (see Analysis §7).
- **FFA licensing curve** (see Analysis §8).
- **Framework P-property snapshots** per level on the generator output, independent of human trials:
  - P1 (Spearman source-to-face-space rank correlation)
  - P2 (ArcFace cosine identity preservation)
  - P4 (Fisher ratio over designed clusters)
  - P6 (attribute preservation — mostly free on designed geometry since we designed attributes as axes)
- **Alignment plot:** P1/P2/P4 on x-axis, human Δ₇₉ on y-axis. Correlation tells us whether machine geometry metrics predict human perception — that's Part 2's Tier 1 vs Tier 2+ decision data.

## What This Curriculum Does Not Claim

- **Not ecological validity.** These are designed geometries, not real job posts. Scam-hunting application is *downstream* of this curriculum passing, not proved by it.
- **Not optimality of Flux.** Other face generators (FLAME, StyleGAN3) would likely also pass the earlier levels; Level 5 is where generator quality starts mattering.
- **Not transfer.** Passing Level 5 on designed geometry is necessary but not sufficient for the qwen-sourced pipeline to work on real data; transfer is a separate question.
- **Not a population claim.** Small-N within-subjects design supports existence / mechanism claims. "Untrained naive users benefit" would require larger N and is explicitly not in scope here.

## Open Design Questions

- **Who is "the trainee"?** Self-trial by the project author is appropriate for Levels 0–2 dev loop (discovery + debugging, not reported). Levels 2–5 reported studies need 3–5 outside participants per Smith & Little; preregister before outside participants run.
- **How many attribute dials does Flux expose cleanly?** Part 2 Stage 0 is the source of this answer. If Flux gives us <4 disentangled dials, Levels 4–5 collapse to effectively Level 3.
- **Best comparator at D>2?** Current plan: report against two comparators (D-bar glyph + 2D scatter with color/size) at Levels 4–5; accept that no single baseline is canonical.
- **Static vs scrolling / animated grids for Level 5?** Real data-mining tools scroll. Static grid is Phase 1; scrolling is a natural follow-up experiment if Phase 1 passes.

## Relation to Part 2's Tier 1 vs Tier 2+ Decision

Part 2 Stage 0 is the numeric Fisher measurement on the existing Flux v3 pipeline. This curriculum is the *perceptual* companion: Stage 0 tells us whether the face space has the right geometry for ArcFace; Levels 1–2 tell us whether humans can read that geometry. Both are inputs to the escalation decision.

Concretely: if Stage 0 Fisher is healthy but Level 2 face-advantage is zero or negative, Tier 2+ training (DRaFT / Flow-GRPO) is unlikely to help — the bottleneck is the human-perception-to-Flux-face mapping, not the Flux-to-ArcFace mapping. If both are healthy, Tier 1 is probably enough. If Fisher is weak but Level 2 face-advantage is present, something is strange and worth a second look.

---

**Next step:** resolve the open design questions (especially demographic-PC identification — run Flux sample → classifier → top-PC extraction, no humans needed), then implement Level 0 as a runnable experiment per the preregistration template. Level 0 is intentionally the least interesting level; its purpose is to de-risk the full pipeline (stimulus generation, timing, logging, psignifit fitting, bootstrap CIs) end-to-end before climbing to levels where the science matters.
