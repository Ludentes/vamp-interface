---
status: live
topic: metrics-and-direction-quality
---

# Prompt-pair iterate — plan (2026-04-23)

## Why

Two findings forced this pivot:

1. **Atom-inject as an edit mechanism is broken.** See
   [2026-04-23-atom-inject-visual-failure.md](2026-04-23-atom-inject-visual-failure.md).
   Ridge-fit atom→δ directions do not produce semantic edits in
   image space. Atoms remain valid as a *measurement* basis; they
   are not valid as *steering vectors.*

2. **Prompt-pair FluxSpace edits carry confounds** (beard-axis is
   mostly an age-axis, smile-makes-younger, etc. — see effect
   matrix v0). These confounds are not fixable at the attention-
   injection level since we have no valid atom-steering primitive.
   They must be fixed at the **prompt** level: better prompt pairs
   produce cleaner atom-space movement with smaller confounds.

The framework needs an iteration loop that lets us propose 2–3
prompt-pair variants, render them on a small fixed eval grid,
measure atom-trajectory + confounds, score each variant by an
objective function, and recommend the winner with numerical
backing. Each iteration persists — after N rounds we have both a
curated prompt-pair dictionary and an audit trail of what was
tried.

## Core principle (user-stated, 2026-04-23)

> "If we want a smile, we should just tell the model to smile. Then
> use atoms only as a way to identify what to subtract/add/gain."

- The edit is specified **in natural language**, fired as a **direct
  prompt pair**, and lands at whatever atom/demographic/identity/probe
  vector the model produces.
- **Atoms are never the target specification.** User says "smile,"
  not "atom_16 = +1."
- **Atoms are the diagnostic layer.** They name what moved alongside
  the target, which is how the solver knows what auxiliary
  prompt-pair edits to compose to cancel the confound.
- Composition is in **prompt-pair weight space**, informed by the
  atoms-×-pairs effect matrix. It is not in atom-direct injection
  space — that path was falsified on 2026-04-23.

## Roles — separation of concerns

| Role | Primitive | Responsibility |
|---|---|---|
| Vocabulary / measurement | 11 NMF atoms in blendshape space | Names the axes; measures edit effects |
| Edit mechanism | FluxSpace pair-averaged attention | The one thing that fires visually |
| Dictionary | Curated prompt-pair library, each tagged with atom-purity profile | Maps atom intent → prompt pair to fire |
| Solver | Linear least squares on atoms×pairs effect matrix | Produces the prompt-pair weights to realize a target atom vector |

The atoms are *how we specify intent and measure effect*. The prompt pairs are *what we actually execute*. The dictionary is *the bridge*; the solver is *what composes multiple pairs to satisfy a target+confound spec*.

## The iterate loop

**Per iteration:**

1. **Inputs** (YAML or inline): axis, target SigLIP probe (optional), target atom (optional), 2–3 candidate prompt-pair variants.

2. **Render on a fixed eval grid**: 3 bases × 2 seeds × 2 scales {0.5, 1.0} = 12 renders per variant. Bases: `young_european_f`, `european_m`, `elderly_latin_m` (same as atom smoke — covers age × gender span).

3. **Score**: MediaPipe blendshapes → atom projection; SigLIP-2 image probes; MiVOLO + FairFace + InsightFace ages/gender/race; ArcFace identity cosine vs seed-matched scale=0 anchor.

4. **Fit slopes per base** (reuse `effect_matrix_v0._cell_slopes`). Aggregate across bases: mean target-probe slope, mean primary-atom slope, off-target atom mass, age drift, identity drift.

5. **Rank by objective**:
   ```
   score = target_slope / (1 + |mv_age_slope| · 0.05 + identity_drift · 1.5)
   ```
   Rewards target signal, penalizes age confound and identity drift. Denominator coefficients tuned so 5y age drift ≈ 0.25 penalty, 0.3 identity drift ≈ 0.45 penalty.

6. **Output markdown recommendation**: table of variants, winner, numerical reasoning, suggested next-round variants based on what won.

7. **Persistence**: each iteration saves a parquet of measurements and a markdown report to `output/demographic_pc/promptpair_iterate/<axis>/iter_NN/`. Builds a history.

## First iteration — smile

Three variants on the smile axis, evaluating against today's baseline.

- **v1 (baseline, overnight)**: pos=`A person smiling warmly.`, neg=`smiling warmly` spliced into base. This is what produced the effect-matrix's `mv_age_slope ≈ −5.3 y` signal.
- **v2 (age-preserving language)**: pos=`A middle-aged person smiling warmly.`, neg=same splice. Tests whether inserting age-stabilising language in the positive prompt reduces the age confound.
- **v3 (mechanistic phrasing)**: pos=`A person with upturned lip corners.`, neg=`A person with lips drawn horizontally.`. Tests whether geometric prompts carry less youth-cluster pull than affect prompts.

Expected outcomes:
- If v2 cuts age drift meaningfully → we have a recipe: add explicit age-language to every axis.
- If v3 wins on atom purity with lower confound → the semantic-affect phrasing is the carrier of the confound, and geometric phrasing should become the default.
- If all three cluster → confound is structural to Flux's prior, not to prompt wording. Demographic-δ batch becomes load-bearing (see `project_pending_demographic_deltas.md`).

## Implementation

Single new module: `src/demographic_pc/promptpair_iterate.py`. Reuses:

- `pair_measure_workflow` from `fluxspace_metrics.py` (FluxSpace rendering)
- `comfy_flux.ComfyClient` (dispatch)
- `score_blendshapes` (MediaPipe)
- `classifiers.{MiVOLOClassifier, FairFaceClassifier, InsightFaceClassifier}`
- `score_clip_probes.Siglip2Backend` (SigLIP text + image)
- `effect_matrix_v0._cell_slopes` (slope fitting)

Writes two artefacts per iteration:

- `output/demographic_pc/promptpair_iterate/<axis>/iter_NN/results.parquet`
- `output/demographic_pc/promptpair_iterate/<axis>/iter_NN/report.md`

## What this explicitly doesn't try to do

- **No solver yet.** Composition solver needs the dictionary to exist first. Iteration is for building the dictionary entries.
- **No rebuild of ridge directions.** That path is closed — see the visual-failure doc.
- **No new render scripts per axis.** The iterate module is generic; axis is an input, not code.

## Exit criteria for the v0 tool

- Fires on smile v1/v2/v3 end-to-end: rendering, scoring, slope fit, markdown report, parquet.
- Resumable (skip rendered PNGs).
- Scale numbers (target, age, identity drift) match the effect-matrix v0 numbers for the baseline variant ±5% — confirms we're measuring the same thing on the same basis.

After smile v0 passes, iterate on beard (where we know the confound is worst) to see whether any prompt-pair phrasing can get target_slope > 0.03 with |age_slope| < 3y. If not, that's the demographic-δ go-signal.
