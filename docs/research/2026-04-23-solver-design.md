---
status: live
topic: metrics-and-direction-quality
---

# Composition solver design — vocabulary-space targets (2026-04-23)

## Purpose

Given a user-supplied intent — **typically natural language ("I want a smile that doesn't make the face look younger")**, sometimes an expert specification in atoms or AUs — translate into a target vector in shared vocabulary space, then output a set of prompt-pair weights `w` and scales `s` that, when fired through FluxSpace, produce an image whose measured vocabulary reads close to target.

The specification language is flexible; the solver is not. Natural language is the expected common case. AU/atom/blendshape target dicts exist for power users or when the user wants fine-grained control over a particular readout, but they're not the primary interface.

Formally: find `(w, s)` that minimize
```
‖A · diag(s) · w − t‖²
```
subject to constraint set `C` (identity_cos ≥ τ, total_drift ≤ δ, etc.), where

- `A` is the effect matrix (readouts × prompt-pairs), rows = vocabulary readouts, columns = prompt pairs from the accumulated dictionary.
- `w ∈ ℝⁿ` = non-negative weights over `n` prompt pairs (sparsity encouraged, L1-penalized).
- `s ∈ ℝⁿ` = per-pair scales.
- `t` = target vector in vocabulary space (only specified readouts; unspecified are "don't care").

In v0 we collapse `s` to a shared scalar so `A · s · w → A · w'` where `w' = s · w` and optimize jointly.

## Vocabulary (= readouts the solver can target)

From the sample_index + effect-matrix pipelines, every render is read into:

| Family | Members | Typical magnitude at scale 1 |
|---|---|---|
| Blendshapes | 52 `bs_*` (ARKit) | 0.0–1.0 |
| NMF atoms | 21 `atom_*` (11 live) | 0.0–2.0 |
| SigLIP probes | 12 `siglip_*_margin` | ±0.2 typical |
| Age | `mv_age, ins_age` (years) | ±30y over scale sweep |
| Gender | `mv_gender_conf`, gender-flip rate | 0–1 |
| Race | `ff_race_probs` (7-dim) | 0–1 simplex |
| Identity | `identity_cos_to_base` | 0–1 |
| Total drift | `siglip_img_cos_to_base` | 0–1 |
| Max-env | collapse predictor (when attn available) | 6–23 |

Any of these is a valid target readout. Targets are sparse — user specifies 1–10 readouts, rest are unconstrained.

## Target specification DSL

Three levels of expressiveness, same downstream:

### Level 1 — natural language (common case)

```yaml
intent: "smile with preserved identity, no age drift"
```

The solver resolves this by picking a seed pair from the dictionary tagged with the closest concept (e.g. a smile-family pair), plus constraint defaults: `identity_cos ≥ 0.75`, `mv_age_slope_abs ≤ 2.0`. Intent-to-target translation uses a lookup table of known concepts → (target readouts, default constraints), with unknown intents falling back to a nearest-concept match in the dictionary.

### Level 2 — constraint-refined natural language

```yaml
intent: "smile"
constraints:
  identity_cos_to_base: {">=": 0.85}   # stronger than default
  mv_age_slope:         {"abs<=": 1.0}
```

The most common case in practice — user intent is a concept, constraints are where they want extra precision.

### Level 3 — explicit vocabulary targets (expert, less common)

```yaml
target:
  bs_mouthSmileL: +0.40
  bs_cheekSquintL: +0.20       # AU6 component
  siglip_smiling_margin: +0.07
constraints:
  identity_cos_to_base: {">=": 0.75}
```

Used when the user needs fine-grained control over a particular readout, or when diagnosing a dictionary coverage gap ("why can't we produce AU6 without AU12?"). Not the daily-driver interface.

### Shared solver parameters

```yaml
solver:
  max_pairs: 3            # L1 sparsity cap — prefer few pairs
  scale_min: 0.0
  scale_max: 1.0
  lambda_l1: 0.1          # L1 penalty weight
```

## Dictionary schema

`output/demographic_pc/effect_matrix_dictionary.parquet` — one row per (axis, variant, base) cell from a `promptpair_iterate` iteration. Accumulates across iterations, no pruning. Columns:

- **Identity**: `iteration_id`, `axis`, `variant`, `base`, `prompt_pos`, `prompt_neg`
- **Slopes** (one per vocabulary readout): `slope_bs_mouthSmileL`, `slope_atom_16`, `slope_mv_age`, `slope_identity_cos_to_base`, `slope_siglip_smiling_margin`, ... — measured per unit scale, averaged across seeds inside the cell
- **Quality flags**: `n_seeds`, `n_scales`, `scale_max`, `target_r2` of the primary signal (for sorting)
- **Provenance**: `rendered_at` timestamp, `spec_path`

The solver reads this parquet at composition time, pivots into the (readouts × pairs) matrix `A`, runs the optimization.

## Solver implementation (v0)

1. **Parse target spec.** Resolve AU-codes and intent-text via fixed alias table; yield `t ∈ ℝᵈ` (sparse, only specified readouts) + constraint list `C`.
2. **Load dictionary.** `A` matrix of shape `(readouts, n_pairs)`. Keep rows only for readouts in `t` ∪ readouts appearing in `C`.
3. **L1-regularized NNLS.** `minimize ||A w − t||² + λ ||w||₁`, `w ≥ 0`, shared scale absorbed into `w`.
4. **Hard-constraint filter.** For each candidate `w`, compute predicted `A_C · w` over constraint readouts, reject if violated.
5. **Output**: ranked list of prompt-pair weights, predicted effect vector, expected residual.

v0 is single-axis, single-step (one composition, no iterative render-then-refine). v1 adds iterative residual cancellation.

## Data-flow

```
natural-language intent
        │
        ▼
target DSL  ──►  solver  ──►  prompt-pair weights + scales
                  ▲                        │
                  │                        ▼
          dictionary parquet       FluxSpace multi-pair render
                  ▲                        │
                  │                        ▼
           promptpair_iterate  ◄──  measure in vocabulary
                (ongoing)                  │
                                           ▼
                                   verify target hit
```

## What this closes and what it doesn't

**Closes:**

- Translation from natural-language / AU-code intent into concrete prompt-pair weights and scales.
- Reuse of every characterized pair — no experiment is wasted; every `promptpair_iterate` iteration adds to the dictionary.
- Principled constraint enforcement: identity preservation is a hard gate, not a scoring term.

**Doesn't close:**

- **Coverage gaps.** If the dictionary has no pair that separates target readouts (e.g. no pair fires AU6 without AU12), the solver returns residual error. That's diagnostic, not failure — the iteration loop fills the gap.
- **Flux-prior entanglement.** Some readouts are structurally coupled in Flux Krea. E.g. "beard" and "youth" co-fire regardless of prompt. The solver can surface these as an L1-infeasible target; fixing them requires either different prompts (iteration) or a different model.
- **Multi-pair FluxSpace execution.** Our current `FluxSpaceEditPair` node takes one pair. Firing multi-pair compositions requires either a node extension or iterative render-and-measure. v0 will use iterative (simpler, works today); v1 adds parallel injection if worthwhile.

## Milestones

1. **Dictionary schema + accumulator** — add a hook in `promptpair_iterate.py` to emit each iteration's per-cell slopes to the canonical dictionary parquet. (cheap; blocker for everything else)
2. **Solver skeleton** — `src/demographic_pc/compose_edit.py`, takes target YAML + dictionary parquet, outputs recommended weights + predicted effect. First version: L1-NNLS with scipy, no constraints. (cheap)
3. **Constraint enforcement** — add hard-constraint filtering.
4. **Iterative residual loop** — render solved composition, measure, compose a residual correction, re-render. Converges when residual error < ε.
5. **Multi-pair FluxSpace node** (optional) — extend ComfyUI node to accept a list of pairs instead of one, eliminating the iterative render for v1.

## Immediate next steps

- Build milestone 1 (dictionary accumulator) now — small diff in `promptpair_iterate.py`.
- Build milestone 2 (solver skeleton) now.
- After iter_02 produces 3 more dictionary entries (v1/v4/v5), we have 6 pairs in the dictionary: enough for a first solver smoke on a simple target like `bs_mouthSmileL: +0.3`.
- After 2–3 more axes have had iterations (beard, anger, surprise), the dictionary is rich enough for real composition.
