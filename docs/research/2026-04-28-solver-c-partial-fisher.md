---
status: live
topic: demographic-pc-pipeline
---

# Solver C as a partial-Fisher estimator; squint feasibility check

Formalizes the Solver-C pair-selection rule from
`docs/research/2026-04-26-solvers-taxonomy-and-next-steps.md` (lines
59–82, 165–196) in information-theoretic terms, and specs the empirical
feasibility check we owe ourselves before authoring an eye-squint
slider.

Context: glasses got rescued by Path A (prompt-pair) once we found
r=32 + peak=20 + long training (`v8`, eval 2026-04-28). The bundle
problem the solvers doc warns about did not bite us hard because the
glasses↔studio bundle was loose enough for a high-capacity LoRA to
absorb the residual signal. Squint is the next axis the pipeline
needs (eyeSquintLeft/Right are core ARKit blendshapes for the
Vox2Face / FLAME bridge thread). Squint bundles in the text encoder
with smile (Duchenne), age (crow's feet), bright lighting, and
scrutiny — plausibly a tighter bundle than glasses. Path A is
structurally weak on bundled axes; the doc predicts squint is where
this becomes load-bearing.

## Information-theoretic restatement

Let θ = scalar intended-axis score (here `bs_eyeSquintLeft +
bs_eyeSquintRight`), η = vector of confound scores (smile, age,
lighting bin, head-pose proxy, hair, clothing, identity drift). The
slider's job is to learn a velocity perturbation that increases θ
without moving η.

The Fisher information matrix of the latent observation z about
(θ, η) decomposes as

```
I(θ, η) = [ I_θθ   I_θη ]
          [ I_ηθ   I_ηη ]
```

What we actually want to maximize is the **partial Fisher
information**

```
I_{θ|η} = I_θθ − I_θη · I_ηη^{−1} · I_ηθ
```

— the information about θ remaining after projecting out whatever η
explains. This is the Schur complement of I(θ, η) on the η block; it
is also the Cramér–Rao lower bound on estimating θ when η is
unknown and must be marginalized over.

Path A maximizes I_θθ (it learns whichever direction has loss-mass,
including the bundle). Solver C and Path C both target I_{θ|η}; they
just attack it at different layers.

## Solver C as a finite-difference partial-Fisher estimator

For a corpus pair (i, j) the latent difference Δz_ij is a
finite-difference column of the latent's Jacobian with respect to
(θ, η). The pair's marginal information content for θ given η is
approximately

```
J(i, j) ≈ (Δθ_ij)² / ( σ_θ² + Δη_ij^T · W · Δη_ij )
```

where σ_θ² is a small floor regularizer and W is a confound-importance
metric. The optimal W under a Gaussian assumption on η is

```
W = Σ_η^{−1}
```

— Mahalanobis on the empirical confound covariance. Concretely: rescale
each confound by its corpus standard deviation, then sum. This
replaces the doc's hand-set "identity weighted heaviest, lighting
lightest" with a calibrated metric.

Maximizing Σ_{(i,j) ∈ S} J(i, j) over a selected pair set S is a
greedy estimator of the empirical I_{θ|η} of the training distribution.
"Δintended large, Δconfound small" in the doc is exactly this ratio.

## What the formalization buys us

1. **Single tunable knob.** σ_θ² is a noise floor; W is set by data.
   The pair-selection objective is parameter-free up to a temperature
   on (Δθ)² — and that temperature only changes which pairs you pick,
   not the ranking.
2. **Stopping criterion.** Keep adding pairs greedily by J. Stop when
   the running estimate of I_{θ|η}(S) saturates — the marginal
   information gain per pair drops below a threshold. If the saturation
   value is small, no pair budget rescues the run; the corpus genuinely
   lacks θ-only variance.
3. **Feasibility check for free.** The histogram of Δθ vs ‖Δη‖_W
   across all candidate pairs *is* the empirical partial-Fisher
   distribution. We can compute it before any training and decide
   whether Solver C has signal on this axis.
4. **Why glasses got lucky.** Glasses' I_{θη} with the studio cluster
   was below the threshold a rank-32 LoRA needs to absorb. Squint's
   bundling with smile/age/lighting is plausibly an order of magnitude
   tighter — capacity will not save us.
5. **Path B vs Path C as estimators.** Path C's φ-gradient
   `−t · ∇_z φ(x̂_0)` is a parametric estimator of the score
   function for θ; its asymptotic variance is I_θθ(φ). Solver C
   feeding Path B is a non-parametric estimator that conditions on η
   at the data layer. They estimate the same quantity (I_{θ|η}); the
   choice between them is a bias-variance and engineering-cost
   tradeoff, not a question of which one is "right."

## Feasibility check on the squint axis

Goal: decide before any training whether Solver C has pairs to find on
eye-squint.

Inputs (all in `models/blendshape_nmf/sample_index.parquet`, 7772 rows
as of 2026-04-28):

- **θ** = `bs_eyeSquintLeft + bs_eyeSquintRight`
- **η** components, z-scored on the corpus:
  - smile: `bs_mouthSmileLeft + bs_mouthSmileRight`
  - age: `age` column (numeric per render)
  - lighting / scene: `base` (one-hot, then weight by 1/n_classes)
  - head-pose proxy: `bs_eyeLookInLeft+Right`,
    `bs_eyeLookOutLeft+Right`, `bs_eyeLookUpLeft+Right`,
    `bs_eyeLookDownLeft+Right` (sum of absolute values — captures
    gaze deviation from frontal)
  - identity drift: `1 − identity_cos_to_base`
- **W** = diag(1/σ²_k) on the z-scored components above (Mahalanobis
  with diagonal Σ_η; we ignore off-diagonal for v1).

Procedure:

1. Form all candidate pairs (i, j) within the same `(ethnicity,
   gender, age_bin)` cell. (Cross-demographic pairs are off-axis by
   construction; we only care about within-cell variance for the
   slider supervision.)
2. For each pair compute Δθ_ij and Δη_ij; compute J(i, j) with
   σ_θ² set to 5% of var(θ) on the corpus.
3. Histogram J across all candidate pairs.
4. Sort by J descending; greedy-select up to 1000 pairs subject to
   each render appearing ≤ 4 times (cap to avoid one outlier render
   dominating).
5. Report:
   - count of pairs with J > J_threshold for J_threshold ∈ {p50, p90,
     p99 of within-cell pair distribution}
   - cumulative I_{θ|η}(S) curve as |S| grows from 0 to 1000
   - top-50 and bottom-50 pair examples (img_path pairs) for visual
     sanity check
   - per-confound Δη histogram inside selected vs unselected pairs
     (selection should compress confound deltas)

Pass criteria for proceeding to Solver-C-driven Path-B training:

- ≥ 200 pairs with J above the within-cell p90.
- Cumulative I_{θ|η}(S) reaches a clear plateau before |S| = 1000
  (i.e., the corpus has more pairs than we need; we are not scraping
  the bottom).
- Visual sanity: top-J pairs show "same identity ± squint" with
  matched smile/age/lighting; bottom-J pairs show the opposite.

Fail modes:

- < 50 pairs above p90 → corpus genuinely lacks intra-cell squint
  variance. Need a render pass with squint-explicit prompts (e.g.
  "narrowed eyes" / "wide-eyed" anchors) before any Solver-C
  training. This is a corpus-rebalance task, not a slider task.
- I_{θ|η}(S) does not saturate by |S| = 1000 → corpus is so
  squint-poor that selection is starving the trainer of signal even at
  full budget. Same outcome: render more.
- Top-J pairs *look* bundled to the eye despite low ‖Δη‖_W → our η
  vector is missing a confound (likely hair, clothing, or some
  non-blendshape facial geometry). Add the missing axis to η and
  re-rank.

## Cost

- Pair enumeration is O(N²) per cell. Largest cell is ≤ ~600 rows (we
  stratified the corpus during render); 600² = 360k pairs/cell, ~30
  cells, ~10M pairs total. Numpy vectorized, well under 1 minute on a
  laptop.
- No GPU, no model load, no rendering. Pure parquet read + linear
  algebra.
- Total wall time: < 5 minutes including writing the report.

## Deliverable

Script: `src/demographic_pc/solver_c_squint_feasibility.py`. Writes:

- `output/solver_c/squint_pairs.parquet` — all candidate pairs with
  Δθ, Δη components, J, selected flag.
- `output/solver_c/squint_feasibility_report.md` — pass/fail summary
  per the criteria above, plus the cumulative-I curve as a PNG.
- `output/solver_c/squint_top50_pairs.png` and
  `..._bottom50_pairs.png` — visual sanity collages.

If the pass criteria are met, the same script is the upstream stage
of a v9-squint-Path-B training run. If they fail, the report dictates
the corpus-rebalance task instead.

## Out of scope for this doc

- Path C φ-training. Independent thread; the partial-Fisher framing
  applies to it too but the engineering is different (5–7 days vs <1
  day for Solver C feasibility). Run Path C gates 1–3 only after
  Solver C either passes feasibility and is in flight, or fails and
  triggers a render pass.
- Off-diagonal Σ_η. Diagonal Mahalanobis is sufficient for v1; if
  selection looks subtly off, recompute W from the full empirical
  covariance.
- Generalizing the formalization beyond this axis. Smile and jawOpen
  are likely safe (not bundled); age and gender are bundled but more
  conventionally so; race is bundled in ways the corpus may not let
  us de-confound at all. Each axis gets its own feasibility report.
