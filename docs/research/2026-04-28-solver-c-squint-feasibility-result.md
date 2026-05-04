---
status: live
topic: demographic-pc-pipeline
supersedes: 2026-04-28-solver-c-partial-fisher.md
---

# Solver C feasibility — squint corpus is Duchenne-bundled (productive failure)

Companion to `2026-04-28-solver-c-partial-fisher.md` (the v1.1 spec
+ adversarial-review revisions). Records what running it actually
produced: a fail, but the kind of fail the framework was built to
catch — diagnostic, pre-compute, and prescriptive about what to do
next.

## Result

Script: `src/demographic_pc/solver_c_squint_feasibility.py` (v1.1).
Inputs: 8,115 Flux corpus rows from `sample_index.parquet` joined
to `arcface_fp32` from `output/reverse_index/reverse_index.parquet`
(317 rows dropped where ArcFace did not detect). Within-(ethnicity,
gender, age) cells; η = 6 blendshape scalars + 17 base one-hots + 8
ArcFace PCs (31 dims); W = full Σ_η^{−1} with Tikhonov λ=1e-3.

Three-criteria pass card:

| criterion | result | value |
|---|---|---|
| ≥ 200 pairs above p90 of J | **PASS** | 487,038 |
| σ_θ² rank stability (min off-diag Spearman ≥ 0.90) | **PASS** | 1.0000 |
| selected Δη effective rank ≥ max(4, η-dim/4) | **FAIL** | 3 / 31 |

Top-3 eigenvectors of selected-Δη covariance, in `eta_names` basis:

```
PC0 (59% of variance):  smile 31% + blink 25% + arc_pc4 23% + cheek 9% + gaze 9%
PC1 (11% of variance):  blink 36% + brow 27% + cheek 20%
PC2 (8% of variance):   brow 61% + small mixes
```

PC0 is the Duchenne bundle. The Flux corpus has ample squint variance
(median |Δθ| in selected pairs = 0.832 ≈ 2.2σ of θ) but **no
squint-isolated pairs**: every high-Δsquint pair within a demographic
cell also moves smile + blink + cheek + identity-shape together. PC1
adds the eye-closure/brow tension residual; PC2 adds independent brow
movement. After three directions, the selected-pair spread collapses
into a thin shell — there is no fourth independent direction to span
across the η complement.

## Why this is a *good* failure

The framework was built to catch exactly this case before any LoRA
training. It worked.

What we did not pay:

- One v9-squint Path B training run (~1 GPU-hour at v8 hyperparameters)
  that would have produced a working LoRA labeled "squint" but
  encoding "Duchenne smile + cheek raise + eye closure."
- One eval battery (~50 min, 216 cells, ~30 GB peak RAM under the
  systemd-run cap) that would have shown high Spearman ρ on the
  intended-axis metric (because eyeSquint *is* part of Duchenne) and
  obscured the bundle by metric design.
- One round of debugging "why does the squint slider also smile."

What we paid: ~50 lines of pair-scoring code, < 1 minute runtime, and
a falsifiable read on the corpus. The diagnostic confirmed in the
data what `2026-04-26-solvers-taxonomy-and-next-steps.md` predicted
qualitatively at lines 191–196 — squint is in the same bundled-axis
family as glasses, but tighter; capacity won't escape it.

The framework's structural value: it gave a *typed result* — not "the
training is bad" or "the slider drifted," but "the η directions
{smile, blink, cheek, brow, identity} co-move with θ and the corpus
contains no compensating variance." That type of result tells us
which route to switch to instead of which knob to retune.

## What this means for v1.1's framing

Two doc-level corrections to `2026-04-28-solver-c-partial-fisher.md`
based on what running revealed:

1. **σ_θ² is irrelevant on this corpus.** The sweep gave perfect 1.0
   Spearman across {1%, 5%, 25%} of var(θ). `Δη^T W Δη` always
   dominates the denominator — no pair is in the σ_θ² floor regime.
   The hyperparameter the adversarial review flagged as fragile turns
   out not to bind. Future axes may differ; keep the sweep, but it's
   currently information-free.
2. **Confound compression dropped 14× → 2.4× under full Σ_η^{−1}.**
   v1's diagonal W produced inflated compression by double-counting
   correlated confounds. The full-matrix metric is fairer and a more
   honest measure of how much Solver C can de-bundle a given corpus.
   Don't read the v1 14× as load-bearing.

## Three forward routes (cost ordering)

The bundle is in the data, not the metric. No reranking saves it.
Three real moves, in increasing engineering cost:

1. **Composition route (Solver A, zero training).** We already have
   `eyeSquint`, `smile`, and `cheekSquint` as FluxSpace prompt-pair
   δs and a working `compose_iterative` that cancels co-firing
   confounds via additive δ composition (cf.
   `project_editing_framework_principle` memory and
   `FluxSpaceEditPairMulti` from 2026-04-23). The squint axis can be
   delivered at render time as `+squint − smile − cheek` without any
   slider training. Cost: minutes of rendering plus existing
   compose-iterative tuning. **Try first.**

2. **Corpus rebalance + rerun Solver C.** Render ~500–1000 prompts
   with explicit "narrowed eyes, neutral mouth, relaxed cheeks" and
   "wide-eyed, neutral mouth" anchors, stratified across demographic
   cells. Goal: introduce the squint-isolated pairs the existing
   corpus does not contain. Rerun v1.1 — if PC0 of selected Δη
   redistributes across {smile, cheek, brow, …} instead of
   collapsing onto Duchenne, Solver C passes and feeds v9-squint
   Path B. Cost: ~4–8 GPU-hours rendering + 1 GPU-hour training +
   eval. Triggered by composition-route failure or insufficiency.

3. **Path C (latent φ surrogate, the original "C is forced by
   squint" route).** Train `φ_squint(z) → squint score` on per-render
   labels (no pair requirement). Use `−t·∇_z φ(x̂_0)` as
   per-input supervision — gradient is computed per-(x_t, t) and can
   in principle find the squint direction even when the empirical
   pair distribution is bundled, because the supervision is not a
   fixed direction in latent space. 5–7 days per
   `2026-04-26-measurement-grounded-slider-plan.md`, with kill-gates
   1–3 first (R² ≥ 0.85, gradient structure visible, no-LoRA gradient
   ascent produces visible squint). Triggered by composition + rebalance
   both failing or insufficient.

## Recommendation

Composition route first. It costs rendering minutes and uses
infrastructure that already exists and is validated cross-axis
(`project_fluxspace_smile_axis`, `project_fluxspace_crossdemo`). If
the `+squint − smile − cheek` composite produces clean perceived
squint at calibrated scales, we ship it for the blendshape bridge
without a slider training run. If it produces uncanny artifacts (over-
cancellation) or fails to compose, the rebalance becomes the cheaper
of the two remaining options.

If we end up running rebalance, the prompt design should explicitly
target the η directions Solver C identified as bundled:

- Mouth-neutral squint anchors at varying squint intensity (cancels
  PC0's smile + cheek loading).
- Eyes-open / eyes-relaxed-but-mouth-smiling anchors (cancels PC0's
  blink loading).
- Brow-neutral squint variants (cancels PC2 residual).

That is, the rebalance prompts should be *informed by the eigenvector
decomposition*, not generic squint synonyms.

## What does NOT need to change

- The Concept Sliders trainer port plan (Path B). If rebalance passes
  Solver C, the trainer port is unchanged from the v9-squint default.
- The arc_latent distillation plan
  (`2026-04-27-arc-latent-distillation-plan.md`). Independent thread;
  Path C's feasibility is unblocked by it but does not require it for
  the kill-gates.
- The glasses v8 production checkpoint. Unaffected.

## Cost in retrospect

- ~50 lines of pair-scoring code (v1 plus v1.1 deltas).
- < 1 minute runtime, no GPU.
- One adversarial review pass that materially improved the metric
  (full Σ_η^{−1}, ArcFace η, eff-rank diversity criterion). The
  σ_θ² sweep turned out to be information-free on this corpus but
  the diagnostic is cheap and worth keeping.
- Avoided: 1 GPU-hour training + 50 min eval + an unbounded debug
  thread on a slider that would have looked like it worked.
