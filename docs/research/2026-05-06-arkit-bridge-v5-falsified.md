---
status: live
topic: arkit-bridge
supersedes: 2026-05-06-arkit-bridge-next-experiments.md
---

# v5 dataset-side stack: falsified

The recommended experiment from
`2026-05-06-arkit-bridge-next-experiments.md` was a stack of four
data-side fixes layered together. Result: v5 plateaued at
**ratio_mean=0.0182** at 90k, **60% worse than v2_120k's 0.0114
baseline**. The dataset-side hypothesis as constructed is falsified.

## What v5 did

1. **Crop-wobble diagnostic** — ran `scripts/diagnose_crop_wobble.py`
   on `MySlate_2`. With EMA(α=0.2) on a 621px-side face, center jitter
   p95=1.5px (0.25%), side jitter p95=7.5px. Even raw mediapipe is
   only 4–7px p95. **Crop wobble is not the bottleneck.** Skipped the
   half-day re-extraction.

2. **Symmetry re-extraction** — added `--flip` to
   `scripts/extract_arkit_pairs.py` and `arkit_bridge.symmetry`
   module. Flips the input image, swaps 20 L↔R blendshape pairs,
   negates eye yaw/roll, swaps L↔R eye rotations. Re-ran teacher on
   flipped frames for all 7 takes. 16507 flipped pkls produced.

3. **Frame-quality filter** — `scripts/build_train_v5.py --filter_p99`
   built `data/arkit_bridge_pairs/all_v5/` by hardlinking originals
   plus train-eligible flipped pairs (excluding flipped versions of
   holdout_v3 source frames), dropping 303 pkls (~1%) where
   `‖m_f‖ > p99=22.39`. Final corpus: 30,941 pairs.

4. **Per-take reweighting** — `scripts/apply_per_take_reweight.py`
   took `runs/v5_shared/stats_raw.npz` (computed on `all_v5/`) and
   multiplied `sample_weights` by `1/sqrt(take_size)`, normalizing
   to mean=1. Result: 7 takes with sizes 655–7078, sample_weight
   range 0.226–8.808.

5. **Trained v5** at 90k with `varnorm_std_tail`, `lam_std=1.0`,
   `lam_tail=0.5`, `active_channel` sampler over the reweighted
   stats. Same architecture and learning rate as v2.

6. **Eval on holdout_v3** — converged smoothly, ratio descended
   monotonically: 0.067 → 0.052 → 0.044 → … → 0.0182 at 90k. R²≥0.7
   hit 0.994 at step 84k (matching v2). Decision: ship v2_120k as
   the deployable student; archive v5 as a falsified augmentation
   experiment.

## Comparison

| run                            | ratio_mean | R²≥0.7  | corpus |
|--------------------------------|------------|---------|--------|
| v2_120k                        | **0.0114** | 1.000   | 15.6k orig |
| v2_lam10_120k                  | 0.0114     | 1.000   | 15.6k orig |
| v4a (weighted_mse, 90k)        | 0.0147     | 1.000   | 15.6k orig |
| v4c (anneal A+B+C → v2, 120k)  | 0.0142     | 1.000   | 15.6k orig |
| **v5 (flip+filter+reweight)**  | **0.0182** | 0.992   | 30.9k aug |

## What this falsifies

The plan's "best case: ratio_mean drops 30–50%" did not materialize.
Specifically:

- **Symmetry augmentation does not transfer.** Doubling the corpus
  with mirror-flipped pairs trained the model to fit the
  joint (orig + flip) distribution, but the flipped distribution is
  evidently not faithful to teacher's natural output for true
  flipped faces — pulling the model away from the holdout's
  un-flipped distribution.

  Specifically: the teacher (PersonaLive's MotEncoder) is *not
  flip-equivariant by construction*. We computed `m_f` from
  flipped images directly, not by flipping the un-flipped `m_f`,
  so we trusted the teacher to handle flipped inputs correctly.
  But its training distribution was probably mostly
  upright/non-mirrored faces, and what it produces on flipped
  inputs may have different statistics (or even non-trivial
  systematic biases). Treating its flipped output as ground truth
  contaminated training.

- **Per-take reweighting hurts here.** The holdout has the same
  per-take distribution as the original training set; equalizing
  per-take contribution moved the training distribution further
  from the holdout's natural skew, not closer.

- **The p99 norm filter is small (1% drop) and probably innocent.**
  Not the cause of regression.

The first point is the load-bearing one. We need a different
augmentation strategy: either (a) flip the **m_f** along the
appropriate axes after running teacher on un-flipped image (i.e.
build a flip-equivariance prior by construction), or (b) accept
that teacher mismatch on flipped inputs is real and use symmetry
only as a soft regularizer (penalize student violation of
equivariance, but don't add flipped pairs as hard targets).

## Falsified stack: full list

After v4 (loss design exhausted) and v5 (dataset augmentation
exhausted), the falsified-by-experiment list is:

- ❌ Loss-design tweaks: weighted_mse, varnorm_jvp, anneal A+B+C
- ❌ Function-class kernel methods (Pareto): KRR Nyström (NMF, k-means,
  random), full-Gram KRR
- ❌ Mirror-flip augmentation as hard targets (this doc)
- ❌ Per-take 1/sqrt(N) reweighting (this doc)

## Open ideas not yet falsified

- **Soft-symmetry consistency loss.** Penalize
  `‖model(flip(b)) − flip(model(b))‖²` at training time. Doesn't
  introduce contaminated targets; only constrains the student's
  *own* equivariance. ~30 LOC change. Worth a try.
- **Flip-equivariant target construction.** Run teacher on
  un-flipped image, then flip its output `m_f` along motion-pile
  spatial axes (assumes m_f has known left/right channel structure
  — needs verification by inspecting MotEncoder output layout).
  This would give "true" flipped targets without trusting teacher
  on flipped inputs.
- **Temporal smoothness loss.** For rendered-quality wobble, not
  offline ratio_mean. Different headroom axis.
- **Crop-wobble re-extraction.** The diagnostic ruled it out at the
  EMA level on one take. Worth checking other takes if any have
  fast head motion.

## Decision

Ship `runs/student_v2_120k/student_best.pt` as the deployable
ARKit→PersonaLive bridge student. Mark v4 family + v5 as falsified
in the plan-of-record. Defer further offline `ratio_mean` work
until the rendered-wobble bottleneck (Bucket 2 / 3 in the
next-experiments doc) is investigated as a separate axis.
