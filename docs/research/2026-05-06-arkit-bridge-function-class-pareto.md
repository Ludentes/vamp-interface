---
status: live
topic: arkit-bridge
---

# Function-class Pareto for ARKit→PersonaLive distill

After three v4 NN variants (`weighted_mse`, `varnorm_jvp`, anneal-A+B+C-to-v2)
all lost to v2_lam10's plain `varnorm_std_tail` on absolute `ratio_mean`
(0.0142–0.0391 vs v2's 0.0059), the v4 thread shifted from "tweak the loss"
to "tweak the function class". The honest read of v4: we explored the
loss-design axis and found v2's varnorm_std_tail is already near-optimal
for `MotEncoderStudent`. To find more headroom we need a different
hypothesis space.

This note records the plan to map a four-point Pareto across function
classes, with NMF atoms as the load-bearing prior.

## What we already know vs what we're asking

Known: at the `MotEncoderStudent` (small NN) function class, v2 ≈ 0.006
ratio is the floor we can hit with reasonable training. Question: is
that the ratio Bayes-floor for the (b_expr → m_f) mapping, or is the
NN class itself leaving headroom?

To answer, we need an *expressive* baseline that is closed-form (no
training noise). KRR with cross-validated λ is the canonical choice —
it's the function-class equivalent of "is the data telling us we can
do better than X?".

## NMF atoms as RBF centers — Nyström-KRR

Standard kernel ridge regression solves `α = (K + λI)⁻¹ y` where K is
the N×N Gram matrix on training inputs. With N≈15.6k, K is ~1 GB
float32; tractable but not interpretable.

The Nyström approximation picks m landmark points and solves a fixed-design
linear regression in the m-dim kernel-feature space:

```
Φ[i, j] = k(b_i, landmark_j)         # N×m design matrix
β = (Φᵀ Φ + λI)⁻¹ Φᵀ Y               # m×m system, all 512 outputs at once
m̂(b) = Σ_j β_j · k(b, landmark_j)    # inference: m kernel evals + matvec
```

The standard recipe picks landmarks at random or by k-means on inputs.
Our prior says **the NMF atoms are the right landmarks**.

We have (from the 2026-04-23 atom-library thread, see
`project_au_library_hybrid.md`): NMF(k=8) on 52-d ARKit blendshapes,
`au_library.npz` with reconstruction R²=0.913. The atoms are sparse
non-negative combinations of blendshapes, each interpretable as
~one action unit. The same NMF basis applied to our 58-d b_expr (re-fit
to the 15.6k corpus, k ∈ {8, 11, 16}) gives a small set of meaningful
centers.

**Why atoms beat random landmarks here.** Atoms encode the corpus's
intrinsic dimensionality of activation patterns — most pairs have low
support on the atom basis (few atoms active), so the kernel design at
atom centers efficiently covers the data manifold. Random landmarks
waste capacity on directions that don't matter. K-means on b_expr
might rediscover similar centers, but starting from atoms also gives
us interpretable β coefficients (β_j tells you "atom j's contribution
to motion cell (i, k)"), which feeds into downstream coupling analysis.

## What we expect to learn

Three bake-off scenarios:

- **NMF-Nyström KRR ≈ v2 (~0.006).** v2 is near-optimal *for this data
  distribution*. Function class doesn't matter much; v4 was always
  doomed. Ship v3, archive the v4 family. KRR becomes a deployable
  alternative student (smaller, deterministic).
- **NMF-Nyström KRR ≪ v2 (e.g. 0.002).** NN is leaving real headroom.
  The 11-atom RBF basis captures the (b → m) mapping more efficiently
  than `MotEncoderStudent` finds it. Either swap the student or rethink
  its architecture (linear-on-atoms readout? larger NN?).
- **Full-Gram KRR ≪ NMF-Nyström KRR.** Atoms are insufficient; the
  manifold has structure beyond the 11 AU-like directions. Re-fit
  NMF with larger k, or move to RFF / k-means landmarks.

## Pareto plot we want

```
function class                 ratio_mean   median R²   inference cost   notes
─────────────────────────────────────────────────────────────────────────────
linear ridge (Φ = b_expr)      ?            ?           58×512 matvec    floor
NMF-Nyström KRR (k=8, 11, 16)  ?            ?           k×512 matvec     headline
Full-Gram KRR (random 2k)      ?            ?           2k×512 matvec    ceiling
v2_lam10 60k (current best NN) 0.0059       ?           NN forward       known
v2_lam10 120k                  ?            ?           NN forward       does v2 plateau?
v4a 90k (weighted_mse)         0.0147       0.893       NN forward       known
v4b 90k (varnorm_jvp)          0.0391       0.767       NN forward       known
v4c 120k (anneal A+B+C → v2)   0.0142       0.898       NN forward       known
```

If ridge ≪ NN, NN is wasting capacity (unlikely). If KRR ≪ NN, the
function class is wrong. If NN ≈ KRR, v2 is genuinely Bayes-near.

## Concretely

Files needed:

- `scripts/fit_nmf_atoms_v4.py` — re-fit NMF on the 15.6k b_expr corpus
  for k ∈ {8, 11, 16}, store atoms.npz with reconstruction R² per k.
- `scripts/fit_krr_baseline.py` — closed-form Nyström-KRR readout, with
  NMF atoms or random/full landmarks depending on `--landmark_mode`.
  Single λ-sweep (log-spaced), eval on holdout_v3, dump JSON +
  per-cell R² npz.
- `scripts/plot_function_class_pareto.py` — small plotting utility,
  reads all eval JSONs, emits one summary plot + table.

Decision rule: if any KRR variant has `ratio_mean < 0.5 × v2_best` we
proceed to deploy KRR alongside NN. Otherwise document v2 as the
deployable student and v4 falsified.

## Why not RBF networks, SVR, boosting, ...

- **Plain RBF networks** (random centers, SGD-trained linear readout)
  are strictly weaker than KRR with the same centers — same hypothesis
  space, worse optimizer. No reason to test.
- **SVR** uses ε-insensitive loss; equivalent to KRR with sparse-α
  regularization. For continuous regression to a smooth target like
  blendshape→motion, KRR with squared loss (matching our `ratio_mean`
  metric) is what we actually want.
- **Boosting (XGBoost/LightGBM)** trains 512 independent regressors
  per output cell — ignores inter-output coupling we *know* exists
  (the cross-cell R² mask is a real metric we care about). Worth
  running as confirmation *only if* KRR underperforms; otherwise skip.

## Open questions

- The 2026-04-23 NMF was on a different corpus (Flux render samples).
  We'll re-fit on b_expr from `data/arkit_bridge_pairs/all` to make
  sure the atoms are appropriate to this distribution. If the new
  atoms drift far from `au_library.npz`, that's a corpus-shift signal.
- KRR's σ (RBF bandwidth) is a hyperparameter. Standard default:
  median-distance heuristic on training pairs. Document the choice.
- We may want to combine KRR readout with a small NN refinement
  (KRR as prior + NN learns the residual). Defer until KRR baseline
  numbers are in.
