---
status: live
topic: arkit-bridge
---

# ARKit-bridge: corrected baseline, function-class Pareto, next-experiments brainstorm

Consolidated session note (2026-05-06 evening). Captures the corrected
v2 baseline finding, the full function-class Pareto, the per-frame
teacher confirmation, and the ranked ideas list for the next training
session. Compaction-resistant — read this and the dated companions if
the conversation summary is missing.

Companions:
- [2026-05-06-arkit-bridge-function-class-pareto.md](2026-05-06-arkit-bridge-function-class-pareto.md) — original plan (some claims now corrected; supersedes are inline below)
- `docs/superpowers/plans/2026-05-06-arkit-bridge-v4.md` — the v4 plan that produced v4a/v4b/v4c
- [project_blendshape_temporal_availability.md](../../../home/newub/.claude/projects/-home-newub-w-vamp-interface/memory/project_blendshape_temporal_availability.md) — referenced for temporal-loss thinking

## State of play (corrected)

Earlier session framing claimed "v4 lost to v2_lam10's 0.0059". That number
came from `runs/student_v2_lam10/eval_log.json` — but that eval was on a
**different holdout** (n=1652, an earlier `holdout_v2`), not `holdout_v3`
(n=822) that all v4 work uses. Re-evaluating `student_v2_lam10/student_best.pt`
on `holdout_v3` gives `ratio_mean=0.7508, R²med=-2.47` — the original v2_lam10
checkpoint **does not generalize** to holdout_v3 because the v3 take content
was likely in its training set.

The real v2 baseline on holdout_v3 is from re-trained v2 runs:

| run                                | steps  | best ratio_mean | R²≥0.7 | notes                       |
|------------------------------------|--------|-----------------|--------|-----------------------------|
| v2_120k (lam_std=1.0)              | 120k   | **0.0114**      | 1.000  | new run, holdout_v3         |
| v2_lam10_120k (lam_std=10)         | 120k   | **0.0114**      | 1.000  | reproducing the "lam10" config |
| v4a (weighted_mse)                 | 90k    | 0.0147          | 1.000  | A+B reweighting             |
| v4b (varnorm_jvp)                  | 90k    | 0.0391          | 0.785  | C: JVP-norm regularizer     |
| v4c (anneal A+B+C → v2)            | 120k   | 0.0142          | 1.000  | β:1→0 over 0–75k            |

**Revised conclusion: v4 didn't lose to v2 — they tied within 30%.** All
loss-design tweaks land in 0.011–0.014 territory on this holdout. The
old "v4 falsified, ship v3" framing was based on a phantom 0.0059
baseline; with the right baseline, v4 and v2 are statistically
indistinguishable on offline ratio_mean.

`lam_std` doesn't matter much: lam_std=1.0 and lam_std=10.0 both hit
0.0114 at 120k. The std-match auxiliary is a small lever.

## Function-class Pareto on holdout_v3

| function class                          | best λ | ratio_mean | R²med | R²≥0.7 |
|-----------------------------------------|--------|------------|-------|--------|
| linear ridge (Φ = b_expr + bias)        | 1e-6   | 0.0553     | 0.614 | 0.189  |
| 11-landmark Nyström KRR (NMF-raw)       | 1e-6   | 0.4605     | -0.82 | 0.000  |
| 11-landmark Nyström KRR (NMF-active)    | 1e-6   | 0.1309     | 0.221 | 0.000  |
| 11-landmark Nyström KRR (k-means)       | 1e-1   | 0.0938     | 0.361 | 0.004  |
| 11-landmark Nyström KRR (random)        | 1e-6   | 0.1012     | 0.318 | 0.000  |
| 128-landmark Nyström KRR (k-means, σ=1.3) | 1e-6 | 0.0461     | 0.677 | 0.408  |
| Full-Gram KRR (N=15.6k, σ=1.3)          | 1e-3   | 0.0273     | 0.809 | 0.908  |
| **Small NN (v2/v4 family)**             | —      | **0.011–0.014** | **1.000** | **1.000** |

σ-tuning on k=128 didn't beat the median heuristic (σ=1.3): {0.3, 0.6, 1.3, 2.6}
gave {0.116, 0.054, 0.046, 0.049} respectively. Median was best.

NMF active-mode (H[j] · w_j_p75 to put landmarks inside the data cloud)
fixed the geometric pathology of NMF-raw (median ‖b−landmark‖/σ went from
catastrophic to 0.91), but k-means still beat NMF for the same k=11.
**The atom prior didn't pay off here.**

Honest read of the Pareto: **the small NN's hidden-layer feature
transform is doing real work that no kernel method we tested can
match cheaply**. Even full-Gram KRR (15.6k centers, σ-tuned) is 2.4×
worse than the NN. Not because the kernel is bad in principle — because
b_expr → m_f apparently has structure that's better fit by a learned
nonlinear feature transform than by Gaussian similarity.

## Architecture: per-frame is correct

We verified that PersonaLive's `MotEncoder` is per-frame
(`src/models/motion_encoder/encoder.py:35-41`):

```python
def forward(self, x):  # x: (b, c, f, h, w)
    latent = self.model(rearrange(x, "b c f h w -> (b f) c h w"))
    # frames flattened, processed independently — no temporal mixing
    ...
```

There is no temporal cross-frame attention or convolution inside the
teacher. The "4-frame" memory is the *consumer* side: PersonaLive's
`wrapper.py` batches multiple frames through MotEncoder and slices the
output along the frame axis to populate `motion_pile`/`motion_buffer`.
Architecture verdict: ✅ our per-frame student type-matches the
teacher exactly.

But: downstream consumers (motion_pile, temporal_window_size) integrate
m_f temporally in the diffusion stack. So per-frame mismatches that
average out in offline `ratio_mean` can still **manifest as visible
wobble in rendered output** because the diffusion pipeline amplifies
per-frame inconsistency.

This reframes "headroom":

- Offline `ratio_mean` is a per-frame metric. Plateauing at 0.011 here
  may mean the per-frame map is genuinely as good as it gets.
- Rendered quality / wobble is a sequence metric. *Different headroom
  axis.* Could be improved without moving offline ratio at all (via
  temporal smoothness loss, sequence student, or — most likely —
  cleaning up the input crop wobble at extraction time).

## Idea inventory

Ranked by expected leverage. Sorted into three buckets.

### Bucket 1: dataset-side (highest leverage, cheapest)

These don't change the model or loss; they fix corpus pathologies.

#### Crop-wobble diagnostic + stabilization (highest leverage, half-day if needed)

The hypothesis (from `project_facemesh_crop_wobble.md`): per-frame
`crop_face` recomputes bbox each frame → input image jitter → teacher
outputs noisy m_f even when b_expr is constant. The student then
learns this noise as part of its target.

Diagnostic (1 minute): on one take's existing pkls, compute
frame-to-frame bbox delta. If >5px on adjacent frames at neutral
expression, the corpus is noisy at the source.

Fix (half-day): refit `extract_arkit_pairs.py` to use EMA bbox or
detect-every-Nth + interp. Re-extract whole corpus. Retrain v2 at 90k.
**Probably the largest single ratio_mean improvement available.**

#### Symmetry re-extraction (high leverage, ~10 min compute)

For each existing video, also run the teacher on the
horizontally-flipped image. Pair with the L↔R-swapped b_expr
(swap ~14 pairs: MouthSmileLeft/Right, EyeBlinkLeft/Right, ...).
Doubles the corpus, bakes in face symmetry as a hard prior.

Cannot just synthesize the flipped m_f because the teacher's not
flip-equivariant. Has to re-run MotEncoder. ~80s teacher inference
for 16k frames.

Implementation: clone `extract_arkit_pairs.py` to also process
`np.fliplr(rgb)` and write `*_frame_NNNNNN_flip.pkl`.

#### Frame-quality filter (cheap, modest impact)

Drop pairs where extraction was lossy:
- `‖m_f‖ > p99` of corpus norm — likely teacher artefact
- mediapipe detection confidence below threshold — wrong b_expr
- consecutive-frame m_f delta huge while b delta small — teacher
  inconsistency, probably crop instability

Likely removes 1–5%, all of which contribute disproportionate gradient
noise. Implement as a one-shot `scripts/filter_pairs.py` that emits a
new `pairs_dir` symlink.

#### Per-take reweighting (free, modest impact)

Corpus is uneven — long expressive takes dominate, short ones
(asym_smile n=11, broad_smile n=46) starved. Reweight by
`1/sqrt(take_size)` so every take contributes comparably.

Trivial extension to `precompute_v4_stats.py`'s `sample_weights`:
multiply existing weight by per-take reweighting factor.

### Bucket 2: loss-side (modest leverage, focused fixes)

#### Symmetry consistency loss (top loss-side pick)

Per-step augmentation: compute prediction on both `b` and `flip(b)`,
penalize `‖model(flip(b)) − flip(model(b))‖²`. Effectively the same as
symmetry re-extraction but at training time and "soft" (a regularizer,
not a hard new dataset). Could combine with re-extraction for
double-benefit.

Cost: 1 extra forward per step (~50% slower training); no extra
extraction. ~30 LOC change to the train loop.

#### Temporal smoothness loss

Sample consecutive-frame pairs `(b_t, m_t)` and `(b_{t+1}, m_{t+1})`.
Add `λ · ‖model(b_{t+1}) − model(b_t)‖² · w(‖b_{t+1} − b_t‖)` where
`w(·)` is a falloff that activates only when input change is small.
Encourages the student to be smoother than the teacher in time, which
helps downstream rendered wobble.

Trade: probably costs some offline ratio_mean. Only worth it if
rendered output quality is the actual bottleneck.

#### Antagonist exclusion prior (small effect)

Penalize predictions for "impossible" blendshape combinations
(BrowDown ↔ BrowOuterUp both >0.5, CheekPuff ↔ CheekSquint, ...).
A regularization term on the input distribution, not the output
distribution. Cheap, small expected gain.

#### Atom-bottleneck loss

Use existing NMF k=11 atoms. Project both student pred and teacher
target onto atom-aligned subspaces, compute MSE in atom-space (in
addition to per-cell MSE). This is "match the model in AU space".
Could help tail recovery on rare AUs. Probably small gain since the
existing varnorm + tail mining already targets the same problem.

#### Hard-example mining (online curriculum)

Train ~10k, run student on full corpus, identify top-5% MSE pairs,
upweight 5× for next 10k. Repeat. Standard online HEM. Diminishing
returns vs the dataset-side fixes for this problem; ~50 LOC + dataloader
rewrite.

### Bucket 3: architecture-side (high cost, uncertain payoff)

#### LSTM / TCN sequence student

Reorganize dataset by `take_id`, train a small temporal model on
context windows. **Won't beat per-frame on offline ratio_mean** because
the teacher itself is per-frame — sequence info doesn't help fit a
per-frame function. **Could** help rendered wobble (the main downstream
quality concern) by producing smoother sequences. ~1–2 days of work
for a payoff that may not show up on our existing metric.

#### Deep kernel learning

NN feature extractor + KRR readout. `MotEncoderStudent` already
approximates this with its hidden layer; the difference would be
fitting the last layer in closed form via ridge. Marginal tweak,
likely negligible gain.

#### RBF networks / boosting / SVR

Explicitly **falsified by the function-class Pareto.** Plain shallow
RBF nets are dominated by KRR (which we tested at 0.027). Boosting
discretizes a smooth target and ignores output coupling — predicted
0.03–0.05 range, worse than NN. SVR ≡ KRR with sparse-α. None of these
will beat the current NN. **Skip.**

## Recommended next-experiment stack

Highest-leverage, lowest-risk path:

1. **Crop-wobble diagnostic** (1 min). Resolves the bottleneck question.
   - If wobble confirmed → schedule crop-stabilized re-extraction as
     a separate experiment, defer to later session.
   - If wobble not detected → proceed.
2. **Symmetry re-extraction** (~10 min compute + ~5 min code). Doubles
   corpus, bakes chirality prior into data.
3. **Frame-quality filter** (~10 min). Removes corpus noise.
4. **Per-take reweighting** (~5 min code). Free fairness fix.
5. **Train v5** at 90k with `varnorm_std_tail`, all four data-side fixes
   active. ~3 min GPU.
6. Eval v5 on holdout_v3 + render yaw clip. Compare to v2_120k baseline.

Total: ~40 min of human time for steps 1–4, then a single 90k training
run. Worst case (no improvement): we have a clean corpus and a falsified
augmentation hypothesis. Best case: ratio_mean drops 30–50%.

If v5 is meaningfully better, layer in **symmetry consistency loss**
(soft version on top of re-extracted dataset) and train v6. If v5 ties
v2_120k, declare offline `ratio_mean` saturated for this dataset and
move attention to **rendered-quality** axis (temporal smoothness loss
or LSTM, depending on how bad the wobble looks).

## What we will NOT do (YAGNI)

- More v4-style loss-mode experiments. Loss design is exhausted.
- Boosting. Falsified by function-class Pareto.
- RBF networks (any flavor). Strictly dominated by KRR.
- Mixup, input noise injection. Wrong tool for smooth-target regression.
- Full Bayesian KRR / Gaussian process tuning. We have the ceiling
  number; tuning σ further didn't help.
- Synthetic blendshape→image data generation. Circular.

## Open questions for the future

- Does the crop-wobble hypothesis actually explain the 0.011 floor, or
  is it irreducible per-frame noise from the teacher itself?
- If we go LSTM, is the right comparison "student gets temporal context
  the teacher didn't" (cheating?) or "student outputs are smoothed
  post-hoc" (degenerate)?
- The R²≥0.7=1.000 saturation in our metrics suggests the threshold is
  too easy; future bake-offs should use ratio_mean alone or move to
  R²≥0.85.
- Should we re-evaluate the original v2_lam10 on its actual training
  holdout to confirm 0.0059 was a real number for *that* split, or is
  it fold-leakage from the get-go?
