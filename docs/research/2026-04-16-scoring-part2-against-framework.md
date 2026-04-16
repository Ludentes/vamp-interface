# Scoring Part 2's proposal against the framework protocol

**Date:** 2026-04-16
**Status:** Draft scoring exercise. Where the Part 2 blog[^blog] proposed a training and evaluation strategy, this doc runs that strategy through the framework's Decision Protocol[^framework] (`v2/framework/math-framework.md` §2.7, v0.11) as if Tier 1 and Tier 2+ were candidate identity-channel mechanisms awaiting evaluation.
**Purpose:** turn Part 2 Claim 8.1 (*"Tier 1 hand-written conditioning + regression on `P` is very likely sufficient"*) from a position statement into a falsifiable scoring sheet, and surface the places where Part 2 implicitly re-litigates decisions the framework already made.
**Audience:** internal — assumes familiarity with both `v2/framework/math-framework.md` and `docs/blog/2026-04-16-part-2-math-of-pattern-preservation.md`.

---

## 1. Why this exercise

The framework is designed for scoring candidate mechanisms `f: X → P` (qwen vector → face) along a set of properties with explicit thresholds. Part 2 proposed two such mechanisms — Tier 1 and Tier 2+ — and their theoretical discussion. But the blog never ran either of them through §2.7's protocol. Doing that here produces three outputs:

1. A feasibility verdict per candidate (pass / fail / needs-measurement per hard constraint).
2. A Pareto comparison between Tier 1 and Tier 2+, with explicit axes.
3. A list of vocabulary overlaps where Part 2 re-defined things the framework already names — opportunities to unify rather than fragment.

It also produces a concrete measurement plan: every "needs-measurement" entry in the feasibility filter maps to one framework experiment (Exp A–E), and that experiment maps back to one Part 2 stage.

---

## 2. The candidates being scored

Both candidates share everything except how `P` is obtained. Formally:

```
f_any(x) = Flux( φ(qwen(x)), seed = hash(x_id) )
```

where `φ: qwen_1024 → Flux_conditioning` is the controllable piece. Frozen: `qwen`, `Flux`, the face encoder `Ψ` used for evaluation, and the v3 drift channel (Cursed_LoRA_Flux + Eerie_horror denoising-strength mechanism).

**Tier 1.** `φ(q) = c_hand( axis_hat(q) )` where `axis_hat` is a linear projection from qwen to axis-label probabilities (or directly to conditioning), fit by least-squares regression on training pairs `(qwen_i, c_hand(axis_i))`. `c_hand` is a lookup table authored by hand per axis level. No Flux gradients in training.

**Tier 2+.** `φ(q) = P_NN(q)` where `P_NN` is a small neural network trained end-to-end on the four-term loss (Part 2 §7.2) via DRaFT-1 / Adjoint Matching / Flow-GRPO. Flux gradients (or policy gradients) are required for training.

The framework's protocol does not score the training procedure per se; it scores the resulting `f`. Tier 1 and Tier 2+ are therefore two instantiations of the same functional type, fed into the same rubric.

---

## 3. Stage-to-experiment map

Part 2's staged buildout plan (§9, Stages 0–7) and the framework's experiment list (§5, Exps A–E) describe overlapping work. The mapping:

| Part 2 stage | Description | Framework experiment | Description | Notes |
|---|---|---|---|---|
| Stage 0 | Baseline metric panel on v3 + ArcFace | **Exp E** | Metrics-toolbox sweep on 543-job v3 baseline | Same experiment, different doc. |
| Stage 1 | Readout-only (FaRL linear probe on v3 faces) | *(not in Exp list)* | Implicit in Exp E's per-encoder comparison | Informs ψ choice; framework doesn't have a separate experiment for this. |
| Stage 2 | Detective corpus generation + readout measurement | **Exp D** (first half) | Hidden-pattern detective experiment, ground-truth planting | Detective corpus is the vehicle for Exp D. |
| Stage 3 | Hand-written conditioning per axis | **Exp B** (input side) | FluxSpace direction-basis extraction for 4+4 axis candidates | Hand-authored axes become candidate directions for Exp B. |
| Stage 4 | Regression-fit `P` (Tier 1 headline) | *(not in Exp list)* | First instantiation of a scored candidate `f` | Tier 1's `f` enters §2.7.1 scoring after Stages 0–3. |
| Stage 5 | DRaFT-1 end-to-end (Tier 2+) | *(not in Exp list)* | Second candidate `f` | Tier 2+'s `f` enters §2.7.1 scoring. |
| Stage 6 | Adjoint Matching / Flow-GRPO | *(not in Exp list)* | Further Tier 2+ variants | Multiple candidates; Pareto-compared. |
| Stage 7 | Unlabeled-regime validation on real data | **Exp A** (Y-source decomposition) + generalization run | Measures whether recovered structure is suspicion or nuisance | Maps partially; Stage 7's full intent exceeds Exp A's 1-hour scope. |
| *(not in stages)* | — | **Exp C** | Editorial/drift distinction pivot experiment | Framework-level architectural decision, runs after Exp A/B. |

**Implication for Part 2:** Stages 0, 2, and 3 are the same work as Exps E, D, and B. The framework's experiment list predates Part 2 by weeks; Part 2 should reference the experiments by name rather than re-describing them as stages. Tier 1's Stage 4 and Tier 2+'s Stages 5–6 are **new candidates being fed into §2.7.1**, not experiments in themselves.

---

## 4. Feasibility filter — §2.7.1 hard constraints

The framework's acceptance criteria. Each row: the constraint, the threshold, the current status on v3 baseline (where known), which experiment measures it, Tier 1 expected outcome, Tier 2+ expected outcome.

| Constraint | Threshold | v3 baseline | Measured by | Tier 1 expected | Tier 2+ expected |
|---|---|---|---|---|---|
| **P*** user-task accuracy vs text baseline | `P* > P*_text + ε` | not measured | Exp D (Stage 2) | unknown; Claim 8.1 asserts sufficient | unknown; Claim 8.1 asserts uncharacterized improvement |
| **P4a global** ρ_Spearman(d_qwen, d_face) | `≥ 0.6` | not yet measured in panel form | Exp E (Stage 0) | plausible if c_hand covers axes; needs measurement | trained directly against this signal (SoftRankSpearman term) |
| **P4a local** k-NN preservation at k=5, 10 | `≥ 0.5, 0.6` | not yet measured | Exp E (Stage 0) | plausible (OLS preserves rank structure implicitly) | trained directly via SoftRankSpearman |
| **P15 stability** variance across seeds | `≤ 0.01`; game-mode bit-equal | v3 ships deterministic | structural | **passes trivially** — P is deterministic, seed is hashed | passes if inference is deterministic (it is once P_NN converges) |
| **P4 Fisher ratio** `F(f) / F(qwen)` | `≥ 0.8` | not yet measured in panel form | Exp E (Stage 0) | plausible; inherits v3 archetype structure | trained directly via CE term |
| **D1/D2 drift correlation** `r(d_anchor, y)` | `≥ 0.7` | **`r = +0.914`** (2026-04-07) | already passed | **inherits, passes** | **inherits, passes** |
| **Y1 orthogonality** pairwise cosine of committed axis directions | `≤ 0.3` | not measured (no committed axes yet) | Exp B | untested; depends on c_hand axis design | untested; not directly optimized |
| **Y2 axis budget** `|E| + |D| ≤ K_max` | `K_max ≤ 5` (FluxSpace-paper initial) | inherited from paper | Exp B | architectural; Tier 1 can commit ≤ 5 axes by construction | same |
| **P10 co-variation invariance** per-axis perceptual consistency across corpus slices | `≤ 0.2` of axis range | not measured | Exp D (Stage 2) | unknown; sensitive to whether c_hand's perceptual effect is slice-stable | trained on four-term loss with no explicit P10 term — could fail |

**Summary verdict on Tier 1's feasibility:**

- **Passes trivially:** P15 (deterministic by construction), D1/D2 (inherits v3).
- **Plausibly passes, needs measurement:** P4a global+local, P4 Fisher. The plausibility comes from the fact that hand-authored axes explicitly span the space we care about, and OLS on `(qwen, c_hand)` pairs minimizes L2 error over that span — rank structure usually survives.
- **Unknown, needs measurement:** P*, Y1, Y2 (architectural), P10.

Tier 1 is **not yet a feasible candidate** under the framework's protocol — three hard constraints (P*, Y1, P10) have no measurement yet. But nothing in Tier 1 obviously fails any constraint. Running Exps E, B, D converts "needs measurement" to "passes" or "fails" and resolves Tier 1's feasibility verdict.

**Summary verdict on Tier 2+'s feasibility:**

Functionally identical to Tier 1 on the constraint filter — same pipeline shape, same drift channel, same determinism story at inference. The gradient-based training changes *where* P4a/P4 are optimized against but doesn't change the shape of the scored `f`. The interesting difference between Tier 1 and Tier 2+ is not feasibility but Pareto position (§5).

**One concerning asymmetry:** Tier 2+ trains explicitly on P4a (SoftRankSpearman) and P4 (CE) but **does not train on P10 or Y1**. If Tier 2+'s biased gradients (DRaFT-1) produce trick-the-readout artifacts (Part 2 §10 mode 9), P10 failure is the likely signature — per-slice confounds that the training signal didn't see. Tier 1 avoids this by never being in a gradient fight with the readout; Tier 1's P10 failure mode would instead be "hand-authored axis doesn't stay perceptually coherent across corpus slices," which is a different (and more diagnosable) failure.

---

## 5. Pareto comparison — §2.7.3 axes

Assuming both Tier 1 and Tier 2+ pass the feasibility filter (TBD until Exps E, B, D run), the framework compares them along four axes. No axis dominates; the output is a Pareto frontier.

| Axis | Measurement | Tier 1 expected | Tier 2+ expected | Framework-side notes |
|---|---|---|---|---|
| **1. Identity separability** (P4 Fisher absolute) | `F(f) / F(qwen)`, higher better, upper-bounded at 1.0 | Strong — hand-authored axes are designed to maximize between-class variance within the axis basis | Potentially stronger — end-to-end optimization can find higher-Fisher conditioning than a human would write, if it exists in Flux's reachable output space | Tier 2+'s theoretical ceiling is higher; whether it reaches meaningfully above Tier 1 is empirical. Tier 1 saturates at human articulation ceiling. |
| **2. Drift controllability** (D1 magnitude / variance / factor-specificity / reversibility) | Multiple sub-measurements | **Tied.** Both inherit v3 drift unchanged. | **Tied.** | Drift channel is orthogonal to identity-channel training. |
| **3. Editorial contribution** (E2 ablation Δ with/without editorial) | `P*_with − P*_without` | **Tied.** Both candidates use the same editorial archetypes (the 10-archetype layer from v3, possibly extended in Stage 3). | **Tied.** | Part 2 inherits the editorial layer from v3; neither tier modifies it. |
| **4. Cost and operational fitness** | GPU-hours, wall-clock per face, license, cache-friendliness | **Dominant.** Training: hours-to-a-day (Stage 4). Inference: identical to v3 + one linear projection. | Dominated. Training: 3–5 days single GPU for DRaFT-1, 1–2 weeks for Adjoint Matching / Flow-GRPO. Inference: identical. | Framework axis 4 is a hard gate on project timeline. |

**Provisional Pareto verdict** (provisional because feasibility is unresolved):

- If Tier 1 passes feasibility with any margin on Axis 1, **Tier 1 dominates or ties Tier 2+ on all four axes** unless Tier 2+ shows a *measured* Axis 1 improvement large enough to justify the Axis 4 cost. The framework stops at the Pareto frontier; the final pick is an engineering decision.
- If Tier 1 fails feasibility on any hard constraint and Tier 2+ passes, Tier 2+ wins by elimination. This is the case that would justify escalation — it's the falsifier for Claim 8.1.

**The decision reduces to a single empirical question:** does Tier 2+'s end-to-end training produce a measurable P4 Fisher improvement over Tier 1 that is *worth* the week of engineering time? This is what §2.7.3's Pareto axes let us say out loud.

---

## 6. Vocabulary overlaps — where Part 2 re-derives the framework

Four places where Part 2 introduced vocabulary the framework already has. These are not errors — the re-derivation is defensible pedagogically — but opportunities to make Part 2 reference the framework rather than parallel it.

### 6.1 `CE(D, D̂)` vs `P*`

- **Part 2 §3 / §7.2:** frames `CE(D, D̂)` as "the training target" and "the supervised end-to-end term."
- **Framework §2.7.1:** names `P*` (user-task accuracy vs text baseline) as the ultimate criterion; everything else is explicitly a proxy for `P*`.
- **The unification:** `CE(D, D̂)` is a *lower-stack proxy for P\**. Framework §2.3 says the proxies (P4a, P4, D1) are computable cheap pre-filters; `P*` is the expensive ground truth. `CE(D, D̂)` fits naturally as a proxy one level deeper — a differentiable surrogate for the full detective-recovery score, which is itself the corpus-scale version of `P*`.
- **What Part 2 should say:** "`CE(D, D̂)` is our differentiable proxy for the detective-corpus version of the framework's `P*`. When Tier 1 minimizes readout CE it is climbing a surrogate gradient; `P*` itself is validated by Exp D."

### 6.2 Channel B vs §2.5 drift rubric

- **Part 2 §5.2 Claim 5.2:** introduces "Channel B — off-manifold drift" as a distinct data pathway with its own mechanism (denoising-strength) and separate from pattern preservation.
- **Framework §2.5:** already has the drift channel as a first-class subsystem with rubric D0–D6. `D1 = factor-wise inconsistency`, measured via ArcFace anchor-distance at `r = +0.914` on v3. §2.5.2 adds joint Y-channel constraints for when multiple drift axes compose.
- **The unification:** Part 2's Channel B is **exactly the framework's drift channel**. Claim 5.2's content ("it's a real distinct channel") is the framework §2.5 preamble. Assumption 5.3 (two channels engineerable independently) is §2.5.2 Y1 orthogonality applied across channels — already named `τ_orth ≤ 0.3`.
- **What Part 2 should say:** "the framework's §2.5 drift-channel rubric (D0–D6) is the specification for Channel B; §5.2 here describes the *perceptual mechanism* (Kätsyri/Diel factor-mismatch) that the framework's D1 measures as factor-wise inconsistency."

### 6.3 Four-term loss vs Pareto axes

- **Part 2 §7.2:** four-term training loss (CE + InfoNCE + VICReg + SoftRankSpearman) with per-term `λ` weights.
- **Framework §2.7.3:** four Pareto-comparison axes (identity separability, drift controllability, editorial contribution, cost).
- **The unification:** these are orthogonal concerns. The loss terms are *training signal*; the Pareto axes are *candidate comparison*. Same candidate, trained with any combination of loss terms, is scored on the same Pareto axes. Part 2 occasionally blurs this — reading §7 can create the impression that the four loss terms are also the four evaluation criteria, which they aren't.
- **What Part 2 should say:** "loss terms drive Adam; framework Pareto axes score the resulting candidate. The two are not in correspondence — loss weights are hyperparameters of training, Pareto axes are properties of the trained mechanism."

### 6.4 Tiers vs §2.7.4 decision structure

- **Part 2 §8:** three tiers of training procedure (hand-written+regression, DRaFT-1, Adjoint Matching / Flow-GRPO).
- **Framework §2.7.4:** two-stage decision structure (feasibility filter → Pareto comparison), agnostic to training procedure.
- **The unification:** each tier produces a candidate `f` of the same functional type. All three tiers feed candidates into the same §2.7.4 protocol. The tier choice is a production-cost decision within the "which candidates do we generate" step, not a protocol-level distinction.
- **What Part 2 should say:** "tiers are training procedures that generate candidates; the framework's §2.7.4 decides which candidate to ship. Candidates from different tiers are comparable because they share functional type."

---

## 7. What the framework does *not* score

Honesty requires naming the limits of this exercise.

- **Choice of optimizer** (Adam vs SGD vs policy gradient, DRaFT-1 vs Adjoint Matching). The framework scores `f`, not how `f` was trained. Tiers 5/6 in Part 2's staged plan are implementation choices *within* the Tier 2+ bucket; the framework collapses them into one candidate type.
- **Biased-gradient dynamics** (three walls, Wall 2 in the north-star memo[^northstar]). The framework assumes `f` is what it is at inference time; whether `f` was reached via a biased or exact gradient doesn't enter the scoring. The three-walls analysis lives upstream — it tells us what we *can* claim about the trained pipeline, not what the pipeline *is*.
- **Identifiability non-uniqueness** (Wall 3). The framework allows multiple viable candidates on the Pareto frontier and explicitly outputs a frontier, not a single winner. This accommodates Wall 3 gracefully — "Adam lands at one of many equally-good points" becomes "one Pareto-frontier candidate among several," which is the expected output of the protocol.
- **Loss hyperparameters** (`λ_task`, `λ_align`, etc.). Framework §2.7.1 doesn't score internal training choices; it scores the inference behavior of `f`. Tuning `λ` is part of generating a candidate, not evaluating one.

---

## 8. What Part 2 surfaces that the framework does *not*

The exchange goes both ways. Part 2 made claims the framework doesn't currently have structure for:

- **The three walls** (north-star memo). Framework §2.7.5 principle #4 ("theoretical eliminators must survive empirical sanity check") is related but narrower — it's about not eliminating candidates on unverified theoretical grounds. The three walls are about not *selling* a candidate as optimal-in-an-absolute-sense. These could be added to §2.7.5 as a fifth principle ("candidates are Pareto-optimal under *declared* criteria, not in an absolute sense").
- **Reward hacking** (Part 2 §10 mode 9). Framework §2.7.5 doesn't name reward hacking. It's arguably absorbed by P10 co-variation invariance (a reward-hacked P fails P10 by construction — the axis is legible to the trained readout but not stable across corpus slices), but making this connection explicit would sharpen §2.5's rubric.
- **Detective-experiment confounds** (Part 2 §10 mode 11). Framework `§5 Exp D` specifies experimenter-detective isolation and decoy axes, but not the full list of confounds Part 2 names. These could be folded into the Exp D spec.

These are additions Part 2 earns the right to propose *back* into the framework. They would be framework v0.12+ material.

---

## 9. Open measurements and run order

Pulling together the "needs measurement" cells from §4, the minimum measurement set to make Tier 1's feasibility verdict computable:

| Order | Experiment | Measures | Cost | Resolves |
|---|---|---|---|---|
| 1 | **Exp E** — metrics panel on v3 baseline | P4a global + local, P4 Fisher | ~1 day | whether v3 + hand-authored conditioning plausibly clears Tier 1 hard constraints |
| 2 | **Exp B** — FluxSpace orthogonality on candidate axes | Y1, Y2 (architectural) | ~1–2 days per axis batch | whether multi-axis composition is structurally sound |
| 3 | **Exp D** (Stage 2 of Part 2) — detective corpus + recovery measurement | P*, P10 | ~1 week per corpus | the ultimate criterion + co-variation invariance |
| 4 | *(skip Tier 2+ until Exps 1–3 indicate Tier 1 fails)* | — | — | — |

If Exps E, B, D run and all four yield Tier-1-passing numbers, **Tier 1 is a viable candidate and Claim 8.1 is supported**. Tier 2+ is then optional (Pareto-check for incremental improvement, not a prerequisite for shipping). If any of Exps E, B, D fails on Tier 1, the specific failure mode determines which of Tier 2+'s capabilities is needed — and the escalation is *targeted*, not a default.

This run order is already the staged plan in Part 2 §9, re-expressed in the framework's vocabulary. Executing it is Phase 1 of the project's research roadmap.

---

## 10. Bottom line

- **Tier 1 is not yet a scored candidate** — three hard constraints (`P*`, Y1, P10) have no measurement. But nothing about Tier 1's design obviously fails any constraint.
- **Exps E, B, D fill the gap.** They are already both scheduled (framework §5) and staged (Part 2 §9). Executing them produces Tier 1's feasibility verdict.
- **Tier 1 vs Tier 2+ Pareto comparison** reduces to one measurable question: does end-to-end training meaningfully improve P4 Fisher over hand-authored+regression, *enough* to justify the week of engineering cost? Every other Pareto axis is tied (drift, editorial) or Tier 1-dominant (cost).
- **Vocabulary unification is low-effort and high-value.** Part 2's blog and the framework agree on substance but use parallel vocabularies; collapsing them to the framework's terms (`P*`, §2.5 drift rubric, §2.7.3 axes, §2.7.4 protocol) would make Part 2 read as an *instance of* the protocol rather than a second opinion *about* it.
- **Limits apply.** The framework scores `f`, not the training process that produced `f`. The three walls, reward-hacking-prone optimizer dynamics, and identifiability non-uniqueness are upstream concerns that live in the north-star memo and Part 2 but not in §2.7.

The framework is the right tool for the job; Part 2 is a candidate waiting for the tool to be used on it.

---

## References

[^blog]: `docs/blog/2026-04-16-part-2-math-of-pattern-preservation.md`
[^framework]: `v2/framework/math-framework.md` (v0.11, 2026-04-16)
[^northstar]: `docs/research/2026-04-16-north-star-adam-optimality.md`
