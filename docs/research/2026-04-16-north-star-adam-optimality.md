---
status: archived
topic: archived-threads
summary: Distinguishes reachable from unreachable optimality claims for face pipelines. Adam-optimality for a specified loss is feasible; optimality for "best representation" or "best game" requires optimizing human perception, which is not differentiable.
---

# Is there an Adam-optimal face representation? A North-Star analysis

**Date:** 2026-04-16
**Status:** Draft position statement. Not an experimental result — a framing that clarifies which North Stars are reachable and which are category errors.
**Audience:** internal — same as the framework and Part 2 blog.

---

## The question

Phrased two ways, both live in the project:

1. **Data-first:** *"Here is the data. Here is an Adam-optimal way of representing it with faces, where optimal is defined in the gradient-descent sense."*
2. **Game-first:** *"Here are the patterns the detective game plants. Here are the Adam-optimal faces for conveying them."*

Both phrasings invite the same kind of promise: a single mathematical object — a trained pipeline — that is *provably optimal* in the same sense a regression line is optimal under least squares.

This memo argues the promise is partly keepable and partly not. The keepable part is worth naming carefully so the project knows which milestones are real.

---

## Short answer

- **Adam-optimal for a specified loss `L`:** reachable. This is a well-posed engineering milestone.
- **Adam-optimal for "best face representation of the data" or "best faces for the game":** not reachable as pure math, because the evaluative target has a human in it and is not a pure function.

Everything below unpacks why, what we can still commit to, and what it means for the project plan.

---

## What "Adam-optimal" requires

For `θ* = argmin_θ L(θ)` to be meaningful, three things must hold:

1. **A loss `L(θ)` that captures what we actually want.**
2. **A differentiable path from `θ` to `L`.**
3. **A search space where gradient descent's convergence is meaningful.**

Each has a wall specific to our pipeline.

---

## The three walls

### Wall 1 — the perception wall

Our true target is: *a human reader recovers the planted patterns from the face*. The reader is a black box. Human perception is not differentiable, not observable per-sample in real time, and inconsistent across individuals and sessions.

We can approximate with **surrogates**:

- Face encoders (FaRL, DreamSim, DINOv2) whose embeddings correlate with some perceptual criteria.
- A trained classifier `face → pattern` whose accuracy stands in for "can be read."
- Preference-learning data from a bank of human ratings, à la RLHF.

Each surrogate is itself a function. Adam optimises the surrogate, not the target. **Goodhart:** given freedom, Adam will find pipeline configurations that score well on the surrogate and poorly on the actual human. The narrower the surrogate, the easier it is to exploit.

Mitigations exist (ensemble surrogates, periodic human recalibration, held-out human evaluations), but the wall cannot be eliminated. Any claim of "optimal faces" is a claim about optimality against *the chosen surrogate*, not against the ground truth the project ultimately cares about.

### Wall 2 — the Flux gradient wall

Even when `L` is differentiable end-to-end, backpropagating through the Flux sampler is approximate:

- **DRaFT-1 / DRaFT-LV** backprops through one (or a few) of the 50 denoising steps. The gradient is a biased estimate — the bias is small in published SDXL/SD results, uncharacterised for our frozen-middle architecture.
- **Adjoint Matching** gives exact ODE-adjoint gradients. Constant memory, ~2× compute, higher variance — and still requires the loss to be expressible on the generated image.
- **Policy-gradient methods** (Flow-GRPO, DanceGRPO) side-step differentiability entirely by treating rollouts as samples and reward as a black box, at the cost of sample efficiency.

Every option gives Adam a *biased or stochastic* gradient of the population loss. The convergence point is a stationary point of the **estimator** — which is not the same as the true stationary point of the true loss. In practice this usually works; in principle, "Adam-optimal" is a statement about the estimator, not the truth.

### Wall 3 — the identifiability wall

Given a corpus and a target property, **multiple trained pipelines produce the same downstream behaviour** and Adam will land at one of them semi-arbitrarily:

- Our projection `P: qwen → conditioning` is a ~60M-parameter linear map. Any invertible reparameterisation of the conditioning space that Flux is invariant to (under appropriate projection into CLIP/T5 subspaces) is an equally good `P`.
- Contrastive-learning identifiability (Daunhawer et al. ICLR 2023) recovers *a shared content block* between paired modalities, not a canonical one. On planted-axis data it coincides with the axes; on organic data it coincides with whatever the two modalities share the most of, which may be topic or formatting rather than the property we care about.
- Adam's specific convergence point depends on initialisation, batch order, learning-rate schedule, and the precise form of the auxiliary losses. Two equally well-trained `P`s can differ meaningfully in which axes they emphasise.

**"The Adam-optimum"** is therefore a singular-the-article claim the math does not support. **"An Adam-stationary point of this specific loss under this specific setup"** is the honest form.

---

## What we CAN hit: Adam-stationary under a declared `L`

This is where the North Star is genuinely reachable.

Write a loss:

```
L(P, R) = α · CE(p̂, p)                       — pattern recovery head (needs labels)
        + β · FaRL_align(Ψ(x), t(p))         — face embedding near pattern-anchor embedding
        + γ · SoftSpearman(d_qwen, d_face)   — rank preservation of pairwise distances
        + δ · VICReg(Ψ(x))                   — variance / covariance regularisation
        + ε · ‖P Σ_e Pᵀ − Σ_c‖              — distribution matching on the conditioning side
```

where

- `P` is the projection (trainable, differentiable).
- `R` is the readout head `face → pattern` (trainable, small, differentiable).
- `Ψ` is a frozen face encoder or ensemble.
- `Flux` is frozen.
- `(α, β, γ, δ, ε)` are declared hyperparameters.

Then **Adam-stationary** means: gradient-descent on `(P, R)` converges (to within tolerance) to parameters where ∂L/∂(P, R) ≈ 0. This is achievable and *that* is a real engineering milestone.

**What needs gradients through Flux:**
- `β` term (loss defined on generated image) → yes, needs DRaFT-1 or Adjoint Matching.
- Every other term is a function of the *conditioning* (`c = P(e)`) or of `Ψ(x)` composed with a rendered image, but can be fit without backprop through Flux if we pre-render and cache targets. `α` and `γ` in particular can be satisfied via offline regression if the target conditioning is known.

**What the milestone says:**

> "For loss `L` with weights `(α, β, γ, δ, ε) = (…)`, Adam converges to `(P*, R*)` achieving `X%` pattern recovery on the held-out detective corpus and `Y` on `N` human subjects."

Defensible. Reviewable. Not overclaiming.

**What the milestone does not say:**

> ~~"`P*` is the optimal projection for face-based data mining."~~

That would require the optimum to exist in an absolute sense. The three walls prevent it.

---

## For the detective game specifically, the goalposts tighten

The game has two properties that shrink (though don't close) Wall 1:

1. **Patterns are discrete, planted, and known.** We control the ground truth. `CE(p̂, p)` is well-defined per-sample without surveying humans.
2. **Success is measurable as classification accuracy.** Both by a classifier (differentiable proxy) and by human subjects (ground truth). We can train against the proxy and evaluate against the ground truth.

This reduces "does the face convey the pattern?" to two concrete questions:

- *Classifier-readable:* does a trained `face → pattern` classifier recover the planted axis? — fully in our pipeline, fully differentiable.
- *Human-readable:* do the planted axes transfer to humans tested in the detective protocol? — observable offline, not differentiable.

When classifier-readable ≈ human-readable, Adam-stationary in the classifier sense *is* Adam-stationary in the human sense (up to noise). When they diverge, we have new information: our surrogate is miscalibrated; either retrain the surrogate or expand the evaluation ensemble.

**The detective game is therefore the version of the North Star with the smallest gap between "Adam-optimal under L" and "actually good"** — which is why it's the right place to pin milestones.

The data-mining phrasing (arbitrary real corpora, no planted ground truth) has a much wider gap and a correspondingly softer North Star: the best we can claim there is "Adam-stationary under a structure-preservation loss whose structure matches the data's dominant shared content" — useful, but not a singular optimum.

---

## Two milestones worth declaring

### Descriptive milestone (closed cycle, no training)

> *"With frozen components `{Qwen, P_handwritten, Flux, Ψ}` and a regression-fit readout `R`, end-to-end pattern recovery on the detective corpus is `X%`."*

Requires no Adam-through-Flux. No pipeline gradients. Closed-cycle measurement of whether signal survives a hand-built pipeline. Anchors the rest.

Corresponds to Part 2 §8 Tier 1 (hand-written conditioning + regression on `P`) and Stage 0–4 of the staged plan.

### Prescriptive milestone (Adam-stationary for a declared `L`)

> *"Under loss `L` with weights `(α, β, γ, δ, ε) = (…)`, Adam on `(P, R)` converges to `(P*, R*)` achieving `Y%` classifier recovery and `Z` human recovery on `N` subjects."*

This is the trained-pipeline result. It is bounded in the three ways above:
- `L` is a proxy; drift between `L` and human performance is observable, not guaranteed-zero.
- The gradient used is biased (DRaFT-1) or stochastic (Flow-GRPO), so "stationary" is in the estimator sense.
- `(P*, R*)` is one of many equally-good Adam limits, not the optimum.

Both milestones are stateable and measurable. The prescriptive one is what gets called "the trained model" in any writeup. The descriptive one is what you measure first, because it tells you whether you needed to train at all.

---

## What this means for the project plan

1. **Do not sell "Adam-optimal faces" as a deliverable.** Sell either (a) "frozen-pipeline end-to-end recovery of `X%`" (descriptive) or (b) "Adam-stationary under this loss" with the loss spelled out (prescriptive). Both are defensible. Neither requires the category-error claim.

2. **Commit the `L` choice early and treat it as the research question.** The choice of `(α, β, γ, δ, ε)` and the choice of surrogate `Ψ` *are* the optimisation. Once declared, Adam runs mechanically. The intellectual work is in `L`, not in the gradient descent.

3. **The detective game is where the North Star is tightest.** Make it the anchoring evaluation for any prescriptive milestone. Real-data domain work is softer and should report bounds, not point estimates of "optimality."

4. **Full-pipeline gradients are only load-bearing for the `β` term.** If we choose an `L` with `β = 0` (or with `β` small enough that Tier-1 regression satisfies it), we never need DRaFT-1 or Adjoint Matching. The walls in §7 get thinner. If we need `β > 0` for perceptual alignment we can't fit via regression, then we need Flux-gradient methods and we inherit Wall 2 in full.

5. **Report three numbers for every prescriptive result:**
   - Classifier recovery under the learned pipeline (proxy outcome).
   - Human recovery on detective subjects (ground truth).
   - Gap between them (surrogate calibration).
   The gap is the most important of the three for long-run project credibility.

---

## Open questions

- **Can we publish Tier 1 end-to-end numbers (descriptive milestone) before committing to an `L`?** If yes, this is the cleanest first milestone and may answer the project's biggest open questions without any training at all.

- **What's the smallest useful `L` for the detective game?** Start with `α · CE(p̂, p)` alone. Add structural terms only when recovery plateaus. The multi-term `L` in §"What we CAN hit" is the ceiling of ambition, not the starting point.

- **When do we run the human-calibration study?** Before prescriptive training would be expensive but tightens the surrogate. After would let us train against a cheaper surrogate and calibrate at evaluation time. Likely answer: run a small pilot early (n ≈ 5), full study at prescriptive milestone.

- **How do we communicate Wall 3 to the reader?** The identifiability wall is the most counterintuitive — two equally-trained pipelines can produce meaningfully different faces. In the writeup, either: (a) name it as a feature (many good trainings exist, we show one), or (b) control it via a canonical parameterisation (e.g. PCA on the conditioning to break rotational ambiguity). Decision deferred.

---

## Relation to existing framework documents

- **`v2/framework/math-framework.md` §2.6.1** names the metric toolbox used for the descriptive milestone.
- **`docs/blog/2026-04-16-part-2-math-of-pattern-preservation.md` §7** spells out a 4-term candidate `L`. This memo is the conceptual frame for why that `L` is a proxy and what "Adam-optimal under it" actually claims.
- **`docs/research/2026-04-16-full-loop-synthesis.md`** covers the four gradient-method options at the mechanics level (DRaFT-1, Adjoint Matching, Flow-GRPO, TexForce). This memo is about whether, and in what sense, those methods converge to "the optimum."

---

## Bottom line

The North Star the project should sell is not *"the Adam-optimal face representation."* It is:

> **"Adam-stationary under a declared, measurable loss `L`, evaluated against human recovery on the detective protocol, with the gap between the two reported."**

That is reachable, defensible, and falsifiable. The alternative framing is an overclaim the math does not support.
