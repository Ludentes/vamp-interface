# Adversarial review — Stage 4.5 blog + comparison report

**Date:** 2026-04-20
**Reviewer:** general-purpose subagent (independent read; no context from the session that produced the artifacts).
**Subjects:**
- [Blog: Demographic-PC Extraction, End to End](../blog/2026-04-20-demographic-pc-sanity-check.md)
- [Report: Stage 4.5 — Ours vs FluxSpace on a single axis](2026-04-20-demographic-pc-stage4_5-comparison.md)

Findings are preserved verbatim so the weaknesses are auditable. Follow-up work to address these is tracked separately.

## BLOCKERS

**B1. The Mahalanobis "prediction" is not actually a prediction; it's a tautology given the direction construction.**
Blog "The Mahalanobis prediction" / report "Direction-only" section. Ours is built by ridge regression on the *same* 1785 conditionings whose Σ is then used to compute Mahalanobis. Ridge + representer theorem forces `w` into the span of those points with heavy shrinkage onto high-variance axes — so `wᵀΣ⁻¹w` is low *by construction*. FluxSpace is built from two prompt embeddings that are almost certainly outside that hull (Euclidean norm 14.75 vs Ours' 0.141 already reveals this is not a like-for-like geometric object). Claiming this ratio "predicts image behaviour on 340 renders without looking at any of them" is not a Bayesian-valid prediction — it's the same distributional fact showing up twice. Needs rewording: the Mahalanobis ratio *describes* the same manifold-alignment property that the image metrics *test*; it does not independently predict them.

**B2. Scale mismatch undercuts half the "FS loses" metrics.**
Ours is scaled to "45 years/λ" via ridge target. FS is scaled to "2 · ||target−base||" — a geometric, non-semantic quantity. The post concedes "it's a gain choice" for local slope, but then still reports **gender-flip, race-flip, and ArcFace identity-drift at |λ|=3 as if they were comparable**. They aren't: Ours at λ=3 asks for +135 years (a classifier-ceiling-capped age push); FS at λ=3 asks for *3× (target−base)*, a pure off-manifold extrapolation with no semantic anchor. The "matched target-slope" rescaling is only applied to AD. The flip-rate and drift-at-λ=3 numbers should either be rematched or explicitly flagged as unrematched.

**B3. FluxSpace coarse is a strawman of the published method.**
FS-fine (with timestep gating) wasn't tested — this is acknowledged but buried. FluxSpace's headline contribution in Dalva et al. is **block-specific + timestep-gated** injection; applying its *direction extraction* at a single pooled-CLIP + T5-mean layer without the gating is not the "coarse variant" from the paper — it's "the prompt-pair contrast baseline FluxSpace was designed to beat." This is Ours vs prompt-pair-contrast, and the "FluxSpace" label is misleading.

## CONCERNS

**C1. Saturation ratio 0.15 averages over sign changes.** Report admits this; blog keeps "0.15 → FS spends 85 % of range by |λ|=1" without the caveat. If outer slope is negative for some portraits, 0.15 isn't "saturation" — it's "reversal". Blog should inherit the caveat.

**C2. n=20, one scale, one axis, one seed trajectory.** R² 0.896 vs 0.216 with n=20 is significant, but per-class AD_race numbers (e.g., FS max on "East Asian" = 4.07e-3 vs "Black" = 1.03e-5 — three orders of magnitude) are dominated by who the 20 portraits happen to be. The "4× more entangled" headline is driven by a single class; report it as a *max over 7 classes with n=20* with the range.

**C3. "Plateau" framing vs continuing drift.** Ours ArcFace drift 0.14 → 0.32 → 0.44 across λ ∈ {1, 2, 3} is a smooth ramp, not a plateau. The cliff/plateau dichotomy is rhetorically tidy but Ours doesn't have a clear plateau edge — just a shallower slope.

**C4. "FS's cliff may be a feature for vamp-interface" reads as a salvage move.** The vamp-interface use-case has never been framed as "needs a direction that deterministically crosses race to look wrong" — race-flipping as a fraud proxy is exactly the kind of confound the perception curriculum was set up to avoid. Rehabilitating FS for vamp-interface by calling race-flip a feature deserves a sharper caveat or deletion.

**C5. Cover image verification.** Top row (Ours): plausible aging, identity preserved — matches. Bottom row (FS): λ=−3 solid teal (matches), λ=−1 visibly East-Asian where baseline reads white (matches "flipped race"), λ=+3 noise static (matches), λ=+2 zombie/lesion (matches). Visual claims check out.

**C6. cos(target,base) is computed but not reported.** Two similarly-worded prompts will have cos≈0.9+, meaning FS's direction is a small tangent to a giant common-mode vector. Normalising by ||target−base|| keeps that tangent tiny in *direction* but still yields a huge absolute Euclidean norm of 14.75. Worth disclosing — it's the mechanical explanation for the 27× Mahalanobis.

**C7. Ledoit-Wolf α=0.005 on a 4864-d problem with N=1785.** Sample covariance is singular; shrinkage must be doing more work than 0.005 suggests for Σ⁻¹ to exist. Worth reporting smallest eigenvalue of the shrunk Σ and a sensitivity check.

## NITS

- Alt text says "ethnicity"; body uses "race". Pick one.
- "Relabel λ_ours and it's closed" — confusing phrasing. Rewrite.
- "Age has the strongest per-head regression … so it's the fair test." It's the *easiest* test, not the fair one. Fair would be race.
- Report: "picks up low-variance off-principal loadings (tokenizer idiosyncrasies, prompt-specific stylistic co-loadings)" — asserted without evidence. One cheap check: project FS direction onto the 121-prompt basis and report residual norm.
- Blog title "Ridge Beats Prompt-Pair on Five of Six Metrics" — body argues at length that FS isn't beaten, just differently-shaped. Title contradicts thesis.

## Summary

The headline 1.92 vs 27.49 Mahalanobis number is real and interesting but oversold as a "prediction" (B1). The "FS loses" framing papers over a scale-matching issue at |λ|=3 (B2) and compares against a stripped FS variant the original paper would disown (B3). The "two kinds of cliff" reframe is defensible but borderline-apologetic. With B1–B3 addressed honestly — rephrase Mahalanobis as descriptive, rematch or flag the extreme-λ metrics, rename "FluxSpace-coarse" to "prompt-pair-contrast baseline" — this becomes a straightforwardly strong post. As written, the prose outruns the evidence in three load-bearing places.
