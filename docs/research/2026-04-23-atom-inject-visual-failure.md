---
status: live
topic: demographic-pc-pipeline
---

# Atom-injection visual validation — scoped failure on one artifact (2026-04-23)

## TL;DR (scoped, 2026-04-23 evening)

`FluxSpaceDirectionInject` fed the **causal-fit** ridge directions
(`directions_resid_causal.npz`) on **atom 16 only**, on **one base
× one seed**, produces either no visible change (scales ≤ 10 after
norm-scaling) or pure noise collapse (scales ≥ 100) with nothing in
between except painterly-texture artifacts at 25–50. That is the
scope of the falsification.

**What this does NOT falsify**:

- The **Phase-3-proper artefact `directions_k11.npz`** (11 atoms,
  tensor norm ~0.02 per atom) — never tested. Scales needed there
  are ~10³ higher than what I tried on the causal file; none of
  those scales were applied to `_k11`.
- `directions_resid.npz` (non-causal, norm ~0.86) — never tested.
- Atoms other than 16 in any file.
- **Prompt-pair FluxSpace edits** (`FluxSpaceEditPair`). Those are
  a different mechanism and continue to fire reliably in the
  overnight corpus.

What *is* falsified:

- **Any claim that `directions_resid_causal.npz` atom_16 can be used as
  an edit primitive** in its current form, at any of the scales we
  probed (0 through 10⁴). The norm-47 direction is too large at low
  scales to matter (dominated by attention noise), and past
  collapse long before it becomes visible.

## Timeline

- **2026-04-22** — Phase-3-proper built ridge-fit atom→δ directions with CV R² 0.82–0.97 across 11 live atoms. Predictive claim validated.
- **2026-04-22** — Same doc flagged *predictive vs constructive asymmetry* and recommended **Option C — Immediate validation pilot**: render atom 5 at a small scale sweep, verify the direction moves its atom as predicted.
- **2026-04-22 → 2026-04-23** — Option C skipped. Moved directly to effect-matrix work on prompt-pair edits using atoms as a *measurement* basis, which worked.
- **2026-04-23** — First actual visual test of atom injection (today). Four rounds of smoke on atom 16, each 4 renders:
  - Scales 0.5, 1.0, 2.0 → no visible change
  - Scales 5, 10 → no visible change
  - Scales 25, 50 → painterly artifact accumulation, no smile geometry
  - Scales 100, 1000, 5000, 10000 → collapse to mosaic / noise
- **Conclusion** — the ridge-fit direction is not a valid edit primitive in any scale regime.

## Why it fails (working theory)

The Phase-3-proper doc spelled out the distributional asymmetry:

> A high CV R² says "given an observed delta_mix, we can predict atom k."
> It does not yet say "injecting our fitted direction moves atom k by the
> predicted amount." That second claim only holds to the extent that
> observed delta_mix vectors live on the same affine subspace as our
> fitted direction.

Concretely: the ridge regresses *atom k coefficient* (scalar) onto the *observed (24, 3072)-tensor delta_mix*. The weight matrix the regressor returns is an **inverse map** — the linear projection from delta_mix onto atom k. Using that projection as an *edit direction* is not the same operation. It would only coincide if the forward and reverse maps were aligned, which would require delta_mix variance to live in a subspace where the predictive direction and the constructive direction align — not a property the fit is designed to produce.

This is a vocabulary-vs-geometry confusion. The ridge gives us a readout projection. It does not give us a steering vector.

## Implications

- **Atom-direct edit via FluxSpaceDirectionInject is dead.** Don't spend more compute on scales, regularization tweaks, or fitting on richer corpora — the problem is structural, not numerical.
- **Atoms remain valid as a measurement basis.** The effect-matrix work (`docs/research/2026-04-23-effect-matrix-v0.md`) uses atoms as blendshape-space readouts; that's correct usage and stands.
- **Prompt-pair edits remain the only working FluxSpace edit mechanism.** FluxSpaceEditPair (pair-averaged attention between two prompts) fires reliably at mix_b=0.5 and scales 0–2.
- **The vocabulary bridge must be rebuilt on the correct side.** Instead of "atom → ridge → δ_attn injected via FluxSpaceDirectionInject," invert the workflow: "prompt pair → render → measure atom trajectory → curate a library keyed by atom-purity." See companion plan doc [2026-04-23-promptpair-iterate-plan.md](2026-04-23-promptpair-iterate-plan.md).

## Supersession

This doc supersedes the "usable edit directions" framing of
`2026-04-22-phase3-proper-directions.md`. The CV-R² numbers and caching
pipeline stand; the "FluxSpace-usable edit directions" claim does not.
