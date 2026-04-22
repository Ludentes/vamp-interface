---
status: live
topic: manifold-geometry
summary: Synthesis of FluxSpace findings: manifold linearity with directional curvature, max_env saturation envelope predictor outperforms Mahalanobis, geometry clusters by base antiparallel magnitude.
---

# FluxSpace Day: What We Now Know

**Date:** 2026-04-21
**Scope:** Synthesis across seven investigations completed today — port, single-prompt confound, pair-averaging fix, collapse predictor, cross-demographic confirmation, smile-axis second-axis test, intensity-dial sweep. Contextualised against the FluxSpace paper (Dalva, Venkatesh, Yanardag 2024; `docs/papers/fluxspace-2412.09611.pdf`).

## The manifold picture

The per-(block, step) joint-attention output `l_θ(x, c, t)` is the space everything happens in. Three empirical properties dominate what we've built:

- **It's approximately linear at the attention level in a limited neighbourhood.** Adding `s · δ` to the base attention output for s ∈ roughly [−0.6, +1.4] produces a clean, prompt-aligned edit on a held-out image. This is the entire operational premise of FluxSpace and it survives contact with our cross-demographic and cross-axis tests.
- **The manifold has directional curvature.** The safe window around s=0 is **asymmetric** on the glasses axis (roughly [−0.6, +1.4]) and **symmetric** on the smile axis. Positive-side collapse looks like colour-speckle noise overlay; negative-side collapse on glasses looks like feature wash-out. The curvature direction is axis-dependent.
- **Base prompt geometry determines how much room `s` has.** Window widths vary 10× across six demographics on the same axis; the widest-window bases have the *most antiparallel* δ to the base attention (`cos(δ, attn_base) ≈ −0.93`). Antiparallel means δ cancels the base, producing a valley in `|attn_base + s · δ|` that extends the safe range before saturation.

Concretely: the manifold is a locally linear neighbourhood around each base's attention profile, bounded by a saturation envelope that rotates with the base-δ geometry and pinches tighter where the two align.

## What metrics actually work

We tried four predictors of collapse-on-scale from a single s=1.0 measurement:

| predictor | form | in-sample | cross-seed | status |
|---|---|---|---|---|
| `max_env = max_d |attn_base_d + s · δ_d|` | piecewise linear | 92.86% | 82.14% | **winner** |
| `zmax = max_d |(steered_d − μ_d) / σ_d|` | piecewise linear, normalised | 78.57% | 75–79% | second |
| `d_mahal = ‖steered − μ‖_Σ⁻¹` | quadratic | 78.57% | 71–79% | third |
| `fro_ratio = ‖steered‖_F / ‖base‖_F` | quadratic | 71.43% | 64–71% | worst |

`max_env` wins. Two further findings changed how we use it:

- **Absolute threshold does not generalise across bases.** `max_env(0)` varies ~30% across demographics, so a threshold fit on one base predicts zero safe range for others. The **ratio** `max_env(s) / max_env(0)` is base-prompt-invariant; `T_ratio=1.275` gives 93% accuracy on glasses.
- **Scalar thresholds hide positive/negative asymmetry.** A two-sided `(T⁺, T⁻)` fit would recover glasses' asymmetric window and match smile's symmetric one. Not yet implemented.

A fifth predictor came from geometry:

- **`cos|p95|(δ, attn_base)`** — the 95th percentile of `|cos(δ, attn_base)|` over (block, step). Computed from the same single measurement pass. Splits bases into narrow (`cos|p95| ≈ 0.85`) and wide (`cos|p95| ≈ 0.93`) window clusters. **The per-base ranking is identical across glasses and smile axes** — this is the strongest cross-axis invariant we have found. It implies one geometry pass per base can predict narrow-vs-wide-window behaviour across *every* axis for that base.

## What we know about Mahalanobis

Mahalanobis was the theoretically motivated metric (shape-aware, on-manifold distance) and it under-performed on every dimension we tested today and yesterday:

- **As a collapse predictor** (above table): 78.57% vs `max_env` 92.86%. The diagonal Σ built from prompt-to-prompt spread of `attn_base.mean_d` across the 10-prompt calibration corpus is *tighter* than the actual on-manifold tolerance around any single prompt. Reason: prompt-to-prompt variance captures variation across *different identities*, which is much broader than the deformation a single identity tolerates before collapse. Σ under-counts within-identity slack and over-counts between-identity slack.
- **As a direction-quality metric** (Stage 4.5): the adversarial review flagged that Σ was built from the same 1785 conditionings used to fit the ridge direction, so the 14.3× Mahalanobis ratio vs prompt-pair was descriptive, not predictive. The memory file explicitly records this as "don't sell as prediction".

Practical conclusion: **don't use Mahalanobis anywhere load-bearing.** It's seductive because it looks principled and ties to multivariate stats, but the Σ we actually have access to doesn't match the decision boundary we care about. `max_env` — a dumb max-of-absolute — wins because the decision boundary is a saturation envelope, not a statistical outlier distance.

## Why our initial direction-extraction failed

Before FluxSpace, we built directions two ways:

1. **Ridge regression on conditioning embeddings** (Stage 4). Fit a linear map from a 16-d demographic factor vector to the T5+CLIP conditioning tensor, extract per-axis direction vectors, add them to the conditioning.
2. **Prompt-pair contrast at the conditioning layer** (Stage 4.5). Difference between two full-prompt conditionings ("adult Latin American woman" − "elderly Latin American woman").

Both worked geometrically — the directions existed, the math was fine — and both produced **no visible glasses** when applied as conditioning offsets. The image changed (age, pose, hair, lighting) but the target attribute didn't land.

The FluxSpace result inverts this: the same prompts in the attention-output space produce a clean edit at scale 0.5–1.0. Reason: **T5 and CLIP embeddings were not trained to respond linearly to arbitrary direction shifts.** The embedding model is a text encoder; its geometry is organised around producing legal text tokens, not around composing visual attributes. The attention-output space in the DiT blocks is downstream of that encoding, closer to where visual features get assembled, and is where the paper's empirical linearity hypothesis actually holds.

This was the single highest-leverage lesson of the project: **don't edit upstream of where linearity lives.** Direction extraction in conditioning space is the wrong layer.

## Why FluxSpace works (and what our variant adds)

The paper's mechanism:

- At every attention layer, extract `l'_θ(x, c_e, t) = l_θ(x, c_e, t) − proj_{l_θ(x,φ,t)} l_θ(x, c_e, t)` — the edit condition's output with the null-condition's output projected out.
- Add `λ_fine · l'_θ` to the base attention output. Optionally mask by a threshold `τ_m` on the edit condition's attention map.
- Linearity holds because joint-attention blocks "gradually add image content to the latent representation" (paper §1), and the edit attention output minus its projection onto the prior is an approximately content-pure direction at each step.

Our `FluxSpaceEditPair` extends this in a way **not in the paper**:

- Replace the single edit condition `c_e` with **two edit conditions** A and B chosen so their *confounds* point in opposite directions. For glasses on a Latin American woman: A = `"A person wearing thick-rimmed eyeglasses."` (drifts younger), B = the full base-prompt splice (drifts older). Cache attention per block per step for each, average at `mix_b=0.5`.
- The common signal (glasses) reinforces; the confounds (age ±) cancel.

This is the paired-delta trick applied at the attention layer instead of the embedding layer — the same idea that failed upstream works downstream, for the same reason FluxSpace itself works downstream.

## Novel findings vs the paper

| finding | in paper? | ours |
|---|---|---|
| orthogonal projection `l'_θ = l_θ(c_e) − proj_φ l_θ(c_e)` | yes | we use paper's math inside the pair node |
| single edit condition + null | yes | **we use two edit conditions averaged** |
| `λ_fine` is the edit scale | yes | same |
| attention-map masking `τ_m` | yes | not adopted |
| pooled-embedding coarse edit `λ_coarse` | yes | not adopted |
| collapse predictor from a single measurement | **no** | `max_env` ratio T=1.275 |
| geometry clustering of bases by `cos|p95|` | **no** | narrow/wide window ranking, axis-invariant |
| window asymmetry (axis-dependent) | **no** (they just pick one λ) | quantified on glasses vs smile |
| B-ladder (prompt-graduated edit conditions) | **no** (they vary λ_fine only) | works as a true intensity dial |
| start_percent category dependence | partially (§5.5 timestep ablation) | we split by object-add vs geometric-deformation |

Three of ours are genuinely additive to the paper: the **pair-averaging variant**, the **preflight collapse/geometry predictors**, and the **B-ladder as intensity control**. The paper's ablations on `t`-start (Fig. 6d) match qualitatively — earlier start means more drastic edit — but they don't observe that the same `t`-start can help one axis and destroy another.

Worth also noting: the paper's Table 1 reports CLIP-T for smile at 0.35 and for eyeglasses at 0.32, and we got CLIP P(smile) up to 0.94 vs P(glasses) only to 0.74 on the same detector (ViT-B-32). Attribute-dependent detector ceiling is a real gotcha for anyone using CLIP as the only scoring signal.

## Best approach to building directions

Based on today:

1. **Work in attention-output space, not embedding space.** Settled.
2. **Use pair averaging with opposing confounds when you know the confound direction.** mix_b=0.5 default. Extend to N-way when useful.
3. **One measurement per (base, axis) at s=1.0 is the unit of work.** From that one pass you can:
   - predict the safe `s` window (via `max_env` ratio)
   - classify the base's window width (via `cos|p95|`)
   - decide whether this axis is object-addition or geometric-deformation (by comparing the δ-vs-attn_base structure — hypothesis, not yet tested analytically)
4. **Vary B-prompt content for intensity, not s magnitude.** A 4-rung ladder at fixed s≈1.0 gives cleaner control than a 5-value scale sweep.
5. **Set `start_percent` per axis category**, not per base.
   - Object-addition (glasses): later start (0.30–0.40) preserves identity on narrow bases, edit still fires.
   - Geometric-deformation (smile): keep default 0.15. Delaying kills the edit because the deformation must lock in during structural phase.
6. **Don't trust any one CLIP score.** Different attributes have different detector ceilings on downsampled portraits. Use curve *shape* (peak location, peak sharpness), not absolute values.

## How much can we automate?

Substantial.

**Already automatable (code exists):**
- Calibration corpus (`--calibrate`): 10 base renders, per-prompt-domain, reusable.
- Measurement (`--crossdemo-measure --axis X`): one render per base.
- Verification scales (`--crossdemo-verify --axis X`): reads predicted edges, renders straddling sweep.
- Collage + primary metrics (drift, CLIP, cos|p95|) per axis.
- Ratio-based edge prediction.

**Partially automatable with small additions:**
- **B-ladder generation.** Prompt an LLM with the axis name and base demographic to emit 4–5 splice prompts grading attribute intensity. Validated by running through the intensity sweep.
- **`start_percent` selection.** Once we confirm the "object-addition vs geometric-deformation" categorisation holds on a third axis (beard? hair colour?), the rule becomes: `sp=0.30 if object_add else 0.15`. Category itself could be auto-detected by looking at the δ's block distribution (object edits are concentrated in mid-late blocks; geometric edits pervade all blocks — testable hypothesis, not tested).
- **Two-sided threshold fit.** Given any labelled sweep, fit `(T⁺, T⁻)` per axis. 10 minutes of analysis code.
- **Axis-category classifier.** Run the sweep on two new axes (beard, age), compare cos|p95| ranking across all four axes. If ranking remains identical, the category classifier may reduce to a single scalar lookup.

**Not automatable (needs human or stronger model):**
- **Intensity-band labelling** (Mona Lisa vs warm vs Joker). Current CLIP P(smile) can rank but not name bands. Py-feat AU12 (lip-corner-puller) + AU25 (lips-part) gets us physiology-based labels without human.
- **"Is this edit clean or uncanny?"** The collapse predictor catches the saturation cliff, not the soft failure modes (wrong teeth count, asymmetric mouth, skin-tone shift). Needs dedicated detectors per failure mode or human review.
- **Selecting an axis worth editing in the first place.** The whole pipeline assumes a target attribute with a text prompt; there is no procedure for discovering directions that don't have an obvious linguistic anchor.

## What we actually have after today

A pipeline that, for a new attribute:

1. Accepts a target axis name and 6 base demographics.
2. Auto-generates A and a per-base B.
3. Runs one calibration (if not cached) + one measurement pass per base.
4. Predicts safe `s` window and per-base window-width category from the measurements alone.
5. Renders a straddling verification sweep.
6. Builds a 4-rung B-ladder, renders it at s ∈ {0.4, 1.0, 1.4} for visual QA.
7. Outputs collages, predictions.json, primary metrics, an intensity grid.

No human tuning between step 1 and 7. The only per-axis decision left is `start_percent`, and the object-add-vs-geometric-deformation rule likely closes even that.

## Open follow-ups

- Third axis (beard or explicit age) to confirm `cos|p95|` ranking invariance.
- Two-sided (T⁺, T⁻) per-axis fit.
- Test whether δ block-distribution distinguishes object-addition from geometric-deformation.
- AU-based (py-feat / OpenFace) intensity labelling replacing CLIP.
- Real-image editing via RF-Inversion (paper §5.3) — we're currently synth-only.
- N-way averaging: 3+ edit prompts with orthogonal confounds, does the cancellation tighten?

## Artefacts

- Paper: `docs/papers/fluxspace-2412.09611.pdf`
- Component docs: `docs/research/2026-04-21-fluxspace-{collapse-prediction,crossdemo-confirmation,smile-axis,smile-intensity-dial}.md`
- Blog post: `docs/blog/2026-04-21-fluxspace-end-to-end.md`
- Pipeline: `src/demographic_pc/fluxspace_metrics.py`, `fluxspace_intensity_sweep.py`, `fluxspace_intensity_collage.py`, `fluxspace_primary_metrics.py`
- Custom node: `~/w/ComfyUI/custom_nodes/demographic_pc_fluxspace/`
