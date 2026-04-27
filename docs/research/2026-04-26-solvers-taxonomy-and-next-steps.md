---
status: live
topic: demographic-pc-pipeline
---

# Three solvers, only one built; v4 → v5 resume notes

This doc disambiguates three different "solvers" we've been discussing
across the slider thread, captures the open ideas in spec form so they
do not get re-derived from scratch next session, and lists the explicit
resume-here steps for after v4 ends.

## Solver taxonomy

We have been using "solver" loosely. Three distinct things, only one
of which exists today.

### A. FluxSpace composition solver (built, in production)

This is the existing infrastructure that produced
`models/blendshape_nmf/sample_index.parquet` and the ~7652 measured
renders, plus the cached attention pkls archived to USB. Inputs:
prompt-pair edits expressed as FluxSpace δs; pair libraries with
`attn_a` / `attn_b` caches; an anchor render with `attn_base`.
Operation: `FluxSpaceEditPair` and `FluxSpaceEditPairMulti` patch
attention online during a render to produce an image at a chosen
`(axis, scale, mix_b)` configuration. Composition (`compose_iterative`)
stacks multiple edits to cancel known confounds (e.g. smile+age).

Status: **built, validated** on glasses, smile, demographic axes;
cross-prompt-invariant geometry confirmed (cos|p95| ranking matches
across glasses↔smile, project memory `fluxspace_smile_axis`).

Use: this is the rendering substrate. It *generated the corpus* the
two new solvers below would consume. Not a slider tool, not a
checkpoint tool — a measurement-time edit composer.

### B. Forward checkpoint-mixing solver (idea, not built)

Inputs: a trained slider's checkpoint ladder (e.g. v4's 15 checkpoints),
a measurement vector per checkpoint per prompt
(`m_k = [intended, identity, hair, clothing, ...]`), an intended-axis
target intensity `T`.

Operation: solve constrained least squares for stack strengths `s_k`
such that `Σ s_k·Δm_k[intended] = T` and `Σ s_k·Δm_k[drift]` is
minimum-norm in a weighted metric (identity weighted heaviest, lighting
lightest). Implemented at inference by stacking multiple LoRA loads at
fractional strengths (mathematically exact for the linear-LoRA regime).

Inputs come from the **slider quality measurement procedure**
(`docs/research/2026-04-26-slider-quality-measurement.md`). The
measurement parquet *is* the input matrix.

Status: **idea, not built, not specced beyond this paragraph.** Build
only after a single checkpoint fails the slider-quality battery, since
if one checkpoint already passes, the solver is unnecessary complexity.

### C. Inverse training-pair selection solver (idea, not built)

Inputs: the measured render corpus (`sample_index.parquet` ∪ classifier
scores ∪ blendshape ∪ ArcFace), a target intended axis (e.g. glasses), a
list of confound axes the entanglement observation flagged
(clothing, beard, hair, lighting, age — bundle-specific).

Operation: query the corpus for pairs `(i, j)` such that
`Δintended(i,j)` is large and `‖Δconfound(i,j)‖_w` is small, where the
confound weights `w` come from the stereotype-bundle observation. Output
~200-1000 image pairs that **break the cluster correlation**. Those
pairs become the training data for a Path B image-pair Concept Slider
training run.

Why this matters: Path A (prompt-pair) and naive Path B (prompt-curated
pairs) both learn the bundle because the supervision signal *is* the
inter-cluster shift. Solver C breaks the bundle by curating
intra-cluster pairs that exist in the corpus but are statistically rare
in unfiltered prompts.

Status: **idea, not built, not specced beyond this paragraph.** This is
the upstream-of-Path-B prerequisite. Should be specced first because
without it, Path B is not meaningfully different from Path A on a
stereotype-bundled axis like glasses.

### Why three is not too many

A consumes user-authored edits and produces measured renders. C consumes
A's measured renders and produces training pairs for a slider trainer.
B consumes a slider trainer's checkpoints and produces a stacked
inference-time configuration. They sit at three different layers
(measurement-time, training-time-input, inference-time-output) and use
the same measurement matrix as currency. Naming them prevents
"the solver" from meaning whichever one is in working memory.

## Path C kill-gates (from prior turn, captured here for non-loss)

Path C from the measurement-grounded plan
(`docs/research/2026-04-26-measurement-grounded-slider-plan.md`) has a
falsification track record on related problems. Before any LoRA training,
run these gates in order, each with a fail-fast criterion. Total
time-to-kill if dead: ~1 hour, before any meaningful compute.

- **Gate 1 — φ regression quality.** Train φ to predict
  `siglip_glasses_margin` from latent. Hold out 20%. **Required:** R² ≥
  0.85 on holdout. If φ cannot predict the metric on a held-out latent,
  either the metric is not latent-determined or the feature input is
  insufficient. *Cost: ~30 min. Kill: numeric.*
- **Gate 2 — gradient structure.** Compute `∇_z φ(z)` for ~50 latents,
  decode `x_0(z) − ε·∇_z φ`, eyeball. If gradient is high-frequency
  noise, the latent-space gradient will not survive the VAE; Path C is
  dead. If glasses-shaped structure appears in the eye region, proceed.
  *Cost: ~10 min. Kill: visual.*
- **Gate 3 — gradient ascent without LoRA.** Take a no-glasses render's
  latent, K=20 steps of `z ← z + η·∇_z φ(z)`, decode, measure glasses
  score. **Required:** glasses score increases monotonically AND visible
  glasses appear on the decoded image. If score rises but pixels do
  not, surrogate is exploiting a metric blind spot — Path C is dead.
  *Cost: ~20 min. Kill: do real glasses appear?*
- **Gate 4 — short LoRA pilot.** Only if 1-3 pass. 200-step LoRA
  training against φ on 50 latents. Evaluate via slider-quality battery.
  *Cost: ~1 GPU-hour. Kill: pass criteria from quality doc.*

Solver C is independent of these gates; it can proceed in parallel
because Path B does not depend on φ.

## Resume-here checklist

After this conversation compacts, the next session resumes here:

1. **v4 wraps at step 1500.** Confirm process exited cleanly. Best
   checkpoints to evaluate first: 300, 400, 600 (males found glasses
   early, females at 600).
2. **Run the slider quality measurement procedure** on the most
   promising v4 checkpoint per the doc
   (`2026-04-26-slider-quality-measurement.md`). The script
   `measure_slider.py` does not exist yet and needs to be written.
   Build minimal: single checkpoint, render grid, write parquet, generate
   collage.
3. **Decide v5 config** based on v4 eval. Default plan: cosine LR
   schedule, η=2.5, EMA off, surgical scope (`add_k_proj`+`add_v_proj`
   only), save_every=50, 800 steps.
4. **Decide v6 config** in parallel. Default plan: same data + scope as
   v5, switch optimizer to Lion (lr ≈ 3e-5) or Prodigy (auto-LR). Run
   sequentially after v5 (one GPU). v6 runs *regardless* of v5 outcome
   per user instruction — we want the optimizer comparison data
   independent of v5's pass/fail.
5. **While v5 trains:** discuss and spec solver C (inverse training-pair
   selection). Solver B (forward checkpoint-mixing) only needed if v5
   single checkpoint fails the quality battery.
6. **Path C gates** can be run while v5 or v6 is training (different
   GPU process — actually no, single GPU, so queue after v6).

## What this doc deliberately does NOT do

- Implement any solver. All three new pieces are spec only.
- Pick between Path B and Path C. Both stay open; solver C unblocks B,
  Gates 1-3 unblock C. Run both in parallel as research threads.
- Tune slider quality thresholds. The numbers in the quality doc are
  starting points and will be adjusted after v4 + v5 evals reveal what
  passes-but-looks-bad and what fails-but-looks-fine.
- Build the reverse-index schema. Premature; one slider's eval parquet
  exists by the end of step 2 above; second slider's parquet exists by
  the end of v5 eval. Schema design happens after we have ≥2 to
  compare.

## Stereotype-bundle observation (load-bearing for solver C)

Captured here so it survives compaction even if the chat journal does
not. v4 step 600 +1.5 column for the black female prompt showed not just
glasses but a bundled mode shift: from bare-shouldered, neutral, simple
hair, even lighting (m=0) to afro, hoop earrings, black t-shirt, harder
studio lighting, slight smile, plus tortoise-frame glasses (m=+1.5). The
−1.5 column for east_asian_m showed parallel structure: shaven,
muscled, no shirt, casual lighting (m=0 →) bearded, polo, studio
lighting (m=+1.5).

Interpretation: the LoRA learned a *cluster shift* from "casual /
unadorned" mode to "professional / studio portrait" mode. Eyewear is
one feature of the destination cluster, not the LoRA's actual axis. The
bundle is upstream of the LoRA — it lives in the text encoder
(T5-XXL primarily, CLIP-pool secondarily) and the image distribution
Flux trained against. The supervision signal `(pos − neg)` traces the
inter-cluster mean; no trainer hyperparameter can subtract a confound
that lives in the supervision itself.

Implication: prompt-pair Concept Sliders (Path A) is *structurally*
weak on stereotype-bundled axes — not a tuning problem. Solver C is
the cleanest path to break the bundle (curate intra-cluster pairs the
training never sees in unfiltered prompts). Path C also breaks it but
by replacing supervision entirely.

This bundle structure is **diagnostic, not specific to glasses.** Smile
likely safe (within-portrait modulation, doesn't move clusters). Age
risky (associated with wedding/yearbook stereotypes). Sus level
unknown — likely diffuse, may not have strong cluster correlates.
First test of solver C and Path C on the next axis tells us if our
reasoning about clusters is right.
