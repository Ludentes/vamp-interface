# Perception Before Training — A Curriculum Interlude Between Parts 2 and 3

**Date:** 2026-04-20
**Series:** Sits between [Part 2: The Math of Pattern Preservation](2026-04-16-part-2-math-of-pattern-preservation.md) and whatever Part 3 turns out to be. Part 2 argued for a training pipeline that preserves patterns from text through face images to a recoverable label. This post notices the question Part 2 skipped: **can humans read the patterns out of the faces at all?** If the answer is no, the training pipeline is optimizing for a perceptual channel that doesn't exist. If the answer is yes-but-only-in-a-specific-regime, we need to know the regime.
**Audience:** same as Parts 1 and 2.
**Status:** Interlude memo, written same day the pivot happened. Revised post-adversarial-review.

---

## The pivot

Part 2 spent a lot of words on a frozen-middle supervised pipeline: text → qwen → projection → Flux → face → face-encoder → readout → recovered label. Five frozen links, two learned adapters, a four-term loss, a staged buildout, a Fisher-ratio diagnostic. All defensible; the adversarial review is in the drawer.

And yet we had never measured, end-to-end, whether a human looking at one of these faces can recover anything. We had pipeline metrics, machine metrics, framework P-properties. We had argued extensively from priors that specialized face-processing should help. We had not once put a face in front of an eye and asked it a question.

The pivot today was: **start smaller, and start with perception.** Not scam posts, not qwen, not even the idea that the faces should mean something in any ecological sense. Hand-designed D-dimensional vectors, D ∈ [1, 6], rendered through the existing Flux pipeline, plus a matched non-face comparator, plus trained human viewers, plus the actual measurement: does the face channel let people read patterns that the chart channel doesn't, and if so where does that crossover sit?

The Part 2 pipeline stays. Only the source of the D-dim vectors changes — from "qwen of a job post" to "points we placed on purpose." Everything else in Part 2 (four-term loss, Fisher diagnostic, continuity properties) remains applicable; we just stop pretending we know the output side of the pipeline works before we measure it.

> ⚠ **Claim I.1** The perceptual-readability question is logically prior to the training-pipeline optimization: a pipeline that looks healthy under machine metrics may still fail for human perception, and the two can decorrelate in ways neither ArcFace nor CLIP-derived encoders catch.
>
> **Status:** *prior*, yes. *Separable* — weaker: Part 2's Stage 0 already feeds perceptual D₇₉ into the Tier 1-vs-Tier 2+ escalation decision, which is coupling, not separation. What this claim licenses is measuring perception *before* spending training compute on losses that might be optimizing a dead proxy. The machine-vs-human decorrelation is plausible on priors (FaRL is caption-aligned; ArcFace is identity-aligned; neither was trained against human perceptual-similarity judgments for attribute axes) but has not been measured in this pipeline. Measuring it is the curriculum.

## Why a curriculum, not one experiment

Two reasons carried the design — not claimed to be exhaustive; these are the two that survived triage.

**Training the observer.** Human face-processing machinery does not spontaneously deploy on unfamiliar synthetic faces. Show a naive viewer a grid of 256 generated portraits and the perceptual channel you wanted to test is the channel that never woke up. Behavioral face-expertise paradigms (Gauthier greebles being the canonical example) explicitly train observers before measurement; skipping the training step is a recipe for null results that look like "faces don't encode the signal" but are actually "observers haven't calibrated." Not every face-perception paradigm trains — inversion-effect localizers often use passive viewing — but for a test of *patterns-in-a-grid* specifically, with unfamiliar synthetic stimuli, the training step is load-bearing.

**Falsifying at each step.** If a D=1 single-dial discrimination task fails on faces, that's a pipeline problem (the dial isn't monotone) and no Level 5 AHA demo will paper over it. If D=1 succeeds but glance odd-one-out at D=3 fails, that's a different problem (the dials are local rather than configural) and knowing which one told you what matters for the fix. A single grand experiment at Level 5 that fails tells you nothing about why. A ladder of experiments with staged pass/fail criteria tells you exactly where the claim stops holding.

Six levels. Level 0 is a deliberately trivial single-dial 2-AFC that faces are *expected to lose* to a plain bar chart; its purpose is to verify the axis is monotone in face-space and to de-risk the experimental infrastructure before the science levels. Level 5 is the 256-face glance-grid with embedded anomalies — the AHA demo that either establishes an existence proof or doesn't. Levels in between escalate in D, number of visible items, and (critically for the FFA hypothesis) brevity of exposure.

The full six-level design — parameter tables, pass criteria, training protocol, analysis pipeline, preregistration templates — is in [`docs/research/2026-04-20-ffa-progression-curriculum.md`](../research/2026-04-20-ffa-progression-curriculum.md)[^curriculum]. This post is about the design discipline the curriculum imports, why that discipline is load-bearing, and what is being bet.

## What the curriculum imports from psychophysics

The first draft of the levels had a shape but not the methodology. It specified paradigms and difficulty but not exposure times, not inversion controls, not a matched non-face comparator as a first-class encoding, not psychometric-function fitting, not bootstrap CIs, not a sample-size justification, not preregistration. A reader from the face-perception community would have asked where Yin (1969) was, where Rossion (2013) was, and whether we understood that "faces encode X better than charts" without an inversion comparison is not a face-specific claim at all — it's a claim that faces happen to work while charts don't, which is different and weaker.

The corrections the curriculum imports, each load-bearing:

**Inversion as a licensing control.** The inversion effect — that upright faces are recognized or discriminated disproportionately better than inverted ones, compared to other object classes — is the canonical behavioral evidence that face processing uses specialized machinery (Yin 1969; reviewed for the composite paradigm specifically in Rossion 2013 and Richler & Gauthier 2014). The standard move: if an effect on upright faces is mediated by that specialized machinery, it should shrink under inversion. If it doesn't — if upright and inverted perform the same — the effect is generic image-level discrimination that happens to occur on face-shaped stimuli. The curriculum requires an inverted-face block at every FFA-bet level (2 through 5), and reports the face-vs-comparator contrast and the face-vs-inverted contrast as **separate CIs**. A positive first CI with a null second CI means "faces work" without licensing any face-specific mechanism.

> ⚠ **Claim I.2** Whether the inversion-licensing structure transfers to Flux-generated synthetic portraits at short SOAs with backward masks is itself an open empirical question — not a background assumption the curriculum can rely on.
>
> The literature establishing inversion as an FFA-licensing control was developed on photographs of real faces and tightly-controlled line-drawings. Diffusion-generated portraits have known artifact signatures (symmetry violations, texture priors, hand/ear anomalies, a "synthetic smoothness" that some viewers notice subliminally) that may either suppress or spuriously trigger face-specialized responses. The inversion effect is also known to attenuate for caricatured, atypical, or distribution-shifted faces in the published literature. **Status:** this is *itself* a curriculum-level question. If Level 2 produces a face-vs-comparator advantage that does *not* shrink under inversion, the clean reading is not "our methodology failed" — it's "the Flux portraits in this pipeline are not recruiting face-specialized processing, even though they look photorealistic." That is a useful and reportable result. The curriculum is prepared to land that result rather than paper over it.

**A matched non-face comparator.** Every level has a non-face encoding — bars, scatter, D-glyphs — carrying the *same* D-dimensional information, matched on luminance / contrast / spatial-frequency envelope, pre-session audited by a simple classifier so we know the two encodings carry equivalent machine-level discriminability for the task. Without this, "faces work" is an absolute claim, not a comparative one, and the project's motivating hypothesis — face channel beats chart channel in some regime — is untestable. Richler & Gauthier (2014) and Rossion (2013) both emphasize comparator discipline for holistic-processing claims specifically, and the adjacent composite-face paradigm (Young 1987; formalized in Richler & Gauthier) is arguably the more modern holistic-processing diagnostic than inversion alone. The curriculum uses inversion because it's simpler to implement at scale; if Levels 2-3 results are ambiguous, a composite-face variant is the natural follow-up.

**Psychometric-function fits with bootstrap CIs.** Per level per subject per encoding, fit a psychometric function (`psignifit 4` or Palamedes) with lapse rate bounded and a fixed chance level by paradigm. Extract threshold Δ₇₉ with 95% bootstrap CIs (Wichmann & Hill 2001, part II). Pool across subjects with mixed-effects logistic rather than by concatenating raw trials — per-subject first, pool last. Accuracy alone conflates sensitivity with bias (Macmillan & Creelman 2005); d′ belongs next to every accuracy number.

**Small-N as a design class, not a sample-size cutoff.** Smith & Little (2018) defends small-N designs where the individual participant is the unit of replication, with dense per-subject data and strong within-subject measurement — as in classical vision-science psychophysics. That framing licenses N = 3–5 *when* each subject contributes many trials, within-subject effect size is large relative to between-subject variance, per-subject analyses are reported, and the effect replicates across the small N. Citing the paper is not the same as meeting its conditions; the curriculum reports per-subject curves alongside any pooled contrast, and explicitly marks "does the effect replicate across all N" as a checkable outcome rather than an assumed one.

**Preregistration as a floor.** Every reported level gets an AsPredicted entry before data collection. Report-honest floor is the "21-word solution" from Simmons, Nelson & Simonsohn (2012) — "We report how we determined our sample size, all data exclusions, all manipulations, and all measures." That note sits on top of the 2011 *Psychological Science* paper "False-Positive Psychology" (Simmons et al.), which supplied the evidence that undisclosed flexibility in reporting inflates false-positive rates enough to matter. A six-level curriculum where later-level parameters depend on earlier-level measurements is not trivially straightforward to preregister — adaptive designs require different structures (sequential analysis, alpha-spending if we're careful) — and this is acknowledged as an open design question for Levels 3-5.

**Demographic-axis confound handling.** A treated-as-engineering-hypothesis, not an established result: Flux's latent conditioning space is almost certainly *not* uniform in human perceptual distance. Directions aligned with apparent age, gender, or ethnicity are perceived as much larger than orthogonal directions of equal L2 distance. If a difficulty direction for a trial happens to align with a demographic axis, the task collapses to "find the young face" and the threshold Δ is an artifact. The curriculum's mitigation — sample 500 faces, run a demographic classifier, project the top classifier-aligned components out of the sampling distribution for Δ — is a plausible engineering move, not a proven one. The assumption that demographic contributions to perceptual distance are approximately linearly decomposable is itself the kind of assumption the curriculum elsewhere says not to make. So: orthogonalization is a Level-0 engineering task, and its efficacy is itself one of the first things Level 1 (continuity) has a chance to reveal.

> ⚠ **Claim I.3** Face-specific claims about generative-model stimuli require, at minimum, a face-vs-non-face-comparator CI *and* a face-vs-inverted-face CI, reported separately. Collapsing them licenses "faces work well" but not "faces work well *because* face-specialized processing is doing something charts don't."
>
> **Status:** the two-CI decomposition is straightforward application of standard licensing logic (Yin 1969; Rossion 2013; Richler & Gauthier 2014). What the decomposition licenses in this project is narrower than "FFA involvement" — as Claim I.2 names, inversion-licensing transfer to synthetic portraits is itself unmeasured. Inversion is *necessary, not sufficient* for holistic-processing claims. The stronger composite-face paradigm (Rossion 2013; Richler & Gauthier 2014) is a natural follow-up if Levels 2-3 produce the cleaner face-vs-inverted pattern but we want tighter evidence before publishing mechanism claims.

## What we need to be true, and aren't sure is

One assumption the curriculum leans on and cannot resolve from within:

> 🔍 **Assumption I.4** The Flux conditioning space exposes enough effectively-independent attribute dials to instantiate D = 4-6 meaningfully. **Status:** Part 2's Stage 0 includes measuring this. If Flux gives us fewer than ~4 disentangled dials, Levels 4-5 collapse to effectively Level 3 — the D>3 claims become untestable in this pipeline without generator changes. That would reshape the plan but not invalidate the earlier levels, which are the ones actually under dispute for the main hypothesis.

## Cross-level synthesis: what the curriculum is actually measuring

Per level, the primary measurement is a threshold Δ₇₉ per encoding with a CI. Cross-level, two curves:

**The face-advantage curve.** X-axis: level index, ordered roughly by `task complexity × item count × 1/exposure`. Y-axis: `Δ₇₉(comparator) − Δ₇₉(face)`, in σ units, per subject. Zero-crossing is the crossover between chart-territory (bars win) and face-territory (faces win). If the curve rises monotonically into face-territory as we move toward high-n, short-exposure, holistic-judgment conditions, the project's motivating hypothesis is confirmed at the existence-proof level.

**The FFA-licensing curve.** X-axis: level. Y-axis: `Δ₇₉(inverted) − Δ₇₉(upright)`, per subject, Levels 2+. Positive margin with CI excluding 0 licenses face-specialized-processing interpretation; flat margin forces retreat to "face-encoding-specific" language without mechanism claims. These two curves decompose the result in exactly the way the published literature asks a reader to see it.

If both curves behave as hoped, the training-pipeline work Part 2 proposed has an identified target: a perceptual channel that exists, is face-specialized, and has a measured operating regime. If the face-advantage curve rises but the FFA-licensing curve stays flat, the project still has a product — faces as a convenient encoding in a specific regime — but the uncanny-valley / holistic-processing / FFA framing should be walked back to "dense pictorial encoding." That's a downgrade but not a failure; the scam-hunting application would still work on its own terms.

## What this interlude doesn't claim

- **Not that the curriculum will pass.** Level 2 might fail and force us to revise what Flux faces encode. That is a real outcome this design is supposed to be able to deliver.
- **Not that N = 3-5 is enough for all claims.** It's enough for existence / mechanism claims under the Smith & Little per-participant-replication framing. "Untrained naive users benefit" is a population claim requiring larger N and is explicitly out of scope here.
- **Not that passing Level 5 makes the scam-hunting product work.** Level 5 is on hand-designed geometry. Real job-post data has upstream uncertainty the curriculum deliberately excludes. Ecological transfer is a separate study.
- **Not that the six levels are exhaustive.** They're a design of discrete plausible steps, not a MECE decomposition of the whole perception question. Adjacent paradigms — composite-face variant (Rossion 2013), visual search with set-size manipulation, change-detection — are all candidate Level-2.5 or Level-3.5 additions depending on what the reported levels surface.
- **Not that face inversion licenses FFA cleanly in this setting.** Claim I.2 above: transfer of inversion-licensing to synthetic portraits is itself unmeasured; a positive face-vs-inverted CI is *necessary, not sufficient* for FFA-specific mechanism claims.

## What changed in our plan of record

- **Part 2's Stage 0** (the baseline measurement on the existing Flux v3 pipeline, added after the Fisher-ratio decision) now has a perceptual companion: Level 0 of this curriculum on the same pipeline. The Tier 1 vs Tier 2+ escalation decision takes two inputs, not one — machine-metric Fisher *and* human-perceptual Δ₇₉ at Level 2.
- **Demographic-PC identification** becomes a prerequisite engineering task, not an optional one. Cheap (no humans), a few hours of work. Do it before any curriculum-level data collection.
- **The scam-hunting application is explicitly downstream of the curriculum.** Level 5 passing is an existence proof; the ecological study on real job-post data comes after, and can fail independently even if the curriculum passes.

## What this interlude is about

That the perceptual readability of the faces is the hinge question the project had been routing around. Measuring it requires psychophysics, not product-UX. The curriculum is how we measure it.

Nothing in this post moves the training pipeline Part 2 described. Everything in this post is a prerequisite before the training pipeline will turn out to have been worth building. Part 3, whenever it arrives, can lean on whichever of the two curves turned out to exist — the broad one (faces beat matched charts for glance pattern-recovery, mechanism unspecified) or the narrow one (face-specialized processing is what powers the effect). That choice will be made on data rather than priors.

---

**Next concrete step:** demographic-PC extraction on the Flux v3 pipeline. Sample 500 faces, run a FairFace / DEX / MiVOLO classifier for age + apparent gender + apparent ethnicity, regress each attribute onto the conditioning vector, extract the top principal components per attribute. Output: a list of "don't sample Δ in these directions." No humans involved. After that, Level 0 as the first runnable experiment.

---

## Assumption inventory

| # | Label | What we assume | Status |
|---|---|---|---|
| I.1 | ⚠ | Perceptual-readability is logically prior to training-pipeline optimization; machine and human metrics can decorrelate | prior yes; fully separable no; decorrelation unmeasured in this pipeline |
| I.2 | ⚠ | Inversion-licensing transfers to Flux synthetic portraits at short SOAs | itself a curriculum-level open question |
| I.3 | ⚠ | Face-specific claims require two separate CIs: face-vs-comparator and face-vs-inverted | standard licensing logic; narrower than full FFA license (necessary, not sufficient) |
| I.4 | 🔍 | Flux conditioning exposes ≥ ~4 effectively-independent attribute dials | being measured in Part 2 Stage 0; if false, Levels 4-5 collapse to Level 3 |

---

[^curriculum]: `docs/research/2026-04-20-ffa-progression-curriculum.md` — the full six-level design, parameter tables per level, training protocol, analysis pipeline, preregistration template, and the open design questions. The curriculum's design discipline is also captured as a reusable skill at `~/.claude/skills/designing-perception-experiments/` so it survives beyond this project; the skill's validation is that it will be used and revised, not that it closed gaps in one TDD run.

[^kingdomprins]: Kingdom, F. A. A., & Prins, N. (2016). *Psychophysics: A Practical Introduction* (2nd ed.). Academic Press. The single best modern all-in-one reference covering paradigms, adaptive methods, SDT, psychometric fitting, and bootstrap CIs. (A 3rd edition of Macmillan & Creelman's SDT text — Hautus, Macmillan & Creelman, 2021, Routledge — is newer, but Kingdom & Prins remains the broader reference.)

[^wichmannhill]: Wichmann, F. A., & Hill, N. J. (2001). The psychometric function: I. Fitting, sampling, and goodness of fit; II. Bootstrap-based confidence intervals and sampling. *Perception & Psychophysics*, 63(8), 1293–1313 and 1314–1329.

[^macmillancreelman]: Macmillan, N. A., & Creelman, C. D. (2005). *Detection Theory: A User's Guide* (2nd ed.). Lawrence Erlbaum Associates.

[^yin]: Yin, R. K. (1969). Looking at upside-down faces. *Journal of Experimental Psychology*, 81(1), 141–145.

[^rossion]: Rossion, B. (2013). The composite face illusion: A whole window into our understanding of holistic face perception. *Visual Cognition*, 21(2), 139–253.

[^richlergauthier]: Richler, J. J., & Gauthier, I. (2014). A meta-analysis and review of holistic face processing. *Psychological Bulletin*, 140(5), 1281–1302.

[^smithlittle]: Smith, P. L., & Little, D. R. (2018). Small is beautiful: In defense of the small-N design. *Psychonomic Bulletin & Review*, 25(6), 2083–2101. **Note on scope:** the paper defends small-N as a *design class* with the individual participant as replication unit and dense per-subject measurement; it does not prescribe N=3-5 as a numeric range. The curriculum's use of N=3-5 is consistent with standard vision-science practice under the paper's framing, but citing Smith & Little is not the same as meeting its conditions (within-subject replicability across all N, strong measurement, constrained theory). Per-subject curves are reported for exactly this reason.

[^simmons2012]: Simmons, J. P., Nelson, L. D., & Simonsohn, U. (2012). A 21 word solution. *SSRN 2160588*. Sits on top of their earlier 2011 *Psychological Science* paper "False-Positive Psychology" (Simmons, Nelson, & Simonsohn, 2011, *Psych. Sci.* 22(11), 1359–1366), which supplied the evidence that undisclosed reporting flexibility inflates false-positive rates enough to matter. Cite both together if the blog is read as methodology grounding.
