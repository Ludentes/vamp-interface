# vamp-interface Mathematical Framework v0.6

**Date:** 2026-04-14 (second external critique pass — formal decision protocol)
**Status:** Draft v0.6. Inputs: user spec, theory constraints doc, blind-alleys doc, paper-findings log, 2026 portrait-animation scan, 2026-04-07 final findings (measured baseline on 543 jobs), 2026-04-09 v2 system description, 2026-04-12 algebra research, 2026-04-12 StyleGAN-vs-diffusion decision review, 2026-04-12 FLAME technical overview, **2026-04-14 math-foundations synthesis (Perplexity + Tavily pro research on trustworthiness/continuity, Gromov-Wasserstein, Information Bottleneck, typical-set OOD, uncanny-as-mismatch)**. Rebuild plan and deeper-research queue deliberately NOT read — they are pre-framework and will be reconciled AFTER the framework is stable.
**Purpose:** Before touching the rebuild plan again, formalize what vamp-interface actually wants in terms of spaces, maps, metrics, and channels; name the hidden inconsistencies in the informal spec; produce an evaluation grid that lets candidate pipelines be scored rather than asserted.

This document does *not* make architectural choices. It specifies the scoring rubric. The rewrite of the rebuild plan will instantiate the rubric over candidates.

### Changelog

**v0.1 → v0.2 (same afternoon):**

- **Lipschitz demoted, distance-rank correlation promoted.** The informal "similar→close, different→far" is a *relational* (order-preserving) property, not a metric one. Spearman rank correlation (new P4a) is the acceptance criterion; Lipschitz is a supporting diagnostic. See §1.5.
- **Two-stage architecture made explicit.** Identity subsystem (on-manifold) and drift subsystem (off-manifold) are separately scored on separate rubrics. P9 removed from identity rubric entirely. See §1.7, §2.2, §2.5.
- **Neural-machinery weak-proof clause.** Having any working uncanny mechanism is weak proof the product engages face-reading perception rather than icon-recognition. §1.7.

**v0.2 → v0.3 (same evening, after reading pre-existing research):**

- **P0 broadened to include the anchor-bridge adapter class.** v1 and v2 already used this pattern. `qwen → cluster → caption → T2I` is a legal bounded-cost adapter and every text-to-image model trivially passes P0 under it. Previously excluded models (PhotoMaker, InstantID) re-enter as identity-subsystem candidates via the anchor bridge. See §2.3 and §3.1.5.
- **Regime C' named.** Top-k-gated soft interpolation in hand-curated natural-language prompt space (as opposed to vector space). This is what v2 does (softmax T=0.02, top-1≈63%, effective_N≈2.3 over 10 hand-curated archetype prompts). Distinct from Regimes A/B/C from the theory doc. See §1.5 and §3.1.6.
- **§3.1.6 added — v2/v3+Flux is the measured baseline**, not a hypothetical candidate. `r(anchor_distance, sus_level) = +0.914`, ArcFace cluster separation = 0.2179, measured on 543 jobs on 2026-04-07. This is the bar any new candidate must clear before displacement.
- **FaceNet replaced by ArcFace IR101** as the measurement tool across P4a / P4 / P6 / D1 / D2. Production tool is `minchul/cvlface_arcface_ir101_webface4m` via `src/face_distinctness.py --model arcface`. LPIPS stays for D4 region-wise leakage.
- **Drift subsystem D1 + D2 are already measured as passed.** `Cursed_LoRA_Flux` + `Eerie_horror` at sus_factor strength produces r=+0.914 drift correlation on Flux v3. The drift mechanism is built, measured, and working. Remaining work is LoRA-curve *tuning* (v8c overshoot into zombie) and optional per-flavor decomposition via Concept Sliders. §2.5 updated accordingly.
- **§3.1.2 StyleGAN3 W entry is wrong for a second reason.** Beyond the Lipschitz over-commitment walked back yesterday, FFHQ-trained StyleGAN3 has near-zero expression range (decision review 2026-04-12), which is fatal for the uncanny mechanism. Fine-tuning on AffectNet (4–8 GPU hours) is a viable variant; raw FFHQ StyleGAN3 is eliminated on product grounds.
- **§3.1.4 FLAME entry reframed.** FLAME is best understood as a **stage-1.5 readable-Y editor** (invert via SMIRK/EMOCA, edit in structurally-orthogonal parameter space, re-render), not a standalone identity candidate. The one FLAME-as-identity path that might work (`qwen → FLAME → Arc2Face-with-expression-adapter`) is now pointed at **arXiv:2510.04706, ICCVW 2025** "ID-Consistent Precise Expression Generation with Blendshape-Guided Diffusion" — the specific paper that operationalizes it.
- **Two new identity candidates added:** §3.1.8 h-space direction finding on frozen Flux (Asyrp + Haas + Self-Discovering Directions); §3.1.9 Arc2Face + blendshape-guided expression adapter (arXiv:2510.04706). Both have specific blocking open questions, listed per entry.
- **New drift entry §3.2.5 — Concept Sliders (ECCV 2024).** LoRA weight-pair semantic directions in diffusion latent space, composable (50+ tested), text-or-image-defined, preserves protected concepts, ~1–2h training per concept. Flux-native via LoRA; **paper shipped SDXL sliders**, Flux versions need our training. This was the 2026-04-12 recommended next step and was lost across three sessions of rediscovery until this pass surfaced it.
- **§1.5 Fisher-ratio claim corrected.** Fisher ratio on the identity channel is **upper-bounded by the source space**, not a floor to exceed. If qwen genuinely clusters `cleaning_legit` and `construction_legit` together (measured inter-sim 0.84 in ArcFace space), no identity candidate can fabricate a separation without destroying information. The correct goal is preservation of qwen-level Fisher structure, plus optional categorical overrides (v3 clothing differentiation) at the editorial level.
- **Same-archetype collapse is now a structural feature of the framework, not a failure mode.** §1.8 added.

**v0.3 → v0.4 (same evening, after adjacent-literature pass):**

- **§2.6 added — Math foundations.** Names the specific theorems / metrics / tools from the two research passes (Perplexity + Tavily) so that the framework has citable footing instead of hand-waving. Source: `docs/research/2026-04-14-math-foundations-synthesis.md`.
- **P4a gains concrete tool pointers.** Trustworthiness/continuity (Venna & Kaski 2001) as the optional deeper diagnostic; DreamSim (NeurIPS 2023) as an alternate perceptual backbone alongside ArcFace IR101; isotonic regression / PAV as the D2 post-hoc monotonicity-enforcement step. See §2.3, §2.5, §2.6.
- **§1.5 Fisher upper-bound anchored on Information Bottleneck.** `I(Z;Y) ≤ I(X;Y)` (Tishby & Zaslavsky 2000) is now the cited theorem for "Fisher ratio upper-bounded by source space." DIB (CVPR 2024) is the operational tool if we ever want to measure it.
- **§1.5 Lipschitz-demotion reframed via Kantorovich-Rubinstein duality.** `W1(µ,ν) = sup_{Lip(f)≤1} E_µ[f] − E_ν[f]` makes Lipschitz functions the *dual* of W1, not a replacement for it. P5 (distribution preservation) is the thing we actually care about; Lipschitz is its dual form.
- **New P5 definition anchored on Gromov-Wasserstein.** GW distance between metric-measure spaces (Mémoli 2011; SCOT single-cell analog 2020) is the coordinate-free measure for "qwen-space pushforward aligns with face-space." Remains an optional diagnostic, not an acceptance floor — we lack the tool to compute it on the full corpus cheaply.
- **D1 refined: typical-set vs. raw-likelihood distinction.** Nalisnick et al. ICLR 2019 — raw likelihood is a broken OOD score. Our current `ArcFace_anchor_distance` proxy is pragmatically fine; the principled upgrade path (energy-based OOD, typical-set distance) is documented for future revisit.
- **D4 reframed around perceptual mismatch.** Kätsyri et al. 2015 + Diel et al. 2022 meta-analysis: uncanny = mismatched cues, not uniformly degraded faces. D4's goal is not to minimize collateral leakage but to ensure leakage is *mismatched across channels* (realistic skin + wrong geometry triggers the valley; uniformly ugly does not).
- **Readability-drop decision validated by glyph literature.** Borgo et al. 2013 + Fuchs et al. 2017 systematic review: Chernoff-style face glyphs underperform simpler glyphs on most tasks. Stage-1.5 (readable Y-channels) continues to have zero tenants. No framework change, but the empty stage is now principled, not provisional.

**v0.4 → v0.5 (same evening, external critique pass):**

- **Architecture restructured from two-stage to three-channel.** Identity / editorial / drift are now peers, not {identity + drift + optional override}. The editorial channel (v3's 10 hand-curated work_type archetypes, clothing differentiation) empirically does most of the visible perceptual work on our corpus and deserves first-class status alongside the geometric identity channel. §2.2 rewritten.
- **New P\* — downstream user-task accuracy — added as top-level required property.** All other P-numbers and D-numbers are *proxies* for P\*. A 5–10 person user study comparing (text baseline / faces without drift / faces with drift) on "which job is more suspicious" and "which jobs are similar" is the *actual* acceptance criterion. §2.3 updated; §5 adds the user study as a high-leverage experiment.
- **P4a local-rank diagnostic promoted from optional to required co-equal with Spearman ρ.** Global ρ hides catastrophic short-range inversions, and our product is *visual clustering* — humans read local neighborhoods, not global order. Trustworthiness and continuity (Venna & Kaski 2001) or k-NN preservation rate at k∈{5,10} now sit alongside ρ as required measurements. §2.3 P4a updated.
- **§1.5 gains the distortion-budget allocation principle.** Source and target manifolds may not be homeomorphic; perfect rank preservation is impossible in general. The framework now explicitly says: *allocate* rank errors to intra-cluster geometry (where humans don't read) rather than to cluster boundaries (where they do). This is the principled form of §1.9's "same-archetype collapse as structural feature."
- **§1.5 also names discrete/hybrid maps as first-class citizens.** Under P4a (rank preservation) + Fisher ratio (P4), step functions and soft-interpolation-over-discrete-archetypes (Regime C') are viable candidates. The measured v3+Flux baseline is Regime C' — hybrid, not smooth. The framework no longer implicitly prefers continuity beyond what the human-perception requirements demand.
- **D1 reframed from "off-manifold reach" to "factor-wise inconsistency" across face factors (identity / expression / symmetry / anatomy / texture).** ArcFace anchor-distance remains the computable proxy, but the *definition* of drift is now grounded in factor mismatch (Kätsyri 2015 + Diel 2022 perceptual-mismatch literature, operationalizable via FLAME's β/ψ orthogonality and Arc2Face + blendshape factor separation). This is a theoretical upgrade; the measured r=+0.914 baseline remains valid because the LoRA mechanism empirically produces factor mismatch even without being designed for it. §2.5 D1 rewritten.
- **P4a gains a seed-stability sub-diagnostic.** We ship with `seed = hash(job_id)` so production `f` is deterministic. But a candidate whose rank preservation holds at seed=42 and collapses at seed=43 is brittle regardless. Measure ρ at 3–5 seeds on a small subset; report variance. Not a full distributional reframe (over-engineered for our deterministic product), just a robustness check.

**v0.5 → v0.6 (second external critique pass, "Decision Protocol" structural polish):**

- **New §2.7 Decision Protocol** — formal structure in the paper-ready form proposed by the external reviewer: (4.1 Hard Constraints, 4.2 Rejection Conditions, 4.3 Comparative Metrics, 4.4 Decision Structure, 4.5 Interpretation). Renamed to §2.7.1–§2.7.5 to fit inside the framework section. Hard constraints get labeled inequalities with named thresholds (`τ_ρ`, `τ_k`, `τ_var`, `τ_F`). Two-stage decision flow: feasibility filter → Pareto comparison. **The reviewer's proposal would have regressed `P*` from top-level to implicit, dropped editorial as first-class, and reverted D1 to raw manifold distance; v0.6 takes the form and keeps v0.5's substance.**
- **P15 — Identity Stability** promoted from "sub-diagnostic inside P4a" to a top-level required property with its own threshold. Seed-stability is structurally different from rank preservation and deserves its own cell. Added to §2.3 table.
- **§2.7.1 "Structure preservation is necessary but not sufficient"** is named as a framing principle and linked back to §1.5's distortion-budget argument. This is the one-liner a reader should leave with.
- **No rubric cells deleted.** v0.6 is presentational + one new cell (P15). No content regressions.

---

## 0. Scope and non-goals

- **In scope:** mathematical structure of the pipeline; properties we will demand, measure, or merely hope for; a consistency audit of the informal spec; a sketch application of the framework to known candidates to check the rubric behaves.
- **Out of scope (this pass):** final candidate selection, training-cost accounting, license review, product UX.
- **Audience:** future-me at the next pass, and any agent picking this up cold.

---

## 1. Consistency audit of the informal spec

The user's informal statement contains eight tensions worth naming before formalizing. Naming them is what keeps us from baking one side of a tension into the framework without realizing.

### 1.1 Betweenness vs. cluster separation

The spec wants **both** (a) "semantically-between ⟹ metrically-between" (a geodesic/convexity property) and (b) "within-cluster different enough, between-cluster very different" (a separation/ratio property).

These are not incompatible but they pull in opposite directions. Pure (a) with no (b) gives a smooth ramp with no visible cluster structure; pure (b) with no (a) gives discrete jumps with no interior. The reconciliation is a **ratio target**: pick a scale σ at which within-cluster distances live and demand between-cluster distances be ≫ σ, while still requiring the map to be Lipschitz at scales > σ. This is essentially Fisher's criterion recast for our purposes.

This tension matches last session's Regime-A-vs-B split for HyperFace. The user already accepts that inbetween-points may exist without clean semantic interpretation — they just shouldn't destroy the metric.

### 1.2 Algebra vs. manifold nonlinearity

The spec speculates that `emb("suspicious job") + emb("painting of a man") ≈ emb("painting of a man doing a suspicious job")` — classic word2vec analogy-arithmetic. This property is *not* reliably held by modern transformer embeddings. It holds locally, inconsistently, and depends on the model.

Practical consequence: algebra is an **optional** evaluation property. If a candidate embedding supports it we get a bonus mechanism (direct vector manipulation), but we cannot premise the architecture on it. Relying on it would rebuild the same over-commitment failure we just walked back.

### 1.3 Multi-space composition is multiplicatively expensive

The spec acknowledges we may need multiple spaces (Qwen → some latent → ArcFace/FLAME → pixels). Every transition `f_i : Z_i → Z_{i+1}` has its own Lipschitz constant `L_i` and its own metric validity (which metric is meaningful on `Z_{i+1}` need not be the same as on `Z_i`). The composition has:

- Lipschitz constant bounded by `∏ L_i` (in the worst case, multiplicative)
- Metric inheritance **not** guaranteed — cosine-valid on Qwen does not mean cosine-valid on ArcFace, even if both are 512-d unit-spheres
- Injectivity preserved only if every `f_i` is injective, and injectivity is brittle under composition

This means each added stage is a real cost, not free. The framework has to score "fewer stages" as a positive.

### 1.4 Disentanglement vs. face-manifold geometry

The spec treats disentangled direction vectors (StyleGAN age, brows, eyes) as a fallback crutch. In reality, disentanglement is a property that exists only in some face spaces and only approximately. StyleGAN W has roughly 10–20 usable semi-disentangled directions; Arc2Face's conditioning space has essentially none (it's an ArcFace embedding, optimized for identity verification not editability); FLAME has *explicitly* disentangled shape/expression parameters but only for a bounded set of 3DMM axes.

Framework consequence: "disentanglement is available" is a space-dependent property, not a fallback you can assume any stack has. It enters the grid as a per-candidate score.

### 1.5 The formal core: rank preservation + Fisher ratio (not Lipschitz) — and Fisher is upper-bounded by the source

**v0.3 correction (anchored on IB in v0.4):** Fisher ratio at the identity-channel output is **upper-bounded by the source space's own Fisher ratio on the same clustering**. This is a special case of the **Information Bottleneck inequality** `I(Z;Y) ≤ I(X;Y)` (Tishby & Zaslavsky 2000; generalization-gap bounds by Vera & Piantanida): no map from X to Z can produce more class-discriminative information about labels Y than X already carries. If two clusters genuinely overlap in qwen 1024-d space (measured on our corpus for `cleaning_legit` vs. `construction_legit`: ArcFace inter-sim 0.84 on generated faces, corresponding to genuinely overlapping qwen embeddings because physical-worker job postings share language), no mapping `f: qwen → face` can fabricate a distinction without injecting information not present in the source. The correct framework goal is **preservation, not maximization**. An identity candidate that achieves higher output Fisher than qwen's own cluster structure is either (a) exploiting editorial overrides (v3 hand-assigned clothing categories for cleaning vs. construction, which add no biometric information but are human-parseable) or (b) hallucinating distinctions. Both are legitimate in some contexts, neither is "better rank preservation." See §1.8 for the full treatment of same-archetype collapse.

---


Buried in the informal spec are **two** formally tractable requirements, and neither is Lipschitz:

**(i) Distance-rank preservation.** "Similar→close, different→far" and "semantically between ⟹ metrically between" are statements about the *ordering* of pairwise distances, not about absolute metric preservation. The right formalization is **Spearman rank correlation** between `{d_X(a_i, a_j)}` and `{d_P(f(a_i), f(a_j))}` over a sample of source pairs: if the ordering is preserved, similar pairs stay nearer than distant pairs, and the visualization reads correctly regardless of absolute scale. This property is scale-free and does not penalize discrete maps — a step function preserves rank at cluster granularity trivially.

**(ii) Fisher's discriminant ratio.** `(mean inter-cluster face-distance) / (mean intra-cluster face-distance) ≫ 1`, where "face-distance" is measured in whatever perceptual metric we commit to (LPIPS, ArcFace cosine, human rating). Measurable, optimizable, directly captures the "within-cluster different enough, between-cluster very different" requirement.

**Why not Lipschitz?** Lipschitz is a one-sided bound (`d(f(a), f(b)) ≤ L · d(a, b)`). It does not prevent collapse (a constant map is 0-Lipschitz), it does not preserve ordering, and it disqualifies discrete/step solutions (HyperFace Regime A has global Lipschitz constant ∞) that we already accept as viable. I was reaching for Lipschitz because "smooth" sounded formal; the formal property actually matching "similar→close, different→far" is rank correlation, not Lipschitz. Lipschitz stays in the framework as a **supporting diagnostic** — useful because papers report it natively and it composes cleanly per stage, but not an acceptance criterion.

**v0.4 reframing via Kantorovich-Rubinstein duality.** The honest statement is stronger: Lipschitz functions are the *dual* of 1-Wasserstein distance via `W1(µ,ν) = sup_{Lip(f)≤1} E_µ[f] − E_ν[f]`. So the property we actually want — distribution preservation — lives in W1 (or Gromov-Wasserstein for heterogeneous-space alignment, Mémoli 2011), and Lipschitz is the diagnostic of an individual map's contribution to the W1 objective. This rescues Lipschitz as supporting and makes P5 (distribution preservation) the thing it was a proxy for. We lack the tool to compute W1/GW cheaply on the full corpus, so P5 remains an optional diagnostic, but the math is now properly grounded. See §2.6.

**These two together are the first-class acceptance criteria for the identity channel.** Everything else is consequence, nice-to-have, or failure-mode guard.

**v0.5 — distortion budget allocation.** Source (qwen 1024-d) and target (face-pixel) manifolds have **no structural reason to be homeomorphic**, let alone isometric. Any nontrivial map between them must *distort* the source distance structure — this is not a failure mode, it is a mathematical inevitability. The framework's job is not to demand zero distortion (impossible) but to **allocate the distortion to places where it does not cost us perceptually**. The allocation principle, made explicit:

- **Preserve cluster boundaries.** Between-cluster distances must be large in face-space; confusing construction-legit with fraud-high is a product failure.
- **Sacrifice intra-cluster geometry.** Within-cluster rank inversions are acceptable and possibly desirable — humans do not discriminate two cleaning-legit postings by biometric microstructure, they bucket them as "same cluster."
- **Equivalently:** optimize for *continuity* (Venna & Kaski 2001 — don't tear qwen neighbors apart) at the expense of *trustworthiness* (don't worry if face-space creates neighbors that weren't in qwen, as long as they're intra-cluster).

This is the principled form of §1.9's "same-archetype collapse is a structural feature." Under distortion-budget allocation, a map that preserves cluster boundaries perfectly while collapsing intra-cluster structure into a point is *better* than a map that preserves everything slightly worse — the first allocates its distortion budget wisely.

**v0.5 — discrete and hybrid maps are first-class.** Under P4a (rank preservation) + P4 (Fisher ratio) + distortion-budget allocation, **step functions and soft interpolations over discrete archetypes are viable identity candidates**, not fallback options. A map `g: qwen → {anchor_1, ..., anchor_10}` that assigns each qwen vector to its nearest archetype is Lipschitz-infinity but preserves all relevant cluster structure. Our measured v3+Flux baseline is Regime C' — soft interpolation over 10 hand-curated archetypes with top-1≈63% — which is hybrid, not smooth. The framework does not implicitly prefer continuity beyond what human perception requires. HyperFace Regime A (theory doc §HyperFace), discrete anchor assignment, and prompt-space interpolation (Regime C') are all legal moves, evaluated on the same rubric as continuous maps.

### 1.6 Coded vs. uncoded channels (last session's split, generalized)

Yesterday's theory doc concluded that the map has to be decomposed into **coded channels** (Y-directions, hand-aligned to face-feature axes, readable) and **uncoded residual** (identifier-only, monotone similarity, not readable). The informal spec now extends this: the user explicitly distinguishes Ys — which "definitely need a direction, a modification solution" — from non-Y dimensions where "axis based manipulation is optional and disentanglement requirements are lax."

This is the same split generalized from 1 Y (sus_level) to `k` Ys. The framework formalizes it as: the pipeline is a sum of `k` named direction-maps plus one residual identity-map, and each has its own evaluation criteria. This is no longer a theoretical move; it is now the spine of the framework.

### 1.7 Drift is a separate subsystem, not a property of the identity candidate

The spec notes that the core perceptual mechanism is not Chernoff-style feature-reading but **human sensitivity to faces that are wrong** — unnatural eyes, missing mouth, doubled nose, lopsided expression. This machinery fires when the face is *off-manifold* relative to natural-face expectation.

A smooth on-manifold map will *never* trigger this response by design: on-manifold means "still in the natural face distribution." The user's decision (2026-04-14): do **not** demand that the identity-channel candidate also produce uncanny output. Most candidates are not built for off-manifold work; disqualifying them on that basis is wrong. Instead, the pipeline is a **two-stage composition**:

    X  →  [ identity subsystem: on-manifold ]  →  photorealistic face I
              (StyleGAN3 / Arc2Face / HyperFace-A / FLAME / ...)

    (I, Y)  →  [ drift subsystem: off-manifold ]  →  corrupted face I'
                (secondary pipeline explicitly built to "mess up" the face under control)

The two stages are scored on **separate rubrics** (see §2.2 and §2.5). Identity candidates are judged only on identity-channel properties (rank correlation, Fisher ratio, σ-injectivity, etc.), never penalized for lacking uncanny reach. Drift candidates are judged only on drift-channel properties (off-manifold reach under control, Y-monotonicity, identity preservation at low Y, no collateral channel leakage).

**Weak proof of neural-machinery activation.** The reason we commit to having a drift subsystem *at all* — even a crude one — is that its mere existence is a weak but non-trivial proof that the product uses the **face-reading perceptual system**, not just icon-recognition. A user study showing that high-sus faces are rated "wrong" by subjects who can't say *why* is direct evidence that the uncanny channel is firing. Without any drift mechanism, vamp-interface collapses into "photorealistic Chernoff faces" and the neural-machinery claim becomes unfalsifiable and probably false. So: drift is **required for the product**, but its implementation is **independent from** the identity subsystem and its quality bar is much lower (it just needs to work, not to be SOTA).

**Candidate drift subsystems (first sketch).** From the 2026 portrait-animation scan (`docs/research/2026-04-14-neural-portrait-animation-2026-scan.md`) and prior work:
- **Classical image-space corruption.** Landmark detection + explicit warp/duplicate/erase of features. Cheapest, most controllable, ugliest in the good way.
- **Inverted-face latent corruption.** Invert the identity-face into StyleGAN W or FLAME params, perturb *anti-disentangled* directions (deliberately activate multiple features at once), re-render. Requires invertibility of the identity candidate's output to some editable latent.
- **Diffusion img2img with "wrong face" prompt.** The 2025 vamp-interface v8 LoRA approach. Empirically overshoots into zombie/skin-lesion territory (see `feedback_lora_uncanny_tuning.md`). Tuning problem, not architecture problem.
- **Asyrp h-space editing.** Latent-DM editing with a trained CLIP-directional "uncanny" vector. Validated only on unconditional DDPM in the paper; risk noted.
- **Animator-family failure-mode exploitation.** Feed the identity face into LivePortrait / MMFA / DyStream / SuperHead as a still-reference, drive it with an *exaggerated* motion signal that the animator was not trained to handle. The animator's natural out-of-distribution failure mode *becomes* the uncanny signal. This is the clever option: we aren't asking the animator to "do uncanny," we're asking it to "fail visibly under push." The LivePortrait lineage is the ecosystem where this is most likely to work and is why the tavily scan actually matters to the framework despite none of those papers accepting a source embedding directly.

All five are framework-legal drift subsystems. Scoring them is §2.5's job.

### 1.8 Smoothness vs. density gaps

Semantic embeddings of real corpora have **density gaps**: regions of the embedding space where no job postings live. A pure Lipschitz-continuous map will dutifully interpolate over these gaps, rendering interior points as "faces of jobs that don't exist." Whether that is good (smooth visualization) or bad (visually indistinguishable noise) depends on what you want the visualization for.

The framework handles this by tracking **support** as a property: does the map preserve the support of the source distribution, or does it fill in gaps with interpolated faces? Neither answer is automatically right; both are choices to make explicitly.

### 1.9 Same-archetype collapse is a structural feature, not a failure mode

From measured data on 543 jobs (2026-04-07 final findings), ArcFace IR101 on the v1 generation run reports:

- `cleaning_legit ↔ construction_legit` inter-sim = **0.84** (highest pairwise similarity across the entire cohort grid)
- `office_scam ↔ pay_mismatch_scam` = 0.70
- `easy_money_scam ↔ remote_scam` = 0.71

These are not failures of the identity mapping. They are faithful renderings of the fact that **these cohorts are genuinely close in qwen space**: physical-worker job postings use overlapping vocabulary, and scam postings of similar structural patterns use overlapping rhetoric. The identity subsystem's correct behavior is to **preserve this collapse**, not fix it — because fixing it would require fabricating a distinction the source does not contain.

Framework consequences:

1. **P4 Fisher ratio is bounded by `d_X`.** An identity candidate cannot be credited for "improving" separation beyond the source space's own Fisher structure unless it injects external information. Scoring a candidate against a fixed numeric Fisher floor (e.g., "≥ 3") is wrong; the floor is source-dependent and must be computed as a percentage of the *qwen-level Fisher ratio on the same clustering*. Candidates preserving 80–100% of qwen's Fisher ratio are passing; candidates below 80% are losing information in the map; candidates above 100% are either using an editorial override (§1.9.1) or hallucinating.
2. **Editorial overrides are legal and must be separately tagged.** v3's solution (hand-assigned clothing: apron for cleaning vs. hi-viz vest for construction) adds no biometric information to the ArcFace-measured identity. It adds a **human-parseable categorical cue** on top of the identity channel, orthogonal to embedding geometry. This is a legitimate design move — it respects P4 (the biometric Fisher ratio is unchanged) while improving human-readable cohort distinction. The framework should treat editorial overrides as a **separate, optional layer** on top of the identity channel, not as an attempt to exceed P4.
3. **The source space is not neutral.** qwen was trained for semantic similarity on general text, not for our specific clustering goals. Whether that clustering matches what the product wants is a **product decision, not a framework decision**. The framework's role is to make the tradeoff visible (how much information is in the source? how much is preserved? what editorial overrides are layered on top?), not to pick a side.

#### 1.9.1 The editorial override layer

Whenever the product wants a distinction that the source space does not contain, it must be injected from outside — either as a hand-assigned categorical cue (v3 clothing), as a structured side-channel (`work_type` parsed separately from `raw_content`), or as a learned probe trained against explicit labels. This is a **fourth subsystem** in the framework that sits between identity and drift:

    Identity subsystem (respects qwen geometry)  →  Editorial override layer (injects named categorical cues from outside qwen)  →  Drift subsystem (off-manifold under Y)

The editorial override layer is not scored on P4a or P4 — it is scored on **auditability** (can a human inspect the override and decide if it's reasonable?) and **non-contamination** (does the override leak into the biometric identity channel, inflating P4 fraudulently?). The v3 work_type archetype system is an editorial override layer applied via T5 text conditioning, with 10 hand-curated archetypes and 20 hand-written (clean, scam) variant prompts.

---

## 2. The framework

### 2.1 Objects

A **candidate pipeline** is a tuple

    (X, d_X) →_{f_0} (Z_1, d_1) →_{f_1} ... →_{f_{n-1}} (Z_n, d_n) →_{render} (P, d_P)

where:

- `(X, d_X)`: **source space.** The embedding of a job posting. Today: qwen3-embedding:0.6b, 1024-d, cosine metric expected-valid. Eventually: possibly a raw 16-d sus_factors vector, or a structured concatenation.
- `(Z_i, d_i)`: **intermediate spaces.** Examples: PCA-whitened qwen, CLIP token space, StyleGAN W, ArcFace 512-sphere, FLAME (shape, expression, pose) parameters, SDXL UNet h-space.
- `(P, d_P)`: **perceptual/pixel space.** The production tool is **ArcFace IR101 via `minchul/cvlface_arcface_ir101_webface4m`** (512-d, LFW 99.83%, IJB-C 97.25%, identity-invariant to lighting/expression/pose) as `d_P` for identity-channel measurements (P4a, P4, P6) and drift-channel monotonicity (D1, D2). Invoked via `src/face_distinctness.py --model arcface`, which handles the `transformers >= 5.5` loading fix documented in `docs/research/2026-04-07-final-findings.md`. **LPIPS** is retained as a *secondary* metric specifically for D4 (region-wise drift collateral leakage) where identity-invariance is not wanted. Pixel L2 is useless. CLIP ViT-B/32 is biased toward SDXL output and unreliable for cross-model comparison (Flux faces land in a tight cluster regardless of actual distinctness) — do not use it for P4a/P4 even though it has been used historically.
- `f_i`: the map at each stage. May be linear (PCA, affine), learned (MLP, LoRA, UNet), or hand-specified (hash-to-W, cluster-to-anchor).
- `render`: the final step that produces pixels from the last latent. For StyleGAN: synthesis network. For Arc2Face/InfiniteYou: diffusion sampling. For FLAME: a renderer.

A candidate pipeline is **scored** against the properties in §2.3.

### 2.2 Three-channel architecture: identity + editorial + drift

The pipeline is the **composition of three independently-scored channels**, plus an optional readable-Y stage that rides on whichever channel best supports it. The v0.4 two-stage phrasing (identity + drift with editorial as optional override) understated the role of the editorial layer; in practice v3's 10 hand-curated work_type archetypes with clothing differentiation do most of the visible perceptual work on our corpus.

**Channel 1 — Identity (geometric embedding).** Maps source X to a photorealistic on-manifold face image I via the geometric structure of the source space. Responsible for cluster membership visibility via rank-preserving similarity: close qwen points map to visually similar faces, and the ordering of pairwise qwen distances survives into face distances (with distortion-budget allocated per §1.5). Scored on identity rubric (§2.3): P\* user-task accuracy, P4a local + global rank correlation, P4 Fisher ratio, P6 σ-injectivity. **Not** scored on off-manifold reach or semantic readability.

**Channel 2 — Editorial (semantic labeling).** Layered on top of identity: hand-curated categorical overrides that inject human-parseable information *not derivable from the geometric embedding alone*. Examples from v3: "cleaning archetype wears apron, construction archetype wears hi-viz vest, office archetype wears suit." These differences are biometrically invisible (ArcFace IR101 cannot see a vest) but carry most of the perceptual clustering signal for humans. The editorial channel has its own rubric (§2.3a, new): E1 readability (can a human name what they see?), E2 information contribution (how much of P\* is editorial vs. identity?), E3 leakage into identity (do editorial cues shift biometric embeddings?), E4 scalability (how many archetypes before the editor's job becomes intractable?).

**Channel 3 — Drift (perceptual salience amplifier).** Takes (I, y) and produces a corrupted face I', where y is the uncanny-type Y (currently `sus_level`). Responsible for triggering the face-reading perceptual machinery via controlled factor-wise inconsistency (the mechanism reframed in §2.5). Scored on drift rubric (§2.5): D1 factor-mismatch reach, D2 Y-monotonicity, D3 identity preservation at y=0, D4 mismatch *signature* (not uniform leakage), D5 reversibility. Already measured passing on Flux v3 with r=+0.914.

**Optional stage-1.5 — Readable Y-channels.** For any Y that is supposed to name a face axis (e.g., "formal job → suited/older face"), a learned or hand-coded direction injected at whatever point supports disentanglement (StyleGAN W, FLAME params, LoRA). Currently **zero committed readable Ys**; the prior over-commitment to "qwen axis → face axis" is retracted (see theory constraints doc). This sub-stage may remain empty at launch.

The three channels are engineered, trained, and evaluated independently. A final pipeline is a **triple** (identity-candidate, editorial-candidate, drift-candidate), with joint end-to-end checks: (a) editorial cues do not corrupt biometric identity, (b) drift at y=0 does not corrupt the identity+editorial output, (c) the combined output optimizes P\* — downstream user-task accuracy.

**Why this restructure matters.** Under the v0.4 two-stage phrasing, v3's success was partly attributed to the identity channel and partly ascribed to "editorial override as patch." Under v0.5, editorial is credited as a first-class information channel doing its own perceptual work, which matches what we actually measured. This changes what "replacing the identity candidate" means: a new identity candidate must either *include* equivalent editorial capacity or be paired with an explicit editorial layer, not compared head-to-head with v3's identity alone.

This split resolves tensions §1.5–1.7 and §1.9. It also explains why the 2026 portrait-animation literature matters: those papers are candidate drift mechanisms, not identity mechanisms.

### 2.3 Identity-channel rubric

Each property gets a status: **required** (candidate fails if absent), **target** (scored), **supporting** (diagnostic only, not gating), **optional** (bonus), **NA** (not measured this pass). **P0** is an interface gate — candidates that fail P0 never reach scoring. **P\*** is the top-level acceptance criterion that all other properties are proxies for.

| # | Property | Status | Measurement |
|---|----------|--------|-------------|
| **P\*** | **Downstream user-task accuracy** — can humans correctly perform the vamp-interface tasks (classify-by-suspicion, find-similar-jobs, cluster-by-archetype) using generated faces? | **required ultimate criterion** | 5–10 person user study, three conditions: (a) text-only baseline, (b) faces without drift, (c) faces with full drift+editorial. Tasks: "which of these is more suspicious?" and "which are similar jobs?" Metric: task accuracy relative to ground-truth labels; improvement over baseline. P4a/P4/P6 and D1/D2 are all *proxies* for P\*; any conflict between a proxy and P\* is resolved in favor of P\*. |
| **P0** | **Input-interface compatibility** — does the candidate accept our source X (possibly via a bounded-cost adapter)? | **required gate** | Inspection. Previously eliminated PhotoMaker, InstantID, RigFace, NoiseCLR; the v0.3 anchor-bridge broadening re-admitted them. |
| P1 | Source metric fitness — is `d_X` meaningful on our corpus (not just in general)? | **required** | Neighbor-retrieval sanity on a labeled subset; k-NN cluster purity vs. sus-label / scam-type. |
| **P4a** | **Distance-rank preservation — global AND local** — does the ordering of pairwise distances in X survive into face-distance at *both* granularities? | **required** (both halves) | **Global half:** Spearman ρ on sampled pairs using ArcFace IR101 cosine (biometric) or DreamSim cosine (holistic, NeurIPS 2023, ~96% human triplet agreement) as face-distance. Soft floor: ρ ≥ 0.6. **Local half:** k-NN preservation rate at k∈{5, 10} — fraction of qwen-k-nearest-neighbors that remain face-k-nearest-neighbors. Soft floor: ≥0.5 at k=5, ≥0.6 at k=10. Equivalently, trustworthiness and continuity (Venna & Kaski 2001, sklearn `sklearn.manifold.trustworthiness`) each ≥0.7 at k=10. **Why both:** global ρ hides catastrophic local inversions, and our product is *visual clustering* — humans read local neighborhoods, not global order. A candidate with ρ=0.8 but k-NN preservation 0.2 is useless; a candidate with ρ=0.5 but k-NN preservation 0.8 may well pass P\*. **Per distortion-budget allocation (§1.5), continuity is weighted above trustworthiness:** tearing qwen neighbors apart is worse than inventing new ones within the same cluster. **Seed-stability sub-diagnostic:** measure ρ across 3–5 seeds on a subset; report variance. Production uses `seed = hash(job_id)` so seed is deterministic given input, but a candidate chaotic across seeds is brittle. See §2.6. |
| P4 | Fisher ratio — `(mean inter-cluster face-distance) / (mean intra-cluster face-distance)` for a held-out clustering on X | **required**, upper-bounded by source per Information Bottleneck `I(Z;Y) ≤ I(X;Y)` (Tishby 2000) | ArcFace IR101 on generated faces, measure both as absolute value **and** as ratio to `Fisher(qwen, same clustering)`. A candidate passing at ≥80% of qwen-level Fisher is preserving source structure; above 100% requires explanation (editorial override or hallucination). See §1.9, §2.6. DIB (Yan et al. CVPR 2024) is the operational tool if we ever want the actual mutual-information ratio. |
| P6 | Identity σ-injectivity — distinct X points at `d_X > σ` map to distinguishable faces | **required** | Face-hash collision rate at chosen σ. |
| P2 | Per-stage Lipschitz bound — `d_{i+1}(f_i(a), f_i(b)) ≤ L_i · d_i(a, b)` | **supporting** | Empirical 95th-percentile pair-distance-ratio. Reported if measurable, used to bound P3 when end-to-end measurement is expensive, **not** an acceptance criterion. Discrete maps pass P4a + P4 + P6 without a finite Lipschitz constant, and that is fine. |
| P3 | Composite distortion (was: composite Lipschitz) — end-to-end spread of face-distance at fixed `d_X` | **supporting** | Line sweep in X, LPIPS dispersion on output. Diagnostic only — P4a is the acceptance criterion, P3 tells us *how* a failing P4a is failing (collapse vs. explosion vs. noise). |
| P5 | Betweenness / geodesic approximation | **optional** | Triangle-inequality-tightness on curated triples. Optional because modern manifolds rarely satisfy this. |
| P7 | Disentangled direction support for *readable* Y-channels (stage 1.5) | **target** per direction | Empirical, per direction per candidate. |
| P8 | Y-channel monotonicity for readable Ys | **required per committed readable Y** | α-sweep. Currently vamp-interface has **zero committed readable Ys**, so this line has no tenant this pass. |
| P10 | Readable-Y leakage into identity | **target** | Paired generation with readable-Y varied, identity inputs fixed. |
| P11 | Support preservation vs. gap-filling | **NA this pass** | Revisit if a candidate's behavior forces the question. |
| P12 | Algebraic closure (local) | **optional** | Single-point test. Not blocking. |
| **P15** | **Identity stability (stochastic consistency)** — for a fixed input X, how stable is the face embedding across seed variation? | **required** | Sample `n ∈ {3, 5}` seeds per input on a 30-job subset, compute ArcFace IR101 embedding for each, report `E_x[Var({p_i})] ≤ τ_var`. Soft floor: `τ_var ≤ 0.01` cosine-variance. Production ships with `seed = hash(job_id)` so `f` is deterministic *per input*, but a candidate whose embedding shifts significantly under seed perturbation is brittle and should be rejected even if P4a looks clean at the production seed. Promoted from a P4a sub-diagnostic in v0.5 to a top-level cell in v0.6. |
| **P13** | **Verification status** — for every cell, is the answer (a) measured on our corpus, (b) inherited from a paper's claim on a different corpus, or (c) assumed? | **required metadata** | Every cell tagged `{measured / inherited / assumed}`. |

**Note on P9.** The old P9 ("Y-channel off-manifold reach") has been removed from the identity-subsystem rubric entirely. It now lives in §2.5 as drift-subsystem property D1. Identity candidates are not penalized for being on-manifold; that is their job.

P13 is the discipline hook. Last session's failures came from treating "the paper says X" as "X is true for our use" — exactly the inherited-vs-measured conflation. The grid forces that distinction explicit.

### 2.3a Editorial-channel rubric (new in v0.5)

The editorial channel layers hand-curated categorical semantic labels onto the identity channel. Its purpose is to inject human-parseable information that the geometric embedding alone cannot (or does not) carry — clothing, accessories, setting, pose-class. On v3, the editorial layer is "10 work_type archetype prompts with clothing differentiation"; it is the channel doing most of the visible cluster-differentiation work.

| # | Property | Status | Measurement |
|---|----------|--------|-------------|
| **E0** | **Editorial interface compatibility** — can the identity-channel mechanism accept a categorical override from the editorial layer without degrading identity properties? | **required gate** | Inspection. Most T2I models pass trivially (prompt concatenation). Vector-space candidates (Arc2Face) may need a second conditioning slot. |
| **E1** | **Readability** — can a human name the difference between two editorial categories by looking at the face? ("apron vs hi-viz") | **required** | Small rating study, 5 raters, 10 archetype pairs. ≥70% correct identification. |
| **E2** | **Information contribution** — how much of P\* (user-task accuracy) comes from the editorial channel vs. from the identity channel alone? | **required** | Ablation: P\* study run twice, once with editorial-level archetypes, once without (uniform base prompt). The P\* gap is the editorial contribution. |
| E3 | **Leakage into identity** — do editorial cues shift biometric identity embeddings beyond tolerance? | **target** | Paired generation, same identity-channel input, varying editorial archetype. ArcFace IR101 cosine drift ≤ 0.1. |
| E4 | **Scalability** — beyond ~10 hand-curated archetypes, editorial bandwidth saturates. How many archetypes before the editor's job becomes intractable or cross-archetype collisions dominate? | **target** | Practical: 10 is fine, 30 probably fine, 100 probably not. Empirical; no sharp threshold. |
| **E13** | Verification status tag for every cell | **required metadata** | Same discipline as P13. |

**Passing the editorial channel.** Viable if E0, E1, and E2 pass. E3 is a ranking factor. E4 is a scoping constraint — a candidate that demands 200 archetypes to cover the corpus is operationally untenable even if other properties pass.

**Why this is a first-class channel.** Under v0.4 the editorial layer was a §1.9.1 "optional override" patch. Under v0.5 it carries its own information budget and gets explicit evaluation. This matches v3's empirical behavior: `cleaning_legit` and `construction_legit` have ArcFace inter-sim 0.84 (biometrically collapsed per §1.9), but human raters differentiate them easily because the editorial channel supplies the apron-vs-vest distinction. Attributing that success entirely to the identity channel would mis-locate the credit and lead us to propose identity replacements that don't carry equivalent editorial capacity.

### 2.4 What counts as "passing" (identity channel)

A candidate is **viable as an identity-channel mechanism** if:
- It passes P0 (interface).
- It passes all required identity-channel properties: P1, P4a (global + local rank preservation), P4 (Fisher ratio), P6 (σ-injectivity).
- P13 tagging is filled in for every scored cell.
- It contributes to P\* (user-task accuracy) above the text-only baseline when paired with an editorial channel and drift channel.

It is **preferred** over another viable candidate if it scores higher on **target** properties, with P\* weighted highest, then P4a (both halves) and P4. Supporting properties (P2/P3) inform diagnosis but do not move the ranking by themselves. Optional properties break ties only.

A candidate failing P4a or P4 on the identity channel is dead for that role. It may still be viable as a drift channel (§2.5) or as a readable-Y-channel mechanism — evaluated on its own rubric there, not retroactively rehabilitated.

### 2.5 Drift-channel rubric

A drift channel takes `(I, y) ∈ P × ℝ` and produces `I' ∈ P`, where `I` is a photorealistic face from the identity+editorial channels, `y` is the uncanny-type Y (currently `sus_level ∈ [0, 1]`), and `I'` is the corrupted output displayed to the user.

**v0.5 definition — drift as factor-wise inconsistency, not raw off-manifold reach.** The v0.4 D1 phrasing ("off-manifold reach") is a *proxy*, not a definition. Uncanny-valley empirical literature (Kätsyri et al. 2015 *Frontiers in Psychology*; Diel et al. 2022 meta-analysis) supports **perceptual mismatch across face factors** as the eeriness driver, not uniform degradation or raw distance from the natural-face manifold. Faces are implicitly factorized into roughly-orthogonal dimensions:

- **Identity** (ArcFace 512-d embedding — who is this?)
- **Expression** (FLAME ψ 100 PCs — what emotion?)
- **Anatomy** (FLAME β 300 PCs — bone structure, symmetry)
- **Texture** (skin detail, pore structure, lighting-dependent realism)
- **Geometry consistency** (eye alignment, feature proportions, left/right symmetry)

Drift, formally, is **any mechanism that produces inconsistent assignments across these factors**. Realistic skin + wrong geometry = uncanny. Realistic identity embedding + impossible expression asymmetry = uncanny. Uniform degradation across all factors = "ugly," *not* uncanny — the face-reading machinery does not fire.

This reframing has two consequences:
1. **D1 measurement remains anchor-distance in practice** — ArcFace IR101 anchor-distance is a computable aggregate proxy for factor-mismatch (if any factor is violated, the aggregate embedding drifts). The baseline r=+0.914 on Flux v3 stands.
2. **D4 gets sharper teeth** — the goal is to produce a *mismatch signature* (some channels corrupted, others preserved), not uniform LPIPS degradation. This is what the `Cursed_LoRA_Flux + Eerie_horror` mechanism empirically does on Flux, even though it was not designed factor-wise.

**v0.5 note on baseline status:** D1 and D2 are **already measured as passed on Flux v3** with `r(ArcFace_anchor_distance, sus_level) = +0.914` on 543 jobs (2026-04-07 final findings). The drift channel is built, measured, and working. The α-sweep pre-flight has been executed. The empirical LoRA mechanism produces factor mismatch implicitly — this is good fortune, not principled design, and the factor-mismatch framing gives us the vocabulary to design explicit factor-wise drift mechanisms later (e.g., FLAME ψ-only perturbation via Arc2Face+blendshape adapter, keeping β and identity untouched). Remaining drift work: LoRA curve tuning (§3.2.2 v8c overshoot) and optional per-factor decomposition via Concept Sliders or Arc2Face+blendshape (§3.2.5, §3.1.9).

| # | Property | Status | Measurement |
|---|----------|--------|-------------|
| **D0** | **Input-interface compatibility** — accepts a face image (and optionally a scalar control, or we can add a thin wrapper that converts y to the native control, e.g., motion magnitude or prompt strength) | **required gate** | Inspection. |
| D1 | **Factor-wise inconsistency** — as `y → max`, does the output `I'` exhibit *mismatched* assignment across face factors (identity / expression / anatomy / texture / geometry-consistency) in a way humans detect as "wrong"? | **required** | **Objective half:** ArcFace IR101 anchor-distance monotone in `y`, Pearson r ≥ 0.7 floor — this is a computable aggregate proxy; any factor violation shows up as embedding drift. **Baseline measured at r=+0.914 on Flux v3 (2026-04-07).** Principled upgrade (future): per-factor measurement via FLAME β/ψ inversion (SMIRK) + texture/geometry subscores. **Subjective half:** small-scale user rating study confirming "wrong but I can't say why" at high `y`. Subjective half is the product's whole point; objective half is a necessary-but-not-sufficient proxy. Raw-likelihood OOD is broken (Nalisnick ICLR 2019); factor-mismatch framing is the principled replacement. See §2.5 preamble, §2.6. |
| D2 | Y-monotonicity — perceived wrongness increases monotonically in `y` across the full range | **required** | α-sweep via ArcFace anchor distance. **Baseline measured at r=+0.914 on Flux v3 (2026-04-07).** This IS the drift pre-flight, already executed. **Optional post-hoc calibration:** isotonic regression via Pool Adjacent Violators (PAV) straightens minor non-monotonic wiggles in O(n) with no retraining. `sklearn.isotonic.IsotonicRegression`. See §2.6. |
| D3 | Identity preservation at y=0 — `I' ≈ I` when `y = 0` | **required** | LPIPS(I', I) < ε at y=0. Without this, stage 2 corrupts the stage-1 work before any signal arrives. |
| D4 | Collateral-channel leakage — varying `y` should produce *mismatched* perturbation across channels (realistic skin + wrong geometry = uncanny), not uniform degradation | **target** | Paired generation with `y` varied, measure region-wise LPIPS on non-face regions. **Reframed per Kätsyri 2015 + Diel 2022 meta-analysis:** the goal is a mismatch *signature* (high LPIPS in some channels, low in others), not zero LPIPS everywhere. Uniform degradation yields "ugly" not "uncanny." See §2.6. |
| D5 | Controllability / reversibility — the mapping `(I, y) → I'` is deterministic and well-defined across the whole `y` range | **target** | Repeated generation with same (I, y) produces same output; sweep continuity on y. |
| D6 | Cost and latency compatible with pre-generate-and-cache | **target** | Wall-clock per frame; target < 5s on local GPU. |
| **D13** | Verification status tag for every cell | **required metadata** | Same discipline as P13. |

**Passing a drift candidate.** A drift subsystem is viable if it passes D0, D1, D2, D3. D4/D5/D6 are ranking factors. D1 is explicitly **two-part** — the classifier measurement is necessary but the subjective rating is the ground truth. A drift mechanism that fools the classifier without firing the human face-reading response has failed.

**The weak proof.** A drift candidate that satisfies D1 (subjective half) is itself weak evidence that vamp-interface engages the face-reading perceptual system rather than decorative icon-recognition. This makes D1 doubly important: it is both the drift-subsystem acceptance criterion *and* the product's answer to "is this a Chernoff-plus-photorealism or something more?"

### 2.6 Math foundations — what the rubric rests on

This section names the adjacent-literature concepts each rubric cell inherits from, so that the framework is citable instead of hand-wavy. Full synthesis in `docs/research/2026-04-14-math-foundations-synthesis.md`. This section does NOT add new rubric cells — it anchors the existing ones.

**P4a (rank preservation).** The scalar Spearman ρ floor is the coarse metric. The principled decomposition is **trustworthiness and continuity** (Venna & Kaski 2001; Lee & Verleysen 2008 co-ranking matrix), which split rank errors into (a) *false* neighbors invented by the map and (b) *torn* neighbors lost by the map. These are Pareto-incompatible: any nontrivial map must choose. The identity subsystem cares more about **continuity** (don't tear qwen clusters apart). The perceptual backbone for measuring face-distance is either **ArcFace IR101** (identity-invariant, current production tool) or **DreamSim** (NeurIPS 2023, `D(x,x̃) = 1 − cos(fθ(x), fθ(x̃))`, ~96% human triplet agreement) depending on whether we care about biometric or holistic similarity — both are legal P4a backbones. Scalar Spearman ρ remains the soft floor; trustworthiness/continuity is an optional deeper diagnostic for disputed cases.

**P5 (distribution preservation).** The coordinate-free measure is **Gromov-Wasserstein distance** (Mémoli 2011) on metric-measure spaces. GW is invariant to isometries and does not require a shared basis, which is correct for aligning qwen-1024 with face-pixel-space. SCOT (Demetci et al. 2020) demonstrates GW on single-cell multi-omics, which is mathematically our problem (align two spaces with no correspondences). **Brenier's theorem** (1991) gives the existence result: for quadratic cost and AC source measure, a unique pushforward map `T = ∇φ` with `T#µ = ν` exists — so P5 is achievable in principle, not just in hope. P5 remains an optional diagnostic in the framework: we lack the tool to compute GW on the full corpus cheaply, and no candidate is being rejected on P5 grounds today.

**§1.5 Fisher upper-bound.** The claim "Fisher ratio on the identity channel is upper-bounded by the source space" is a special case of the **Information Bottleneck** inequality `I(Z;Y) ≤ I(X;Y)` (Tishby & Zaslavsky 2000; generalization-gap bounds by Vera & Piantanida). No map from X to Z can manufacture class-discriminative information about Y that X did not already carry. For any future measurement, **Differentiable Information Bottleneck** (Yan et al. CVPR 2024) is the operational tool — kernel-eigenvalue mutual information, no variational approximation. The framework does not gate on an IB ratio today, but §5 adds this as a post-baseline experiment.

**Lipschitz demotion.** The walked-back over-commitment to Lipschitz is honest under **Kantorovich-Rubinstein duality**: `W1(µ,ν) = sup_{Lip(f)≤1} E_µ[f] − E_ν[f]`. Lipschitz functions are *dual* to 1-Wasserstein distance, not a replacement for distribution preservation. So "we care about W1, and Lipschitz is a diagnostic of a map's contribution to W1" is the correct phrasing. Lipschitz is supporting, not core; W1 (or GW) is the core object if we ever compute it.

**D1 (off-manifold reach) — principled form.** The current `ArcFace_anchor_distance` proxy is a pragmatic OOD score. The principled form is **typical-set distance**, not raw likelihood (Nalisnick et al. ICLR 2019 "Do Deep Generative Models Know What They Don't Know?"). Flows and diffusion models assign high likelihood to OOD data in pathological ways; the typical set is what matters. Energy-based OOD scores and CLIP-embedding-norm variability predictors (ACL 2024 "Words Worth a Thousand Pictures") are candidate upgrade paths. The framework retains ArcFace anchor-distance as the D1 objective-half measurement because it already correlates r=+0.914 with `sus_level` on our corpus — fixing a working proxy is not a priority. The *subjective* half (user rating) remains ground truth regardless.

**D4 (collateral leakage) — mismatch reframing.** Uncanny valley empirical literature (Kätsyri et al. 2015; Diel et al. 2022 meta-analysis) supports **perceptual mismatch** as the valley driver, not raw human-likeness. Implication for D4: the goal is not to minimize collateral leakage uniformly, but to ensure leakage is *mismatched across channels* — realistic skin + wrong geometry triggers the valley; uniformly ugly faces do not. This reframes the existing "region-wise LPIPS on non-face regions" measurement: what we want is a mismatch signature (high LPIPS in some channels, low in others), not zero LPIPS everywhere.

**D2 monotonicity — cheap enforcement.** **Isotonic regression / Pool Adjacent Violators (PAV)** gives us O(n) post-hoc monotonicity calibration of any scalar-to-scalar function. If `ArcFace_anchor_distance(sus_level)` has minor non-monotonic wiggles, PAV straightens them for free — no retraining. Added to the D2 measurement pipeline as an optional calibration step.

**Readability drop — glyph literature cross-check.** The stage-1.5 readable-Y-channels decision (framework §2.2: currently zero tenants) is validated by empirical glyph studies (Borgo et al. 2013 systematic review; Fuchs et al. 2017): Chernoff-style face glyphs underperform simpler glyphs on most tasks. Our product's identity channel is an identifier, not a readout — this is in line with what the visualization literature would recommend.

**What's still genuinely missing** (flagged as open by both research passes):

1. No finite-sample, neural-parameterized Brenier map estimation rate. Any distribution-preservation claim we make about a trained map is empirical, not theoretical.
2. No differentiable Kendall-τ surrogate with generalization bounds. If we ever train a projection, the loss (PCC-style Pearson+Spearman on distance matrices) has no clean sample-complexity guarantee.
3. No psychophysically validated uncanny functional tied to latent density. D1 subjective half remains a user study.
4. No formal channel-allocation theory for glyphs. Irrelevant at this moment (stage 1.5 is empty), but would bite if we ever reopen readability.

These are on the "accept and move on" side. The framework rests on theorems that exist (Brenier, Kantorovich-Rubinstein, Information Bottleneck, trustworthiness/continuity) plus measurements we have (r=+0.914, cluster sep 0.2179), not on theorems we wish existed.

### 2.7 Decision Protocol

The framework defines necessary conditions and comparative criteria for assessing semantic-to-perceptual mapping systems. The goal is not to collapse performance into a single scalar score but to **filter invalid systems** and **compare viable ones along interpretable axes**. Decision flow has two stages: feasibility filtering (§2.7.1), then Pareto comparison (§2.7.3).

#### 2.7.1 Acceptance criteria (hard constraints)

A candidate `f: X → P` is considered **valid** only if it satisfies the following — each with an explicit threshold. Below-threshold on any hard constraint is a rejection (§2.7.2), regardless of performance on other axes.

**Ultimate criterion — `P*` Downstream user-task accuracy.**

    P*(f) > P*(text-baseline) + ε_P*            (§2.3)

A candidate must measurably improve user task accuracy over the text-only baseline when paired with an editorial channel and a drift channel. All other constraints below are *proxies* for `P*` — cheap pre-filters run before the expensive user study. Any conflict between a proxy and `P*` is resolved in favor of `P*`.

**Rank preservation — global.**

    ρ_Spearman(d_X, d_P) ≥ τ_ρ,     τ_ρ = 0.6    (P4a global)

Where `d_X` is qwen cosine distance and `d_P` is ArcFace IR101 (biometric) or DreamSim (holistic) cosine distance, sampled over held-out pair set.

**Rank preservation — local (neighborhood consistency).**

    |N_k^X(x) ∩ N_k^P(x)| / k ≥ τ_k              (P4a local)

averaged over dataset, with `τ_k = 0.5` at `k=5` and `τ_k = 0.6` at `k=10`. Equivalently, trustworthiness and continuity (Venna & Kaski 2001) each `≥ 0.7` at `k=10`. **Local structure dominates human perceptual judgments** — see §1.5 distortion-budget allocation. Continuity is weighted above trustworthiness: tearing qwen neighbors apart is worse than inventing new intra-cluster neighbors.

**Identity stability (stochastic consistency).**

    E_x [ Var({p_i : i = 1..n}) ] ≤ τ_var,   τ_var = 0.01    (P15)

Where `p_i = ArcFaceIR101(f(x, seed_i))` for `n ∈ {3, 5}` seeds per input. Production ships deterministic via `seed = hash(job_id)`, but a candidate unstable under seed perturbation is brittle and rejected.

**Cluster structure preservation — Fisher ratio as ratio to source.**

    F(f) / F(qwen, same clustering) ≥ τ_F,   τ_F = 0.8    (P4)

Upper-bounded by the source per Information Bottleneck `I(Z;Y) ≤ I(X;Y)` (§1.5); a candidate above 100% is either using editorial overrides (§2.3a) or hallucinating distinctions.

**Drift — factor-mismatch reach + monotonicity.**

    r(d_anchor(f(x, y)), y) ≥ τ_drift,   τ_drift = 0.7    (D1, D2)

Where `d_anchor` is ArcFace IR101 anchor-distance as an aggregate factor-mismatch proxy (§2.5). Measured baseline on Flux v3: `r = +0.914`.

#### 2.7.2 Rejection conditions

A candidate is **rejected** (not scored further) if any of the hard constraints above fail:

- `P*` does not exceed text baseline (ultimate failure — the entire premise fails)
- `ρ_Spearman < τ_ρ` → global structure failure
- k-NN preservation `< τ_k` at either `k=5` or `k=10` → local perceptual coherence failure
- `Var > τ_var` → generative stability failure
- `F(f) / F(qwen) < τ_F` → cluster collapse beyond tolerance
- `r(d_anchor, y) < τ_drift` → drift mechanism does not track `y`

A candidate that clears all hard constraints enters §2.7.3 comparative scoring.

#### 2.7.3 Comparative metrics (within valid candidates)

Among candidates that pass §2.7.1, comparison is performed along the following axes. No single axis dominates; the output is a **Pareto frontier of viable candidates**, not a single optimum.

**Axis 1 — Identity separability.**

    F = (between-class variance) / (within-class variance)    (P4 absolute)

Higher is better. Upper-bounded by source; candidates at `F(qwen)` or above are fully preserving the source structure.

**Axis 2 — Drift controllability.**

Within the drift channel, evaluate:
- **Magnitude** — mean `d_anchor` at `y = max`
- **Variance** — `Var[d_anchor | y]` (low is better, stable across inputs)
- **Factor-specificity** — which face factors (identity / expression / anatomy / texture / geometry-consistency, §2.5) are violated at high `y`. A drift mechanism that targets expression specifically is more controllable than one that degrades all factors uniformly.
- **Reversibility** — D5 reproducibility under (I, y) repetition.

Drift is not penalized per se; it is evaluated for **consistency and controllability** along these axes.

**Axis 3 — Editorial contribution.**

    Δ_editorial = P*_with_editorial − P*_without_editorial   (E2)

Ablation study comparing uniform-base-prompt condition to full-editorial condition. Non-zero Δ_editorial is expected on our corpus; it quantifies how much user task accuracy is owed to hand-curated semantic labels vs. the geometric identity channel. Candidates that require larger editorial layers to cover the corpus are penalized via E4 scalability.

**Axis 4 — Cost and operational fitness.**

Training cost (GPU-hours), inference latency (wall-clock per face), license compatibility, and pre-generate-and-cache friendliness. Not a math axis, but a gate on viability for the product.

#### 2.7.4 Decision structure

Two-stage:

1. **Feasibility filter.** For each candidate, fill in §2.7.1 cells and apply §2.7.2 rejection rules. Output: set of surviving candidates `V = { f : f passes all hard constraints }`.

2. **Pareto comparison.** For each `f ∈ V`, compute the four axes in §2.7.3. Retain candidates on the Pareto frontier — no other `f' ∈ V` dominates them on all four axes simultaneously. This typically leaves 2–5 candidates, each representing a different trade-off (e.g., "best identity separability but mediocre drift," "best drift controllability but expensive to train").

The framework stops at the Pareto frontier. The final pick is a product / engineering decision informed by the comparison, not dictated by it.

#### 2.7.5 Interpretation — three framing principles

The protocol reflects three principles that distinguish this framework from single-metric benchmarks:

1. **Structure preservation is necessary but not sufficient.** Rank preservation (P4a global + local) and cluster preservation (P4 Fisher) are hard constraints because without them the geometric hypothesis fails. But a candidate that passes all geometric constraints can still fail `P*` — structure is a floor, not a ceiling.

2. **Perceptual validity is governed by local relationships.** Our product is visual clustering; humans read k-neighborhoods, not global rank order. The framework weights local P4a above global P4a in exactly the cases they disagree, per distortion-budget allocation (§1.5).

3. **Controlled deviations — drift and editorial cues — are first-class design dimensions.** Drift is not "failure of on-manifold fidelity"; it is a deliberate off-manifold mechanism with its own rubric. Editorial overrides are not "patches on top of the identity channel"; they are a first-class information channel with measured contribution. A framework that treats either as secondary will mis-locate credit and pick the wrong identity replacement.

This framing enables systematic comparison across qualitatively different approaches, including continuous mappings (h-space direction finding), discrete cluster-based methods (HyperFace Regime A, anchor-bridge adapter), and hybrid soft-interpolation methods (v3 Regime C'). All three are legal moves under the same rubric.

---

## 3. Sketch application (rubric shakedown)

Populating the full grid is the next pass's work. Here we apply the framework to candidates at a coarse grain just to verify the rubric discriminates sensibly under the v0.2 changes.

### 3.1 Identity-subsystem candidates

#### 3.1.1 Qwen → MLP → Arc2Face conditioning slot

- P0 interface: ok (accepts a 512-d ArcFace-shaped vector via zero-padded 768-d single token, per paper reread).
- P1 source: ok.
- P4a rank correlation: **unverified.** This is the continuity pre-flight gate we've been deferring. Whether a trained `g: qwen → ArcFace` preserves Spearman ρ is the first thing to measure.
- P4 Fisher ratio: unverified, same pre-flight.
- P6 σ-injectivity: depends on `g`.
- P2/P3 diagnostic: no paper-inherited Lipschitz bound for Arc2Face's conditioning sensitivity — the Arc2Face paper does not study it.
- **Status:** identity-viable pending P4a/P4 measurement. Interface clean, photorealism good, license CC-BY-NC.

#### 3.1.2 Qwen → PCA → StyleGAN3 native W (FFHQ)

- P0 interface: ok via an MLP adapter qwen→W.
- **Product-level eliminator (2026-04-12 decision review):** FFHQ-trained StyleGAN3 has near-zero expression range. W-space variation is almost entirely identity variation — age, hair, face shape, skin tone — with strong smile, cold eyes, hollow practiced smile, tense jaw all outside FFHQ's support. **This is fatal for the vamp uncanny mechanism**, which lives on *expression wrongness*, not identity variation. Independent of any Lipschitz argument. Even a theoretically-perfect rank-preserving `g: qwen → W` maps onto a face space that does not contain the signal the product depends on.
- **Variant that is viable:** StyleGAN3 **fine-tuned on AffectNet** (~4–8 GPU hours from FFHQ checkpoint per 2026-04-12 research) has expression range, and InterFaceGAN / SeFa / GANSpace find named directions in the expanded W-space. This is a real option but is not off-the-shelf — it requires training.
- P7 disentanglement (for optional readable Ys later): strong after AffectNet fine-tune — 10–20 known semi-disentangled directions from SeFa/GANSpace/StyleSpace analyses in the FFHQ W-space, which partially transfer.
- P2 diagnostic: paper-inherited (StyleGAN3 is designed to be Lipschitz in W).
- **Status (FFHQ):** eliminated on product grounds. **Status (AffectNet fine-tune):** viable identity candidate with ~4–8 GPU hours training cost, pending measurement of expression range on our actual sus-axis labels. License: NVIDIA research (non-commercial).

#### 3.1.3 Qwen → cluster-index → HyperFace Regime A (one face per cluster)

- P0 interface: ok (clustering + lookup).
- P1 source: ok.
- **P4a rank correlation: passes trivially at cluster granularity.** Within-cluster pairs all have face-distance 0, between-cluster pairs all have face-distance `≥ min_pairwise_angle`. Spearman ρ is 1.0 when ties are handled correctly (or undefined-but-fine; the interpretation is "monotone at cluster scale"). Under the v0.1 framework this candidate looked disqualified by Lipschitz; under v0.2 it is the cleanest pass on the entire rubric.
- **P4 Fisher ratio: maximal by construction.** HyperFace packs anchors to maximize the minimum pairwise angle.
- P6 σ-injectivity: passes if σ ≥ cluster radius; fails for sub-cluster resolution. User decides whether this matters for their use case (it does for the analyst scenario, may not for the scam hunter).
- P2 diagnostic: globally ∞ (step function). v0.2 treats this as OK because P2 is supporting, not gating.
- **Status:** identity-viable, arguably the cleanest candidate. Inherits the original vamp-interface fixed-anchor thesis as a special case. Main open question: cluster granularity N, which trades off P6 (finer N = better sub-cluster resolution) against visual distinctness (coarser N = cleaner Fisher).

#### 3.1.4 FLAME — better understood as a stage-1.5 readable-Y editor, not a standalone identity candidate

FLAME is a parametric 3D rig, not a renderer (5023-vertex mesh, PS2-era raw output). Using FLAME as an identity subsystem *requires* a two-stage composition: `qwen → MLP → (β, ψ, pose) → [neural renderer] → photorealistic face`. The viable renderer options for our no-per-person-data constraint are narrow:

- **GaussianAvatars / SPARK:** photoreal and real-time, but require per-person multi-view video. **Dead for us.**
- **Arc2Face + blendshape-guided expression adapter** (arXiv:2510.04706, ICCVW 2025): this is the specific off-the-shelf path and is listed separately as candidate §3.1.9.
- **RigFace:** no code released, dead.
- **Custom diffusion LoRA trained on FLAME conditioning:** research project.

**The more productive role for FLAME is as a stage-1.5 readable-Y-channel editor** stacked on top of any identity candidate via a round trip:

    I  →  [ SMIRK (CVPR 2024 SOTA) or EMOCA v2 (CVPR 2022) ]  →  (β, ψ, pose)_from_I
                                                                      ↓
                                                  perturb ψ and/or pose by a named Y
                                                                      ↓
    (β, ψ', pose')  →  [ diffusion renderer ]  →  I'_with_Y_edit

Under this framing, FLAME is **not a competing identity candidate**. It is an editor with **structural orthogonality guarantees** (β, ψ, pose are trained from three different data sources and occupy non-overlapping parameter dimensions), which means D10 (readable-Y leakage) **passes by construction** for any Y that can be expressed in FLAME's parameterization — an a-priori guarantee no empirical candidate can match.

**P7 for FLAME as stage-1.5 editor, split by parameter space:**

- **Pose axes (neck, head, jaw, eye):** semantically named by construction. "Gaze averted," "jaw open," "head tilt asymmetric" are direct Y-channel controls. No probe needed.
- **Expression (ψ, 100 PC):** statistically disentangled but not semantically named (PC1 = largest variance mode, not "smile"). Needs a probe. EMOCA v2 provides valence/arousal probes (human PCC 0.78/0.69). SMIRK provides perceptually-supervised emotion recovery.
- **Shape (β, 300 PC, first ~50 meaningful):** morphologically-semantic (face width, jaw prominence, brow length). Usable but less affect-relevant.

**FLAME cannot be the drift subsystem.** Its prior is on-manifold by design — "FLAME renders only valid human faces within its prior." Pushing FLAME parameters to extreme edges gives awkward-but-on-manifold faces, not uncanny-off-manifold faces. Drift remains a pixel-space or diffusion-latent operation.

- P0 interface: ok via MLP adapter (if used as standalone identity) or via SMIRK inversion (if used as stage-1.5 editor on top of another candidate).
- P7 for stage-1.5: **strongest of all candidates for pose; strong-with-probe for expression; moderate for shape.**
- License: **CC-BY-4.0** (FLAME 2023) — more permissive than Arc2Face (CC-BY-NC). Meaningful if commercialization is ever considered.
- Cost: FLAME param extraction ~12ms; ARKit↔FLAME conversion ~1ms. Negligible vs. diffusion renderer latency (~1–2s/image).
- **Status:** recommended as the stage-1.5 readable-Y-channel substrate for any identity candidate that wants structural Y-axis orthogonality. Not recommended as standalone identity unless §3.1.9 (Arc2Face + blendshape adapter) proves viable.

#### 3.1.5 P0 status of candidates that take face-image or text-token inputs

Under **v0.2** these were eliminated at the P0 identity gate as "candidates that do not accept an abstract source embedding" — PhotoMaker (multi-photo), InstantID (ArcFace + landmarks + IP-Adapter), RigFace (face image through FaceFusion features, also no code), NoiseCLR (CLIP text pseudo-tokens), 2026 portrait animators (reference portrait + motion).

Under **v0.3**, P0 is broadened to include the **anchor-bridge adapter class**: `qwen → cluster → (LLM or hand-curated) caption → T2I → anchor face`. Every text-to-image model trivially passes P0 under this bridge, and so do face-consuming models that can take the generated anchor as input:

- **PhotoMaker:** re-enters P0 via anchor bridge (feed generated anchor as reference). Identity-subsystem viable; needs measurement.
- **InstantID:** re-enters P0 via anchor bridge (extract ArcFace + landmarks from the anchor, feed both to IdentityNet). Identity-subsystem viable; needs measurement.
- **RigFace:** P0 passes interface-wise, **still blocked on missing code**. Dead.
- **NoiseCLR:** stays eliminated; its conditioning is learned CLIP text pseudo-tokens for attribute *editing*, not identity projection — wrong operation class.
- **2026 portrait animators (LivePortrait / PersonaLive / MMFA / DyStream / SuperHead / Splat-Portrait / UniTalking):** passes P0 for drift-subsystem (§3.2.3), not identity-subsystem (they animate, they don't project).

#### 3.1.6 v2/v3 + Flux anchor-bridge — the measured baseline

**This is not a hypothetical candidate. It is the system that exists and is running.** v3 directional anchors over Flux (`output/dataset_faces_flux_v3/`) is the current best identity-subsystem implementation and the baseline any new candidate must beat.

**Architecture** (per `docs/research/2026-04-09-embedding-space-and-face-conditioning.md`):

1. Clustering: HDBSCAN on PaCMAP-reduced qwen → 42 coarse clusters → hand-assigned to 10 archetypes (`data/cluster_centroids.pt`).
2. Anchor captions: 10 archetypes × 2 variants (clean / scam) = 20 hand-written T5 prompts in `data/archetypes.json`. Categorical override at the prompt level (cleaning → apron, construction → hi-viz vest, etc.) for cohorts that genuinely collapse in qwen embedding space (§1.9).
3. **Regime C'** — top-k-gated soft interpolation in hand-curated natural-language anchor space. Top-3 nearest cluster centroids by squared L2, RBF softmax with temperature 1.5, per-archetype accumulation. Top-1 ≈ 63%, effective_N ≈ 2.3.
4. Two conditioning channels: **T5 text** (primary, hard clean/scam switch at sus_factor=0.5) and **CLIP-L pooled** (secondary, smooth blend of precomputed archetype centroids). Mediated through `QwenPooledReplace` custom ComfyUI node.
5. Drift: `Cursed_LoRA_Flux` at strength = `sus_factor` plus `Eerie_horror` at `0.75 × sus_factor`. Seed deterministic via `seed = hash(job_id) % 2^32`.
6. Backend: Flux.1-dev fp8 scaled, 20 steps, guidance 3.5, txt2img at denoise=1.0.

**Measured properties (543 jobs, ArcFace IR101, 2026-04-07):**

| Property | Value | Source |
|---|---|---|
| **P4 identity cluster separation (ArcFace)** | **0.2179** | `dataset_faces_flux_v3/face_distinctness_arcface.json` |
| **P4 (Flux v4 PaCMAP variant)** | 0.2294 | `dataset_faces_flux_v4/` — beats v3 on separation, loses on sus correlation |
| **D1+D2 drift correlation r(anchor_dist, sus)** | **+0.914** | strongest of any tested combination; SDXL v1–v3 all at +0.688 to +0.724, Flux v4 at +0.847 |
| P4a rank correlation (Spearman ρ, qwen vs ArcFace) | **NOT YET MEASURED** in final-findings format; needs one script run |
| Known failure: `cleaning_legit ↔ construction_legit` inter-sim | 0.84 (highest pairwise) | same source — faithful source-space collapse, §1.9 |
| Known failure: CLIP-L pooled rank-inversion | cross-archetype 23.8° < same-archetype 25.0° | `2026-04-09-embedding-space-and-face-conditioning.md` §6 — measured on 20 test jobs, **sus axis dominates work-type axis in pooled vector space** |

**Framework interpretation:**

- **Drift subsystem D1 + D2 are already measured as passed** (r=+0.914). The drift mechanism is built, measured, working. Remaining work is LoRA-curve tuning (v8c overshoot) and optional per-flavor decomposition — neither is an architectural decision.
- **Identity subsystem passes P4 and P6 at v3-cluster granularity.** What is *not* yet measured is P4a (Spearman ρ between qwen pairwise distances and generated-face ArcFace pairwise distances on the 543-job set). That measurement would convert the grid from "baseline measured for drift, unmeasured for rank" to "baseline fully characterized." **This is the single most leverage-positive experiment the framework recommends.** It uses only existing data and existing tools.
- **The CLIP-L pooled channel is known rank-inverted.** This is the one measured architectural failure in the baseline. The current system works because the T5 text channel carries the work-type signal; CLIP-L pooled only contributes to the clean/scam axis (where it happens to be adequate). An improvement target is: either drop CLIP-L pooled entirely, or replace it with something rank-preserving, or relegate it to drift-axis only.
- **The editorial override layer (§1.9.1) is load-bearing.** v3's hand-assigned clothing categories (cleaning→apron, construction→hi-viz vest) are what make the physical-worker cohorts visually distinguishable in the final images, *not* the identity channel — the biometric Fisher ratio between these cohorts is capped by qwen's own geometry (inter-sim 0.84). The framework must acknowledge this layer as intentional, not as a bug to fix.

**Status:** strong measured baseline across P4 and D1+D2. Missing P4a measurement. This is the reference point for every other §3.1 entry. A new identity candidate has to either (a) beat v3 Flux at r(sus) **and** cluster separation on the same 543 jobs with the same ArcFace IR101 tool, or (b) pass P7 for a Y-channel that v3 does not support, or (c) offer a meaningful cost/license improvement.

#### 3.1.7 Anchor-bridge generalization (N&gt;10, Concept-Slider-enhanced)

Same architecture as §3.1.6 but generalized along three axes: (a) finer cluster granularity (N&gt;10, up to 42 native HDBSCAN clusters or more), (b) per-flavor sus decomposition via Concept Sliders (§3.2.5) instead of a single Cursed_LoRA dial, (c) replace or supplement the CLIP-L pooled channel with a rank-preserving mechanism (h-space direction injection from §3.1.8, or direct ArcFace centroid blending).

- P0: passes via the anchor-bridge (already exercised by v2/v3).
- P4a: unverified; should improve over v3 because per-flavor decomposition unbundles the collapsed multi-attribute sus axis.
- P4 Fisher ratio: potentially worse than v3 at coarse granularity (N=10) and better at fine granularity (N=42), because finer clusters better preserve qwen source structure (per §1.9 upper-bound argument).
- P7 readable Ys: strong via Concept Sliders (per-concept LoRA weight pairs, 50+ composable, explicit ground-truth training).
- **Status:** the framework-native evolution of v3. Scores better than v3 on every framework property *if* the CLIP-L pooled channel is fixed and Concept Sliders are trained for Flux. Both are bounded-cost engineering tasks, not research.

#### 3.1.8 h-space direction finding on frozen Flux (Asyrp / Haas / Self-Discovering Directions)

**The StyleGAN W-space paradigm inside the Flux model we already use.** Per 2026-04-12 algebra research:

- **Asyrp (ICLR 2023):** h-space (UNet bottleneck activations per timestep) is homogeneous, linear, robust, timestep-consistent. Validated on unconditional DDPM/iDDPM.
- **Haas et al. (IEEE FG 2024):** unsupervised PCA on h-space finds pose/gender/age directions for face editing without classifier training.
- **Self-Discovering Interpretable Directions (Li et al., CVPR 2024):** learns a direction vector for an arbitrary named concept through the denoising process itself, no labeled data after concept definition.
- **DiffFERV (IJCAI 2025):** h-space linear and geodesic traversal for facial expression video editing.

Architecturally: `qwen → learned projection → h-space perturbation` added to the v3 Flux pipeline at a specific timestep range. Zero migration cost — stays in ComfyUI, stays on Flux, adds learnable semantic direction vectors alongside existing anchor blending. The arithmetic is `h + α · v_concept → modified face` in the face latent space of the model we're already running.

- P0 interface: ok (h-space is internal to Flux, perturbation is added at sampling time).
- P4a / P4: should inherit v3 baseline (same Flux backend, same T5 conditioning) plus any gains from adding rank-preserving perturbations in h-space.
- P7 disentanglement: plausibly strong per Haas et al. and Self-Discovering Directions; **untested on our corpus**.
- **Blocking open question (flagged in algebra doc, not yet answered):** Asyrp / Haas / Self-Discovering Directions all use *single-prompt* denoising. v3 uses **ConditioningAverage-blended multi-anchor** conditioning. Whether h-space directions remain stable across blended conditionings is untested. The pre-flight experiment: run PCA on h-space activations across a batch of blended generations, check whether top-k directions remain stable across different blend weights. **~1 day, zero training, existing infrastructure.** This is the single highest-leverage experiment the framework can recommend.
- **Status:** if the pre-flight passes, h-space directions become a cheap additive layer to §3.1.6/§3.1.7 for both identity-channel enhancement and drift-axis decomposition. If the pre-flight fails, the path is blocked and StyleGAN-W-style arithmetic in face space requires either AffectNet-fine-tuned StyleGAN3 (§3.1.2) or parametric bundled (§3.1.9). **Pre-flight experiment runs first.**

#### 3.1.9 Arc2Face + blendshape-guided expression adapter (arXiv:2510.04706, ICCVW 2025)

"ID-Consistent, Precise Expression Generation with Blendshape-Guided Diffusion." Takes a 512-d ArcFace identity vector *and* a FLAME blendshape expression vector as two orthogonal conditioning channels via cross-attention on fine-tuned Stable Diffusion. This is the specific paper that operationalizes the "FLAME as identity substrate → photoreal diffusion renderer" path that had been hand-wavily called FEM-family in prior tiers.

- P0: accepts (ArcFace 512-d identity, FLAME 12-d expression) as two channels. Our adapter is `qwen → MLP → ArcFace 512-d` (the Arc2Face adapter pattern) plus `sus_factors → MLP → FLAME expression delta`. Both are bounded-cost.
- **Structural orthogonality guarantee** on the expression channel: FLAME shape vs. expression are trained from different data and occupy non-overlapping parameter dimensions. D10 leakage from expression to identity passes by construction for any expression-axis Y.
- P4a / P4 / P6: unverified; inherits SD's expression flexibility (unlike FFHQ-StyleGAN), so the expression-range objection to §3.1.2 does not apply.
- P7 readable Ys: strong for expression axes, weak for pose (FLAME pose is a separate channel the paper may or may not include — needs paper read).
- **Blocking task:** first-hand read of arXiv:2510.04706 to confirm exact input/output shapes, training requirements, checkpoint availability, license. This is a **specific** Tier-1b read, not a family hand-wave. Until it's done, this candidate is promising-but-unknown.
- **Status:** strongest structural-orthogonality FLAME-based identity path. Blocked on a single paper read. Should be tier-1b-prioritized above all other FLAME literature.

### 3.2 Drift-subsystem candidates (stage 2)

Evaluated on §2.5's D0–D6.

#### 3.2.1 Classical image-space corruption (landmark-driven warp/dup/erase)

- D0: ok (takes face image).
- D1 off-manifold reach: maximal — we directly place features off-manifold by construction.
- D2 monotonicity: easy by parameterizing warp magnitude with `y`.
- D3 identity preservation at y=0: trivially (the warp has amplitude 0).
- D4 leakage: minimal (background untouched if we mask).
- D5 controllability: deterministic.
- D6 cost: negligible.
- **Weakness:** D1's *subjective* half is the risk. Chernoff-style warps might look cartoonish-wrong rather than uncanny-wrong. Needs user study at low priority — but even if so, this is the cheapest working baseline and passes D0–D6 on the measurable side immediately.
- **Status:** viable baseline. Ship as the fallback drift mechanism.

#### 3.2.2 Cursed_LoRA_Flux + Eerie_horror (the running v3+Flux drift mechanism) — **measured as working**

**This is the production drift subsystem and it is already passing D1 + D2 at r = +0.914 on 543 jobs.** The approach: at generation time, apply `Cursed_LoRA_Flux` at strength = `sus_factor` *plus* `Eerie_horror` at strength = `0.75 × sus_factor`, where `sus_factor = (sus_level / 100.0) ** 0.8` compresses the 0–100 input range toward the top. Deterministic per `seed = hash(job_id) % 2^32`.

- D0: ok. Takes the Flux generation pipeline and modifies its LoRA stack at sampling time.
- **D1 off-manifold reach: measured passing.** r(ArcFace anchor distance, sus_level) = +0.914 on Flux v3, which is the strongest D1+D2 result of any tested generation pipeline on 543 real jobs. The high-sus cohorts reach anchor distances of 0.94–1.14 (vs. 0.57–0.70 for legitimate cohorts) — clearly off-manifold in identity space.
- **D2 monotonicity: measured passing.** r = +0.914 is a direct monotonicity measurement at the cohort level; ordering of per-cohort mean anchor distances is clean.
- D3 identity preservation at y=0: ok (sus_factor=0 gives LoRA strength 0, which collapses the LoRA contribution to identity).
- D4 leakage: unmeasured per-region but anecdotally some drift into pose/lighting at high strength.
- **D5 controllability/reversibility:** known **tuning problem** per `feedback_lora_uncanny_tuning.md`: the v8c LoRA curve overshoots into zombie / skin-lesion territory at high sus_factor. This is a curve-shape problem (need lower max strength or delayed ramp), not an architecture problem. The fix is to re-tune the strength function, not to replace the mechanism.
- D6 cost: known Flux fp8 txt2img at 20 steps (the current production config).
- **Status: measured baseline.** Any drift-candidate replacement must beat r=+0.914 on the same 543 jobs. The framework's D1+D2 cells for the drift subsystem are **filled with measured values**, not flagged as "to measure." Remaining drift work is: (a) tune the LoRA curve to stop overshooting into zombie territory at high sus_factor, (b) optionally decompose the sus axis into per-flavor sliders via §3.2.5.

#### 3.2.3 Animator-family failure-mode exploitation (LivePortrait / MMFA / DyStream / SuperHead)

- D0: ok (they take a reference portrait image).
- D1: **unverified and speculative** — the idea is to drive them with exaggerated motion signals so their OOD failure modes become the uncanny signal. No paper reports this, no animator author has tested it, we would need to run the experiment. This is why the tavily scan matters.
- D2: needs a mapping from `sus_level` to an exaggerated motion control; monotonicity is plausible but unverified.
- D3: at natural motion magnitude, the animator returns something close to the reference.
- D4: likely high leakage — animators couple motion and expression tightly.
- D5: deterministic within the animator.
- D6: 2026 animators range from real-time-capable (DyStream claim, unverified) to several seconds per frame.
- **Status:** speculative but interesting. Worth a small experiment after the classical and diffusion drift candidates are in place. Parked in the deeper-research queue until stage 1 is decided.

#### 3.2.4 Asyrp h-space editing with CLIP-directional "uncanny" vector

- D0: requires access to the diffusion backbone's h-space at a specific timestep range.
- D1: the Asyrp paper validates on unconditional DDPM/iDDPM only. Latent-DM / CFG backbones are explicit future work. Haas et al. (FG 2024) extends to diffusion face editing on single-prompt denoising. **Blocking open question same as §3.1.8:** stability under blended ConditioningAverage conditioning is untested. Resolved by the same ~1-day pre-flight.
- **Status:** research bet at stage-2-specific drift. Subordinated to §3.1.8 — if h-space direction finding works at all on blended Flux, it is available for both identity enhancement *and* drift axis simultaneously. The pre-flight answers both at once.

#### 3.2.5 Concept Sliders (ECCV 2024, Gandikota et al.) — the named next-step overlay

**LoRA weight-pair semantic directions in diffusion model parameter space.** Each concept is a pair of LoRAs: one increases the target attribute, the paired LoRA decreases it. Applied at Flux (or SDXL) inference time by adding the LoRA weights to the model with a signed strength scalar.

- **50+ sliders compose without quality degradation**, minimally affecting other attributes. Composability is tested and confirmed in the paper.
- **Text-or-image-defined concepts.** You can define "hollow practiced smile," "evasive gaze," "urgency pressure," "warmth drain" as separate named concepts and learn a direction per concept via the standard LoRA fine-tuning recipe, ~1–2h per concept on a V100-equivalent.
- **Preserves protected concepts.** Tested for race / gender spillover — adding an age slider does not drift ethnicity. This directly addresses D4 (collateral channel leakage) for Y-axes that should not touch identity.
- **For our use case:** replace the monolithic Cursed_LoRA + Eerie_horror dial with a composition `w_warmth · slider_warmth + w_pressure · slider_pressure + w_evasive · slider_evasive + w_hollow · slider_hollow`, each driven from a different component of `sus_factors` (the 16-d factor vector that already exists in the telejobs DB).
- **Flux status:** paper shipped **SDXL sliders**, Flux versions are not published as of 2026-04. We would need to train our own sliders using the standard Concept Sliders training recipe on Flux. Training cost estimate: 1–2h per concept × 4–8 concepts = 4–16 GPU hours total, one-time.

Scores:
- D0: ok (LoRA inference-time composition with Flux).
- D1 off-manifold reach: inherits v3+Flux baseline (already measured r=+0.914); the question is whether decomposition *preserves* that r under per-flavor assignment from sus_factors, which depends on sus_factors' relationship to the ground-truth fraud signal.
- D2 monotonicity: each slider is monotone by training construction (weight-pair definition).
- D3 identity preservation at y=0: ok by construction (all slider strengths = 0 returns base model).
- **D4 leakage: best of any candidate in the framework** — Concept Sliders' composability and protected-concept preservation are explicit training objectives, tested and validated on SDXL.
- D5: deterministic.
- D6: inference cost identical to current Flux pipeline; training cost one-time and bounded.
- **Status:** **strong recommended candidate** for the drift subsystem upgrade and the readable-Y decomposition layer, simultaneously. Surfaced on 2026-04-12 decision review, lost across three framework sessions, recovered 2026-04-14. This was the single most load-bearing insight the older docs contained that the framework was missing.

### Rubric shakedown conclusions under v0.3

- **The measured baseline (§3.1.6, Flux v3 anchor-bridge) is what other candidates must beat.** No longer "HyperFace A is cleanest on paper" — the correct sentence is "v3+Flux is cleanest on the 543-job corpus with r=+0.914, and any candidate that cannot beat this is not an improvement."
- **Drift subsystem is already built and measured.** §3.2.2 is the production drift mechanism with r=+0.914 D1+D2 correlation. The "mandatory drift α-sweep pre-flight" from the prior rebuild plan is **already done**, it just predated the framework's D-rubric formulation. The remaining drift work is LoRA curve tuning and optional per-flavor decomposition via §3.2.5 Concept Sliders.
- **Identity subsystem has one high-leverage unmeasured cell (§3.1.6 P4a Spearman ρ on baseline).** Running that measurement is a ~1-day task that uses existing data and tools (ArcFace IR101 against existing `dataset_faces_flux_v3/` output). It converts the baseline from "partially measured" to "fully characterized" at near-zero cost.
- **The one architectural experiment that moves everything forward is the h-space blended-conditioning pre-flight (§3.1.8 / §3.2.4).** ~1 day, zero training, existing infrastructure. If it passes, h-space directions become a cheap additive layer for both identity enhancement and drift-flavor decomposition, opening §3.1.8 / §3.1.7 as framework-preferred candidates. If it fails, the frontier path is §3.1.9 (Arc2Face + blendshape adapter, arXiv:2510.04706) which needs a paper read first.
- **StyleGAN3-FFHQ is eliminated on product grounds, not just Lipschitz grounds.** §3.1.2 is downgraded from "viable with measurement" to "eliminated as FFHQ, viable only as AffectNet fine-tune with 4–8 GPU hours training cost." The Lipschitz argument I over-committed to is no longer load-bearing here; the expression-range argument independently does the same work and was made on 2026-04-12.
- **FLAME is a stage-1.5 readable-Y editor, not an identity candidate** (§3.1.4). Its unique property — structural orthogonality — is most productively used as a round-trip editor via SMIRK/EMOCA inversion + re-render, not as an identity-channel substrate. The standalone FLAME identity path is blocked on the arXiv:2510.04706 read.
- **P4a and P13 remain the discriminators.** P4a especially, because the measured baseline has it as the one empty cell.

---

## 4. What the framework does not do

- **It does not pick a winner.** The recommendation from §3 is: v3+Flux is the measured baseline; §3.1.6 is the reference point; §3.1.7/§3.1.8/§3.2.5 are the improvement targets; experiments 1–4 in §5 move the grid from "partially measured" to "fully characterized." The framework gives a structured scoring apparatus, not an architectural verdict.
- **It does not replace training-cost / license / inference-latency concerns.** Those live in a second grid applied to framework-viable candidates. License data (FLAME CC-BY-4.0, Arc2Face CC-BY-NC, Flux Dev license, StyleGAN3 NVIDIA research) is captured per-entry in §3 but not weighted.
- **It does not handle the distribution-preservation question (P11).** If it becomes important it gets promoted from NA to target in a future revision.
- **It does not validate the algebraic-closure speculation** (§1.2, §P12) beyond marking it optional. The 2026-04-12 algebra research doc surveyed this and concluded the StyleGAN W-arithmetic paradigm is alive in 2025–26 via three modern substrates (h-space, 3DMM, disentangled flow matching); the framework incorporates this by treating the substrates as candidates (§3.1.8, §3.1.4/§3.1.9) rather than by demanding algebraic closure as a universal property.
- **It does not formalize uncanny-valley perception directly** — only the **mechanism** that pushes off-manifold. Whether the off-manifold push subjectively reads as "wrong" is a user study question downstream of D1 passing objectively. The framework's D1 two-part definition (classifier + subjective) makes this split explicit.
- **It does not reconcile against the pre-framework docs.** `docs/design/2026-04-14-vamp-rebuild-plan.md` and `docs/design/2026-04-14-deeper-research-queue.md` were written before the framework existed and are expected to contain over-commitments that the framework invalidates. The reconciliation is deferred until experiments 1–4 in §5 produce measurements; until then, those docs are advisory-only and the framework is the current authority.

---

## 5. Next steps (v0.5 — ordered by leverage, zero-training first)

All framework-recommended experiments use existing data and existing tools. None require training before a measurement result lands. Ordered from cheapest and most leverage-positive to most speculative.

1. **Measure P4a on the existing v3+Flux baseline — global ρ + local k-NN preservation + seed stability** (`output/dataset_faces_flux_v3/`). Script: load qwen embeddings for all 543 jobs, load existing ArcFace IR101 embeddings (or re-run `src/face_distinctness.py --model arcface`), compute pairwise distance matrices in both spaces. Report:
   - **Global:** Spearman ρ on the pair set, stratified by cohort.
   - **Local:** k-NN preservation rate at k∈{5, 10}; trustworthiness and continuity at k=10 via `sklearn.manifold.trustworthiness`.
   - **Seed stability:** regenerate a 30-job subset at 3 additional seeds; report ρ variance across seeds.

   **~1 day, zero training, zero new data.** Fills the §3.1.6 baseline scoring. Global ρ alone was the v0.4 plan; v0.5's critique-driven expansion adds local and stability halves. If global passes but local fails, we learn something important: v3 preserves cluster structure but tears local neighborhoods — fixable via editorial-channel refinement without replacing the identity channel.
2. **Run the h-space blended-conditioning pre-flight** (§3.1.8). Generate ~100 v3 faces at varying ConditioningAverage blend weights, capture h-space activations at a fixed timestep range, run PCA or learned-direction-search on the activations, check whether top-k directions remain stable across different blend weights. **~1 day, zero training, existing ComfyUI infrastructure.** Outcomes:
   - Pass ⇒ §3.1.8 (h-space direction injection) and §3.2.4 (Asyrp-style drift-axis direction) become cheap additive layers over v3; Concept Sliders becomes an optional refinement rather than a necessity.
   - Fail ⇒ §3.1.9 (Arc2Face + blendshape adapter) is the frontier path for Y-channel structural orthogonality; §3.2.5 (Concept Sliders) is the drift-refinement path.
3. **Re-run ArcFace IR101 on v8 manifold-face-gen output** (`output/` directory for manifold-face-gen phases 0/1/3) to discover whether v8 passed its phase-4 acceptance gate or not. The framework currently assumes v8 status is unknown. If v8 beat v3 on r(sus) or cluster separation, the baseline changes and §3.1.6 needs re-pointing. If v8 failed, the manifold-face-gen plan can be closed out. **~1 hour, existing tool, existing data.**
4. **Read arXiv:2510.04706** ("ID-Consistent, Precise Expression Generation with Blendshape-Guided Diffusion," ICCVW 2025). Confirm input shapes, checkpoint availability, training requirements, license. This is the specific unblock for §3.1.9 and for FLAME-as-identity candidate class generally. **~1 hour first-hand read.**
5. **Reconcile framework conclusions with `docs/design/2026-04-14-vamp-rebuild-plan.md` and `docs/design/2026-04-14-deeper-research-queue.md`.** Both were written pre-framework and likely contain StyleGAN3-primary over-commitments from yesterday's HyperFace pivot. The reconciliation is: either rewrite both docs against this framework, or replace them with pointers to this doc and the cross-references in Appendix B. Do not touch until experiments 1, 2, 3 have produced measurements — the decisions in those docs depend on results that don't exist yet.
6. **Run the P\* user study — downstream task accuracy.** 5–10 raters, three conditions (text baseline / v3 identity-only / v3 identity + editorial + drift). Tasks: (a) "which of these two jobs is more suspicious?" (b) "group these 10 jobs into clusters." Measure: accuracy vs. ground-truth labels; improvement delta per channel. **~1 day to set up, 1 day to run, 1 day to analyze.** This is the *ultimate* acceptance criterion — every proxy metric above is trying to predict this number. Do not defer indefinitely.
7. **Editorial ablation for E2** (information contribution). Re-run the P\* user study with a uniform base prompt (no work_type archetype differentiation), compare to the full-editorial condition. The P\* gap is the editorial channel's contribution. Without this, we cannot honestly answer "how much of v3's success is identity vs. editorial?" — we will be designing identity replacements blind to what they need to carry.
8. **Only after 1–7:** decide whether the per-flavor sus decomposition via §3.2.5 Concept Sliders is worth 4–16 GPU hours to train Flux-native sliders. If baseline r=+0.914 + P\* user study is adequate, Concept Sliders is a nice-to-have; if user testing reveals the sus axis is collapsing multiple readable dimensions into a single indistinguishable "wrong" signal, Concept Sliders becomes urgent.
9. **Framework validation against the user.** Ongoing. The specific open questions as of v0.5:
   - Does the three-channel restructure (§2.2) match how the user thinks about the pipeline, or should editorial fold back into identity?
   - Is the distortion-budget allocation principle (§1.5) — "continuity > trustworthiness, cluster boundaries > intra-cluster geometry" — the right default, or should different product goals reweight it?
   - Is the factor-mismatch drift reframing (§2.5) the right model, or does it over-theorize a mechanism that works empirically without factor decomposition?

---

## Appendix A — Glossary of properties

- **Lipschitz constant (L):** smallest L such that `d(f(a), f(b)) ≤ L · d(a, b)` for all `a, b`. Locally: measure L within a neighborhood.
- **Fisher ratio:** `(between-class variance) / (within-class variance)`. Adapted here to `(mean cross-cluster face-distance) / (mean within-cluster face-distance)`.
- **On-manifold / off-manifold:** a point is on the natural-face manifold if it lies in the high-density region of a reference face distribution (e.g., FFHQ). Off-manifold points are the uncanny-valley territory.
- **Injectivity at scale σ:** `d_X(a, b) > σ ⇒ f(a) ≠ f(b)`. Weaker than full injectivity; tolerates collisions among near-identical source points.
- **Disentanglement of a direction v:** moving along v in Z produces a change in the rendered face that is localized to one semantic feature, within tolerance, over a usable range of α.
- **Coded vs. uncoded channel:** coded channels are hand-assigned to named semantic variables (the Ys). Uncoded channels carry arbitrary residual information that distinguishes samples without being individually nameable.

---

## Appendix B — Cross-references

**Incorporated (framework v0.3 has read and folded in):**

- `docs/research/2026-04-07-final-findings.md` — 543-job measured baseline (ArcFace IR101), r(sus)=+0.914 on Flux v3, same-archetype collapse
- `docs/research/2026-04-07-metrics-research.md` — ArcFace IR101 as production metric
- `docs/research/2026-04-09-embedding-space-and-face-conditioning.md` — v2/v3 system description, CLIP-L pooled rank inversion, Regime C' interpolation
- `docs/research/2026-04-12-flame-technical-overview.md` — FLAME structural orthogonality, renderer options, SMIRK/EMOCA
- `docs/research/2026-04-12-embedding-to-face-latent-arithmetic-2026.md` — h-space algebra, Asyrp/Haas/Self-Discovering-Directions, arXiv:2510.04706
- `docs/research/2026-04-12-stylegan-vs-diffusion-decision-review.md` — 2026-04-12 decision to stay on Flux, FFHQ expression-range argument, Concept Sliders as next-step overlay
- `docs/research/2026-04-14-vamp-theory-constraints.md` — Statement 1 (HyperFace regime analysis), Statement 2 (continuity-vs-readability contradiction), channel split
- `docs/research/2026-04-14-rebuild-blind-alleys.md` — killed paths analysis
- `docs/research/papers/2026-04-14-paper-findings.md` — paper verification log
- `docs/research/2026-04-14-neural-portrait-animation-2026-scan.md` — 2026 drift-mechanism candidates

**Deliberately not incorporated (pre-framework; reconcile after experiments 1–4):**

- `docs/design/2026-04-14-vamp-rebuild-plan.md` — likely contains StyleGAN3-primary over-commitment from HyperFace/Lipschitz pivot; framework v0.3 invalidates via §3.1.2 product-level eliminator
- `docs/design/2026-04-14-deeper-research-queue.md` — Tier-1b reads list; should be re-prioritized against §5 experiments (arXiv:2510.04706 is now top priority)
- `~/.claude/projects/-home-newub-w-vamp-interface/memory/project_vamp_rebuild_primitives.md` — pre-framework primitives, superseded by `project_vamp_measured_baseline.md` memory

**Production code (reference, not authoritative for framework):**

- `src/face_distinctness.py --model arcface` — ArcFace IR101 measurement tool, with `transformers >=5.5` loading fix
- `src/generate_dataset.py --face-version 3 --flux` — v3+Flux generation pipeline
- `data/archetypes.json` — 10 archetypes × 2 variants T5 prompts
- `data/cluster_centroids.pt` — 42-cluster qwen centroids
- `data/clipl_centroids.pt` — archetype CLIP-L pooled centroids (known rank-inverted)
- `output/dataset_faces_flux_v3/face_distinctness_arcface.json` — baseline measurements
- `output/phase1/phase1_anchor.png` — drift-reference neutral anchor (seed=42)
