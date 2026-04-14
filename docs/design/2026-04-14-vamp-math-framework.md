# vamp-interface Mathematical Framework v0.2

**Date:** 2026-04-14 (post-compact rigor pass)
**Status:** Draft v0.2. Inputs: user spec (late-session verbal dump), theory constraints doc (`docs/research/2026-04-14-vamp-theory-constraints.md`), blind-alleys doc, paper-findings log, 2026 portrait-animation scan.
**Purpose:** Before touching the rebuild plan again, formalize what vamp-interface actually wants in terms of spaces, maps, metrics, and channels; name the hidden inconsistencies in the informal spec; produce an evaluation grid that lets candidate pipelines be scored rather than asserted.

This document does *not* make architectural choices. It specifies the scoring rubric. The rewrite of the rebuild plan will instantiate the rubric over candidates.

### Changelog (v0.1 → v0.2)

- **Lipschitz demoted, distance-rank correlation promoted.** Lipschitz is a supporting diagnostic, not an acceptance criterion. The user's informal "similar→close, different→far" is a *relational* (order-preserving) property, not a metric one. Strict Lipschitz wrongly disqualifies discrete/step maps (HyperFace Regime A) that our §3.3 analysis already accepted. The required acceptance criterion is now **Spearman rank correlation** of pairwise distances (new P4a), complementing Fisher ratio (P4). See §1.5.
- **Two-stage architecture made explicit.** Identity channel (on-manifold, photorealistic face) and drift channel (off-manifold, uncanny corruption) are now formalized as **separate subsystems** scored on separate rubrics. P9 (off-manifold reach) is removed from the identity-candidate rubric entirely — identity candidates are not penalized for being on-manifold; that is their job. Drift is a second-stage pipeline applied to the identity output. See §1.7, §2.2, §2.5.
- **Neural-machinery weak-proof clause.** Having *any* working uncanny/off-manifold mechanism is itself a weak proof that the product engages the face-reading perceptual system (not just cartoon-icon recognition). This is a discriminator between "Chernoff with photorealism" and "vamp-interface." Recorded in §1.7.

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

### 1.5 The formal core: rank preservation + Fisher ratio (not Lipschitz)

Buried in the informal spec are **two** formally tractable requirements, and neither is Lipschitz:

**(i) Distance-rank preservation.** "Similar→close, different→far" and "semantically between ⟹ metrically between" are statements about the *ordering* of pairwise distances, not about absolute metric preservation. The right formalization is **Spearman rank correlation** between `{d_X(a_i, a_j)}` and `{d_P(f(a_i), f(a_j))}` over a sample of source pairs: if the ordering is preserved, similar pairs stay nearer than distant pairs, and the visualization reads correctly regardless of absolute scale. This property is scale-free and does not penalize discrete maps — a step function preserves rank at cluster granularity trivially.

**(ii) Fisher's discriminant ratio.** `(mean inter-cluster face-distance) / (mean intra-cluster face-distance) ≫ 1`, where "face-distance" is measured in whatever perceptual metric we commit to (LPIPS, ArcFace cosine, human rating). Measurable, optimizable, directly captures the "within-cluster different enough, between-cluster very different" requirement.

**Why not Lipschitz?** Lipschitz is a one-sided bound (`d(f(a), f(b)) ≤ L · d(a, b)`). It does not prevent collapse (a constant map is 0-Lipschitz), it does not preserve ordering, and it disqualifies discrete/step solutions (HyperFace Regime A has global Lipschitz constant ∞) that we already accept as viable. I was reaching for Lipschitz because "smooth" sounded formal; the formal property actually matching "similar→close, different→far" is rank correlation, not Lipschitz. Lipschitz stays in the framework as a **supporting diagnostic** — useful because papers report it natively and it composes cleanly per stage, but not an acceptance criterion.

**These two together are the first-class acceptance criteria for the identity channel.** Everything else is consequence, nice-to-have, or failure-mode guard.

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

---

## 2. The framework

### 2.1 Objects

A **candidate pipeline** is a tuple

    (X, d_X) →_{f_0} (Z_1, d_1) →_{f_1} ... →_{f_{n-1}} (Z_n, d_n) →_{render} (P, d_P)

where:

- `(X, d_X)`: **source space.** The embedding of a job posting. Today: qwen3-embedding:0.6b, 1024-d, cosine metric expected-valid. Eventually: possibly a raw 16-d sus_factors vector, or a structured concatenation.
- `(Z_i, d_i)`: **intermediate spaces.** Examples: PCA-whitened qwen, CLIP token space, StyleGAN W, ArcFace 512-sphere, FLAME (shape, expression, pose) parameters, SDXL UNet h-space.
- `(P, d_P)`: **perceptual/pixel space.** Two candidate metrics: pixel L2 (useless), LPIPS (OK for photorealistic), ArcFace-cosine-on-generated-face (for identity-channel monotonicity tests), or explicit human rating.
- `f_i`: the map at each stage. May be linear (PCA, affine), learned (MLP, LoRA, UNet), or hand-specified (hash-to-W, cluster-to-anchor).
- `render`: the final step that produces pixels from the last latent. For StyleGAN: synthesis network. For Arc2Face/InfiniteYou: diffusion sampling. For FLAME: a renderer.

A candidate pipeline is **scored** against the properties in §2.3.

### 2.2 Two-stage architecture: identity subsystem + drift subsystem

The pipeline is the **composition of two independently-scored subsystems**, plus optionally some readable (non-uncanny) Y-channels that ride on whichever subsystem supports them best:

**Stage 1 — Identity subsystem.** Maps source X to a photorealistic on-manifold face image I. Responsible for cluster membership visibility via rank-preserving similarity. Scored on identity-channel rubric (§2.3): rank correlation (P4a), Fisher ratio (P4), σ-injectivity (P6), composite distortion (P3 as diagnostic). **Not** scored on off-manifold reach.

**Stage 2 — Drift subsystem.** Takes (I, Y_drift) and produces a corrupted face I'. Responsible for triggering the face-reading perceptual machinery via controlled off-manifold push as a function of the uncanny-type Y (currently `sus_level`). Scored on drift-channel rubric (§2.5): off-manifold reach (D1), Y-monotonicity (D2), identity preservation at Y=0 (D3), collateral-channel leakage (D4), reversibility (D5). **Not** scored on rank correlation or Fisher ratio — those are stage-1 concerns.

**Optional stage-1.5 — Readable Y-channels.** For any Y that is supposed to be *readable* (name a face axis, e.g., "this is a formal job → suited/older face"), a learned or hand-coded direction is injected at whatever point in stage 1 supports disentanglement — typically StyleGAN W, FLAME params, or a LoRA direction. These ride on stage 1 and are scored on identity-compatible P7/P8/P10 properties (disentanglement, monotonicity, leakage). **vamp-interface currently has zero readable Y-channels committed** (the prior over-commitment to "qwen axis → face axis" is retracted; see theory constraints doc). This stage may be empty at launch and added later.

The two stages are engineered, trained, and evaluated independently. A final pipeline is a **pair** (identity-candidate, drift-candidate), with a joint end-to-end check on identity preservation at Y_drift=0 (i.e., stage 2 at rest does not visibly corrupt stage 1's output).

This split directly resolves tensions 1.5–1.7. It also explains why the 2026 portrait-animation literature matters: those papers are candidate **stage-2** mechanisms even though none of them is a candidate stage-1 mechanism.

### 2.3 Identity-subsystem rubric (stage 1)

Each property gets a status: **required** (candidate fails if absent), **target** (scored), **supporting** (diagnostic only, not gating), **optional** (bonus), **NA** (not measured this pass). **P0** is an interface gate — candidates that fail P0 never reach scoring.

| # | Property | Status | Measurement |
|---|----------|--------|-------------|
| **P0** | **Input-interface compatibility** — does the candidate accept our source X (possibly via a bounded-cost adapter)? | **required gate** | Inspection. Eliminates PhotoMaker, InstantID, RigFace, NoiseCLR, and all 2026-portrait-animator-family papers at the identity-subsystem level. |
| P1 | Source metric fitness — is `d_X` meaningful on our corpus (not just in general)? | **required** | Neighbor-retrieval sanity on a labeled subset; k-NN cluster purity vs. sus-label / scam-type. |
| **P4a** | **Distance-rank preservation (Spearman ρ)** — does the ordering of pairwise distances in X survive into face-distance? This is the formal version of "similar→close, different→far". | **required** | Sample N pairs from a labeled subset, compute `d_X` and face-distance (LPIPS or ArcFace-cosine), report Spearman ρ. Soft floor: ρ ≥ 0.6 for the identity subsystem to be acceptable. Scale-free by construction, so does not penalize discrete maps. |
| P4 | Fisher ratio — `(mean inter-cluster face-distance) / (mean intra-cluster face-distance)` for a held-out clustering on X | **required** | LPIPS or ArcFace-cosine on generated faces; soft floor `≥ 3`. |
| P6 | Identity σ-injectivity — distinct X points at `d_X > σ` map to distinguishable faces | **required** | Face-hash collision rate at chosen σ. |
| P2 | Per-stage Lipschitz bound — `d_{i+1}(f_i(a), f_i(b)) ≤ L_i · d_i(a, b)` | **supporting** | Empirical 95th-percentile pair-distance-ratio. Reported if measurable, used to bound P3 when end-to-end measurement is expensive, **not** an acceptance criterion. Discrete maps pass P4a + P4 + P6 without a finite Lipschitz constant, and that is fine. |
| P3 | Composite distortion (was: composite Lipschitz) — end-to-end spread of face-distance at fixed `d_X` | **supporting** | Line sweep in X, LPIPS dispersion on output. Diagnostic only — P4a is the acceptance criterion, P3 tells us *how* a failing P4a is failing (collapse vs. explosion vs. noise). |
| P5 | Betweenness / geodesic approximation | **optional** | Triangle-inequality-tightness on curated triples. Optional because modern manifolds rarely satisfy this. |
| P7 | Disentangled direction support for *readable* Y-channels (stage 1.5) | **target** per direction | Empirical, per direction per candidate. |
| P8 | Y-channel monotonicity for readable Ys | **required per committed readable Y** | α-sweep. Currently vamp-interface has **zero committed readable Ys**, so this line has no tenant this pass. |
| P10 | Readable-Y leakage into identity | **target** | Paired generation with readable-Y varied, identity inputs fixed. |
| P11 | Support preservation vs. gap-filling | **NA this pass** | Revisit if a candidate's behavior forces the question. |
| P12 | Algebraic closure (local) | **optional** | Single-point test. Not blocking. |
| **P13** | **Verification status** — for every cell, is the answer (a) measured on our corpus, (b) inherited from a paper's claim on a different corpus, or (c) assumed? | **required metadata** | Every cell tagged `{measured / inherited / assumed}`. |

**Note on P9.** The old P9 ("Y-channel off-manifold reach") has been removed from the identity-subsystem rubric entirely. It now lives in §2.5 as drift-subsystem property D1. Identity candidates are not penalized for being on-manifold; that is their job.

P13 is the discipline hook. Last session's failures came from treating "the paper says X" as "X is true for our use" — exactly the inherited-vs-measured conflation. The grid forces that distinction explicit.

### 2.4 What counts as "passing" (identity subsystem)

A candidate is **viable as an identity subsystem** if:
- It passes P0 (interface).
- It passes all required identity-channel properties: P1, P4a (rank correlation), P4 (Fisher ratio), P6 (σ-injectivity).
- P13 tagging is filled in for every scored cell.

It is **preferred** over another viable candidate if it scores higher on **target** properties, with P4a and P4 weighted highest (they *are* the identity-channel definition). Supporting properties (P2/P3) inform diagnosis but do not move the ranking by themselves. Optional properties break ties only.

A candidate failing P4a or P4 on the identity channel is dead for that role. It may still be viable as a drift subsystem (§2.5) or as a readable-Y-channel mechanism — evaluated on its own rubric there, not retroactively rehabilitated into the identity rubric.

### 2.5 Drift-subsystem rubric (stage 2)

A drift subsystem takes `(I, y) ∈ P × ℝ` and produces `I' ∈ P`, where `I` is a photorealistic face from stage 1, `y` is the uncanny-type Y (currently `sus_level ∈ [0, 1]`), and `I'` is the corrupted output displayed to the user.

| # | Property | Status | Measurement |
|---|----------|--------|-------------|
| **D0** | **Input-interface compatibility** — accepts a face image (and optionally a scalar control, or we can add a thin wrapper that converts y to the native control, e.g., motion magnitude or prompt strength) | **required gate** | Inspection. |
| D1 | Off-manifold reach — as `y → max`, does the output `I'` leave the natural-face distribution in a way a human detects as "wrong"? | **required** | FFHQ-realism classifier score (or ArcFace-distance to nearest FFHQ neighbor) decreases monotonically with `y`; **and** a small-scale user rating study confirms the "wrong but I can't say why" effect at high `y`. The classifier alone is necessary but not sufficient — the subjective half is the whole point of the product. |
| D2 | Y-monotonicity — perceived wrongness increases monotonically in `y` across the full range | **required** | α-sweep, this is the Asyrp/drift pre-flight restated for stage 2. LPIPS from I (at y=0) monotone in y. |
| D3 | Identity preservation at y=0 — `I' ≈ I` when `y = 0` | **required** | LPIPS(I', I) < ε at y=0. Without this, stage 2 corrupts the stage-1 work before any signal arrives. |
| D4 | Collateral-channel leakage — varying `y` does not shift pose, lighting, background, or readable-Y features (if any exist) beyond tolerance | **target** | Paired generation with `y` varied, measure region-wise LPIPS on non-face regions. |
| D5 | Controllability / reversibility — the mapping `(I, y) → I'` is deterministic and well-defined across the whole `y` range | **target** | Repeated generation with same (I, y) produces same output; sweep continuity on y. |
| D6 | Cost and latency compatible with pre-generate-and-cache | **target** | Wall-clock per frame; target < 5s on local GPU. |
| **D13** | Verification status tag for every cell | **required metadata** | Same discipline as P13. |

**Passing a drift candidate.** A drift subsystem is viable if it passes D0, D1, D2, D3. D4/D5/D6 are ranking factors. D1 is explicitly **two-part** — the classifier measurement is necessary but the subjective rating is the ground truth. A drift mechanism that fools the classifier without firing the human face-reading response has failed.

**The weak proof.** A drift candidate that satisfies D1 (subjective half) is itself weak evidence that vamp-interface engages the face-reading perceptual system rather than decorative icon-recognition. This makes D1 doubly important: it is both the drift-subsystem acceptance criterion *and* the product's answer to "is this a Chernoff-plus-photorealism or something more?"

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

#### 3.1.2 Qwen → PCA → StyleGAN3 native W

- P0 interface: ok via an MLP adapter qwen→W.
- P1 source: ok.
- P4a rank correlation: plausible — StyleGAN3 native W is locally Lipschitz by explicit design, and a well-trained adapter should preserve rank. **Unverified on our corpus.**
- P4 Fisher ratio: unverified.
- P6 σ-injectivity: W is locally injective under natural training.
- P7 disentanglement (for optional readable Ys later): strong — 10–20 known semi-disentangled directions from SeFa/GANSpace/StyleSpace analyses.
- P2 diagnostic: paper-inherited (StyleGAN3 is designed to be Lipschitz in W).
- **Status:** identity-viable pending measurement. Strong bonus on P7 for future readable-Y-channel extension. Photorealism: FFHQ-dependent. License: NVIDIA research.

#### 3.1.3 Qwen → cluster-index → HyperFace Regime A (one face per cluster)

- P0 interface: ok (clustering + lookup).
- P1 source: ok.
- **P4a rank correlation: passes trivially at cluster granularity.** Within-cluster pairs all have face-distance 0, between-cluster pairs all have face-distance `≥ min_pairwise_angle`. Spearman ρ is 1.0 when ties are handled correctly (or undefined-but-fine; the interpretation is "monotone at cluster scale"). Under the v0.1 framework this candidate looked disqualified by Lipschitz; under v0.2 it is the cleanest pass on the entire rubric.
- **P4 Fisher ratio: maximal by construction.** HyperFace packs anchors to maximize the minimum pairwise angle.
- P6 σ-injectivity: passes if σ ≥ cluster radius; fails for sub-cluster resolution. User decides whether this matters for their use case (it does for the analyst scenario, may not for the scam hunter).
- P2 diagnostic: globally ∞ (step function). v0.2 treats this as OK because P2 is supporting, not gating.
- **Status:** identity-viable, arguably the cleanest candidate. Inherits the original vamp-interface fixed-anchor thesis as a special case. Main open question: cluster granularity N, which trades off P6 (finer N = better sub-cluster resolution) against visual distinctness (coarser N = cleaner Fisher).

#### 3.1.4 Qwen → MLP → FLAME (shape, expression, pose)

- P0 interface: ok via MLP adapter.
- P1 source: ok.
- P4a rank correlation: plausible; FLAME parameter space is explicitly disentangled and Euclidean-meaningful, so a well-trained adapter preserves rank. Unverified.
- P4 Fisher ratio: achievable by construction (FLAME's expression axes are independent).
- P6 σ-injectivity: passes under natural training.
- P7 disentanglement (for readable Ys): **strongest of all candidates** — FLAME is designed for this.
- **Note on drift:** FLAME renders only valid human faces within its prior, so it is a **bad fit for the drift subsystem** — but that is no longer a penalty at the identity-subsystem level. It just means stage 2 is a different model.
- **Status:** identity-viable, strongest on P7 for future readable-Y-channel extension. Photorealism: depends on renderer (worst if using a classical rasterizer, better if paired with a neural renderer). License: permissive (FLAME is research-only for model weights but well-distributed).

#### 3.1.5 Known-failed-at-P0 candidates (PhotoMaker, InstantID, RigFace, NoiseCLR, 2026 portrait animators)

All eliminated at the **P0 identity gate**: none of them accepts an abstract source embedding. They take face images, landmarks, or CLIP text tokens. They are **not** retroactively rehabilitated as identity candidates under v0.2. They are, however, candidate **stage-2 drift subsystems** — see §3.2.

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

#### 3.2.2 Diffusion img2img with "wrong face" prompt (the v8 LoRA approach)

- D0: ok.
- D1: validated by the existing v8c LoRA work — it pushes off-manifold. User feedback noted overshoot into zombie/skin-lesion (see `feedback_lora_uncanny_tuning.md`). **Tuning problem, not an architecture failure.**
- D2: proven monotone in denoising strength; just needs a softer curve at the top.
- D3: at strength=0 the image is returned unchanged.
- D4: some prompt-leakage into pose/lighting at high strength.
- D5: seed-deterministic, reversible via strength.
- D6: known (SDXL img2img ~2–4s/image on local GPU).
- **Status:** viable, currently in best working-order of any candidate for stage 2. The v8 LoRA is already this.

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
- D1: the Asyrp paper validates on unconditional DDPM only. Latent-DM generalization is explicit future work. **Risk.**
- **Status:** research bet, not a near-term candidate. Remains on the blind-alleys reopened list.

### Rubric shakedown conclusions under v0.2

- **HyperFace Regime A is now the cleanest identity candidate on paper**, not because it is technologically sophisticated but because it is the most honest about what the framework actually demands. The Lipschitz demotion was the change that unblocked it.
- **StyleGAN3 native W and Qwen→Arc2Face remain strong identity candidates** with measurement pending; they beat HyperFace A only on sub-cluster P6 and on P7 (disentanglement for future readable Ys).
- **FLAME is a compelling dark horse for identity** because it's the only candidate that would also give us first-class readable Y-channel support if we decide to add one later. Photorealism is the open question.
- **Drift subsystem is decoupled.** The current v8 LoRA (classical-plus-diffusion hybrid) is a working drift pipeline right now — the rebuild does not need to replace it to make progress on stage 1. This decoupling removes an artificial constraint that was blocking the rebuild plan.
- **P4a and P13 are the discriminators**, same as before. But under v0.2 they are asking cleaner questions than P3 (Lipschitz) was asking under v0.1.

---

## 4. What the framework does not do

- It does not pick a winner. That requires actually populating the grid, which requires the residual Tier 1b reads (FEM, InfiniteYou, LoRA weight-space editing) and hands-on experiments.
- It does not replace training-cost / license / inference-latency concerns. Those live in a second grid applied only to framework-viable candidates.
- It does not handle the distribution-preservation question (P11). If it becomes important it gets promoted from NA to target in a future revision.
- It does not validate the algebraic-closure speculation (1.2 / P12) beyond marking it optional.
- It does not formalize uncanny-valley perception directly — only the **mechanism** that pushes off-manifold. Whether the off-manifold push subjectively reads as "wrong" is a user study question downstream of P9 passing.

---

## 5. Next steps

1. **Validate this framework against the user.** Do the tensions named in §1 match the user's intent? Is the channel split (identity + Y) the right spine? Are any required properties (P1, P4, P6, P8, P9) actually optional for some users? (E.g., the student scenario may not need P6 below cluster radius.)
2. **Populate the grid over all serious Step-2 candidates.** Grid axes: candidate × {P0–P13}, cells tagged {measured on our corpus / inherited from paper / assumed}. This is the rebuild-plan rewrite's core artifact.
3. **Run the continuity pre-flight** for the top-ranked identity-channel candidate. This is the first measurement that moves P4 from "unverified" to "measured" for any candidate. Until it runs, *no candidate has a measured P4*, and picking between them is premature.
4. **Define σ.** The within-cluster scale. Pick it from the clusterability analysis of the corpus, not from the face-space side. Everything downstream depends on it.
5. **Fold the tavily orphaned request into the deeper-research queue**: 2026 neural portrait animation papers (PersonaLive, FantasyTalking2, UniTalking, DyStream, Hallo3/4, EMO/FLOAT successors). These are candidate **drift mechanisms** in this framework (Y-channel side, not identity side), so they belong in the Y-channel evaluation once identity is settled.
6. **Only then** rewrite `docs/design/2026-04-14-vamp-rebuild-plan.md` and the primitives memory.

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

- Theory constraints doc (last session): `docs/research/2026-04-14-vamp-theory-constraints.md`
- Blind alleys: `docs/research/2026-04-14-rebuild-blind-alleys.md`
- Paper findings: `docs/research/papers/2026-04-14-paper-findings.md`
- Not-yet-rewritten rebuild plan (target for next pass): `docs/design/2026-04-14-vamp-rebuild-plan.md`
- Not-yet-rewritten primitives memory: `~/.claude/projects/-home-newub-w-vamp-interface/memory/project_vamp_rebuild_primitives.md`
