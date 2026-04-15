# Math Foundations Synthesis — What vamp-interface will actually use

**Date:** 2026-04-14
**Inputs:**
- `2026-04-14-math-foundations-perplexity.md` (Perplexity deep research)
- `2026-04-14-math-foundations-tavily-raw.json` (Tavily pro research, JSON-wrapped)

**Purpose:** Flat "cite-and-use" list of the mathematical concepts, metrics, and tools identified by the two adjacent-literature passes. Filters out decorative or out-of-scope material. Target consumers: framework v0.4 §2.6, and any future paper read that needs a math anchor.

---

## 1. Rank preservation (P4a)

**Accepted canon:**

- **Trustworthiness and continuity** (Venna & Kaski 2001; Lee & Verleysen co-ranking matrix 2008). Rank-based neighborhood preservation metrics with scores in [0,1]. Trustworthiness penalizes *false* neighbors (invented closeness); continuity penalizes *torn* neighbors (lost original closeness). **They are Pareto-incompatible in general** (LMDS parameter traces the trade-off). Our framework should pick per-subsystem: identity subsystem cares about continuity (don't tear qwen clusters apart); drift subsystem is exempt.
- **Spearman ρ / Kendall τ** on distance matrices. Coarser than trustworthiness/continuity but directly usable as an acceptance floor. The framework's P4a soft floor `ρ ≥ 0.6` lives here.
- **DreamSim** (Fu et al., NeurIPS 2023, https://dreamsim-nights.github.io). Learned perceptual distance `D(x,x̃) = 1 − cos(fθ(x), fθ(x̃))` with ~96% human triplet agreement after fine-tuning. Candidate backbone for P4a *perceptual-side* distance. Compare against ArcFace IR101 on a subset — IR101 is identity-invariant, DreamSim is holistic. For our P4a (rank preservation), DreamSim may be more appropriate than IR101 because we care whether *perceived* face similarity tracks qwen similarity, not whether biometric identity tracks qwen.
- **LPIPS** (Zhang et al., 2018). Retained for D4 collateral leakage only.
- **PCC — Preserving Clusters and Correlations** (arXiv:2503.07609). Explicit optimization objective maximizing Pearson+Spearman between source and target distance matrices relative to reference points. If we ever *train* a projection, this is the loss.
- **Isotonic regression + Pool Adjacent Violators (PAV)**. O(n) post-hoc calibration of a scalar output to be monotone in an input. Directly applicable to D1/D2: if `ArcFace_anchor_distance vs. sus_level` has minor non-monotonic wiggles, PAV straightens them for free. Trivial to implement, zero training.

**Named but skipped:**
- **MPAD** (arXiv:2504.16335). ANN-retrieval-specific, not a general generative-model metric.
- **RankGAN / RankSRGAN**. Training-time ranker substitution. Interesting as a *mechanism* but we are not retraining the Flux backbone; this is a reference for future fine-tune work, not v0.4.

**Framework impact:** §2.3 P4a gets two concrete backbones (ArcFace IR101 and DreamSim). The co-ranking / trustworthiness-continuity split becomes an optional deeper diagnostic we can run if scalar ρ is ambiguous. PAV is added to §2.5 as the D2 monotonicity-enforcement post-processor.

---

## 2. Distribution preservation (P5)

**Accepted canon:**

- **Gromov-Wasserstein distance** (Mémoli 2011; Peyré & Cuturi). Coordinate-free discrepancy between metric-measure spaces; invariant to isometries. This is the correct P5 metric for heterogeneous-space alignment (qwen-1024 → ArcFace-512 or qwen → face-pixel-space). **SCOT** (Demetci et al. 2020) demonstrates GW on single-cell multi-omics — mathematically the same problem (align two spaces with no known correspondences). Applies to us even though we are not doing biology.
- **Brenier's theorem** (1991). For quadratic cost and absolutely continuous source measure, there exists a unique optimal transport map `T = ∇φ` with `T#µ = ν`. **This is our "a pushforward map exists" theorem** — the formal grounding for "P5 is achievable in principle." Citable in framework §1.5.
- **Kantorovich-Rubinstein duality.** `W1(µ,ν) = sup_{Lip(f)≤1} E_µ[f] − E_ν[f]`. **This is the bridge that rescues the demoted-Lipschitz discussion:** Lipschitz functions are *dual* to W1, so when we say "Lipschitz is a supporting diagnostic, not a core criterion," the honest reformulation is "we care about W1 distance, not the Lipschitz constant of individual maps." Citable in framework §1.5.
- **Sinkhorn / entropic OT** (Cuturi 2013). Makes OT/GW computationally tractable. Relevant if we ever compute W1 or GW distances on real corpus data — standard tool.
- **Unbalanced OT with marginal penalization** (ICML 2025). Handles mass-creation/destruction when supports differ. Potentially relevant for the same-archetype-collapse scenario where qwen has mass in regions our image generator can't cover.
- **Gromov-Monge Embedding** (arXiv:2311.01375). "Monotone generative modeling" — directly on topic. Should be read in full if we ever commit to a trained projection.

**Named but skipped:**
- **Modality gap / CLIP symmetry fixes** (Liang et al.). Orthogonal to our pipeline (we don't use CLIP alignment at the inference path).
- **WassersteinProcrustes hybrids**. Interesting framing but no concrete tool we'd run.

**Framework impact:** §2.6 gains a P5 definition anchored on GW distance. §1.5 gains Brenier and K-R as citations instead of hand-waving. The current framework does not have a P5 acceptance floor because we lack the tool to compute GW on our full corpus; adding it as "optional diagnostic, run if we ever suspect distribution collapse" is honest.

---

## 3. Cluster / class structure preservation (§1.5 Fisher upper-bound)

**Accepted canon:**

- **Fisher discriminant ratio / LDA** (Fisher 1936). `Fisher(projection) = between-class-variance / within-class-variance`. The framework's existing §1.5 argument ("Fisher ratio at work_type level is upper-bounded by the source space") is exactly this.
- **Information Bottleneck** (Tishby et al. 2000; Vera & Piantanida generalization-gap bounds). Formalizes the Fisher upper-bound as `I(Z;Y) ≤ I(X;Y)` — a hard information-theoretic ceiling on how much discriminative information about class labels can survive any map `X → Z`. **This is the citation for §1.5's "upper-bounded by the source" claim.**
- **Differentiable Information Bottleneck (DIB)** (CVPR 2024). Kernel-eigenvalue operationalization — no variational approximation. The tool that would let us *compute* `I(qwen; sus_cluster)` and `I(face; sus_cluster)` and check the ratio.
- **Supervised DR theory** — LOL (Vogelstein et al. 2021, Nature Comms), DLRPP, DROP-D. Families of projections with provable discriminability preservation under model assumptions. Background reading; not a direct tool for us.
- **Johnson-Lindenstrauss bounds.** Target dimension `m = O(ε⁻² log n)` preserves pairwise distances within `1±ε`. Theoretical grounding for our PCA step (qwen 1024 → PCA-whitened). Citable background.
- **Projection-cost sketches.** `O(k/ε²)` rows preserve rank-k projection cost. Related theoretical grounding.

**Framework impact:** §1.5 gains IB and Brenier citations. The Fisher upper-bound becomes a named theorem, not a heuristic. DIB is added to §5 as a possible experiment ("measure the mutual-information ratio on real data") but not a gating metric.

---

## 4. Off-manifold / uncanny formalization (D1, D4)

**Accepted canon:**

- **Typical-set reasoning** (Nalisnick et al. 2019 "Do Deep Generative Models Know What They Don't Know?"). Raw likelihood is a broken OOD score — flows assign *higher* likelihood to OOD data. The right notion is distance to the *typical set*, not pointwise density. **Framework impact:** our D1 "anchor distance" is a crude proxy; the principled version is typical-set distance in the image generator's own latent space.
- **Energy-based OOD** (Liu et al., and UCL UDL 2021 survey). Corrected density-based scores that handle the typical-set pitfall. Candidate operational OOD score for D1 beyond anchor-distance.
- **Uncanny valley as perceptual mismatch** (Kätsyri et al. 2015 *Frontiers in Psychology*; Diel et al. 2022 meta-analysis). Empirical support for "mismatch between cues is the eeriness driver," not raw human-likeness. **Framework impact:** D4 (collateral leakage) should be reframed — the *point* of D4 is not to minimize leakage but to ensure leakage is *mismatched* across channels (realistic texture + wrong geometry = uncanny; uniformly degraded = just ugly).
- **Human-aligned uncanny functionals** (thinkmind ACHI 2024 "Systematic Approach to Quantify UV"). Parameterized curves from psychophysics — monotone-in-parameter measurable versions of Mori's sketch. Candidate reference for D1 subjective-half study design.
- **Manifold-aware generative models** (arXiv:2503.03963, CVPR 2022 "Low-density regions"). Diffusion-map + score-model pipelines that give tangent/normal decomposition explicitly. Background reading; not directly applicable to frozen Flux.
- **CLIP embedding norm as variability predictor** (ACL 2024, "Words Worth a Thousand Pictures"; W1KP). Empirical: `||CLIP(prompt)||` strongly predicts perceptual variability of generations. Could be a **cheap D1-alternative OOD score** for prompt-conditioned generation. Untested for our pipeline.

**Named but skipped:**
- **COIL constrained VAE.** Mechanism for enforcing constraints in VAE latents. Not applicable to a frozen Flux pipeline.
- **BIPE (adaSVR).** Identity-preservation via singular-value rescaling. Plausible mechanism for the inverse problem (preserve identity under drift) but not our current bottleneck.

**Framework impact:** §2.5 gains a D1 footnote on typical-set vs. raw distance. §2.5 D4 gets reframed around mismatch, not uniform minimization. The D1 subjective-half study can cite ACHI 2024 for protocol design.

---

## 5. Glyph / channel allocation (semantic visualization)

**Accepted canon:**

- **Glyph design surveys** (Borgo et al. 2013; Fuchs et al. 2017 systematic review). Empirical: Chernoff-style face glyphs **perform worse** than simpler glyphs on most tasks. Validates our decision to drop readability (framework §2.2 stage-1.5 is empty).
- **Perceptual channel orderability** (Bertin retinal variables; Munzner; Edwards EuroVis 2016). Spatial position > length/angle > area > color for quantitative; distinct hues for categorical. Standard visualization canon.
- **MCDA framework for glyph design** (arXiv:2303.08554). Criteria: typedness, discernability, separability, comparability, searchability, learnability, memorability. Structured way to evaluate glyphs.

**Named but skipped:**
- **PaCMAP** — we already used this for v4, but it is not a semantic visualization theorem, it is a DR algorithm. Duplicate of existing knowledge.
- **Face-Based Glyphs Revisited** (FACS-structured glyphs, emoji glyphs). Interesting but we are not building a glyph — we are rendering photorealism. Dropping readability means we are not in this literature.

**Framework impact:** Nothing directly actionable. This whole strand confirms the dropping-readability decision made in `2026-04-14-vamp-theory-constraints.md`. §2.2 stage-1.5 continues to have zero tenants.

---

## 6. Tool catalog — "what we would actually run"

Concrete operational tools lifted from the two reports, ranked by leverage-to-cost:

| Tool | Source | Cost | What it measures / does | Framework slot |
|---|---|---|---|---|
| **Spearman ρ on distance matrices** | standard | trivial, existing scipy | P4a rank floor | §2.3 P4a (already in framework; IR101 and DreamSim as alternate backbones) |
| **PAV / isotonic regression** | sklearn | trivial | Post-hoc monotonicity enforcement on D2 | §2.5 D2 calibration step |
| **Trustworthiness / continuity** | sklearn `sklearn.manifold.trustworthiness` | trivial | Deeper P4a diagnostic if scalar ρ is ambiguous | §2.3 P4a optional sub-diagnostic |
| **DreamSim** | https://github.com/ssundaram21/dreamsim | small (preload model) | Perceptual-distance backbone alternative to LPIPS, 96% human agreement | §2.1 perceptual space note |
| **DIB (kernel-eigenvalue IB)** | CVPR 2024 repo | medium | Operational `I(X;Y)` proxy | §5 experiment (mutual-info ratio) |
| **GW distance via POT** | python optimal transport lib | medium (n² memory) | P5 distribution alignment metric | §2.6 P5 optional diagnostic |
| **Sinkhorn (entropic OT)** | POT | small | Tractable OT computation | Supporting tool for GW and W1 runs |
| **Typical-set / energy OOD score** | EBM lit | medium (need calibration) | Principled D1 beyond anchor-distance | §2.5 D1 alternative measurement |
| **CLIP embedding norm** | ACL 2024 W1KP | trivial | Cheap variability/OOD predictor | §2.5 D1 cheap proxy |
| **Gromov-Monge Embedding** | arXiv:2311.01375 | read-only for now | Reference for future trained projection | §5 reading list |

---

## 7. What the two reports did NOT resolve

Both reports flag these as open problems with no canonical answer in the literature:

1. **Finite-sample, neural-parameterized Brenier map estimation rates.** No clean theorem about how well a learned MLP approximates an optimal transport map in high dimension. Implication for us: any claim about distribution preservation by our `qwen → face` map is empirical, not theoretical.
2. **A differentiable Kendall-τ surrogate with generalization bounds.** Nobody has packaged rank preservation as a clean training loss with sample-complexity guarantees. Implication: if we ever train a projection, we will use PCC-style Pearson+Spearman as a proxy and accept that the theory is loose.
3. **A psychophysically validated uncanny functional tied to latent density.** Meta-analyses support the *shape* of the valley, but there is no tool that takes a face image and returns a calibrated `uncanny_score ∈ [0,1]` that correlates with human ratings. Implication: D1's subjective half remains a user study, not an automated metric.
4. **Formal channel-allocation theory for glyphs.** Nobody has proven which variables go on which channels under perceptual budget. Implication: we cannot rigorously decide what readable Y-channels to add later. (We currently have zero, so this does not bite.)

These are all on the "accept and move on" side of the framework, not the "block until resolved" side.

---

## 8. Citation bundle for framework v0.4

Minimal set of citations the framework should carry after the v0.4 pass:

- Venna & Kaski 2001 — trustworthiness, continuity
- Lee & Verleysen 2008 — co-ranking matrix
- Fu et al. NeurIPS 2023 — DreamSim
- Zhang et al. 2018 — LPIPS
- Mémoli 2011 — Gromov-Wasserstein
- Brenier 1991 — Monge optimal transport theorem
- Kantorovich-Rubinstein — W1 duality (textbook citation)
- Cuturi 2013 — Sinkhorn entropic OT
- Demetci et al. 2020 — SCOT (GW for multi-omics alignment, our analog)
- Tishby & Zaslavsky 2000 — Information Bottleneck
- Vera & Piantanida — IB generalization-gap upper bound
- Yan et al. CVPR 2024 — Differentiable Information Bottleneck
- Nalisnick et al. ICLR 2019 — typical set vs. likelihood for OOD
- Kätsyri et al. 2015 — uncanny valley perceptual-mismatch review
- Diel et al. 2022 — uncanny valley meta-analysis
- Fisher 1936 — linear discriminant
- Johnson-Lindenstrauss — pairwise-distance preservation bounds

Plus the existing framework citations (Asyrp ICLR 2023, Concept Sliders ECCV 2024, Arc2Face+blendshape ICCVW 2025, ArcFace IR101).
