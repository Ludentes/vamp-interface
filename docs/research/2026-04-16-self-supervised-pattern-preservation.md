---
status: archived
topic: archived-threads
summary: Combine VICReg (collapse prevention) + Spearman distance-matching + InfoNCE to preserve multivariate structure when projecting embeddings to face space.
---

# Self-Supervised Pattern Preservation for Cross-Modal Projection

**Date:** 2026-04-16
**Question:** Without labels, which objective most reliably forces Φ: Q → F to preserve the multivariate structure of Q, so that patterns recoverable from Q remain recoverable from Φ(Q)?
**Scope:** paired data `(q_i, f_i = encode_face(flux(P(q_i))))`, no axis labels, arbitrary unknown pattern set in Q.

---

## 1. TL;DR

**Use a weighted combination of (a) a decorrelating variance/covariance regularizer on F and (b) a distance-structure term between Q and F.** Concretely: **VICReg on F** (prevents the face encoder from collapsing any axis of variation) **plus a relational/distance-matching term** (Spearman on pairwise-distance matrices, or a soft Gromov-Wasserstein surrogate) **between Q and F**. If you can afford one more term, add **InfoNCE over the paired batch** to force alignment.

One-line justification: VICReg alone guarantees nothing about *which* structure is preserved; distance-matching alone can be satisfied by a degenerate embedding with correct-looking statistics; InfoNCE alignment alone maximizes MI but not geometry. The three together lower-bound MI, prevent collapse, *and* preserve pairwise structure — which is what "pattern-preserving" actually means operationally.

---

## 2. Comparison Table

| Objective | What it provably preserves | Assumptions | Differentiable | Cost |
|---|---|---|---|---|
| **InfoNCE** (van den Oord 2018; Chen 2020; Radford 2021) | Lower bound on `I(Q; F)` up to `log N` | Large batch / many negatives; positive pair is informative | Yes, cleanly | Low (batch dot-products) |
| **MINE** (Belghazi 2018) | Direct MI estimate (Donsker–Varadhan) | Needs separate critic network; unstable gradients | Yes, but noisy | Medium; notorious variance |
| **Barlow Twins** (Zbontar 2021) | Cross-corr matrix → I (decorrelated features, no collapse) | Redundancy reduction captures invariance | Yes | Low |
| **VICReg** (Bardes 2022) | Explicit variance floor + covariance off-diag → 0 + invariance | Same spirit as Barlow, decoupled | Yes | Low |
| **IRM** (Arjovsky 2019) | Features whose optimal classifier is invariant across envs | Requires *environments* (labels-lite) and a downstream risk | Yes (penalty) | Medium; fragile in practice |
| **IB** `min I(Q;F) − β I(F;Y)` (Tishby 2000; Alemi 2017) | Compresses Q, retains Y-relevant info | Needs Y | Yes with VIB | Not applicable here |
| **Distance-matrix Spearman/Pearson** | Rank/linear correlation of `d_Q` and `d_F` | Patterns are distance-revealed; smooth Φ | Pearson yes; Spearman needs soft rank | Very low |
| **Gromov-Wasserstein** (Mémoli 2011; Peyré 2016) | Isomorphism of metric-measure spaces (up to relabel) | Metric spaces, mass distribution fixed | With entropic regularization (Sinkhorn) | High (O(N²) or more) |
| **Nonlinear ICA / iVAE** (Khemakhem 2020; Hyvärinen 2019) | Identifies true latent factors up to permutation+scale | Auxiliary variable, conditional independence of factors | Yes (VAE) | Medium; strong assumptions |
| **CLIP-style alignment** (Radford 2021) | Same as InfoNCE with learned τ | Paired data; modality encoders frozen or trained | Yes | Low |

---

## 3. Per-Objective Notes

**InfoNCE.** Maximizes `log p(f_i | q_i) / Σ_j p(f_i | q_j)` over a batch. Proven to lower-bound `I(Q; F) ≤ log N + L_InfoNCE` (Poole et al. 2019, arXiv:1905.06922). In our setup the "views" are the *same underlying item* rendered in two modalities; this is exactly CLIP (arXiv:2103.00020). Strength: scales, easy gradients. Failure: with small batches, the bound is loose; it preserves *identifiability-of-pairs* but not necessarily geometry — two items can be perfectly distinguishable yet have a shuffled pairwise-distance structure.

**MINE** (arXiv:1801.04062). Trains a statistics network T to estimate `I = sup E_P[T] − log E_Q[exp T]`. Works but is numerically fragile; has been largely superseded by InfoNCE for representation learning. Not recommended unless you specifically need an MI scalar to report.

**Barlow Twins** (arXiv:2103.03230). Forces the `D×D` cross-correlation between two view-embeddings to equal identity: on-diag → 1 (invariance), off-diag → 0 (redundancy reduction). Elegant, no negatives needed. Failure mode: with anisotropic feature dimensions or mismatched scales across modalities, the identity target is hard to hit and the loss saturates.

**VICReg** (arXiv:2105.04906). Three terms: **V**ariance (each dim std ≥ 1), **I**nvariance (MSE between paired views), **C**ovariance (off-diagonal covariance → 0). Same goal as Barlow Twins but decoupled: easier to tune the collapse-prevention term independently of the alignment term. Has become a default in cross-modal work where one encoder is fixed.

**IRM** (arXiv:1907.02893). "Learn a representation whose optimal classifier is the same in every environment." Needs (i) multiple environments and (ii) a downstream risk. Our cross-modal alignment is not naturally an IRM setup — there is no environment variable. IRM is the right tool when you want Φ to ignore a *known nuisance*; it is not a generic pattern-preservation objective. Also, IRMv1's penalty is known to be a poor proxy (Rosenfeld et al., arXiv:2010.05761).

**Information Bottleneck.** `min I(Q; F) − β I(F; Y)`. With no Y, the objective reduces to `min I(Q; F)` which *removes* information and is the opposite of what we want. Confirmed: IB is inapplicable unlabeled. (It *is* applicable the moment you have any labels or auxiliary targets — §5.)

**Distance-matrix correlation.** Compute `D_Q[i,j] = ||q_i − q_j||`, `D_F[i,j] = ||f_i − f_j||`, then maximize Spearman or Pearson between the upper triangles. Theoretical content: Pearson = 1 iff `D_F = α D_Q + β`, i.e., Φ is an isometry up to scale. Spearman = 1 iff Φ preserves the *rank order* of distances (a weaker, more robust invariant). Cheap and scalar — good for monitoring. As a *loss*, differentiable Spearman needs a soft-rank operator (Blondel et al., arXiv:2002.08871). This is the right cheap proxy in §2.6.1 of the framework.

**Gromov-Wasserstein.** Directly optimizes metric-space matching: minimizes `Σ_{ijkl} |d_Q(i,j) − d_F(k,l)|² π_{ik} π_{jl}` over couplings π. Entropic GW (Peyré 2016, arXiv:1602.05441) is differentiable via Sinkhorn. This is the **theoretically cleanest** pattern-preservation loss — it penalizes every relational mismatch, not just nearest-neighbor — but the `O(N²)` cost limits batch size. Use it for final validation or on subsampled batches.

**ICA family.** Linear ICA is unidentifiable without non-Gaussianity. Nonlinear ICA was long considered impossible without further structure; Hyvärinen & Morioka (arXiv:1605.06336) and iVAE (Khemakhem et al., AISTATS 2020, arXiv:1907.04809) showed identifiability **up to permutation and element-wise transform** given an observed auxiliary variable u (e.g., time index, class, environment) such that sources are conditionally independent given u. Klindt et al. (arXiv:2007.10930) extended to temporal sparse coding.

**CLIP.** Practical recipe: symmetric InfoNCE with a learned temperature. Same theoretical content as InfoNCE; the lesson is engineering (temperature, batch size, normalization).

---

## 4. The Identifiability Story

The ICA-family results are the only ones that promise to **recover the true latent factors** of Q, not just a representation that preserves them up to some equivalence. The key insight (Khemakhem 2020): if latent sources `s` are conditionally independent given an auxiliary `u`, and the conditional `p(s|u)` belongs to an exponential family with sufficient variability in u, then observing `x = g(s)` lets you recover `s` up to permutation and element-wise rescaling. This is called **identifiability up to the equivalence class ~A**.

Does this apply to our cross-modal setup? **Partially, and only if we treat one modality as the auxiliary.** Gresele et al. (arXiv:1905.06642, "Incomplete Multi-View ICA") and Daunhawer et al. (arXiv:2303.09166, "Identifiability results for multimodal contrastive learning", ICLR 2023) showed that **paired multimodal data is a form of auxiliary variable**: if Q and F share latent content but have modality-specific style, contrastive learning recovers the *shared content block* identifiably, under an assumption that content and style are statistically independent. This is exactly our situation — shared content = "what makes a job-posting what it is", modality-specific style = text surface vs. face rendering. **This is a real theoretical win for InfoNCE-style alignment in paired cross-modal data** and is the strongest a-priori argument for contrastive loss in our pipeline.

Caveat: identifiability theorems assume the data-generating process has the assumed structure. When the face encoder discards information (e.g., ArcFace's identity bottleneck), the *recoverable* content block shrinks to whatever survives that bottleneck. That's a property of F, not of the loss.

---

## 5. Combined / Semi-Supervised Recipes

The graceful-degradation recipe: **one loss, three terms, three switches.**

```
L = λ_align · L_InfoNCE(q, f)              # always on
  + λ_struct · L_dist(D_Q, D_F)            # always on; Spearman or entropic GW
  + λ_collapse · L_VICReg(f)               # always on; prevents F collapse
  + λ_task · CE(classifier(f), y)          # only when labels exist
```

With labels (detective regime, planted axes): `λ_task > 0`, all four terms active; this is supervised contrastive + structure preservation. Without labels (real scam corpus): `λ_task = 0`; the three self-supervised terms still enforce (i) paired-MI, (ii) pairwise geometry, (iii) no axis collapse. This lets us train and evaluate on the *same loss form* across labeled and unlabeled regimes, and removes the "which objective did we optimize?" confound when comparing results.

Concept-Sliders-style editing (arXiv:2311.12092) can be layered on top as a downstream interpretability tool — it is orthogonal to the projection loss.

---

## 6. Open Questions

1. **Does VICReg's variance term conflict with Flux's conditioning-space prior?** Forcing unit variance on each face-encoder dim might push P into parts of CLIP-conditioning space Flux handles poorly. Needs an ablation.
2. **Soft-rank Spearman vs. entropic Gromov-Wasserstein** as the structure term: at what batch size does GW become affordable and does it actually recover more patterns on the detective benchmark?
3. **Daunhawer-style identifiability**: can we empirically estimate the "shared content dimension" between Q and F via CCA or SVCCA (Raghu 2017) to get an upper bound on pattern-recovery performance *before* training?
4. **IRM revisited**: if we treat domain (scam corpus vs. labor-market corpus vs. detective) as an environment, does an IRM penalty help Φ generalize across corpora? This is a legitimate use of IRM we have not yet explored.
5. **Subsampled GW as validation metric** even if not as loss: does GW distance between Q and Φ(Q) correlate with detective-benchmark top-1 accuracy? If yes, GW becomes our unsupervised model-selection metric.

---

## Key Citations

- van den Oord, Li, Vinyals (2018). *Representation Learning with Contrastive Predictive Coding*. arXiv:1807.03748
- Chen et al. (2020). *SimCLR*. arXiv:2002.05709
- Radford et al. (2021). *CLIP*. arXiv:2103.00020
- Belghazi et al. (2018). *MINE*. arXiv:1801.04062
- Poole et al. (2019). *On Variational Bounds of Mutual Information*. arXiv:1905.06922
- Zbontar et al. (2021). *Barlow Twins*. arXiv:2103.03230
- Bardes, Ponce, LeCun (2022). *VICReg*. arXiv:2105.04906
- Arjovsky et al. (2019). *Invariant Risk Minimization*. arXiv:1907.02893
- Rosenfeld, Ravikumar, Risteski (2020). *The Risks of IRM*. arXiv:2010.05761
- Tishby, Pereira, Bialek (2000). *The Information Bottleneck Method*. physics/0004057
- Alemi et al. (2017). *Deep Variational Information Bottleneck*. arXiv:1612.00410
- Peyré, Cuturi, Solomon (2016). *Gromov-Wasserstein Averaging of Kernel and Distance Matrices*. arXiv:1602.05441
- Mémoli (2011). *Gromov-Wasserstein Distances and the Metric Approach to Object Matching*.
- Blondel et al. (2020). *Fast Differentiable Sorting and Ranking*. arXiv:2002.08871
- Hyvärinen, Morioka (2016). *Unsupervised Feature Extraction by Time-Contrastive Learning*. arXiv:1605.06336
- Khemakhem et al. (2020). *Variational Autoencoders and Nonlinear ICA: A Unifying Framework* (iVAE). arXiv:1907.04809
- Klindt et al. (2020). *Towards Nonlinear Disentanglement in Natural Data with Temporal Sparse Coding*. arXiv:2007.10930
- Gresele et al. (2019). *The Incomplete Rosetta Stone Problem: Identifiability Results for Multi-View Nonlinear ICA*. arXiv:1905.06642
- Daunhawer et al. (2023). *Identifiability Results for Multimodal Contrastive Learning*. ICLR 2023, arXiv:2303.09166
- Raghu et al. (2017). *SVCCA*. arXiv:1706.05806
- Gandikota et al. (2023). *Concept Sliders*. arXiv:2311.12092
