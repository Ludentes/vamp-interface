# Evaluating Manifold Transformations Between Qwen and T5 Embeddings

## Overview

This report describes how to evaluate the quality of a mapping between two high-dimensional embedding spaces, with a focus on transforming Qwen embeddings into T5 embeddings and, optionally, back again (Qwen → T5 → Qwen).
The primary example is a "human faces" manifold represented in Qwen and T5 embedding spaces, with the downstream goal of using T5-like embeddings to condition a generative model such as Flux.[^1][^2]

The report covers:

- Gromov–Wasserstein (GW) and related optimal-transport metrics as manifold-level similarity measures.[^3][^4][^5]
- Global structure metrics: Procrustes alignment error, regression loss, and representation-similarity metrics such as CKA and pairwise-distance correlations.[^6][^7][^8][^9][^10]
- Local topology metrics: k-nearest-neighbor (k-NN) overlap, trustworthiness, and continuity.[^11][^12][^13]
- Task-level metrics: cross-model face recognition performance and generative identity preservation for models like Flux.[^2][^14][^15][^1]
- How to interpret forward mappings (Qwen → T5) versus round-trip mappings (Qwen → T5 → Qwen) using cycle consistency.

The goal is to provide a practical toolbox and experimental design for diagnosing whether the mapping preserves the geometry and semantics of the face manifold and to understand how close the mapping is to an invertible transformation.

## Problem Setup

Assume two embedding spaces:

- Qwen space: embeddings \(e_Q \in \mathbb{R}^{d_Q}\) for faces.
- T5 space: embeddings \(e_T \in \mathbb{R}^{d_T}\) for the same faces (e.g., via a text encoder or a face encoder that is architecturally similar to T5).[^1][^2]

Given paired samples \((e_Q^{(i)}, e_T^{(i)})\) for many identities, the objective is to learn:

- A forward map \(f: Q \to T\) such that \(f(e_Q^{(i)}) \approx e_T^{(i)}\).
- Optionally, an inverse map \(g: T \to Q\) such that \(g(e_T^{(i)}) \approx e_Q^{(i)}\).

The core questions are:

1. How well does \(f\) align the Qwen face manifold with the T5 face manifold at the geometric and semantic levels?
2. Is the mapping approximately invertible on the data manifold, as seen via a round trip Qwen → T5 → Qwen?
3. Which metrics are most informative and correlate with downstream tasks (face verification, Flux generation)?

Cycle-consistency analysis is used throughout, by considering both one-step and round-trip transformations.[^16][^17][^18][^19]

## Gromov–Wasserstein Manifold Alignment

### Definition and Use

Gromov–Wasserstein (GW) distance compares two metric measure spaces using only their internal pairwise distances, without requiring a cross-space cost function or direct correspondences.
Given point sets in Qwen and T5 spaces with distance matrices \(D^{(Q)}, D^{(T)}\), GW finds a soft matching that minimizes discrepancy between these distance structures.[^4][^5][^3]

In the Qwen → T5 context:

- Treat Qwen embeddings and T5 embeddings as two metric spaces (using, for example, cosine or Euclidean distances).
- Compute GW between the mapped Qwen manifold \(f(Q)\) and the native T5 manifold \(T\).

A smaller GW distance indicates that the intrinsic geometry of \(f(Q)\) is closer to that of \(T\), which is useful even when the two spaces differ in dimensionality or global orientation.[^5][^3][^4]

### GW Under a Round Trip

With both \(f\) and \(g\) available, round-trip analysis considers Qwen → T5 → Qwen via \(g(f(\cdot))\):

- Compute GW between original Qwen embeddings and reconstructed ones: \(\text{GW}(Q, g(f(Q)))\).

Interpretation:

- If \(\text{GW}(f(Q), T)\) is small and \(\text{GW}(Q, g(f(Q)))\) is also small, then the Qwen and T5 manifolds are close to isometric on the sampled data, and \(f, g\) approximate a bi-Lipschitz equivalence.[^20][^3][^4][^5]
- If \(\text{GW}(f(Q), T)\) is small but \(\text{GW}(Q, g(f(Q)))\) is large, then the forward map produces a geometry that looks T5-like, but information necessary to reconstruct Qwen geometry is lost or difficult to recover; the mapping is many-to-one or too constrained for invertibility.[^3][^5][^20]
- If \(\text{GW}(f(Q), T)\) is large but \(\text{GW}(Q, g(f(Q)))\) is small, the composite map behaves almost like identity in Qwen space, yet the mapped manifold is still geometrically misaligned with the T5 manifold; this indicates differing intrinsic geometries that cannot be reconciled by the chosen parametric family for \(f\).[^5][^20][^3]

Because GW is a global manifold measure, it is computationally heavier but gives a broad view of how similar the spaces are as metric measure spaces.
It is especially valuable when only unpaired data are available, but with paired data it remains a strong sanity check for global geometry.[^4][^3][^5]

## Global Geometry Metrics Beyond GW

### Procrustes Alignment and Regression Loss

Classical manifold alignment often embeds datasets and then fits an optimal linear or orthogonal transform to align the coordinates.
Given matrices of aligned embeddings for corresponding points, Procrustes analysis chooses an orthogonal matrix (and optionally a scale factor) to minimize Frobenius norm \(\lVert XQ - Y \rVert_F\).[^7][^6]

In the Qwen–T5 setting:

- Fit \(f\) as a linear or affine map using paired embeddings.
- On a held-out set, compute alignment error:
  - Mean squared error (MSE): \(\lVert f(e_Q^{(i)}) - e_T^{(i)} \rVert_2^2\).
  - If \(f\) is restricted to orthogonal (or orthogonal + scale), report normalized Procrustes residual as a manifold-alignment quality measure.[^10][^6][^7]

Round-trip view:

- Forward residual: \(\lVert f(e_Q^{(i)}) - e_T^{(i)} \rVert_2^2\).
- Backward residual: \(\lVert g(e_T^{(i)}) - e_Q^{(i)} \rVert_2^2\).
- Cycle residuals:
  - \(\lVert g(f(e_Q^{(i)})) - e_Q^{(i)} \rVert_2^2\) (Qwen → T5 → Qwen).
  - \(\lVert f(g(e_T^{(i)})) - e_T^{(i)} \rVert_2^2\) (T5 → Qwen → T5).[^21][^20]

Patterns of interest:

- Small forward residual, large backward and large cycle residuals suggest Qwen → T5 is relatively easy but T5 → Qwen is lossy, implying that Qwen encodes less information or a different structure than T5 for the same faces.[^21][^1]
- Large forward residual, smaller backward residual suggests T5 space is easier to approximate from Qwen or that the available model class favors T → Q mapping.[^20][^21]
- Moderate forward and backward residuals but very small cycle residuals indicate both mappings use an implicit shared coordinate system; individually they may be imperfect in original coordinates, but together they behave almost as identity on the manifold.[^21][^20]

These behaviors are directly studied in cross-model embedding transformation work, where regression-based losses are combined with other constraints to enforce alignment.[^20][^21]

### Representation Similarity: CKA and Pairwise Distance Correlation

Representation-similarity measures quantify how similarly two embeddings represent relationships among a set of points without focusing on absolute coordinates.[^8][^9]
Common metrics include:

- Centered Kernel Alignment (CKA): compares kernel (Gram) matrices of embeddings.
- Pearson or Spearman correlation of vectorized pairwise distances or similarities.

Given paired data, compute:

- Kernel or distance matrix for mapped Qwen embeddings in T5 space, \(K_f\) or \(D_f\).
- Kernel or distance matrix for native T5 embeddings, \(K_T\) or \(D_T\).
- CKA(\(K_f, K_T\)) or correlation between entries of \(D_f\) and \(D_T\).[^9][^8]

Under round trip:

- Compute kernels or distances for original Qwen embeddings and reconstructed Qwen embeddings \(g(f(Q))\).
- Evaluate CKA or distance correlation between Q and g(f(Q)).

Expectations:

- If \(f\) and \(g\) are near-isometric inverses on the manifold, CKA(Q, g(f(Q))) and distance correlation between distance matrices should approach 1.
- If forward and backward mappings each weaken representational similarity, but cycle similarity remains high, this indicates that the composite mapping preserves relational structure while coordinate systems differ.

Recent work on comparing neural representations validates CKA and related measures as robust to invertible linear transforms, making them suitable for this type of analysis.[^8][^9]

### Distribution-Level Metrics (FID-Style and Wasserstein)

Another perspective is to treat the mapped Qwen embeddings and T5 embeddings as samples from distributions in the same space and compare them using distributional distances:

- Fit Gaussian approximations to T5 embeddings and mapped Qwen embeddings in T5 space and compute a Fréchet-like distance.
- Compute Wasserstein distance (not Gromov–Wasserstein) between the two empirical distributions in T5 space.

The same can be done for Q and g(f(Q)) in Qwen space.
These metrics are simpler than GW and lack explicit structural interpretation but are useful as sanity checks that the global distribution of mapped points matches that of native embeddings.[^13][^5]

## Local Topology and Neighborhood Preservation

### k-NN Overlap and Neighbor Stability

Local topology preservation is often crucial for face manifolds, where nearest neighbors reflect identity and fine-grained visual similarity.
Neighbor-based metrics are standard in manifold alignment and representation-quality evaluations.[^12][^11][^13]

For a given k (e.g., 10 or 20):

- For each point, compute its k nearest neighbors in the native target space (T) and in the mapped space (f(Q)).
- Measure overlap using Jaccard index or recall at k.

Under round trip:

- Compute k-NN sets in Qwen space and in reconstructed space g(f(Q)).
- Evaluate overlap between these neighbor sets.

Interpretation:

- High k-NN overlap in both forward and round-trip spaces indicates that local neighborhoods are preserved, aligning with the intuition of a well-preserved manifold.
- If forward mapping preserves neighborhoods moderately but the round trip exhibits much lower overlap, the inverse mapping introduces local distortions even if some global statistics are preserved.

### Trustworthiness and Continuity

Trustworthiness and continuity are classical measures to evaluate the quality of low-dimensional embeddings and manifold learning methods and can be repurposed here.
Trustworthiness penalizes points that appear as neighbors in the embedding but are far apart in the original space, while continuity penalizes the reverse.[^11][^12]

For Qwen → T5:

- Compute trustworthiness and continuity with Q as reference and f(Q) as embedding.

For round trip:

- Compute trustworthiness and continuity with Q as reference and g(f(Q)) as embedding.

High trustworthiness and continuity in both steps support the claim that the mapping is approximately topology-preserving.
If these scores drop significantly in the round-trip comparison, then the inverse map or the composition may be introducing topological artifacts such as folding or tearing of the manifold.[^12][^11]

## Task-Level and Semantic Metrics

### Cross-Model Face Embedding Compatibility

From a practical perspective, the goal is often to reuse a model’s scoring protocol (verification/identification) with embeddings originating in another model.
Recent work shows that embeddings from different face networks can be made compatible through simple linear mappings and evaluates these maps by standard face recognition metrics.[^14][^22][^2][^1]

For Qwen → T5:

- Train \(f: Q \to T\) using paired face embeddings.
- In evaluation, feed f(Q) into the T5-based recognition system and compute:
  - Verification metrics such as ROC-AUC, TAR at fixed FAR.
  - Identification metrics such as rank-1 accuracy and CMC curves.
- Compare performance to native T5 embeddings and to a baseline with no mapping.

For T5 → Qwen, perform the symmetric procedure with g(T) evaluated using the Qwen-based recognition system.

Under a round trip:

- Evaluate Q → T → Q by scoring g(f(Q)) in the Qwen system.
- Evaluate T → Q → T by scoring f(g(T)) in the T5 system.

If round-trip performance closely matches single-step performance, the composite mapping is not significantly degrading the information relevant for recognition.
If performance drops sharply, especially compared to relatively mild changes in geometry metrics, this indicates that the mappings distort semantic identity structure in a way that local or global geometric metrics do not fully capture.[^2][^14][^1]

### Generative Metrics for Flux and Similar Models

When mapped embeddings are used as conditioning for a generative model such as Flux, downstream evaluation can be based on identity and quality of generated images.
Cycle-consistency ideas have been applied to image–text alignment and cross-modal embeddings, with consistency used as a reward or regularization signal.[^18][^19][^15]

An evaluation protocol might be:

- For each face identity, obtain a reference image and corresponding native T5 embedding (via text description or an image encoder aligned with T5) and a Qwen embedding.
- Generate images with Flux conditioned on:
  - Native T5 embeddings.
  - f(Qwen embeddings).
  - Mapped embeddings after a round trip if relevant (e.g., Q → T → Q processed through a diagnostic path).
- Measure identity preservation by comparing generated images to ground-truth faces using a third-party face recognition model, reporting verification or similarity scores.

If Flux outputs conditioned on f(Q) match the identity-preservation quality of native T5 conditioning, then the mapping is adequate for the generative task.
If quality remains comparable even after a round trip, this is strong evidence that the mappings preserve the task-specific semantics of the face manifold.[^19][^15][^18]

## Cycle Consistency and What It Reveals

### Concept of Cycle Consistency

Cycle consistency requires that going from one domain to another and back recovers the starting point: for Qwen and T5 spaces, this is the requirement that \(g(f(e_Q)) \approx e_Q\) and \(f(g(e_T)) \approx e_T\) for data points on the manifold.[^17][^16][^19]

Cycle consistency is used in many cross-modal

---

## References

1. [Are Face Embeddings Compatible Across Deep Neural Network ...](https://arxiv.org/html/2604.07282v1) - Our findings reveal surprising cross-model compatibility: low-capacity linear mappings ... embedding...

2. [Are Face Embeddings Compatible Across Deep Neural Network ...](https://arxiv.org/abs/2604.07282) - Our findings reveal surprising cross-model compatibility: low-capacity linear mappings substantially...

3. [Gromov-Wasserstein Alignment of Word Embedding Spaces - arXiv](https://arxiv.org/abs/1809.00013) - In this paper, we cast the correspondence problem directly as an optimal transport (OT) problem, bui...

4. [Gromov-Wasserstein Alignment: Statistical and Computational ...](https://www.mathtube.org/lecture/video/gromov-wasserstein-alignment-statistical-and-computational-advancements-duality) - The Gromov-Wasserstein (GW) distance quantifies dissimilarity between metric measure (mm) spaces and...

5. [Structure-Preserving Multi-View Embedding Using Gromov ... - arXiv](https://arxiv.org/html/2604.02610v1)

6. [[PDF] Manifold Alignment using Procrustes Analysis - UMass ScholarWorks](https://scholarworks.umass.edu/bitstreams/73434785-959f-497a-a825-484e3e7a47f3/download) - In this paper we introduce a novel approach to manifold alignment, based on Procrustes analysis. Our...

7. [[PDF] Manifold Alignment using Procrustes Analysis](https://icml.cc/Conferences/2008/papers/229.pdf) - In this paper we introduce a novel approach to manifold alignment, based on Procrustes analysis. Our...

8. [1 Introduction - arXiv](https://arxiv.org/html/2510.22953v1) - Centered kernel alignment (CKA) is a popular metric for comparing representations, determining equiv...

9. [Deciphering Molecular Embeddings with Centered Kernel Alignment](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00837) - Centered kernel alignment (CKA) has emerged as a promising model analysis tool that assesses the sim...

10. [[PDF] A Survey of Cross-lingual Word Embedding Models](https://www.jair.org/index.php/jair/article/download/11640/26511/21826) - Mapping Methods. There are four types of mapping methods that have been proposed: 1. Regression meth...

11. [Practical Cross-modal Manifold Alignment for Grounded Language](https://ebiquity.umbc.edu/_file_directory_/papers/1039.pdf)

12. [Assessing and improving reliability of neighbor embedding methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC12125374/) - The authors introduce a statistical framework to detect intrinsic map discontinuities and improve th...

13. [Mind the Gap A Generalized Approach for Cross-Modal Embedding ...](https://arxiv.org/html/2410.23437v1) - Our approach emphasizes speed, accuracy, and data efficiency, requiring minimal resources for traini...

14. [IEEE TRANSACTIONS ON BIOMETRICS, BEHAVIOR, AND IDENTITY SCIENCE](https://publications.idiap.ch/attachments/papers/2022/Krivokuca_IEEET-BIOM_2022.pdf)

15. [[PDF] Improving Cross-Modal Retrieval With Set of Diverse Embeddings](https://openaccess.thecvf.com/content/CVPR2023/papers/Kim_Improving_Cross-Modal_Retrieval_With_Set_of_Diverse_Embeddings_CVPR_2023_paper.pdf) - The model is trained with the loss using our smooth-Chamfer similarity. (b) Details of our set predi...

16. [[1904.07846] Temporal Cycle-Consistency Learning - arXiv](https://arxiv.org/abs/1904.07846) - We introduce a self-supervised representation learning method based on the task of temporal alignmen...

17. [Cycle Consistency Mechanism - Emergent Mind](https://www.emergentmind.com/topics/cycle-consistency-mechanism) - Cycle consistency is a mechanism that constrains paired forward and backward mappings, ensuring that...

18. [CycleMatch: A Cycle-consistent Embedding Network for Image-Text Matching](https://oulurepo.oulu.fi/bitstream/handle/10024/30774/nbnfi-fe2020120399215.pdf;jsessionid=F4D804386772020E996F56F8C1797D84?sequence=1)

19. [[PDF] Cycle Consistency as Reward: Learning Image-Text Alignment ...](https://openaccess.thecvf.com/content/ICCV2025/papers/Bahng_Cycle_Consistency_as_Reward_Learning_Image-Text_Alignment_without_Human_Preferences_ICCV_2025_paper.pdf) - Given an image and generated text, we map the text back to image space using a text-to-image model a...

20. [A Unified Framework for Cross-Model Embedding Transformation](https://preview.aclanthology.org/landing_page/2025.acl-long.1237.pdf)

21. [[PDF] A Unified Framework for Cross-Model Embedding Transformation](https://aclanthology.org/2025.acl-long.1237.pdf) - This section examines the impact of different loss functions and architectural choices on Embedding-...

22. [[PDF] Towards Protecting Face Embeddings in Mobile Face Verification ...](https://publications.idiap.ch/downloads/papers/2022/Krivokuca_IEEET-BIOM_2022.pdf)

