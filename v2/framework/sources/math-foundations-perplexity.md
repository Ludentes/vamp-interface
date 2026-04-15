# Rank- and Distribution-Preserving Maps for Generative Interfaces: Adjacent Mathematical Foundations
## 1. Overview and Mapping to P4a/P5
This report surveys five strands of literature adjacent to a vamp-interface–style framework that cares about (i) rank-preserving image/latent maps and (ii) distribution-preserving manifold alignment, especially for evaluating and constraining generative models.
It focuses on established fields that already define metrics or theorems about neighborhood preservation, manifold/distribution alignment, class-structure preservation, semantic glyph design, and off‑manifold/uncanny behaviour.

Where useful, P4a is interpreted as a “soft floor” on rank preservation (relative order of neighbors/distances is not allowed to degrade beyond a tolerance), and P5 as distribution preservation (pushforward of the source distribution stays close to a reference distribution in an appropriate sense).

The key questions for each strand are:

- What are the core metrics/theorems?
- Are there ready-made substitutes for P4a/P5?
- Are there additional framework-level concepts that seem missing from the current axioms?

***
## 2. Rank-Preserving Embeddings and Neighborhood Quality
### 2.1 Trustworthiness, Continuity, and Co-ranking
Venna & Kaski introduced **trustworthiness** and **continuity** as rank-based measures of local neighborhood preservation for nonlinear projections and manifold learning.
Trustworthiness penalizes points that appear as neighbors in the embedding but were not close in the original space, while continuity penalizes original neighbors that are lost in the embedding.[^1][^2]
Formally, for a given neighborhood size \(k\), trustworthiness \(T(k)\) compares input-space and embedding-space ranks and lies in \([0,1]\); an implementation is standard in scikit-learn.[^1]

Lee & Verleysen and follow-ups recast trustworthiness/continuity in terms of a **co-ranking matrix**, from which additional quality measures such as mean relative rank error and local continuity meta-criteria can be derived.[^3]
These metrics separate different failure modes (creating false neighbors vs. tearing apart true neighbors) and can be evaluated as a function of \(k\), giving a curve rather than a single scalar.

Venna et al. later propose precision–recall–like measures over ranks and show how different nonlinear mappings trace different trade-offs between trustworthiness and continuity, effectively parameterizing a **Pareto frontier of neighborhood preservation**.[^4][^5]
Local multidimensional scaling (LMDS) explicitly introduces a tunable parameter that trades trustworthiness against continuity, highlighting that any projection must compromise between the two.[^4]
### 2.2 Rank- and Neighbor-Preserving DR post‑2020
Recent DR work continues to treat neighborhood and order preservation as first-class objectives rather than side effects.
A 2025 DR review emphasizes neighborhood-preservation metrics (trustworthiness, continuity) and cluster consistency as stability diagnostics for embeddings.[^6]

MPAD (Maximum Pairwise Absolute Difference, 2025) is especially relevant: it is an unsupervised DR algorithm explicitly designed to preserve **nearest-neighbor rank structure** for approximate vector search, prioritizing the ordering of true \(k\)-NN over non-neighbors rather than global distance fidelity.[^7]
MPAD includes a geometric analysis with bounds describing how distortion accumulates under non-uniform scaling and shows that its embedding preserves topological neighborhood information up to affine transformations.[^7]

These works collectively show that:

- Rank preservation can be directly optimized rather than implicitly targeted via Lipschitz or isometry constraints.
- There are established, tunable metrics for neighborhood quality that separate input→embedding and embedding→input errors.
- Theoretical bounds exist for how much rank structure can be preserved under perturbations and embeddings (e.g., MPAD’s condition-number-based bound).
### 2.3 Relevance to P4a (Rank-Preservation Soft Floor)
Trustworthiness and continuity already implement soft floors on rank preservation: one can demand \(T(k) \ge \tau_T\) and \(C(k) \ge \tau_C\) for relevant \(k\), or bound the area under their curves.[^1][^2][^3]
The co-ranking-framework-derived metrics (e.g., local continuity meta-criterion, mean relative rank error) give more fine-grained control over where rank errors are allowed (short vs. long range).[^3]
MPAD contributes an explicit optimization objective that maximizes separation between neighbor and non-neighbor rank-classes, and provides bounds on how much this ordering can deteriorate under distortions.[^7]

Taken together, these offer near drop-in, mathematically grounded substitutes for P4a:

- Use **trustworthiness/continuity or co-ranking–based metrics as explicit constraints or penalties** on mappings.
- Adopt **MPAD-style objectives** to train projections or conditioning maps that directly preserve nearest-neighbor ranking for the specific task (e.g., retrieval or conditioning neighborhoods), rather than relying on generic Lipschitz bounds.


***
## 3. Manifold Alignment and Distribution-Preserving Maps (P5)
### 3.1 Gromov–Wasserstein and Optimal Transport Alignment
Optimal transport (OT) and especially Gromov–Wasserstein (GW) OT provide a natural formalism for aligning datasets that live in different metric spaces by matching their **intrinsic distance structures**.[^8][^9][^10]

Key elements:

- **Gromov–Wasserstein distance:** compares two metric-measure spaces by finding a coupling that makes pairwise distances as consistent as possible; this is inherently distributional and structure-preserving across heterogeneous domains.[^9][^8]
- **SCOT** (Single-Cell alignment using Optimal Transport) applies GW OT to align single-cell multi-omics datasets without known correspondences, preserving local geometry across modalities; the GW distance itself acts as an unsupervised alignment and model-selection criterion.[^10][^9]
- Recent multi-view embedding work (e.g., Multi-GWMDS, 2026) explicitly uses GW-based couplings to produce **structure-preserving multi-view embeddings**, with experiments showing improved robustness to view heterogeneity and better preservation of manifold geometry than Euclidean or correlation-based baselines.[^8]

From the P5 perspective, GW distance is an OT-based measure that is explicitly **invariant to isometries** and cares about consistency of distance relations, not pointwise coincidence.
This provides a formal notion of “distribution-preserving” when mapping between heterogeneous embedding spaces.
### 3.2 Joint Embeddings via Unbalanced OT and Wasserstein–Procrustes
Recent work at ICML 2025 proposes a joint metric space embedding for unsupervised alignment of heterogeneous datasets using **unbalanced OT with GW marginal penalization**, proving existence of minimizers and convergence toward minimizers of an embedded Wasserstein distance as penalization parameters grow.[^11]
This formulation blends GW structure preservation with marginal regularization, giving control over how much mass can be created or destroyed when aligning datasets with different supports or densities.

Wasserstein–Procrustes and related hybrids combine OT with Procrustes analysis, simultaneously optimizing a transport plan and an orthogonal transform (or more general alignment map), with applications to joint visualization, domain adaptation, and graph/protein alignment.[^12]
These methods effectively yield maps that are **approximately isometric up to rigid transformations** while respecting distributional structure.
### 3.3 Cross-Modal Alignment Beyond CLIP
CLIP-style contrastive models align image and text representations by minimizing distances for matched pairs and maximizing them for mismatched pairs.[^13]
However, analyses of CLIP-like models reveal a persistent **modality gap**: the centers of the image and text subspaces are offset and the spaces remain only partially symmetric, meaning that cross-modal distances are not perfectly aligned even at converged training.[^13]

Recent work explicitly formalizes this gap as the Euclidean difference between modality centers and introduces loss augmentations with in-modal and cross-modal symmetry terms that enforce more symmetric, co-located embedding spaces.[^13]
ICLR 2025 work on “closing the modality gap” proposes a modality-agnostic framework that yields perfectly aligned multimodal representations in theory and demonstrates improved cross-modal retrieval and reconstruction performance in practice.[^14]

LIFT (Language-Image alignment with Fixed Text encoder) shows that fixing an LLM-based text encoder and training only an image encoder can achieve or surpass CLIP on compositional tasks, suggesting that **alignment to a strong semantic manifold (LLM) plus a constrained map from images to that manifold** can be an effective alignment strategy.[^15]
### 3.4 Relevance to P5 (Distribution Preservation)
OT/GW-based methods give ready-made P5 substitutes:

- **P5 as a GW bound:** require that the mapping between source and target embedding spaces achieves GW distance below a threshold, effectively constraining how much structural distortion is allowed.[^8][^9][^10]
- **Unbalanced OT with marginal penalties:** captures cases where supports or densities differ but structural alignment is still desired, using the penalty weights as an explicit “distribution preservation tolerance.”[^11]
- **Embedded Wasserstein distance:** convergence results for joint embedding schemes offer guarantees that, in the limit of strong penalization, joint embeddings preserve both marginal and relational structure in a Wasserstein sense.[^11]

Cross-modal work adds another angle: **modality gap metrics** such as center distance between modality-specific embeddings and symmetry measures can be viewed as distribution-preservation constraints in a shared latent space.[^14][^13]
These metrics could serve as P5-style regularizers on text→image latent conditioning maps, ensuring that conditioning-transformed latents do not drift into modality-specific subspaces in a way that breaks semantic symmetry.

***
## 4. Cluster-Preserving Projection and Fisher/Information Bounds
### 4.1 Fisher Discriminant Ratio and LDA as Optimal Projections
Fisher’s Linear Discriminant (FLD) and Linear Discriminant Analysis (LDA) maximize a **Fisher discriminant ratio**: the ratio of between-class variance to within-class variance after projection.[^16][^17][^18]
This can be expressed as a generalized Rayleigh quotient, with optimal projection directions given by eigenvectors of \(S_W^{-1} S_B\), where \(S_W\) is within-class scatter and \(S_B\) is between-class scatter.[^19][^16]

Interpreting FLD/LDA as supervised dimensionality reduction emphasizes that there is an optimal subspace (within the class of linear projections) that preserves class discrimination; projecting further or differently necessarily reduces Fisher ratio and thus linear separability.[^17][^18]
### 4.2 Supervised DR with Theoretical Guarantees
Recent work on supervised DR for big data introduces **Linear Optimal Low-Rank Projection (LOL)**, which incorporates class-conditional moments into the projection and is shown to preserve discriminative information with statistical guarantees at very high dimensions.[^20]
LOL is designed so that, under model assumptions, the low-dimensional representation retains sufficient information for near-optimal classification performance, effectively providing a lower bound on discriminability given the projected dimension.[^20]

Other methods such as DLRPP (Discriminative Low-Rank Preserving Projection) combine low-rank representation, discriminant analysis, and manifold regularization; here, the projection is optimized jointly with a low-rank representation to preserve both global structure and local discriminative information.[^21]

DROP-D and related orthogonal-projection-based DR for discrimination carefully analyze subspaces that remove within-class variance while preserving between-class structure, again emphasizing that there are **information-theoretically special subspaces** for classification.[^22]
### 4.3 Information Bottleneck and Mutual-Information Limits
The Information Bottleneck (IB) principle formalizes trade-offs between compression \(I(X;Z)\) and relevance \(I(Z;Y)\) for representations \(Z\) of input \(X\) with respect to targets \(Y\).[^23][^24][^25]
IB-based analyses derive **upper bounds on generalization error** and characterize phase transitions in learnability as the bottleneck multiplier \(\beta\) varies, showing that if \(\beta\) is too large, the trivial representation (independent of \(X\)) becomes optimal and discriminability collapses.[^24][^23]

More recent “decodable information bottleneck” variants explicitly connect mutual information to the capacity of downstream classifiers, providing bounds that link representation compression to attainable classification accuracy under specific decoder families.[^25]
### 4.4 Relevance to Cluster-Preserving Projections
For a vamp-interface-like system, these results point to two complementary kinds of constraints:

- **Fisher-style class-structure preservation:** require that downstream maps (e.g., from one latent space to another) do not reduce Fisher discriminant ratios below a threshold for certain class partitions, effectively imposing a cluster-preservation soft floor.
  This can be done by supervising on proxy labels (domains, known classes) or pseudo-labels (clusters) and constraining post-map Fisher ratios to stay within a factor of pre-map ratios.[^16][^17][^20]
- **IB-style information limits:** treat the map as a bottleneck and require that \(I(Z;Y)\) (or a variational lower bound) not fall below a specified fraction of \(I(X;Y)\), which bounds how much discriminative information about classes/clusters can be lost.[^24][^25]

While few works provide closed-form global upper bounds on discriminability loss under arbitrary nonlinear maps, the Fisher/LOL/IB literature strongly supports the idea that **class structure is fragile under dimensionality reduction and can be quantified in terms of Fisher ratios and mutual information**.
These notions can be folded into the framework as explicit cluster-preservation axioms or diagnostics, adjacent to P4a/P5 but focusing on label- or cluster-conditional structure.

***
## 5. Semantic Visualization and Glyph Design (Chernoff Successors)
### 5.1 Glyph Design Surveys and Empirical Results
Glyph-based visualization encodes multivariate data into small graphical marks (glyphs) whose visual channels map data dimensions to visual variables; Chernoff faces are the classic example using facial features.[^26][^27]
A systematic review of 64 glyph user studies catalogues glyph types, tasks, and results, highlighting trade-offs among accuracy, speed, and memorability, and noting that Chernoff faces and similar face-based glyphs often yield poor accuracy and slow responses compared to alternatives.[^27][^26]

Comparative experiments show that **spatial layouts and simple glyphs** often outperform complex face-like glyphs for tasks such as binary data comparison or multivariate scalar field analysis, emphasizing that human perception favors certain visual channels and layouts for different tasks.[^28][^29]

Recent “Face-Based Glyphs Revisited” and emoji-glyph work refine facial glyphs using the Facial Action Coding System and controlled channel assignments, achieving better discrimination and leveraging pre-attentive processing of emotional expressions.[^30][^31]
### 5.2 Perceptual Channel Allocation and Orderability
Research on perceptual orderability and effectiveness of visual channels (Bertin’s retinal variables, Ware, Munzner and others) shows that channels differ in their **perceived order**, discriminability, and suitability for quantitative vs. categorical data.[^32][^33]
Crowdsourced experiments demonstrate that channels like luminance/value are perceived as more ordered and are better for tasks involving ordered data, while hue is less ordered and better suited to categorical distinctions.[^34][^32]

Design frameworks and teaching materials explicitly rank visual channels by their **discriminating capacity** and support guidelines such as: spatial position > length/angle > area > color for quantitative comparisons; distinct hues and shapes for categorical differences; and careful avoidance of “crosstalk” where multiple dimensions interfere perceptually.[^35][^33][^36]

A 2024 multi-criteria decision analysis (MCDA) framework for glyph design formalizes glyph evaluation using criteria like typedness, discernability, separability, comparability, attentional importance and balance, searchability, learnability, and memorability.[^37]
This provides a structured way to choose which variables to map to which perceptual channels under dimensionality and salience constraints.
### 5.3 Relevance to Semantic Visualization in the Framework
From a vamp-interface perspective, semantic visualizations or control panels for generative models can be treated as glyph design problems under **perceptual information-budget constraints**.
The glyph literature suggests:

- There is a **quasi-axiomatic ranking of visual channels** in terms of discriminability and orderability, which can be encoded as constraints on which latent semantics may be assigned to which channels.[^32][^33]
- MCDA-style frameworks give a principled way to trade off variables when the glyph has limited “slots” (e.g., position, color, size), effectively acting as a perceptual analogue of P4a/P5 where the preserved structure is not metric but **human-readability and semantic salience**.[^26][^37]
- Successors to Chernoff faces (action-based face glyphs, emotion glyphs, emoji glyphs) demonstrate how to leverage pre-attentive perception for multivariate semantics, suggesting possible designs for high-density latent controls that remain interpretable.[^30][^31][^27]

These concepts are not direct replacements for P4a/P5 but suggest a parallel layer: **perceptual allocation constraints** on how semantic axes of a model are exposed to humans, which could be formalized using channel rankings and MCDA criteria.

***
## 6. Off-Manifold Generation and Uncanny-Valley Formalization
### 6.1 Manifold-Aware Generative Modeling
Score-based and diffusion models often learn data supported on low-dimensional manifolds embedded in high-dimensional spaces, raising questions about how they behave off-manifold.[^38][^39]
Analyses of score-based generative models show that as noise decreases, the learned score field mixes samples along the manifold while denoising with normal projections off the manifold, suggesting that these models implicitly reconstruct a manifold-like structure and restrict off-manifold excursions.[^39]

Recent manifold-aware generative frameworks combine diffusion maps with score-based models to sample densities on manifolds: diffusion maps find a low-dimensional latent manifold; a score-based model samples densities in that latent space; and geometric harmonics or Nyström extensions lift samples back to ambient space.[^40][^38]
This pipeline explicitly encodes the idea that valid samples lie near a learned manifold and provides tools for estimating density and tangent vs. normal directions.
### 6.2 Density-Based OOD and Low-Density Generation
Density-based OOD detection for deep generative models has revealed that naive likelihood often fails as an OOD indicator, with flows assigning higher likelihoods to OOD data due to issues like mismatch between high-density regions and the “typical set.”[^41]
Energy-based models and methods that adjust for background statistics or input complexity attempt to correct this, yielding better uncertainty estimates and OOD detection.[^41]

Diffusion-based approaches have been proposed to specifically generate high-fidelity samples from **low-density regions of the data manifold**, using modified sampling strategies that bias towards regions underrepresented in the training data.[^42]
These methods explicitly identify low-density regions and show that standard models undersample them, while targeted diffusion sampling can populate them without obvious visual artifacts.[^42]
### 6.3 Empirical and Theoretical Work on the Uncanny Valley
The uncanny valley describes a non-monotonic relationship between human-likeness and affinity, with a dip (the valley) where almost-but-not-quite human entities evoke discomfort.[^43][^44]
Empirical meta-analyses and reviews suggest that a simple “more human-like → more eerie” curve is not universally supported; instead, **perceptual mismatch** between cues (e.g., realistic texture with inconsistent motion, or mismatched facial features) is a strong driver of uncanny responses.[^45][^44]

A 2022 meta-analysis of uncanny valley experiments reports consistent evidence for valley-shaped affinity curves in some conditions and supports perceptual mismatch as a key mechanism; it also argues that movement and categorization difficulty play roles but are not sole determinants.[^46][^45]
More recent work proposes systematic methods to **quantify uncanny valley effects**, using controlled stimulus sets and rating scales to map human-likeness vs. eeriness, effectively turning Mori’s sketch into parameterized curves.[^47]

Parallel strands in face perception and deepfake detection show that modern generative faces can be indistinguishable from real faces for human observers, with biases toward judging images as real and strong effects of attractiveness and familiarity on detection performance.[^48][^49][^50]
These results suggest that current generators are often on the “far side” of the valley for many faces, but also that **small off-manifold defects** in certain attributes may still trigger uncanniness.
### 6.4 Relevance to Off-Manifold and Uncanny Metrics
For a generative interface concerned with off-manifold behaviour:

- **Distance-to-manifold and density-based scores:** manifold-aware generative frameworks and diffusion maps naturally yield estimates of distance to the learned manifold or density level sets, which can be used as monotone-in-parameter measures of “off-manifoldness.”[^38][^39]
- **Likelihood/energy corrections for OOD:** energy-based models and adjusted density methods suggest how to define OOD scores that avoid the pitfalls of raw likelihood, potentially providing better proxies for “unrealness.”[^41]
- **Uncanny curves as human-aligned metrics:** empirical valley curves (human-likeness vs. eeriness) can be fitted for specific tasks or populations and used as **ground truth functions** against which automatically computed features (e.g., mismatch between texture realism and geometric consistency) are regressed.[^46][^45][^44]

In other words, off-manifold generation and uncanny-valley behaviour can be formalized via a combination of **manifold distance/density** and **perceptual mismatch metrics** trained on human ratings.
This yields plausible candidates for monotone-in-parameter “uncanny” scores in the framework.

***
## 7. Framework-Level Concepts Potentially Missing
Pulling across strands, several mathematical concepts appear adjacent to—but not always explicit in—the current vamp-interface-style axioms:

1. **Co-ranking and Pareto trade-offs between trustworthiness and continuity.**  
   The co-ranking matrix and trustworthiness/continuity curves provide a principled way to describe where in the rank lattice the interface chooses to be faithful, and where it allows distortion.[^2][^4][^3]
   Encoding P4a in terms of acceptable regions in the co-ranking plane ("no more than ε mass in certain error quadrants") might be more precise than scalar rank-correlation thresholds.

2. **Metric-measure-space alignment via Gromov–Wasserstein.**  
   Framing P5 as a GW bound between metric-measure spaces (latent manifolds) gives a coordinate-free notion of distribution preservation that is robust to heterogeneity and changes of parametrization.[^8][^9][^10]
   This is stronger and more geometrically principled than, say, KL divergence between marginal distributions after an arbitrary embedding.

3. **Supervised discriminability preservation (Fisher/LOL) and information bottlenecks.**  
   Beyond unsupervised rank and distributional constraints, cluster/class structure can be constrained via Fisher ratios and mutual information, giving a third axis that bounds how much discriminative information about particular partitions can be lost.[^16][^24][^20][^25]

4. **Perceptual-channel allocation and MCDA for glyphs.**  
   When the interface exposes latent directions as visual controls, there is a parallel optimization problem over assigning semantics to visual channels under perceptual information budgets; this suggests importing MCDA frameworks and channel-orderability theory as first-class constraints.[^26][^32][^37][^33]

5. **Manifold-aware generative geometry and typical-set reasoning.**  
   Manifold/diffusion work plus energy-based OOD analysis suggests that off-manifold behaviour should be defined with respect to the **typical set** and the learned manifold, not just pointwise density.[^38][^39][^41]
   This can lead to more robust definitions of off-manifold perturbations and controlled excursions for exploration.

6. **Human-aligned “uncanny” functionals.**  
   Uncanny valley meta-analyses and face perception experiments point towards constructing an explicit functional \(U(h, m)\) over human-likeness features \(h\) and mismatch features \(m\) that predicts eeriness, learned from human data; such a functional could be used as a regularizer when exploring extreme regions of the generative manifold.[^46][^45][^48][^44]

***
## 8. Ready-Made Substitutes for P4a and P5
### 8.1 P4a (Rank-Preservation Soft Floor)
The following constructs can serve as direct or near-direct substitutes for P4a:

- **Trustworthiness and continuity constraints:** impose lower bounds \(T(k) \ge \tau_T\), \(C(k) \ge \tau_C\) across a range of \(k\), or constraints on the integrated curves; these explicitly bound the number and severity of rank violations in local neighborhoods.[^1][^2][^3]
- **Co-ranking–based error budgets:** define permissible mass in co-ranking matrix regions corresponding to particular types of rank errors (e.g., allow some long-range distortions but few short-range ones).[^3]
- **MPAD-style rank objectives:** train projection/conditioning maps using losses that maximize margin between neighbor and non-neighbor ranks, with theoretical bounds on distortion under transformations.[^7]
- **Rank-based correlations for generative evaluation:** Spearman/Kendall correlations between distance matrices of real vs. generated embeddings can be seen as coarser rank-preservation metrics; these can be complemented by trustworthiness/continuity for a full picture.
### 8.2 P5 (Distribution Preservation)
Similarly, P5 can be grounded in existing distributional alignment concepts:

- **Gromov–Wasserstein distance thresholds:** require that mappings between latent spaces or between data and latent space keep GW distance below a preset tolerance, capturing structure-preserving distribution alignment even across heterogeneous spaces.[^8][^9][^10]
- **Embedded Wasserstein/OT with marginal penalties:** unbalanced OT formulations with convergence guarantees provide knobs for mass-creation/destruction tolerance while still aligning geometry; penalty parameters become interpretable P5 hyperparameters.[^11][^12]
- **Modality-gap and symmetry metrics in multimodal spaces:** treat center distance between modality-specific embeddings and symmetry constraints on similarity matrices as additional distribution-preservation metrics in shared latent spaces, particularly for text↔image conditioning.[^14][^13][^51]

These substitutes are mathematically mature, with both optimization formulations and theoretical properties, and can be composed with Fisher/IB and perceptual/glyph constraints to yield a multi-layered vamp-interface framework grounded in well-established fields.

---

## References

1. [trustworthiness — scikit-learn 1.8.0 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.trustworthiness.html) - Jarkko Venna and Samuel Kaski. 2001. Neighborhood Preservation in Nonlinear Projection Methods: An E...

2. [[PDF] Manifold learning with approximate nearest neighbors](https://www.monash.edu/business/ebs/research/publications/ebs/wp03-2021.pdf) - Trustworthiness & Continuity (T&C). Venna & Kaski (2006) defined two quality measures, trustworthine...

3. [[PDF] Quality assessment of nonlinear dimensionality reduction based on ...](http://proceedings.mlr.press/v4/lee08a/lee08a.pdf) - Venna and S. Kaski. Neighborhood preservation in nonlinear projection methods: An experimental study...

4. [[PDF] Local multidimensional scaling - Department of Computer Science](https://research.cs.aalto.fi/pml/papers/wsom05-nn.pdf) - We propose a new method, local MDS, which is a derivative of CCA with the ability to control the tra...

5. [venna10a.dvi](https://jmlr.org/papers/volume11/venna10a/venna10a.pdf)

6. [Comprehensive review of dimensionality reduction algorithms - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12453773/) - Stability can be quantified using metrics such as trustworthiness, continuity, neighborhood preserva...

7. [MPAD: A New Dimension-Reduction Method for Preserving Nearest ...](https://arxiv.org/html/2504.16335v1) - Section 2 reviews classical and modern dimensionality reduction techniques, with emphasis on their l...

8. [Structure-Preserving Multi-View Embedding Using Gromov ... - arXiv](https://arxiv.org/html/2604.02610v1) - In this work, we propose two geometry-aware multi-view embedding strategies grounded in Gromov-Wasse...

9. [Gromov-Wasserstein optimal transport to align single-cell multi ...](https://www.biorxiv.org/content/10.1101/2020.04.28.066787v2.full) - We present Single-Cell alignment using Optimal Transport (SCOT), an unsupervised learning algorithm ...

10. [Gromov–Wasserstein based optimal transport to align ...](https://mlcb.github.io/mlcb2020_proceedings/papers/74_Demetci_etal_2020.pdf)

11. [Joint Metric Space Embedding by Unbalanced Optimal Transport ...](https://icml.cc/virtual/2025/poster/46678) - We propose a new approach for unsupervised alignment of heterogeneous datasets, which maps data from...

12. [UNSUPERVISED MANIFOLD ALIGNMENT WITH](https://arxiv.org/pdf/2207.02968v1.pdf)

13. [Understanding and Comparing Latent Space Characteristics of Multi ...](https://www.journalovi.org/2024-humer-amumo/)

14. [ICLR Closing The Modality Gap Enables Novel Multimodal Learning ...](https://iclr.cc/virtual/2025/36848) - In medical multimodal learning, our method enhances alignment between radiology images and clinical ...

15. [Language-Image Alignment with Fixed Text Encoders - arXiv](https://arxiv.org/html/2506.04209v1) - We propose to learn Language-Image alignment with a Fixed Text encoder (LIFT) from an LLM by trainin...

16. [Fisher Discriminant Ratio - an overview](https://www.sciencedirect.com/topics/computer-science/fisher-discriminant-ratio) - Fisher's Discriminant Ratio is defined as a criterion in linear discriminant analysis that aims to m...

17. [Dimensionality reduction and classification with Fisher's linear ...](https://dfdazac.github.io/04-fisher-example.html) - In this notebook we will deal with two interesting applications of Fisher’s linear discriminant: dim...

18. [Fisher's Linear Discriminant: Intuitively Explained](https://towardsdatascience.com/fishers-linear-discriminant-intuitively-explained-52a1ba79e1bb/) - LDA uses Fisher's linear discriminant to reduce the dimensionality of the data whilst maximizing the...

19. [An illustrative introduction to Fisher's Linear Discriminant](https://sthalles.github.io/fisher-linear-discriminant/) - In this piece, we are going to explore how Fisher's Linear Discriminant (FLD) manages to classify mu...

20. [Supervised dimensionality reduction for big data - Nature](https://www.nature.com/articles/s41467-021-23102-2) - We introduce an approach to extending principal components analysis by incorporating class-condition...

21. [Discriminative low-rank preserving projection for dimensionality ...](https://www.sciencedirect.com/science/article/abs/pii/S1568494619305496) - As an effective image clustering tool, low-rank representation (LRR) can capture the intrinsic repre...

22. [[PDF] dimension reduction by orthogonal projection for discrimination - HAL](https://hal.science/hal-01251257/document) - This approach uses orthogonal projection in order to clean some of the within-class variability from...

23. [The Role of the Information Bottleneck in Representation Learning | Semantic Scholar](https://www.semanticscholar.org/paper/The-Role-of-the-Information-Bottleneck-in-Learning-Vera-Piantanida/0cb57625ac7d8ab80ef377d38598f28371b5d707) - This work derives an upper bound to the so-called generalization gap corresponding to the cross-entr...

24. [Learnability for the Information Bottleneck - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7514257/) - The Information Bottleneck (IB) method provides an insightful and principled approach for balancing ...

25. [Information bottleneck approaches to representation learning](https://ui.adsabs.harvard.edu/abs/2022APS..MARB42003S/abstract) - Extracting relevant information from data is crucial for all forms of learning. The information bott...

26. [A Systematic Review of Experimental Studies on Data Glyphs](https://d-nb.info/1135916578/34)

27. [The Many-Faced Plot: Strategy for Automatic Glyph ...](https://cdv.dei.uc.pt/wp-content/uploads/publications-cdv/cunha2018IV.pdf)

28. [Evaluation of glyph-based multivariate scalar volume visualization ...](https://dl.acm.org/doi/10.1145/1620993.1621006) - We present a user study quantifying the effectiveness of Scaled Data-Driven Spheres (SDDS), a multiv...

29. [An Empirical Evaluation of Chernoff Faces, Star Glyphs, and](https://dl.acm.org/doi/pdf/10.5555/857080.857081)

30. [King’s Research Portal](https://kclpure.kcl.ac.uk/ws/portalfiles/portal/104723075/Glyph_PVis2019_Post_Review_Copy_preprint.pdf)

31. [[PDF] Face-Based Glyphs Revisited - Eurographics](https://diglib.eg.org/server/api/core/bitstreams/9f4e8166-9a86-463a-83ab-375d51a76229/content)

32. [How Ordered Is It? On the Perceptual Orderability of Visual Channels](https://kclpure.kcl.ac.uk/portal/en/publications/how-ordered-is-it-on-the-perceptual-orderability-of-visual-channe)

33. [@tamaramunzner](https://www.cs.ubc.ca/~tmm/courses/532-18/slides/lect-5-6-4x4.pdf)

34. [Eurographics Conference on Visualization (EuroVis) 2016](http://darrenedwards.info/index_files/EVis-visualOrderabilityFinal%20(1).pdf)

35. [[PDF] Marks and Channels - Csl.mtu.edu](https://www.csl.mtu.edu/cs5631.ck/common/05-Marks-Channels.pdf)

36. [CSE 5544: Introduction](http://web.cse.ohio-state.edu/~machiraju.1/teaching/CSE5544/ClassLectures/PDF/Lecture-5-1.pdf)

37. [Multi-Criteria Decision Analysis for Aiding Glyph Design - arXiv.org](https://arxiv.org/html/2303.08554v2) - Glyph-based visualization is one of the main techniques for visualizing complex multivariate data. W...

38. [Generative Learning of Densities on Manifolds - arXiv](https://arxiv.org/html/2503.03963v2) - A generative modeling framework is proposed that combines diffusion models and manifold learning to ...

39. [Score-based generative model learnmanifold-like structures with ...](https://neurips.cc/virtual/2022/61139) - How do score-based generative models (SBMs) learn the data distribution supported on a lower-dimensi...

40. [Generative learning of densities on manifolds - ScienceDirect.com](https://www.sciencedirect.com/science/article/abs/pii/S0045782525005389) - A generative modeling framework is proposed that combines diffusion models and manifold learning to ...

41. [[PDF] On Out-of-distribution Detection with Energy-based Models](http://www.gatsby.ucl.ac.uk/~balaji/udl2021/accepted-papers/UDL2021-paper-021.pdf) - Several density estimation methods have shown to fail to detect out-of-distribution (OOD) sam- ples ...

42. [[PDF] Generating High Fidelity Data From Low-Density Regions Using ...](https://openaccess.thecvf.com/content/CVPR2022/papers/Sehwag_Generating_High_Fidelity_Data_From_Low-Density_Regions_Using_Diffusion_Models_CVPR_2022_paper.pdf) - In this section, we present our approach to generating samples from low-density regions of data mani...

43. [Uncanny valley - Wikipedia](https://en.wikipedia.org/wiki/Uncanny_valley)

44. [Too real for comfort? Uncanny responses to computer generated faces](https://pmc.ncbi.nlm.nih.gov/articles/PMC4264966/) - As virtual humans approach photorealistic perfection, they risk making real humans uncomfortable. Th...

45. [Frontiers | A review of empirical evidence on different uncanny valley hypotheses: support for perceptual mismatch as one road to the valley of eeriness](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2015.00390/full) - The uncanny valley hypothesis, proposed already in the 1970s, suggests that almost but not fully hum...

46. [1 A Meta-analysis of the Uncanny Valley's Independent ...](http://www.macdorman.com/kfm/writings/pubs/Diel-2022-Meta-Analysis-Uncanny-Valley-ACM.pdf)

47. [A Systematic Approach to Quantify the Uncanny Valley Effect](https://www.thinkmind.org/articles/achi_2024_1_10_28001.pdf)

48. [AI-generated images of familiar faces are indistinguishable from real ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC12521686/) - Abstract. Human users are now able to generate synthetic face images with artificial intelligence (A...

49. [Attractive faces are less likely to be judged as artificially generated](https://www.sciencedirect.com/science/article/pii/S0001691824005481) - In this study, 150 participants rated 109 pictures of faces on 4 characteristics (attractiveness, be...

50. [Testing human ability to detect 'deepfake' images of human faces](https://academic.oup.com/cybersecurity/article/9/1/tyad011/7205694) - This study aims to assess human ability to identify image deepfakes of human faces (these being uncu...

51. [[PDF] Is CLIP ideal? No. Can we fix it? Yes!](https://www.openaccess.thecvf.com/content/ICCV2025/papers/Kang_Is_CLIP_ideal_No._Can_we_fix_it_Yes_ICCV_2025_paper.pdf)

