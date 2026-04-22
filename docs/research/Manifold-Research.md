---
status: superseded
topic: manifold-geometry
superseded_by: 2026-04-22-manifold-papers-distillation.md
---

> **Superseded note (2026-04-22).** This survey oversold the
> applicability of two references: RJF (ref [22][23]) is
> training-time only and requires an analytical hyperspherical
> manifold Flux's attention cache does not have; Hessian-Geometry
> (ref [13]) is validated only on 2D latent slices of SD 1.5 / Ising
> / TASEP and needs a 2D slice before its Fisher metric is defined.
> The String-Method reference (ref [11][12]) is the only one that
> directly matches our non-monotonic-smile phenomenology. See
> `2026-04-22-manifold-papers-distillation.md` for the paper-by-paper
> verdicts against our open questions. The survey below is kept for
> its reference list and general framing only.

# Geometric Properties of Diffusion and Flow-Matching Model Manifolds

## Executive summary

This report surveys recent work on the geometry of distributions learned by diffusion and flow-matching models, with a focus on how the training objective, data representation, and architecture shape the effective data manifold and the generative trajectories that traverse it.[1][2][3] It connects classical manifold-hypothesis views with newer results on log-domain smoothing, tubular neighbourhoods, Riemannian score/flow models, and the behaviour of modern DiT-based systems such as Flux.

At a high level, score-based diffusion training implicitly encourages log-density smoothing that is tangential to the data manifold, so that the learned score field respects low-dimensional structure despite operating in a high-dimensional ambient space.[4][3] The geometry of the learned manifold is further constrained by (i) the choice of data space or representation (pixels vs encoder features, Euclidean vs hyperspherical), (ii) the objective (score matching vs Euclidean flow matching vs Riemannian variants), and (iii) architectural inductive biases (U-Net vs DiT, guidance schemes), which together determine whether trajectories stay in a thin tubular neighbourhood of the data manifold or cut through low-density regions.


## Background: data manifolds and generative dynamics

The manifold hypothesis posits that high-dimensional data such as images are concentrated near a much lower-dimensional manifold $$M$$ embedded in an ambient space $$\mathbb{R}^D$$. Generative models then aim to learn a distribution whose support is concentrated in a tubular neighbourhood of $$M$$, and whose sampling dynamics define paths that move along, rather than orthogonally away from, this manifold.[5][6]

Diffusion models implement this via a forward noising process that gradually pushes data toward a simple reference distribution (often Gaussian) and a reverse-time generative process driven by a learned score field $$\nabla_x \log p_t(x)$$. Score-based generative modelling on Euclidean space can be viewed as learning a velocity field whose integral curves map noise to data across time, with the associated probability paths described by SDEs or equivalent ODEs.[1][7]

Flow-matching and rectified-flow models instead formulate a deterministic ODE transporting a simple base distribution to the data by regressing a time-dependent velocity field toward an analytically constructed target, under assumptions about the marginals. Under Gaussian assumptions, flow matching and diffusion can be unified in a single probabilistic framework, with different parameterisations of the same underlying transport.[8][9]


## Geometry of diffusion models in Euclidean ambient space

### Quasi-linear trajectories and denoising geometry

“A Geometric Perspective on Diffusion Models” analyses the ODE associated with a variance-exploding SDE and finds that the sampling dynamics connect data and noise by a quasi-linear trajectory plus an implicit denoising trajectory.[1] The denoising trajectory governs the curvature of the sampling path; finite-difference approximations of this trajectory correspond to the second-order samplers used in practice, linking geometric curvature to numerical integrators.[1]

This work further relates optimal ODE-based sampling to the mean-shift algorithm: asymptotically, the model behaves like a mode-seeking procedure on the learned density, so that sampling paths follow ridges of high density along the data manifold, while deviations in the learned score translate to curvature changes and potential off-manifold excursions.[1]

### Tubular neighbourhoods and singularities

Sakamoto et al. study diffusion dynamics in relation to the tubular neighbourhoods of the data manifold, introducing an algorithm to estimate the injectivity radius—the maximal radius for which the normal exponential map is a diffeomorphism and tubular neighbourhoods are well-defined.[2] They show that geometric quantities such as manifold curvature and the ratio between intrinsic and ambient dimensions control singularities in the generative dynamics, including emergent critical phenomena and spontaneous symmetry breaking.

When the noise schedule or dynamics push trajectories outside the injectivity radius, the mapping from latent variables to data can self-intersect or become ill-conditioned, which manifests as sampling instabilities and mode structure that does not align with the true data manifold. This provides a formal geometric lens on issues such as over-smoothing or collapsing modes in diffusion sampling.

### Log-domain smoothing and manifold adaptivity

Farghly et al. revisit the manifold hypothesis for diffusion models and show that the score-matching objective implicitly performs smoothing in the log-density domain.[3][10] They prove that, in an affine setting, log-domain smoothing with a generic kernel is equivalent to smoothing with a geometry-adaptive kernel that acts only along planes parallel to the data manifold, leaving normal directions comparatively less smoothed.[4]

The analysis is extended to curved manifolds, where the approximation error between generic log-domain smoothing and strictly manifold-adapted smoothing can be bounded in terms of manifold dimension, curvature, and the smoothing scale.[4] These results support the view that diffusion training naturally favours tangential smoothing and therefore adapts to the intrinsic geometry of the data manifold, with the choice of smoothing scale controlling how sharply the model concentrates around a particular manifold of generalisation.[3]

### Global geometry via string methods

Moreau et al. introduce a “Diffusion String Method” that evolves discrete curves between samples under the learned score field, allowing the computation of Minimum Energy Paths (MEPs) and Principal Curves in the learned distribution without retraining.[11][12] They find a “likelihood–realism” paradox: MEPs that maximise likelihood tend to pass through visually degenerate, low-entropy regions (cartoon-like images), whereas finite-temperature principal curves traverse high-entropy, realistic regions more aligned with the typical set.

This method reveals the global modal structure and barrier heights between modes, showing that naive interpolation or purely energy-based paths can leave the data manifold, while entropy-aware dynamics better respect its geometry.[11][12] In practice, this indicates that the manifold relevant for human perception is not simply the locus of maximum probability density, but an entropy-weighted subset of the typical set shaped by both the learned score and the volume form of the ambient space.

### Hessian and Fisher geometry of latent spaces

Recent work on “Hessian Geometry of Latent Space in Generative Models” applies Hessian-based techniques to extract a Fisher information metric from generative latent spaces, including diffusion models.[13] The authors observe a fractal structure of phase transitions: the Fisher metric exhibits abrupt changes delineating different “phases” in latent space, with geodesic interpolations being approximately linear within each phase but breaking down at phase boundaries where the effective Lipschitz constant diverges.[13]

This strongly suggests that the learned manifold is piecewise-regular with sharp geometric transitions, and that architectural or training changes that modify the Hessian spectrum will reshape these phase boundaries, altering where geodesics are straight and where they kink or fold.


## Riemannian score-based models and diffusion on manifolds

### Riemannian Score-Based Generative Modelling

De Bortoli et al. extend score-based generative modelling to data supported on Riemannian manifolds, defining Riemannian Score-Based Generative Models (RSGMs).[7][14] They replace Euclidean Brownian motion with diffusion processes intrinsic to the manifold, derive time-reversal formulas in this setting, and show how to estimate manifold-valued scores using Riemannian gradients and exponential/log maps.[15]

RSGMs demonstrate strong performance on data lying on spheres and other manifolds, where classical Euclidean SGMs either perform poorly or require extrinsic embeddings that ignore curvature.[7][16] Geometrically, RSGMs ensure that the forward noising and reverse denoising processes remain on the manifold by construction, so the learned vector field operates in tangent spaces and respects geodesic distances rather than Euclidean ones.

### Flow matching on general geometries

Chen and Lipman propose Riemannian Flow Matching (RFM), which trains continuous normalising flows directly on manifolds using closed-form target vector fields defined via a premetric.[17][18] They use spectral decompositions of the Laplace–Beltrami operator to construct these premetrics efficiently, enabling flows on complex geometries such as triangular meshes with nontrivial curvature and boundaries.[17]

By defining the flow in terms of intrinsic geometry, RFM ensures that trajectories remain on the manifold and that the induced metric governs both training and sampling. This provides a direct analogue of rectified flow on manifolds and shows how modifying the premetric or manifold structure changes the learned transport paths and the resulting manifold of generated samples.[18]

### Diffusion geometry for estimating curvature and dimension

Jones’ “Manifold Diffusion Geometry” develops estimators for manifold curvature, tangent spaces, and intrinsic dimension using diffusion maps and Markov diffusions.[19][20] The methods are robust to noise and low-density regions and outperform prior approaches particularly when data is sparse or noisy, situations typical in high-dimensional generative modelling.[21]

Although not specific to generative models, these tools can be applied to samples from diffusion or flow-matching models to empirically recover the local geometry of the learned data manifold, providing a way to measure how architectural or training changes alter curvature, dimension, and tangent structures.


## Geometry in modern Diffusion Transformers and Flux-like models

### Geometry interference in representation encoders

Recent work on “Learning on the Manifold: Unlocking Standard Diffusion Transformers with Representation Encoders” identifies a failure mode termed Geometric Interference when training diffusion transformers directly on hyperspherical representation encoder features.[22][23] Standard Euclidean flow matching forces probability paths through the low-density interior of the hyperspherical feature space instead of along the manifold surface, causing training instability and convergence failures.[22]

The authors propose Riemannian Flow Matching with Jacobi regularisation (RJF), constraining the generative process to manifold geodesics and correcting curvature-induced error propagation.[23] This keeps the intermediate states strictly on the hypersphere and velocities within tangent spaces, enabling standard DiT-B architectures to converge without width scaling while achieving strong FID scores.[22]

### Flow matching, rectified flow, and Flux-style models

Flow matching reparameterises generative modelling as fitting a time-dependent velocity field whose ODE transports a simple base distribution to the data distribution. Educational notes on flow matching highlight the geometric picture of a flow $$\psi_t$$ acting on the ambient space via a velocity field $$u_t$$, with trajectories that can be understood as straightened counterparts of stochastic diffusion paths.[8]

Score-distillation analyses show that, under Gaussian assumptions, flow matching objectives and diffusion objectives are theoretically equivalent, allowing score distillation techniques originally designed for diffusion to transfer to flow-matching models including modern text-to-image systems such as SANA, SD3, and Flux.[9][24] This unification implies that many geometric insights derived for diffusion (e.g., log-domain smoothing and manifold adaptivity) extend to Flux-style models at the level of distributions and probability paths.

Flux.1 Kontext exemplifies a DiT-based flow-matching model that unifies image generation and editing by operating on sequences of image and text tokens.[25] Its architecture uses multimodal DiTs that maintain separate image and text streams, coupled via joint attention, allowing the learned velocity field to respect both visual and semantic manifolds while enabling efficient, few-step generation and editing.[25][26]

### Stability and shared manifolds under perturbations

“The Amazing Stability of Flow Matching” empirically investigates how flow-matching models behave under significant changes to the training data and architecture.[27] When training separate models on disjoint subsets of the data, integrating the velocity fields from the same noise seeds yields highly similar outputs; even when trained on different but related datasets (CelebA-HQ vs FFHQ), latent trajectories remain closely aligned, suggesting that both datasets lie on a shared manifold learned by the flow.[27]

The authors further show that reducing model capacity or switching between different architectures (e.g., UNet vs DiT variants) still preserves coarse attributes of generated images, as measured by identity similarity, despite noticeable changes in fine details.[27] This indicates that the global geometry of the learned manifold—its coarse structure and main modes—is robust to such perturbations, while local curvature and higher-order structure are more sensitive.

### Manifold-preserving guidance and tubular stability

Classifier-free guidance (CFG) is known to pull diffusion and rectified-flow trajectories off the data manifold when applied naively, especially at large guidance strengths. Recent work on Rectified-CFG++ for flow-based models (including Flux and SD3) introduces a predictor–corrector scheme that combines rectified-flow deterministic updates with a geometry-aware conditioning rule.[28][29]

Theoretical analysis proves that the resulting conditional velocity field remains within a bounded tubular neighbourhood of the data manifold and is marginally consistent, ensuring stability even at high guidance strengths.[28] Empirically, this geometry-aware guidance yields better prompt fidelity and robustness, while avoiding the off-manifold artifacts observed with naive CFG, highlighting how conditioning mechanisms directly affect manifold adherence.[29]


## How training objectives shape the learned manifold

### Score matching and log-domain smoothing scale

Score matching minimises the squared error between the model score and the empirical data score under a noise-perturbed distribution, which can be interpreted as smoothing the log-density. Farghly et al. show that the scale of this log-domain smoothing determines how geometry-adaptive the smoothing becomes: small normal-direction smoothing relative to tangent directions ensures that probability mass remains concentrated near the manifold, whereas large smoothing scales can wash out fine manifold features and encourage broader, less geometric generalisation.[4][3]

This perspective clarifies how design choices such as the noise schedule, early stopping, and regularisation affect the resulting manifold: more aggressive smoothing and longer training push the model toward a coarser, lower-curvature approximation of the data manifold, while milder smoothing preserves finer curvature at the cost of potentially higher variance and sensitivity to sample sparsity.[4][10]

### Flow matching and rectified objectives

In flow matching and rectified-flow models, the objective is to regress a velocity field toward a target that realises a specific transport plan between base and data. Because the flow is deterministic, there is no explicit entropy term in the dynamics, so the geometry of the learned manifold depends strongly on how the target vector field is constructed (e.g., choice of interpolation path in space or feature space) and on regularisers that control Lipschitz constants and curvature.[8][9]

Rectified-flow objectives that encourage approximately straight paths between noise and data in an appropriate representation space can be seen as imposing a geodesic-like structure on the learned manifold; however, if the representation space has nontrivial curvature (e.g., hyperspherical encoder features), naive Euclidean interpolation induces geometric interference unless corrected by Riemannian flow methods such as RJF.[22][23]

### Riemannian vs Euclidean formulations

Riemannian SGMs and RFMs make the manifold structure explicit by defining noise and transport processes intrinsically on $$M$$, so that trajectories are constrained to tangent spaces and geodesics.[7][17] In contrast, Euclidean formulations operating on embedded manifolds rely on implicit regularisation from score matching and model architecture to avoid leaking probability mass into normal directions.

Empirical results show that Riemannian models outperform Euclidean baselines on manifold-supported data and are more robust to curvature and topological complexity.[7][17] This suggests that making the geometric structure explicit in the training objective can significantly tighten the tubular neighbourhood around the data manifold and reduce pathological singularities in the dynamics.


## How architecture and representation impact manifold geometry

### U-Nets vs Diffusion Transformers

Classical diffusion architectures based on U-Nets are strongly biased toward local convolutional processing, which emphasises local smoothness and translation invariance in the learned manifold.[1] Diffusion Transformers (DiTs), by tokenising images into patches and using global self-attention, introduce inductive biases toward long-range interactions and global coherence, which change how modes are arranged and connected in the learned distribution.[30][8]

Empirical studies in flow-matching setups indicate that changing architecture capacity or even switching between UNet and DiT backbones preserves global identity and coarse attributes but alters fine geometry, such as texture-level curvature and the shape of phase boundaries in latent space.[27][13] Thus, architecture mainly reshapes local curvature and connectivity in the manifold while leaving its coarse embedding and main modes relatively stable.

### Representation encoders and hyperspherical structure

Using pre-trained representation encoders (e.g., DINOv2 features) as the space on which diffusion or flow models operate introduces an explicit manifold structure, often close to a hypersphere due to normalisation.[22][31] When generative models ignore this geometry and apply Euclidean flow matching, learned trajectories traverse low-density interior regions, degrading convergence and sample quality.[22]

RJF demonstrates that correcting this via Riemannian flow matching, enforcing trajectories on the manifold surface, eliminates geometric interference and allows standard-width DiTs to converge, effectively aligning the learned manifold with the encoder’s intrinsic geometry.[23] This shows that the choice of representation and whether its geometry is respected in the objective has a first-order effect on the manifold learned by the generative model.

### Conditioning mechanisms and guidance

Conditioning mechanisms such as classifier-free guidance modify the effective vector field by adding or scaling conditional components, potentially inducing off-manifold drift. Rectified-CFG++ shows that guidance rules can be made geometry-aware, preserving a bounded tubular neighbourhood around the data manifold while still steering samples strongly toward prompts.[28][29]

Similarly, score-guided editing methods based on proximal projection and rectified-flow editing interpret guidance as optimisation in an energy landscape balancing fidelity to input with realism, making explicit use of local curvature and normal contraction properties of the learned manifold.[32] These approaches highlight that conditioning is not merely a semantic steering tool but a geometric transformation that can either preserve or distort manifold structure.


## Practical mental models and open questions

### Practitioner’s mental model

For diffusion and Flux-style models, a useful mental model is that training learns a vector field whose normal component (relative to the data manifold) performs denoising and whose tangential component shapes how mass flows along the manifold. Score matching plus log-domain smoothing implicitly dampens normal components more strongly, keeping trajectories near the manifold, while architectural choices and conditioning rules govern tangential flow and mode connectivity.[1][3][27]

Flux and other flow-matching models can be seen as choosing a particular family of paths—often straighter and lower-curvature in an appropriate representation space—subject to the constraints imposed by the chosen representation (pixels vs encoder features, Euclidean vs Riemannian) and any geometric regularisers like RJF or tubular-neighbourhood-preserving guidance.[8][22][28]

### Open research directions

Several directions remain active and incompletely understood:

- **Quantifying learned curvature and dimension**: Applying diffusion-geometry estimators to large diffusion/flow models at scale to map curvature, dimension, and injectivity radius as functions of architecture and training regime.[19][2]
- **Phase transitions and Fisher geometry**: Systematically relating Hessian/Fisher phase boundaries to semantic boundaries in image space and to failure modes such as mode collapse or over-smoothing.[13]
- **Entropy vs energy in paths**: Extending string-method analyses to conditioned settings and complex architectures to better align likelihood-based trajectories with human-perceived realism and typical-set geometry.[11][12]
- **Unified Riemannian frameworks for modern DiTs**: Bringing Riemannian SGMs/RFMs together with large-scale DiT and representation-encoder pipelines (as in RJF) to make manifold-respecting modelling the default for foundation generative models.[7][22][17]

Addressing these questions will provide a sharper geometric understanding of how training objectives and architectures carve out manifolds in high-dimensional spaces, and how to deliberately design models whose trajectories stay on, and move meaningfully along, those manifolds.