# Flow-Based Generative Models for Faces and Face Animation (2022–2026): Foundations and Evaluation-Relevant Results
## 1. Scope and goals
This report surveys load‑bearing foundations for flow-based generative modeling as a substrate for face image generation and face animation, focusing on 2022–2026.
It emphasizes flow matching, rectified flow, and normalizing-flow–style invertible generators, with an eye toward: (i) suitability as a face generator backbone; (ii) exact / tractable density and inversion for OOD and uncanny-valley analysis; and (iii) disentangled latents for algebraic editing.

For each major paper, the report lists arXiv ID, venue (when available), a brief contribution summary, and whether code/weights are public.
It is structured along five strands:

1. Flow matching and rectified flow theory and general foundations.
2. Flow matching / rectified flow applied to faces and talking-head animation.
3. Normalizing-flow “revival” and architectural advances 2024–2026.
4. Invertible / bijective generators for OOD/uncanny analysis.
5. Disentangled flow matching for algebraic latent operations.

A final subsection summarizes “where we are in 2026” from a design‑evaluation standpoint.

***
## 2. Foundational flow-matching and rectified-flow papers
### 2.1 Flow Matching for Generative Modeling (Lipman et al., 2022)
- **Paper:** “Flow Matching for Generative Modeling” (Lipman et al.), arXiv:2210.02747, accepted at ICLR 2023.[^1][^2]
- **Contribution:** Introduces *flow matching* as a way to train deterministic ODE flows between a base and target distribution by matching a prescribed *probability path* and vector field, avoiding stochastic differential equations and score estimation; shows equivalence to Schrödinger bridges under certain choices of path, and demonstrates competitive or improved sample quality vs. diffusion with fewer steps on image benchmarks (CIFAR-10, FFHQ, ImageNet).[^3][^2]
- **Code / weights:** Several open-source implementations exist (official and third-party), including didactic “Flow Matching Guide and Code” (arXiv:2412.06264) that provides reference implementations and experimental scripts.[^2][^4]

From an evaluation perspective, Lipman et al. supply the core mathematical framework for deterministic flows and clarify the role of the path measure; this is the main theoretical reference when treating a flow-based face generator as an ODE solver over latent space.[^2]
### 2.2 Rectified Flow: A Marginal Preserving Approach to Optimal Transport (Liu et al., 2022)
- **Papers:**  
  - “Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow” (Liu et al.), arXiv:2209.03003.[^5][^6]
  - “Rectified Flow: A Marginal Preserving Approach to Optimal Transport” (Liu et al.), arXiv:2209.14577.[^7][^8]
- **Contribution:** Rectified flows are defined via a path that linearly interpolates between base and target distributions in data space while preserving marginals, leading to nearly straight ODE trajectories and simpler training objectives; they provide tight connections to optimal transport and prove convergence properties.[^7][^5]
  Experimental results show faster sampling and more stable training than diffusion on standard image datasets and demonstrate straightforward transfer to style transfer and super-resolution tasks.[^9][^5]
- **Code / weights:** Public implementations for rectified flow and tutorials are widely available; the authors and community maintain reference code in PyTorch and JAX, used as a basis for later rectified-flow models.[^10][^9]

Rectified flow’s straight trajectories and OT interpretation are central when assessing path regularity, Lipschitz properties, and how easily conditioned trajectories can be steered for identity-preserving edits.
### 2.3 Minibatch Optimal Transport for Flow-Based Generative Models (Tong et al., 2023–2024)
- **Paper:** “Improving and generalizing flow-based generative models with minibatch optimal transport” (Tong et al.), arXiv:2302.00482, published in TMLR 2024.[^11][^12][^13]
- **Contribution:** Introduces *minibatch OT* losses to train normalizing flows and flow-matching models using mini-batches while approximating full OT couplings; improves training stability and sample quality on continuous and image data.[^12][^11]
  The paper provides convergence analysis for minibatch-OT training and demonstrates improved FID and NLL on CIFAR-10 and ImageNet against standard maximum-likelihood training of flows.[^14][^11]
- **Code / weights:** Official code is released (PyTorch), including flow architectures and training scripts, though pretrained face-specific weights are not the focus.[^13][^12]

This work is relevant as it replaces log-likelihood-only training with OT-regularized objectives, which may better align with perceptual quality for face synthesis.
### 2.4 Later flow-matching foundations and surveys (2024–2026)
- **Flow Matching Guide and Code (2024):** arXiv:2412.06264 gives a consolidated tutorial, code, and best practices for training flow-matching models efficiently, including connections to diffusion, optimal transport, and numerical ODE/SDE solvers.[^15][^4]
- **Laplacian Multi-scale Flow Matching (LapFlow, 2026):** “Laplacian Multi-scale Flow Matching for Generative Modeling,” arXiv:2602.19461, ICLR 2026, proposes a multi-scale Laplacian-pyramid architecture combined with flow matching and transformer backbones, scaling flow matching to high-resolution images (up to 1024×1024) with improved FID and reduced compute compared to single-scale flows.[^16]
- **Survey / tutorial material:** A 2026 MIT CSAIL tutorial “Flow Matching and Diffusion Models — 2026 Version” summarizes the landscape, treating flow matching and rectified flows as the likelihood-based, ODE-centric branch of generative modeling and outlining best practices for architecture choice, path design, and numerical solvers.[^17][^18]

These later works codify flow matching as a mature alternative to diffusion, with multi-scale architectures and public code bases suitable for face datasets, though they are not face-specific.[^18][^16]

***
## 3. Flow matching & rectified flow applied to faces and talking-head animation
### 3.1 FLOAT: Generative Motion Latent Flow Matching for Audio-driven Talking Portraits (ICCV 2025)
- **Paper:** “FLOAT: Generative Motion Latent Flow Matching for Audio-driven Talking Portraits,” arXiv:2412.01064, ICCV 2025.[^19][^20]
- **Contribution:** FLOAT trains a latent-space flow-matching model to generate face motion trajectories (lip and expression dynamics) conditioned on audio, operating in a motion-latent space rather than pixel space; it decouples identity (kept fixed) from motion and learns a generative motion prior via flow matching.[^21][^19]
  It shows improved temporal consistency and lip-sync accuracy over diffusion-based talking-head models, while benefiting from the deterministic and straight-flow properties of flow matching.[^19]
- **Code / weights:** Official implementation and pretrained models are released on GitHub, enabling direct reuse for audio‑driven face animation experiments.[^22]

FLOAT is a key foundation if you want a flow-based latent motion prior for a vamp-style face-animation interface, especially when combining fixed-identity base renderers with learned motion flows.
### 3.2 FlowChef: Steering Rectified Flow Models for Controlled Generations (ICCV 2025)
- **Paper:** “FlowChef: Steering of Rectified Flow Models for Controlled Generations,” ICCV 2025, open-access at CVF.[^23][^24]
- **Contribution:** FlowChef analyzes rectified-flow vector-field properties (straightness, smooth Jacobians) and develops a training- and gradient-free steering method for rectified-flow models that can handle classifier guidance, inverse problems, and image editing by modifying the vector field along the ODE trajectories.[^24][^23]
  It demonstrates controlled image editing and inverse problems on several image benchmarks and scales to billion-parameter rectified-flow models such as Flux, showing effective steering without re-training or backprop through solvers.[^23]
- **Code / weights:** Official implementation is on GitHub (FlowChef/FlowChef) with examples for Flux; it relies on existing pretrained rectified-flow models rather than training its own from scratch.[^25]

For face generators, FlowChef is foundational for *deterministic, controllable editing* of rectified flows, which is relevant to identity-preserving transformations and interactive control of facial attributes.
### 3.3 Stable Diffusion 3 and Flux: Rectified-Flow Transformers for High-Resolution Images
- **Paper:** “Scaling Rectified Flow Transformers for High-Resolution Image Synthesis,” arXiv:2403.03206, the Stable Diffusion 3 research paper.[^26][^27]
- **Contribution:** Introduces a rectified-flow formulation with multimodal diffusion transformers (MMDiT) as the backbone, scaling to 8B parameters and high-resolution image generation; shows that rectified flows can match or exceed diffusion quality at significantly fewer steps, and details training heuristics for stability and conditioning.[^27][^28]
- **Code / weights:** Stability AI has released SD3‑family models and APIs, though full training code is partially closed; model weights or derivatives are available via Stability’s releases and community ports.[^26]

- **Flux models:** Flux is a family of rectified-flow transformer models (by Black Forest Labs et al., 2024) used as backbone in works like FlowChef and FluxSpace; blog and technical notes (e.g., “Flux, rectified flow transformers explained”) describe its architecture and training, positioning Flux as a rectified-flow analogue of SDXL.[^23][^29]
  Flux model weights are available under various licenses via official releases and community distributions; combined with FlowChef, they form an off-the-shelf rectified-flow backbone for faces.[^25][^29]

While SD3 and Flux are not face-specific, their widespread use and released weights make them the default high-capacity rectified-flow substrates on which face-identity evaluation can be performed (e.g., measuring identity consistency, attribute disentanglement and inversion stability on face subsets).
### 3.4 Other rectified-flow and flow-matching face/portrait works
- **Real-Time Person Image Synthesis Using a Flow Matching Prior (2025):** arXiv:2505.03562 proposes a person image generator conditioned on pose and appearance using a flow-matching prior in latent space, targeting real-time synthesis for human images.[^30]
  Code availability is not yet clear, but the method exemplifies using flow matching for person/face synthesis under structured conditioning (pose, identity embeddings).[^30]
- **Learning Flow Fields in Attention for Controllable Person Image Generation (2024):** referenced in FluxSpace’s related work; learns flow fields in attention to control person image generation, often using rectified-flow training under the hood.[^31]

These provide additional examples of flow-based architectures specifically tuned for human/face data, demonstrating that flow matching and rectified flow scale to high-quality human synthesis.

***
## 4. Normalizing-flow revival and architectural advances (2024–2026)
### 4.1 General advances in normalizing flows
While Glow (2018) and RealNVP (2017) remain canonical, recent work has focused on improving expressivity, stability, and channel mixing.
A representative example is **Entropy-informed weighting channel normalizing flow (EIW-Flow)**, which proposes a regularized channel shuffling operation guided by entropy, preserving invertibility while improving density estimation and sampling quality on image datasets.[^32]

Other 2025–2026 works explore:

- Multi-scale coupling and “unsqueeze/squeeze” operations building on Glow to reduce compute while maintaining exact likelihood.[^32]
- Normalizing-flow–based metrics for image generation/retrieval that leverage flows as learned similarity measures rather than direct generators.[^33]
- Flow-based visuomotor policies (e.g., NeurIPS 2025 “Normalizing Flows are Capable Visuomotor Policy Generators”), highlighting flows as expressive distributions over complex structured outputs.[^34]

These are not face-specific but re-establish flows as competitive density models with improved architectures and are relevant if you want a flow-based, exact-likelihood component in your stack.
### 4.2 Normalizing flows for image and face generation
Direct 2024–2026 Glow/RealNVP successors targeting face images are relatively sparse compared to diffusion/flow matching.
However:

- Open implementations of Glow and variants continue to use CelebA/CelebA‑HQ as reference datasets, with reproducible code and scripts (e.g., `glow.py` with `--dataset=celeba`) demonstrating that exact-likelihood flows scale to 64×64 and 128×128 faces with moderate quality.[^35][^36]
- Some normalizing-flow methods target human motion and animation (e.g., NFlowAD: a normalizing-flow model for anomaly detection in human motion animations, 2026), modeling human pose trajectories with invertible flows that can detect anomalous or off-distribution motions.[^37]

In terms of load-bearing foundations, Glow/RealNVP remain the main invertible convolutional architectures, while recent channel-weighting and entropy-informed flows show how to boost expressivity without sacrificing exact density.[^32]
### 4.3 Normalizing flows plus flow matching / hybrid approaches
Several 2025–2026 works blend normalizing flows with flow matching or diffusion, for example:

- **Amortized Sampling with Transferable Normalizing Flows (NeurIPS 2025):** trains flows to approximate expensive samplers (e.g., diffusion) in latent space, enabling faster approximate sampling while retaining the invertible structure and density evaluation of flows.[^38]
- **LR2Flow / Enhancing Low-resolution Image Representation Through Normalizing Flows (arXiv:2601.06834, 2026):** uses normalizing flows to model low-resolution image representations and improve downstream generation and super-resolution tasks.[^39]

These hybrids are promising if you want to use flows primarily for *likelihood / OOD estimation* while delegating high-fidelity synthesis to flow matching or rectified-flow models.

***
## 5. Invertible / bijective generators for OOD and uncanny-valley analysis
### 5.1 Invertible models for OOD detection
A core motivation for using flows as substrates is exact density and exact inverse, which can support OOD and uncanny-valley diagnostics.
However, naive likelihood-based OOD detection with flows is known to fail, as flows can assign higher likelihoods to OOD data than in-distribution data.[^40][^41]

Key works addressing this include:

- **Likelihood-free OOD detection with invertible generative models (Ahmadian & Lindsten, IJCAI 2021):** proposes using invertible generative models to build *representation spaces* and complexity-adjusted statistics, combined with one-class SVMs, instead of raw likelihood; improves OOD performance and shows that representations learned by invertible models are more useful for OOD than likelihood alone.[^42]
- **HYBOOD: a hybrid generative model for OOD detection (AAAI 2025):** uses a normalizing-flow density model followed by a simple linear classifier, combining label predictions and flow-based density to detect OOD; uses likelihood ratios that correct for image complexity following Serrà et al. (2020).[^43]
- **Revisiting Likelihood-Based OOD Detection by Modeling Representations (2025):** proposes modeling the distribution of *representations* (e.g., ViT features) using diffusion-like generative models and shows that likelihood estimated in representation space is a more reliable OOD signal than image-space likelihood.[^40]

These works suggest that for uncanny-valley or factor-mismatch detection, flows are best used as invertible encoders into feature spaces where specialized OOD or “mismatch” scores are computed, rather than relying directly on image-level likelihood.
### 5.2 Flows as exact inverses for face factor mismatch
While not face-specific, the above methods can be instantiated on face data: a Glow/RealNVP-style flow trained on faces yields exact log-likelihood and an invertible latent representation, which can be combined with separate mismatch metrics (e.g., attribute classifiers, geometry errors) to estimate uncanny-valley scores.

Normalizing-flow–based metrics for image generation and retrieval (e.g., Moonlight 2025 review) explore using flows as learned similarity measures, showing that flow-induced distances in latent space correlate with semantic similarity, which is promising for defining *factor-consistency* metrics in face space.[^33]
Combined with representation-based OOD detection, this gives a principled path toward density-informed but not density-only uncanny metrics.

***
## 6. Disentangled flow matching and algebraic latent operations
### 6.1 SCFlow: Implicitly Learning Style and Content Disentanglement with Flow Models (ICCV 2025)
- **Paper:** “SCFlow: Implicitly Learning Style and Content Disentanglement with Flow Models,” arXiv:2508.03402, ICCV 2025.[^44][^45]
- **Contribution:** SCFlow uses flow matching to learn bidirectional mappings between *entangled* and *disentangled* representations for style and content, without explicit disentanglement supervision.[^46][^44]
  It trains only on the task of merging style and content invertibly, using a synthetic dataset of 510k samples (51 styles × 10k content instances) with full combinatorial coverage; disentangled representations emerge naturally, and the model generalizes in zero shot to datasets like ImageNet-1k and WikiArt.[^44][^46]
- **Code / weights:** Project page (CompVis) indicates code and data will be released; at least demo code and visualizations are public, though full pretrained models may be limited to certain setups.[^46]

SCFlow provides a clear blueprint for invertible, factored latent spaces suitable for algebraic operations like “content A with style B,” which is closely related to face A − face B + face C operations if style/content are mapped to identity/attribute factors.
### 6.2 Disentangled Representation Learning via Flow Matching (2026)
- **Paper:** “Disentangled Representation Learning via Flow Matching,” arXiv:2602.05214 (2026).[^47][^48][^49]
- **Contribution:** Formalizes disentangled representation learning in a flow-matching context by learning factor-conditioned flows in a compact latent space with an orthogonality regularizer between factors, encouraging non-overlapping subspaces for different attributes.[^48]
  The method improves disentanglement metrics and controllability on synthetic data and standard image datasets (including face-like datasets such as CelebA) and supports compositional editing via algebraic operations in the latent space.[^47][^48]
- **Code / weights:** At time of writing, the paper provides algorithmic details and pseudo-code; public code/weights are likely but need to be confirmed as repositories appear after 2026-02 in some cases.[^48]

This is the most direct foundational piece for algebraic latent arithmetic in a flow-matching framework, giving an explicit notion of factor-specific flows and orthogonal latent subspaces.
### 6.3 FluxSpace: Disentangled Semantic Editing in Rectified Flow Transformers (2024)
- **Paper:** “FluxSpace: Disentangled Semantic Editing in Rectified Flow Transformers,” arXiv:2412.09611 (2024).[^31][^50][^51]
- **Contribution:** Proposes a domain-agnostic semantic editing method that works in the *representation space* of rectified-flow transformers such as Flux, defining semantically interpretable directions corresponding to attributes (e.g., facial expressions, eyeglasses) and using orthogonal projections in attention-layer feature spaces.[^50][^51]
  FluxSpace supports fine-grained and coarse-level edits with limited interference between attributes and requires no re-training of the base model, effectively turning Flux into a linearizable semantic space for editing.[^51][^31]
- **Code / weights:** At least partial code is typically released (editing toolkit); full pretrained Flux models are imported rather than trained from scratch.[^31]

FluxSpace shows how rectified-flow transformers can support near-linear semantic arithmetic akin to “face A − face B + face C” by working in internal embedding spaces, even if the base model was not explicitly trained for disentanglement.
### 6.4 Other disentangling efforts relevant to faces
FluxSpace’s related work references several rectified-flow–based editing methods:

- **Taming Rectified Flow for Inversion and Editing (2024):** improves inversion and editing stability for rectified flows, which is critical for identity-preserving face edits.[^31]
- **Stable Flow: Vital Layers for Training-Free Image Editing (2024):** proposes modifications to flow architectures that make them more amenable to training-free editing; useful when designing future flow backbones for faces.[^31]

Although not all of these are face-specific, they collectively demonstrate that flow matching and rectified flows can support interpretable, linear-ish semantic spaces with algebraic manipulation, bridging the gap to the kind of latent arithmetic used in early GAN work on faces.

***
## 7. 2026 “where we are now” snapshot
By early 2026, flow matching and rectified flows have matured into practical, high-quality substrates for face and person image generation, while normalizing flows occupy a more specialized role as exact-likelihood encoders and OOD tools.

- **Flow matching:** With works like Laplacian Multi-scale Flow Matching (LapFlow) and the Flow Matching Guide, flow matching is a standard, well-tooled method capable of high-resolution image generation on datasets such as CelebA-HQ, with multi-scale architectures and clear best practices.[^4][^16]
- **Rectified flows:** Rectified-flow transformers (Stable Diffusion 3, Flux) dominate large-scale image generation; ancillary work on inversion and steering (FlowChef, RF-Solver-Edit, rectified-flow priors) turns them into controllable, identity-sensitive backbones suitable for image editing and conditioning-heavy tasks like talking-head generation.[^25][^24][^26][^52][^53][^54]
- **Normalizing flows:** Classic Glow/RealNVP architectures are no longer state-of-the-art for large-scale face synthesis but are revived in forms like EIW-Flow, LR2Flow, and hybrid OOD methods (HYBOOD, representation-likelihood flows) where exact density and invertibility are central.[^32][^43][^33][^39]

For a vamp-interface concerned with face/face-animation:

- Flow matching (Lipman, LapFlow) provides the main *generative backbone* theory and practice, including multi-scale paths suitable for high-res faces.[^2][^16]
- Rectified flows (Liu, SD3, Flux) provide *fast, straight trajectories* and well-understood vector field properties, crucial for controllable and identity-preserving edits.[^23][^7][^5][^26][^29]
- FLOAT and related works provide *motion priors* for talking-head animation, explicitly separating identity and motion.[^19][^22]
- SCFlow, Disentangled Flow Matching, and FluxSpace establish *disentangled latent and representation spaces* on top of flow matching and rectified flows, enabling algebraic operations akin to face A − face B + face C.[^44][^50][^48]
- Invertible normalizing flows and OOD work (Ahmadian & Lindsten, HYBOOD, representation-likelihood models) provide *exact-density and representation-based OOD detection* schemes to quantify off-manifold and uncanny-valley factor mismatches.[^42][^40][^43]

As of 2026, combining these strands gives a coherent design space: a rectified-flow or flow-matching face backbone (e.g., Flux/SD3 fine-tuned on faces) augmented with SCFlow/FluxSpace-style disentangled latent spaces and normalizing-flow-based OOD detectors, yielding both a powerful generator and an analyzable, invertible representation suitable for vamp-interface–style evaluation and control.

---

## References

1. [[PDF] Flow Matching for Generative Modeling - Semantic Scholar](https://www.semanticscholar.org/paper/Flow-Matching-for-Generative-Modeling-Lipman-Chen/af68f10ab5078bfc519caae377c90ee6d9c504e9) - This work presents the notion of Flow Matching (FM), a simulation-free approach for training CNFs ba...

2. [[2210.02747] Flow Matching for Generative Modeling - arXiv](https://arxiv.org/abs/2210.02747) - We introduce a new paradigm for generative modeling built on Continuous Normalizing Flows (CNFs), al...

3. [Flow Matching for Generative Modeling - alphaXiv](https://www.alphaxiv.org/resources/2210.02747v1) - View recent discussion. Abstract: We introduce a new paradigm for generative modeling built on Conti...

4. [[2412.06264] Flow Matching Guide and Code](https://arxiv.org/abs/2412.06264) - Flow Matching (FM) is a recent framework for generative modeling that has achieved state-of-the-art ...

5. [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) - We present rectified flow, a surprisingly simple approach to learning (neural) ordinary differential...

6. [[PDF] Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow | Semantic Scholar](https://www.semanticscholar.org/paper/Flow-Straight-and-Fast:-Learning-to-Generate-and-Liu-Gong/244054a4254a2147e43a3dad9c124b9b7eb4a04a) - We present rectified flow, a surprisingly simple approach to learning (neural) ordinary differential...

7. [Rectified Flow: A Marginal Preserving Approach to Optimal Transport](https://arxiv.org/abs/2209.14577) - We present a flow-based approach to the optimal transport (OT) problem between two continuous distri...

8. [[PDF] Rectified Flow: A Marginal Preserving Approach to Optimal Transport | Semantic Scholar](https://www.semanticscholar.org/paper/Rectified-Flow:-A-Marginal-Preserving-Approach-to-Liu/d13f262e72c805c52e4c5767b5caa69e24d8bef7) - The method iteratively constructs a sequence of neural ordinary differentiable equations (ODE), each...

9. [Rectified Flow: A Marginal Preserving Approach to Optimal Transport](https://huggingface.co/papers/2209.14577) - Join the discussion on this paper page

10. [A Visual Introduction to Rectified Flows](https://alechelbling.com/blog/rectified-flow/) - These models leverage neural networks to transform random noise into complex data by applying a sequ...

11. [Improving and generalizing flow-based generative models with minibatch optimal transport](https://arxiv.org/abs/2302.00482) - Continuous normalizing flows (CNFs) are an attractive generative modeling technique, but they have b...

12. [Edinburgh Research Explorer](https://www.pure.ed.ac.uk/ws/portalfiles/portal/472344876/TongEtalTMLR2024ImprovingAndGeneralizingFlow-based.pdf)

13. [[PDF] Improving and generalizing flow-based generative models with minibatch optimal transport | Semantic Scholar](https://www.semanticscholar.org/paper/Improving-and-generalizing-flow-based-generative-Tong-Malkin/5396c55bee2a2abf2207e1cc5e5ae72c9edef9fa) - The generalized conditional flow matching (CFM) technique is introduced, a family of simulation-free...

14. [Fugu-MT 論文翻訳(概要): Improving and generalizing flow-based generative models with minibatch optimal transport](https://fugumt.com/fugumt/paper_check/2302.00482v2)

15. [[Paper Note] Flow Matching for Generative Modeling, Yaron Lipman+, ICLR'23 · Issue #2166 · AkihikoWatanabe/paper_notes](https://github.com/AkihikoWatanabe/paper_notes/issues/2166) - URL https://arxiv.org/abs/2210.02747 Authors Yaron Lipman Ricky T. Q. Chen Heli Ben-Hamu Maximilian ...

16. [Laplacian Multi-scale Flow Matching for Generative Modeling](https://arxiv.org/abs/2602.19461) - Abstract:In this paper, we present Laplacian multiscale flow matching (LapFlow), a novel framework t...

17. [Image Generation Models: A Technical History](https://arxiv.org/html/2603.07455v2) - We dedicate Section 4 to Normalizing Flows, which offer a likelihood-based alternative for generativ...

18. [Flow Matching and Diffusion Models — 2026 Version](https://diffusion.csail.mit.edu) - This course is ideal for those who want to explore the frontiers of generative AI through a mix of t...

19. [FLOAT: Generative Motion Latent Flow Matching for Audio- ...](https://arxiv.org/abs/2412.01064) - This paper presents FLOAT, an audio-driven talking portrait video generation method based on flow ma...

20. [FLOAT: Generative Motion Latent Flow Matching for Audio ...](https://openaccess.thecvf.com/content/ICCV2025/papers/Ki_FLOAT_Generative_Motion_Latent_Flow_Matching_for_Audio-driven_Talking_Portrait_ICCV_2025_paper.pdf) - This paper presents FLOAT, an audio- driven talking portrait video generation method based on flow m...

21. [FLOAT: Generative Motion Latent Flow Matching for Audio- ...](https://huggingface.co/papers/2412.01064) - This paper presents FLOAT, an audio-driven talking portrait video generation method based on flow ma...

22. [[ICCV 2025] FLOAT: Generative Motion Latent Flow ...](https://github.com/deepbrainai-research/float) - This paper presents FLOAT, an audio-driven talking portrait video generation method based on flow ma...

23. [FlowChef - ICCV 2025 Open Access Repository](https://openaccess.thecvf.com/content/ICCV2025/html/Patel_FlowChef_Steering_of_Rectified_Flow_Models_for_Controlled_Generations_ICCV_2025_paper.html) - FlowChef: Steering of Rectified Flow Models for Controlled Generations ... (ICCV), 2025, pp. 15308-1...

24. [Steering of Rectified Flow Models for Controlled Generations](https://openaccess.thecvf.com/content/ICCV2025/papers/Patel_FlowChef_Steering_of_Rectified_Flow_Models_for_Controlled_Generations_ICCV_2025_paper.pdf) - FlowChef steers the trajectory of Rectified Flow Models during inference to tackle linear inverse pr...

25. [FlowChef/FlowChef: [ICCV 2025] Official Implementation of ...](https://github.com/FlowChef/flowchef) - FlowChef introduces a novel approach to steer the rectified flow models (RFMs) for controlled image ...

26. [Stable Diffusion 3: Research Paper](https://stability.ai/news-updates/stable-diffusion-3-research-paper) - We conduct a scaling study for text-to-image synthesis with our reweighted Rectified Flow formulatio...

27. [Scaling Rectified Flow Transformers for High-Resolution ...](https://huggingface.co/papers/2403.03206) - The Stable Diffusion 3 research paper broken down, including some overlooked details! ... UNIMO-G: U...

28. [Stable Diffusion 3: Scaling Rectified Flow Transformers for ...](https://www.youtube.com/watch?v=6XatajQ-ll0) - Website paper: https://stability.ai/news/stable-diffusion-3-research-paper Paper: https://arxiv.org/...

29. [Flux, rectified flow transformers explained - Ozan Ciga](https://ozanciga.wordpress.com/2024/08/18/flux-rectified-flow-transformers-explained/) - The paper spends quite a bit of time experimenting with the best configuration (optimizers, samplers...

30. [Real-Time Person Image Synthesis Using a Flow Matching ...](https://arxiv.org/abs/2505.03562) - Pose-Guided Person Image Synthesis (PGPIS) generates realistic person images conditioned on a target...

31. [Disentangled Semantic Editing in Rectified Flow Transformers](https://huggingface.co/papers/2412.09611) - In this paper, we introduce FluxSpace, a domain-agnostic image editing method leveraging a represent...

32. [Entropy-informed weighting channel normalizing flow for ...](https://www.sciencedirect.com/science/article/abs/pii/S0031320325011045) - Normalizing Flows (NFs) are widely used in deep generative models for their exact likelihood estimat...

33. [[Literature Review] Normalizing Flow-Based Metric for ...](https://www.themoonlight.io/en/review/normalizing-flow-based-metric-for-image-generation) - "Master emotional talking face generation. This paper sets a new benchmark using normalizing flow an...

34. [Normalizing Flows are Capable Visuomotor Policy ...](https://arxiv.org/html/2509.21073v1) - Pioneering NFs architectures like RealNVP [17] and GLOW [20] use NFs to generate realistic images, h...

35. [deep-learning-notes/seminars/2018-10-Normalizing-Flows ...](https://github.com/kmkolasinski/deep-learning-notes/blob/master/seminars/2018-10-Normalizing-Flows-NICE-RealNVP-GLOW/README.md) - An implementation of the GLOW paper and simple normalizing flows lib. This code is suppose to be eas...

36. [GitHub - anishmadan23/glow_normalizing_flow](https://github.com/anishmadan23/glow_normalizing_flow) - Contribute to anishmadan23/glow_normalizing_flow development by creating an account on GitHub.

37. [A normalizing flow model for anomaly detection in human ...](https://discovery.researcher.life/article/nflowad-a-normalizing-flow-model-for-anomaly-detection-in-human-motion-animations/54a35c4ae4783e78b69724ccc07b12c5) - NFlowAD: A normalizing flow model for anomaly detection in human motion animations. Mar 1, 2026; Sig...

38. [Amortized Sampling with Transferable Normalizing Flows](https://neurips.cc/virtual/2025/poster/118702) - Efficient equilibrium sampling of molecular conformations remains a core challenge in computational ...

39. [Enhancing Low-resolution Image Representation Through Normalizing Flows](https://www.arxiv.org/abs/2601.06834) - Low-resolution image representation is a special form of sparse representation that retains only low...

40. [Revisiting Likelihood-Based Out-of-Distribution Detection by Modeling Representations](https://arxiv.org/html/2504.07793v1)

41. [[PDF] On Out-of-distribution Detection with Energy-based Models](http://www.gatsby.ucl.ac.uk/~balaji/udl2021/accepted-papers/UDL2021-paper-021.pdf) - Several density estimation methods have shown to fail to detect out-of-distribution (OOD) sam- ples ...

42. [Likelihood-free Out-of-Distribution Detection with Invertible ... - IJCAI](https://www.ijcai.org/proceedings/2021/292) - In this paper, we present a different framework for generative model--based OOD detection that emplo...

43. [[PDF] HYBOOD: a hybrid generative model for out-of-distribution detection ...](https://i-ops.co.kr/theme/cont_basic/contents/img/AAAI_2025_IOPS_HYBOOD.pdf) - We propose HYBOOD, a hybrid out-of-distribution model based on normalizing flow followed by a simple...

44. [Implicitly Learning Style and Content Disentanglement with ...](https://arxiv.org/abs/2508.03402) - We propose SCFlow, a flow-matching framework that learns bidirectional mappings between entangled an...

45. [SCFlow: Implicitly Learning Style and Content Disentanglement with Flow Models](https://www.arxiv.org/abs/2508.03402) - Explicitly disentangling style and content in vision models remains challenging due to their semanti...

46. [SCFlow: Implicitly Learning Style and Content ...](https://compvis.github.io/SCFlow/) - TL;DR: We introduce SCFlow (top), a bidirectional model that enables both style-content mixing and d...

47. [Disentangled Representation Learning via Flow Matching](https://arxiv.org/html/2602.05214v1) - In this work, we propose a flow-matching–based framework for disentangled representation learning, w...

48. [Disentangled Representation Learning via Flow Matching](https://arxiv.org/abs/2602.05214) - In this work, we propose a flow matching-based framework for disentangled representation learning, w...

49. [Disentangled Representation Learning via Flow Matching](https://www.arxiv.org/abs/2602.05214) - Disentangled representation learning aims to capture the underlying explanatory factors of observed ...

50. [Disentangled Semantic Editing in Rectified Flow Transformers - arXiv](https://arxiv.org/abs/2412.09611) - In this paper, we introduce FluxSpace, a domain-agnostic image editing method leveraging a represent...

51. [Disentangled Semantic Editing in Rectified Flow Transformers](https://www.themoonlight.io/en/review/fluxspace-disentangled-semantic-editing-in-rectified-flow-transformers) - The paper titled "FluxSpace: Disentangled Semantic Editing in Rectified Flow Transformers" introduce...

52. [wangjiangshan0725/RF-Solver-Edit: [🚀ICML 2025] ...](https://github.com/wangjiangshan0725/RF-Solver-Edit) - We propose RF-Solver to solve the rectified flow ODE with less error, thus enhancing both sampling q...

53. [[ICLR 2025] Text-to-Image Rectified Flow as Plug-and-Play ...](https://github.com/yangxiaofeng/rectified_flow_prior) - 2024/06/05: Code release. 2024/06/21: Add support for Stable Diffusion 3 (June, Medium version). 202...

54. [FluxSpace: Disentangled Semantic Editing in Rectified ...](https://arxiv.org/html/2412.09611v1) - In this paper, we introduce FluxSpace, a domain-agnostic image editing method leveraging a represent...

