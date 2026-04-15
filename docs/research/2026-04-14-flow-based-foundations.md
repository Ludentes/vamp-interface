# Flow-based foundations for face generation — framework lens

**Date:** 2026-04-14
**Context:** Face-Research [12-conclusions.md](../../../Face-Research/12-conclusions.md) prediction #3 — flow matching will displace diffusion for several near-real-time face tasks, following FLOAT's lead. User speculation: flow-based methods die and revive every ~2 years; the next face-adjacent revival may be on the horizon. This note surveys the load-bearing foundational papers and reads them through the [V2 framework](../../v2/framework/math-framework.md) rubric: identity (P4a rank preservation, P6 σ-injectivity), editorial (E1–E4), drift (D1 factor-mismatch, D5 reversibility), and algebraic closure (§1.2 / §P12).

**Sources merged:** tavily pro research 2026-04-14 (request `e29dcccd-7310-40a2-87fd-3e85ef62a045`) + perplexity research report [Flow-Based Generative Models for Faces and Face Animation (2022–2026)](Flow-Based%20Generative%20Models%20for%20Faces%20and%20Face%20Animation%20%282022–2026%29%20%20Foundations%20and%20Evaluation-Relevant%20Results.md). Where the two sources disagreed on arXiv IDs (Lipman FM, FluxSpace), perplexity's are used because they are the canonical preprint IDs and tavily's were derived URLs.

**2026-04-14 verification pass.** Three papers (RectifID, FluxSpace, FlowChef) were fetched and verified in depth after this note's first draft; corrections from that pass have been folded in below. Full verification record at [2026-04-14-rectifid-fluxspace-flowchef-verification.md](2026-04-14-rectifid-fluxspace-flowchef-verification.md). The most important correction: RectifID's "anchor" is **not** a reference face — it is a reference ODE trajectory used as a numerical stabilizer for classifier-guidance fixed-point iteration. V1's anchor-bridge is **not** a prior-art conflict with RectifID. The word "anchored" is a false friend.

---

## 1. Flow matching / CFM foundations

- **Lipman et al., "Flow Matching for Generative Modeling"** — arXiv:2210.02747, ICLR 2023. Simulation-free training of continuous normalizing flows by regressing a neural vector field on prescribed conditional probability paths. Shows equivalence to Schrödinger bridges for certain path choices. Meta reference implementation at `facebookresearch/flow_matching`.
  - **Framework lens:** the foundation document for the §1.2 "disentangled flow matching" algebraic-closure substrate. The vector field $v_\theta(x,t)$ defines an ODE whose flow map is bijective by construction → directly supports D5 reversibility and makes noise↔image correspondence deterministic.

- **Liu et al. — two companion papers.**
  - **"Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow"** — arXiv:2209.03003.
  - **"Rectified Flow: A Marginal Preserving Approach to Optimal Transport"** — arXiv:2209.14577.
  - Together: trains an ODE along a straight interpolant between noise and data; iterative "Reflow" straightens trajectories, enabling 1–4 step sampling; the second paper proves marginal preservation and connects to optimal transport convergence. Code at `gnobitab/RectifiedFlow`.
  - **Framework lens:** straight-line trajectories between noise and data are the strongest structural argument for P4a rank preservation in any generative-model family. Linear interpolation in noise → a path that stays near the data manifold → neighborhood structure should survive. Combined with determinism, this also underwrites σ-injectivity (P6). **The OT companion paper (2209.14577) is the direct Brenier-transport anchor** — the same theorem §2.6 already cites for the identity channel now has its flow-training analogue.

- **Tong et al., "Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport"** — arXiv:2302.00482, TMLR 2024. Introduces OT-CFM: minibatch Sinkhorn OT to select training pairs, variance reduction, straighter trajectories without reflow.
  - **Framework lens:** second Brenier-anchored foundation alongside Liu's 2209.14577. If we ever train flow components ourselves, OT-CFM is the objective to use for framework-native justification.

- **Meta "Flow Matching Guide and Code"** — arXiv:2412.06264 (Dec 2024). Consolidated tutorial + reference code + best practices. First-stop reading when we need to run something.

- **Laplacian Multi-scale Flow Matching (LapFlow)** — arXiv:2602.19461, ICLR 2026. Multi-scale Laplacian-pyramid + transformer flow matching to 1024² with improved FID vs. single-scale flows. **Read-relevant** — this is the architecture that brings flow matching to native face resolution.

- **Blockwise Flow Matching** — NeurIPS 2025. Temporal-block partitioning of the trajectory, specialized velocity networks per block, multi-× inference speedup.

- **FastFlow** — arXiv:2602.11105 (2026 preprint). Bandit-inference scheduling + higher-order ODE solvers; deterministic noise↔data correspondences useful for inversion/editing.

- **"Image Generation Models: A Technical History"** — arXiv:2603.07455. Section 4 treats normalizing flows as the likelihood-based alternative branch; useful historical-structural framing.

- **MIT CSAIL "Flow Matching and Diffusion Models — 2026 Version"** — `diffusion.csail.mit.edu`. Course-level tutorial; good for establishing shared vocabulary if we discuss flow matching across sessions.

## 2. Rectified flow applied to faces and at-scale backbones

- **Stable Diffusion 3 — "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis"** — Esser et al., arXiv:2403.03206. Rectified-flow formulation with MM-DiT transformer backbone scaled to 8B parameters. Middle-weighted trajectory sampling schedule. Stability AI released SD3-family models.
  - **Framework lens:** critical reattribution — **Flux v3, our current production baseline with `r=+0.914`, is a rectified-flow model**, not a plain diffusion model. We have been describing the v3 baseline as a "diffusion era" generative AI result; it is actually the most successful public rectified-flow face generator. Its measurement is structurally explained by straight-line trajectory properties, not by "diffusion happens to work." Framework §3.1 entries should be re-phrased: the axis is "rectified flow vs. W-space GAN vs. pure diffusion," not "diffusion vs. GAN."

- **Flux (Black Forest Labs, 2024).** Rectified-flow transformer family; the SDXL-analogue in the rectified-flow world. Public weights; used as backbone by FlowChef, FluxSpace, RF-Solver-Edit.

- **FlowChef** — arXiv:**2412.00100**, ICCV 2025 (Patel, Wen, Metaxas, Yang), `FlowChef/FlowChef`. ArXiv title: *Steering Rectified Flow Models in the Vector Field for Controlled Image Generation*; ICCV title adds the "FlowChef" brand. **Training-free and gradient-free through the ODE solver**: exploits straight trajectories + smooth Jacobians of rectified flow to derive a "gradient skipping" approximation that steers the denoising trajectory without backprop through the ODE integration. Demonstrated tasks: classifier guidance, linear inverse problems (super-resolution, inpainting, deblurring), inversion-free image editing. Base models: Flux.1-dev, InstaFlow, SD3. Baselines: FreeDoM, MPGD, LGD.
  - **Framework lens (revised 2026-04-14 post-verification).** FlowChef is a **general-purpose controlled-generation primitive** on rectified flows, NOT a face-specialized editor. Neither the project page nor the arXiv abstract features face/portrait results prominently. To use FlowChef as a vamp-interface editorial mechanism we would have to define a classifier (or inverse-problem objective) corresponding to the semantic axis we want, which is strictly more work than FluxSpace's prompt-pair-orthogonal-projection mechanism. FlowChef's real relevance to us is as **the primitive that becomes necessary when we need classifier-guided steering with an arbitrary discriminator** — ArcFace-guided identity preservation, or an uncanny-valley regressor, or a factor-mismatch classifier driving drift. These are not editorial operations; they are alternative identity-channel or drift-channel conditioning pathways. Worth reading as the "how do we inject an arbitrary differentiable objective into Flux without training anything?" reference, not as a drop-in editorial axis. **Medium priority read.**

- **FluxSpace** — arXiv:2412.09611, **CVPR 2025 confirmed** (Dalva, Venkatesh, Yanardag), pp. 13083–13092 in the CVF open-access proceedings. Domain-agnostic semantic editing on frozen Flux.1 via **orthogonal decomposition of joint-attention outputs**: given a base condition `c` and an edit condition `c_e`, compute the edit signal by subtracting the projection onto the base direction — `l'θ(x,c_e,t) = lθ(x,c_e,t) − proj_φ lθ(x,c_e,t)` — then add it at inference scaled by `λ_fine ∈ [0,1]`: `l̂θ(x,c,c_e,t) = lθ(x,c,t) + λ_fine · l'θ(x,c_e,t)`. Dual-level variant: fine edits use attention-output projections, coarse edits use projections on pooled CLIP embeddings before modulation. Attributes demonstrated on faces: eyeglasses, smiles, age, gender, beards, sunglasses, stylization. Baselines: LEDITS++, TurboEdit, RF-Inversion, Sliders-FLUX. Training-free, no fine-tuning, no LoRA, no optimization loop.
  - **Framework lens (tightened 2026-04-14 post-verification).** Paper-reported metrics are **CLIP-I and DINO content preservation plus a user study (4.19/5)** — NOT ArcFace identity similarity. CLIP-I ≈ 0.94 (eyeglasses) and DINO ≈ 0.94 are image-content-preservation numbers, approximating but not equal to E3 "identity leakage." Framework E3 scoring of FluxSpace will require **our own ArcFace measurement** on edited vs. unedited pairs; we cannot inherit the paper numbers directly. Everything else holds: E1 readable (named attribute axes), E2 ablatable (`λ_fine` is the exact ablation parameter), E4 scalable (training-free). This is the **highest-leverage editorial-channel candidate** the survey found.
  - **Framework lens:** editorial-channel candidate with explicit linear/orthogonal structure — this is as close as the literature gets to "W-space arithmetic but on a flow backbone." E3 identity-leakage measurement is reported in the paper. **High priority read.**

- **RectifID — "Personalizing Rectified Flow with Anchored Classifier Guidance"** — arXiv:2405.14677, **NeurIPS 2024** (Sun et al.). Training-free identity-preserving personalization via classifier guidance on rectified flows. Fixed-point formulation of classifier guidance; "anchored" refers to anchoring the **fixed-point iteration to a reference (un-guided) ODE trajectory** for numerical stability and convergence (Proposition 2). Identity signal comes from a separate off-the-shelf discriminator (**ArcFace** for faces, **DINOv2** for general objects) computing similarity to a reference image. Base model: **piecewise rectified flow fine-tuned from SD 1.5** (not Flux). Eval: 200 CelebA-HQ reference images × 20 prompts; reports ArcFace ID similarity 0.5930 at 100 iterations vs InstantID 0.5806. Code: `github.com/feifeiobama/RectifID`.
  - **Framework lens (retracted-and-reframed 2026-04-14 post-verification).** Prior note claimed RectifID is a "direct structural analogue of V1's anchor-bridge in rectified-flow space." **This claim is retracted.** The word "anchor" is a false friend: RectifID's anchor is an **un-guided reference ODE trajectory** used as a numerical stabilizer for the classifier-guidance fixed-point iteration. V1's anchor is a **fixed neutral face image** serving as an img2img starting point for seeded perturbation. These are different mechanisms on different axes — V1 is img2img-seeded perturbation where drift is measured by distance from the anchor image, RectifID is classifier-guided text-to-image where the anchor does not appear in any distance metric. Also: RectifID's base model is **SD 1.5 PeRFlow**, not Flux — it is not a drop-in on our current backbone. V1 is **not** implicitly reinventing RectifID; there is no prior-art conflict. RectifID remains worth knowing as an **unrelated primitive**: the training-free recipe for injecting an off-the-shelf discriminator (like ArcFace) as a personalization signal into a rectified flow. If V2 ever wants reference-photo identity preservation on top of a rectified flow, RectifID is the relevant foundation. vamp-interface's actual use case (generate a face from a job embedding, no reference photo) is a different problem and RectifID does not directly address it. **Medium-low priority read; demoted from "high priority anchor-bridge analogue."**

- **Taming Rectified Flow for Inversion and Editing** (2024, referenced in FluxSpace related work). Improves inversion stability — critical for identity-preserving edits.

- **RF-Solver-Edit** — ICML 2025, `wangjiangshan0725/RF-Solver-Edit`. Lower-error ODE solver for rectified-flow, better sampling quality and editing fidelity.

- **Stable Flow: Vital Layers for Training-Free Image Editing** (2024). Identifies which transformer layers in a rectified-flow model matter for editing; training-free editing recipes.

- **Latent PMRF (Latent Posterior-Mean Rectified Flow)** — arXiv:2507.00447. Rectified flow for face restoration with explicit identity preservation. Reports improved identity fidelity vs. non-PMRF baselines.

- **Optimal Transport for Rectified Flow Image Editing** — arXiv:2508.02363. Unifies inversion-based and direct editing via transport theory. Relevant to E2 editorial-ablation and D5 drift-reversibility simultaneously.

- **Real-Time Person Image Synthesis Using a Flow Matching Prior** — arXiv:2505.03562. Pose-conditioned person-image generator via latent flow matching. Targets real-time. Code availability unclear.

- **Rectified-flow stability caveat — Reflow / RCA** (OpenReview). Iterative rectified-flow training without real-data mixing collapses. Reverse Collapse-Avoiding Reflow and Online RCA mitigate. Matters only if we train our own reflow; if we consume Flux/SD3 weights this is upstream risk, not ours.

## 3. Face-specific flow — talking-head / animation lineage

- **FLOAT** — arXiv:2412.01064, ICCV 2025. Flow matching in a **learned orthogonal motion latent space** for audio-driven talking portrait. Transformer velocity predictor. Decouples identity from motion by construction. Code + weights at `deepbrainai-research/float`.
  - **Framework lens:** not directly our generator (we don't need audio-driven motion) but the **orthogonal motion latent** is methodologically important — it demonstrates that a flow model can be architected to separate identity from expression. That structure is P7 disentanglement + D1 factor separation handed to you by the architecture.

- **DEMO — "Disentangled Motion Latent Flow Matching for Fine-Grained Controllable Talking Portrait Synthesis"** — arXiv:2510.10650. Direct FLOAT successor with explicit factor disentanglement.

- **DyStream** — arXiv:2512.24408 (v1 Dec 2025, v2 Feb 2026). Streaming dyadic talking heads via flow-matching autoregressive. ~34 ms/frame. Direct streaming successor to FLOAT. Already documented in [Face-Research 06-talking-heads.md](../../../Face-Research/06-talking-heads.md:195).

## 4. Normalizing flows (Glow/RealNVP/IAF lineage) — revival status

**Partial revival, confined to specialized roles.** The Glow/RealNVP architectural line did not come back as a face *generator*. It revived as **exact-likelihood encoders + OOD tools** in hybrid stacks. Key 2024–2026 representatives:

- **EIW-Flow (Entropy-Informed Weighting Channel Normalizing Flow)** — Pattern Recognition 2025. Regularized entropy-guided channel shuffling while preserving invertibility. Improved density estimation and sample quality on image datasets.
- **LR2Flow — "Enhancing Low-resolution Image Representation Through Normalizing Flows"** — arXiv:2601.06834 (2026). Flows for low-res representations improving downstream generation.
- **Amortized Sampling with Transferable Normalizing Flows** — NeurIPS 2025. Flows trained to approximate expensive samplers in latent space — use flows for fast-ish sampling while retaining invertibility + exact density.
- **"Normalizing Flows are Capable Visuomotor Policy Generators"** — arXiv:2509.21073, NeurIPS 2025. Not face-relevant but shows flows re-establishing competitive performance on structured outputs.
- **NFlowAD** (2026). Normalizing-flow model for anomaly detection in human motion animations — exact bijection used explicitly for anomaly scoring.

**Interpretation — the user's 2-year revival hypothesis.** The 2024–2026 flow revival *happened*, but it happened as **rectified flow / flow matching**, not as a Glow-lineage bijective-architecture revival. The invertible-by-construction exact-likelihood family absorbed its *spirit* (continuous flows between noise and data) without its architectural commitment (layer-wise exact bijection). If the user's intuition about a second revival being on the horizon specifically for faces is right, the form it would take is **bijective-architecture Glow-lineage work on high-res faces with released checkpoints** — and neither source found such a paper. **The conceptual slot is open.**

## 5. Invertible generators for OOD / uncanny-valley detection

**Not a total gap — but the key result is a negative one.** Naive image-space likelihood OOD detection with flows **is known to fail**: flows routinely assign higher likelihood to OOD data than in-distribution data. This is documented across multiple independent findings. Papers that work around it:

- **Ahmadian & Lindsten — "Likelihood-free Out-of-Distribution Detection with Invertible Generative Models"** — IJCAI 2021. Uses invertible models as *representation learners* and applies one-class SVMs on the learned representations rather than scoring raw likelihood. Outperforms likelihood-only baselines.
- **HYBOOD — "A hybrid generative model for OOD detection"** — AAAI 2025. Normalizing-flow density + linear classifier, using likelihood ratios that correct for image complexity (following Serrà et al. 2020).
- **"Revisiting Likelihood-Based OOD Detection by Modeling Representations"** — arXiv:2504.07793 (2025). Models the distribution of ViT features with a generative model; **representation-space likelihood is reliable where image-space likelihood is not**.
- **"Normalizing Flow-Based Metric for Image Generation"** (Moonlight review, 2025). Flows as learned similarity measures — flow-induced distances in latent space correlate with semantic similarity.

**Framework lens.** The framework's §2.5 D1 factor-mismatch reframing currently plans to measure uncanny via **hand-designed factor decompositions** (identity/expression/symmetry/anatomy/texture). The flow-OOD literature offers a complementary measurement path: **train an invertible flow on a face representation space (e.g., ArcFace features) and score representation-space log-likelihood**. Three things to note:

1. **Raw image-space flow likelihood is contraindicated** — do not try this. The negative result is the point.
2. **Representation-space likelihood is viable** and matches our existing ArcFace IR101 infrastructure (our §V1 baseline is already measured in ArcFace feature space, so this is one additional step away, not a full retraining).
3. **Nobody has applied this to faces specifically for uncanny detection.** The foundational papers use CIFAR/ImageNet for generic OOD. Applying the representation-space flow-OOD recipe to faces to produce a uncanny metric is an unclaimed research opening.

**Correction to my tavily-only draft.** I previously called this "fully open." That was wrong. The OOD literature exists and has figured out the likelihood-fails-on-raw-pixels issue; the gap is specifically "no face-uncanny application," not "no method."

## 6. Disentangled flow matching and algebraic latent operations

**Tighter than I thought.** Two papers explicitly formalize latent algebra in flow models — one of them directly addresses faces:

- **"Disentangled Representation Learning via Flow Matching"** — arXiv:2602.05214 (2026). **Factor-conditioned flows in a compact latent space with an orthogonality regularizer between factors**, encouraging non-overlapping subspaces for different attributes. Evaluated on disentanglement metrics and CelebA. Paper states it "supports compositional editing via algebraic operations in the latent space." Code may be TBD but algorithmic pseudo-code is in the paper.
  - **Framework lens:** this is the most direct match to framework §1.2 / §P12 algebraic-closure speculation. It is a flow-matching framework that explicitly claims algebraic latent operations, validated on a face dataset. **Highest priority read for anything algebraic-closure adjacent.** This closes the "no formal flow-based face arithmetic" claim I made in the tavily-only draft.

- **SCFlow — "Implicitly Learning Style and Content Disentanglement with Flow Models"** — arXiv:2508.03402, ICCV 2025. Flow-matching bidirectional mapping between entangled and disentangled representations **without explicit disentanglement supervision**. Trained only on style-content merging; disentanglement emerges naturally from the invertible structure. Generalizes zero-shot to ImageNet-1k and WikiArt. Project page at `compvis.github.io/SCFlow`.
  - **Framework lens:** style/content decomposition maps onto vamp-interface's identity/editorial split. If SCFlow's implicit disentanglement is as clean as claimed, it is a candidate editorial-channel mechanism *trained from scratch* rather than projected after-the-fact like FluxSpace. Worth comparing head-to-head with FluxSpace.

- **FluxSpace** (already covered §2) also contributes here — orthogonal attention-space projections are algebraic latent operations on a deployed rectified-flow face backbone.

**Correction to my tavily-only draft.** I previously said "no formal flow-based face arithmetic benchmarks exist." That was wrong — arXiv:2602.05214 explicitly does this on CelebA. The remaining gap is narrower: no **A − B + C identity-preservation benchmark with the same protocol used in the StyleGAN latent-arithmetic literature** exists in flow models, to my knowledge. That is a more precise and smaller gap.

## 7. Framework-view synthesis

Given the framework, four conclusions:

1. **Flux v3 already IS a rectified-flow face generator.** The v3 baseline measurement is a rectified-flow result. Framework §3.1 should reattribute it accordingly, and the "straight-line trajectory → rank preservation" argument becomes the structural P4a justification for why the measurement worked.

2. **One editorial-channel candidate lands in our lap (not three).** Post-verification (2026-04-14), **FluxSpace alone** is the near-term editorial-channel candidate: it works on frozen Flux.1 and has a drop-in prompt-pair orthogonal-projection mechanism. FlowChef is a general-purpose controlled-generation primitive (not face-specialized), and adapting it to editorial axes requires defining a classifier per axis — more work, and the wrong tool for this job. RF-Solver-Edit remains an inversion-quality primitive. The pre-verification framing "three editorial mechanisms on the same backbone" was too loose; only FluxSpace qualifies as drop-in editorial.

3. **D1 drift has a complementary measurement path via representation-space flow density.** Train a normalizing flow on ArcFace IR101 feature vectors of "natural" faces, score generated faces under the flow's log-density. The framework's hand-designed D1 factor decomposition and a learned flow-OOD score would be **independent measurements of the same thing**, and their agreement or disagreement is diagnostic. Important caveat: raw image-space flow likelihood is contraindicated; do it on representations only. This goes in framework §5 experiments as a low-priority exploratory measurement (requires training a flow; not free).

4. **Algebraic closure has a 2026 reference implementation.** arXiv:2602.05214 is the closest foundation paper we have for §1.2 / §P12. Even if we do not adopt it, citing it elevates the framework's algebraic-closure note from "speculation" to "speculation supported by one explicit existence proof on CelebA." Updating the framework to cite it is a minor change with good hygiene value.

## Reading priority (final)

1. **Liu et al. Rectified Flow — both 2209.03003 + 2209.14577.** Structural P4a argument + OT companion.
2. **Esser et al. SD3/Flux — arXiv:2403.03206.** Understand the backbone we are running.
3. **FluxSpace — arXiv:2412.09611.** Only drop-in editorial-channel candidate on current backbone. Verified 2026-04-14; see [verification doc](2026-04-14-rectifid-fluxspace-flowchef-verification.md).
4. **Disentangled Representation Learning via Flow Matching — arXiv:2602.05214.** Algebraic closure foundation, CelebA-evaluated.
5. **FlowChef — arXiv:2412.00100, ICCV 2025.** General-purpose steering primitive; read when we need classifier-guided injection with an arbitrary discriminator (ArcFace, uncanny regressor), not as an editorial drop-in.
6. **Lipman Flow Matching — arXiv:2210.02747.** Theoretical foundation doc.
7. **Tong OT-CFM — arXiv:2302.00482.** Brenier-anchored training objective.
8. **Ahmadian & Lindsten (IJCAI 2021) + arXiv:2504.07793.** OOD recipe for representation-space flow density, if we pursue the D1 complementary-measurement path.
9. **RectifID — arXiv:2405.14677, NeurIPS 2024.** Demoted from "prior-art for V1 anchor-bridge" to "unrelated reference-photo personalization primitive." Read only if V2 ever needs reference-photo identity preservation on rectified flow.
10. **LapFlow — arXiv:2602.19461.** Only if we need multi-scale flow matching for a future native-resolution face backbone.

## Surprising finds (final, after merge)

- **Flux v3 is a rectified-flow model.** Our baseline measurement is secretly the most successful public rectified-flow face result. Framework reattribution follow-up.
- **Training-free editorial steering on Flux already exists** via FluxSpace (CVPR 2025, verified). The editorial channel is not a 6-month research project; it is an afternoon of adapting a published technique. FlowChef is a different, more general primitive and not a drop-in editorial — correction to the pre-verification claim.
- ~~RectifID = anchor-bridge in rectified-flow space.~~ **Retracted.** V1's anchor-bridge and RectifID's anchoring are structurally different mechanisms sharing only the word "anchor" — RectifID anchors a fixed-point iteration to a reference ODE trajectory for numerical stability, V1 anchors img2img to a reference face image for seeded perturbation. V1 is not implicitly reinventing RectifID. Full reasoning in [verification doc](2026-04-14-rectifid-fluxspace-flowchef-verification.md).
- **Image-space flow likelihood for OOD is a known failure mode.** If anyone proposes "use a flow to score uncanny," we should cite the negative result and redirect to representation-space recipes.
- **arXiv:2602.05214 (2026) gives flow-matching algebraic operations on CelebA.** §P12 algebraic closure now has an existence proof to cite.
- **Bijective Glow-lineage revival for faces did not happen.** The conceptual slot (scalable exact-likelihood invertible generator on high-res faces with released checkpoints) is still unoccupied. If the user's "revival every 2 years" intuition is right about its next form, this is the form.

## Evidence gaps (revised)

These are genuine gaps — narrower than my tavily-only draft claimed but still present:

- **No 2024–2026 primary paper revives Glow/RealNVP as a scalable exact-likelihood high-resolution face generator with released checkpoints.** The bijective-architecture branch did not come back for face synthesis.
- **Representation-space flow-OOD has not been applied to face uncanny detection specifically.** The generic method exists (Ahmadian & Lindsten, HYBOOD, 2504.07793); the face-uncanny application does not.
- **No A − B + C identity-preservation benchmark with the StyleGAN-era protocol exists in flow models.** arXiv:2602.05214 does algebraic operations on CelebA but not that specific protocol.

Any framework claim that load-bears on these three gaps being filled is currently resting on assumption.

## References

Consolidated from both source reports. IDs preferred where available.

- Lipman et al., Flow Matching — arXiv:2210.02747 · ICLR 2023 · https://arxiv.org/abs/2210.02747
- Lipman et al., Flow Matching Guide and Code — arXiv:2412.06264 · https://arxiv.org/abs/2412.06264
- `facebookresearch/flow_matching` · https://github.com/facebookresearch/flow_matching
- Liu et al., Flow Straight and Fast: Rectified Flow — arXiv:2209.03003 · ICLR 2023 spotlight · https://arxiv.org/abs/2209.03003
- Liu et al., Rectified Flow: Marginal Preserving OT — arXiv:2209.14577 · https://arxiv.org/abs/2209.14577
- `gnobitab/RectifiedFlow` · https://github.com/gnobitab/RectifiedFlow
- Tong et al., OT-CFM — arXiv:2302.00482 · TMLR 2024 · https://arxiv.org/abs/2302.00482
- `atong01/conditional-flow-matching` · https://github.com/atong01/conditional-flow-matching
- LapFlow — arXiv:2602.19461 · ICLR 2026 · https://arxiv.org/abs/2602.19461
- Blockwise Flow Matching — NeurIPS 2025 · https://neurips.cc/virtual/2025/37053
- FastFlow — arXiv:2602.11105 · https://arxiv.org/html/2602.11105v1
- "Image Generation Models: A Technical History" — arXiv:2603.07455
- MIT CSAIL FM/Diffusion 2026 course · https://diffusion.csail.mit.edu
- Esser et al., SD3 Scaling Rectified Flow Transformers — arXiv:2403.03206 · https://arxiv.org/abs/2403.03206
- SD3 research paper · https://stability.ai/news-updates/stable-diffusion-3-research-paper
- FlowChef — arXiv:2412.00100 · ICCV 2025 · `FlowChef/FlowChef` · https://arxiv.org/abs/2412.00100 · https://github.com/FlowChef/flowchef
- FluxSpace — arXiv:2412.09611 · CVPR 2025 · https://arxiv.org/abs/2412.09611 · CVF: https://openaccess.thecvf.com/content/CVPR2025/html/Dalva_FluxSpace_Disentangled_Semantic_Editing_in_Rectified_Flow_Models_CVPR_2025_paper.html
- RectifID — arXiv:2405.14677 · **NeurIPS 2024** · https://arxiv.org/abs/2405.14677 · `feifeiobama/RectifID`
- RF-Solver-Edit — ICML 2025 · `wangjiangshan0725/RF-Solver-Edit`
- Latent PMRF — arXiv:2507.00447
- OT for Rectified Flow Editing — arXiv:2508.02363
- Real-Time Person Image Synthesis w/ FM Prior — arXiv:2505.03562
- RCA / model-collapse in rectified flows · OpenReview `95d2147205827749dd8d53166d2dac2709425923`
- FLOAT — arXiv:2412.01064 · ICCV 2025 · `deepbrainai-research/float`
- DEMO — arXiv:2510.10650
- DyStream — arXiv:2512.24408
- AlignYourFlow — NVIDIA Toronto AI · https://research.nvidia.com/labs/toronto-ai/AlignYourFlow/
- MeanFlow / Re-MeanFlow — arXiv:2511.23342
- EIW-Flow — Pattern Recognition 2025 · ScienceDirect S0031320325011045
- LR2Flow — arXiv:2601.06834
- Amortized Sampling with Transferable Normalizing Flows — NeurIPS 2025
- NFs are Capable Visuomotor Policy Generators — arXiv:2509.21073 · NeurIPS 2025
- NFlowAD (2026) — human motion anomaly detection via NF
- Ahmadian & Lindsten — Likelihood-free OOD with Invertible Models — IJCAI 2021 · https://www.ijcai.org/proceedings/2021/292
- HYBOOD — AAAI 2025
- Revisiting Likelihood-Based OOD via Representations — arXiv:2504.07793
- SCFlow — arXiv:2508.03402 · ICCV 2025 · https://compvis.github.io/SCFlow/
- Disentangled Representation Learning via Flow Matching — arXiv:2602.05214
- `awesome-normalizing-flows` · https://github.com/janosh/awesome-normalizing-flows
- Taming Rectified Flow for Inversion and Editing (2024) — referenced in FluxSpace related work
- Stable Flow: Vital Layers for Training-Free Image Editing (2024)
