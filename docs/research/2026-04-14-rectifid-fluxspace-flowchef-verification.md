# Verification: RectifID, FluxSpace, FlowChef

**Date:** 2026-04-14
**Purpose:** Due diligence on the three flow-based face-adjacent papers that the [flow-based foundations](2026-04-14-flow-based-foundations.md) note flagged as "high-priority read" before any framework edits. Read each paper, verify claims I made in the earlier note, and document corrections.

**Verdict at a glance:**

| Paper | Prior claim held up? | Action on prior claim |
|---|---|---|
| RectifID | **No** — "anchor-bridge analogue" is wrong in the sense that matters | Retract the claim. Keep RectifID as an unrelated primitive worth citing. |
| FluxSpace | **Yes, with one tightening** | Prior claim survives; tighten "identity-leakage metric" wording. |
| FlowChef | **Partially** — prior framing was too face-specific | Reframe as general-purpose rectified-flow steering, not a face editor. |

---

## 1. RectifID — arXiv:2405.14677

### Fixed facts

- **Full title:** *RectifID: Personalizing Rectified Flow with Anchored Classifier Guidance*
- **Authors:** Zhicheng Sun, Zhenhao Yang, Yang Jin, Haozhe Chi, Kun Xu, Liwei Chen, Hao Jiang, Yang Song, Kun Gai, Yadong Mu
- **Venue:** **NeurIPS 2024** (I had referred to it as "HF paper page arXiv:2405.14677" in the flow-foundations note without committing to a venue; the correct venue is NeurIPS 2024, not CVPR 2025)
- **Code:** `github.com/feifeiobama/RectifID`
- **Base model:** **piecewise rectified flow finetuned from Stable Diffusion 1.5**, K=4 time windows. **Not Flux.** A secondary SD 2.1 variant is reported.

### What the paper actually does

Classifier-guidance personalization for rectified flows, training-free, using off-the-shelf discriminators (ArcFace for faces, DINOv2 for general objects) to steer generation toward the identity of a reference image. The novelty is making vanilla classifier guidance work for rectified flow by reformulating it as a fixed-point iteration, and stabilizing that iteration by anchoring it to a reference ODE trajectory.

Key phrase verified from arxiv HTML v3 §3:

> "we propose to anchor the classifier-guided flow trajectory to a reference trajectory to improve the stability of its solving process"

The anchor iteration has the form:

$$\hat{z}_1 = z_1 + s \cdot [\nabla_{z_0} z_1]\, \nabla_{\hat{z}_1} \log p(c \mid \hat{z}_1)$$

where $z_1$ is the endpoint of the **reference (un-guided) ODE trajectory**. Proposition 2 gives a convergence guarantee under the straightness assumption.

Identity classifier used on faces:

$$p(c \mid \hat{z}_1^{(i)}) = \text{sim}\bigl(f \circ g(\hat{z}_1^{(i)}),\ f \circ g(z_{\text{ref}})\bigr)$$

where $f$ is the discriminator (ArcFace) and $g$ is a region detector on the generated image.

### The "anchor" is NOT a face

This is the critical verification result. Re-read the definition carefully: **the anchor is the un-guided reference ODE trajectory**, used as a numerical stabilizer for the classifier-guided fixed-point iteration. The anchor is not a reference face image. The reference face image is used separately, by the classifier $p(c \mid \hat{z}_1)$, to compute the similarity signal that guides the trajectory.

### Comparison to V1 anchor-bridge

| Aspect | V1 anchor-bridge (vamp-interface) | RectifID |
|---|---|---|
| What "anchor" refers to | A fixed neutral face image (`output/phase1/phase1_anchor.png`, seed=42) | An un-guided reference ODE trajectory |
| Where the anchor lives | In pixel space, as an img2img input | In latent/flow trajectory space, as a stabilizer for fixed-point iteration |
| Identity preservation mechanism | Denoising strength perturbs the anchor face; `sus_level` scales the perturbation | ArcFace similarity between generated image and a separate reference image, injected via classifier guidance |
| Conditioning input | `job_id → seed` (deterministic) + archetype prompt | Text prompt + reference image |
| Base model family | Flux (rectified flow at Flux-dev scale) | SD 1.5 piecewise-rectified-flow |
| Role of anchor in distance metric | `d(face, anchor) = √(2·(1 − cos_sim))` measures drift from anchor | Anchor does not appear in a distance metric; only in the trajectory-solving equation |

**These are different mechanisms that share the word "anchor."** V1 is img2img-with-fixed-start-image. RectifID is classifier-guided text-to-image with fixed-point numerical stabilization.

### Retraction

The flow-foundations note (§2 reading priority, §7 surprising finds) claims:

> "RectifID = anchor-bridge in rectified-flow space. Direct structural analogue to V1. We should read it and see if V1's mechanism is implicitly reinventing a published technique."

**Retracted.** V1 is not implicitly reinventing RectifID. The word "anchored" in RectifID's title is unrelated to V1's anchor-bridge. V1's mechanism (img2img-seeded perturbation of a fixed face image, drift measured by ArcFace distance from that image) has no prior-art conflict with RectifID.

RectifID is still worth citing, but as a **different** primitive: a training-free way to add classifier-guidance identity preservation on top of a rectified-flow model via a stable fixed-point solver. For vamp-interface it is not a drop-in for Flux v3 (wrong base model) and not an anchor-bridge analogue. If V2 ever wants "generate a face that preserves the identity of a reference photo via ArcFace guidance," RectifID is the relevant primitive; vamp-interface's actual use case (generate a face from a job embedding, no reference photo) is a different problem.

### Numbers the paper reports (for the record)

- 200 CelebA-HQ reference images × 20 prompts evaluation set
- Primary reported identity similarity (ArcFace, SD 1.5 variant, 100 iterations): **0.5930**, vs InstantID 0.5806
- Prompt consistency (CLIP): 0.2933
- SD 2.1 variant: 0.5034 ID / 0.3151 prompt consistency

These numbers are modest and on a small eval set. They are not directly comparable to vamp-interface's Flux v3 measurements because (a) different base model, (b) different task (reference-image identity preservation vs. embedding-to-face), (c) ArcFace cosine similarity rather than anchor-distance.

---

## 2. FluxSpace — arXiv:2412.09611

### Fixed facts

- **Full title:** *FluxSpace: Disentangled Semantic Editing in Rectified Flow Transformers*
- **Authors:** Yusuf Dalva, Kavana Venkatesh, Pinar Yanardag
- **Venue:** **CVPR 2025 confirmed**, pp. 13083–13092 in the CVF open-access proceedings
- **Code:** project page at `fluxspace.github.io` (editing toolkit public; full pipeline status from the page was not fully extracted)
- **Base model:** **frozen Flux.1** (no fine-tuning, no LoRA)

### Verified mechanism

The paper operates on the **outputs of Flux's joint attention layers** inside the MM-DiT blocks — i.e., the attention output tensors $\ell_\theta(x, c, t)$ that come out of the joint (text+image) attention operation at each timestep.

The edit is an **orthogonal decomposition**, not a subtraction:

$$\ell'_\theta(x, c_e, t) = \ell_\theta(x, c_e, t) - \text{proj}_\phi\, \ell_\theta(x, c_e, t)$$

where $c_e$ is the edit condition, $\phi$ is a base/null condition, and the projection removes the base-condition component, leaving the orthogonal semantic signal. The final edit is then applied at inference:

$$\hat{\ell}_\theta(x, c, c_e, t) = \ell_\theta(x, c, t) + \lambda_{\text{fine}}\, \ell'_\theta(x, c_e, t)$$

with $\lambda_{\text{fine}} \in [0, 1]$ scaling the edit strength. The paper also has a coarse-level variant that applies orthogonal projection to pooled CLIP embeddings before modulation, and an optional attention-derived spatial mask.

### What it is training-free on

Fully inference-time on frozen Flux weights. The paper explicitly states the method operates "without altering Flux's pretrained parameters." No LoRA, no optimization loop, no fine-tuning.

### Face attributes and metrics

**Attributes demonstrated:** eyeglasses, smiles, age, gender, beards, sunglasses, plus stylization (comic / 3D cartoon).

**Reported image-similarity metrics (fine-grained edits):**
- CLIP-I: 0.9417 (eyeglasses), 0.9038 (smile)
- DINO: 0.9402 (eyeglasses), 0.9347 (smile)
- User preference: 4.19 / 5.0

**Baselines compared:** LEDITS++, TurboEdit, RF-Inversion, Sliders-FLUX.

### Tightening of my prior claim

The flow-foundations note said "E3 identity-leakage measurement is reported in the paper." This is **partially true and needs tightening**. The metrics I verified are **CLIP-I and DINO** (content / image similarity) and a user study — not **ArcFace ID similarity** specifically. CLIP-I and DINO approximate "how much of the unedited image is preserved overall" and indirectly upper-bound identity leakage, but they are not identity-specific. A proper framework E3 score would want ArcFace (or similar face-ID) cosine on the unedited-vs-edited pair. I did not find that number in the verification fetch; it may exist in the full PDF tables. **Treat FluxSpace's reported metrics as "image-content preservation" rather than "identity preservation."**

This does not invalidate the prior claim that FluxSpace is a strong editorial-channel candidate. It does mean: if we use FluxSpace under framework E1–E4 scoring, E3 will need us to **run our own ArcFace measurement** on FluxSpace outputs rather than inherit a paper number.

### What holds up from the prior claim

- CVPR 2025 ✅
- Orthogonal projection in attention-layer features ✅
- Training-free on frozen Flux ✅
- Attribute axes demonstrated (eyeglasses, age, etc.) ✅
- Direct editorial-channel candidate on our existing backbone ✅

### Framework E1–E4 pre-scoring

Even without reading the full PDF tables, the verified mechanism supports preliminary scoring:

- **E1 readability:** strong. Named semantic axes (eyeglasses, smile, age) with orthogonal structure. Axis names are explicit.
- **E2 information contribution (ablation):** strong. $\lambda_{\text{fine}} \in [0,1]$ is exactly the ablation parameter; set to 0 to ablate the edit, to 1 to apply it.
- **E3 identity leakage:** **unverified**. Paper reports image-content preservation (CLIP-I, DINO), not face-ID similarity. Measurement needed by us.
- **E4 scalability:** strong. Training-free; marginal cost per edit is a forward pass modification, not a new model.

---

## 3. FlowChef — arXiv:2412.00100

### Fixed facts

- **arXiv title:** *Steering Rectified Flow Models in the Vector Field for Controlled Image Generation*
- **ICCV 2025 title:** *FlowChef: Steering of Rectified Flow Models for Controlled Generations* (same paper; title was revised for the published version)
- **Authors:** Maitreya Patel, Song Wen, Dimitris N. Metaxas, Yezhou Yang
- **arXiv ID:** 2412.00100 (correction: my tavily note had a wrong URL; the correct preprint ID is 2412.00100)
- **Code:** `github.com/FlowChef/flowchef`; project page at `flowchef.github.io`
- **Base models:** Flux.1-dev, InstaFlow, SD3

### Verified mechanism

FlowChef is a **unified framework** for three distinct tasks: classifier guidance, linear inverse problems, and image editing. The core insight is that rectified flows have **straight trajectories with smooth Jacobians**, and this property lets you derive a mathematical relationship that **skips backprop through the ODE solver**. The technique is called "gradient skipping": the method jumps between nearby trajectories until convergence, driven by the straight-line structure of the vector field.

Concretely (from the verified abstract and project page):

- The method modifies the vector field at each timestep to steer the denoising trajectory
- Backpropagation through the full ODE solver is avoided; gradients of the guidance objective are approximated via rectified-flow geometry
- This is both **training-free** (no fine-tuning) and **gradient-free through the solver** (you still compute gradients of the classifier / inverse problem objective, but not through the ODE integration)

Tasks demonstrated:
- **Classifier guidance**: e.g., steering toward a target class
- **Linear inverse problems**: super-resolution, inpainting, deblurring
- **Image editing**: inversion-free editing

Baselines: FreeDoM, MPGD, LGD — all gradient-through-solver guidance methods. FlowChef claims improvements in "performance, memory, and time requirements"; specific numerical speedups were not visible in the abstract/project-page fetch and would need the full PDF.

### Face-specific results

**Not explicitly highlighted.** Neither the project page nor the abstract features portrait or face results prominently. This is a distinct departure from my prior framing.

### Correction to my prior claim

The flow-foundations note said:

> "this is a first-class editorial-channel candidate on top of the Flux backbone we already run. E1 readability (explicit semantic steering at inference), E2 ablation (steering strength is a knob), E4 scalability (no training cost)."

**This is partially correct but overstated in one direction.** FlowChef is:
- ✅ A training-free editorial-channel primitive on Flux
- ✅ A steering framework with explicit control objectives
- ❌ **Not** a face-specialized editor in the published form
- ❌ **Not** a drop-in "editorial axis" like FluxSpace

Adapting FlowChef to a vamp-interface editorial axis would require defining a classifier (or inverse-problem objective) that corresponds to the semantic axis we want. This is **more work** than FluxSpace, where the semantic axes are already defined as prompt-pair orthogonal projections.

**Revised positioning:** FluxSpace is the near-term editorial-channel candidate because its mechanism maps directly to "define an attribute prompt pair → get a linear edit direction." FlowChef is the **general steering primitive** that becomes relevant if we need controlled generation via an arbitrary classifier — e.g., ArcFace-guided identity preservation, or an uncanny-valley regression model — where FluxSpace's prompt-pair mechanism doesn't apply. They solve different problems on the same backbone. Both are worth knowing; only one is a drop-in face editor.

---

## Summary of corrections to [2026-04-14-flow-based-foundations.md](2026-04-14-flow-based-foundations.md)

1. **Retract** the "RectifID = anchor-bridge in rectified-flow space" claim in §2 and in "Surprising finds" §7. V1 is not implicitly reinventing RectifID. The word "anchored" in the RectifID title is about ODE trajectory numerical stability, not about reference-face-as-img2img-start.
2. **Fix** RectifID venue: **NeurIPS 2024**, not "HF paper page" (that was the distribution URL, not the venue).
3. **Fix** RectifID base model: **SD 1.5 piecewise-rectified-flow**, not Flux. It is therefore NOT a drop-in for our Flux backbone.
4. **Tighten** the FluxSpace "identity-leakage measurement" claim: paper reports **CLIP-I and DINO content preservation**, not ArcFace identity. Framework E3 requires our own measurement.
5. **Reframe** FlowChef: general-purpose controlled-generation framework on rectified flows, not a face-specialized editor. Worth reading as the primitive that becomes relevant when we need classifier-guided steering on Flux (e.g., ArcFace identity preservation, or an uncanny-valley-regressor-driven generator).
6. **Fix** FlowChef arXiv ID: **2412.00100**, not 2602.11105 (that was FastFlow, conflated in my first draft).

## What survives the verification unchanged

- Flux v3 is a rectified-flow model. (This was the biggest finding and did not depend on these three papers.)
- FluxSpace is a legitimate editorial-channel candidate on frozen Flux with an orthogonal-projection-in-attention-space mechanism that is amenable to framework E1–E4 scoring. Read it in full before scoring.
- Rectified flow's straight-trajectory property is load-bearing for all three of these papers, and is the structural reason for the framework's P4a rank-preservation argument.
- The flow-OOD-on-representations path for D1 measurement is unchanged and independent of these three papers.

## Next actions (suggestion, for user approval — no edits yet)

1. **Update [2026-04-14-flow-based-foundations.md](2026-04-14-flow-based-foundations.md)** with the six corrections above. Small, local edits. No framework change yet.
2. **Do not** edit the framework based on these three papers alone. The verified findings change priorities but do not change rubric cells. Framework edits should wait until we've either scored FluxSpace under E1–E4 or decided not to.
3. **Queue FluxSpace for framework §5 Experiment N+1**: "Score FluxSpace on Flux v3 under E1–E4, with ArcFace E3 measurement run by us because the paper reports CLIP-I/DINO not ArcFace."
