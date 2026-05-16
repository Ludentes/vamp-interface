---
status: live
topic: arkit-controlnet
---

# ARKit-conditioned ControlNet on FLUX (InfiniteYou / InfuseNet)

**Date:** 2026-05-16
**Purpose:** Record the design analysis for an ARKit-blendshape ControlNet that
poses a generated face while ArcFace holds identity steady — i.e. InfiniteYou's
InfuseNet with its spatial-control channel re-purposed for expression. Covers
the architecture fit, the loss design (the slider-thread latent-loss lesson and
how it does / does not transfer), the ready-made training corpus, and prior art.

## The idea in one line

Keep InfiniteYou's identity path untouched (ArcFace → residual injection) and
feed its existing spatial-control channel a render of an ARKit-driven FLAME
mesh. The result: a generative FLUX face that is *posed* by 52 blendshape
coefficients and *pinned* by an ArcFace embedding — fine expression control
over a steady identity.

## How InfuseNet works (verified against the PDF)

`docs/papers/infiniteyou-2503.16418.pdf`, ICCV 2025, ByteDance.

- **InfuseNet is "a generalization of ControlNet"** (the paper's own words). It
  is a trainable copy of `N` DiT blocks; with `M` base blocks and `M = N·i`,
  `i=4`, InfuseNet block `j` predicts the output residuals of base blocks
  `(j−1)·i+1 … j·i`. Outputs are **added back as residuals**, not substituted
  into attention.
- **Identity path:** a frozen ArcFace-class face encoder → an IP-Adapter-style
  projection network → **8 identity tokens**, fed into InfuseNet through
  attention. ID Loss in the paper is defined as `1 − cos(ID_gen, ID_ref)`.
- **The spatial-control slot already exists.** Verbatim: *"An optional control
  image, such as a five-facial-keypoint image, can be input into InfuseNet to
  control the generation position of the subject. If no control is needed, a
  pure black image can be used instead."* Today this channel carries crude pose
  keypoints — it is the natural slot for a FLAME/ARKit render.
- **Residual injection, not attention, is deliberate.** Fig 2(b) + Table 2
  ("w/ IPA") show that injecting a face signal through attention layers
  improves identity (ID Loss 0.180) but craters text alignment
  (CLIPScore 0.318 → 0.241) and quality. Residual injection keeps identity
  *distinct* from the text path.
- **Training:** base DiT frozen; only InfuseNet + projection train, with the
  standard conditional flow-matching loss `L_CFM = ‖v_Θ(z,t) − u_t(z|ε)‖²`.
  Two stages — pretrain on 43M real single-person-single-sample (SPSS) images,
  then SFT on 2M synthetic **single-person-multiple-sample (SPMS)** pairs.
- **The SPMS ablation is load-bearing.** Table 2: replacing SPMS with SPSS in
  stage 2 collapses ID Loss 0.209 → 0.368. The paper's diagnosis: SPSS makes
  the model "learn a reconstruction of synthetic data rather than transforming
  reference real data" — copy-paste instead of decoupling.
- **Plug-and-play with off-the-shelf ControlNets** (Fig 6 b/c): InfU runs
  alongside a stock Depth or Pose ControlNet with no retraining of InfU.

## The proposal

Two conditioning modalities into one InfuseNet-style framework — which is
exactly what InfuseNet was built for ("ingest more modalities via residual
connections"):

- **Identity** — ArcFace embedding → 8 tokens → residual injection.
  *Unchanged from InfiniteYou.* The user's "combine ArcFace + ARKit" is really
  "keep ArcFace, add ARKit" — ArcFace is already the identity path.
- **Expression + pose** — drive a FLAME / ARKit-rigged head with the 52
  blendshape coefficients at the target pose, render it (surface normals +
  depth, or normals + landmark overlay), feed that as InfuseNet's control image.

**Feed expression as a render, not as tokens.** 52 scalars projected to tokens
have no spatial grounding — a token cannot know *where* the mouth is. That is
the slider thread's v1h failure mode (critic satisfied, render unchanged).
A rendered FLAME mesh is spatially aligned, so the ControlNet sees where the
expression happens. This is structurally the right channel and is independently
confirmed by InfiniteYou's own "w/ IPA" ablation.

### Two paths, cheapest first

- **Path 1 — zero-training spike.** Drive a stock FLUX depth-or-normal
  ControlNet with a FLAME render, stacked on *frozen* InfU (the paper
  demonstrates exactly this stacking, Fig 6 b/c). Tests "does a FLAME render
  steer FLUX expression while ArcFace holds identity" for the cost of a ComfyUI
  graph. No GPU-months at risk.
- **Path 2 — fine-tune InfuseNet's control encoder.** Replace the 5-keypoint
  control image with the FLAME render and retrain InfuseNet (base DiT frozen,
  as InfU does). Tighter decoupling; 64–128 GPU scale for the full recipe, less
  for a LoRA-grade version.

## Loss design

This is the section that connects to the slider thread. The question was:
*the slider thread's latent-space blendshape loss caused VAE-decode artifacts,
fixable by gating the loss to early denoising steps — does that transfer to a
ControlNet?*

### The slider artifact is not a property of "latent-space loss"

Runs v1h / v1j (`docs/research/2026-04-30-bs-loss-classifier-fooling.md`): the
distilled blendshape critic, read off the FLUX latent `z_t`, reported the
target ARKit value correctly, but after VAE decode produced **no visible
expression change or identity collapse** (aged skin, lighting drift). The
diagnosed cause:

> *"In the latent space there are off-manifold directions where the critic
> confidently predicts whatever blendshape we want, while VAE decoding produces
> a face that has the right blendshape geometry plus arbitrary correlated drift."*

The artifact was caused by supervising with a **distilled critic and no
ground-truth target** — a critic alone has an off-manifold shortcut (fool the
readout without moving the geometry) and the optimizer always takes it. It was
*not* caused by supervising in latent space per se.

### The base CFM loss is grounded — no artifact, no gating needed

InfuseNet trains with conditional flow matching toward a *real target image*.
That is a latent-space loss, and it has **no foolable shortcut**: there is a
known pixel-grounded answer. So:

**If you train an ARKit-InfuseNet the standard way — CFM reconstruction against
real `(photo, tracked-ARKit-render)` pairs — you do not inherit the artifact,
and the early-gate fix is not needed for the base loss.** The problem does not
arise.

### The gating idea transfers — to an *auxiliary* loss

The fix transfers cleanly the moment you bolt an auxiliary blendshape-fidelity
or ArcFace-identity loss onto the CFM base — because that reintroduces the
foolable latent-space term from the slider thread.

You may want to. CFM reconstruction supervises "match this exact photo"; it
only *implicitly* supervises control-following. On an expression-skewed corpus
(the `project_blendshape_bridge_state` corpus-skew problem) the model can
half-ignore the control render and still score low CFM loss. An auxiliary
critic loss — "the decoded image must read the *intended* ARKit coefficients" —
supervises control-following directly. For that auxiliary loss:

- **`bs_loss_t_max ≈ 0.5` transfers unchanged.** That number is a property of
  the `x0 ≈ z_t − t·v` reconstruction quality — garbage at high `t`, critic
  goes OOD — which is model-agnostic (LoRA or ControlNet, identical). Reuse it.
- **The early/late split is *more* principled in a ControlNet.** Expression is
  structural geometry, laid down in early/high-noise steps; late steps are
  texture/detail. Gating an expression-fidelity loss off late is correct on
  both counts — late, the critic is unreliable (the per-channel R²(t) map) and
  it can only corrupt skin/lighting (the v1j collapse band).
- **The per-channel R²(t) temporal-availability map is reusable** as the
  gating window — small channels early, large channels late, all under `t≤0.5`.
  Use the PGD-robust `bs_v4_pgd` critic rather than `v3_t` for the auxiliary
  term once it clears its validation gates.

### Three safeguards stack, where the slider had one

1. **CFM is grounded** — a real target image, no shortcut. (slider had only a critic)
2. **The base DiT is frozen** — InfuseNet only adds residuals; the frozen
   generative manifold is a built-in prior that resists off-manifold drift.
   The slider LoRA modified the base's own attention and could perturb anything.
3. **Timestep-gating** the auxiliary loss — the slider thread's fix.

Even CFM + gated auxiliary critic is structurally *less* foolable than v1h.

### Recommended loss sequencing

InfiniteYou deliberately **rejected** an identity loss — PuLID-FLUX trains with
alignment + identity losses; InfiniteYou does not, fixing identity with SPMS
data + residual architecture instead. Mirror that:

1. Train ARKit-InfuseNet on **CFM alone**, with SPMS-shaped data (one identity
   × many ARKit expressions). Check whether expression-following is already
   adequate. Likely it is — and the foolable loss is avoided entirely.
2. **Only if** control-following is weak (corpus skew) → add the auxiliary
   blendshape critic loss, gated `t_max ≈ 0.5` per channel via the R²(t) map.
   This is where the slider thread's machinery (`_bs_loss_term`, the gating
   code in `ConceptSliderTrainer.py`, `bs_v4_pgd`) ports over directly.

Optional inference knob: feed the FLAME render as InfuseNet's control image
only in the structural band and a black image late, so late steps refine
texture without the control fighting. Tunable, not required — the render is
mild conditioning, not a foolable loss.

## The corpus is already built — `reverse_index.parquet`

`output/reverse_index/reverse_index.parquet` — 79,116 rows × 110 columns.
Per image it carries: the image SHA, **52-d ARKit blendshapes** (`bs_*`),
**512-d ArcFace embedding** (`arcface_fp32`), FairFace / MiVOLO / InsightFace
demographics, 20 NMF atoms, 12 SigLIP probe margins, a 1152-d SigLIP-2 image
embedding. Sources: 70,000 real FFHQ portraits + 7,772 Flux renders + 1,344
Solver-A grid renders.

This is `(image, ARKit-coefficients, identity-embedding)` triples already
assembled — the exact training corpus an ARKit-InfuseNet needs, with the 70K
real FFHQ images standing in as the InfiniteYou Stage-1 SPSS set. The only
missing step is rendering each blendshape vector through FLAME to produce the
control image; that is a scriptable batch job.

Two gaps to note:
- FFHQ is one image per identity (SPSS, not SPMS). True same-identity /
  different-expression pairs are rare. But the index carries *both* ArcFace
  embeddings and blendshapes, so it can mine ArcFace-near / blendshape-far
  pairs as SPMS approximations — or, better, select decorrelated expression
  exemplars to drive synthetic SPMS generation (the InfiniteYou Stage-2 recipe).
- The `bs_*` coefficients are MediaPipe FaceLandmarker readings, not a
  high-fidelity tracker. Apply the standing verification gate
  (`cos(tracked, teacher) > 0.95`) before trusting them as control ground truth.

## Prior art

- **FG-Portrait** (CVPR 2026, `docs/papers/fg-portrait-2603.23381.pdf`) — the
  closest published method to this thread; drives a portrait animator from
  FLAME / parameter vectors via learning-free 3D-flow correspondence. Read
  before building; position the contribution against it. No code at fetch time.
- **DiffusionRig** (CVPR 2023) — conditions diffusion on DECA/FLAME renderings
  (normals, albedo, lighting) for identity-preserving portrait editing.
  Per-identity finetuned, plain SD. The amortized-identity delta over it is the
  contribution here.
- **Face-Adapter** (ECCV 2024) — identity + 3DMM control for reenactment /
  swapping.

The contribution, framed crisply: *InfiniteYou's amortized ArcFace identity
⊕ DiffusionRig's 3DMM-render conditioning, on FLUX.*

## Caveats

- Per-frame FLUX is not real-time and not temporally consistent — InfiniteYou
  is a single-image model. If the goal is live animation this is an
  offline-quality renderer / distillation teacher, not a replacement for the
  LAM / LivePortrait streaming path. Natural payoff: render an
  `(ArcFace-ID, ARKit-expression) → photoreal` corpus and distill into the fast
  student, slotting into the existing `arc_distill` / PersonaLive-student
  machinery.
- A static ARKit-ControlNet is a smaller project than FLUX video animation
  (the rendering-stack doc estimates 3–6 months for the latter) but still
  multi-week — an open research project, not a recipe.
- This *helps* the vamp-interface continuity hypothesis: 52 smooth coefficients
  + a smooth FLAME render is a far more continuous control surface than the
  denoising-strength dial.

## Next step

Path 1 — the zero-training ComfyUI spike: stock normal/depth ControlNet driven
by a FLAME render, on frozen InfU, evaluated against `reverse_index` baseline
distributions (ArcFace-cos identity drift, re-measured blendshape fidelity).
See the topic index `_topics/arkit-controlnet.md` for current status.
