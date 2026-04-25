---
status: live
topic: demographic-pc-pipeline
---

# Research: Identity-preserving LoRA / personalization for Flux when sweeping an attribute axis

**Date:** 2026-04-24
**Author:** Claude (research subagent)
**Sources:** ~12 primary URLs — DreamBooth (CVPR 2023), Concept Sliders
(ECCV 2024), Celeb-Basis (NeurIPS 2023), FastComposer, FaceChain +
FaceChain-FACT, PhotoMaker, IP-Adapter-FaceID, InstantID, PuLID
(NeurIPS 2024), Arc2Face (ECCV 2024 oral), ID-Booth (FG 2025),
ai-toolkit (Ostris).

---

## Executive summary

For our 12-atom-atlas use case — one fixed identity rendered at every
value of 12 learned attribute sliders — the literature splits cleanly
into two families:

1. **Per-identity fine-tuning** (DreamBooth / LoRA-DreamBooth /
   Celeb-Basis / ID-Booth). Trains weights for *this* face. The recent
   identity-preservation papers in this branch (ID-Booth, FaceChain)
   add an ArcFace-derived identity term to the diffusion loss as a
   regulariser.
2. **Encoder-conditioned zero-shot** (IP-Adapter-FaceID, PhotoMaker,
   InstantID, PuLID, Arc2Face). Trains *once* on a large face corpus,
   then inject any new identity via an ArcFace embedding fed through a
   cross-attention adapter. No per-identity training; ArcFace is in the
   *training* objective for some (PuLID's "accurate ID loss"; Arc2Face
   conditions on the embedding directly).

The "self-LoRA" / CivitAI wave from 2023 was almost entirely category
(1) without an explicit identity loss — pure DreamBooth-LoRA on
~10–30 selfies, relying on data and the prior-preservation loss. The
identity-loss-augmented variants (ID-Booth, FaceChain-FACT, PuLID) are
mostly 2024–2025 work.

For sweeping an attribute axis while holding identity, the empirically
better-attested path right now is **encoder-conditioned + slider LoRA
stacked at inference**, specifically PuLID-Flux + Concept-Slider LoRA,
because it decouples the "who" from the "what" at the architecture
level rather than in a single set of trained weights. See ranked
recommendation at the end.

---

## Identity-loss-augmented LoRA / DreamBooth literature

**DreamBooth** (Ruiz et al., CVPR 2023, arXiv 2208.12242) is the
canonical few-shot personalisation method. Original DreamBooth uses
**no identity-recognition loss** — only the standard ε-prediction
denoising loss plus a *class-specific prior preservation* loss
(generate "a [class]" with the original model and keep matching them).
ArcFace shows up only in evaluation, not the objective.
([arXiv 2208.12242](https://arxiv.org/abs/2208.12242))

**Custom Diffusion** (Kumari et al., CVPR 2023): tunes only K/V
projections of cross-attention. No identity loss; data-driven.

**Celeb-Basis** (Yuan et al., NeurIPS 2023, arXiv 2306.00926): builds a
PCA basis over ~691 celebrity name-token embeddings and learns a 1024-
parameter projection of a target face into this basis. No ArcFace
training term — the basis acts as a prior. Cheap (1024 params, ~3
min/identity) but limited fidelity vs newer methods.
([arXiv 2306.00926](https://arxiv.org/abs/2306.00926),
[celeb-basis.github.io](https://celeb-basis.github.io/))

**FastComposer** (Xiao et al., 2023): pretrained on a large human
dataset (150k steps, batch 128, 8×A6000), uses CLIP image features for
identity and a localisation/cross-attention regulariser to keep
multiple subjects from blending. Identity preservation is data-driven
through pretraining, not an ArcFace loss term.

**FaceChain** (ModelScope, arXiv 2308.14256): productised pipeline
with a separate face-LoRA and style-LoRA stacked. Vanilla FaceChain
uses standard DreamBooth-LoRA training; ArcFace enters only as a
quality filter on the input crops, not the training loss.
**FaceChain-FACT** (arXiv 2410.12312, Oct 2024) is the upgrade: a
TransFace-based face adapter with *decoupled* training so the LoRA
captures ID rather than entire face appearance. Closer to encoder
methods than to LoRA-DreamBooth.
([arXiv 2308.14256](https://arxiv.org/abs/2308.14256),
[arXiv 2410.12312](https://arxiv.org/abs/2410.12312))

**ID-Booth** (Tomašević et al., FG 2025, arXiv 2504.07392): the
clearest current example of "DreamBooth + ArcFace identity loss". Uses
a **triplet loss** on ArcFace ResNet-100 features extracted from MTCNN
crops: anchor = generated, positive = training images, negative =
prior images from the unfine-tuned base model. Beats vanilla
DreamBooth/LoRA on intra-identity consistency *and* on
inter-identity separability. It is LoRA-shaped (fine-tunes a latent
diffusion model with low-rank adapters).
([arXiv 2504.07392](https://arxiv.org/abs/2504.07392),
[github.com/dariant/ID-Booth](https://github.com/dariant/id-booth))

**Imagic, StyleGAN-NADA**: cited in the user's prompt for
completeness. StyleGAN-NADA's CLIP-direction loss is a different
animal (text-driven domain adaptation on StyleGAN, not ArcFace). Imagic
is single-image text-driven editing of diffusion, not personalisation.
Neither is load-bearing here.

**Net of category (1):** DreamBooth, Custom Diffusion, Celeb-Basis,
LoRA-DreamBooth → **no ArcFace term in training**. ID-Booth and
(partially) FaceChain-FACT → **explicit ArcFace identity loss in
training**. ID-Booth is the cleanest reference implementation.

---

## The training-objective math

Across the papers that *do* add an ArcFace term, the recurring pattern is:

```
L_total = L_diffusion(ε_pred, ε)                           # standard MSE on noise
        + λ_id · L_id(arcface(decode(x0_pred)), arcface(x_target))
        + λ_prior · L_prior                                 # optional
```

The decisive design choice is **what x0 you score with ArcFace**.
Options seen in the wild:

- **(a) One-step x0 estimate from the current noisy latent.** Cheapest.
  Used implicitly when a paper says "decoded x0 prediction during
  training". Quality is poor at high noise levels, so most
  implementations only apply identity loss when the timestep t is
  small (low noise). This is the dominant choice in production
  pipelines because it keeps gradient cost ~1× the base step.
- **(b) Few-step DDIM rollout from current t to 0.** Better x0 quality
  → cleaner ArcFace gradient. Memory-expensive (need to backprop
  through k UNet calls). PuLID's "Lightning T2I branch" is a variant
  of this: it runs an SDXL-Lightning few-step branch in parallel to the
  standard branch and applies the ID loss on the Lightning branch's
  near-clean output. This is the most cited "make the identity loss
  actually informative" trick.
  ([PuLID arXiv 2404.16022](https://arxiv.org/abs/2404.16022))
- **(c) Auxiliary identity-preservation pass with frozen base.** The
  prior-preservation pattern from DreamBooth, generalised: render with
  the frozen model and use it as a *negative* anchor (ID-Booth's
  triplet) to keep the LoRA from collapsing prior identities into the
  trained one.
- **(d) Pure regularisation on LoRA delta** (no ArcFace, just norm /
  orthogonality penalties). Used in Concept Sliders. Doesn't help
  identity per se; helps disentanglement on the attribute axis.

What works empirically (synthesising claims across PuLID, ID-Booth,
FaceChain-FACT):

- (a) alone is weak — high-noise gradients dominate and the ArcFace
  signal at low t is too late.
- (b) is what the SOTA papers do. PuLID reports its "accurate ID loss"
  on the Lightning branch as a key ingredient.
- (c) is cheap and complementary; ID-Booth shows it's worth the
  forward pass.
- Composing (b)+(c) is the ID-Booth + PuLID recipe.

**Single-source caveat:** the relative weight of (a) vs (b) gradients
is a claim I'm aggregating from PuLID's ablation and ID-Booth's
discussion. I have not independently verified.

---

## The "self-LoRA" / personalisation wave

Yes, ~2023 saw a flood of CivitAI-era "train a LoRA on yourself" tools
(kohya-ss/sd-scripts, dreambooth-lora notebooks, EveryDream). Looking
at what they actually did:

- **No explicit identity loss.** Almost universally just
  ε-prediction MSE + sometimes prior-preservation images. Identity
  came from data scale (10–50 selfies, captioning, regularisation
  images) and rank (4–32 LoRA rank).
- **Captioning tricks** (rare token like `ohwx person`,
  class-preservation prompts) carried more weight than loss
  engineering.
- **Mixing/stacking culture**: stacking face-LoRA at strength 0.7 with
  style-LoRA at strength 0.5 was the empirical answer to "preserve
  identity while changing style". This is the proto-version of the
  encoder-conditioned + slider LoRA stack in section
  *Recommendation* below.

For Flux specifically, the dominant trainers in 2025–2026 are:

- **ai-toolkit (ostris)**, the most popular Flux LoRA trainer.
  Standard recipe: rank 16–32 for likeness, ε-prediction loss, no
  identity loss. The Ostris GitHub has an open issue
  ([#166](https://github.com/ostris/ai-toolkit/issues/166)) about
  "facial contamination" — face-LoRAs leak the trained face into other
  characters in group photos — which is exactly the failure mode an
  ArcFace term would mitigate but the toolkit doesn't ship one.
- **kohya-flux, SimpleTuner, x-flux**: same story. Vanilla MSE.

Net: **no, the popular FLUX trainers do not use face-identity loss by
default.** Anyone doing it is rolling custom code or porting from
ID-Booth. ([ai-toolkit README](https://github.com/ostris/ai-toolkit))

---

## Encoder-conditioned alternatives

These avoid per-identity training. Compared head-to-head:

| Method | ID encoder | Trained on | Composability with attribute slider | Editability |
|---|---|---|---|---|
| IP-Adapter-FaceID | ArcFace | adapter only | high (no UNet edit) | medium — text degrades |
| IP-Adapter-FaceID-Plus | ArcFace + CLIP | adapter | high | medium — style degrades |
| PhotoMaker (v1/v2) | stacked CLIP | adapter | medium | text-edit hurt |
| InstantID | ArcFace + IdentityNet (ControlNet-style) | adapter+CN | high — UNet untouched, ControlNet-compatible | best of the SDXL bunch |
| PuLID (SDXL & Flux) | ArcFace + CLIP | adapter, contrastive + Lightning ID loss | high — explicitly designed for "minimum disruption to base" | best, especially text-edit |
| Arc2Face | ArcFace only (no text) | full SD UNet fine-tune | low — sacrifices text controllability | very high ID, bad for sliders |

Sources, in order:
[IP-Adapter-FaceID HF](https://huggingface.co/h94/IP-Adapter-FaceID),
[InstantID](https://instantid.github.io/),
[PuLID GitHub](https://github.com/ToTheBeginning/PuLID),
[Arc2Face GitHub](https://github.com/foivospar/Arc2Face),
[Patreon comparison post](https://www.patreon.com/posts/comparing-v1-v2-109318688).

**Stacking with attribute sliders.** The community evidence
(comparison post above + ComfyUI-PuLID-Flux + concept slider repos):

- **PuLID + Concept Slider on Flux** is the most-cited working stack.
  PuLID is explicitly designed to "minimize disruption to the original
  model's behaviour" via contrastive alignment between
  ID-injected and ID-not-injected branches sharing a prompt and
  initial latent — i.e. the rest of the model still responds to text
  and to other LoRAs. Concept-Sliders LoRAs apply at the standard LoRA
  injection points, so they compose linearly with PuLID's adapter.
  *Caveat:* I couldn't find a peer-reviewed quantification of "how
  much identity drifts when the slider is at α=1.0 with PuLID
  active". The qualitative evidence is from workflow posts on
  [openart.ai](https://openart.ai/) and
  [runcomfy.com](https://www.runcomfy.com/).
- **InstantID + Concept Slider** works in principle on SDXL.
  InstantID doesn't train the UNet, so it should compose with LoRAs.
  Single-source claim from the InstantID landing page —
  "compatible with existing pre-trained models and ControlNets".
- **Arc2Face is a poor fit** for sliders. It trades text
  controllability for ID similarity (paper explicitly notes this) and
  trains the whole UNet. Don't stack our text/image-pair sliders onto
  it.

**Identity-axis decoupling quality when stacked.** Best evidence I
have is PuLID's own ablation: its contrastive-alignment loss is
specifically constructed so the *non-ID* features (background,
lighting, composition, attribute response to prompts) match the
no-ID-injected reference. So when a slider perturbs an attribute,
PuLID's ID branch is contractive on identity and approximately
identity on everything else. This is the cleanest theoretical story
for our use case. Single-source — I'd want to verify before betting
the workflow on it.

---

## What would specifically help the 12-atom-atlas use case

We have:

- One anchor face we want to render at every value of 12 attribute
  sliders (think a 12-D grid).
- Concept-Sliders-style image-pair LoRAs trained per-axis.
- 67% ArcFace ≥ 0.75 retention at α=1.0 currently.
- ArcFace IR101 already extracted as a metric.

Three options to consider, ranked:

### (1) RECOMMENDED — PuLID-Flux as identity injector + our 12 slider LoRAs as attribute axes

- **Architecture**: at inference, freeze the anchor's ArcFace
  embedding, run PuLID-Flux to inject it, apply slider LoRA i at
  strength α_i for each axis i. No retraining required for new
  identities.
- **Why**: separates "who" (encoder, ArcFace-conditioned) from
  "what" (slider LoRAs trained without identity in the loop).
  PuLID's contrastive alignment specifically protects non-ID
  attributes from being clobbered by ID injection, so slider
  effects should remain visible. Lower training cost than
  retraining 12 axes with an identity loss.
- **Risk**: composition with multiple sliders simultaneously is
  unverified at scale. Start with a single-axis sweep and confirm
  ArcFace ≥ 0.75 holds, then sweep 2-D grids.
- **Disk/cost**: zero extra training. Inference cost is PuLID
  adapter (~one extra forward of an ArcFace + projector) per image.

### (2) FALLBACK — Train one identity LoRA (ID-Booth recipe) for the anchor, stack with the existing 12 slider LoRAs

- **Architecture**: ID-Booth-style fine-tune (LoRA + ArcFace triplet
  loss with anchor as positive, prior renders as negative) for the
  one anchor face. At inference, stack identity-LoRA + slider LoRA i
  at α_i.
- **Why**: works without an encoder dependency. ID-Booth has
  a public reference implementation
  ([github.com/dariant/ID-Booth](https://github.com/dariant/id-booth))
  porting cleanly to Flux is non-trivial but tractable. The
  triplet objective is the cleanest single recipe in the literature.
- **Risk**: LoRA stacking interference. With our 67% baseline, a
  rank-32 identity LoRA stacked onto a rank-16 slider LoRA at
  α=1.0 may still drift — but starting from a face that is
  *trained-in* to the model rather than encoder-injected gives
  a stronger prior to fall back on at high α.
- **Cost**: ~one A100-hour per identity (extrapolating from
  ID-Booth's reported numbers). Doesn't scale to many identities.

### (3) NOT RECOMMENDED for this use case — Single multi-axis LoRA conditioned on the 12-d atom vector with ArcFace regulariser

- **Architecture**: train one LoRA whose adapter is conditioned on
  the 12-d atom vector (e.g. via a learned MLP that produces
  per-layer scalings), with an ArcFace identity regulariser à la
  ID-Booth.
- **Why not**: this is a research project, not an off-the-shelf
  recipe. No paper does exactly this. SliderSpace and SliderEdit
  are adjacent (single-LoRA, multiple discovered axes) but the
  axes are *discovered*, not externally specified by our atom
  atlas. Building this bespoke would absorb weeks before we know
  if it's better than (1).
- **When to revisit**: if (1) shows that PuLID + stacked sliders
  hits a composition ceiling (sliders interfere multiplicatively
  beyond ~3 simultaneously active axes), (3) becomes the natural
  next step.

### Concrete next experiment

Cheapest decision-driver: take the existing v3 anchor and v3
slider LoRAs, plug in PuLID-Flux at the anchor's ArcFace embedding,
sweep one axis from α=0 to α=1.0, and re-measure the ArcFace ≥ 0.75
retention rate. If it goes from 67% to ≥85% with no quality loss
on the slider effect, option (1) is decided. If it doesn't move,
fall back to option (2) and budget an ID-Booth port.

---

## Open questions / single-source claims worth verifying

- PuLID's "the rest of the model still responds normally" claim is
  taken at the paper's word; the paper's ablations are mostly on
  text editability, not LoRA stacking. **Test before betting.**
- Identity-loss gradient signal at low t vs few-step rollout: I'm
  averaging across PuLID and ID-Booth's discussions. Worth reading
  the PuLID Lightning-branch ablation directly before re-implementing.
- ai-toolkit "facial contamination" issue is a single GitHub thread;
  the magnitude isn't quantified in any paper I found.
- No published quantification of "Concept-Sliders α=1.0 identity
  drift with vs without PuLID" exists, that I could find. We'd be
  generating that number ourselves.

---

## Sources

- [DreamBooth — arXiv 2208.12242](https://arxiv.org/abs/2208.12242)
- [Concept Sliders — sliders.baulab.info / arXiv 2311.12092](https://sliders.baulab.info/)
- [Celeb-Basis — arXiv 2306.00926](https://arxiv.org/abs/2306.00926)
- [FaceChain — arXiv 2308.14256](https://arxiv.org/abs/2308.14256)
- [FaceChain-FACT — arXiv 2410.12312](https://arxiv.org/abs/2410.12312)
- [IP-Adapter-FaceID — HF](https://huggingface.co/h94/IP-Adapter-FaceID)
- [PhotoMaker comparison post](https://www.patreon.com/posts/comparing-v1-v2-109318688)
- [InstantID](https://instantid.github.io/)
- [PuLID — arXiv 2404.16022](https://arxiv.org/abs/2404.16022)
- [PuLID GitHub](https://github.com/ToTheBeginning/PuLID)
- [ComfyUI-PuLID-Flux](https://github.com/balazik/ComfyUI-PuLID-Flux)
- [Arc2Face — ECCV 2024](https://github.com/foivospar/Arc2Face)
- [ID-Booth — arXiv 2504.07392](https://arxiv.org/abs/2504.07392)
- [ai-toolkit (Ostris)](https://github.com/ostris/ai-toolkit)
