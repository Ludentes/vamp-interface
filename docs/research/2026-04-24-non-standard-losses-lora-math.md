---
status: live
topic: demographic-pc-pipeline
---

# Non-standard losses on diffusion LoRAs: PuLID's mechanism and why ArcFace-on-LoRA is hard

**Date:** 2026-04-24
**Author:** Claude (research subagent)
**Companion to:** [2026-04-24-identity-preserving-lora-survey.md](2026-04-24-identity-preserving-lora-survey.md) — read that first for the literature map. This doc goes deeper on two questions the survey didn't answer: *what exactly does PuLID do at the math level*, and *why isn't an ArcFace term a free addition to a vanilla LoRA training loop*.

## Why this doc exists

The survey established that ID-Booth and PuLID are the only well-attested
recipes that put ArcFace inside the training objective for diffusion
identity preservation, and that essentially nobody in the kohya/ai-toolkit
ecosystem ships such a loss. The natural follow-up: "OK but why not — it's
just another loss term, isn't it?" The honest answer is that it isn't,
and unpacking why is the same as understanding why diffusion + reward
models is its own subfield (DDPO, DRaFT, ReFL, ImageReward) rather than a
two-line patch to an MSE loop.

---

## Part 1 — What PuLID actually does

### The skeleton: two parallel branches

PuLID trains an IP-Adapter-style identity injector on top of a frozen
base diffusion model (SDXL in v1, FLUX in v0.9). Trainable weights are
*only*:

- the ID encoder (ArcFace-50 features + EVA-CLIP CLS, fused through an
  MLP in SDXL or a small Transformer in the FLUX port) producing
  ~55 global + ~55 local ID tokens;
- additional cross-attention `K_id, V_id` projections in parallel with
  each text-conditioned cross-attention layer (SDXL) or extra
  cross-attention blocks injected every few DiT blocks à la Flamingo
  (FLUX). The base UNet/DiT and the VAE are frozen.

This is not a LoRA in the kohya sense (no rank-decomposed delta on
existing weight matrices). It is structurally an *adapter*: new modules
running parallel to existing ones. Sources:
[arXiv 2404.16022 v2](https://arxiv.org/html/2404.16022v2),
[`docs/pulid_for_flux.md`](https://github.com/ToTheBeginning/PuLID/blob/main/docs/pulid_for_flux.md).

The training run carries two forward branches per step that share the
same prompt and same initial noise:

1. **Standard diffusion branch.** The frozen base model + the new ID
   modules denoise a noisy latent at a random `t ~ U(0,T)`, supervised
   by the usual ε-prediction MSE — call it `L_diff`. This is the
   "behave like a diffusion model" anchor.
2. **Lightning T2I branch.** The same modules are run inside a
   *4-step* SDXL-Lightning rollout from `x_T` all the way to a clean
   `x_0`. The Lightning branch generates at 768×768 to keep VRAM
   in budget. This is where the identity loss and the alignment loss
   are computed.

### Why a 4-step rollout instead of one-step x̂0

The mechanical reason is gradient quality. From `x_t` and the
ε-prediction `ε̂_θ(x_t,t)` you can always write the one-step estimate

```
x̂0(x_t, t) = (x_t − √(1−ᾱ_t) · ε̂_θ(x_t,t)) / √(ᾱ_t)
```

but at large `t`, `ᾱ_t` is small and the inverse scaling blows the
ε noise into a high-variance "best guess" of the clean image. In
ε-MSE training the noisy prediction is fine because the *target* is
also noise; the per-pixel error has variance `O(1)`. But the moment
you push `x̂0` through a perceptual function `φ` (ArcFace,
LPIPS, CLIP, a reward model), at high `t` the function sees garbage
and returns a gradient that is uncorrelated with what the LoRA
should have learned. Empirically this is well-known across the
reward-finetuning literature; ID-Booth side-steps it with a timestep
weighting `λ_t = 1 − (t/T)²`
([arXiv 2504.07392](https://arxiv.org/html/2504.07392v1)) and ReFL
side-steps it by sampling t only near the output
([ImageReFL background, arXiv 2505.22569](https://arxiv.org/html/2505.22569)).

PuLID picks a different option. Run a *distilled* sampler
(SDXL-Lightning) for 4 steps from pure noise to a near-clean image and
take the ArcFace loss on that. The Lightning teacher's distillation
already makes its 4-step output close to a real sample, so `φ(x_0)`
returns a meaningful gradient. The trade-off: backprop through 4
sequential UNet calls. PuLID handles this by truncating the gradient
path to "the last few timesteps" of the Lightning rollout for `L_id`,
and by randomly sampling a single step for the alignment loss
backward pass. This is the same DRaFT-K trick from
[arXiv 2309.17400](https://arxiv.org/html/2309.17400v2): full rollout
is unstable (gradient explosion), one-step x̂0 is too noisy, K=1–4
steps near the end is the working point. Reported VRAM with the
Lightning trick: 41–63 GB at training, vs OOM on 80 GB A100s if you
try to backprop a 4-step rollout of the *non-distilled* base model
([arXiv 2404.16022 v2 §implementation](https://arxiv.org/html/2404.16022v2)).

### The two losses in formulas

**Accurate ID loss.** The Lightning branch produces
`x̂0_L = L-T2I(x_T, c_id, c_txt)` from initial noise `x_T`, identity
condition `c_id`, and text condition `c_txt`. The loss is

```
L_id = 1 − cos( φ(x̂0_L), φ(I_id) )
```

where `φ` is ArcFace-50 features (the recognition embedding, not the
detector) and `I_id` is the source identity image. This is "accurate"
in PuLID's framing because `x̂0_L` is a high-quality near-clean
sample, not a noisy x̂0 — a direct response to the gradient-quality
issue above.

**Contrastive alignment loss.** Same x_T, same prompt, *two* forwards
through the Lightning branch:

- `Q_tid` — query features at a given cross-attention layer with the
  ID condition injected;
- `Q_t` — query features at the same layer with ID injection
  *removed* (the standard diffusion branch's behaviour at the same
  state).

Then

```
L_align-sem    = ‖ softmax(K Q_tid^T / √d) Q_tid
                  − softmax(K Q_t^T / √d)  Q_t  ‖_2
L_align-layout = ‖ Q_tid − Q_t ‖_2
L_align        = λ_sem · L_align-sem + λ_layout · L_align-layout
```

with reported `λ_sem=0.6, λ_layout=0.1`. The full objective is

```
L = L_diff  +  L_align  +  λ_id · L_id        (λ_id = 1.0)
```

Both align terms are computed on cross-attention features inside the
UNet, not on pixels. The "contrastive" wording is loose — it is not an
InfoNCE-style with-negatives contrast; it is a *paired alignment*
between the ID-on and ID-off branches sharing an initial noise and a
prompt. The intuition: ArcFace cosine alone says nothing about whether
the model still responds to the prompt or whether the layout / lighting
/ pose still match what the base model would have produced. Aligning
the with-ID and without-ID query maps says "inject identity in a way
that leaves everything else alone." This is the part that prevents the
ID loss from collapsing the model into copy-pasting the source pixels
(see *Identity loss can collapse to memorization* below).

### Architecture (Flux specifically)

PuLID for FLUX, per [`pulid_for_flux.md`](https://github.com/ToTheBeginning/PuLID/blob/main/docs/pulid_for_flux.md):

- ID encoder switched from MLP (SDXL) to a Transformer for FLUX.
- Additional cross-attention blocks inserted "every few DiT blocks"
  (FLUX has 19 double-stream + 38 single-stream blocks per the
  reference architecture; the exact insertion frequency isn't given
  in docs and would have to come from the source).
- Inspired by Flamingo: ID tokens enter via these new cross-attention
  blocks, the base double/single-stream weights stay frozen.

The published `pulid_flux_v0.9.1.safetensors` is the trainable-only
delta, on the order of a few hundred MB, vs FLUX's ~24 GB base — so
the trained module is roughly an order of magnitude smaller than a
full FLUX LoRA at rank 32, but considerably larger than the ~50 MB
typical of a kohya FLUX face LoRA.

### Training data and scale

- 1.5 M high-quality human images, BLIP-2 auto-captioned.
- 8× NVIDIA A100. Batch size, LR, total steps not explicitly disclosed
  in the paper text I was able to read; the staged-training recipe is:
  stage 1 = `L_diff` only, stage 2 = `+L_id`, stage 3 = full objective
  with `+L_align`.
- Per-step VRAM: 41–63 GB with Lightning + truncated backprop; 50–80+
  GB without acceleration; 4-step backprop of the non-distilled base
  OOMs on 80 GB cards. Source: [arXiv 2404.16022 v2](https://arxiv.org/html/2404.16022v2).
- FLUX-specific training details (data scale, steps) aren't broken
  out separately; the paper's main exposition is on SDXL with a port
  appendix. **Single source for FLUX numbers — verify against the repo
  before relying on them.**

This is *not* a recipe a hobbyist runs on a single 24 GB card. Even
inference uses 12–16 GB, but training has the 8× A100 dependency and
the Lightning teacher requirement.

---

## Part 2 — Why a non-standard loss is hard to bolt onto a LoRA

The user's framing: "Why can't we just add it as any other loss?" The
answer has six interlocking pieces, and most of the engineering effort
in DDPO/DRaFT/ReFL/PuLID/ID-Booth is spent on these.

### The training-time observation problem

A diffusion LoRA is trained by sampling `(x_0, t, ε)` with
`t ~ U(0,T), ε ~ N(0,I)`, computing `x_t = √(ᾱ_t) x_0 + √(1−ᾱ_t) ε`,
and minimising `‖ε̂_θ(x_t, t) − ε‖²` (or equivalently a v-prediction
or rectified-flow MSE). At every step *the model only ever sees and
predicts noise.* It never produces an image during training. ArcFace,
LPIPS, CLIP, ImageReward — every interesting auxiliary loss — needs an
*image* as input. So you cannot simply add a term `L_aux(model
output)`. You first have to *materialise an image from the model's
prediction at this step*. That is the entire problem.

### One-step x̂0 vs multi-step rollout

The cheapest way to get an image is the one-step estimate above:

```
x̂0(x_t, t) = (x_t − √(1−ᾱ_t) ε̂_θ(x_t, t)) / √(ᾱ_t)
```

This requires no extra forward passes. Its gradient w.r.t. LoRA
parameters is exactly the gradient of `ε̂_θ` rescaled by
`√(1−ᾱ_t)/√(ᾱ_t)`. Two problems:

1. **At high `t` it's garbage.** `√(ᾱ_t) → 0`, so dividing by it
   amplifies any error in `ε̂_θ`. ArcFace on this returns a feature
   vector that is essentially noise at `t > T/2`. This is why
   ID-Booth weights its identity loss by `λ_t = 1 − (t/T)²`
   ([arXiv 2504.07392](https://arxiv.org/html/2504.07392v1)) and
   ImageReFL goes further, sampling `t` only "near the output"
   ([arXiv 2505.22569](https://arxiv.org/html/2505.22569)) — they
   are admitting the loss is uninformative at high noise.
2. **You biased the LoRA toward late-timestep edits.** If the
   identity loss only fires at small `t`, the LoRA learns to express
   identity in the *last* few denoising steps. That's where high-
   frequency facial details live, but it also means the LoRA is
   doing zero work on the global structure of the face — the
   composition, the head pose, the silhouette — because at low `t`
   those are already locked in by the base model's earlier steps.
   This is a real failure mode: the LoRA becomes a "skin texture"
   patch that does nothing for likeness on novel poses.

The multi-step rollout avoids the high-`t` garbage by actually running
the sampler from `x_T` to `x̂0`. PuLID's 4-step Lightning branch is
the canonical instance. The cost: every step is a full UNet/DiT
forward, and to compute a gradient through the rollout you need the
activations of every step in the chain.

### Backprop-through-sampling memory

A single Stable-Diffusion-1.5 UNet forward at 512×512 stores ~8–10 GB
of activations for the backward pass; an SDXL UNet at 1024×1024 is
~20 GB; a FLUX DiT at 1024×1024 is comparable. Naively:

- Backprop through 1 step of UNet ≈ 1× the memory.
- Backprop through 4 steps ≈ 4× the memory.
- Backprop through 50 steps ≈ 50× → out of any consumer GPU.

**Three mitigations** are used in the literature, and PuLID combines
two of them:

1. **Gradient checkpointing.** Recompute activations on the backward
   pass instead of storing them. Memory cost drops to ~√k for k steps,
   compute roughly doubles. Used by DRaFT
   ([arXiv 2309.17400](https://arxiv.org/html/2309.17400v2)) and
   AlignProp.
2. **Truncated backprop (DRaFT-K, ReFL).** Forward through all k
   steps, but only let gradients flow through the last K steps and
   detach the earlier ones. K=1 (ReFL,
   [arXiv 2505.22569](https://arxiv.org/html/2505.22569)) makes
   the rollout cheap; K=k (full backprop) gives exploding gradients
   per the DRaFT paper. Sweet spot in the literature is K=1–4. PuLID
   describes its accurate ID loss as backpropping "only for the last
   few timesteps."
3. **Distilled few-step sampler.** Replace the 50-step base model
   with a 4-step distilled student (SDXL-Lightning, LCM, Hyper-SD).
   k itself is small to begin with. PuLID does this *and* the
   truncation; the FLUX port can use FLUX-Schnell or 4-step Hyper-SD
   variants.

Concrete numbers: PuLID reports 41–63 GB training VRAM with
Lightning + truncation, OOM on 80 GB without one of these tricks
([arXiv 2404.16022 v2](https://arxiv.org/html/2404.16022v2)). DRaFT-1
with checkpointing fits SD-1.5 LoRA training on a single A100-40GB
([arXiv 2309.17400](https://arxiv.org/html/2309.17400v2)). ID-Booth
runs on 4× A100-40GB or a single RTX 4090 — but using the
*one-step* x̂0 estimate (no rollout) precisely because rollout
backprop wouldn't fit a 4090 ([arXiv 2504.07392](https://arxiv.org/html/2504.07392v1)).

The pattern: the cheaper your auxiliary-loss path, the noisier the
gradient. The cleaner your gradient, the bigger your training rig.

### Loss-scale balancing

ε-MSE has roughly unit-variance per-pixel noise → loss values of
~0.5–1.0 across training. ArcFace cosine distance lives in [0, 2]
with most realistic values in [0.1, 0.5]. LPIPS is in [0, 1]. CLIP
similarity ranges similarly. The numerical scales are not catastrophic
but the *gradient* magnitudes can differ by orders of magnitude,
because:

- the diffusion loss gradient is `d L_diff / d θ ∝ (ε̂ − ε)` —
  zero-mean and small once trained;
- the auxiliary loss gradient comes through a fixed pretrained
  network's Jacobian, which has its own scaling and is non-zero-mean.

Reported behaviour:

- **Fixed weight (most common).** PuLID uses `λ_id=1.0, λ_sem=0.6,
  λ_layout=0.1` and a staged schedule (turn on losses one at a time
  to stabilise). ID-Booth uses fixed weights plus the per-timestep
  `λ_t = 1 − (t/T)²` cooldown.
- **Learned weights.** Multi-task uncertainty weighting (Kendall et
  al.) is occasionally cited but I did not find it adopted in any
  of the surveyed identity-preservation papers.
- **Gradient surgery (PCGrad, GradNorm).** Not used by any of
  PuLID, ID-Booth, FaceChain-FACT, DRaFT, ReFL. The community has
  apparently found fixed weights with timestep masking and staged
  schedules sufficient. **Single-source observation:** I haven't
  seen a paper test PCGrad on a diffusion+reward objective; absence
  of evidence isn't evidence of absence.

Reported failure mode: if you crank `λ_id` too high, the training
loss for `L_diff` stops decreasing and the LoRA's samples lose prompt
adherence. PuLID's staged schedule (diff → diff+id → diff+id+align)
is essentially a manual curriculum to prevent this.

### Identity loss can collapse to memorisation

If `L_id` is the only thing pushing on identity and you weight it
strongly, gradient descent has an easy minimum: make the model
ignore the prompt and the noise and emit pixels close to the source
identity image. ArcFace cosine ≈ 1, `L_id` ≈ 0, prompt fidelity
destroyed. This is the same collapse mode that DreamBooth's
prior-preservation loss is designed to prevent in the data-driven
setting.

Two structural defences in the literature:

1. **PuLID's contrastive alignment.** The `L_align` term *requires*
   that the model's cross-attention queries with-ID and without-ID
   match. If the with-ID branch collapses to "render the source
   image", `Q_tid` deviates massively from `Q_t` (which is still
   following the prompt), and `L_align` punishes it. This is a
   regulariser disguised as an alignment objective; it is the
   actual mechanism by which PuLID stays composable with the base
   model's other behaviours, including LoRA stacking.
2. **ID-Booth's triplet anchor.** Anchor = generated image, positive
   = training images of the target identity, negative = images
   produced by the *frozen base model* with the same prompt. The
   triplet loss `max(0, d(a,p) − d(a,n) + m)` prevents the LoRA
   from drifting all of identity space toward the target — the
   negative anchor is "what the base model would have done", and
   moving the trained model away from it on every prompt is exactly
   collapse.

These two are fundamentally the same idea — penalise deviation from
the frozen base model on everything except identity — implemented at
different layers (cross-attention features vs ArcFace embedding).

### VAE in the loop

For latent diffusion (SD, SDXL, FLUX), `x̂0` is a *latent*. ArcFace
needs RGB pixels. So the path is

```
x_t ── ε̂_θ ──> x̂0 (latent) ── VAE.decode ──> x̂0 (RGB) ── ArcFace ──> features
```

Three implementation choices:

- **Backprop through the VAE.** Adds memory (the VAE decoder stores
  its own activations) and compute. The VAE is frozen in PuLID and
  ID-Booth, but gradients still need to flow through it to reach
  the diffusion model. PuLID's Lightning branch backprops through
  the VAE only at the truncated final-K steps.
- **Use a latent-space ArcFace surrogate.** A learned MLP that
  predicts ArcFace from latents — proposed by
  [arXiv 2411.15247](https://arxiv.org/html/2411.15247v1) for
  reward fine-tuning. Avoids the VAE entirely. Not standard in
  identity preservation yet.
- **Detach + re-encode.** Generate the image without gradients,
  re-encode for an MSE on latent or run the loss outside the
  graph. Used for things like CLIP-direction losses where you only
  need the gradient to point in *some* useful direction. Loses the
  fine-grained gradient signal but is memory-cheap.

PuLID and ID-Booth take option 1 — backprop through a frozen VAE
decoder. The decoder is part of the gradient path, not detached. The
extra memory is part of the 41–63 GB number.

### Why the 2023 self-LoRA wave skipped this

It's worth being concrete about why kohya-ss/sd-scripts and
EveryDream and the CivitAI face-LoRA culture didn't ship an ArcFace
term, even though every face LoRA trainer would benefit:

1. **The reference open-source diffusion+ArcFace+VAE+timestep-mask
   stack didn't exist in early 2023.** Releasing one required
   implementing each of the pieces above and validating it. PuLID's
   April 2024 release is the first proper public reference;
   ID-Booth's April 2025 release is the cleaner LoRA-shaped
   companion. The stack is now public; before it was, anyone who
   tried got stuck on the gradient-quality and memory issues
   independently.
2. **Compute budget.** A 24 GB consumer card running kohya at
   batch=1 has ~5 GB of room above forward+backward of an SDXL
   UNet at rank 16. Not enough for a VAE-decode + ArcFace +
   gradient through the rollout. So even when the recipe was
   knowable, it wasn't runnable on the gear of the people writing
   the trainers.
3. **The community accepted that "more data + better captions +
   prior preservation" was the answer.** And empirically, for
   single-identity face LoRAs, it kind of is — likeness from 30
   captioned selfies + prior preservation is good enough that
   nobody felt acute pain. The pain became visible only when LoRAs
   were composed (the ai-toolkit "facial contamination" issue,
   [#166](https://github.com/ostris/ai-toolkit/issues/166)) or
   when scaled to attribute editing with sliders, which is exactly
   our regime.
4. **Reward fine-tuning was its own subfield.** DDPO, DRaFT, ReFL
   showed up in mid-2023 and matured through 2024
   ([BAIR DDPO blog](https://bair.berkeley.edu/blog/2023/07/14/ddpo/),
   [DRaFT arXiv 2309.17400](https://arxiv.org/html/2309.17400v2),
   ImageReward) but they targeted aesthetic / preference rewards,
   not identity. The identity-preservation community and the
   reward-finetuning community have only really merged in PuLID
   and ID-Booth.

---

## Adjacent attempts: the same problems, the same patterns

Across non-MSE losses on diffusion the machinery converges:

- **CLIP-direction loss (StyleGAN-NADA → diffusion ports).** CLIP
  embedding distance between generated image and a target text. Same
  one-step-x̂0-or-rollout problem. StyleGAN-NADA gets away with it
  because StyleGAN is single-step (no timestep ladder) — directly
  transferring the recipe to diffusion failed until people added
  timestep gating. Used in some inversion-based editing methods
  but not as a routine LoRA training loss.
- **LPIPS.** Pixel-space perceptual loss. Same pipeline as ArcFace
  (decode → run → backprop) and same memory cost. Used in some
  super-resolution diffusion fine-tunes.
- **DDPO ([arXiv 2305.13301](https://arxiv.org/abs/2305.13301)).**
  Treats sampling as an MDP and uses policy gradients on the reward.
  Avoids backprop through sampling entirely (REINFORCE-style), which
  removes the memory problem but introduces high variance. Can
  fine-tune with non-differentiable rewards, which DRaFT/ReFL cannot.
- **DRaFT ([arXiv 2309.17400](https://arxiv.org/html/2309.17400v2)).**
  Backprop through the full sampling chain with gradient checkpointing
  and LoRA-only weights. Reports gradient explosion at full backprop;
  recommends DRaFT-K with K=1.
- **ReFL / ImageReward
  ([arXiv 2505.22569](https://arxiv.org/html/2505.22569)).** Restricts
  gradient flow to the final step and adds a diffusion-loss
  regulariser to prevent reward hacking. Closest community sibling
  to PuLID's `L_align` — same job (don't drift the base distribution
  while you optimise the auxiliary term), simpler implementation
  (just keep `L_diff` in the objective).
- **DPO-Diffusion / Diffusion-DPO.** Pairwise preference learning,
  reformulates the problem to avoid backprop-through-sampling — the
  preference loss is computed at noise-prediction level using the
  DPO trick. Sidesteps the memory issue entirely but requires paired
  preference data.

The recurring pattern: **(a) something to make `x̂0` informative
(rollout, distilled sampler, or timestep mask), (b) something to
keep the base distribution intact (alignment loss, diffusion-loss
regulariser, prior-preservation negatives), (c) something to keep
memory tractable (gradient checkpointing, truncated backprop, or
both)**. Any non-standard loss recipe that omits one of these three
fails in a predictable way: noisy gradient, distribution collapse,
or OOM.

---

## Implication for this project

Our concrete situation: 12 prompt-pair attribute slider LoRAs on
FLUX, identity preservation at α=1.0 measured at 67% (ArcFace ≥
0.75). The survey recommended PuLID-Flux + slider stacking at
inference. This deeper read tightens that recommendation:

- **Why PuLID's design fits.** The contrastive alignment loss is
  literally optimised to leave the base FLUX model's response to
  prompts (and therefore to LoRAs applied at the standard
  injection points) unchanged outside the identity dimension. That
  is the property we need for slider stacking to work without
  re-training.
- **Why an ID-Booth-style LoRA-with-ArcFace alternative is more
  expensive than it looks.** Porting ID-Booth to FLUX means: a
  one-step x̂0 path with timestep weighting, a VAE backprop,
  ArcFace in the gradient graph, plus a triplet-loss replay buffer
  of frozen-base renders. None of this is conceptually hard but
  none of it exists in ai-toolkit and porting it to FLUX with
  rectified-flow timestep semantics is a real engineering project
  (a few weeks). The single-anchor case may not justify it given
  PuLID exists.
- **Why we shouldn't try to add an ArcFace term to our
  Concept-Sliders distillation pipeline.** Concept Sliders is a
  small-scale LoRA training (a few hundred image pairs, single
  axis). Bolting an ArcFace identity term onto that training loop
  reproduces every problem in this doc with none of the data scale
  PuLID had to absorb the regularisation cost. The right separation
  is: train sliders without identity loss (current path), inject
  identity at inference via a frozen PuLID-Flux adapter that was
  itself trained on 1.5 M images with the right machinery.

The cleanest mental model after this writeup: **PuLID is a finished
identity-preservation system with the gradient machinery already
solved; our job is to verify it composes with our sliders, not to
rebuild it from MSE+ArcFace inside a kohya trainer.**

---

## Sources

- [PuLID — arXiv 2404.16022](https://arxiv.org/abs/2404.16022) (HTML v2: [arxiv.org/html/2404.16022v2](https://arxiv.org/html/2404.16022v2))
- [PuLID GitHub — ToTheBeginning/PuLID](https://github.com/ToTheBeginning/PuLID)
- [`docs/pulid_for_flux.md`](https://github.com/ToTheBeginning/PuLID/blob/main/docs/pulid_for_flux.md)
- [NeurIPS 2024 PuLID poster](https://neurips.cc/virtual/2024/poster/96055)
- [ID-Booth — arXiv 2504.07392](https://arxiv.org/html/2504.07392v1)
- [DDPO — arXiv 2305.13301](https://arxiv.org/abs/2305.13301), [BAIR blog](https://bair.berkeley.edu/blog/2023/07/14/ddpo/)
- [DRaFT — arXiv 2309.17400](https://arxiv.org/html/2309.17400v2)
- [ImageReFL — arXiv 2505.22569](https://arxiv.org/html/2505.22569)
- [Latent-space surrogate reward — arXiv 2411.15247](https://arxiv.org/html/2411.15247v1)
- [ai-toolkit issue #166 (facial contamination)](https://github.com/ostris/ai-toolkit/issues/166)

## Single-source / unverified claims (be cautious)

- PuLID-Flux's exact insertion frequency of cross-attention blocks
  ("every few DiT blocks") and exact trainable parameter count —
  documented qualitatively only; would need a code read of `pulid/`
  to confirm.
- FLUX-specific PuLID training data scale and step count — paper
  exposition is on SDXL, FLUX port specifics are abbreviated.
- The PCGrad / GradNorm claim in *Loss-scale balancing* is
  absence-of-evidence based on the surveyed papers; I did not search
  exhaustively for diffusion-reward gradient-surgery work.
- 67% ArcFace ≥ 0.75 retention figure is from our own pipeline
  (memory `project_demographic_pc_stage4_5.md`), not the literature.
