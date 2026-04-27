---
status: live
topic: demographic-pc-pipeline
---

# Distilling perception models into latent space — identity and blendshape oracles for slider training

A deep design doc for two companion training losses that both operate on
representations we already produce, instead of decoding to RGB and
running pretrained ArcFace / MediaPipe at every training step:

1. **Latent ArcFace** — identity-preservation loss distilled from pixel
   ArcFace.
2. **Latent MediaPipe blendshape predictor** — direct-target supervision
   for slider axes, distilled from MediaPipe FaceLandmarker.

They share a distillation architecture and a role in the training loop,
so the plan covers them together. Either alone is useful; together they
enable a clean-room rebuild of the slider trainer where identity and
axis targets are *both* specified declaratively rather than implicitly
via image-pair data.

Written 2026-04-24, extended same day after the v1.1 training run
surfaced a sublinearity + identity-contamination failure mode (see
*Motivation* below) that a blendshape-target loss addresses
structurally.

## Motivation — why this became urgent

The first eye_squint v1.1 run (image-pair training on a 409-row curated
manifest) showed two failure modes that the current training setup
cannot cleanly fix:

- **Identity contamination at all α.** Even at inference α=0.1 the
  LoRA shifts identity — because training cells labelled α=0.15
  already had median identity cos=0.91 (9% drift) vs their anchor.
  The LoRA learned "scale=0.15 means 9% toward the training-mean
  face" as part of the axis.
- **Sublinearity mismatch.** Training cells have measured blendshape
  effect that grows 2× from α=0.30 to α=1.00 while the training α
  label grows 3.3×. A linear LoRA cannot fit that; it compromises by
  distorting the edit direction into identity space.

Both problems are *data-shape* problems: the image-pair signal encodes
what we want plus what we don't want (identity drift), and the α↔effect
relationship is nonlinear. Blendshape-loss training — where the target
vector is authored directly, not measured from a rendered edit —
sidesteps both.

## What we want

A function `φ_ours(input) → 512-d identity embedding` such that:

- `cos(φ_ours(a), φ_ours(b)) ≈ cos(ArcFace_pixel(decode(a)), ArcFace_pixel(decode(b)))`
  — the pair-cosine geometry is preserved.
- `input` is something we *already* compute during a Flux training or
  inference step. Candidates:
  1. **Cached attention features** — per-block (L, D) tensors we
     already extract for FluxSpace measurement. ~110 MB / render.
  2. **Latent z** — Flux's 16-channel VAE latent. Always present,
     no extra cost, ~256 KB / render.
  3. **Noised latent z_t at training time** — same shape as latent,
     plus a timestep scalar. Used during LoRA training.
- `φ_ours` is small (few MB), differentiable, and cheap to backprop
  through — small enough that it adds negligible memory to a LoRA
  training loop.

The point is to enable an identity-preservation loss term during
slider LoRA training without paying for VAE backprop, multi-step DDIM
rollout, or pretrained-ArcFace forward passes on decoded RGB. PuLID
needed all three because it had no other path; we have cached features
and a closed pipeline, so we have a cheaper one.

## Why this is reasonable, in one paragraph

ArcFace is just a function trained with margin loss to map images to
unit-norm 512-d vectors where same-identity vectors cluster. There is
nothing magic about pixels as the input. The geometry — the *cosine
structure* on identities — is the real artefact. That structure can
be transferred to any input domain that carries enough identity
information, by training a new model whose only job is to reproduce
ArcFace's cosines from the new input. This is knowledge distillation
in its most direct form: ArcFace acts as a frozen oracle that labels
each training example with its target embedding, and the new model
learns to imitate it.

If our cached attention features or VAE latents carry sufficient
identity signal — and they almost certainly do, given that we render
recognisable faces from them — then a distilled φ_ours can stand in
for ArcFace anywhere our pipeline needs an identity score, at a
fraction of the cost.

## Why this matters specifically for slider training

Today our slider LoRAs (Concept Sliders style) are trained with pure
flow-matching MSE loss. There is no identity term. Identity drift is
the dominant failure mode of the corpus, and we currently fight it
with prompt sanitisation and pair-averaging at the *data* level, not
the loss level. Even after that, only 67% of α=1.0 training cells pass
the ArcFace cos ≥ 0.75 gate.

The standard fix from the literature (PuLID-style: VAE-decode the
predicted x̂0, run pretrained ArcFace, compute identity loss, backprop
through everything) is incompatible with our regime:

- **Memory.** PuLID reports 41–63 GB VRAM for SDXL with the trick. On
  our 32 GB 5090, with Flux-Krea-dev's 12B params and bf16 transformer,
  we'd OOM before the ArcFace forward pass.
- **Compute.** Multi-step rollout adds k× transformer forwards per
  training step; identity loss alone would 3–5× the per-step cost of
  what is already a 50-min run per axis.
- **Data.** PuLID's regularisation against identity collapse uses 1.5M
  images of contrastive prompts. Concept Sliders trains on a few
  hundred image pairs. We can't afford the regulariser, and without
  it identity loss collapses to anchor memorisation.

A distilled φ_ours sidesteps all three:

- Memory: a few-MB MLP is rounding error against a 12B-param trunk.
- Compute: one MLP forward + one backward per LoRA training step.
- Data: the slider trainer doesn't need contrastive anti-collapse
  data, because it's not learning *to inject* identity; it's just
  being penalised when its existing edits drift.

The loss enters at one place:

```
ε̂        = LoRA-Flux(z_t, t, prompt)
ẑ0       = (z_t − √(1−ᾱ_t) · ε̂) / √ᾱ_t
id_pred  = φ_ours(ẑ0)            # OR φ_ours(features_tapped_from_LoRA-Flux)
id_anch  = φ_ours(z_anchor)      # or anchor's cached features; precomputed
L_id     = 1 − cos(id_pred, id_anch)
L_total  = L_diffusion + λ_id · L_id
```

`id_anch` is precomputed once per anchor base. φ_ours is loaded
frozen at training time. No VAE, no rollout, no ArcFace.

## The three input choices

There is a real design decision about *which* representation we
distill into. Tradeoffs:

### Option A — cached attention features

**Input.** A subset of the (L, D) tensors we already pickle in
`measure_path` from `FluxSpaceEditPair`. Probably the residual stream
or attention features at one mid-network DiT block, at a fixed pivot
timestep.

**Pros.**
- Zero extra runtime cost during training — these features fall out
  of the forward pass anyway.
- We already have ~thousands of cached pickles paired with PNGs.
- Mid-network features are known to carry rich semantic content
  (FluxSpace, DAAM, the entire mech-interp on diffusion literature).
- Very short backprop path through φ_ours.
- The framework principle in our project memory says atoms / classifiers
  read from cached features; this is the same shape.

**Cons.**
- Vendor-locked to Flux's specific block layout. Switching base model
  retrains φ_ours from scratch.
- Cached features only exist when we *enable* caching, which costs
  ~110 MB / render. We turned this off for v3.1 to save disk.
- Dimensionality is huge (L × D where L is hundreds of thousands of
  tokens × layers). φ_ours needs careful pooling design to be small.
- Less obvious how to use this representation at LoRA *training* time,
  where we don't have a pre-rendered cache — we'd need to re-tap the
  features inside the training forward pass.

### Option B — VAE latent z (clean)

**Input.** The 16-channel Flux latent at clean t=0. Shape ~128×128×16.

**Pros.**
- Always present, no extra cost, tiny (~256 KB / render).
- Single canonical representation independent of which DiT block.
- Standard ConvNet / small ViT architectures fit naturally.
- Easy to assemble training data: VAE-encode FFHQ / CelebA / our
  existing PNG corpus.

**Cons.**
- VAE encode is a small extra step (~100 ms per image; one-time, not
  per-training-step).
- Less semantic than cached features; identity may need more capacity
  to extract.
- Spatial alignment varies with face placement in the frame —
  training data needs portrait-cropped or position-equivariant design.

### Option C — VAE latent at noised t (timestep-conditioned)

**Input.** `(z_t, t)` — noised latent at training timestep t.

**Pros.**
- *Exactly* the representation available at training time, no further
  transformation needed.
- Removes the "denoise to t=0 first" requirement — this is the cleanest
  fit for slider training.
- Distillation supervision is still ArcFace on the clean image, which
  is what we want.

**Cons.**
- Highest difficulty for the discriminator: extract identity through
  noise. May need bigger model, may saturate.
- Training data assembly: for each clean image with target embedding,
  generate (z_t, t) pairs at varying t. Standard but adds bulk.
- Quality likely degrades at high t; we'd need to restrict the
  identity-loss timesteps to t < some threshold.

### Recommendation

Build A first, B as fallback, C as the eventual production form.

A is cheapest to validate with existing data. If A works (pair-cosine
correlation r ≥ 0.9 on holdout), B and C become engineering exercises:
re-train the same head architecture against latent and noised-latent
inputs, compare quality. If A *doesn't* work, the cached features are
too lossy for identity at our chosen tap point and we know to reach
for B before doing more architecture work.

## Distillation as the training procedure

The training loop for φ_ours, for any input choice:

```
INPUTS:
  - face image dataset (FFHQ / CelebA / our render corpus)
  - frozen pixel ArcFace φ_pixel
  - frozen Flux VAE encoder E_VAE
  - frozen Flux trunk (only if we need cached features)

PER-EXAMPLE:
  x       = face image (RGB, 112×112 aligned for ArcFace; arbitrary
            crop for our trunk)
  target  = φ_pixel(x)                   # 512-d, unit-norm
  input_  = chosen rep:
              A — cached features at fixed block / timestep
              B — E_VAE(x), clean
              C — noise-add E_VAE(x) at random t, output (z_t, t)

LOSS:
  L = 1 − cos(φ_ours(input_), target)
```

Notes on the procedure:

- **No identity labels.** We do not need "this is identity 47, photo 12"
  annotations. The embedding is its own label. Any face image works.
- **Frozen oracle.** φ_pixel is never updated. We are not improving
  ArcFace; we are translating it.
- **L2 vs cosine.** Cosine on unit-norm targets is the simplest, since
  our downstream use is cosine similarity. MSE on the raw 512-d also
  works and is sometimes more stable; pick after a sanity run.
- **Architecture.** Start small. For B/C: a 4-block ResNet adapted to
  16-channel input, ~10M params, GAP → MLP → 512-d. For A: a small
  attention pooling + MLP, sized to whatever feature shape we tap.
  Resist scaling until pilot data forces it.
- **Augmentation.** For B/C, train on the same random crops / flips /
  light colour-jitter that pixel ArcFace was trained on, so identity
  invariances transfer. Don't crop so hard that the face leaves the
  frame; the latent has no way to recover information that left.

## The pilot — what to run first

A single notebook-scale experiment that either validates the whole
approach or kills it cheap.

### Step 1 — assemble training data

Per memory, `output/demographic_pc/classifier_scores.parquet` already
contains ArcFace embeddings for every render in our 7332-row index.
Per memory, ~315 GB of cached attention pkls live on the external
drive. The training data pretty much already exists; it just needs
joining.

```
for each row in sample_index where measure_path is not None:
    target_embedding = classifier_scores[row.id].arcface_embedding
    cached_features  = load_pkl(row.measure_path)
    yield (cached_features, target_embedding)
```

If we cache ~50% of renders this gives us several thousand pairs,
spanning bases × seeds × αs × axes. Not enormous, but plenty for a
small head.

### Step 2 — train a small head

Tap one block (probably mid-network, where identity tends to be
densest) at one fixed pivot timestep. Pool spatially. MLP to 512-d.
Train with cosine loss. Tens of minutes on a single GPU.

### Step 3 — evaluate pair-cosine correlation

Held-out 1000-pair set: pairs of renders, half same-identity (same
base prompt + similar α), half different. Compute:

- `c_pixel = cos(ArcFace_pixel(png_a), ArcFace_pixel(png_b))`
- `c_ours  = cos(φ_ours(features_a), φ_ours(features_b))`

Decision rules:

- **r(c_pixel, c_ours) ≥ 0.9, |bias| < 0.05** → φ_ours is a usable
  drop-in. Move on to integrating into the LoRA training loop.
- **0.7 ≤ r < 0.9** → signal is there but lossy. Try Option B (latent),
  bigger head, or different tap block, before giving up.
- **r < 0.7** → cached features at this tap are too lossy. Switch to
  Option B and re-evaluate; if B also fails, the whole approach is in
  trouble (more diagnosis needed).

This is a one-day experiment using data we already own.

### Step 4 — integrate into slider training

Once φ_ours is validated, modify `train_flux_image_slider.py`:

- Load φ_ours frozen at startup.
- For each training step, after the LoRA forward pass, tap whatever
  representation φ_ours expects (extra forward through the trunk if
  needed; cheap if it's a single block tap).
- Compute `id_pred` and `id_anch` (anchor precomputed at run start).
- `L_total = L_diffusion + λ_id · (1 − cos(id_pred, id_anch))`

Start λ_id small (0.1) and check that diffusion loss still goes down.
PuLID reports λ_id = 1.0 with their Lightning rollout; we have a
cheaper but possibly weaker signal, so the right scale is empirical.

### Step 5 — A/B comparison

Train two eye_squint v1.2 sliders side by side:

- Identical hyperparameters.
- One with `λ_id = 0` (baseline, like v1.1).
- One with `λ_id ∈ {0.1, 0.5, 1.0}`.

Measure ArcFace pass rate at α ∈ {0.45, 0.75, 1.0} on a held-out
preview grid. Goal: lift α=1.0 pass rate above 67% (the corpus
ceiling) without sacrificing visible squint Δ.

This is the only test that matters. The pilot in step 3 proves that
φ_ours has the right *geometry*; step 5 proves that geometry is
useful as a training signal.

## Why warm-starting from ArcFace weights is not the move

Tempting because "we already have those weights, why not load them?"
but in practice:

- ArcFace's first conv layer expects 3-channel RGB. Our latent input
  is 16-channel. Replacing the stem and reusing the rest *can* work
  but throws away the receptive-field calibration the rest of the net
  was trained against.
- Spatial scale differs (ArcFace: 112×112, latent: 128×128).
- The mid-late-layer "face features" we want to inherit might not
  transfer across input modalities in the way they would across
  similar-modality datasets — that's exactly the unstated assumption
  that warm-starting bets on, and it's brittle.
- Distillation gives us the same end state with stronger guarantees:
  pair-cosines are preserved by construction, with no opaque feature-
  transfer assumption.

Warm-starting could help φ_ours converge faster on Option B (where the
input is at least image-shaped), but the architecture-mismatch tax
makes it unlikely to dominate. Skip unless distillation pilot shows
slow convergence.

## Risks and unknowns, in honest order

- **The cached features at our chosen tap might not carry identity
  cleanly.** They carry semantic content for FluxSpace edits, but
  identity-preservation may need a different tap. Mitigated by
  step 3 of the pilot: r < 0.7 kills the option early.
- **Identity collapse into anchor pixels.** Even a perfect φ_ours,
  used naively as a training loss, can drive the LoRA to memorise the
  anchor exactly. Mitigation in PuLID was the contrastive alignment
  loss. Our smaller setup may not need it because (a) Concept Sliders'
  pair-MSE pulls the LoRA *toward* the edit, not the anchor, so the
  identity loss is a regulariser rather than the dominant gradient,
  (b) we train with α-augmented data, so the LoRA can't lock onto a
  single anchor pose. We should *measure* collapse risk by checking
  whether `L_id` stays > 0 throughout training (not collapsing to 0
  by overfit).
- **Loss-scale balancing.** Diffusion MSE and identity cosine have
  very different magnitudes. The right λ_id is empirical and probably
  axis-dependent. Sweep, don't guess.
- **Distillation drift across image domains.** φ_ours is trained on
  rendered face crops at a particular size / lighting / framing. If
  we slider into more extreme domains (heavy expressions, occlusions),
  φ_ours may misbehave because its training distribution didn't cover
  them. Monitor: keep an OOD validation set of weird renders and
  check that pair-cosine geometry holds.
- **Vendor lock to Flux.** φ_ours is specific to whichever input we
  picked. Switching base model = retrain. Acceptable cost given that
  retrain is a few hours at most, and the pipeline is already
  Flux-locked through ComfyUI custom nodes.

## What we are *not* doing

- We are not retraining ArcFace itself on faces.
- We are not building a new identity-preserving model from scratch.
- We are not trying to compete with PuLID on absolute identity
  fidelity for arbitrary identities. PuLID is meant for "given any
  ArcFace embedding, generate that person." We just want "during
  slider training, don't drift from the existing identity in the
  pair." Much narrower problem, much cheaper solution.

## Sequence of work, when we pick this up next week

1. **Prep:** make sure curator and v1.1 slider are done so we have a
   v1.1 baseline to compare against.
2. **Data join:** build the (cached_features, ArcFace_embedding) parquet
   from existing artefacts. (Half day.)
3. **Pilot head training:** small MLP, cosine loss, hold out 1000
   pairs. (Half day.)
4. **Pair-cosine evaluation:** decide which option survives. (Hours.)
5. **Integration:** modify trainer to load φ_ours and add L_id.
   (Half day.)
6. **A/B run:** v1.2 with vs without identity loss on eye_squint.
   ~2 hours of training + collage comparison.
7. **Document:** what worked, what surprised us, λ_id sweep, recommend
   whether to retrofit other axes.

Total estimated cost: ~3 days of one person's attention if everything
goes smoothly, ~1 week with normal bumps. No new GPU procurement, no
new dataset acquisition, no model-family change. All risk lives in
"will the cached features be informative enough", which step 3
answers in hours.

## The one-line summary

Distil pixel ArcFace into a small head over our existing cached
features, validate with pair-cosine geometry against the pixel
oracle, then drop the resulting `φ_ours` into the slider training
loss. ArcFace pretrained weights are reused as the *labelling
oracle*, not as initialisation, which is the cleanest reuse pattern
the literature has — we just haven't applied it to slider training
because no one has needed to before.

---

# Companion plan — distil MediaPipe into `φ_bs_latent` and train sliders with direct blendshape targets

Added 2026-04-24 evening after the v1.1 result. Same distillation
mechanism, different oracle, different role in the loss. Where latent
ArcFace is a *regulariser* that keeps identity near the anchor, latent
MediaPipe is the *primary supervisor* of the slider axis.

## What changes about the slider trainer

Current (image-pair) training:

```
pick edited cell (anchor_image, edited_image, α)
encode both to latents, noise the edited
LoRA-Flux predicts velocity
loss = MSE(predicted velocity, true velocity from edited latent)
```

The blendshape effect is *implicit* in the data — we measure it
afterwards to see whether the LoRA learned what we wanted.

Blendshape-loss training:

```
pick anchor image, pick target α
anchor_bs = mediapipe_blendshapes(anchor_image)  # 52-d ARKit
target_bs = anchor_bs.clone()
target_bs[axis_idx] += α * delta_per_unit_α       # synthetic target

z_0 = VAE.encode(anchor_image)
t   ~ sample_timestep()
z_t = noise-add(z_0, t)
ε̂    = LoRA-Flux(z_t, t, prompt, scale=α)
ẑ0   = (z_t − √(1−ᾱ_t) · ε̂) / √ᾱ_t
L_bs = ‖φ_bs_latent(ẑ0) − target_bs‖²             # primary axis supervision
```

The blendshape effect becomes the *training signal*, not a measurement
downstream. The target is authored directly, so α ↔ effect is linear
by construction.

## Why this structurally fixes the v1.1 failure modes

- **Identity contamination gone.** The loss only measures 52-d
  blendshape output. The anchor's identity features have no reason to
  shift unless shifting them is the cheapest way to reduce L_bs —
  and shifting a specific blendshape by α × δ is a smaller perturbation
  than re-rendering a different person, so the LoRA converges on a
  near-pure blendshape edit. Bundle with L_id (latent ArcFace) for a
  hard identity guarantee.
- **Linearity by construction.** We define `target_bs(α) = anchor_bs +
  α · unit_delta`. The sublinearity of FluxSpace editing disappears
  because *we* control the target, not a nonlinear rendering process.
- **ARKit-compatible by construction.** The LoRA is directly optimised
  to hit ARKit blendshape values. Round-trip (ARKit vector → render →
  measure ARKit) works by design — it's literally what the loss is.
- **Per-axis scales to the 12-atom atlas trivially.** Adding a new axis
  = adding a new one-hot target template to a config. No new rendering
  corpus per axis.

## What `φ_bs_latent` is

A small differentiable PyTorch model that takes a Flux 16-channel latent
(or noised latent + t) and outputs a 52-d ARKit-compatible blendshape
vector matching what MediaPipe FaceLandmarker would return on the
VAE-decoded image.

- **Architecture.** Start with the same shape as `φ_latent` for ArcFace
  — a 4-block ResNet adapted to 16-channel input, GAP, MLP head to 52-d.
  The output head uses a sigmoid or clipped-linear activation because
  ARKit blendshape values are in [0, 1].
- **Size target.** Few-MB. It is translating a single-network output,
  not learning face geometry from scratch; capacity requirements are
  modest.
- **Loss on distillation.** MSE on the 52-d vector, optionally weighted
  by per-axis variance so over-expressed axes don't dominate.

## Why MediaPipe is the right oracle for this

- **Outputs are standardised (ARKit 52).** The ecosystem compatibility
  we'd want the sliders to hit eventually is what MediaPipe already
  reports.
- **Blendshape regression is robust** within its operating envelope
  (frontal face, reasonable framing). Our corpus is all frontal
  portraits — close to MediaPipe's training distribution.
- **We already have 7332 rows of (image, blendshape)** in
  `models/blendshape_nmf/sample_index.parquet` with 52 `bs_*` columns.
  The distillation corpus already exists; we just need to pair it with
  latents.

## What we're NOT doing with this plan

- Not reimplementing MediaPipe's internal face-mesh pipeline (BlazeFace
  + FaceMesh + blendshape regression MLP). Too much engineering, and we
  don't need it: distillation on the final output IS the shortcut.
- Not training a latent-space face-mesh model with landmark supervision.
  We only need the 52-d output; everything upstream can stay in
  MediaPipe's world during corpus generation.
- Not throwing away image-pair training. Image-pair data becomes a
  *validation oracle* — trained LoRA renders 20 faces at α=0, 0.5, 1.0,
  MediaPipe scores them, we check that measured blendshapes match
  targets. Same infrastructure, new role.

## Obstacles and how they map to the ArcFace plan

| Obstacle | ArcFace plan solution | Blendshape plan solution |
|---|---|---|
| Oracle expects RGB | distil into latent | same — distil into latent |
| Oracle not differentiable (for ours-in-loop use) | distillation produces differentiable φ | same |
| x̂0 garbage at high t | timestep-condition φ, or restrict loss to low-t | same |
| Flow-matching loop produces ε not image | one-step x̂0 estimate, cheap | same |
| LoRA "cheats" the predictor | pair with L_diffusion regulariser | pair with L_id AND L_diffusion_on_anchor |

The blendshape plan inherits every solution the ArcFace plan has. The
distillation module is literally the same code, just different head
output and different training labels.

## Composed loss for the slider trainer

Once both oracles exist:

```
L_total = λ_bs   · ‖φ_bs_latent(ẑ0) − target_bs‖²             # axis target
        + λ_id   · (1 − cos(φ_id_latent(ẑ0),
                             φ_id_latent(z_anchor)))           # identity
        + λ_diff · L_diffusion_on_anchor_at_scale_0            # LoRA zero-init regulariser
```

Three terms, each addressing one failure mode:

- `L_bs` pushes the blendshape axis to the target value.
- `L_id` pulls the face toward the anchor's identity.
- `L_diff` ensures the LoRA has near-zero action at scale=0 — without
  this, nothing prevents the LoRA from memorising a distorted face that
  happens to have the right blendshape.

## Pilot and validation — same two-step pattern as ArcFace

For each oracle, validate that the latent predictor matches the pixel
oracle before trusting it in training:

- **ArcFace pilot:** pair-cosine correlation r ≥ 0.9 between
  `cos(φ_pixel(img_a), φ_pixel(img_b))` and
  `cos(φ_latent(z_a), φ_latent(z_b))` on a 1000-pair holdout.
- **Blendshape pilot:** per-axis MSE between
  `MediaPipe(img)` and `φ_bs_latent(VAE.encode(img))` on a 1000-image
  holdout. Target: median ≤ 0.05 per axis (ARKit values are in [0,1]).

If either fails, the whole approach is fiction for that oracle — know
this cheaply, before sinking GPU into a retrained LoRA.

## What we already own

- `output/demographic_pc/classifier_scores.parquet` — ArcFace embeddings
  for every render (covers ArcFace distillation corpus entirely).
- `models/blendshape_nmf/sample_index.parquet` — 52 `bs_*` columns per
  row (covers MediaPipe distillation corpus entirely).
- ~315 GB of cached attention pkls and ~7000 PNGs on disk/archive —
  source for building the (latent, target_embedding) and
  (latent, target_bs) parquets.
- Trainer, VAE, Flux — working.

Literally every ingredient for the distillation corpora already exists.
The new build is: joining parquets, training two small heads, adding a
loss branch to the trainer.

## Combined sequence of work, when we pick this up

Order the steps so earlier ones validate the approach before committing
to the later ones.

1. **Distillation corpora** — build
   (latent, arcface_embedding) and (latent, mediapipe_bs) parquets by
   joining existing tables with freshly VAE-encoded images.
   Half a day. CPU/small-GPU only.
2. **Train φ_id_latent and φ_bs_latent** with cosine and MSE losses
   respectively. One day for both, single GPU.
3. **Pair-cosine eval (ArcFace) + per-axis MSE eval (MediaPipe).** If
   either falls below the bar, stop and iterate on the head.
4. **Trainer integration — identity loss only first.** Add `L_id` on
   top of existing image-pair objective. Rerun v1.2-baseline with this
   term and see whether α=0.1 identity drift goes away.
5. **Trainer integration — blendshape loss.** Replace image-pair MSE
   with `L_bs`; keep `L_id` and add `L_diff` regulariser. This is the
   full-rebuild trainer.
6. **A/B compare** image-pair v1.2 vs combined-loss v2 on eye_squint
   at α ∈ {0.25, 0.5, 0.75, 1.0} on held-out bases.
7. **Generalise to other axes.** Pick a second axis (brow_lift or
   mouth_stretch) and train v2 with the same trainer, different
   target_bs template. Zero new rendering.
8. **Atlas plan.** Once two axes work, scope the 12-atom atlas: config
   file, axis definitions, batch training, delivery as ARKit-compatible
   LoRA bundle.

Total estimated cost: ~1 week for steps 1–6, another week for 7–8. No
new rendering, no new GPUs, no new model family. Risk lives in
distillation quality (step 3); known cheap to evaluate.

## The two-line summary

Distillation solves the same obstacle twice. ArcFace gives us a
differentiable latent identity oracle to use as a regulariser.
MediaPipe gives us a differentiable latent blendshape oracle to use as
the primary axis supervisor. Running them together turns "train a
slider" from "render, curate, hope" into "declare a target blendshape
vector and an identity anchor, train."

## Companion docs

- `2026-04-24-identity-preserving-lora-survey.md` — broader survey of
  identity-preservation in LoRA training, ranks PuLID-Flux stacking
  vs custom training. This doc is the "if we want to go custom, how
  cheaply can we do it" follow-up.
- `2026-04-24-non-standard-losses-lora-math.md` — the math of why
  non-standard losses are hard to add to diffusion training, and what
  PuLID does about it. Distillation sidesteps most of those obstacles.
- `2026-04-23-framework-procedure.md` — the canonical procedure doc;
  this plan is consistent with its principle that classifiers /
  scorers are diagnostic readouts over cached features.
