---
status: live
topic: demographic-pc-pipeline
---

# Concept slider math: why our LoRAs learn global, not local

User observation that prompted this: at slider strength −1.5 our renders
wash out / lose detail, and at +1.5 they over-expose / shift "vibe"
(gym-bro vs academic), but the *local* feature we want (rims on eyes)
under-engages. This is the local/global descriptor distinction from
classical CV applied to the LoRA's training signal. Below is the math
that explains why the L2 / xattn / rank-r setup we're using makes this
the *expected* failure mode rather than a tuning issue.

## The training step, one forward pass

**Inputs.**

- Pair $(x_{pos}, x_{neg})$ from the dataset — for v2 with-glasses and
  without-glasses, same demographic, **not** same identity (FluxSpace
  α=0 vs α=0.6/0.8 renders).
- Caption $c_{demo}$ from the `.txt` file ("a photorealistic portrait
  photograph of a Lebanese woman, plain grey background, studio
  lighting").
- Slider prompts (yaml `slider:` block):
  - $c_{target}$ = "a portrait photograph of a person"
  - $c_{pos}$ = $c_{target}$ + "wearing eyeglasses, glasses on face,
    eyewear"
  - $c_{neg}$ = $c_{target}$ + "without glasses, no eyewear, bare face"
- Sample $t \sim U(0, 1)$, noise $\varepsilon \sim \mathcal{N}(0, I)$.

**Encode + noise.** Latents from VAE (cached on disk):

$$z_{pos} = \mathrm{VAE}(x_{pos}), \qquad z_{neg} = \mathrm{VAE}(x_{neg})$$

Flow-matching corruption with **shared** $\varepsilon$ at **shared** $t$:

$$x_t^{pos} = (1-t)\,z_{pos} + t\,\varepsilon, \qquad
  x_t^{neg} = (1-t)\,z_{neg} + t\,\varepsilon$$

**Frozen-model targets** (LoRA off; $\eta$ = `guidance_strength` = 4.0
in v4/v6):

$$v_{pos}^{tgt} = v_\theta(x_t^{pos}, c_{pos}) + \eta\,
  \bigl(v_\theta(x_t^{pos}, c_{pos}) - v_\theta(x_t^{pos}, c_{neg})\bigr)$$

$$v_{neg}^{tgt} = v_\theta(x_t^{neg}, c_{neg}) - \eta\,
  \bigl(v_\theta(x_t^{neg}, c_{pos}) - v_\theta(x_t^{neg}, c_{neg})\bigr)$$

i.e. classifier-free-guidance-style amplification of the (pos − neg)
direction in velocity space, by factor $\eta$. **The LoRA's job is to
match these amplified targets while only seeing $c_{target}$ at training
time.**

**LoRA-active predictions** (multiplier $\pm 1$):

$$v_{pos}^{pred} = v_{\theta + \Delta}(x_t^{pos}, c_{target}, m{=}{+}1)$$

$$v_{neg}^{pred} = v_{\theta - \Delta}(x_t^{neg}, c_{target}, m{=}{-}1)$$

where $\Delta = \tfrac{\alpha}{r}\, B A$ applied at xattn projection
weights. ($\alpha$ = `linear_alpha`, $r$ = `linear` rank, both yaml
knobs; v6 has $\alpha{=}1, r{=}16 \Rightarrow$ effective scale
$\tfrac{1}{16}$.)

**Loss.** Per-pixel, per-channel L2, averaged uniformly:

$$L = \frac{1}{N} \sum_{i=1}^{N}
  \Bigl[(v_{pos}^{pred} - v_{pos}^{tgt})_i^2
       + (v_{neg}^{pred} - v_{neg}^{tgt})_i^2\Bigr]$$

with $N = C \times H \times W$ of the latent. For Flux at 512×512 the
patched latent is $\approx 16 \times 32 \times 32 \approx 16{,}000$ cells.

## Backprop into the LoRA

Gradient into LoRA params:

$$\frac{\partial L}{\partial B} \;\propto\;
  \sum_{i=1}^{N} (v_i^{pred} - v_i^{tgt}) \cdot
  \frac{\partial v_i^{pred}}{\partial B}$$

The sum runs uniformly over all $N$ spatial-channel cells. **L2 has no
notion of which cells are "the eye region" and which are "background
sky".** Every cell contributes equally to gradient mass.

This is the entire mechanism. The next three sections explain why this
formulation systematically prefers global over local.

## Why the gradient prefers global over local

### (1) Pixel-mass asymmetry

A *global* attribute (overall contrast, "academic vibe", skin tone,
background formality) differs across $N_{global} \approx N$ cells. A
*local* attribute (lens rims, hinges) differs across $N_{local} \approx
K$ cells with $K \ll N$ (eye region $\sim$ 200 of 16,000 = 1.25%).

Their L2 contributions scale by their support:

$$\|\delta_{global}\|_2^2 \sim N\,\delta^2, \qquad
  \|\delta_{local}\|_2^2 \sim K\,\delta^2$$

For the same per-cell amplitude $\delta$, the global signal contributes
$N/K \approx 80\times$ more to the loss. **Gradient descent climbs the
steepest direction first → global features before local.**

This is the core mathematical fact behind the user's intuition.

### (2) Rank truncation

A rank-$r$ LoRA captures at most the top-$r$ singular components of
the residual matrix $R = v^{tgt} - v^{pred}$ across the training
distribution. SVD ranks features by

$$\sigma_k(R) \approx \text{(per-cell amplitude)}_k \times
  \sqrt{\text{support size}_k}$$

The top $\sigma_k$ are exactly the global wash / contrast / semantic-
bundle features (large support, moderate amplitude). Local features
(rim edges, lens highlights) sit deep in the spectrum.

For $r = 16$ (our config), the LoRA reserves all 16 singular directions
for the loudest signal — global. There is **no capacity left** to fit
the local feature even if its gradient were non-zero. The user's
"LoRA looks in the wrong area" intuition is the LoRA picking up the
top-16 PCs of a residual whose top PCs are global.

### (3) xattn structural bias

LoRA at $W_q, W_k, W_v$ shifts the cross-attention by a *spatially
uniform* weight delta. Cross-attention is

$$\text{attn}(x) = \mathrm{softmax}\!\bigl(QK^\top / \sqrt{d_k}\bigr)\,V$$

with $Q = (W_q + \Delta W_q)\,x_{tokens}$. A uniform $\Delta W_q$ shifts
**every** spatial token's query by the same direction. The softmax then
mixes spatial tokens before producing the output. The LoRA cannot
encode "modify token 17 differently from token 18" via a single weight
matrix delta — that would require spatially-conditional weights, which
this architecture does not provide.

So the natural expressive language of an xattn LoRA is "shift the
global semantic field of all spatial positions together," not "edit
the pixels in this specific region." Localization in the output must
come from *content differences* between tokens (i.e., from the frozen
attention pattern picking out which tokens to attend to), not from
the LoRA itself.

## Putting it together

The user's wash-out / over-exposure / semantic-bundle triad is the
**expected output** of:

1. L2 loss summed uniformly over spatial cells → global features have
   $N/K$ times more loss-mass than local features.
2. Rank-$r$ LoRA truncating to top-$r$ singular components → those
   components are global.
3. xattn-only scope → uniform weight delta → no architectural
   selectivity for spatial regions.

These three compound. **L2 does not know "eyes only matter," xattn
cannot say "eye region only," and rank-$r$ truncation cannot store
"rims as a localized basis function."** The LoRA learns whichever
signal has the most loss-mass — global, by construction.

The intuition from classical CV is exact here: we are training a
*global descriptor* and asking it to act *local*.

## Fixes, in order of leverage

1. **Anchor class** (one line of yaml; available now). Set
   `anchor_class: "a portrait photograph of a person"`. Adds a third
   loss term

   $$L_{anchor} = \|v_{\theta + \Delta}(x_t, c_{anchor}, m{=}0)
                  - v_\theta(x_t, c_{anchor}, m{=}0)\|^2$$

   penalising the LoRA for changing anything when no slider intent
   is expressed. Reduces drift orthogonal to the slider direction.
   Doesn't fix the local/global asymmetry but cuts the bundle.

2. **Spatial loss masking**. Replace $\sum_i$ with $\sum_i w_i$ where
   $w_i = 1$ in eye region, $w_i = \varepsilon$ elsewhere. MediaPipe
   gives eye landmarks already. Inverts the support asymmetry — local
   gradient now exceeds global. **Directly addresses (1).** Requires
   ai-toolkit code patch (concept_slider loss is in
   `jobs/process/ConceptSliderProcess.py` or similar).

3. **High-frequency loss term.** Add a Laplacian/Sobel residual:

   $$L_{HF} = \lambda\, \|\nabla v^{pred} - \nabla v^{tgt}\|^2$$

   Edges and rims live at high spatial frequencies; wash-out lives at
   DC. This penalises DC matching and rewards edge matching. Most
   physics-correct answer to the local/global problem. Requires code.

4. **Larger rank.** $r = 64$ or $128$ gives the LoRA more SVD
   components — local features can join the basis after the global
   ones. Helps (2). But increases bundle capacity proportionally;
   needs to be paired with anchor.

5. **Pixel-aligned pairs.** The cleanest mathematical fix. If $x_{pos}$
   and $x_{neg}$ differ only at the eye region (same identity, edited
   in place), the diff signal is sparse spatially → the gradient is
   sparse spatially → support asymmetry inverts. Helps (1) at the
   data level.

   Our v2 pairs are NOT pixel-aligned — they are different FluxSpace
   renders that happen to share a base prompt. Same demographic, same
   seed, but different α gives different overall renders, not the
   same identity edited. To get true pixel-aligned pairs we'd need
   either (a) GAN-inversion + targeted edit, (b) image-based slider
   training (Concept Sliders Path B), or (c) a different generation
   pipeline that produces same-identity-different-attribute pairs.

## Lowest-friction next experiment

`anchor_class: "a portrait photograph of a person"` in the yaml. One
line, no code. Tests whether the bundle component of the
local/global problem is the dominant pain. Run as v6.5 or v7.

The architectural fix (spatial mask in loss, code-level) is the right
long-term answer but needs ai-toolkit modification.
