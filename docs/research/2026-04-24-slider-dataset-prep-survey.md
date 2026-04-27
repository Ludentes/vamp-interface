---
status: live
topic: demographic-pc-pipeline
---

# Slider-LoRA dataset prep: what canonical recipes actually do

Background: training eye_squint Concept Sliders on Flux from a 409-cell image-pair corpus produced a forced choice — v1.1 ships the axis but destroys identity; v1.2 preserves identity but has no axis. Diagnosed root cause: training cells labelled α=0.15 already carry median ArcFace id_cos=0.91 vs the α=0 anchor, so the LoRA learns "scale=0.15 means 9% toward training-mean face" as part of the slider direction. Target edit magnitude also grows sublinearly (2× from α=0.3→1.0 while α label grows 3.3×), which a linear LoRA scaling can't fit without distortion.

Before iterating blindly, we want to know: what do the canonical recipes actually do? This doc is the answer.

## What the canonical Concept Sliders repo does

Source: [rohitgandikota/sliders/README.md](https://github.com/rohitgandikota/sliders/blob/main/README.md), [trainscripts/imagesliders/train_lora-scale.py](https://github.com/rohitgandikota/sliders/blob/main/trainscripts/imagesliders/train_lora-scale.py), [trainscripts/imagesliders/data/config.yaml](https://github.com/rohitgandikota/sliders/blob/main/trainscripts/imagesliders/data/config.yaml).

The README is blunt:

> To train image-based sliders, you need to create a ~4–6 pairs of image dataset (before/after edit for desired concept). ([README](https://github.com/rohitgandikota/sliders/blob/main/README.md))

Not 400. Not 4000. **Four to six**. Two folders (`bigsize/`, `smallsize/`), matched filenames, CLI arg `--scales '1, -1'` (binary endpoints, not a graded α grid).

Reading `train_lora-scale.py` (lines 172–334 is the full training loop), the mechanics are:

- **Binary endpoints**, not a graded α grid. `scales_unique` is derived from the `--scales '1,-1'` CLI arg. Each step samples `scale_to_look = abs(random.choice(list(scales_unique)))` — if you pass `'1,-1'` this is always `1`. The repo supports multi-scale (you could pass `'1,0.5,-0.5,-1'` with folders named accordingly) but the documented recipe is binary.
- **Same seed, same timestep, paired noise** between high and low branches. The script reuses `torch.manual_seed(seed)` before each `get_noisy_image` call, so `img1` and `img2` are noised along the same trajectory. This is the identity-control mechanism. The LoRA only has to explain the difference between two images sharing a diffusion trajectory — not between two points in full pixel space.
- **Images resized to 256×256** regardless of source resolution.
- **Two losses per optimizer step**, accumulated (no `zero_grad` between). This is inherited from Ostris's ai-toolkit recipe, as the code comment states:
  > `## NOTICE NO zero_grad between these steps (accumulating gradients) # following guidelines from Ostris (https://github.com/ostris/ai-toolkit)`
  - **High-side loss**: `network.set_lora_slider(scale=+scale)`; predict noise on `denoised_latents_high` with text embeddings `(unconditional, positive)` at `guidance_scale=1`; MSE against `high_noise` (the noise that was added to img2 to produce the noised latent).
  - **Low-side loss**: `network.set_lora_slider(scale=-scale)`; predict noise on `denoised_latents_low` with text embeddings `(unconditional, unconditional)` (yes, unconditional twice — so the conditioning is empty); MSE against `low_noise`.
- **The low side is trained without text conditioning.** At `scale=-1` the LoRA must reconstruct an unconditional noise prediction on the `low` image. This is what forces the slider to be symmetric and prevents the negative side from being a free variable that can drift anywhere.
- Rank 4, alpha 1, lr 2e-4, 1000 iterations, batch size 1, SD1.x. Trained at 256×256 for image sliders.
- `max_denoising_steps=50`, timestep `t` uniformly sampled in [1, 48] every step.

The training signal is not "α=0 anchor → α=+1 full edit". It is **"same noise trajectory, add LoRA at ±1 to flip between the two endpoint images"**. There is no intermediate α in the dataset. The continuous slider scale at inference time is an extrapolation that works because the LoRA parametrises a rank-4 direction, not a lookup table.

Additional decision from the repo: `--attributes 'male, female'` for text sliders is specifically for disentanglement when the primary edit correlates with a confound. The age slider would otherwise drag gender along; adding `attributes='male, female'` trains the direction to be invariant across those prompts ([README](https://github.com/rohitgandikota/sliders/blob/main/README.md) — "age slider makes all old people male"). There is no analogous `attributes` flag documented for the image slider path.

## What the official Flux slider notebook does

Source: [rohitgandikota/sliders/flux-sliders/train-flux-concept-sliders.ipynb](https://github.com/rohitgandikota/sliders/blob/main/flux-sliders/train-flux-concept-sliders.ipynb).

**The Flux notebook does NOT do image-pair training at all.** It's text-pair only. This is a significant finding for us — we assumed the Flux port followed the image-slider recipe. It does not.

Concrete hyperparameters from cell 3:

```
pretrained_model = black-forest-labs/FLUX.1-schnell (or FLUX.1-dev)
rank = 16
alpha = 1
train_method = 'xattn'
lr = 0.002
max_train_steps = 1000
eta = 2                  # guidance multiplier in the synthetic-target construction
weighting_scheme = 'none'  # uniform timestep sampling, NOT logit-normal μ=0.5
batch_size = 1
height = width = 512
guidance_scale = 3.5 (dev) / 0 (schnell)
num_inference_steps = 30 (dev) / 4 (schnell)
target_prompt = 'picture of a person'
positive_prompt = 'photo of a person, smiling, happy'
negative_prompt = 'photo of a person, frowning'
```

Training loop (cell 8):

1. Sample timestep `t` uniformly (weighting_scheme='none'; `logit_mean=0.0` is defined but unused).
2. Run the Flux pipeline forward from t=0 to the sampled timestep with **the target_prompt**, producing `packed_noisy_model_input`. This is the training latent — a self-generated midway denoising state from the target prompt. There is no image dataset.
3. Run three teacher forward passes (no-LoRA, `torch.no_grad()`) with target / positive / negative prompts on the *same* noisy latent.
4. Construct the synthetic target:
   ```python
   gt_pred = target_pred + eta * (positive_pred - negative_pred)
   gt_pred = (gt_pred / gt_pred.norm()) * positive_pred.norm()
   ```
   This is a CFG-style extrapolation in prediction space: push `target_pred` along the `(positive − negative)` direction by `eta=2`, then renorm to the positive branch's magnitude (the renorm is the identity-preservation trick — prevents the LoRA from learning a bigger-magnitude output).
5. Student forward pass: LoRA-on, target_prompt, same latent. MSE against `gt_pred`.
6. `eta=2` is the only place α lives during training. At inference, `networks[i].set_lora_slider(scale=S)` with S ∈ [-5, -2.5, 0, 2.5, 5] (cell 16) extrapolates.

Critical implications:

- **No α grid during training.** The LoRA is trained at a single effective "strength" (eta=2) and extrapolated linearly at inference. The graded slider behaviour at test time is purely a property of the LoRA scaling linearity, not of graded training data.
- **No image pairs.** Identity preservation is handled entirely by (a) sharing the noisy latent across target/positive/negative/student, (b) the `gt_pred = (gt_pred/||gt_pred||)*||positive_pred||` renorm, and (c) `train_method='xattn'` limiting LoRA to cross-attention blocks only.
- **`rank=16`, `alpha=1`** — note alpha=1 at rank=16 means effective scale 1/16, much gentler than our `alpha=rank` convention.
- **`train_method='xattn'`** — only xattn, not "xattn + proj_out" as we're doing. Our proj_out targeting may be part of the identity-contamination problem: proj_out is downstream of all block interactions, so perturbing it affects residual-stream identity more than a pure xattn edit.
- **Uniform timestep sampling** (weighting_scheme='none'), not logit-normal μ=0.5. Our logit-normal bias toward mid-timesteps is plausibly good for edit-strength but is not what the reference Flux notebook does.

## SliderSpace (Gandikota et al. 2026)

Source: [arxiv.org/abs/2502.01639](https://arxiv.org/abs/2502.01639) (abstract; full PDF not fetched), [github.com/rohitgandikota/sliderspace](https://github.com/rohitgandikota/sliderspace) (linked from Concept Sliders README).

Abstract summary: discovers multiple interpretable directions from a single text prompt; each direction is a low-rank adaptor. No image pairs required — it's an unsupervised decomposition of the model's own creative variance given a single prompt. **This is actively the opposite direction from our recipe**: instead of curating 409 carefully-scaled cells, SliderSpace is argmax-variance on a single prompt's generation diversity. Single-source (abstract only; not read full paper).

## Ostris ai-toolkit slider trainer

Source: [config/examples/train_slider.example.yml](https://github.com/ostris/ai-toolkit/blob/main/config/examples/train_slider.example.yml).

Ostris's recipe is also **text-pair, not image-pair**. Key settings:

- `rank: 8, alpha: 4` ("Do about half of rank")
- `steps: 500` ("I rarely go over 1000")
- `lr: 2e-4` ("4e-4 to 1e-4 at 500 steps")
- `batch_full_slide: true` — a single gradient step covers all four combined sub-steps of slider training
- `targets`: list of `{target_class, positive, negative, weight}` — polar-opposite text prompts. Comment: "You want it to be the extreme of what you want to train on... 'an extremely fat, morbidly obese person' as the prompt. Not just 'fat person'"
- `anchors` (optional, and explicitly discouraged for beginners): "a person with a face mask" as anchor for a smile slider — an image that's identity-adjacent but invariant to the edit, used to regularise identity. The config's own comment: *"these are NOT necessary and can prevent the slider from converging if not done right. leave them off if you are having issues."*

The "anchors" idea is the closest thing in the canonical toolchain to what we need — a regularising prompt that should be identity-close but edit-invariant. But even Ostris flags it as a footgun.

## Community failure mode: "the slider destroys identity"

Community guidance converges on the same diagnosis, though no single canonical "here's the fix" thread exists. Relevant references:

- [Civitai Flux Guide I: Lora Training](https://civitai.com/articles/9360/flux-guide-part-i-lora-training): "Lower Network Dimension values (8–16) make LoRAs effective only at high strengths, requiring exaggerated weight application to be noticeable." — our rank=16 sits in this zone and needs test-time `scale≥3–5`, which matches the Flux notebook's default `slider_scales=[-5,-2.5,0,2.5,5]`.
- [Concept Sliders issue #95 — "SDXL Visual Sliders training new concept has no effect at all"](https://github.com/rohitgandikota/sliders/issues/95): corroborates that image-pair training silently fails when pairs are too visually different (trajectory divergence too large to learn a rank-4 direction).
- [Avatar Concept Slider (arXiv 2408.13995)](https://arxiv.org/html/2408.13995v1) explicitly addresses identity preservation in human-avatar sliders; cited as evidence the failure mode is well-known. (Not read in detail — single source.)
- [CompSlider (arXiv 2509.01028)](https://arxiv.org/html/2509.01028): "Current challenges for slider-based generation include maintaining continuity and structural consistency such as identity preservation of a person across attribute adjustments." — canonical framing of our failure mode as an open problem. (Not read in detail.)

No "here's the fix" post found in 30 minutes of searching. The consensus workaround in community configs is: fewer pairs, higher-contrast endpoints, xattn-only target modules, train-scale=1 at training time, higher-scale at inference.

## Flux-specific gotchas we missed

1. **Our α grid is wrong by construction.** The canonical recipes train at binary endpoints (image: ±1; Flux text: eta=2) and rely on LoRA linearity to give graded sliders at inference. **There is no evidence in the canonical toolchain that multi-α supervised training is better than binary.** Our 8-value α grid is unprecedented, and the sublinear edit-magnitude growth we observe (2× vs α=3.3×) suggests the α labels don't correspond to what the LoRA can actually represent.
2. **Our α=0 anchor is contaminated.** Median id_cos=0.91 at α=0.15 means the edit direction is genuinely not a pure edit direction — it's (edit + small identity drift). The canonical image-slider recipe avoids this entirely by sharing the diffusion trajectory between high and low images (same seed, same timestep, different clean latent). Our FluxSpace pair-averaging pipeline produces images at different effective latents because the injected attention deltas accumulate across the full denoising trajectory, so α=0 and α=1 already are different seeds in effect.
3. **train_method='xattn' only.** The Flux notebook is explicit. Our `xattn + proj_out` targeting is more aggressive and likely contributes to identity leakage. proj_out is post-attention and modulates the residual stream directly.
4. **α=1 equivalent at training is eta=2, not α=1.** In the Flux notebook the LoRA is trained to match `target + 2*(positive - negative)` renormed. The "strength" the LoRA learns is fixed; graded output at inference is just multiplying the delta by `scale/1.0`. We trained multi-α and then set LoRA scale to α. Those are not the same operation.
5. **alpha=1 at rank=16, not alpha=rank.** The Flux notebook's effective LoRA magnitude is 1/16 per rank channel. Our alpha=16 at rank=16 gives 16× the update magnitude the reference code uses.
6. **Uniform timestep sampling**, not logit-normal. Our μ=0.5 logit-normal concentrates updates at mid-timesteps where structure is already partially formed — plausibly entangles edit with face structure. Canonical Flux recipe is uniform.

## Concrete recommendations for eye_squint v1.3

In descending priority.

1. **Abandon the multi-α image-pair corpus as the primary training signal for v1.3.** Switch to the canonical Flux **text-pair** recipe with `target='photo of a person, eyes open, neutral expression'`, `positive='photo of a person, eyes squinting, eyelids narrowed'`, `negative='photo of a person, eyes wide open'`. Train per the Flux notebook: eta=2, rank=16, **alpha=1 (not 16)**, train_method='xattn' only (remove proj_out), lr=2e-3, 1000 steps, uniform timestep sampling (`weighting_scheme='none'`), batch=1, 512×512. This is the recipe rohitgandikota/sliders/flux-sliders ships. It doesn't touch our image corpus at all.
2. **If image-pair training remains desired** (to ground the slider in the FluxSpace-measured squint direction rather than T5's interpretation), collapse the corpus to **binary endpoints only**: take cells at α=0 and α=1.00 that pass all gates, drop the 0.15/0.30/0.45/0.60/0.75/0.90 middle. Target ~4–6 pairs per base × 14 bases = 56–84 pairs, not 409. Train with `--scales '1,-1'` per the canonical image-slider recipe. Supplementary reason: 400 pairs at rank=4 or rank=16 is vastly over-parameterised for a rank-4-to-16 subspace; the extra capacity absorbs identity noise.
3. **Share the seed across α=0 and α=1.00 cells in each pair.** Our pipeline generates these at the same base seed but the FluxSpace attention injection changes the effective denoising trajectory, so they diverge. Check that the α=0 image is literally `FluxSpaceEditPair(scale=0)` on the exact same seed — if not, regenerate the anchor side at scale=0 rather than reusing the cached unperturbed render, to match what the LoRA will see at inference.
4. **Drop proj_out from target modules.** xattn only. This alone is likely worth most of the identity-preservation gap.
5. **Set `alpha=1` at `rank=16`**, not `alpha=rank`. Lower effective LoRA strength per training step; gives the optimizer less room to overfit identity drift.
6. **Add a "scale=0 reconstruction regulariser"** if any image-pair training remains. With λ ≈ 0.3 weight, mix in steps where LoRA scale=0 and the loss is MSE between LoRA-on and LoRA-off predictions on the α=0 anchor. This explicitly teaches the network "scale=0 means don't change anything" rather than hoping it infers that from the binary-endpoint data. Not in canonical recipe but this is a sanity-bound penalty that addresses our specific α=0 contamination problem.
7. **Uniform timestep sampling.** Drop the `logit_mean=0.0, logit_std=1.0, weighting_scheme='logit_normal'` μ=0.5 bias. Set `weighting_scheme='none'`. Our mid-timestep bias plausibly contributes to identity entanglement because structure is already forming at those timesteps.
8. **Do NOT add anchors in v1.3.** Ostris explicitly flags them as a convergence footgun; debug the simpler recipe first. Anchors are v1.4 if identity is still compromised after the above.
9. **Keep the scatter-gather curator's identity + confound gates, but use them for post-hoc LoRA evaluation, not dataset selection.** Filter rendered v1.3 LoRA outputs across the α grid and compute the same identity/edit/confound metrics. This is the diagnostic readout, per the editing-framework-principle memo: axes/atoms are never input, always output. Our 409-cell parquet manifest stays — it becomes the test set, not the training set.

### One-screen diff from current trainer config

```
- rank: 16,  alpha: 16            →  rank: 16, alpha: 1
- train_method: xattn + proj_out  →  train_method: xattn
- weighting_scheme: logit_normal  →  weighting_scheme: none
- α grid: [0, 0.15, 0.30, ..., 1.00] with scale=α per step
  →  text-pair recipe (eta=2) OR binary image-pair (--scales '1,-1')
- dataset: 409 curated cells     →  0 cells (text) OR ~56–84 cells (binary image)
- loss: MSE(pred, α-labeled cell latent)
  →  MSE(student, target_pred + 2*(pos_pred - neg_pred) renormed to pos_pred)
- add: λ=0.3 LoRA-scale=0 reconstruction regulariser on α=0 anchors (if image-pair retained)
```

## What we did not confirm in 30 min

- Whether SliderSpace changes the dataset recipe or just the optimisation (abstract only).
- Whether Avatar Concept Slider's identity-preservation trick transfers to Flux (single-source citation, not read).
- Whether any published Flux slider uses LPIPS as an identity-preservation term (no hits in 30 min of search).
- Whether the `alpha=1` at `rank=16` in the notebook is intentional or vestigial from a copy-paste. Ostris uses `alpha = rank/2`; Concept Sliders SD uses `alpha=1 at rank=4`. The Flux notebook runs alpha=1 at rank=16 — consistent with the SD defaults scaled to larger rank. Likely intentional.

## Sources

- [rohitgandikota/sliders README (canonical repo)](https://github.com/rohitgandikota/sliders/blob/main/README.md)
- [trainscripts/imagesliders/train_lora-scale.py](https://github.com/rohitgandikota/sliders/blob/main/trainscripts/imagesliders/train_lora-scale.py)
- [trainscripts/imagesliders/data/config.yaml](https://github.com/rohitgandikota/sliders/blob/main/trainscripts/imagesliders/data/config.yaml)
- [flux-sliders/train-flux-concept-sliders.ipynb](https://github.com/rohitgandikota/sliders/blob/main/flux-sliders/train-flux-concept-sliders.ipynb)
- [ECCV 2024 paper PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05660.pdf) — arXiv:[2311.12092](https://arxiv.org/abs/2311.12092)
- [Ostris ai-toolkit slider config](https://github.com/ostris/ai-toolkit/blob/main/config/examples/train_slider.example.yml)
- [SliderSpace arXiv 2502.01639](https://arxiv.org/abs/2502.01639) (abstract only — single-source)
- [Avatar Concept Slider arXiv 2408.13995](https://arxiv.org/html/2408.13995v1) (not read in detail — single-source)
- [CompSlider arXiv 2509.01028](https://arxiv.org/html/2509.01028) (not read in detail — single-source)
- [Text Slider arXiv 2509.18831](https://arxiv.org/html/2509.18831v1) (fetch exceeded size — not read)
- [Concept Sliders issue #95 (SDXL visual slider no-effect)](https://github.com/rohitgandikota/sliders/issues/95)
- [Civitai: Flux Guide I — LoRA Training](https://civitai.com/articles/9360/flux-guide-part-i-lora-training)
- [Civitai: Flux LoRA Training Parameters](https://civitai.com/articles/11394/understanding-flux-lora-training-parameters)
- [Civitai: Flux Age Slider](https://civitai.com/models/720018/flux-age-slider)
