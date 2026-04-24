---
status: live
topic: demographic-pc-pipeline
---

# Research: Slider training for Flux

**Date:** 2026-04-24
**Sources:** 11 sources — official Concept Sliders repo + ECCV paper,
SliderSpace (ICLR/arXiv Feb 2025), Text Slider (arXiv Sep 2025),
SliderEdit (CVPR 2026 oral, Nov 2025), Flux LoRA training baselines
(SimpleTuner, Apatero, Modal).

---

> **Correction (addendum below):** earlier claim that the Flux
> Concept-Sliders trainer supports visual (image-pair) mode is wrong.
> The `flux-sliders/` directory has only a text-prompt notebook;
> image-pair scripts are SDXL-only. See "Notebook inspection" section.

## Executive Summary

Training continuous attribute sliders on Flux has four viable method
families as of April 2026. **Concept Sliders (Gandikota et al., ECCV
2024)** remain the canonical approach and ship with an official
`flux-sliders` training notebook, though the authors flag Flux support
as experimental vs SDXL [1, 2, 4]. For our pipeline specifically, the
visual (image-pair) slider mode matches our corpus format one-for-one
— it requires only 4–6 before/after pairs per axis, and our overnight
screening produces 90 renders per axis × 15 alpha×seed×base cells,
which is 15–22× oversampled for direct slider training [4]. Three
newer methods reduce cost or generalize training: **Text Slider**
(arXiv 2509.18831) fine-tunes only the text-encoder LoRA and claims 5×
faster training with zero-shot transfer to Flux-schnell [3];
**SliderSpace** (arXiv 2502.01639) automatically decomposes the model
into 32–64 unsupervised sliders and works on Flux but is orthogonal to
our prescribed-axis approach [5]; **SliderEdit** (CVPR 2026 oral,
arXiv 2511.09715) learns a single adapter that generalizes across
many edits and targets FLUX-Kontext rather than base Flux [6].

## Key Findings

### Concept Sliders is the canonical Flux slider training path, and its visual mode accepts exactly our corpus format

The official repository `rohitgandikota/sliders` ships a
`flux-sliders/` subdirectory with a dedicated training notebook
(`train-flux-concept-sliders.ipynb`) and separate `flux-requirements.txt`;
the authors state "Concept Sliders now support FLUX.1 models, and the
same formulation that works for SDXL works for FLUX models too" while
simultaneously warning "right now it is experimental. Please be
patient if it doesn't work as good as SDXL. FLUX is not designed the
same way as SDXL" [1, 4].

Two supervision modes are supported. The **textual** mode needs only
four prompts per slider — `target`, `positive`, `unconditional`
(opposite of positive), and `neutral` — e.g. `target=person,
positive=old person, unconditional=young person, neutral=person`
[4]. The **visual** mode "you need to create a ~4-6 pairs of image
dataset (before/after edit for desired concept). Save the before
images and after images separately" and is recommended "for concepts
that are hard to describe in text or concepts that are not understood
by the model" [4, 7]. Our overnight-screening corpus
(`crossdemo/<axis>/<base>/s<seed>_a{0,0.25,0.5,0.75,1.0}.png`) is
already a before/after image-pair dataset along α with six demographic
bases per axis — exactly the input format the visual-slider training
code expects, but at 90 pairs per axis vs the 4–6 minimum.

### Training cost for a Flux LoRA is ~1 hour on an A100, and slider training inherits this envelope

General Flux LoRA training guides converge on: rank 16 is the default
for simple concepts (rank 32–64 for complex styles); learning rate
1e-3 to 4e-3 (a full order of magnitude higher than SDXL's 1e-4 to
3e-4); 500–1500 steps; batch size 6–8 for stability; ~30 GB VRAM
unquantized, dropping to ~9 GB with NF4 quantization on a rank-16
LoRA [8, 9]. Reported wall-clock is "under an hour" on a single A100
40 GB or comparable cloud GPU [10]. The Concept-Sliders training loop
is a guided-score objective over a rank-1 or rank-2 LoRA applied to
attention projections, which is strictly cheaper than a general LoRA
fine-tune — but the Flux-specific notebook does not publish
hyperparameters in the README so the exact slider training time on
Flux is single-source from the repo itself [1].

### Text Slider moves the LoRA to the text encoder and generalizes zero-shot to Flux, but numeric Flux results are not published

Jung et al. (arXiv 2509.18831, Sep 2025) observe that Concept Sliders
backpropagate through the denoiser, which is the expensive part, and
that for Flux this means backprop through a 12B-parameter DiT. Their
**Text Slider** applies LoRA only to the text encoder's
self-attention projections, which "eliminates the need for
backpropagation through the diffusion model" and achieves "5× faster
training than Concept Slider and 47× faster than Attribute Control"
while cutting GPU memory ~2×. On SD-XL a Text Slider trains in ~550
seconds and 5.68 GB VRAM with 1.53 M trainable parameters [3]. The
Flux claim is specifically about *transfer*: because Flux shares its
text encoder (CLIP-L + T5) with some SD-family models, a Text Slider
trained on SD-XL transfers "zero-shot to FLUX.1-schnell and SD-3
without retraining" [3]. The paper does not publish quantitative
edit-quality numbers on Flux, so for our axis dictionary this should
be treated as a promising but single-source speedup claim.

### SliderSpace finds axes automatically on Flux, which is the opposite of what we want

Gandikota et al. (arXiv 2502.01639, Feb 2025) describe a three-step
unsupervised pipeline: (1) generate ~5000 images from a prompt by
varying seeds; (2) extract CLIP features and run PCA to find the
principal semantic directions; (3) train a rank-1 LoRA per direction
aligned with cosine loss to that PC [5]. Typical experiments use 32–64
sliders; it works on both U-Net (SD) and DiT (Flux Schnell) and
outperforms Concept Sliders on art-style exploration in user studies
(73% win rate on "diverse", 67% on "useful") [5]. Despite the "no
training required" pitch, **SliderSpace does train LoRA adapters** —
the unsupervised part is only direction *discovery*, not the final
edit mechanism.

Relevance to our pipeline is *inverse*: we explicitly committed on
2026-04-24 to not discover axes (we use prescribed MediaPipe blendshape
+ demographic axes). SliderSpace is the right tool if we flip that
decision; with our current positioning its main use is as a sanity
check — do any of our 11 hand-picked axes align with Flux's natural
PCA directions?

### SliderEdit is the state-of-the-art as of CVPR 2026 but targets FLUX-Kontext, not base Flux

Zaier et al. (arXiv 2511.09715, CVPR 2026 oral) train "a single set of
low-rank adaptation matrices that generalize across diverse edits,
attributes, and compositional instructions" using Selective Token LoRA
(STLoRA). They apply it to FLUX-Kontext and Qwen-Image-Edit — i.e., to
instruction-editing models, not to base Flux.1-dev/schnell [6]. For a
project like ours that generates faces from scratch (not edits an
input), SliderEdit is architecturally adjacent but not a drop-in
replacement for Concept Sliders.

### Visual-pair slider training recipe aligns with our existing corpus

Cross-referencing [1, 4, 7]: the image-pair slider training script
`train_lora-scale.py` expects two folders, `before/` and `after/`,
with matching filenames per pair, and at least 4–6 pairs per slider.
Our overnight corpus structure
(`crossdemo/<axis>/<axis>_inphase/<base>/s<seed>_a{α}.png`) can be
remapped: α=0.0 → `before/`, α=1.0 → `after/`, with the 18 (base, seed)
pairs per axis forming the image set. The intermediate α values
(0.25/0.5/0.75) are currently unused by the official trainer but match
the "dataset with varied intensity effect" option the repo mentions for
training slider strength continuously rather than as a binary
endpoint [4, 7]. The intermediates are therefore directly usable as
intensity supervision.

## Comparison

| Method | Year | Flux support | Train data per axis | Cost | Our fit |
|---|---|---|---|---|---|
| Concept Sliders [1,2] | ECCV 2024 | Experimental, official notebook | 4 prompts OR 4-6 image pairs | ~1 h A100 rank-16 LoRA [8,9] | **Best match**: direct slider per axis |
| SliderSpace [5] | arXiv Feb 2025 | Flux-Schnell confirmed | 5k unlabelled images → PCA | ~1 h/slider × 32-64 sliders | Orthogonal — we pick axes, it discovers |
| Text Slider [3] | arXiv Sep 2025 | Zero-shot transfer, no Flux training | 4 prompts (SD-XL) | 550 s / 5.7 GB (SD-XL) | Cheap but Flux numbers not published |
| SliderEdit [6] | CVPR 2026 oral | FLUX-Kontext only | Single adapter, many axes | Not published | Wrong target (editor, not generator) |

## Notebook inspection (2026-04-24 addendum) — exact Flux training recipe

Pulled the raw notebook to `docs/research/external/train-flux-concept-sliders.ipynb`
(11 MB) and extracted the hyperparameters and training objective directly.

**Hyperparameters (cell 3):**
- Base model: `FLUX.1-dev` or `FLUX.1-schnell` (both supported, same script)
- `max_train_steps = 1000`, `bsz = 1`, `lr = 2e-3`, `lr_scheduler =
  'constant'`, `lr_warmup_steps = 200`
- LoRA: `rank = 16`, `alpha = 1`, `train_method = 'xattn'` (LoRA on
  **cross-attention only**, not on the full DiT — much cheaper than a
  full Flux LoRA)
- `num_sliders = 1` (one slider per LoRA file)
- Precision: `bfloat16`
- Training resolution: 512 × 512
- Sampler: `FlowMatchEulerDiscreteScheduler`, timestep weighting scheme
  `'none'` (uniform)
- Dev-specific: `num_inference_steps = 30, guidance_scale = 3.5,
  max_sequence_length = 512`. Schnell-specific: `num_inference_steps = 4,
  guidance_scale = 0, max_sequence_length = 256`.

**Training objective (cell 8) — the actual guided-score formulation:**

```python
# 3 frozen teacher forwards + 1 student (LoRA) forward per step
model_pred    = transformer(xt, target_prompt,   lora=ON)    # student
target_pred   = transformer(xt, target_prompt,   lora=OFF)   # teacher ref
positive_pred = transformer(xt, positive_prompt, lora=OFF)
negative_pred = transformer(xt, negative_prompt, lora=OFF)

gt_pred = target_pred + eta * (positive_pred - negative_pred)    # eta=2
gt_pred = gt_pred / gt_pred.norm() * positive_pred.norm()        # renorm
loss    = MSE(model_pred, gt_pred)
```

The student LoRA is trained to reproduce a *synthesised* flow-matching
velocity equal to the target prediction plus `eta=2` × (positive −
negative), renormalized to positive's magnitude. This is a
classifier-free-guidance-style amplification applied at training time
rather than inference time.

**Per-step cost:** 4 forward passes through the 12 B DiT (1 trainable
+ 3 frozen). For Flux-dev this is heavy; for Flux-schnell the
4-step sampler makes the `from_timestep→till_timestep` pre-roll cheap.
The `xattn`-only LoRA target keeps trainable params small but does not
reduce forward cost.

**Input format — text prompts only, not image pairs:** Only 3 prompts
are needed per slider: `target_prompt` (baseline), `positive_prompt`
(α=1 end), `negative_prompt` (α=0 end). Example from the notebook:

```python
target_prompt   = 'picture of a person'
positive_prompt = 'photo of a person, smiling, happy'
negative_prompt = 'photo of a person, frowning'
```

**Correction to earlier finding:** The `flux-sliders/` directory
contains only this text-prompt notebook. The image-pair training
scripts (`train_lora-scale.py`, `train_lora-scale-xl.py`) are
**SDXL-only** — there is no `train-flux-visual-sliders.ipynb`. Our
overnight corpus therefore does not plug into a pre-existing Flux
visual-sliders trainer; either (a) we use the text-pair Flux notebook
with our exact overnight prompts (natural since we generated the corpus
from those prompts), or (b) we port the SDXL image-pair script to
Flux, which involves rewriting the diffusion-training inner loop.

Option (a) is the obvious first move. Our 990-render overnight corpus
then serves as *evaluation* data (per-axis per-base MediaPipe /
SigLIP-2 scores) rather than training data, which is actually the
cleaner split: train from prompts, evaluate from renders.

## Decision (2026-04-24): build our own Flux image-pair slider trainer

Grid of the four methods vs "can it consume our
`(base_prompt, α=0.png, α=1.png)` corpus on Flux":

| Method | Accepts image pairs on Flux? | Why not |
|---|---|---|
| Concept Sliders, Flux text mode | no | 3 prompts only, no images |
| Concept Sliders, SDXL visual mode | no (on Flux) | image-pair trainer exists but SDXL-only |
| Text Slider | no | LoRA on text encoder, attribute strings only |
| SliderSpace | no | generates its own random-seed images, unsupervised |
| SliderEdit | no (wrong architecture) | trains adapter on FLUX-**Kontext** (image editor), not base Flux generator |

None of the public methods natively consume our corpus on Flux. We
therefore develop our own trainer. Full cloned reference
implementations live in `docs/research/external/`:
- `train-flux-concept-sliders.ipynb` (Flux text mode scaffolding)
- `sliders-repo-SDXL train_lora-scale.py` at
  https://raw.githubusercontent.com/rohitgandikota/sliders/main/trainscripts/imagesliders/train_lora-scale.py
  (SDXL image-pair training inner loop to port)
- `sliderspace-repo/` (Flux direction-discovery reference)
- `text-slider-repo/` (text-encoder LoRA reference)
- `slideredit-repo/` (FLUX-Kontext adapter reference; wrong target)

Build plan — three swaps on top of the Flux text notebook scaffold:
1. **Data loader**: for each (axis, base, seed), read `α=0.png`
   (before) and `α=1.png` (after); VAE-encode to latents, cache per
   epoch. Optional: use `α ∈ {0.25, 0.5, 0.75}` as intensity supervision
   (pattern exists in SDXL image-pair script).
2. **Loss target**: replace
   `gt_pred = target_pred + eta·(pos_pred − neg_pred)` prompt-pair
   teacher with an image-pair flow-matching target — at noised
   `x_t = lerp(noise, latent_after, σ_t)`, supervise the LoRA student
   against velocity toward `latent_after − latent_before`. Same
   `FlowMatchEulerDiscreteScheduler`.
3. **VAE precompute**: batch-encode all 18 (base, seed) α=0 and α=1
   PNGs per axis once at startup to avoid repeated VAE passes.

Reuse from Flux text notebook: transformer/VAE/T5/CLIP loading,
`train_method='xattn'`, `rank=16`, `alpha=1`, `bs=1`, `lr=2e-3`,
`max_train_steps=1000`, bf16, `lr_warmup_steps=200`,
`lr_scheduler='constant'`, 512 × 512.

Cost estimate: ~1 h/axis on A100 (same envelope as text mode; only the
teacher targets change, the 4-forward student+teacher structure stays).

First test axis: `eye_squint` (strongest signal, least base-dependence
from overnight screening) before scaling to all 8 keepers +
2 v2-reprompted axes.

## Open Questions
- Text Slider claims zero-shot Flux transfer but publishes only SD-XL
  numbers; edit-quality on Flux is unknown. Single-source.
- Whether α-intermediate images (0.25, 0.5, 0.75) in our corpus actually
  improve slider calibration vs using only (α=0, α=1) endpoint pairs
  is not addressed in any source. Empirical question for the first
  training run.
- No source addresses whether Concept-Sliders Flux training remains
  viable on the newer Flux Krea or Flux 2 checkpoints (the Civitai
  results at [7] list Flux.2 Klein-era sliders, suggesting community
  extension but not official support).

## Sources

[1] Gandikota, R. et al. "Concept Sliders — official repo with flux-sliders subdir." https://github.com/rohitgandikota/sliders (Retrieved 2026-04-24)
[2] Gandikota, R. et al. "Concept Sliders: LoRA Adaptors for Precise Control in Diffusion Models." ECCV 2024. https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05660.pdf
[3] Jung, H. et al. "Text Slider: Efficient and Plug-and-Play Continuous Concept Control for Image/Video Synthesis via LoRA Adapters." arXiv 2509.18831. https://arxiv.org/html/2509.18831v1
[4] "sliders/README.md" — training data format. https://github.com/rohitgandikota/sliders/blob/main/README.md
[5] Gandikota, R. et al. "SliderSpace: Decomposing the Visual Capabilities of Diffusion Models." arXiv 2502.01639. https://ar5iv.labs.arxiv.org/html/2502.01639
[6] Zaier, A. et al. "SliderEdit: Continuous Image Editing with Fine-Grained Instruction Control." CVPR 2026 oral. https://armanzarei.github.io/SliderEdit/
[7] "Flux Concept Sliders — enabling finer control." https://flux1.cc/posts/flux-concept-sliders
[8] "Flux Training Tips & Tricks 2025 (Apatero)." https://apatero.com/blog/flux-training-tips-tricks-complete-guide-2025
[9] "SimpleTuner FLUX quickstart." https://github.com/bghira/SimpleTuner/blob/main/documentation/quickstart/FLUX.md
[10] "Fine-tuning a FLUX.1-dev style LoRA (Modal)." https://modal.com/blog/fine-tuning-flux-style-lora
[11] "How to Train a FLUX.1 LoRA for $1." https://medium.com/@geronimo7/how-to-train-a-flux1-lora-for-1-dfd1800afce5
