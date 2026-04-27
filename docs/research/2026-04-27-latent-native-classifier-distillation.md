---
status: live
topic: demographic-pc-pipeline
---

# Prior art: latent-native classifiers for diffusion training

**Date:** 2026-04-27
**Sources:** 15 papers / artifacts. Load-bearing: DOODL [1], DRaFT [2], PuLID [3],
LPL [4], REPA [5], Diffusion Classifier [6], Concept Sliders [7], precomputed
latent datasets [8][9].

## Executive summary

The exact recipe we're considering — distill pixel-space ArcFace / SigLIP into
twins that consume Flux VAE latents directly, then use them as the training
loss for a slider LoRA without VAE-decoding every step — is **novel
territory**. No paper we found does precisely this. But every adjacent
ingredient is well-established: training-time identity-preservation losses
[3], full-pipeline differentiable reward fine-tuning [2], latent-path
perceptual losses [4], and projection-head alignment of noisy intermediate
representations to a frozen vision encoder [5]. The reason nobody has
combined them this way is plausibly path-dependence: PuLID and DRaFT-style
methods accept the VAE-decode cost per step, and that has been good enough
on per-image budgets. For our case — many slider-axis training runs, many
metric heads — the latent twin pays for itself within a single training
campaign. Engineering risk is low; novelty is real but moderate.

We **do not need to render new images** for distillation: FFHQ (70k portraits)
and CelebA-HQ are already available [8][9], and we only need to encode them
through Flux's VAE once.

## What's been done

### Identity / reward losses through full decode (PuLID, DRaFT)

PuLID's ArcFace identity loss is computed on **decoded pixels**, not
latents. The architecture spawns a "Lightning T2I" branch that runs full
4-step SDXL-Lightning sampling from pure noise to a complete image, then
applies ArcFace to the decoded RGB output and computes
`1 − cos(φ(C_id), φ(L-T2I(x_T, C_id, C_txt)))` [3]. **This requires VAE
decode at every training step**, which is exactly the cost we're trying to
avoid. PuLID accepts it because Lightning-T2I makes the decode cheap (4
steps).

DRaFT (Direct Reward Fine-Tuning) goes further: it backpropagates through
the **entire sampling chain plus VAE decode** to align the diffusion model
to any differentiable reward — HPSv2, PickScore, LAION-Aesthetic, OWL-ViT,
even adversarial ResNet-50 classification [2]. Same architectural pattern:
classifier on pixels, gradient flows back through decode + multi-step
sampling. This is the dominant paradigm for "use a pretrained
classifier/reward as a training signal" right now.

Both of these confirm the *information* we want to inject (ArcFace identity,
classifier reward) is the right kind of supervision — they just pay the
decode cost we're trying to skip.

### Inference-time classifier guidance with full decode (DOODL)

DOODL ("Direct Optimization of Diffusion Latents") works at **inference
time, not training time**. It optimizes the initial latent x_T against a
pretrained pixel-space classifier's gradient, using an EDICT-based
invertible diffusion process to backprop through the full sampling chain
with constant memory [1]. DOODL also requires VAE decode (the classifier
operates on `the true generated pixels`), and the classifier is frozen.

DOODL is the closest thing in the literature to "use frozen classifier
gradients on latent-space artifacts," but it's a sampling-time method, not
a training objective for fine-tuning. Adapting DOODL's invertible-backprop
trick to slider training is plausible but heavy.

### Latent-path perceptual loss without external classifier (LPL)

The closest published work to "perceptual signal in the latent path" is
"Boosting Latent Diffusion with Perceptual Objectives" (LPL) [4]. It's
notable for what it **doesn't** do: it does not distill an external
classifier (LPIPS / CLIP / ArcFace). Instead, it routes both clean and
predicted latents through the **VAE decoder** and computes feature-matching
loss on the *decoder's* intermediate hidden states.

LPL still incurs decode cost (it pushes through the decoder's first L
blocks), but only the decoder forward, not the full RGB-out + external
classifier pipeline. Reported gains: 6%–20% FID improvement on ImageNet-1k,
CC12M, S320M. **This validates the principle that the latent path can carry
meaningful semantic-perceptual loss.** It's evidence the gradient signal we
want is reachable; LPL just chose not to distill an external twin.

### Aligning noisy latents to a frozen vision encoder (REPA)

REPA (ICLR 2025 Oral) is the most directly relevant precedent [5]. It
trains a small projection head from the noisy DiT internal hidden state
onto frozen DINOv2 features computed from the *clean* image, with cosine
similarity loss. REPA reports >17.5× faster DiT convergence.

Direction matters: REPA goes "noisy DiT representation → DINOv2 (pixel)
embedding." We want the inverse and stronger version: "Flux noisy latent →
ArcFace embedding," via a small distilled ResNet, with the ArcFace teacher
operating on the decoded image during distillation only. REPA is therefore
half the proposal: it confirms a small head can map noisy intermediate
representations into a frozen vision-encoder space, and the resulting loss
is a useful training signal. It does **not** prove the inverse direction
(latent → external embedding via a standalone student net), but the
gradient information is already known to live in the noisy latent.

REPA's HASTE follow-up [5] adds another finding: alignment loss helps early
training and should be turned off later. For a slider LoRA the analogous
question — should the metric loss decay over training steps — is
unanswered in the literature for our setting.

### Diffusion Classifier and noise-aware classification

The "Diffusion Classifier" line [6] uses the diffusion model itself as a
zero-shot classifier by ranking conditioning hypotheses against denoising
loss. "Your Diffusion Model is Secretly a Noise Classifier" [10] adds a
contrastive denoising objective. Both confirm that noisy-latent
representations carry classifier-relevant information, but neither trains
a separate classifier head against pretrained external semantics. Adjacent
support, not direct precedent.

### Concept Sliders' use of ArcFace

In the original Concept Sliders paper [7], ArcFace appears as an
**evaluation** metric for ID consistency (the headline result table reports
ArcFace cosine pre/post slider edit). It is not part of the training
objective. Avatar Concept Slider [11] introduces a PCA-based
attribute-preservation loss but again on decoded faces, not latents.

So the slider-LoRA literature has consistently **measured** identity with
ArcFace and **trained** with latent MSE, never closed the loop.

## Comparison table

| Method | When | Classifier ops on | Trains classifier? | Decode in train loop? |
|---|---|---|---|---|
| Concept Sliders [7] | training | n/a (latent MSE) | n/a | no |
| PuLID [3] | training | decoded RGB | no (frozen ArcFace) | **yes** (every step) |
| DRaFT [2] | training | decoded RGB | no (frozen reward) | **yes** (every step + sample chain) |
| DOODL [1] | inference | decoded RGB | no (frozen classifier) | yes (invertible backprop) |
| LPL [4] | training | decoder hidden states | no | partial (decoder forward only) |
| REPA [5] | training | clean RGB → DINOv2 | no (frozen DINOv2) | partial (DINOv2 on clean image) |
| **Proposed** | training | **noisy latent directly** | **yes (one-time distill)** | **no** |

The empty cell — train-time, latent-native, distilled student of a frozen
external classifier, used as fine-tuning loss — is what we'd be filling.

## Do we need to render the corpus?

No.

The distillation step needs (image, Flux-VAE-latent) pairs — that's it. We
have two options ordered by ease:

1. **Reuse FFHQ / CelebA-HQ raw images and encode through Flux VAE.** FFHQ
   is 70k 1024² portraits with broad demographic variation [12]. Encoding
   through Flux's 16-channel VAE is a one-pass `VAE.encode()` over the
   dataset — a few GPU-hours on one card. The image data is free; the
   compute is small.
2. **SDXL precomputed latent datasets exist** (`Birchlabs/sdxl-latents-ffhq`
   [8], `korexyz/celeba-hq-256x256-sdxl-vae-latents` [9]) but they're SDXL
   VAE, not Flux. Flux uses a different 16-channel VAE, so these are
   wrong-shape and not directly usable. Mentioned only to confirm the
   community pattern of caching latents at scale is normal.

For the **slider training itself** (Flavor 2 in our recent design): a
single anchor face plus a multiplier `m ∈ [-1.5, +1.5]` is the entire
input. No paired corpus needed. The whole demographic-pair rendering
pipeline becomes obsolete if metric-space loss replaces velocity-MSE loss.
This is a substantial win unrelated to whether the distillation works.

For Flavor 1 (keep pairs, swap loss): existing v2 pairs reused as-is. No
re-render.

So: zero new generation work. FFHQ → Flux-VAE-encode → distill heads →
train sliders on metric loss. The only synthesis we'd already need to keep
doing is whatever sample-prompts we run to *evaluate* sliders (already
cheap).

## Risks and unknowns

**Distillation quality is the floor.** None of the cited works publish
distillation accuracy of an ArcFace-equivalent latent twin, because nobody
has done it. This is a real unknown — could be 0.95 cosine R² (great) or
0.6 (problematic). The cheapest sanity check is to distill ArcFace to a
*pixel*-input ResNet-18 student first; if that hits >0.9 we know the task
is well-posed. Then redo with a Flux-latent input student.

**Noisy x_t vs clean z_0.** REPA [5] works because it uses noisy DiT
intermediate features — but it aligns to DINOv2 on the *clean* image. For
our setting, the cleanest version is to distill on **clean** Flux latents
`z = VAE.encode(x)` and use the student only at low-noise timesteps
(`t < 0.3`) during slider training, falling back to the existing
velocity-MSE at high t. The harder, less-explored variant — distill a
twin that takes `(x_t, t)` and works at all noise levels — exists in the
"Diffusion Classifier" lineage but adds training cost.

**Inherited failure modes.** The latent ArcFace twin will have ArcFace's
same demographic-bias failures (poorer on children, dark skin, occluded
faces) — REPA, PuLID, DRaFT all share this. Not a new risk introduced by
the latent twin specifically.

**Compute overhead per slider step.** Forward through 3–4 small distilled
heads is cheap (~a few ms each), much smaller than VAE decode. Net win
even if we add 4 metrics.

## What this implies for the roadmap

- The recipe is novel enough to be a small contribution if published, low
  enough engineering risk to build now. We're not racing a well-trodden
  benchmark; we're filling an unoccupied cell.
- LPL [4] and REPA [5] together are the strongest evidence the gradient
  signal we want is reachable on the latent path.
- PuLID [3] and DRaFT [2] are the workable fallback if distillation
  quality fails: pay the decode cost, eat the slowdown, get the metric
  loss anyway.
- A single ArcFace latent twin unlocks **identity, age, gender, race,
  glasses** simultaneously (all of those are linear ridges on ArcFace
  embedding space we already have), so the distillation ROI is high before
  we even consider SigLIP.

## Suggested next experiments

1. **Render-free distillation corpus prep.** FFHQ raw images, encode all
   70k through Flux VAE, persist as latent shards. ~6 GPU-hours, one-time.
2. **ArcFace-pixel sanity baseline.** Train a small ResNet-18 student on
   FFHQ to predict ArcFace embeddings. If cosine R² < 0.9, the task itself
   is hard and we should diagnose before adding the latent-input twist.
3. **`arc_latent` distillation.** Same ResNet-shaped student, 16-channel
   latent input. Target: cosine R² > 0.9 vs teacher on held-out. This is
   the gate.
4. **v9 trainer (Flavor 1):** existing pair cache, replace latent-MSE with
   metric loss using the new heads at low t, retain MSE at high t. Compare
   against v7's spatial-mask result.
5. **v10 trainer (Flavor 2):** anchor-only, no pairs, full metric-space
   loss. The interesting science.

## Sources

[1] Wallace et al. "End-to-End Diffusion Latent Optimization Improves
Classifier Guidance" (DOODL). ICCV 2023. https://arxiv.org/abs/2303.13703

[2] Clark et al. "Directly Fine-Tuning Diffusion Models on Differentiable
Rewards" (DRaFT). https://arxiv.org/html/2309.17400v2

[3] Guo et al. "PuLID: Pure and Lightning ID Customization via Contrastive
Alignment". https://arxiv.org/html/2404.16022v2

[4] "Boosting Latent Diffusion with Perceptual Objectives" (LPL).
https://arxiv.org/html/2411.04873v1

[5] Yu et al. "Representation Alignment for Generation" (REPA). ICLR 2025
Oral. https://github.com/sihyun-yu/REPA — and follow-up "REPA Works Until
It Doesn't" https://arxiv.org/html/2505.16792v1

[6] Li et al. "Your Diffusion Model is Secretly a Zero-Shot Classifier"
(Diffusion Classifier).
https://diffusion-classifier.github.io/static/docs/DiffusionClassifier.pdf

[7] Gandikota et al. "Concept Sliders: LoRA Adaptors for Precise Control
in Diffusion Models". ECCV 2024.
https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05660.pdf

[8] `Birchlabs/sdxl-latents-ffhq`.
https://huggingface.co/datasets/Birchlabs/sdxl-latents-ffhq

[9] `korexyz/celeba-hq-256x256-sdxl-vae-latents`.
https://huggingface.co/datasets/korexyz/celeba-hq-256x256-sdxl-vae-latents

[10] "Your Diffusion Model is Secretly a Noise Classifier and Benefits
from Contrastive Training". NeurIPS 2024.
https://arxiv.org/abs/2407.08946

[11] "Avatar Concept Slider: Manipulate Concepts In Your Human Avatar
With Fine-grained Control". https://arxiv.org/html/2408.13995v1

[12] FFHQ dataset.
https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq
