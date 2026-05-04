---
status: live
topic: mediapipe_distill
---

# bs_v3_t critic is foolable by LoRA — diagnosis and paths forward

**Date:** 2026-04-30
**Context:** ai-toolkit ConceptSliderTrainer using `BlendshapeStudent (bs_a)` distilled from MediaPipe FaceLandmarker as a classifier-loss critic for Flux LoRA training (squint slider thread).

## TL;DR

Two-orthogonal-failures diagnosis.

1. **The bs critic generalizes to natural Flux renders.** Tested on `squint_slider_v0` step 1800 samples: eyeSquint reads rise 0.20→0.59 monotonically with slider m on all 3 demographics. The critic is not broken in the trivial sense.
2. **The bs critic is fooled by LoRA-induced latent perturbations.** A pure-classifier-loss LoRA (`bs_only_mode`, no slider polarity, no prompt-pair) can drive critic readings to target while producing renders that either look identical to baseline (subtle squint case) or change *the right blendshape* but with severe identity drift bundled in (jawOpen sanity case).

The squint failure was a special case — eyeSquint baseline overlaps the engage target (0.5) on 2/3 demographics, and eyeSquint is genuinely correlated with eyeBlink in the FFHQ training distribution. With a clean target (jawOpen, baseline ≈ 0.01, target = 0.9), the LoRA visibly produces the correct blendshape but identity collapses simultaneously.

## Evidence — sanity readings on rendered samples (`scripts/sanity_bs_critic.py`)

Each cell is `bs_v3_t/final.pt` applied at t=0 to a Flux VAE encoding of the rendered JPG.

```
=== v0 step 0 baseline (no LoRA) ===
demographic           eyeSquint(L+R)/2   eyeBlink(L+R)/2   jawOpen
east_asian_man        0.4487             0.1632            0.0061
black_woman           0.1656             0.0698            0.0074
european_man          0.5681             0.1236            0.0154

=== v0 squint_slider step 1800 — m=+1.5 (visibly squinting) ===
east_asian_man        0.5938  (+0.39)    0.3131  (+0.26)   ~0
black_woman           0.5009  (+0.45)    0.3012  (+0.28)   ~0
european_man          0.4475  (+0.28)    0.2469  (+0.22)   ~0
↑ critic correctly tracks visible squint, BUT eyeBlink rises in lockstep

=== v1h_bs_only step 200, multiplier=1.0 (NO visible squint) ===
east_asian_man        0.561              0.080             0.019
black_woman           0.585              0.074             0.013
european_man          0.579              0.088             0.024
↑ critic reads eyeSquint > 0.5 target without raising eyeBlink — but renders show no squint

=== v1j_jaw_sanity step 200, multiplier=1.0 (visibly OPEN MOUTHS, drifted identity) ===
east_asian_man        0.092              0.021             0.9551
black_woman           0.148              0.031             0.9562
european_man          0.229              0.049             0.9146
↑ critic AND visual agree on jawOpen; identity collapse independent of bs target
```

Visual reference (v1j): `2026-04-30-bs-loss-classifier-fooling-collage.png` — every 25 steps for 3 demographics. Mouth-open emerges by step 75; identity progressively collapses through step 200 (aged skin, dramatic lighting, weathered features).

## Why this happens

The critic was distilled on a fixed (clean Flux latent → blendshape) corpus (`compact_blendshapes`, ~33 k FFHQ + Flux corpus rows; warm-started from v2c, random-t noise augmentation). The training distribution doesn't include LoRA-perturbed latents.

In the latent space there are *off-manifold directions* where the critic confidently predicts whatever blendshape we want, while VAE decoding produces a face that has the right blendshape geometry plus arbitrary correlated drift (skin age, lighting, contrast). The LoRA, optimizing only the bs loss, finds these directions because they're locally cheaper than producing a clean-identity, correct-blendshape change.

This is the classic **classifier-fooling** / adversarial-direction failure mode. It's also why "raise the weight" / "more iterations" don't help — the optimizer keeps finding the easiest direction, and the easy direction is fooling.

## What we already tried

| Variant | Architecture | Outcome |
|---|---|---|
| v1c–v1g | slider + bs_loss | eyeSquint↔eyeBlink coupling; closure at high m |
| v1h | bs_only_mode | critic satisfied, render unchanged at training scale; identity collapse at amplified scale |
| v1i | v1h + SigLIP-distill anchor cosine (sg_b student, w=5000) | critic still fooled; sg cosine preserved numerically while pixel-level identity drifts; no squint |
| v1j (jaw sanity) | v1h with single jawOpen→0.9 term | mouths visibly open; identity drift bundled in regardless |

Pattern: every attempt produces some bs-target satisfaction with severe correlated identity drift, regardless of regularizer strength or specific blendshape.

## Two paths forward

### Path 1 — stabilize the LoRA with a stronger anchor (keep critic as-is)

The LoRA finds an off-manifold direction. Pin the LoRA more tightly to the on-manifold no-LoRA prediction and the cheap fooling direction becomes expensive.

- **a. Geometric latent mask** — given per-image eye/mouth masks (we already have these for the squint trainer at `output/squint_path_b/eye_masks`), forbid LoRA perturbation outside the region of interest:
  `L_geo = w · mean( (1 − M) ⊙ (z_lora_on − z_lora_off)² )`
  Surgical, cheap, ~30 min to wire. Limitation: assumes mask is correct; deformable expressions (open mouth → cheek shape changes) may need soft masks.

- **b. ArcFace-on-x0 identity anchor** — VAE-decode `x0_recon` for both LoRA-on and LoRA-off forward passes, run ArcFace on both, cosine-anchor. ArcFace is trained to be invariant to expression and pose, sensitive to identity — the inverse of what we need to penalize. We already have `arc_distill` infrastructure. Cost: VAE decode + ArcFace forward in training loop, ~3-5× slower per step. Most likely to actually solve it; this is what every identity-preserving face-LoRA paper uses.

- **c. Combine a + b** — belt and suspenders. Probably the production end-state.

The SigLIP-anchor we already tried (v1i) is a *softer* version of (b); it failed because sg_b distilled from the VAE bottleneck is not strict enough about pixel-level identity. ArcFace's inductive bias is much closer to "identity vs expression," exactly the axis we need.

### Path 2 — retrain bs_v3_t to be robust to LoRA-style perturbations

Fix the classifier-fooling at the root. The critic should be Lipschitz over small latent perturbations, so the LoRA can't find a direction that flips the reading without changing the underlying geometry.

- **a. Adversarial-perturbation training** — at each batch sample δ ~ small random direction in latent space, add a consistency term `|| bs(z + δ) − bs(z) ||²`. Forces local Lipschitz smoothness. Modify `src/mediapipe_distill/train_t.py`, retrain ~40 epochs (≈ 1 hour on RTX 4090). Known cost: 2-5% R² drop on clean inputs (irrelevant — we're at t_max=0.5 anyway, R²=0.89 is plenty).
- **b. PGD-style adversarial examples** — instead of random δ, use projected gradient descent to find worst-case δ within an ε-ball. Stronger but slower per epoch.
- **c. Ensemble distillation** — train K critics with different inits + augmentations, use min over readings as the loss target. Cheap to ensemble at training time.

Path 2 produces a reusable robust critic — every future LoRA against any subset of channels benefits, not just squint/jaw.

## Recommended sequence

1. **v1k = v1j + ArcFace-on-x0 anchor.** Tests Path 1b on the cleanest signal we have (jawOpen visibly engages, so identity drift is the only remaining failure mode). ~1 hour to wire + run. If renders show open mouths *with preserved identity*, we have a working recipe and can apply it to squint.
2. **In parallel: queue Path 2a (adversarial-δ retrain).** ~1 hour compute, modest engineering. Result is a robust critic that all future bs-loss runs can use.
3. **Geometric mask (Path 1a) is a fallback / additive option** if ArcFace alone proves insufficient or too expensive in production.

## Open questions

- Is the natural eyeBlink↔eyeSquint correlation in FFHQ a critic problem or a *data* problem? Both — the targets are correlated in the training distribution because FFHQ rarely contains "eye-narrow without partial-closure." A robust critic might still reflect this. The geometric-mask + raised target may be needed regardless.
- Does adversarial-δ training of bs_v3_t generalize across the full 52-d output, or just the channels we explicitly probe? Standard result says all channels benefit because the trunk is shared.
- For the classifier-fooling failure specifically: does ArcFace anchor compose multiplicatively with adversarial-trained critic (i.e., we get even better results combining both) or do they overlap? We'd need a 2×2 ablation (with/without each).
