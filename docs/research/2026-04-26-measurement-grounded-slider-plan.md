---
status: live
topic: demographic-pc-pipeline
---

# Measurement-grounded slider training — plan

**Date:** 2026-04-26

## Why

Path A (prompt-pair Concept Sliders, Flux Krea, ai-toolkit) is failing the way today's experiments document: v0 drift, v1 immobile, v2 found-then-lost glasses, v3 EMA-buried, v4 extreme-amplification probe in flight. The core diagnosis (`2026-04-26-slider-experiments-journal.md`):

- Flux's guidance distillation makes the prompt-pair signal `(positive_pred − negative_pred)` small in magnitude at OOD `guidance_embedding_scale=1.0`; even at in-distribution 3.5 the signal isn't direction-pure.
- We are using a method designed for arbitrary text concepts on a problem where every axis we care about (glasses, smile, eye-squint, age, gender, race, identity) already has a cheap, well-validated quantitative measurement function in `models/blendshape_nmf/sample_index.parquet`.

The structural fix is to stop pretending we don't know what "glasses" means and let the measurement function supervise the LoRA directly. This doc lays out three paths in math-level detail, ranks them, and identifies the gotchas.

## What measurements we already have

Per row in `models/blendshape_nmf/sample_index.parquet` (7652 rows as of 2026-04-26):

| Quantity | Column | Source | Range |
|---|---|---|---|
| Glasses | `siglip_glasses_margin` | SigLIP-2 zero-shot probe | typically ±0.05, range ~[-0.02, +0.10] |
| Facial expression (52 axes) | blendshape vector | MediaPipe FaceLandmarker | each ∈ [0, 1] |
| Identity | ArcFace IR101 cosine to anchor | InsightFace | [-1, 1], anchor-relative |
| Age, gender, race | classifier logits | InsightFace + custom | softmax-style |
| Latent | cached VAE encoding | Flux VAE | 16 × 64 × 64 fp16 |
| Attention cache | cached `attn_base`, `attn_a`, `attn_b` | FluxSpace measurement node | per-block (L, D) tensors |

Pair counts under various filter regimes:

- **High-margin glasses pairs at matched composition** (same demographic, same seed, varying α): roughly 40-100 pairs from the existing corpus, 7 currently filtered tightly in `datasets/ai_toolkit_glasses_v1/`.
- **Smile pairs from blendshape**: hundreds; smile coefficient varies naturally across the corpus.
- **Identity-controlled pairs**: same seed × demographic, varying classifier output, gives us nearly arbitrary axis pairs by ranking on the relevant column.

We are sitting on the data. The trainer just isn't using it.

## Three paths

### Path D — direct latent-direction extraction (no LoRA, validation only)

**Goal.** Confirm in <1 day that the glasses direction is recoverable from existing measurement-grounded data, before committing to a LoRA training run. If this fails, paths B and C also fail and the issue is elsewhere (data quality, attention measurement basis).

**Math.**

For a single Flux double_block `b` and timestep range `[t1, t2]`, we have for each render `i`:
- `a_i ∈ R^{L × D}` — cached attention output (L tokens, D dim) at block `b`, averaged over `[t1, t2]`
- `m_i ∈ R` — measured SigLIP glasses margin

Find the direction `d ∈ R^{L × D}` that maximally correlates with `m`:
```
d* = argmin_d  Σ_i (m_i − ⟨a_i − ā, d⟩)² + λ ‖d‖²
```
with `ā = mean(a_i)`. Closed-form via ridge:
```
d* = (A^T A + λI)^{-1} A^T m         where A_i = vec(a_i − ā)
```

At inference, inject:
```
attn'_b = attn_b + α · d*           for t ∈ [t1, t2], α ∈ R user-controlled
```

This is the FluxSpace mechanism but with measurement-grounded direction instead of prompt-pair difference.

**What this tests.**

1. *Is the direction recoverable from our corpus?* If `d*` exists with reasonable norm and significant correlation on holdout, yes. If not (low R² on holdout, near-zero direction), the corpus doesn't contain enough signal — this falsifies B and C too.
2. *Does direct injection produce visible glasses?* Run the FluxSpace edit with `d*` at varying α. If yes, the direction works as an inference-time edit. If yes but only at certain (block, t-range), that tells us where to put the LoRA.
3. *What's the effective scale?* Map α → SigLIP score on rendered output. Establishes the magnitude scale we'd target in B / C.

**Cost.** ~1 day. Existing corpus, existing FluxSpace node, scikit-learn ridge.

**Failure mode.** If `d*` exists but injection produces nothing visible (similar to the falsified cached-δ replay from 2026-04-23), the issue is the static-direction assumption — Flux editing requires live forward passes, not constant additive shifts. In that case Path B (which trains a LoRA that *learns* to compute the right shift per-input) is the correct response.

---

### Path B — image-pair Concept Sliders (the smallest move from current trainer)

**Goal.** Replace prompt-pair supervision `(positive_prompt, negative_prompt)` with image-pair supervision `(I_pos, I_neg)` where the pairs are filtered by measured score. Same Concept Sliders algebra; different data source.

**Math.**

Build pairs `(I_pos, I_neg)` filtered by:
- `siglip_glasses_margin(I_pos) > 0.025`
- `siglip_glasses_margin(I_neg) < 0.005`
- Same demographic, same seed (so VAE-encoded latents differ primarily by glasses presence)

VAE-encode: `z_pos = VAE.encode(I_pos)`, `z_neg = VAE.encode(I_neg)`.

Per training step:
1. Sample timestep `t ~ U(0,1)`
2. Sample noise `ε ~ N(0, I)` — **same noise applied to both halves**
3. Build noisy latents:
   ```
   x_t,pos = (1−t) z_pos + t ε
   x_t,neg = (1−t) z_neg + t ε
   ```
4. Velocity targets (Flux flow-matching convention):
   ```
   v*_pos = ε − z_pos
   v*_neg = ε − z_neg
   ```
5. LoRA forward at multiplier `+w`: predict `v̂_pos = predict_noise(x_t,pos, c, t)`. Loss: `‖v̂_pos − v*_pos‖²`.
6. LoRA forward at multiplier `−w`: predict `v̂_neg = predict_noise(x_t,neg, c, t)`. Loss: `‖v̂_neg − v*_neg‖²`.
7. Average and backward.

**What the LoRA learns.**

Treat the model as `v(x_t, c, t; LoRA, mult) = v_base(x_t, c, t) + mult · δv(x_t, c, t)` where `δv` depends on LoRA weights.

At convergence:
```
v_base + w · δv ≈ ε − z_pos          (training endpoint at +w)
v_base − w · δv ≈ ε − z_neg          (training endpoint at −w)
```

Adding the two equations:
```
2 v_base ≈ 2ε − (z_pos + z_neg)
v_base   ≈ ε − (z_pos + z_neg)/2
```
The base model is implicitly being asked to match the average — it's not retrained, this just means "the pair midpoint should be a plausible base prediction." Reasonable since both halves are real samples.

Subtracting the two:
```
2w · δv ≈ (ε − z_pos) − (ε − z_neg) = z_neg − z_pos
δv     ≈ (z_neg − z_pos) / (2w)
```

So **the LoRA learns to produce a velocity perturbation proportional to the latent-space difference between the without-glasses and with-glasses latents**, scaled by `1/(2w)`.

This is direction-pure by construction. The supervision signal magnitude is `‖z_neg − z_pos‖` — a real measured pixel-level difference, not a small text-prompt-difference inside a guidance-distilled network. **Bypasses Path A's signal-magnitude problem entirely.**

At inference, LoRA at multiplier `α`:
```
v_inf = v_base(x_t, c, t) + α · δv(x_t, c, t)
      ≈ v_base + (α / 2w) · (z_neg − z_pos)
```
Negative α → push toward `z_pos` (more glasses); positive α → push toward `z_neg` (less glasses). Sign convention can be flipped at training time by swapping pair assignment.

**Math gotchas.**

1. **Same-noise constraint is critical.** If pos and neg use different noise, the difference `(v*_pos − v*_neg) = (ε_pos − ε_neg) − (z_pos − z_neg)` includes a noise-difference term that is sample-mean-zero but per-step large. Optimizer sees noisy target and learns nothing direction-specific. Same noise must be enforced.
2. **`δv` is not actually constant.** It's the LoRA's output at `x_t`, which depends on `x_t`. The training data at varying `x_t,pos / x_t,neg` (different t, different ε per step) lets the LoRA learn the *function* `δv(x_t, t)` whose value approximates `(z_neg − z_pos) / (2w)` at the specific training latents but generalizes to unseen `x_t`. **Generalization requires diverse pairs.** Seven pairs is too few.
3. **Pair filter must control composition.** If `I_pos` and `I_neg` differ in lighting, hair, demographic, *and* glasses, then `z_pos − z_neg` encodes all of those — not glasses alone. The LoRA learns the entangled direction. Filter strictly: same seed, same demographic, ideally same α-anchor.
4. **Bidirectional polarity is symmetric.** The trainer runs both `+w` and `−w` per step. With pairs swapped, both polarities have well-defined targets. No degeneracy from one-sided training.

**Data plan.**

Existing corpus: `datasets/ai_toolkit_glasses_v1/{train,reg}/` has 7 strict-filter pairs. Need ~50-200 for generalization.

Steps:
1. Re-filter `models/blendshape_nmf/sample_index.parquet` for the broader pair set:
   - Group by `(base_demographic, seed)`
   - Within each group, find rows with `siglip_glasses_margin > 0.025` and rows with `< 0.005`
   - Build pairs (one positive per group, one negative per group); take cartesian product if multiple of each
2. Render any missing α=0 anchors if the group is missing one
3. Build side-by-side images: `[neg | pos]` horizontally concatenated, 1024×512 (so each half is 512×512)
4. Save with caption file = target_class
5. Output to `datasets/ai_toolkit_glasses_v2_pairs/`

**Trainer plan.**

`extensions_built_in/image_reference_slider_trainer/ImageReferenceSliderTrainerProcess.py:hook_train_loop` is the SDXL/SD trainer. Three changes for Flux flow-matching (~30-50 lines):

1. Replace `noise_scheduler.add_noise(latents, noise, timesteps)` with linear interpolation:
   ```python
   x_t = (1 - t).view(-1,1,1,1) * latents + t.view(-1,1,1,1) * noise
   ```
2. Replace `target = noise` with `target = noise - latents` (velocity).
3. Replace timestep sampling. Currently `torch.randint(0, max_denoising_steps, ...)`. Should use Flux's logit-normal sampling with sigma shift, OR uniform `t ~ U(0,1)`. Match the canonical Flux Concept Sliders notebook (uniform).
4. Plumb `guidance_embedding_scale=3.5` through (we already patched ConceptSliderTrainer for this; same fix needed here).

This is a self-contained port. Branch `vamp/flux-image-pair-slider` off the existing `vamp/flux-slider-guidance-3p5`.

**Pass criteria.**

- ±1.5 sweep on 4 demographics, all show clear glasses gradient
- Identity preserved across sweep (less drift than v2 because supervision is image-grounded, less reason for LoRA to perturb identity)
- m=0 baseline matches Krea baseline (no drift; image-pair training shouldn't bias the "off" state)

---

### Path C — latent-space measurement surrogate (the cleanest mathematical statement)

**Goal.** Skip the image-pair indirection entirely. Train a tiny scoring network `φ` that maps latents to measured scores; use `φ`'s gradient as the supervision signal directly.

**Math.**

Train `φ_glasses: R^{16×64×64} → R` to predict `siglip_glasses_margin` from VAE latents:
```
φ* = argmin_φ  Σ_i (φ(z_i) − m_i)²       (i over 7600+ corpus rows)
```
Architecture: small CNN (4-5 conv layers + GAP + MLP, ~1-5M params). Validate on held-out split; require r ≥ 0.85 to use.

Slider trainer per step (replaces velocity-MSE, or runs alongside it):
1. Sample noisy latent `x_t` and noise `ε`, with `t ~ U(0,1)` (downweight high-t per ID-Booth).
2. LoRA forward at multiplier `+w`: predict `v̂_+ = predict_noise(x_t, c, t)`.
3. Reconstruct one-step `x̂_0,+`:
   - From `v = ε − x_0` and `x_t = (1−t) x_0 + t ε`:
   - `ε = v + x_0`
   - `x_t = (1−t) x_0 + t(v + x_0) = x_0 + t v`
   - **`x̂_0,+ = x_t − t · v̂_+`** ← clean closed form, no `ᾱ` baggage from DDPM
4. Compute `φ(x̂_0,+)` — predicted glasses score of the LoRA's reconstruction
5. Target score:
   - `m_target,+ = m_baseline + Δm` where `m_baseline = mean φ on neutral renders` and `Δm = +η_φ` is a chosen step (e.g. 0.05, halfway between corpus mean and 95th percentile)
6. Loss on +1 polarity: `L_φ,+ = λ_t · (φ(x̂_0,+) − m_target,+)²`
7. Same at multiplier `−w`, `Δm = −η_φ`. Loss `L_φ,−` analogously.
8. Optional regularizer: keep velocity-MSE on a "neutral" forward (no LoRA active or LoRA at 0) to prevent base-distribution drift. `L_diff = ‖predict_noise(x_t, c, t; LoRA_off) − v_baseline‖²` or just use existing dataset latents.
9. Total: `L = L_φ,+ + L_φ,− + λ_diff · L_diff`

**Math gotchas.**

1. **High-t x̂_0 is noisy.** Per the 04-24 doc and ID-Booth, at `t → 1` the model's velocity prediction is essentially noise prediction and `x̂_0 = x_t − t · v̂` recovers a high-variance estimate. `φ(x̂_0)` returns something close to `φ(noise)`, which is uninformative. **Mitigation:** weight the φ-loss by `λ_t = 1 − t²` (smooth from 1 at t=0 to 0 at t=1).
2. **φ trained on clean latents may not generalize to noisy x̂_0.** φ sees clean `z_i = VAE.encode(I_i)` during training; at slider-train time it sees noisy reconstructions. **Mitigation:** train φ with noise augmentation — sample `t' ~ U(0, 0.5)`, build `z_t' = (1−t') z + t' ε`, train `φ(z_t') → m` (with the score still being the original clean-latent score). Makes φ noise-robust over the timestep range we care about.
3. **Reward hacking.** φ-only loss could be satisfied by a degenerate LoRA output that fools φ but doesn't visually have glasses. Classic DRaFT/ReFL pathology. **Mitigations:**
   - `L_diff` regularizer keeps the model close to base (penalizes drift on prompts other than the slider concept)
   - φ trained adversarially: also include latents-with-noise-perturbation as anti-hack training data
   - Periodic visual validation: render samples every N steps, compute SigLIP on the actual decoded image, compare to φ prediction
   - If `|φ(x̂_0) − siglip(decode(x̂_0))| > threshold` consistently, retrain φ
4. **No VAE backprop.** φ operates on latents directly; we never decode to RGB during training. This is the key efficiency vs PuLID/ID-Booth's image-level losses. The 04-24 doc's six-piece machinery doesn't apply.
5. **Gradient through one-step x̂_0.** `dL_φ/dv̂ = −t · dφ/dx̂_0(x̂_0)`. The `−t` factor naturally upweights high-t (because `−t` is large in magnitude there) — but combined with the `λ_t = 1 − t²` weighting that downweights, the net effect is a smooth weighting peaked around mid-timesteps. Worth empirical validation; could also just sample `t ~ U(0, 0.5)` and skip the weighting.
6. **Choice of `η_φ` (target Δm).** Too small → slider doesn't move much; too large → demands an output outside the achievable range of the model. Calibrate from corpus distribution: target is some percentile (e.g. 90th) of measured scores. For glasses: 90th percentile of `siglip_glasses_margin` is roughly +0.07 (would need to confirm); use that as `m_target,+`.

**Why the math is sounder than Path B.**

Path B's supervision is `(z_neg − z_pos) / (2w)` — a fixed direction in latent space derived from one specific pair. The LoRA learns this direction *averaged over the training pair distribution*, which assumes the pair distribution is representative of "glasses" in general.

Path C's supervision is `−t · ∇_x̂_0 φ(x̂_0)` — a context-dependent direction at every (x_t, t) that points toward "more glasses by SigLIP." The gradient is computed *per input*, so the LoRA learns a function of the input rather than a constant direction. More expressive; harder to overfit to specific compositions.

In the limit of perfect φ and perfect optimization, Path C is the theoretically correct slider — it's a learned approximation of "the velocity perturbation that maximally increases the SigLIP score." Path B is a dataset-driven approximation of the same thing.

**Engineering cost.**

Higher than Path B. Roughly:
- φ training: 1-2 days (small CNN, big corpus, well-defined task)
- Trainer integration: 2-3 days (new loss term, x̂_0 reconstruction, weighting, anti-hack diagnostics)
- Validation: 1 day (visual + SigLIP-on-decoded comparison)

Total ~5-7 days vs Path B's ~2-3 days.

---

## Decision matrix

| Path D result | Path B result | Path C result | Inferred truth |
|---|---|---|---|
| `d*` exists, injection works | (skip, validate) | (skip, validate) | Direction is in the corpus and reachable. Move to B for production LoRA. |
| `d*` exists, injection fails | works | (skip) | Static direction insufficient; LoRA's per-input adaptation is necessary. B handles it. |
| `d*` weak / corpus-limited | fails | (try) | Pair-level supervision insufficient. φ-supervised learns the function from richer corpus. |
| `d*` weak | fails | fails | Corpus genuinely lacks the signal. Need to render larger / cleaner glasses corpus first. |
| `d*` strong | works | works | All paths viable; pick by deployment cost (B is cheapest LoRA). |

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Path B underfits with current 7-pair corpus | Re-filter sample_index for 50-200 pairs before launch (data prep step is cheap) |
| Path B port to Flux flow-matching has bugs | Diff against existing ConceptSliderTrainer's flow-matching plumbing; both go through the same `predict_noise` API |
| Path C φ overfits or doesn't generalize | Hold out 20% of corpus for φ validation; require r ≥ 0.85 before using |
| Path C reward hacking | `L_diff` regularizer + periodic visual cross-check |
| Single-pair supervision (B) entangles non-glasses changes | Strict filter on (demographic, seed, α); allow only glasses to vary |
| Same-noise constraint in B silently violated by the trainer | Read the port carefully; assert in code that pos/neg get same noise tensor |
| Either path bypasses the guidance-distillation problem but introduces a new one | Run cheap inference sweep at multiple LoRA multipliers; check linearity of `φ(decode(x̂_0))` vs multiplier — should be roughly monotone |

## Recommended order

1. **Path D first** (1 day, no trainer changes). If glasses direction is recoverable + injectable, paths B and C have signal to learn from. If not, address corpus quality before training.
2. **Path B second** (2-3 days). Most aligned with current ai-toolkit infrastructure; minimal new code; validates the algebra holds with measurement-grounded data.
3. **Path C third** (5-7 days). Cleaner mathematically, more general (one φ per axis works for any LoRA architecture), but heavier engineering. Worth doing once we've validated the data with B.

## Cross-axis generalization

The above is written for glasses. The same plan applies axis-by-axis:

| Axis | Measurement column | Pair filter |
|---|---|---|
| Glasses | `siglip_glasses_margin` | high vs low margin |
| Smile | blendshape `mouthSmileLeft + mouthSmileRight` | high vs zero smile coefficient |
| Eye squint | blendshape `eyeSquintLeft/Right` | high vs zero squint |
| Jaw open | blendshape `jawOpen` | high vs zero |
| Age | classifier age logit | high age vs low age, identity-controlled |
| Gender | classifier gender logit | controlled |
| Race | classifier race logit | controlled |

For each axis: same pipeline (B or C), different filter and target. φ is per-axis (one small CNN per slider). All axes share the same trainer code.

The blendshape axes are the easiest test cases because:
- The measurement is geometric and continuous (not a noisy classifier)
- Pairs at controlled coefficient values are abundant in the existing corpus
- The expected slider effect is well-defined (smile coefficient should monotonically increase with LoRA multiplier)

Once Path B works for glasses, the blendshape axes should fall out trivially. They're better-conditioned problems with the same machinery.
