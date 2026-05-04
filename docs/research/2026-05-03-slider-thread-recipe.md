---
status: live
topic: arc-distill
---

# Slider thread — combined recipe (parking note)

**Date:** 2026-05-03
**Purpose:** Park the LoRA-slider thread as *semi-successful*, in a state where a future session (or a future GPU budget) can resume without re-deriving the lessons. Combines: solvers, trained student/critic models, observed failure modes, and the working recipe we landed on.

## TL;DR

We have a working recipe — `bs_only_mode` LoRA + arc_distill identity anchor (v1k pattern) — that produces a visible target blendshape *with* identity preserved, on a clean channel like jawOpen. The thread is paused because (a) generalising the recipe from jawOpen to squint requires raising the engage target above the FFHQ baseline overlap and re-running, (b) the critic remains foolable on adversarial directions, and (c) inference must use a step-gated LoRA application, which we have designed but not implemented in the demo path. None of these are conceptual blockers — they are GPU-time and engineering-time blockers.

## Status by component

### Critic — `BlendshapeStudent (bs_a)` family

| ckpt | Path | Trained on | R²(t=0) | R²(t=0.5) | Status |
|---|---|---|---|---|---|
| v3_t | `models/mediapipe_distill/bs_v3_t/final.pt` | FFHQ + Flux corpus, random-t noise aug, 33k rows | jaw 0.92, squint 0.88 | jaw 0.89, squint 0.84 | **production critic** for current bs_loss |
| v4_pgd | `models/mediapipe_distill/bs_v4_pgd/final.pt` | v3_t + PGD adversarial inner loop, r2_median ckpt selection | TBD | TBD | **trained, not yet validated as drop-in** — the gate (G4) is "on v1h_bs_only fooling renders should NOT report eyeSquint > 0.30" |
| v2c, v2d, v2e | `models/mediapipe_distill/v2*/` | clean (no random-t) | high R²(0), bad R²(>0) | n/a | superseded — only safe for clean-latent diagnostics |

**What v3_t does well.** Generalises from FFHQ to natural Flux renders (sanity_bs_critic.py confirmed eyeSquint reads rise 0.20→0.59 monotonically with the v0 squint slider).

**What v3_t does badly.** Foolable by LoRA-induced latent perturbations. v1h (squint bs_only) drove eyeSquint critic to >0.5 with no visible squint; v1j (jawOpen bs_only) drove jawOpen to 0.95 with severe identity collapse. The critic was trained on a clean (Flux latent → blendshape) corpus and the LoRA finds adversarial latent directions outside that distribution.

**What v4_pgd is supposed to fix.** Adversarial-δ training (PGD inner loop in `train_t.py`) makes the critic locally Lipschitz over small latent perturbations, so the LoRA can't find a cheap fooling direction. **Validation is the next step on this thread** — run `scripts/sanity_bs_critic.py` (already updated to multi-checkpoint) on v1h, v1j, v1k renders and check the four gates in `docs/superpowers/plans/2026-04-30-bs-v4-pgd-robust-critic.md` (G1: clean R² preserved within 5%; G2: degrades gracefully under δ; G3: rejects v1h fooling; G4: still fires correctly on v1j real signal).

### Identity anchor — `arc_distill` (`AdapterStudent`, latent_a2_full_native_shallow)

- `models/arc_distill/checkpoint.pt` — frozen-IResNet50 ArcFace distilled to take Flux latents directly (no decode in training loop). 43.6 M params.
- Already wired into `ConceptSliderTrainer` as `id_loss_*`.
- v1k confirmed: at `id_loss_weight=5000, id_loss_t_max=0.5` the LoRA produces visible jawOpen with identity preserved across all 3 demographics.
- ArcFace's inductive bias (invariant to expression/pose, sensitive to identity) is exactly the inverse of what we want to penalise — that's why it works where SigLIP-distill (v1i, sg_b @ w=5000) failed.

### Identity/semantic anchor — `siglip_distill` (`SigLIPStudent` sg_b)

- `models/siglip_distill/v1/` — SigLIP distilled to Flux latents, 1152-d L2-normed.
- Tried as anchor in v1i; **failed** to prevent classifier-fooling (sg cosine preserved numerically while pixel-level identity drifted — sg lives on the VAE bottleneck, not at strict pixel/identity level).
- Useful as a *semantic* anchor for a different problem (e.g., concept-preservation), not as an identity anchor. Park it.

### Solvers / training scripts

- `extensions_built_in/concept_slider/ConceptSliderTrainer.py` (in ai-toolkit) — the trainer. Has `bs_loss_*`, `bs_loss2_*`, `id_loss_*`, `sg_loss_*` knobs all wired.
- `bs_only_mode: true` — the structural-LoRA training path that bypasses slider polarity / negative passes. **This is the path we use.**
- `bs_loss_t_max: 0.5` — only apply the bs critic on z_t at t≤0.5 (matches v3_t training distribution).
- `bs_loss_engage_target` — the scalar to push the named channel to (0.5 was too low for squint due to FFHQ baseline ~0.4–0.6; **use 0.85–0.9 for any future bs_only run**).
- `mask_dir`, `eye_mask_*` — per-image eye-region geometric mask amplifier; harmless on non-eye losses, additive when you want to discourage off-region perturbation.

### Working recipe — v1k pattern

```yaml
slider:
  bs_only_mode: true
  bs_loss_weight: 10000
  bs_loss_t_max: 0.5
  bs_loss_checkpoint: models/mediapipe_distill/bs_v3_t/final.pt   # or bs_v4_pgd once validated
  bs_loss_variant: bs_a
  bs_loss_mode: engage
  bs_loss_engage_target: 0.9     # well above any FFHQ baseline
  bs_loss_channels: [<single channel>]

  id_loss_weight: 5000
  id_loss_t_max: 0.5
  id_loss_checkpoint: models/arc_distill/checkpoint.pt
  id_loss_variant: latent_a2_full_native_shallow
```

200 steps, batch 1, lr 1.25e-4, adamw8bit, lora linear=32, ignore single_transformer_blocks/ff/norm. **Sample renders during training will look identity-collapsed at the network_multiplier sweep until step ~100; this is normal and recovers as the arc anchor catches up.**

### Inference — step-gated, structural

`bs_only` LoRAs trained at `t_max=0.5` are **structural**. Apply only at early-mid Flux denoising steps:
- `start_percent ≈ 0.0–0.05`
- `end_percent ≈ 0.75–0.85`

Without step-gating, the LoRA perturbs detail bands it never trained on → identity drift / lighting collapse at the multiplier=1.0 demo render. ComfyUI `LoRAControl` / ModelSamplingFlux step-gate covers this. **Not yet wired into our standard demo workflow** — pending engineering item.

## Failure ladder we climbed

| Run | What changed | Result | Lesson |
|---|---|---|---|
| v1c–v1g | classic slider+bs | eyeSquint ↔ eyeBlink coupled, closure at high m | Channel correlation in FFHQ training distribution → can't isolate squint from blink with bs alone |
| v1h | bs_only_mode (no slider polarity) | critic satisfied, render unchanged | Classifier-fooling at training scale; off-manifold direction |
| v1i | v1h + SigLIP anchor (sg_b, w=5000) | critic still fooled, sg cosine preserved, identity drifts | SigLIP-on-latent is too soft for pixel-identity preservation |
| v1j | v1h with single jawOpen→0.9 (sanity) | mouths visibly open, severe identity collapse | bs_only infrastructure works; identity drift is the universal failure mode regardless of channel |
| v1k | v1j + arc_distill anchor (w=5000) | mouths open, identity preserved across all 3 demos | **Working recipe.** ArcFace inductive bias is the missing piece. |

Pending (not yet run):
- **v1l_squint_arc** — apply v1k recipe to squint, with `engage_target: 0.9` (raised from 0.5 to clear FFHQ baseline overlap) and the dual eyeSquintLeft/Right channels. Optional: add geometric eye mask amplifier (already in v1h–v1k yaml, harmless when on).
- **v4_pgd validation** — confirm the robust critic actually rejects the v1h fooling case before swapping it into v1l. If it does, the arc anchor weight may be reducible.
- **Step-gated demo** — wire the start/end_percent step-gate into the standard demo workflow so v1k renders at multiplier=1.0 keep their identity through the final detail bands.

## Open questions worth holding

1. Does v4_pgd compose multiplicatively with the arc anchor (i.e., do we get even cleaner v1l with both, or does one dominate)? 2×2 ablation if compute permits.
2. Per-channel R²(t) curve says small/large blendshapes emerge at different points in the schedule — should `bs_loss_t_max` be **per-channel** rather than a global 0.5? See `feedback_blendshape_per_channel.md` and `project_blendshape_temporal_availability.md`.
3. Composition of multiple bs channels at training time (e.g., squint + smile in one LoRA): does the arc anchor scale, or do you need per-channel anchor weighting?
4. Does the recipe survive the move from Flux Krea to Flux Schnell / Flux1-dev? (only tested on Krea so far)

## Why we're parking

The recipe works on the cleanest case (jawOpen). Generalising to every ARKit channel is a per-axis `~1 hour training + per-axis ~1 hour validation × 52 channels` problem on current hardware, plus the per-axis engage_target tuning to clear baseline overlap. That's a ~100-hour GPU loop on a single 4090. Before committing to that path we want to evaluate **neural-deformation alternatives** (LivePortrait family, IM-Animation, FG-Portrait) that potentially give us all 52 ARKit channels post-hoc on any rendered portrait without per-axis LoRA training. See companion docs `2026-05-03-neural-deformation-for-blendshape-control-{topic,practical}.md` (in flight as of writing).

If neural deformation is the right call, this thread becomes a measurement / validator rather than the production edit mechanism. If it isn't, we resume here with v1l_squint_arc as the next experiment and the v4_pgd swap as the second-priority task.

## Resume sequence

```bash
# 1. Validate v4_pgd vs the foolability gates
.venv/bin/python scripts/sanity_bs_critic.py   # already multi-checkpoint

# 2. (If G4 passes) draft v1l_squint_arc.yaml from lora_v1k_jaw_arc.yaml:
#    - bs_loss_channels: [eyeSquintLeft, eyeSquintRight]
#    - bs_loss_engage_target: 0.9
#    - bs_loss_checkpoint: models/mediapipe_distill/bs_v4_pgd/final.pt
#    - keep id_loss_* at 5000

# 3. Launch via ai-toolkit (NOT through `uv run` — direct venv to avoid oyaml miss):
/home/newub/w/ai-toolkit/.venv/bin/python /home/newub/w/ai-toolkit/run.py \
  /home/newub/w/ai-toolkit/config/lora_v1l_squint_arc.yaml

# 4. Step-gated render at inference:
#    ComfyUI LoRAControl with start_percent=0.0, end_percent=0.8
```

## Cross-references

- Diagnosis: `docs/research/2026-04-30-bs-loss-classifier-fooling.md`
- Per-channel R² curves: `feedback_blendshape_per_channel.md` (memory)
- Step-gating reasoning: `project_bs_lora_step_gating.md` (memory)
- v4_pgd plan: `docs/superpowers/plans/2026-04-30-bs-v4-pgd-robust-critic.md`
- arc_distill model card: `models/arc_distill/README.md`
- Neural-deformation alternative: `docs/research/2026-05-03-neural-deformation-for-blendshape-control-{topic,practical}.md`
