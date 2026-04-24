---
status: live
topic: demographic-pc-pipeline
---

# Step 4 — Hyperparameters for v1.0 smoke test

Locks concrete values for the single-axis trainer. Every number here is
**motivated by step 2 (family refresher), step 2.6 (improvements), or
the reference notebook**; nothing is invented.

## The training objective — flow-matching velocity on the edited trajectory

Derivation (important — drives every other hyperparam choice):

Base Flux at prompt `P` predicts `v(z_t, t | P) ≈ ε − z_unedited`
(noise-to-clean velocity). We want *slider-on* Flux to predict `ε −
z_edited(α)`. Training target:

```
α_train ∼ U{0.25, 0.5, 0.75, 1.0}        # multi-α supervision
z_after = VAE(img_at_α=α_train)           # cached
ε ∼ N(0, I)
t ∼ LogitNormal(μ=0.5, σ=1.0)             # logit-normal bias
z_t = (1 − t) · z_after + t · ε

# forward with LoRA scale set to α_train:
v̂ = flux(z_t, t, base_prompt_embeds, LoRA_scale=α_train)

v_target = ε − z_after
loss = MSE(v̂, v_target)
```

Interpretation: at LoRA-scale=`α_train`, the model must predict the
velocity field whose integration lands on the `α_train`-edited image.
At scale=0 the LoRA contribution vanishes (B init zero) → base
velocity, unedited. This makes the slider scale **semantically
equal to the data's α** by construction.

## LoRA parametrisation

From step 2:

| Param | Value | Rationale |
|---|---|---|
| Rank `r` | **16** | Reference recipe; ~30 MB total params; conservative capacity for single-axis edit |
| Alpha `α_lora` | **1** | scale = α/r = 1/16; damps injection |
| Target modules | **xattn** (all `attn.*` linears in both double and single blocks) | 304 linears; leaves MLPs frozen → preserves base-model identity |
| Init A | Kaiming uniform | step 2 |
| Init B | Zeros | ΔW=0 at init → LoRA'd model == base Flux at step 0 |
| LoRA dropout | **0** | no regularisation in v1; reintroduce if overfitting shows |

**Class-name match:** inspect downloaded Flux Krea state dict at first
load; if diffusers version uses `FluxAttention` instead of `Attention`,
update the module-class filter accordingly (gap flagged in step 2).

## Optimizer

| Param | Value | Source |
|---|---|---|
| Optimizer | **AdamW 8-bit** (bitsandbytes) | saves ~60 MB optim state |
| Learning rate | **2e-3** | reference notebook |
| β1 | 0.9 | default |
| β2 | 0.999 | default |
| ε | 1e-8 | default |
| Weight decay | **0** | LoRA with init-zero B does not benefit from WD in practice |
| Grad clip | **max_grad_norm=1.0** | stability safeguard |

LR=2e-3 is much higher than full-finetune typical (1e-5), justified by
LoRA's small parameter count + α/r = 1/16 scaling which effectively
reduces the apparent LR by that same factor.

## LR schedule

| Param | Value | Source |
|---|---|---|
| Schedule | **constant with warmup** | reference notebook |
| Warmup steps | **200** | reference notebook |
| Total steps | **1000** | reference notebook; revisit if loss still dropping at end |
| Warmup shape | linear | diffusers default |

## Batch & precision

| Param | Value | Rationale |
|---|---|---|
| Batch size | **1** | reference notebook; 5090 VRAM-safe |
| Gradient accumulation | **4** | effective batch 4, smoother gradients |
| Effective batch | **4** | — |
| Mixed precision | **bf16** | diffusion gradient underflow on fp16 (step 1) |
| Grad checkpointing | **on** | required for 32 GB fit |
| torch.compile | **off** (v1.0) | adds debugging surface; enable in v1.5 |

## Timestep sampling (logit-normal bias, tier-1 improvement)

Standard Flux training uses uniform `t ∼ U[0,1]`. Edit-training
benefits from biasing toward mid-to-high-noise timesteps where
structural decisions are made.

```
u ∼ N(μ=0.5, σ=1.0)
t = sigmoid(u)       # peak at t ≈ 0.62 (mid-high noise)
```

Why μ=0.5 (not 0): the first 30% of denoising (high t) is where
semantic edits imprint; boosting that region over-samples the regime
with the strongest edit signal. μ=0.5 is conservative — more
aggressive choice would be μ=0.8 (peak at t=0.69). Revisit if v1.0
convergence is slow.

## α sampling (multi-α supervision, tier-1 improvement)

```
α_train ∼ U{0.25, 0.5, 0.75, 1.0}    # discrete uniform, 4 values
```

Skips α=0: target would be `z_0 − z_0 = 0`, tautologically low-loss.
v1.1 anchor loss handles the scale=0 case explicitly. Endpoint α=1 is
the strongest training signal; uniform (not weighted) because over-
weighting endpoints caused monotonicity issues in SDXL image-sliders
reports.

## Data

| Param | Value | Rationale |
|---|---|---|
| Corpus | `output/demographic_pc/fluxspace_metrics/crossdemo/eye_squint/eye_squint_inphase/` | step 1 |
| Bases in train | **5** (hold-out = `european_m`) | step 3 LOBO |
| Seeds in train | **3** (1337, 2026, 4242) | all |
| α values in train | **5** (0, 0.25, 0.5, 0.75, 1.0) | multi-α |
| Total samples | **5 × 3 × 5 = 75** | — |
| Augmentation | **none** | step 2.6; horizontal flip ruled out |
| Resolution | **512×512** | VAE-compatible, FaceAPI-adequate |
| Data loader | cached VAE latents + cached T5+CLIP embeds | no re-encode per step |

**One-shot precompute step before training:**
1. VAE-encode all 75 training images → cache as `.safetensors`.
2. T5+CLIP-encode 6 base prompts → cache.
3. Drop VAE + text encoders from GPU.

**Target prompt** per base: the base's characterisation prompt
(e.g. `"portrait photo of a young european woman, neutral expression"`
for `young_european_f`). From the FluxSpaceEditPair corpus's
`base_prompt` field in `measure_*.pkl` — need to pull these at cache
time.

## Logging & checkpointing

| Param | Value |
|---|---|
| Log cadence | every step (loss, LR, grad_norm, t, α_train) |
| Logger | `wandb` if `WANDB_API_KEY` in env, else stdout + CSV |
| Checkpoint cadence | steps 500, 1000 |
| Checkpoint format | `diffusers`-compatible LoRA safetensors + PEFT state dict |
| Eval during train | quick eval every 250 steps (1 base × 1 seed × 5 scales) |
| Final export | merged-LoRA safetensors + metadata card |

## Wallclock / VRAM budget

Expected on 5090 (ComfyUI killed first):

| Component | Cost |
|---|---|
| Flux-Krea-dev bf16 (frozen) | 23.8 GB resident |
| LoRA params + 8-bit Adam state | ~0.1 GB |
| Activations @ bsz=1 + grad ckpt | 3–4 GB |
| Cached latents + embeds (precomputed) | ~0.5 GB |
| **Peak VRAM** | **~28 GB** |
| Headroom to 32 GB | ~4 GB |
| Forward pass (4 DiT forwards not needed — image target) | ~1.5 s |
| Training step (1 forward + backward) | ~2.5–3 s |
| 1000 steps | **~45–50 min** |
| Eval (quick @ 250/500/750/1000 steps) | ~2 min |
| **Total wallclock** | **~55 min** |

Tighter than the earlier 60–90 min estimate because image-pair target
eliminates the teacher-blend (3 extra forwards/step in the notebook).

## Reproducibility

```
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
```

`torch.backends.cudnn.deterministic = True` is intentionally **not**
set — determinism costs ~20% throughput and we don't need bit-exact
reproducibility. Seed fixed for loss-curve comparability across runs.

## Exit criteria for v1.0 → v1.5

All P-metrics in pass, no G-metric in fail (step 3 table). If yes:
- Commit trainer + eval scripts
- Commit slider card for eye_squint
- Proceed to v1.5 (extend trainer to multi-adapter, add ortho reg, add
  P4/P5 additivity eval)

If warn: tune μ for logit-normal (try 0.0 and 0.8 as ablation), or
bump steps to 1500. Max 3 iteration budget before treating as a bug.

If fail: pause, investigate. Likely suspects: LoRA hook insertion
(wrong module class), α-scale multiplier not actually wired to LoRA
scale at forward, VAE cache stale, timestep schedule miscomputed.

## Next

Step 5 — write the trainer.
