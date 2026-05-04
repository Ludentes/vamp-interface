---
status: live
topic: arc-distill
---

# Noise-conditional distill students — design (2026-04-30)

Cross-cutting upgrade plan for the three teacher-distill heads
(`arc_distill`, `mediapipe_distill`, `siglip_distill`) so they can be used
not only as **slider/LoRA training losses** (Use A — works today) but also
as **inference-time classifier guidance** (Use B — out of distribution
today).

This doc captures the *design*; per-head implementation plans (training
scripts, validation plots) get their own dated docs when we pull the
trigger on a specific head.

## Problem

All three current students see only **clean** Flux VAE latents `z_0` during
training. At inference-time guidance, the diffusion sampler needs gradients
through the student at **partially-noised** latents
`z_t = α_t · z_0 + σ_t · ε` for every step in the schedule. The student has
no training signal at `t > 0`, so its embedding at noisy `z_t` is
out-of-distribution and the gradient direction is unreliable except in the
last 1–2 steps where `z_t ≈ z_0`.

This is the open question already flagged in
`docs/research/_topics/arc-distill.md`:

> Partially-noised latents at t > 0 of the diffusion schedule: behaviour
> untested; the student was trained on clean VAE-encoded latents only.

The fix is to teach the student about the noise schedule. Two paths.

## Path 1 — Cheap retrain (same architecture, random t)

Mirror the diffusion forward process during training:

```python
# in dataset / collate or training loop
t = torch.rand(B, device=device) * T_max          # uniform in (0, 1]
sigma = sigma_schedule(t)                          # Flux's σ(t)
alpha = alpha_schedule(t)                          # Flux's α(t)
eps = torch.randn_like(z_0)
z_t = alpha[..., None, None, None] * z_0 + sigma[..., None, None, None] * eps
loss = teacher_loss(student(z_t), target)          # same target as Use A
```

Same student architecture, same target, same loss. Only the input
distribution widens.

**Cost:** ~20 % more wall time (extra noise sampling + the model has to
fit a wider input manifold). No new data. Drops into the same Use B
inference path.

**Risk:** capacity bottleneck. The student now has to map *every*
`(z_0, t, ε)` triple to the same target embedding, which is a much wider
function. The 12 M-param trunk may saturate. If val cos at large `t`
collapses by > 0.05 vs the clean-only baseline, capacity is the bottleneck
and we go to Path 2.

**Validation primitive:** measure val cos as a function of `t` bucket. Plot
`val_cos_mean` for `t ∈ {0, 0.1, 0.25, 0.5, 0.75, 1.0}`. Should be a
roughly flat curve; any cliff says the student isn't fitting that regime.

## Path 2 — Noise-conditional student (`(z_t, t)` input)

Add `t` as a conditioning input. Inject via FiLM or AdaLN at each ResNet
block (and the stem):

```python
t_emb = sinusoidal_embedding(t, dim=128)           # standard diffusion timestep emb
gamma_l, beta_l = MLP(t_emb)                       # per-block (gamma, beta)
x = block(x) * (1 + gamma_l[..., None, None]) + beta_l[..., None, None]
```

At inference the caller passes `(z_t, t)` — the student knows where on the
schedule it is and can specialise.

**Cost:** ~5 % parameter increase (the FiLM MLPs), ~20 % wall-time on top
of Path 1's cost. Architectural change, so checkpoints don't transfer
directly from the clean-only models — needs a fresh training run.

**Risk:** None significant beyond Path 1. This is the canonical
noise-conditional adapter pattern; if Path 1 saturates, Path 2 is the
answer.

## Per-head application

| Head | Variant tag | Target | Use A status | Cost of Path 1 retrain |
|---|---|---|---|---|
| `arc_distill`   | `latent_a2_full_native_shallow_t` | 512-d ArcFace | val cos 0.881, shipped | ~25 min on shard (was ~20 min) |
| `mediapipe_distill` | `bs_v2c_t` (or fresh `bs_v3`) | 52-d ARKit blendshapes | v1 median R² 0.761; v2c training now | ~30 min |
| `siglip_distill`  | `sg_a_t`             | 1152-d SigLIP-2 emb     | v1 training now; cos ≥ 0.85 expected | ~25 min |

**Order to pull them:** start with `siglip_distill` because the slider
glasses_v9 vs v8 downstream test is the one we actually care about for
Use A right now, and the noise-conditional version is the one we'd use if
glasses_v9's eval rig later wanted *guidance* instead of *training loss*.
Then `arc_distill` (the most-tested head, baseline 0.881 to beat). Then
`mediapipe_distill` (R² is already noisy, so guidance from this head is
the most fragile use case and worth deferring until we've seen the other
two work).

## Implementation notes shared across heads

- **Schedule source.** Use Flux's actual schedule, not a generic one — the
  student must see the same `(α_t, σ_t)` pairs as inference. Pull from
  `comfy_extras` or wherever Flux's noise schedule is defined; pin the
  version.
- **Pre-compute or on-the-fly?** Pre-computing noisy latents is wasteful
  (storage explodes by `n_t_buckets`× and we lose the continuous-`t`
  benefit). Generate on-the-fly per batch, costs ~one extra randn +
  multiply per sample.
- **Train/val split unchanged.** SHA-prefix `'f'` → val. The val noise is
  freshly sampled per evaluation; if val cos is unstable across runs, fix
  a per-row noise seed for the val set so the comparison is deterministic.
- **Reuse the existing compact pair files.** `compact.pt` (latents),
  `compact_blendshapes.pt`, `compact_siglip.pt` — no rebuild needed.
- **Save diagnostic curves.** Every retrained head should ship a
  `val_cos_vs_t.json` (or `val_r2_vs_t.json` for mediapipe) so we can see
  the schedule-shape of the student's quality, not just an averaged number.

## Validation criteria

For each head, the noise-conditional version is "good enough" if:

- **Clean-input parity.** Val cos / R² at `t = 0` is within 0.01 of the
  clean-only baseline. If the noise-conditional student gets noticeably
  worse at `t = 0`, capacity is leaking.
- **Schedule coverage.** Val cos at every `t` bucket in
  `{0.1, 0.25, 0.5, 0.75, 1.0}` is within 0.05 of the `t = 0` value. A
  cliff anywhere says the student isn't fitting the schedule.
- **Use B sanity.** Run a 1-prompt round-trip: encode a real face → add
  noise at `t = 0.5` → student → cosine to teacher embedding of the clean
  face. Should be > 0.7 (looser bar than `t = 0` clean cos because the
  signal really is degraded). This is the smallest end-to-end test of
  guidance-readiness.

If a head fails clean-input parity → Path 2 (architecture change). If it
fails schedule coverage at high `t` → Path 2 with stronger conditioning.
If it fails the round-trip but passes the others → the loss formulation is
the bottleneck; iterate on the per-head loss design (e.g. weight by `α_t`).

## Why we are NOT doing this now

Use A (slider/LoRA training loss) is the load-bearing application for the
near-term roadmap (`glasses_v9` slider, smile axis, squint axis). Use B
(inference guidance) is speculative until at least one Use A demonstrates
real value over the v8 baseline. Building the noise-conditional version
*before* Use A is YAGNI.

This doc exists so that **once Use A demonstrates value and somebody asks
"can we use the same student to guide the sampler?"**, the upgrade plan is
already specced and we don't have to re-derive it.

## Trigger condition

Pull this off the shelf when **one** of:

1. A slider built with the Use-A loss (e.g. `glasses_v9`) outperforms the
   no-loss baseline (`glasses_v8`) on the existing eval rig, **and**
   somebody wants to test whether the same student can also guide
   inference for the same axis.
2. Identity-preservation guidance becomes a feature request — i.e. someone
   wants the sampler to actively pull toward an ArcFace target rather
   than just to train a slider that does so.
3. A research thread (e.g. classifier guidance experiments on Flux Krea)
   needs an embedding head and we want to skip training one from scratch.

Until then, the clean-only students are sufficient for everything we
actually do today.
