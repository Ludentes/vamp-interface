---
status: live
topic: demographic-pc-pipeline
---

# Slider trainer — phased plan (v1.0 → v1.5 → v1.1 → v2)

Revision of the earlier `v1 / v1.1 / v2` roll-up after bumping
composability into v1 scope. The single-axis smoke-test now gets its
own milestone (v1.0) so the whole-pipeline plumbing is validated
before we commit to a joint 10-axis training run.

## Why this ordering

- **v1.0 first** so we can fail fast: if VAE caching, LoRA hook
  insertion, multi-α supervision, or timestep biasing is broken, we
  find out on a 60–90 min run, not an overnight 10-axis run.
- **v1.5 second** — once the single-axis path works, turning on joint
  training + orthogonality reg is a mechanical extension, not a new
  project. Composability gets built in, not bolted on.
- **v1.1** (warm-start from cached δ) stays in the queue as a
  derisking experiment on one axis, to run *after* v1.5 ships.
- **v2** stays as "full dictionary + further axis additions".

## v1.0 — single-axis smoke test

**Goal:** prove the end-to-end pipeline on `eye_squint`. Ship nothing.

**What we build:**
- `src/demographic_pc/train_flux_image_slider.py` — single-axis trainer.
- `src/demographic_pc/eval_slider.py` — validation-protocol (step 3) runner.
- Cached VAE latents + cached T5+CLIP embeddings for `eye_squint`.

**Recipe (from steps 2, 2.6, 4-to-come):**
- LoRA rank=16, α=1, xattn, bf16
- Multi-α supervision (random α ∈ {0.25, 0.5, 0.75, 1.0})
- Logit-normal timestep bias (peak ~0.3)
- 1000 steps, LR 2e-3, 8-bit Adam, grad ckpt
- Trained on 5 bases × 3 seeds × 5 α = 75 training samples
- Held out: 1 representative base (`european_m` — middle-of-pack in
  screening) + 1 hardest base (TBD after calibration pass; `eye_squint`
  didn't have a hardest base per se but `black_f` was tightest in Δ)

**Pass criteria (step 3 applied to one axis):**
- P1 target-Δ ≥ 0.72 (= 0.8 × +0.90 prompt-pair effect)
- P2 monotonicity ρ ≥ 0.85
- G1 ArcFace cos ≥ 0.70
- G2 off-target L¹ ≤ 0.03

**Diagnostic output (pass OR fail):**
- Loss curve, LR schedule plot
- VRAM peak, wallclock per step, total wallclock
- Sample renders at scale ∈ {0, 0.25, 0.5, 0.75, 1.0} on held-out bases
- Slider card (step 3 format)

**Go / no-go decision for v1.5:**
- **Go** if all step-3 pass criteria met.
- **Iterate** if any P-metric in warn: tune timestep bias or α-sampling
  distribution, re-run. Max 3 iteration budget.
- **No-go** if any P-metric fails: pause, investigate (likely a loss
  or data-loader bug, not a recipe problem), fix before v1.5.

## v1.5 — joint-composable training

**Goal:** ship 10 composable sliders.

**What changes from v1.0:**
- 10 LoRA adapters loaded simultaneously (~300 MB combined).
- Per-step axis sampling: uniform over 10 axes (re-weighted later if
  some axes converge slower).
- Orthogonality regulariser: `λ · Σ_{i,j:i≠j} ||A_i · A_jᵀ||²_F / r²`
  on LoRA down-projections, sampled pairwise per step to keep O(N)
  cost.
- Step count: ~10 000 (1000 × 10 axes, no re-loading overhead).
- PEFT `set_adapter(name)` / `disable_adapters()` per step.

**Training-data policy:** no rendered composition pairs. Axes never
see each other in training; orthogonality reg is the only compositional
signal. If P4/P5 fail on v1.5, rendering composition pairs becomes a
v1.6 experiment.

**Validation additions (step 3.5 — to be written):**
- P4. Additivity across axis pairs (MAE over pair-metrics ≤ 0.15 ×
  max-individual-effect).
- P5. Interference (each axis's solo effect retained ≥ 0.7× under
  pairwise composition).

## Inference/rendering path (v1.5 task, not blocking)

v1.0 rendering uses diffusers `FluxPipeline` directly (pre-encode
prompts, drop text encoders to fit 32 GB). Works, but leaves memory
and fp8 stability on the table.

**Migration target:** ComfyUI custom node (`demographic_pc_slider/`)
that (1) loads our PEFT-format LoRA safetensors (keys like
`base_model.model.transformer_blocks.0.attn.to_q.lora_A.slider.weight`
— the standard LoRA loader node may need adaptation), (2) exposes a
`slider_scale` float input that mutates the adapter's scaling at
forward time. Lets us reuse Comfy's fp8 Flux path, auto-offload, and
our existing `inject_workflow` infrastructure
(`au_library_smoke.py`, `compose_iterative.py`).

Not blocking v1.0 evaluation — diffusers path works.


## v1.1 — warm-start ablation (parked)

Run on one axis post-v1.5, SVD-decompose cached `FluxSpaceEditPair`
attention deltas → rank-16 `(A_init, B_init)`. Compare convergence +
final quality vs v1.0 Kaiming/zero init.

## v2 — dictionary expansion

New axes, refined recipe based on v1.5 lessons, possibly rendered
composition pairs if orthogonality proves insufficient.

## Summary

```
v1.0  single-axis smoke test      — eye_squint, 60–90 min, gating validation
  ↓   (all step-3 criteria pass)
v1.5  joint 10-axis composable    — overnight, ship sliders
  ↓
v1.1  warm-start ablation         — compare on eye_squint, adopt if wins
  ↓
v2    dictionary expansion        — additional axes, refined recipe
```

## Next

Step 4: hyperparameters (for the v1.0 smoke test specifically — same
recipe carries forward to v1.5 with addition of ortho-reg `λ`).
