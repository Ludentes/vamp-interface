---
status: live
topic: arc-distill
---

# Layer 3 spec — slider A/B with vs without student loss

**Question:** Does adding `λ · (1 − cos(student(x_edited), student(x_anchor)))`
to a Concept-Sliders training objective measurably reduce identity drift at
matched edit magnitude?

This is the real Use-A test (see [2026-04-30-arcface-frozen-adapter.md](2026-04-30-arcface-frozen-adapter.md)
"Possible uses" section). Layer 1.1–1.3 + Layer 2 say the student is plausibly
a usable loss; Layer 3 is the operating-conditions check.

## Setup

**Slider axis.** Pick **smile** (we already have v3 + v3.1 corpora and a working
`FluxSpaceEditPair` for it; small, cheap to retrain).

**Two LoRA training runs**, identical except for the loss term:
- **A (control)**: existing Concept-Sliders objective.
- **B (student-loss)**: same objective + `λ · L_id` where
  `L_id = 1 − cos(student(z_edited.detach()), student(z_anchor))`.
  `z_anchor` and `z_edited` are the (16, 64, 64) latents at the diffusion step
  the slider is supervised at. Compute the student forward in fp32 (or bf16
  with a manual cast); detach `z_edited` if you only want to constrain the
  *prediction*; keep grads if you want to constrain the *trajectory*. Default:
  no detach — student-loss steers the trajectory.
- λ tuning: 0.5 / 1.0 / 2.0. Pick by visual triage on a 4-image probe.

**Seed and step parity.** Same seeds, same step count, same LR schedule, same
training pairs. Only differ in the added loss term and `λ`.

## Eval protocol

**Held-out test set:** 30 anchor identities not seen in slider training, balanced
across age × gender × race using `face_attrs.pt` quantiles.

**For each anchor:** generate {scale 0, 0.5, 1.0, 1.5} images with both the A
and B LoRAs. 30 anchors × 4 scales × 2 LoRAs = 240 renders. ~30 min on shard.

**Identity drift metric (the headline):**
- Run *real* InsightFace ArcFace (the buffalo_l ONNX teacher, NOT our student
  — judge must be independent of the loss being trained against) on every
  rendered face.
- For each anchor, compute `id_drift(scale) = 1 − cos(emb(scale), emb(scale=0))`.
- Plot `id_drift` vs `scale` per LoRA. **The win condition is B's curve below
  A's at matched scale, especially at scale ≥ 1.0** where edits are largest.

**Edit magnitude (sanity, must be matched):**
- Visual edit strength: per-anchor pairwise LPIPS between scale=0 and scale=k.
  If B has lower LPIPS at the same scale, B is just editing less, not preserving
  identity better. Report `id_drift` *normalised by* LPIPS to control for this.

**Secondary readouts:**
- Demographic-ridge transfer (age/yaw): does the student-loss preserve
  age/pose information at scale=1 better than the control?
- Failure-mode triage: 5 worst-anchor renders from A vs B, side-by-side. Shows
  *what kind* of identity drift the student-loss prevents.

## Decision rule

Ship the student-loss (move to MediaPipe head + treat ArcFace student as
production-validated) **iff**:
1. B's median `id_drift` at scale=1 is at least 15% lower than A's, AND
2. LPIPS-normalised drift confirms the win isn't just smaller edits, AND
3. Visual triage shows no new failure mode introduced (e.g., faces collapsing
   to the anchor, smile suppression beyond what control shows).

If only (1) holds without (2): student-loss over-constrains — try smaller λ or
detach `z_edited`.

If neither (1) nor (2) holds: student is not a usable loss head at this fidelity
(0.881). Either retrain with stronger backbone unfreeze, switch architectures,
or accept that 0.881 is too low and demote the project to a Use-A measurement
tool only.

## Build sequence

1. Identify where Concept-Sliders training computes its objective (in
   `/home/newub/w/ai-toolkit/...` — needs spelunking; not done yet).
2. Add an optional `--id-loss-checkpoint` arg + `--id-loss-weight` arg that
   loads `latent_a2_full_native_shallow` and adds the term.
3. Smoke-test: 5 training steps, both runs produce checkpoints, no NaN.
4. Full A/B: ~1 h training each on shard.
5. Render the 240-image probe + compute metrics + write up.

Checkpoint to use: `C:\arc_distill\arc_adapter\latent_a2_full_native_shallow\checkpoint.pt`
(0.881 / 0.939 / 87.6%).

## Open questions

- Operating point for the student during slider training: slider LoRAs supervise
  at `t > 0` denoising steps where the latent is still partially noised. The
  student was trained on *clean* VAE-encoded latents. Behaviour on partially
  denoised latents is untested. Quick check: run student inference on a clean
  vs noised latent at t=0.5; compare cosine distance.
- Detach vs not on `z_edited`: detaching constrains only the prediction; not
  detaching constrains the trajectory. Default not-detach but worth a side
  experiment.
- LoRA capacity interaction: at strong λ, the LoRA may simply not learn the
  edit. Need to confirm the smile axis still moves under loss.

## Build status

Spec only. Implementation not started. Estimated time-to-completion: ~half a day
once the Concept-Sliders training script is located and instrumented.
