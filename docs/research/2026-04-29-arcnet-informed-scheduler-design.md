---
status: live
topic: arc-distill
---

# ArcNet-informed Flux scheduler — design (2026-04-29)

A scheduler for diffusion sampling decides two things: how many steps to
take and where to place them on the noise schedule. Flux's default uses a
rectified-flow timestep grid with a resolution-dependent "shift" parameter.
None of that placement is informed by *what is happening* in the
generation — it's a static heuristic.

Once we have a noise-conditional latent-space critic — `arc_student(z_t, t)`
returning a 512-d ArcFace embedding usable across the full schedule — we
can replace the heuristic with an *empirically grounded* schedule. This
document captures the experiment program.

## Background: what counts as a scheduler intervention

A typical diffusion sampler at inference computes, per step:

```
ε̂_t  = base_model(z_t, t, prompt_emb)
x̂₀_t = (z_t − σ_t · ε̂_t) / α_t          # Tweedie estimate of clean
z_{t-Δ} = step_fn(z_t, ε̂_t, t)            # rectified-flow / DPM / etc.
```

The scheduler controls (a) the set of `t` values visited, (b) `Δ` between
consecutive visits, (c) optionally early-termination, and (d) per-step
guidance scale. None of these knobs ride on what the model is *doing*; the
schedule is fixed before sampling begins.

Our students give us a per-step measurement of *content* (identity,
expression, attributes) at zero pixel-decode cost. That measurement is the
input to a smarter scheduler.

## Three scheduler designs

### A — Step-density redistribution (most ambitious)

Find where on the trajectory identity actually commits. Reallocate step
budget to match.

Procedure:

- Run a baseline 30-step Flux generation on N seeds.
- At each step, compute `x̂₀_t` and `arc_student(x̂₀_t, t)`.
- Record cosine of `arc_student(x̂₀_t)` to `arc_student(x̂₀_final, 0)`.
- Average over N seeds → "ID emergence curve" `c(t)`.
- The steep portion of `c(t)` = where ID is locking in. The flat portions
  (early: not yet emerged; late: already settled) are wasted compute.
- Build a new schedule that distributes steps in proportion to `|dc/dt|`.

This is conceptually the same construction as NVIDIA's *Align Your Steps*
(AYS, 2024) but with an ID-specific metric instead of a generic
data-distribution metric. Stronger because task-aware: the "where do steps
matter" answer for an identity-preservation use case is different from the
answer for a creative-prompt use case.

**Cost:** ~1 evening for the measurement + curve. Then schedule
construction is closed-form (cumulative inverse of `|dc/dt|`).

**Risk:** the curve may be seed-dependent enough that no single redistributed
schedule beats the baseline averaged across seeds. Mitigation: per-seed
adaptive schedule (more expensive at inference, but possible since the
critic is cheap).

### B — Early termination on identity convergence

Once `arc_student(x̂₀_t, t)` cosine to its prior step is > 0.99 for k
consecutive steps, stop. Skip the final 3–5 steps when nothing
identity-relevant is changing.

**Cost:** trivial to test.

**Risk:** steps after ID-stabilization may still refine non-identity
features (skin detail, lighting). Quality / step-savings trade-off needs
visual inspection. Likely safe-by-default with large k (e.g. ≥ 3).

### C — Adaptive guidance scale

Use student-output stability as a confidence proxy for CFG:

- When ID is volatile (early steps, big direction change in `arc_student`
  outputs), turn CFG up — bias toward the prompt while there's still
  something to bias.
- When ID is locked (late steps, near-identity outputs across consecutive
  steps), turn CFG down — don't blow the converged identity with cumulative
  classifier nudges.

**Cost:** cheap; piggy-backs on the per-step student output we already
have to compute for A or B.

**Risk:** standard CFG-tuning interactions. Worth testing as an add-on, not
a primary path.

## Generalisation beyond ArcFace

The same construction works with any of our distilled critics:

- **MediaPipe-informed:** redistribute where *expression* commits. Likely
  later in the trajectory than ID (expression is finer-grained).
- **SigLIP-informed:** redistribute where high-level attributes (smile,
  glasses, age, beard) commit. Probably overlaps with the
  expression-commit window.
- **Multi-objective:** combine. Each student produces its own commit-curve;
  the optimal schedule is a weighted union of their steep zones.

This is the most direct demonstration that the three distill heads are
complementary at inference, not redundant.

## Why this is interesting

Most published adaptive samplers use cheap-to-compute proxies — DPM-Solver
uses the noise prediction itself; AYS uses a known data distribution.
A *trained* attribute-aware critic is more expressive: it sees the same
generation through the lens of "is this person yet?" / "is this person
smiling yet?" / "does this person have glasses yet?" and can place steps
to lock those attributes individually.

It also makes the distill heads pull their weight at *inference*, not just
at *training* time. The Use A (slider/LoRA loss) thread treated the
students as offline tools. This thread treats them as online tools that
ride along with the sampler.

## Prerequisite: noise-conditional students

The whole programme requires students that produce reliable embeddings at
noise levels other than `t = 0`. Tracking the cosine of
`arc_student(x̂₀_t, t)` to its final value across the trajectory only
tells you something coherent if the student is reliable across `t`.

Status as of 2026-04-29 evening:

- `siglip_distill` Path 2 (sg_c) — schedule curve nearly flat to t=0.75
  (cos drop ≤ 0.015), reliable across most of the trajectory. Trained
  to epoch 10, stopped before plateau but already deployable for
  measurement-only use.
- `arc_distill` Path 1 (latent_a2_full_native_shallow_t) — schedule curve
  cliffs hard: t=0 0.869, t=0.5 0.525, t=1.0 0.03. Reliable only in
  `t ≤ ~0.25`. Architectural ceiling from frozen IResNet50.
- `arc_distill` Path 2 (latent_a2_full_native_shallow_t2) — running. FiLM
  at five stage boundaries. Early indicators show small constant boost
  over Path 1 but not transformative; possibly will land at t=0.5 ≈ 0.55.

If Path 2 lands ≥ 0.83 at t=0.5, all three scheduler designs are viable
across the full trajectory. If it lands ≈ 0.55 (current trajectory), the
useful measurement window for arc_distill is `t ≤ ~0.4`. That happens to
be exactly where ID typically commits in diffusion (the late half of
denoising), so it may be enough for the scheduler use case anyway.

## Experiment 1 — the baseline measurement

This is the gating experiment: it tells us whether the rest of the
programme is worth pursuing, and what the empirical commit curve actually
looks like.

**Inputs:**

- 8 fixed seeds.
- 1 fixed prompt that produces a clear face (e.g. the demographic-PC neutral
  anchor prompt).
- Default Flux scheduler with 30 steps.
- Noise-conditional students for arc, siglip (mediapipe optional).

**Procedure:**

```python
for seed in seeds:
    initial_noise = torch.randn(...)
    z_t = initial_noise
    history = []
    for t in flux_default_schedule:
        eps = flux_model(z_t, t, prompt_emb)
        x_hat0 = (z_t - sigma(t) * eps) / alpha(t)
        arc_emb = arc_student_t(x_hat0, t)
        sg_emb = siglip_student_t(x_hat0, t)
        history.append({"t": t, "arc": arc_emb, "sg": sg_emb})
        z_t = step_fn(z_t, eps, t)
    save(history)
```

**Outputs:**

- `arc_emergence_curve.png` — cos(arc_t, arc_final) vs t, mean ± std over
  seeds.
- `siglip_attr_emergence.png` — for each of the 12 sg probes, plot
  `dot(siglip_t, probe_dir)` vs t. Reveals when each attribute commits.
- `commit_windows.json` — per-attribute (t_start, t_end) where commit is
  happening (where `|dc/dt|` > threshold).

**Decision rule:**

- If `arc_emergence_curve` is sharp (steep transition over ≤ 5 of 30
  steps): redistribution is high-leverage. Build A.
- If smooth ramp from step 5 to step 25: redistribution is subtle, build B
  (early termination) first as the cheaper win.
- If the curve is flat (ID commits before step 0?? unlikely): something is
  wrong with the measurement; debug.

## Open questions

- **Schedule placement vs schedule count.** Most of the discussion above
  is about *where* steps go. Is the optimal *count* also a function of the
  ID emergence curve? Probably yes — if ID commits in 5 steps, we don't
  need 30. But the budget knob may be application-dependent (creative use
  may want more even though ID is locked).
- **Per-prompt vs global schedule.** Different prompts likely commit at
  different `t`. Is the optimal schedule prompt-conditional? If so,
  experiment design extends to multi-prompt commit-curve clustering.
- **Interaction with CFG.** All measurements above are at default CFG.
  Does the commit curve shift under different CFG values? If yes, scheduler
  and CFG are jointly optimisable.
- **MediaPipe student in the noise-conditional pipeline.** Currently v2c
  is clean-only. If the scheduler experiments produce signal, retraining
  v2c noise-conditionally becomes a priority instead of a nice-to-have.

## What this is not

- Not a replacement for the Use A (slider/LoRA training loss) work in the
  parallel thread. That's offline; this is online.
- Not a fix for slow Flux inference per se. Step *redistribution* is
  free; step *count reduction* is where actual speedup comes from, and
  that's design B (early termination) — at most a 10-15% saving if it
  works.
- Not a new sampler algorithm. The step function (rectified-flow Euler,
  or DPM, or whatever Flux ships) stays the same; only the schedule
  changes.

## Trigger condition

Pull this off the shelf when **either**:

1. `arc_distill` Path 2 lands at t=0.5 ≥ 0.83 (clean schedule coverage),
   making all three designs viable, OR
2. We accept the t ≤ 0.25 reliable window from Path 1 as sufficient for
   late-trajectory commit measurement, and run Experiment 1 on that window
   only.

Until then, the noise-conditional distill experiments are gating this work.
