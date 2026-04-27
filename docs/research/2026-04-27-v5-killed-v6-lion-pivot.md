---
status: live
topic: demographic-pc-pipeline
---

# v5 killed at step ~450; v6 = Lion + v4-shape config

v5 ran for ~16 min (450 steps of 800) before being killed. At training-time
sample renders through step ~400, **glasses did not appear** on any of the
9 sample prompts. v4 also had no glasses at step 400 — engagement first
showed at step 600. So step-400 absence isn't itself "failure". What
makes us pivot is the *shape* of the v5 trajectory vs v4 plus the visible
artefacts.

## v4 vs v5 configs side by side

| Lever              | v4         | v5          | Δ                       |
|--------------------|------------|-------------|-------------------------|
| α (linear_alpha)   | 1          | 16          | 16× per-step LoRA scale |
| η (guidance_strength) | 4.0     | 2.5         | 0.625×                  |
| effective per-step (α/r)·η | 0.25 | 2.5      | **10× v5**              |
| LR                 | 1.25e-4    | 1.25e-4     | same                    |
| LR schedule        | constant   | cosine→0    | v5 decays over 800      |
| Steps              | 1500       | 800         | 0.53×                   |
| Dataset            | v1 generic | v2 demo-conditioned, +acetate variant | richer  |
| Optimizer          | adamw8bit  | adamw8bit   | same                    |
| Scope              | xattn      | xattn       | same                    |
| EMA                | off        | off         | same                    |

v5 has **10× the per-step LoRA gradient pressure** of v4. With same lr,
v5 should engage earlier than v4 — but doesn't.

## What this teaches

- **"Effective signal" is not equal to "engagement speed"** in this
  setup. Concept-slider's pos/neg symmetric supervision has a lazy
  fixed point ("change nothing → both branches predict the same → low
  loss"). With α=1·η=4 (v4), per-step pressure is small enough that
  the optimizer must accumulate gradient across hundreds of steps to
  break out of the lazy basin — and the gradual accumulation explores
  the loss landscape. With α=16·η=2.5 (v5), per-step pressure is large
  enough to "commit" early to a basin in the first ~50 steps. If that
  basin is the lazy fixed point, no amount of subsequent gradient
  helps escape.

- **Path-dependence dominates magnitude here.** Pumping α or η further
  amplifies the same lever that's not working. The diagnostic split
  (alpha-only vs η-only) is moot if the optimization is basin-stuck.

- **"Principled α=rank" was unvalidated for our stack.** Concept-slider
  papers default to α=rank for standard concepts, but the trainer +
  Flux + xattn-only scope + concept_slider loss is a specific stack
  with its own dynamics. v4 proved α=1 works here. v5 jumped to α=16
  on first-principles grounds and regressed.

- **Multi-variable change broke isolation.** v5 changed α, η, scheduler,
  dataset, captions all at once. We can't attribute v5's behaviour to
  any single change. Future iterations should change one variable at
  a time when the previous run was already a known-good (or
  known-semi-good) baseline.

User-observed visual differences at step 400:

- **v5 LoRA contribution at +1.0 too weak** for the local glasses
  feature (no rims, no lenses).
- **v5 has less wash-out / contrast collapse** than v4 sample renders
  at comparable steps (a global improvement).
- **Stereotype bundle remains** — gym-bro vs academic-portrait
  semantic drift is still visible. v2 dataset's demographic captions
  did not eliminate the bundle.

The local-vs-global split is interesting: cosine LR + α=16 may have
softened the bundle's per-step magnitude (less wash-out) without
reaching the engagement basin (no glasses).

## v6 plan: Lion at v4-shape

User direction: "Let's try Lion with settings close to v4 (given the
optimizer change)."

Lion is sign-based — every weight gets a unit-magnitude step in the
direction of the gradient sign, regardless of gradient magnitude. In a
flat basin (like the lazy fixed point), AdamW divides by near-zero
variance and either explodes in noisy dims or stalls in flat dims.
Lion's sign step gives uniform pressure across all dims, more conducive
to escaping the basin.

Hypothesis: **Lion + v4's α=1·η=4 + v2's better data** is the cleanest
test of basin-escape via optimizer dynamics.

Config sketch (v6):

| Lever       | Value                                               |
|-------------|-----------------------------------------------------|
| optimizer   | `lion8bit` (bitsandbytes; no install needed)        |
| lr          | 3e-5 (Lion canonical = AdamW lr × ~1/4)             |
| α           | 1 (revert to v4)                                    |
| η           | 4.0 (revert to v4)                                  |
| LR schedule | cosine→0 over 800 steps (v5 keep)                   |
| Steps       | 800                                                 |
| save_every  | 50                                                  |
| Dataset     | `ai_toolkit_glasses_v2` (60 pairs, demo captions)   |
| Scope       | xattn (FLUX_XATTN_TARGETS)                          |
| EMA         | off                                                 |

What this isolates: **optimizer dynamics**, holding everything else at
known-good v4 values plus v5's data improvements.

## Files

- `output/ai_toolkit_runs/glasses_slider_v5/glasses_slider_v5_000000{050..450}.safetensors`
  — kept for posthoc inspection. Will not be evaluated unless v6 fails.
- `output/ai_toolkit_runs/glasses_slider_v5.log` — full training log
  to step ~450.

## Open questions for after v6

1. If Lion engages glasses by step ~200–400, v5's failure was AdamW's
   basin-stuck behaviour, not the data or scope.
2. If Lion still doesn't engage, the bottleneck is somewhere else
   (data, scope, or concept_slider loss formulation for hard-negative
   concepts on Flux).
3. If Lion engages but bundle remains, the bundle is a property of
   the slider's pos/neg prompt formulation interacting with Flux's
   prior, not an optimizer artefact.
