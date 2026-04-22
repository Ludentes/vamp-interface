---
status: live
topic: manifold-geometry
---

# Session synthesis — 2026-04-22

**What this doc is:** a compact record of what we learned today across
three threads (paper reading, hypothesis falsification, intensity-curve
characterisation) so we don't have to reconstruct the day on the next
session. Chronological-ish, with the load-bearing conclusions at the
end.

## Where we started the day

- Mona-Lisa → Joker α-sweep (completed 2026-04-21) showed a
  step-function cliff in `jawOpen` at α≈0.45 and non-monotonic
  `mouthSmile` in every 1 of 60 trajectories.
- Manifold-theory survey (`Manifold-Research.md`) pointed at Fisher
  metric / RJF / String Method as candidate explanations.
- Open question: combine ridge + FluxSpace using Riemannian geometry
  to overcome their respective failure modes.

## What we did

### 1. Documentation reorganisation

Built `docs/research/_topics/` as a living interpretation layer over
the dated research corpus. Four topic indexes:
`manifold-geometry`, `metrics-and-direction-quality`,
`demographic-pc-pipeline`, `archived-threads`. Each points at the
load-bearing dated docs in priority order with a current-belief
summary. `MEMORY.md` and `CLAUDE.md` were wired to read these first
on every session.

Added `docs/papers/` with four paper PDFs and a README index of
load-bearing claims per paper.

Instituted two-strikes frontmatter rule: when editing a docs/research
or docs/blog doc without frontmatter, add it. Project-local Haiku
agent (`.claude/agents/frontmatter-tagger.md`) generates the block.

### 2. Paper distillation (Hessian-Geometry, RJF, Diffusion String Method)

Read three manifold-theory papers against our specific open questions
— not as generic summaries. Verdicts (detail in
`2026-04-22-manifold-papers-distillation.md`):

- **Hessian-Geometry** (Lobashev et al., ICML 2025) — Proposition 4.1
  is a toy model of exactly the α≈0.45 cliff: diverging Lyapunov
  exponent at a bimodal phase boundary. **But** validated only on 2D
  latent slices of SD 1.5 / Ising / TASEP, not DiT/flow-matching.
  Fisher metric is Hessian of fit log-Z (not sample Σ, not Jacobian
  of generator). Requires 2D slice — not defined on our 1D `mix_b`.
- **RJF** (Kumar & Patel) — training-time only, requires analytical
  hypersphere manifold. **Does not apply to test-time FluxSpace
  editing.** Downgraded.
- **Diffusion String Method** (Moreau et al.) — Fig. 4 directly shows
  non-monotonic likelihood along linear-initialised strings in
  SiT-XL VAE latent. Closest phenomenological match. Two gaps:
  operates on state space not attention cache; requires score +
  velocity (Flux gives only velocity).

Marked `Manifold-Research.md` as superseded — the survey oversold
RJF and Hessian-Geometry applicability; only the String-Method
reference cleanly matches what we observed.

### 3. Reframing: "combine ridge + FluxSpace"

User pushed back on treating Riemannian framing as optional:
understanding the geometry is key, and the real deliverable is a
bridge to the blendshape ecosystem (ARKit → FLAME → MetaHuman
→ FACS). Blendshape-aligned FluxSpace directions become the common
currency.

Reframed the plan: decompose 52-channel blendshape measurements via
PCA→ICA into ~25 AU-like axes; ridge-fit attention-cache features
per axis; test per-axis linearity; apply Riemann correction only
where individual axes fail linearity.

Documented in `2026-04-22-blendshape-bridge-plan.md` with five
phases, stopping conditions, and downstream payoff.

### 4. Phase-1 falsification — cross-phase vs in-phase sweeps

Scored two control datasets on existing attention-sweep renders:

- `smile_inphase/` (330 PNGs): two AU12-dominant prompts.
- `jaw_inphase/` (330 PNGs): two AU26-dominant prompts.

Compared against the cross-phase Mona→Joker sweep
(`alpha_interp/`, 660). Full writeup in
`2026-04-22-inphase-monotonicity.md`. **Result: the α≈0.45 cliff
appears in ALL THREE sweeps at the same α-location, including pure
single-AU in-phase sweeps.** Absolute step amplitudes:

- smile_inphase mouthSmile: +0.41 from α=0.4 to α=0.5.
- jaw_inphase jawOpen: +0.44.
- alpha_interp jawOpen: +0.25.

The cliff is a **FluxSpace `mix_b` injection threshold**, not a
mixture-of-AUs phenomenon. Simple mixture-hypothesis falsified.

Also noted: the original α-interp writeup overstated the
`mouthSmile` non-monotonicity. The dominant signal is the
injection-threshold jump in `jawOpen`; the smile reversal is a
1.5% ripple on a 90%-baseline, not a large concave dome.

### 5. Phase-4 (redirected) — intensity-curve characterisation

Scored `intensity_full/` (340 of 504 — 32% face-detection failures
concentrated at scale ≥ 1.4) and analysed scale-sweeps at fixed
`mix_b=0.5`. Writeup: `2026-04-22-intensity-linearity.md`.

Result: even **above the injection threshold, scale is nonlinear**.
78 trajectories classify: 13% linear (Class A), 28% saturating
monotonic (Class B), **59% non-monotonic (Class C)**. The dominant
non-linearity is:

- **Saturation sigmoid in scale** — most prompts saturate from
  near-zero to ~0.9 within one or two scale steps.
- **Per-base baseline shift** — `european_m` scores mouthSmile=0.47
  at scale=0.2 on `01_faint`; `asian_m` scores 0.07 on the same
  direction. Not base-invariant.
- **Collapse at scale ≥ 1.4** — blendshape scores crash even where
  face detection survives.

Cross-axis leakage is real and prompt-dependent: `01_faint` leaks
3%; `03_broad` leaks 19% onto stretch (compound prompt bundling
AU12+AU25+AU26).

## Revised picture

**Non-linearities in FluxSpace editing are three stacked engineering
curves, not manifold curvature:**

1. Injection threshold in `mix_b` at ≈0.45 (sigmoid).
2. Saturation curve in `scale` with narrow linear window ~[0.2, 0.5].
3. Per-base baseline shift (~0.4 mouthSmile delta across bases).

Plus collapse cliff at `scale ≥ 1.4`.

**Implications:**

- Riemann thread downgraded twice: first by paper reading (none
  directly applicable), then by Phase-1 falsification (cliff is
  engineering, not geometry). Would only become relevant if residual
  non-linearity survives after correcting the three curves.
- The blendshape bridge goal still stands, but Phase 4 must use
  baseline correction + low-scale ridge fits + engineered-single-AU
  prompt pairs rather than coarse compound phrases.
- PCA→ICA on measured blendshape coefficients (~1600 available) is
  the right next step — it grounds the "measure leakage" claim in a
  data-driven basis and may reveal which AUs are intrinsically
  coupled in Flux's output distribution.

## What's on disk as of end-of-session

Research docs (all with frontmatter, all under `_topics/manifold-geometry`):
- `2026-04-22-manifold-papers-distillation.md`
- `2026-04-22-blendshape-bridge-plan.md`
- `2026-04-22-inphase-monotonicity.md`
- `2026-04-22-intensity-linearity.md`

Papers in `docs/papers/`:
- `fluxspace-2412.09611.pdf`
- `hessian-geometry-2506.10632.pdf`
- `learning-on-manifold-rjf-2602.10099.pdf`
- `diffusion-string-method-2602.22122.pdf`

Scripts:
- `analyze_inphase_monotonicity.py` (Phase 1)
- `analyze_intensity_linearity.py` (Phase 4 redirected)

Scored corpora (blendshapes.json):
- `smile_inphase/` (330)
- `jaw_inphase/` (330)
- `intensity_full/` (340 of 504)
- `alpha_interp/` (660, pre-existing)
- `bootstrap_v1/` (288, pre-existing)

Total scored vectors: ~1948. Minus duplicates and per-base variance,
plenty for a 52-dim ICA fit.

## Open questions at end of session

1. **Will PCA→ICA on ~1948 blendshape measurements recover a clean
   AU basis?** Unknown until run.
2. **Which ICA axes are intrinsically coupled vs separable in Flux's
   output distribution?** The question the decomposition will
   answer — informs which "single-AU prompts" are even achievable.
3. **Does the injection threshold location move with start_pct or
   DiT-block subset?** Deferred — not urgent for the bridge.
4. **Could a within-identity Σ rehabilitate Mahalanobis for collapse
   prediction?** Still open; orthogonal to today's work.

## Falsification loop for the day

Three hypothesis revisions in sequence:

- Morning: "α≈0.45 cliff = manifold curvature" → survey suggested
  this, papers didn't confirm it.
- Mid-day: "α≈0.45 cliff = mixture of AUs bundled in prompts" →
  falsified by same-phase sweeps showing identical cliff location.
- Afternoon: "above-threshold, scale is linear per-axis" → falsified
  by `intensity_full` analysis showing 59% non-monotonic.

Three disconfirmations in one day is a good loop. Each was cheap
because the data was already on disk; we just had to look.

## Next actions queued

1. Commit today's work (pending now).
2. Fetch 2-4 papers on PCA→ICA of blendshapes / facial expression
   coefficients (agent running in parallel).
3. Read those papers; update the bridge plan's Phase 2 with their
   recommended component count, reconstruction metric, and AU-labelling
   procedure.
4. Run PCA→ICA on the ~1948-vector corpus.
5. Only then: revised Phase 4 — prompt engineering per measured ICA
   axis, baseline-corrected ridge fits, per-axis sigmoid
   characterisation.
