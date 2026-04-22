## Manifold geometry of Flux edits — current belief

**Status:** live. Last updated 2026-04-22.

### TL;DR

- **FluxSpace's Euclidean linearity assumption holds in
  attention-cache space but fails in decoded image space.** Linear
  `mix_b` interpolation between two endpoint prompts produces
  non-linear blendshape trajectories.
- **The Mona-Lisa→Joker α-sweep has a Lipschitz singularity at
  α≈0.45** — textbook phase transition in `jawOpen`, plus
  non-monotonic `mouthSmile` in all 60 trajectories. This matches
  the Hessian-geometry paper's prediction that generative latent
  spaces are piecewise-regular with sharp geometric transitions.
- **MediaPipe's 52-channel basis is overcomplete.** Effective dim
  ~25-30 at 90-95% variance; recommend PCA→ICA (or SPLOCS) before
  ridge regression on expression targets. Our pilot's ~3 effective
  dims on 5 smile/stretch channels is consistent with bilateral
  coupling.

### Key findings

- **Two-axis structure of "smile intensity."** The Mona-Lisa→Joker
  pair does not trace a single intensity axis — it traces two:
  mouth-aperture (step-function in α) and lip-corner-pull
  (concave, peaking mid-range). Intermediate α-values are their
  own mid-phase species, not "50% of a smile."
- **`max_env` predicts scale-collapse edges from one s=1 render.**
  93%/82% in/out sample on glasses; unconfirmed on other axes at
  the same `T_ratio`.
- **Pair-averaging cancels opposing confounds.** FluxSpace's
  `mix_b=0.5` averages two edit-prompt attention caches, which
  cancels age/race confounds on the glasses axis and generalises
  across 6 demographics.
- **Blendshape effective rank ≈ 25-30** (BFM-2017 29 components,
  FaceWarehouse 25, ARKit ~32 after bilateral pairs). Do not
  ridge-fit all 52 raw channels — they are not independent.

### Theoretical framing (from `docs/papers/`)

- **Hessian Geometry of Latent Space in Generative Models** (Lobashev
  et al., ICML 2025, `hessian-geometry-2506.10632.pdf`). Predicts the
  α≈0.45 phase boundary: Fisher metric has fractal phase transitions,
  geodesics are approximately linear within a phase and break at
  boundaries where the effective Lipschitz constant diverges.
- **Learning on the Manifold / RJF** (Kumar & Patel,
  `learning-on-manifold-rjf-2602.10099.pdf`). Diagnoses
  Euclidean-`mix_b` failures on hyperspherical features as
  geometric interference; proposes Riemannian Flow Matching with
  Jacobi regularisation. Candidate fix for our non-monotonic
  smile trajectories.
- **Probing Diffusion Models with the String Method** (Moreau et al.,
  `diffusion-string-method-2602.22122.pdf`). Minimum-Energy-Path vs
  Principal-Curve distinction; likelihood-vs-realism paradox explains
  why straight attention-cache paths cross low-density regions.
- **FluxSpace** (Dalva et al., CVPR 2025, `fluxspace-2412.09611.pdf`).
  Our working baseline. Pair-averaged attention cache edits on
  joint-attention outputs, scale 0.5–1.0 for stable edits.

### Load-bearing dated docs (in priority order)

1. [2026-04-22 α-interp phase boundary](../2026-04-22-alpha-interp-phase-boundary.md)
   — the Mona-Lisa→Joker finding; core empirical result this thread
   is currently organised around.
2. [2026-04-22 Manifold theory cross-reference](../2026-04-22-manifold-theory-crossref.md)
   — maps each of our empirical findings to a specific theoretical paper.
3. [2026-04-22 Blendshape effective dimensionality](../2026-04-22-blendshape-effective-dimensionality.md)
   — lit review justifying PCA→ICA reduction before ridge.
4. [2026-04-21 FluxSpace synthesis](../2026-04-21-fluxspace-synthesis.md)
   — consolidates what works and why; includes the Mahalanobis
   post-mortem (see also [metrics](metrics-and-direction-quality.md)).
5. [2026-04-21 FluxSpace collapse prediction](../2026-04-21-fluxspace-collapse-prediction.md)
   — `max_env` threshold mechanics.
6. [2026-04-21 FluxSpace crossdemo confirmation](../2026-04-21-fluxspace-crossdemo-confirmation.md)
   — `T_ratio = 1.275` across 6 demographics.
7. [2026-04-21 FluxSpace smile axis](../2026-04-21-fluxspace-smile-axis.md)
   — cross-axis transfer of pair-averaging geometry.
8. [Manifold-Research.md](../Manifold-Research.md) — original survey
   that sourced the Hessian-geometry / RJF / String Method references.

### Open questions (active)

- Can we locate the α≈0.45 phase boundary from attention-cache
  geometry alone, before rendering? Attention-capture α-sweep
  running at time of writing (`alpha_interp_attn/`).
- Does a same-phase endpoint pair (e.g., "faint closed smile" to
  "broad closed smile") produce a monotonic α-sweep, confirming
  that non-monotonicity is a cross-phase phenomenon? Scripted as
  `fluxspace_alpha_inphase.py`.
- Can Riemannian `mix_b` (geodesic instead of Euclidean) remove
  the non-monotonic smile artefact? RJF paper provides the recipe;
  not yet implemented.
- Can simplex mixing over 3 endpoints give independent control
  over mouth-aperture and lip-corner-pull? Not yet attempted.
