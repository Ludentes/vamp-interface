---
status: live
topic: manifold-geometry
summary: Phase-3-proper fits full-tensor ridge on top-24 (step, block) sites per atom; CV R² 0.82–0.97. Predictive claim holds. Constructive claim (injection steers atom k) is UNVERIFIED on this artefact (`directions_k11.npz`). A related but different artefact (`directions_resid_causal.npz`) failed visual validation on 2026-04-23 — see the linked failure doc. The `directions_k11.npz` file itself has not been visually tested.
---

> **⚠ CAUTION (2026-04-23).** A nearby artefact
> (`directions_resid_causal.npz`, norm ~47 per atom, 20 atoms) was
> visually tested on 2026-04-23 and does NOT produce visible edits
> in any explored scale regime. That is NOT this doc's artefact.
> This doc's `directions_k11.npz` (11 atoms, norm ~0.02 per atom)
> has not yet been visually tested. Any follow-on work on *this*
> file must first run the Option-C gating pilot at scales calibrated
> to the 0.02-norm magnitudes. See
> [2026-04-23-atom-inject-visual-failure.md](2026-04-23-atom-inject-visual-failure.md).


# Phase 3-proper — FluxSpace-usable edit directions per NMF atom

**Date:** 2026-04-22
**Scripts:** `cache_attn_features.py`, `fit_nmf_directions.py`
**Artefacts:** `models/blendshape_nmf/attn_cache/{tag}/{delta_mix,attn_base}.npy`,
`models/blendshape_nmf/directions_k11.npz`

## Pipeline

1. **Cache** `delta_mix.mean_d` (3072-d) and `attn_base.mean_d` per
   (step=16, block=57) per paired render. 1320 renders total (330
   smile_inphase + 330 jaw_inphase + 660 alpha_interp_attn). Cache is
   fp16 to halve disk; ~14 GB total across 3 source directories.
2. **Per-site screening**: for each of 912 (step, block) sites, fit
   one multi-output Ridge on the 3072-d delta_mix vector at that site
   against all 11 atom coefficients jointly. Record per-(site, atom)
   in-sample R². This gives a localisation map.
3. **Top-K site selection**: per atom, pick the K=24 sites with
   highest screening R².
4. **Full fit**: per atom, concatenate `delta_mix.mean_d` at the 24
   sites (24 × 3072 = 73,728 features per sample), ridge-fit on 1320
   samples with 5-fold CV. Output direction matrix (24, 3072).

## Results

| # | AU label | CV R² | In-sample R² | Top-site R² |
|---|----------|------:|-------------:|------------:|
| 0 | AU12+AU10 broad smile | 0.924 | 1.000 | 0.974 |
| 1 | AU1+AU2 brow raise | 0.970 | 1.000 | 0.985 |
| 2 | AU7 lid tighten | 0.941 | 1.000 | 0.983 |
| 3 | AU64+AU45 gaze-down+blink | 0.951 | 1.000 | 0.983 |
| 4 | AU4 brow lower | 0.963 | 1.000 | 0.981 |
| 5 | AU12 pure smile | 0.896 | 1.000 | 0.971 |
| 6 | AU16+AU26 lower lip + jaw | 0.960 | 1.000 | 0.980 |
| 7 | AU26 pure jaw | 0.886 | 1.000 | 0.975 |
| 8 | AU24+AU28 lip press (fragile) | 0.843 | 1.000 | 0.959 |
| 9 | AU61/62 horizontal gaze | 0.961 | 1.000 | 0.981 |
| 10 | AU18 pucker (underrep.) | 0.823 | 1.000 | 0.956 |

**Median CV R² = 0.951**, minimum 0.823. Every atom — including the
ones Phase-3 diagnostic flagged weak — is now a strong, held-out
predictor.

## What Phase-3-proper gives us that Phase-3 diagnostic didn't

- **Full-vector directions.** Each output is a (24, 3072) matrix
  that can be injected at inference time into the specific (step,
  block) locations FluxSpace already supports. The diagnostic's
  scalar features only told us *which* (step, block) mattered; now
  we have the actual direction vectors.
- **~0.10 mean CV R² gain** across atoms vs scalar features. The
  full 3072-d tensor at a single (step, block) carries almost all
  the atom's predictable variance; single-best-site R² reaches 0.96+
  for all atoms.
- **Weakest atoms lift the most.** The scalar-feature diagnostic's
  bottom (atom 10 pucker at R²=0.54) was data-coverage-limited *at
  scalar resolution*. At full-tensor resolution we recover R²=0.82
  even for this under-represented atom. Implication: when training
  data is thin, the information-per-sample matters more than the
  sample count.

## Caveats and honest notes

- **Overfitting at K=24.** In-sample R² = 1.0 for every atom (73k
  features × 1320 samples). CV R² is the honest number; gap is
  0.03–0.18. Biggest gap on the known-weak atoms (8 and 10). A
  K=8–12 sweep or higher ridge alpha would probably close the gap
  further on those two. Not blocking for Phase 4 — the CV numbers
  are the ones that matter for held-out behaviour.
- **Predictive vs constructive asymmetry.** A high CV R² says
  "given an observed delta_mix, we can predict atom k." It does not
  yet say "injecting our fitted direction moves atom k by the
  predicted amount." That second claim only holds to the extent
  that observed delta_mix vectors live on the same affine subspace
  as our fitted direction. The training distribution was
  smile/jaw/Mona-Joker prompt pairs — directions *extrapolated* to
  new prompts (pucker, brow lower) inherit whatever linearity Flux
  has in that region. Phase 4 tests this.
- **Corpus coverage bias.** 660 of 1320 samples are Mona-Joker
  α-sweep; only 330 each for smile_inphase and jaw_inphase. Atoms
  that get large coverage in Mona-Joker (smile, jaw, lip-press,
  pucker) have more training signal than others.

## What to do next

Option A — **Phase 4 on the canonical 10 atoms** (skip atom 8
fragile for now). Render a small single-axis sweep per direction at
scale ∈ {-2, -1, -0.5, 0, 0.5, 1, 2} on 3 bases × 3 seeds = 9
trajectories per atom. That's 10 × 9 × 7 = 630 renders at ~10s each
= ~1.75 hours. Measure atom-k response via MediaPipe; report per-axis
monotonicity and R² against a sigmoid.

Option B — **Rehabilitate the weak atoms first.** Add 100–200
render samples at pucker/lip-press-specific prompt pairs, rebuild
cache increment, refit. ~20 min render + 10 min cache/fit. Then
Phase 4 on all 11.

Option C — **Immediate validation pilot.** Pick one atom (e.g.
atom 5 AU12 pure smile) and render a single sweep immediately to
verify the direction actually moves the atom as predicted before
spending compute on all 10. 9 renders, ~2 min.

My vote: **C first**, then A. If the pilot sweep on atom 5 produces
a clean linear response in mouthSmile measurements, we know the
constructive claim holds and we can run the full Phase 4 in
confidence. If the pilot fails, we save 1.75 hours of compute and
investigate why the fitted direction doesn't translate into actual
edits.

## Pyright

Existing sklearn-stub spurious errors persist. Unused imports and
unused variable warnings (`rep`, `attn_base`, `meta`, `N`, `D`) are
real dead code in the current cut; remove on next touch per the
two-strikes rule.

## Implementation notes

- Cache uses fp16 to fit 14.8 GB into disk. Ridge fits convert back
  to fp32 at site slice time to avoid precision artefacts.
- Per-site screening took ~15 min on CPU. Could be sped up 10× with
  GPU or batch-parallel numpy, but it's a one-time pass.
- Per-atom ridge on 73k features × 1320 samples takes ~8 s; the
  pipeline is bottlenecked by cache I/O (mmap-backed numpy), not
  the fits themselves.
