---
status: live
topic: metrics-and-direction-quality
summary: Current-best plan for the cached-δ injection library. Hybrid of NMF
  components (for clean multi-AU patterns) and direct per-blendshape ridge
  vectors (for cases where NMF over-mixes). Grounded in the 2026-04-23 cache
  recon, fidelity, regression, component, and side-by-side analyses.
---

# Cached-δ injection library — current best plan

This is the current-best summary after one day of recon on the compact
attention cache. It is the thing to build next.

## One-line result

A hybrid library of **~8 NMF AU-pattern vectors plus a handful of
direct-blendshape vectors**, each tied to one `(step, block)` peak, is the
right shape for a cached-δ injection store — not per-axis and not per-blendshape.
The 52-d ARKit blendshape space is ~8-d in disguise (NMF recon R² = 0.913), and
per-blendshape ridge ties or slightly beats NMF-component ridge on most AUs.
The hybrid lets us name clean patterns (smile, brow_lift, pucker) while
keeping finer knobs (eyeSquintLeft, jawOpen) available where NMF's mixing hurts.

## How each recon step built the plan

- **Recon A** — `delta_mix` peaks at `single_34` for every axis; only step index
  shifts. One load-bearing block.
- **Recon B** — at `single_34`, emotion axes (anger/surprise/disgust/pucker/lip_press)
  share cos ≥ 0.97 and diverge from the smile cluster at cos ≤ 0.62. Two clusters,
  not eight axes.
- **Recon C** — smile direction is portable across five of six bases (mean
  off-diag cos 0.887); `black_f` is the outlier that justifies the dictionary's
  `(axis, base)` keying on that base specifically.
- **Recon D** — effective dimensionality at peak block is ev1 ≈ 0.66–0.95, k80
  ≤ 4 for every axis. The axis subspace is small.
- **Injection fidelity** — per-channel mean-only reconstruction captures 0.72
  median for the smile cluster and only 0.07–0.10 for the emotion cluster; k=1–3
  PCs recover the emotion signal to 0.56–0.70. Emotion axes need a basis, not a
  single vector.
- **Blendshape regression (per-AU)** — 3072-d channel vector predicts ARKit
  blendshape scores with R² > 0.85 for two dozen AUs; every top-AU's weight
  vector is nearly orthogonal to the axis mean. Axis-mean carries magnitude;
  AU-specific direction is elsewhere in the subspace.
- **Blendshape components (PCA + ICA)** — 8 components explain >90% of the 52-d
  blendshape space. Components correspond to brow_lift, brow_furrow, smile,
  pucker, jaw_open, eye_squint, gaze_lateral, eye_activity. Attention channels
  predict these cleanly (PC1 brow_lift CV R² 0.68–0.96 across every corpus).
- **NMF on blendshapes** — non-negative parts-based basis, recon R² 0.913 at
  k=8, cleaner separation than PCA (e.g. pure smile vs. upper-lip raise).
- **Side-by-side (per-BS vs component)** — Δ = per-BS R² − per-component R² is
  usually near zero. Large positive Δ shows up on C7 eye_squint (NMF mixes in
  lateral gaze) and C0 jaw_open (NMF mixes in lower-lip). Otherwise tied.

## The hybrid library

A flat `{name → (direction, (step, block), R², provenance)}` dict. Entry list:

| name                 | source              | rationale |
|---|---|---|
| `pattern_brow_lift`  | NMF C2              | component = dominant BS within 0.02 R² everywhere it matters |
| `pattern_brow_furrow`| NMF C3              | same |
| `pattern_pucker`     | NMF C4              | component slightly cleaner than pure `mouthPucker` |
| `pattern_smile`      | NMF C6              | smile corpora R² tied (0.81) |
| `pattern_eye_activity` | NMF C1            | universal R² > 0.6 and clean |
| `bs_eyeSquintLeft`   | direct ridge on eyeSquintLeft | beats C7 by +0.62 on disgust, +0.10 elsewhere |
| `bs_jawOpen`         | direct ridge on jawOpen | beats C0 by +0.45 on anger, +0.38 on pucker |
| `bs_mouthFunnel`     | direct ridge on mouthFunnel | not a clean component; only fits via direct |
| `bs_mouthPressLeft`  | direct ridge       | appears in C3 accessories but needs its own knob |

Each entry also carries:

- `(step, block)` flat site index (peak from the source corpus).
- CV R² on the best source corpus.
- `provenance`: `{source: "NMF_C6" | "direct_bs", corpus: "smile_inphase", ...}`.
- Unit-normalised direction vector. Scale is chosen at inject time.

## What this is NOT

- Not a per-axis library keyed by `smile / anger / disgust / ...`. That keying
  loses information — anger-corpus renders are dominated by brow activity, not
  by something called "anger".
- Not a 52-entry per-blendshape store. That overweights the 52-d output basis
  when the true rank is ~8 + a handful of finer AUs.
- Not a full-attention-tensor cache. Our existing cache is channel-mean
  summaries (3072-d per block); this library stores per-channel shifts only.
  Whether those shifts reproduce prompt-pair pixel edits is still unproven —
  that is what the next smoke render tests.

## Open questions before this becomes load-bearing

1. **Does a per-channel shift move pixels at all?** Library vectors are per-channel;
   prompt-pair edits write the whole (tokens, 3072) tensor. Smoke-test answer
   required before committing.
2. **Is a single-site injection enough?** K=1 peak block (`single_34`) vs
   K=5–24 sites across (step, block) grid. The existing NMF-atom injection
   used K=24 sites and visually failed for `directions_resid_causal.npz` on a
   different direction family; the mechanism is known-working with enough sites.
3. **Does `ab_half_diff` give us the emotion-specific structure delta_mix
   collapsed?** Reading from the drive in the background; rerun the same
   component pipeline when it lands.
4. **Noise-free AU coverage is uneven.** `noseSneer*`, `cheekPuff`,
   `cheekSquint*` never fire in the corpus — MediaPipe zero every sample. The
   library cannot cover AUs the corpus doesn't exercise; those need targeted
   pair renders.

## Files produced this round

- `models/blendshape_nmf/au_library.npz` — k=8 NMF basis + per-(component, corpus) ridge `W`.
- `output/demographic_pc/blendshape_vs_nmf_r2.csv` — full 52-row side-by-side.
- `docs/research/2026-04-23-cache-recon.md` — probes A–D.
- `docs/research/2026-04-23-injection-fidelity.md` — mean/basis reconstruction.
- `docs/research/2026-04-23-blendshape-regression.md` — per-blendshape R².
- `docs/research/2026-04-23-blendshape-components.md` — PCA + ICA components.
- `docs/research/2026-04-23-au-library.md` — NMF library summary.
- `docs/research/2026-04-23-blendshape-vs-nmf-r2.md` — per-BS vs NMF side-by-side.
- **this file** — the current-best plan.
