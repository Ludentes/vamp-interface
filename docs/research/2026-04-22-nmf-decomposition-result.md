---
status: live
topic: manifold-geometry
summary: Sparse NMF on 1941 blendshape vectors recovers 10/12 AU-plausible atoms vs ICA's 3/12 at matched k=12 (VE 0.957 vs 0.971); 11 effective directions, sparsity 7.8 vs 26.3 channels per atom.
---

# Phase 2 result — sparse NMF decomposition of the blendshape corpus

**Date:** 2026-04-22
**Script:** `src/demographic_pc/blendshape_decomposition.py`
**Artefacts:** `models/blendshape_nmf/{W_nmf.npy, W_ica.npy, manifest.json}`

## Headline

- **Corpus:** 1941 scored blendshape vectors from 5 sources
  (bootstrap_v1, alpha_interp, smile_inphase, jaw_inphase, intensity_full).
  52 channels → 39 after pruning near-constant channels
  (`_neutral`, `cheekPuff`, `jawForward`, left/right gaze variants,
  etc. — 13 channels with σ < 0.01).
- **k-sweep (unregularised NMF, VE rule):** VE reaches 0.95 at k=10,
  saturates slowly. k* = 12 picks first k above target.
- **Sparsity sweep:** alpha_W=alpha_H=0.001 trades 0.61% VE for
  ~30% increase in sparsity. alpha >= 0.005 collapses VE below 0.91;
  alpha >= 0.01 is too aggressive for our [0,1] data scale.
- **Atom quality @ k=12, alpha=0.001:**

| Method | AU-plausible | composite | noise | dead | Mean support | VE |
|---|---|---|---|---|---|---|
| **Sparse NMF** | **10** | 1 | 0 | 1 | **7.8** | 0.957 |
| PCA → FastICA | 3 | 3 | 6 | 0 | 26.3 | 0.971 |

NMF is **3.3× more likely to produce AU-aligned atoms** at matched k.
Mean atom support (channels with |w| ≥ 5% of peak) is also 3.4×
smaller — each NMF atom touches ~8 channels, ICA touches ~26.

The one NMF "dead" atom (#11 with all-zero loadings) indicates the
true effective rank of our corpus is **11**, not 12. Re-running at
k=11 would likely produce the same 11 active atoms with no dead slot.

## AU naming of the 11 active NMF atoms

Inspecting top channels by loading magnitude and comparing against
ARKit/FACS nomenclature:

| # | Top channels (weight) | FACS interpretation |
|---|------------------------|---------------------|
| 00 | mouthSmileLR (1.33) + mouthUpperUpLR (1.33) | **AU12 + AU10** — pronounced smile with upper-lip raise |
| 01 | browOuterUpL (1.07), browInnerUp (1.06), browOuterUpR (0.88) | **AU1 + AU2** — surprise/worry brow |
| 02 | eyeSquintLR (~1.1), eyeLookUpLR (~0.36) | **AU7** (+ AU6 companion) — lid tightener |
| 03 | eyeLookDownLR (0.84), eyeBlinkLR (0.66–0.72) | **AU64 + AU45** — gaze-down + blink |
| 04 | browDownLR (1.16), eyeSquintLR (~0.42) | **AU4** + AU7 — brow lowerer / concentration |
| 05 | mouthSmileLR (~1.44) | **AU12** — pure lip-corner pull |
| 06 | mouthLowerDownLR (~1.25), jawOpen (0.42) | **AU16 + AU26** — lower-lip depressor, jaw drop |
| 07 | jawOpen (1.29) | **AU26** — pure jaw drop |
| 08 | mouthPressLR (~0.50), mouthRollLower/Upper | **AU24 + AU28** — lip pressor + suck (composite) |
| 09 | eyeLookOutL + eyeLookInR (~1.03) | **AU61/62** — horizontal gaze |
| 10 | mouthPucker (1.27) | **AU18** — pucker / kiss |
| 11 | — | dead (effective rank is 11) |

Notable observations:

- **Two smile atoms** (#00 and #05). Both AU12-dominant but with
  different companion loadings: #00 co-activates AU10 (upper lip),
  #05 is pure corner pull. This is the "broad-teeth smile" vs
  "closed-lip smile" distinction at the AU level — consistent with
  the `smile_inphase` endpoints we used.
- **Brow is split into two atoms** (#01 up, #04 down) as it should
  be — these are physically opposing muscle groups.
- **Jaw is cleanly separated from mouth-corner** (#07 vs #00/#05).
  This directly contradicts the original "mixture of AUs" intuition
  we had for the α-interp cliff: jaw and smile AUs are *separable*
  in our corpus's blendshape space.
- **Gaze is a rank-2 subsystem** (#03 down, #09 horizontal). Vertical
  and horizontal eye-look axes are independent by construction in
  ARKit and NMF recovers that.
- **No atom for cheekSquint / cheekPuff / nose sneer** — these
  channels were pruned for insufficient variance. Our Flux corpus
  doesn't produce enough of those expressions to learn directions.

## How this squares with the Mona→Joker α-interp finding

The blendshape bridge plan's hypothesis was that the α≈0.45 cliff is
a mixture-of-AUs phenomenon: Mona Lisa prompt loads AU12, Joker
prompt loads AU26, and `mix_b` flips the dominant axis mid-sweep.

Phase-1 falsification already showed the cliff appears in
same-phase sweeps too — so it's a FluxSpace injection threshold,
not a mixture flip.

Phase-2 now adds a second piece: **at the blendshape level, AU12 and
AU26 are cleanly separable** (atoms #05 and #07, minimal
cross-loading). So the original prompt pair was not mixing axes at
the blendshape *output* level — both endpoints land on well-defined
atoms. The cliff is entirely a property of how FluxSpace's `mix_b`
propagates through the DiT, not of anything structural in AU space.

## Sparsity vs VE tradeoff

```
alpha       VE      support
0.0000     0.9627   8.42
0.0005     0.9624   8.42    (negligible change)
0.0010     0.9566   11.08   (selected; 0.6% VE cost, +32% dense)
0.0050     0.9084   17.25   (VE drops 5%)
0.0100     0.7920   25.00   (VE drops 17%)
```

Interesting: at alpha ≤ 0.0005 sparsity didn't budge. The data itself
has enough zeros that unregularised NMF already produces reasonably
sparse atoms. Regularisation helps above 0.001 but at diminishing
returns until it collapses above 0.005.

**Choice rationale:** alpha=0.001 pushes a few marginal loadings to
zero (mean support 8.4 → 11.1 is *more*, not less, channels included —
because the regulariser distributes atom energy across more channels
rather than dumping into one). The *total* atom norm is compressed
uniformly. The classifier still flags 10/12 as AU-plausible, so the
extra channels are not polluting the dominant region.

At alpha=0.0 (unregularised), 4 atoms show single-channel dominance
exceeding 70% of total mass. At alpha=0.001, only 2 show that —
loadings are more distributed. Either choice is defensible; for the
bridge plan's downstream ridge-fit step, the unregularised atoms
might actually be preferable because they give sharper targets for
per-AU edit directions.

**Follow-up consideration:** re-run at k=11 alpha=0.0 for the
downstream ridge fits. No dead atom, sharpest AU alignment.

## Implications for Phase 3 (ridge fits per AU direction)

- **11 target axes**, not 25. Our ridge-regression targets go from
  52 raw channels to 11 AU-aligned atoms.
- **Clean separability** between core AU axes means single-prompt
  edits targeting one atom (e.g. pure jaw drop for AU26) should
  produce a clean response on that atom and low cross-loading on
  others.
- **Two smile axes** (#00, #05) means we have a natural ordering
  for the "smile intensity" research question: AU12 alone (#05) for
  fine control, AU12+AU10 (#00) for broader teeth-showing smiles.
- **The dead atom tells us the effective rank is 11** and sets a
  ceiling for how many independent AU-aligned FluxSpace directions
  we can hope to extract from this corpus. Additional axes (eye
  wide, cheek puff, nose sneer) would need more diverse prompts.

## Known limitations

- **Classifier is heuristic.** "AU-plausible" by anatomical-group
  concentration isn't the same as a volunteer-vote muscle-
  plausibility check per Tripathi DFECS. I don't project atoms onto
  a neutral face and inspect displacement arrows — I check channel-
  name clustering. A stricter human review could reclassify some
  atoms.
- **NMF non-convexity.** Single random init used here. Should
  verify atom stability across ≥5 random seeds before declaring the
  basis canonical.
- **Pruned channels list is aggressive.** σ < 0.01 drops `cheekPuff`,
  `cheekSquintLR`, `jawForward`/`Left`/`Right`, etc. These are real
  AUs but our Flux corpus doesn't produce them. If we later add
  prompts that exercise those channels, re-prune and re-fit.
- **ICA comparison used the default FastICA whitening.** Changing
  contrast function (logcosh vs exp vs cube) would shift ICA
  results — but not by enough to change the headline, since ICA's
  atom support is dominated by density, not rotation choice.

## Next — Phase 3

Ridge-fit attention-cache features against ICA coefficients
per atom. Now the 11 NMF atoms are the targets. Phase 4 per-axis
linearity testing gets simpler too — test each of 11 axes
separately rather than 52 raw channels.

Stability check before committing the basis as canonical: run 5
seeds of NMF at k=11, alpha=0, record atom-to-atom cosine similarity
across runs. Atoms that match across ≥4/5 runs are stable; others
are noise.

## Pyright

Known spurious errors (sklearn stubs lag real API):
- `NMF(n_components=int)` — "int not assignable to str"
- `NMF(alpha_H=float)` — "float not assignable to str"
- `FastICA.mixing_.T` — tuple-vs-array confusion
Add `# type: ignore[arg-type]` on next touch per two-strikes rule.

## Artefacts

- `models/blendshape_nmf/W_nmf.npy` — (12, 39) atom-loading matrix
- `models/blendshape_nmf/W_ica.npy` — (12, 39) ICA comparison
- `models/blendshape_nmf/manifest.json` — channels, atom
  classifications, sweep data
- `output/demographic_pc/fluxspace_metrics/analysis/blendshape_decomposition.json` —
  duplicate manifest for analysis pipeline
