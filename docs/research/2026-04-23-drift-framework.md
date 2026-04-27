---
status: live
topic: metrics-and-direction-quality
summary: Formalizes "edit-induced attribute drift" as a measurement problem — each edit direction moves the target attribute AND pulls entangled dimensions (age, identity, hair, beard, eye closure). Plan to measure with already-vendored models, build a drift index, iterate mitigation at three levels of cost.
---

# Edit drift as a measurable attribute-space shift

## The observation that motivates this

v2 smile calibration (faint rung, scale=0.7) aimed to hit
`mouthSmileLeft ≈ 0.7`. 2/12 hit; the 10 misses all overshot. More
importantly, visual and channel-level inspection showed the smile edit
dragged *other* attributes with it:

- `young_european_f` — eyes closed almost fully (ΔeyeBlink = +0.63),
  age appears reduced
- `southasian_f` — age drops noticeably, hair stays put
- `elderly_latin_m` — beard density shifts with the smile
- all bases — eye-squint or eye-blink drift correlated with smile
  magnitude, ratio 0.2–0.75 per unit smile (ratio is base-specific)

This is edit-induced attribute drift: the smile direction in attention
space is not orthogonal to age / identity / eye-closure / beard. Any
edit-based generation pipeline has to reckon with it.

## The formalization

For a render `r` produced by applying edit `e` at scale `s` on base `b`
and seed `k`, define:

```
drift[e, b, s, k, attr] = attr(r) − attr(control[b, k])
```

where `control[b, k]` is the unedited (scale=0) render at the same base
and seed, and `attr` ranges over:

- **MediaPipe channels** (52): mouthSmile*, eyeBlink*, eyeSquint*, etc.
- **Residualised atoms** (20): atom_NN post per-base z-score
- **Identity embedding**: ArcFace IR101 512-d vector → drift reported
  as `1 − cosine(edit, control)`
- **Age prediction**: MiVOLO (primary), DEX (ensemble) — signed years
- **Race / gender attributes**: FairFace 7-way race + binary gender +
  9-bucket age, reported as KL-divergence or argmax flip
- **CLIP probes**: cosine similarity to a small bank of short text
  prompts — `bearded`, `clean-shaven`, `long_hair`, `short_hair`,
  `wrinkled`, `glasses`, `open_mouth`, etc. Extensible without
  training new models.

`drift[e, b, s, k, attr]` per (e, b, s) averaged over seeds gives the
**drift matrix**, shape `(axis × base × scale × n_attrs)`. Columns are
attributes; rows index into the vocabulary of edits.

## What we already have

Vendored under `vamp-interface/vendor/`:

- ArcFace IR101 (memory `project_vamp_measured_baseline.md` confirms
  choice of IR101 over FaceNet)
- MiVOLO (age + gender joint)
- pytorch-DEX (age-only, different training — ensemble partner)
- FairFace (race / gender / age bucket classifier)

What's missing and cheap to add: CLIP probes using `open_clip` and a
short-prompt bank. No training required.

## Three levels of mitigation

**L1 — preservation clauses in the splice prompt.** Cheapest and
often enough. Instead of `"{demo} with a faint closed-mouth smile"`,
use `"{demo} with a faint closed-mouth smile, eyes open, same age, same
hairstyle, same beard"`. Flux generally respects negative qualifiers
appended to splice prompts. Measure drift after; if it shrinks, done.

**L2 — drift-counter-edit composition.** For attributes that persist
after L1, derive a counter-edit from the drift matrix: `counter =
mean(delta_mix) at cells where attr drift was observed, flipped`. Feed
a multi-edit FluxSpaceEdit variant that averages primary + counter
attention caches. Formally a small linear system — pick mix weights
that zero all but the target attribute.

**L3 — closed-loop per-render adjustment.** Render, measure drift, if
outside tolerance render again with adjusted prompt / scale / counter
weights. Expensive, use only for tight-tolerance demos.

Genetic / gradient-descent search over prompt variations feels like
overkill for a ~10-dimensional drift space. L1 + L2 cover it with
analytic tools.

## Build plan

1. **`score_drift.py`** — per-PNG enrichment: ArcFace vec, MiVOLO age,
   DEX age, FairFace race/gender/age, CLIP probe bank. Verifies
   vendored weights load at startup; fails fast if missing.
2. **`build_drift_index.py`** — one-pass over existing scored PNGs,
   adds drift columns to `sample_index.parquet`. Compression keeps
   parquet size bounded (~5 MB total with the 512-d ArcFace vector).
3. **`analyze_drift.py`** — reads extended parquet, computes per-(axis,
   base, scale) drift vs scale=0 control. Reports the drift matrix and
   top offenders.
4. **`compose_preserving_prompt.py`** — L1 implementation: takes a
   primary edit prompt + a preservation list, emits a splice prompt
   with the preservation clauses. Simple template for now.

Tomorrow's work. Tonight's render batch is designed to give the
drift-scorer dense paired (edit, control) samples across more axes and
seeds so the drift matrix has statistical power from day one.

## Concrete evidence we have already

From `output/demographic_pc/smile_calibration_v2/blendshapes.json`:

| base | Δsmile | ΔeyeBlink | ΔeyeSquint | blink-per-smile ratio |
|---|---|---|---|---|
| young_european_f | +0.84 | **+0.63** | +0.41 | 0.75 |
| southasian_f | +0.66 | +0.36 | +0.35 | 0.55 |
| black_f | +0.94 | +0.35 | +0.27 | 0.37 |
| elderly_latin_m | +0.56 | +0.24 | +0.06 | 0.43 |
| european_m | +0.92 | +0.16 | +0.02 | 0.17 |
| asian_m | +0.59 | +0.12 | +0.13 | 0.20 |

Per-unit-smile eye-closure drift varies **4×** across bases. Demographic
drift has clear base structure — will show up in the drift matrix as
coherent patterns.

## Why this matters

Until the drift matrix exists, every generation pipeline is subject to
silent attribute pull. With the matrix, any target `(atom, base,
scale)` can be annotated with expected collateral damage, and the L1 /
L2 mitigation path is explicit rather than heuristic.

The atom + base + age + gender vocabulary we've been building is
exactly the substrate the drift framework lives on — atoms are the
*intended* axes, drift columns are the *unintended* ones.
