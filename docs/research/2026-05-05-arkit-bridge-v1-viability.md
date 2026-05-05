---
status: live
topic: neural-deformation-control
---

# ARKit → PersonaLive bridge v1 — viability criteria

Companion to [`2026-05-05-arkit-bridge-v1-design.md`](2026-05-05-arkit-bridge-v1-design.md)
and [`2026-05-05-arkit-bridge-v1-plan.md`](2026-05-05-arkit-bridge-v1-plan.md).
Locks in what "good enough to ship" means *before* the real-corpus training
run in Task 8 burns 30–60 min, and what would constitute architectural
failure vs corpus-coverage failure.

## What "viable" means here

The bridge replaces real `motion_encoder(face_crop_224)` with student
`S(b_expr)` at deployment. "Viable" = the downstream PersonaLive pipeline
produces output close enough to the RGB-driven baseline that the loss is
acceptable for the use case (selfie-style VTuber with iPhone driving). That
collapses to four nested tiers, evaluated cheapest first; we abort at the
first failure rather than burning further compute.

## Tier 1 — distill fidelity

Held-out (8th take or 10% random hold-out — TBD by Task 8 corpus split)
agreement with the frozen teacher.

| Metric | Threshold | Note |
|---|---|---|
| Aggregate `MSE / Var(m_f_teacher)` | < 0.10 mean | PE alone contributes ~0.85 of teacher var; sub-0.10 means we got most of the *expression* signal on top |
| Per-cell R² mask: ≥ 0.7 | ≥ 80% of (32, 16) = 512 output cells | Cells with R² < 0.3 are likely collapsed-to-mean; tag as do-not-trust rather than abort |
| Sensitivity sweep | top-K driving channels (mouthSmile*, jawOpen, brow*) elicit response in trustworthy cells | If high-load ARKit channels only move untrustworthy cells, the network learned but on the wrong axes |

Tier 1 is a learnability check. **Failing it is an architectural signal**
(MLP capacity / loss shape) rather than a data signal.

## Tier 2 — input-category coverage

Stratify the held-out set into ARKit-input categories. Compute Tier-1 ratio
*per category*. Any category with ≥ 100 frames must pass; categories under
the floor are flagged as "out of scope for v1" rather than failing the run.

| Category | Selector (per-frame) | Min frames |
|---|---|---|
| Neutral / micro | sum\|b[0:52]\| < 1.5 | 200 |
| Broad smile | mouthSmileLeft + mouthSmileRight > 1.0 | 200 |
| Speech / mouth open | jawOpen > 0.3 | 200 |
| Brow raise | browInnerUp > 0.4 OR browOuterUpLeft+Right > 0.6 | 100 |
| Squint / blink | max(eyeBlink*, eyeSquint*) > 0.4 | 100 |
| Asymmetric smile | \|mouthSmileLeft − mouthSmileRight\| > 0.2 | 100 |
| Eye gaze | max(\|eye yaw\|, \|eye pitch\|) > 0.2 rad | 100 |

Per-category threshold: ratio < 0.15 (slight slack vs aggregate).
**Failing a category with ≥ 100 frames is a corpus-quality signal**, not
architecture: record more takes for that category and retrain.

Frame counts under the floor (e.g., tongue-out, deep cheek-puff) become
explicit v1 limitations in the readout, not gates.

## Tier 3 — drop-in delta on a held-out take

The actual question. Pick one held-out take. Run PersonaLive end-to-end
twice with the same anchor:

1. (a) real `motion_encoder(face_crop_224)` per frame — RGB-driven baseline.
2. (b) student `S(b_expr)` per frame — bridge.

Per-frame metrics:

| Metric | Threshold | Reference |
|---|---|---|
| ArcFace cosine to anchor (median) | within 0.03 of (a)'s median; never below buffalo_l τ=0.40 | 2026-05-04-personalive-default-decision.md baseline floor 0.43 |
| Smile macro-correlation with driver's mouthSmile* | ≥ 0.70 | (a) gave 0.81 on the 3×6 grid |
| LPIPS((a), (b)) per frame | median < 0.10, p95 < 0.20 | qualitative ceiling |

Tier-3 failure splits two ways:
- LPIPS / smile-corr fail despite Tier 1+2 pass → **the MSE objective is
  the problem**. Switch to a diffusion-loss fine-tune (Task 8 follow-on),
  not architectural changes to the student.
- ArcFace cosine drops below floor → likely the closed-form keypoint path
  (Task 9 calibration) is wrong, *not* the student. Investigate before
  blaming `S`.

## Tier 4 — subjective viability gate

Render a 30-second held-out clip side-by-side with the RGB-driven version.
Reads "same character, slightly less expressive": **ship**. Reads "uncanny"
or "different person": fail; investigate.

This tier is judged informally against the 2026-05-04 PersonaLive default
decision baseline. It exists because metrics under-weight category drift
(a smile that's the wrong *kind* of smile reads as wrong even if smile-corr
is fine).

## Decision matrix

| Outcome | Action |
|---|---|
| Tier 1 ✓, Tier 2 ✓ in ≥ 5 categories, Tier 3 ✓, Tier 4 ✓ | **Ship as v1** |
| Tier 1 ✓, Tier 2 fails 1–2 categories with ≥ 100 frames | Record more takes for those categories; retrain |
| Tier 1 ✓, Tier 2 ✓, Tier 3 LPIPS p95 > 0.20 *or* smile-corr < 0.50 | **Diffusion-loss fine-tune** (replace MSE with sampling-loss against the appearance UNet); don't touch architecture |
| Tier 1 ✓, Tier 3 ArcFace breaks floor | Closed-form pose (Task 9) calibration issue, not student; widen Euler search before blaming S |
| Tier 1 fails (ratio > 0.10) | **Architectural escalation**: try wider MLP, residual blocks, or a small per-frame transformer over the 58-dim input. Not a data problem — 16,500 pairs is plenty for ~280K params |

## What this implies for Task 7 and Task 8

Task 7 (per-channel sensitivity eval) must produce, in one JSON readout per
checkpoint:

- `held_out_ratio_mean` and `held_out_ratio_p95`
- `per_cell_r2` of shape (32, 16), plus `r2_above_0.7_fraction`
- `category_mse` dict (one entry per category in Tier 2 above), plus
  per-category frame counts
- existing per-channel sensitivity sweep (kept as is — useful for Tier 1
  trustworthiness diagnostic)

Task 8 (real distill) consumes those same readouts every N steps for
early-stop / regression detection rather than running blind to step budget.

## Open thresholds (lock before Task 8)

- 0.10 aggregate ratio — derived from "PE explains ~0.85 of var, want most
  of the rest." Sanity-check empirically on Task 8's first 5k steps; loosen
  to 0.15 only if PE dominates more than expected.
- 0.7 R² cell threshold — chosen so 80% × 512 ≈ 410 trustworthy cells gives
  cross-attention enough signal. Worth re-examining post-Tier-3.
- 0.70 smile-corr — softer than (a)'s 0.81 to allow for student noise.
- 0.10 LPIPS median — empirically the boundary above which differences
  read as *not the same actor*; tune after first Tier-3 run.

These are starting points. Tier-3 outputs let us re-derive thresholds
post-hoc if any feel mis-calibrated; the doc gets updated then.
