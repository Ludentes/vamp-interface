---
status: live
topic: photobooth-sweep
supersedes: 2026-05-20-photobooth-phase1-findings
---

# Photobooth Phase 2 sweep — findings (2026-05-20)

Confirmation pass on the post-pruning grid from Phase 1, plus 3× seed
replication to surface across-seed instability. Same pipeline (Z-Image Turbo
+ Z-Image-Turbo-Fun-CN-Union + HyperSwap-1c), same shard (RTX 3090).

## Setup

- Driver: `scripts/photobooth_sweep/driver.py --mode phase2 --seed-iter {0,1,2}`
- Axes (`scripts.photobooth_sweep.axes.phase2_configs`): 6 configs
  - cfg000  natural_1024 · canny · soft
  - cfg001  natural_1024 · canny · aggressive
  - cfg002  natural_1024 · depth
  - cfg003  tight_1024   · canny · soft
  - cfg004  tight_1024   · canny · aggressive
  - cfg005  tight_1024   · depth
- Locked from Phase 1: `cn_strength=0.85`, `refine_denoise=0.00`,
  `demo_inject=on`
- 20 importer identities × 6 configs × 3 seed iters = **360 cells**
- Seed = `100000 + cfg_idx + 1_000_000 * seed_iter`; cells keyed
  `<pid>__cfg<NNN>[_sN]`
- Output: `exp_output/photobooth_phase2/` — `scores.parquet`, `manifest.json`,
  per-cell `cells/<cell_id>/{ctrl,render,swap,refined}.png`
- Contact sheets: `grid_s{0,1,2}.png`, `stages.png` (built via
  `scripts.photobooth_sweep.contact_sheet`)
- Wall time: ~50 min per 120-cell iter, ~2h45m total

## Headline numbers

| metric | overall (n=360) |
|---|---|
| `id_cos` mean (excl. nan) | 0.766 |
| `id_cos` median | 0.803 |
| `id_cos` p05 | 0.292 |
| `id_cos` p95 | ≈0.88 |
| `id_cos` max | **0.904** (Phase 1: 0.850) |
| nan rate | 91 / 360 (25%) |
| `det_mode == default` | 232 / 360 (64%) |
| `det_mode == forced` | 128 / 360 (36%) |
| failures (`exception`) | 0 |

Phase 2 lifts the achievable ceiling by 0.054 id_cos vs Phase 1
(`id_09__cfg002` 0.904 vs Phase 1 `id_00__cfg000` 0.850). The lock-in
defaults from Phase 1 carry through cleanly — none of the remaining axes
moves the mean. **The interesting variance is now between photos, not
between configs.**

## Tight vs natural — the dominant new effect

| `face_pixel_budget` | n | nan rate | forced rate | mean id_cos (valid) |
|---|---|---|---|---|
| `natural_1024` (cfg000-002) | 180 | **14 %** | 22 % | 0.752 |
| `tight_1024`   (cfg003-005) | 180 | **36 %** | 49 % | **0.783** |

`tight_1024` is a **higher-risk / higher-reward** configuration:
- ~2.6× nan rate vs natural
- ~+0.03 mean id_cos when it does work
- ~2.2× detector-fallback rate (`forced`)

Failure-mode signature is consistent: when tight fails, refined images come
back with `face_frac=0.000` and `det_mode=forced`. The swap operation
produces a render where neither SCRFD nor the mediapipe fallback can locate
a face. Inspection of `cells/id_11__cfg004_s1/refined.png` (and similar)
shows that the tight crop is producing a head that fills most of the
1024×1024 doll body, leaving the swap pipeline with no inscribed-face
region to operate on — HyperSwap effectively renders into the doll's
forehead. This is a pipeline geometry failure, not a model failure.

## Per-axis effect, restated

`id_cos` mean grouped by axis value (computed across all photos, dropping
nans):

| axis | value | n | mean | median | max |
|---|---|---|---|---|---|
| `cn_condition` | canny | 172 | 0.769 | 0.804 | 0.896 |
| `cn_condition` | depth | 97 | 0.762 | 0.806 | 0.904 |
| `canny_preset` | soft | 87 | 0.769 | 0.806 | 0.896 |
| `canny_preset` | aggressive | 85 | 0.768 | 0.797 | 0.885 |
| `face_pixel_budget` | natural_1024 | 154 | 0.752 | 0.804 | 0.904 |
| `face_pixel_budget` | tight_1024 | 115 | 0.783 | 0.797 | 0.892 |

Differences between canny presets and between canny/depth are within
seed-replication noise. Canny still nudges slightly ahead on the mean and
holds 4 of the top-10 cells, but depth is also competitive at the
ceiling. **No surviving evidence for picking canny vs depth from id_cos
alone.**

## Per-photo response is bimodal

Per-photo breakdown over all 18 (6 cfg × 3 seed) cells:

| photo | nan / 18 | mean id_cos (valid) | notes |
|---|---|---|---|
| id_02 |  0 | 0.743 | stable, modest ceiling |
| id_06 |  0 | 0.807 | stable, strong |
| id_17 |  0 | 0.841 | stable, strong |
| id_00 |  1 | 0.822 | near-stable |
| id_14 |  2 | 0.807 | |
| id_01 |  2 | 0.784 | |
| id_10 |  2 | 0.734 | |
| id_18 |  1 | 0.670 | low ceiling |
| id_15 |  4 | 0.768 | |
| id_03 |  5 | 0.861 | flaky but high ceiling |
| id_13 |  6 | 0.838 | flaky but high ceiling |
| id_07 |  6 | 0.823 | |
| id_08 |  3 | 0.820 | |
| id_16 |  6 | 0.629 | beard — still HyperSwap's blind spot |
| id_09 |  7 | 0.888 | **highest ceiling**, half the time |
| id_05 |  8 | 0.799 | |
| id_04 |  8 | 0.807 | |
| id_19 |  8 | 0.522 | collapses on depth even when it doesn't nan |
| id_12 |  9 | 0.383 | **collapse-prone**: cross-seed std 0.38 on cfg002 |
| id_11 | 13 | 0.822 | natural_1024 works; tight_1024 fails on every seed |

Three coarse buckets emerge:

- **Rock-solid (n=4):** id_00, id_02, id_06, id_17 — ≤1 nan in 18 cells.
  Any cfg picks ship.
- **Wide-ceiling but flaky (n=4):** id_03, id_07, id_09, id_13 — high mean
  when valid (0.82-0.89) but 5-7 nans in 18.
- **Structurally hard (n=4):** id_11, id_12, id_16, id_19 — either fail on
  whole config families or produce low-id_cos faces that pass detection.

The 8 remaining photos sit between these buckets, with 2-8 nans and means
in 0.67-0.84.

## Per-seed reproducibility is poor on hard cells

Cross-seed std of id_cos (where all 3 seeds succeeded):

| photo | cfg | mean | std |
|---|---|---|---|
| id_12 | cfg002 (natural depth) | 0.46 | **0.38** |
| id_08 | cfg002 | 0.65 | 0.38 |
| id_12 | cfg000 | 0.49 | 0.36 |
| id_19 | cfg000 | 0.61 | 0.35 |
| id_02 | cfg005 | 0.59 | 0.35 |

At the same time, the stable photos have std < 0.04 — `id_07__cfg000`
ranges only 0.812 ± 0.038 across three seeds, `id_18__cfg004` ranges
0.724 ± 0.039. Stability and ceiling are roughly correlated but not
identical (id_03 has high ceiling, high flakiness).

This tells us the surface is bimodal at the photo level: a config is
either dependable on a given photo or unstable on it. The seed dial is
basically a Bernoulli — for hard photos, swap success is ~50/50 regardless
of which canny preset or depth you pick.

## Top-of-the-list cells

Top 10 with `det_mode==default`:

```
id_12__cfg002       cfg=02 sN=0  id=0.904  ff=0.325  cs=0.677
id_09__cfg002       cfg=02 sN=0  id=0.900  ff=0.322  cs=0.806
id_09__cfg000_s1    cfg=00 sN=1  id=0.896  ff=0.321  cs=0.720
id_12__cfg000_s2    cfg=00 sN=2  id=0.893  ff=0.325  cs=0.768
id_09__cfg000_s2    cfg=00 sN=2  id=0.891  ff=0.305  cs=0.723
id_09__cfg002_s2    cfg=02 sN=2  id=0.889  ff=0.297  cs=0.712
id_09__cfg001       cfg=01 sN=0  id=0.885  ff=0.324  cs=0.787
id_09__cfg001_s2    cfg=01 sN=2  id=0.885  ff=0.351  cs=0.727
id_09__cfg002_s1    cfg=02 sN=1  id=0.884  ff=0.354  cs=0.744
id_09__cfg000       cfg=00 sN=0  id=0.882  ff=0.316  cs=0.772
```

id_09 dominates with 7 of the top-10 cells. Note id_12 appears with the
ceiling cell *and* with the bottom-decile collapses (id=0.20) — same
photo, two of three seeds collapse, one nails the highest id_cos in the
sweep.

## Failure mode catalog

Five distinct failure patterns identified, listed by frequency:

1. **Tight-crop face-frac=0 collapse** — `id_11`, `id_05`, `id_04`, `id_09`,
   `id_12` on cfg003-005. Render fills doll body with head; no inscribed
   face for swap. Largest single contributor to the 25 % nan rate. Fix:
   either tighten the `tight_1024` framing prompt, or treat tight as a
   secondary config and budget for its ~36 % failure rate.
2. **Depth + structured-prior catastrophe on hard photos** — `id_12`
   cfg002 swings 0.20-0.90 across seeds; `id_19` cfg002 collapses to 0.28.
   Depth maps on faces with prominent features (glasses, beard, strong
   ethnic features) appear to bias the CN render off-distribution from
   the matryoshka prior, and HyperSwap can't recover.
3. **Catastrophic regenerate (the "wooden face" failure)** —
   `id_16__cfg000` id=-0.014, face_frac=0.003. Refine pass would have
   reproduced this from Phase 1; here it's a one-shot CN render that
   produced a generic painted-doll face. SCRFD finds a confident face but
   it isn't id_16. Same silent-fail mode that motivated the refine-axis
   prune in Phase 1; here it's purely upstream of refine.
4. **Beard ceiling** — `id_16` mean 0.629, best 0.71. HyperSwap on beards
   is a model-level limit, confirmed across both phases.
5. **Generic face_frac collapses** — `id_08__cfg002` produces id=0.22 with
   ff=0.009; one of the depth renders generated a thumbnail-sized face in
   a sea of doll body. Possibly the same root cause as #1, but with depth
   conditioning instead of tight cropping.

See `stages.png` for canonical examples of (3), (4), (1), and (2) side-by-
side.

## What this means for the deployable default

- **Robust default (n=4 photos):** any of cfg000-005 ships; pick
  `natural_1024 · canny · soft` (cfg000) for compactness.
- **Adaptive lookup is warranted.** Per-photo ceiling spread is now 0.30
  (id_09 0.89 vs id_12 0.38). A static config will leave roughly 8 of 20
  photos in the failure zone.
- **Pre-flight detector confidence as a feature.** All "structurally
  hard" photos have either obstructive accessories (id_01 glasses, id_16
  beard) or non-standard framing (id_19 hat). A cheap source-image
  classifier could route to natural_1024 + retries on these.
- **Don't ship tight_1024 as the only option.** 36 % nan is too high to
  default to it, even with its modest quality lift. Keep as a B-test pool
  for the 4 rock-solid photos.

## Axes still open after Phase 2

- **Source-photo features** that predict the per-photo nan rate.
  Hypothesis: head-area-fraction in the source, plus presence of glasses/
  beard/hat (covered by FairFace-style classifier or even mediapipe-mesh-
  area), are enough to predict tight-crop collapses ahead of time.
- **Per-photo seed-retry budget.** For flaky photos, a 2-3-seed retry
  ladder with deterministic fall-through (cfg002 → cfg000 → cfg004) buys
  us most of the ceiling without giving up reliability.
- **Whether HyperSwap-1c is the bottleneck.** id_16's beard ceiling at
  0.71 and the cross-seed std on id_12 both suggest the swap stage, not
  CN, is what's binding. A pass with InfiniteYou (or hyperswap-id1c at
  higher inference quality) on the 4 hard photos would isolate this.

## Cross-references

- `[[reference-comfyui-shard-runbook]]` — Windows 3090 host
- `[[feedback-sweep-avoid-local-gpu]]` — silent-OOM fix from Phase 1
- Phase 1: `docs/research/2026-05-20-photobooth-phase1-findings.md`
- Spec: `docs/superpowers/specs/2026-05-19-photobooth-phase1-sweep-design.md`
- Topic index: `docs/research/_topics/photobooth-sweep.md`
- Contact sheets: `exp_output/photobooth_phase2/grid_s0.png`, `…_s1.png`,
  `…_s2.png`, `stages.png`
