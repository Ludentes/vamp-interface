---
status: live
topic: photobooth-sweep
---

# Photobooth Phase 1 sweep — findings (2026-05-20)

LHS pilot over six axes on four stratified source photos, isolating the swap
stage to HyperSwap-1c. Goal: identify the robust default config + axis
sensitivities, plus an axis-pruning list for Phase 2.

## Setup

- Spec: `docs/superpowers/specs/2026-05-19-photobooth-phase1-sweep-design.md`
- Plan: `docs/superpowers/plans/2026-05-19-photobooth-phase1-sweep.md`
- Driver: `scripts/photobooth_sweep/driver.py`
- Host: Windows shard (`shard`, RTX 3090)
- Output: `exp_output/photobooth_phase1/` — `scores.parquet`, `manifest.json`,
  per-cell intermediates under `cells/<cell_id>/{ctrl,render,swap,refined}.png`
- 4 photos × 40 LHS cells = **160 cells**, ~36 s/cell wall (depth dominates)

## Live axes

| # | Axis | Values |
|---|---|---|
| 1 | `face_pixel_budget` | `natural_1024` / `tight_1024` / `natural_768` |
| 2 | `cn_condition` | `canny` / `depth` |
| 3 | `cn_strength` | continuous `[0.80, 1.00]` |
| 4 | `canny_preset` | `soft` / `default` / `aggressive` (null for depth) |
| 5 | `refine_denoise` | `0.00` / `0.10` / `0.15` |
| 6 | `demo_inject` | `off` / `on` |

(Axes dropped via the spike — see spec addendum 2026-05-20.)

## Scoring

- `id_cos` — insightface buffalo_l recognition vs source
- `det_score` — SCRFD confidence on the swapped crop
- `det_mode` — `default` / `forced` / `failed` / `exception`
- `face_frac` — face bbox area / render area
- `clip_style` — open_clip ViT-B/32 cosine vs anchor doll image
- `wall_clock` — render + swap + refine seconds

## Run truncation

Driver hit 145/160 cells before being silently OOM-killed when the user's
parallel CFM training job claimed local GPU memory. `scripts/photobooth_sweep/
scorer.py` was initializing `open_clip ViT-B/32` on `cuda:0`; ~500 MB of VRAM
contention was enough to kill the driver mid-cell-25 of id_01. Per-photo
counts: `id_00=40, id_11=40, id_16=40, id_01=25`. Sample is sufficient for
axis-level conclusions; per-photo claims for id_01 are weaker.

Durable fix: scorer now forces CLIP to CPU when running against `--comfy-url`.
Memory pinned at `feedback-sweep-avoid-local-gpu`.

## Headline numbers

| metric | overall (n=145) |
|---|---|
| `id_cos` mean | 0.692 |
| `id_cos` p05 | 0.518 |
| `id_cos` p50 | 0.701 |
| `id_cos` p95 | 0.823 |
| `id_cos` max | **0.850** |
| `det_mode == default` | 84 / 145 (58 %) |
| `det_mode == forced` | 61 / 145 (42 %) |
| failures (`exception`) | 0 |
| wall_clock mean | 35.7 s/cell (RTX 3090) |

**Per-photo ceiling and floor:**

| photo | mean | median | max |
|---|---|---|---|
| id_00 (Black F) | 0.728 | 0.753 | 0.850 |
| id_11 (E.Asian F) | 0.715 | 0.689 | 0.834 |
| id_01 (White F, sunglasses) | 0.706 | 0.706 | 0.805 |
| id_16 (Mid-East M, beard) | 0.627 | 0.664 | 0.726 |

Per-photo dispersion in *mean* id_cos is 0.10 — adaptive lookup gain ceiling
is modest. Per-photo dispersion in *max* is 0.12 — id_16 has a measurably
lower attainable ceiling, consistent with the HyperSwap-on-beard story from
the matryoshka bake-off.

## Per-axis effect

`id_cos` mean grouped by axis value:

| axis | value | n | mean | median | max |
|---|---|---|---|---|---|
| `refine_denoise` | **0.00** | 35 | **0.779** | 0.802 | 0.850 |
| `refine_denoise` | 0.10 | 38 | 0.677 | 0.684 | 0.777 |
| `refine_denoise` | 0.15 | 36 | 0.628 | 0.655 | 0.763 |
| `demo_inject` | **on** | 42 | **0.726** | 0.716 | 0.850 |
| `demo_inject` | off | 67 | 0.673 | 0.683 | 0.840 |
| `cn_condition` | **canny** | 51 | **0.711** | 0.723 | 0.850 |
| `cn_condition` | depth | 58 | 0.679 | 0.680 | 0.834 |
| `face_pixel_budget` | **natural_1024** | 41 | **0.706** | 0.720 | 0.850 |
| `face_pixel_budget` | tight_1024 | 30 | 0.701 | 0.699 | 0.835 |
| `face_pixel_budget` | natural_768 | 38 | 0.674 | 0.678 | 0.840 |
| `canny_preset` | aggressive | 15 | 0.730 | 0.726 | 0.840 |
| `canny_preset` | soft | 19 | 0.718 | 0.723 | 0.829 |
| `canny_preset` | default | 17 | 0.685 | 0.716 | 0.850 |
| `cn_strength` | 0.80-0.87 | 29 | 0.687 | 0.713 | 0.835 |
| `cn_strength` | 0.87-0.93 | 36 | 0.701 | 0.711 | 0.834 |
| `cn_strength` | 0.93-1.00 | 44 | 0.691 | 0.682 | 0.850 |

**Ranked by effect on id_cos mean:**

1. **`refine_denoise`** is the dominant axis — `0.00` beats `0.15` by 0.15
   id_cos. Every non-zero value of refine destroys identity, regardless of the
   other settings. Mechanism: low-denoise img2img re-runs Z-Image over the
   swap output, which pulls the face back toward the painted-matryoshka
   prior. Even denoise=0.10 loses ~0.10 id_cos on average.
2. **`demo_inject`** is real: +0.05 id_cos. The demographic phrase prefix
   primes the CN to render a structurally matching face — gender × age × race
   context biases the U-Net before the swap fixes identity-pixels.
3. **`cn_condition=canny`** beats `depth` by +0.03. Canny encodes face landmark
   geometry; depth encodes overall head silhouette. HyperSwap can use the
   landmark prior better than the silhouette one. Modest effect.
4. **`face_pixel_budget`**: `natural_1024` ≈ `tight_1024` > `natural_768` by
   ~0.03. Render-resolution headroom matters; crop tightness does not.
5. **`canny_preset`**: variation is within noise (range 0.69-0.73 on mean).
6. **`cn_strength`** in `[0.80, 1.00]` is **flat** — no monotone trend on the
   measured range. Effect size is at the noise floor.

## Robust default

Best multi-photo config from this run (cells per config are scarce — these
are read as candidates, not certified winners):

```
face_pixel_budget = natural_1024
cn_condition      = canny
canny_preset      = soft  or aggressive  (default is slightly worse)
cn_strength       = ~0.85   (any value in [0.80, 1.00] is fine)
refine_denoise    = 0.00    (mandatory; any positive value is a net loss)
demo_inject       = on      (use the per-photo demographic prefix)
```

Highest-scoring `det_mode==default` cells (id_cos ≥ 0.81) all share
`refine_denoise = 0.00`; canny dominates the top 10 but two depth cells reach
0.82+ so depth is viable. `demo_inject` is on in 8 of the top 10. Strengths
span 0.81-1.00 with no pattern.

## Failure modes

Looking at the bottom decile (id_cos < 0.55, n = 14):

- **All 14 have `refine_denoise ∈ {0.10, 0.15}`** — the refine pass is the
  proximate cause.
- **id_16 dominates** (9 of 14 bottom cells). The beard remains HyperSwap's
  blind spot.
- **One catastrophic regenerate**: `id_16__cfg010` with `id_cos=0.011` but
  `det_score=0.812`. SCRFD finds a confident face — but the refine pass at
  denoise=0.15 erased the swap and re-rendered a generic painted-doll face,
  which is a face but is *not* id_16. This is the silent-fail mode of the
  refine axis — without `id_cos` you wouldn't notice.
- `det_mode=forced` is 42 % of all cells. SCRFD often fails to detect the
  swapped face on the matryoshka render, but the mediapipe fallback recovers
  most cells. The 42 % rate is uncomfortably high — Phase 2 should reduce
  this (likely by avoiding the small-face `natural_768` budget).

## Phase 2 axis-pruning list

**Drop entirely:**
- `refine_denoise` — lock to `0.00`. Every positive value loses on every
  photo. No information gained by sweeping further.
- `natural_768` — strictly dominated by 1024 variants on every metric.
- `cn_strength` exploration — lock to a single value (0.85 is mid-bin). Use
  the freed degrees of freedom for more per-photo cells.

**Keep, but narrow:**
- `face_pixel_budget` → 2-way `{natural_1024, tight_1024}` (or lock to
  natural; tight gives no measurable benefit in this run).
- `cn_condition` → keep both (`canny` slight winner, `depth` competitive in
  the top cells; useful in Phase 2 to confirm per-photo).
- `canny_preset` → keep, drop `default` (it underperforms `soft`/`aggressive`
  on mean).

**Lock-in winners (consider non-axes for Phase 2):**
- `demo_inject = on`
- `refine_denoise = 0.00`
- `cn_strength = 0.85`

**Phase 2 sketch (post-pruning):**

| axis | values |
|---|---|
| `cn_condition` | `canny` / `depth` |
| `canny_preset` | `soft` / `aggressive` |
| `face_pixel_budget` | `natural_1024` / `tight_1024` |

3 axes, 8 configs × 20 photos = **160 cells** at the same compute footprint
as Phase 1, with 5× per-config replication for confidence intervals. Should
take ~1.5 h on the 3090.

## Open questions surfaced

- Does HyperSwap on beards have a *config-level* fix or is it a model-level
  ceiling? `id_16` ceiling at 0.726 vs others at 0.81-0.85 suggests model
  limit, not config.
- Is `det_mode=forced` a quality signal or just a detector fallback? Need to
  check whether forced cells systematically score lower beyond the obvious
  bottom-10. Cheap regression at Phase-2 time.
- Why does `demo_inject` help if the prompt is dominated by the CN structure?
  Hypothesis: the prompt biases the Z-Image render structure (skin tone bias
  in particular), giving HyperSwap better-shaped target pixels to blend with.
- Refine-as-currently-implemented (whole-image low-denoise) is destructive.
  A future masked refine (mask = swap-affected face region only, leave body
  untouched) might recover its theoretical value. Out of scope for Phase 2.

## Cross-references

- `[[reference-comfyui-shard-runbook]]` — the Windows 3090 host
- `[[feedback-sweep-avoid-local-gpu]]` — the OOM-kill root cause + fix
- `[[project_matryoshka_bakeoff]]` — prior swap baseline
- Spec: `docs/superpowers/specs/2026-05-19-photobooth-phase1-sweep-design.md`
- Plan: `docs/superpowers/plans/2026-05-19-photobooth-phase1-sweep.md`
