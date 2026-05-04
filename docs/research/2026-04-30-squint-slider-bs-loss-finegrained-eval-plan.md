---
status: live
topic: squint-slider-bs-loss
---

# Fine-grained eval plan — squint slider bs_loss variants (v0, v1a, v1c, v1e, v1f)

**Goal:** find each variant's *useful range* — the slider strength below which we
get a clean squint and above which the eyes collapse. The bs_loss work
through v1c→v1f cannot eliminate closure at m=1.5 (corpus visually equates
strong squint with lid compression; v1e showed eyeSquint↔eyeBlink channels
correlated in the student's geometry). What it might do is **push the
closure cliff outward** — buy a wider clean band before saturation.

The metric we care about is "where does the cliff sit," not "is m=1.5 safe."

## What we evaluate

| Run | Ckpt | What it tests |
|-----|------|---------------|
| `squint_slider_v0` | 550 | baseline reference (no bs_loss) |
| `squint_slider_v1a_bs_sanity` | 550 | bs_loss preserve on mouth channels (sanity, off-axis) |
| `squint_slider_v1c_bs_eyeopen_max` | 200 | preserve eyeBlink w=1000 (closest-to-final before kill) |
| `squint_slider_v1e_engage_squint` | 75 | engage eyeSquint + preserve eyeBlink (final before kill) |
| `squint_slider_v1f_engage_both` | TBD | dual engage (eyeBlink→0 + eyeSquint→0.5) — see *Conditional inclusion* |

v1b and v1d are excluded:
- v1b killed at step 50 (no visible movement vs v0)
- v1d froze (catastrophic — t_max=1.0 with clean-trained student injected
  high-variance random gradient, base loss got drowned)

## Grid

- Strengths: `[0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]` (7 points)
- Seeds: `[2026, 4242, 7777]` (3 seeds — deliberately disjoint from
  training-time sample seed 1337 so we're not just re-rendering the
  cells we already eyeballed)
- Prompts: 1 prompt only — the european-man default (`pid="european_man"`
  in measure_slider's prompt list). Single demographic to keep cell count
  small; cross-demographic generalisation can be a later sweep.
- Total per ckpt: 7 × 3 = 21 renders ≈ 10 min on a 4090
- Total grand: 5 ckpts × 21 = 105 renders ≈ 50 min if we include v1f

## Tooling

`src/demographic_pc/measure_slider.py` already accepts:

```bash
python -m demographic_pc.measure_slider \
  --slider-name <name> \
  --checkpoint <path-to-.safetensors> \
  --intended-axis squint \
  --phase render \
  --strengths 0 0.25 0.5 0.75 1.0 1.25 1.5 \
  --seeds 2026 4242 7777
```

Output convention (already in code):
`models/sliders/<name>/<ckpt_tag>/renders/<prompt_id>/seed{S}_str{V}.png`

`--phase render` only — we're not running the full SigLIP / MediaPipe
scoring battery here (that's a downstream task once we know which ckpt is
worth measuring). Visual contact sheet is the deliverable.

## Contact sheet

Build a 7-col × 3-row grid per ckpt: columns = strengths, rows = seeds.
Stack the 5 ckpt grids vertically (or place side-by-side for direct
cliff-position comparison).

Suggested helper: extend `src/demographic_pc/build_ffhq_contact_sheets.py`
with a `--render-glob` mode, or write a fresh small script — it's just
`PIL.Image.paste` over the renders dir tree. Keep it under 50 LOC.

## Pass criteria

For each ckpt, find the **cliff strength** `m_cliff` = lowest m where ≥ 2/3
seeds show one or more of:
- iris fully occluded (no sclera visible)
- lids pressed flat / line-shaped eye opening
- skin of upper and lower lid touching across full eye width

Compare:
- `v1f.m_cliff − v0.m_cliff` ≥ 0.25 → bs_loss does useful work, ship it
- `v1f.m_cliff − v0.m_cliff` < 0.25 → cosmetic, accept v0 range or change data
- v1c, v1e shown for ablation context (which loss form contributed most)

## Conditional inclusion of v1f

v1f is currently training (bash bq097bj9z). Don't burn 10 min eval-rendering
v1f if its training-time samples already show no movement:

1. Wait for v1f to reach step ≥ 200.
2. Eyeball training-time samples at `output/ai_toolkit_runs/squint_slider_v1f_engage_both/samples/*_000000200_2.jpg` (m=+1.5, east asian man), `_5.jpg` (m=+1.5, black woman), `_8.jpg` (m=+1.5, european man).
3. If full closure on all three → skip v1f from this eval (no information beyond v1e).
4. If partial closure or open eyes on any → include v1f at ckpt 200.

## Out of scope here (deliberately)

- SigLIP / MediaPipe / ArcFace scoring — render-only for now.
- Cross-demographic generalisation (1 prompt only).
- Held-out prompts.
- m=−1.5 (negative slider direction) — squint is asymmetric; we only care
  about the engagement direction.
- v1b, v1d (excluded above).

These can each be added later if the cliff-shift result merits a deeper look.

## Schedule

Don't kick off until v1f frees the GPU (or v1f is killed because its
training-time samples show closure already). Then run all 5 ckpts in one
batch; the entire eval should fit in under an hour.
