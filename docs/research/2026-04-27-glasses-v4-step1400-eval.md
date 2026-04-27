---
status: live
topic: demographic-pc-pipeline
---

# Glasses slider v4 step 1400 — quality eval (canonical schema)

Companion to [step 600 eval](2026-04-27-glasses-v4-step600-eval.md).
Step 1400 is near-final under the η=4 hail-mary regime (1500 max);
sampled to test whether late training improves the slider or
overshoots.

Both 600 and 1400 evals are stored in the **canonical schema** that
matches `models/blendshape_nmf/sample_index.parquet` — every column
from the FluxSpace measurement corpus is present (92/92), so
`pd.concat([sample_index, lora_eval], join="outer")` produces one
fusable corpus for downstream solvers (Path B checkpoint mixing,
Path C training-pair selection, cross-axis composition queries). LoRA
evals add 22 slider-specific columns (`slider_name`, `checkpoint`,
`observed_*` classifier predictions, `prompt_pool`, etc.).

## 600 vs 1400 — the comparison

| Metric | step 600 | step 1400 | direction |
|---|---|---|---|
| in-dist frac with glasses @ +1.0 | 89% | 89% | tie |
| held-out frac @ +1.0 | 67% | **50%** | **600 wins** |
| held-out frac @ +1.5 | 100% | 89% | 600 wins |
| ArcFace mean in s∈[0,+1.5] | 0.686 | 0.709 | 1400 wins |
| ArcFace cells <0.4 | 17/108 | 9/108 | 1400 wins |
| Spearman in-dist (one-sided) | 0.77 | 0.80 | 1400 wins |
| Spearman held-out (one-sided) | 0.82 | 0.79 | 600 wins |
| Generalization gap @ +1.0 | 22 pts | **39 pts** | 600 wins |

**Step 1400 is worse on held-out coverage despite 800 more training
steps.** Identity preservation improves marginally (0.69 → 0.71, 17 →
9 failures), but generalization erodes — the slider increasingly only
fires on prompts that look like the training set. This is the
classic overfitting signature: more training tightens the fit on
in-distribution prompts at the cost of held-out behaviour.

## Verdict

**Ship 600, not 1400.** η=4 v4 is past its useful peak by step 1400.
The right late-training regime needs either lower LR (cosine schedule
to 0), reduced η, or earlier early-stopping based on held-out
coverage rather than wall-clock checkpoints.

## Implication for v5

The default plan was cosine LR + η=2.5 + EMA off + surgical scope +
800 steps. Step 600's overfit-by-1400 signal validates the cosine
schedule and the lower step budget — long flat-LR training in v4 ate
its own held-out generalization. **Hold the v5 plan as drafted.**
v6 (Lion or Prodigy) still runs sequentially regardless of v5
outcome per prior decision.

## Files

- `models/sliders/glasses_v4/glasses_slider_v4_000001400/eval.parquet`
  (243 rows × 114 cols, canonical schema)
- `models/sliders/glasses_v4/glasses_slider_v4_000001400/glasses_v4_glasses_slider_v4_000001400_eval_collage.png`
- `models/sliders/glasses_v4/glasses_slider_v4_000000600/eval.parquet`
  (re-scored 2026-04-27 to canonical schema; 243 rows × 114 cols)
