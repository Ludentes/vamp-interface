---
status: live
topic: metrics-and-direction-quality
---

# Smile iter_04 — ablation: tinkering with the primary pair's prompt does NOT cancel race drift

**Framing**: this is the *prompt-only ablation* against the counter-edit approach. Before building a separate race δ and composing it, quantify how much the race drift can be fixed by only modifying the primary smile pair's prompt. Result: it can't — which justifies the heavier machinery.

**Variants compared**: three smile pairs that differ only in how much demographic detail sits in pos+neg:
- v4_age_only: `{age}` only
- v6_full_demo: `{age} {ethnicity} {gender}`
- v7_demo_no_age: `{ethnicity} {gender}`

**Result: prompt anchoring inside the smile pair does not help.** Race drift on elderly_latin_m is the same across all three.

| variant | pos prompt on elderly_latin_m | race (scale 0 → ~0.75) |
|---|---|---|
| v4_age_only | "A elderly person smiling warmly." | Latino_Hispanic → **East Asian** |
| v6_full_demo | "A elderly Latin American man smiling warmly." | Latino_Hispanic → **East Asian** (same) |
| v7_demo_no_age | "A Latin American man smiling warmly." | Latino_Hispanic → **Black** (worse) |

Visual collage (`collage_elderly_latin_m_seed2026.png`) confirms: all three variants produce the same narrower-face, narrower-eye, smoother-skin morphing at scale 0.5–0.8. Ethnicity in the prompt pair is cosmetic; the FluxSpace edit δ carries its own race drift that composes with the base regardless of what the prompt says.

## Why it failed

Both `pos` and `neg` are *edit prompts*, fed to the pair-averaged attention node. Putting "Latin American man" in both cancels out in the δ — so it has no steering effect on the edit direction. The drift is a property of the "smiling warmly" edit token attention itself, which correlates with East Asian features in the underlying Flux training distribution.

Pre-emptive anchoring works for the **base** prompt (scale=0 image); it does not work for the **edit δ**.

## Side findings (from the per-(variant,base) table)

- **v6 preserves age better** than v4 on elderly: mv_age_slope +0.98 y/scale vs +6.31 y/scale. Ethnicity in the prompt pair helps age anchoring somehow (probably via conditioning on "Latin American man" making "elderly" semantically more specific).
- **`{age}` is load-bearing**. v7 (no age word) on elderly_latin_m: mv_age_slope **−45.28 y/scale**. The model de-ages when it isn't told to keep the base's age.
- **v7 also produces the worst identity drift** (0.755 vs 0.536 for v6). Dropping the age anchor destabilises identity.

## Composite score ordering

1. v6_full_demo — 0.052
2. v4_age_only — 0.045
3. v7_demo_no_age — 0.033

v6 ≻ v4 because both target and age are slightly better; but **both have the race confound**. Scoring without a race penalty hides the equivalence on the problem dimension.

## Implication for the framework

The ablation confirms: **race drift must be cancelled via composition with a separate race-only δ**, not via prompt tinkering within the smile pair. The **iterative single-axis corrective loop** (step 4 in the procedure) is the only remaining path for race drift on `elderly_latin_m` smile:

1. Fire primary edit (v6_full_demo or v4_age_only, scale ≈ 0.29–0.5).
2. Measure: race flip detected.
3. Fire a race counter-edit pair (e.g. `pos="A Latin American person."`, `neg="A person."`) at some negative-or-positive weight.
4. Re-render, re-measure.

**Next concrete step**: `race_iter_01` — build counter-edit pairs for Latin American ↔ East Asian anchoring, add to dictionary. Then test composition (even in a crude sequential-render form) on elderly_latin_m smile.

## Dictionary state after iter_04

- 33 rows before → **42 rows after** (+9: 3 variants × 3 bases)
- All 9 new rows tagged `axis=smile`, `iteration_id=smile/iter_04`
