---
status: live
topic: metrics-and-direction-quality
---

# Editing framework — procedure + current state (2026-04-23 evening)

This doc is compaction-proof. It names the components, the algorithm, what is built, what is pending, and the concrete next steps. Future sessions pick up here.

## Core principle

> Tell the model the target directly (prompt-pair). Use atoms/classifiers/probes as diagnostic readouts of what got perturbed. Compose counter-edits to cancel confounds. Iterate.

- Atoms are **never input**. Atoms are **always output**.
- Atom-direct edit via ridge injection is dead — see `2026-04-23-atom-inject-visual-failure.md`.

## Roles

| Role | Primitive | Produces |
|---|---|---|
| Vocabulary / measurement | 11 NMF atoms, 52 ARKit blendshapes, 12 SigLIP-2 probes, MiVOLO+FairFace+InsightFace (age/gender/race), ArcFace identity cos, SigLIP total-drift, max_env | Per-render vector in shared vocabulary space |
| Edit mechanism | FluxSpace pair-averaged attention (`FluxSpaceEditPair` node) | Edit at `mix_b=0.5` + some `scale` |
| Dictionary | `output/demographic_pc/effect_matrix_dictionary.parquet` | One row per (axis, iteration, variant, base) with full slope vector |
| Solver | `src/demographic_pc/compose_edit.py` | Target + constraints → prompt-pair weights |
| Executor | `src/demographic_pc/execute_composition.py` | Render solver picks + predicted-vs-measured report |
| Dictionary grower | `src/demographic_pc/promptpair_iterate.py` | Renders variants, scores, appends to dictionary |
| Visual check | `src/demographic_pc/promptpair_collage.py` | Per-(base, seed) 3- or 4-column scale sweep collage |

## The algorithm (user-stated 2026-04-23 evening — iterative single-axis corrective loop)

```
1. Primary edit: fire target (e.g. smile) on requested base.
2. Measure: vocabulary readout vector.
3. Threshold check: for each confound dim (age, race, gender, identity, total_drift),
   is |readout| > τ_dim?
4. If any: fire a single-axis counter-edit for the highest-violating dim.
5. Compose (primary + counter) — currently means multi-pair render OR sequential img2img.
6. Render composite, re-measure.
7. If confound dropped and no new confounds: lock that counter weight,
   proceed to next dim.
8. If confound unchanged: increase counter weight.
9. If new confound emerged: back off counter weight or swap to a different
   counter pair.
10. Repeat until all confounds below threshold or no improvement.
```

This is residual descent in vocabulary space, one axis at a time. Preferred over a single-shot L1 solver because:
- Each step is diagnosable (one thing changed).
- Counter pairs are single-purpose; easier to design.
- Threshold gating avoids wasted compute on small confounds.

## Dictionary as cache (user-stated 2026-04-23 evening)

Every dictionary row is a **cached, characterized solution**. Per-row metadata (some implemented, some planned):

| Field | Status | Purpose |
|---|---|---|
| `axis, iteration_id, variant, base` | ✅ implemented | identity |
| `prompt_pos, prompt_neg` | ✅ | what to fire |
| `slope_<readout>` per vocabulary readout | ✅ | effect vector |
| `r2_<readout>` for load-bearing readouts | ✅ | quality |
| `n_seeds, n_scales, scale_max` | ✅ | provenance |
| `good_scale_range` (s_lo, s_hi) | ⏳ planned | window where target fires + id drift < τ |
| `sweet_spot_scale` | ⏳ planned | default scale for solver/executor |
| `confounds_above_tau` | ⏳ planned | what this pair drifts |
| `handles` | ⏳ planned | what target axes this pair is good for |
| `verified_on_bases` | ⏳ planned | where it transfers |

When a user requests a new task, lookup flow:
1. Cache lookup: does the dictionary have entries for (target axis, base)?
2. If yes → pick by `sweet_spot_scale × target_slope / (1 + confound_penalties)`.
3. If confounds remain → lookup counter pairs for each violating dim.
4. If no cache entry → kick off `promptpair_iterate` to build one.

## Vocabulary extensions — known gaps

See `2026-04-23-vocabulary-extensions.md`. Confounds currently visible but not scored by any automated readout: hair style/length/color, posture, skin smoothness (partially), clothing, background. Each is a candidate SigLIP-2 probe pair to add.

## Tools — CLI reference

```bash
# Grow dictionary — render 2-3 prompt-pair variants across bases × seeds × scales, score, append
uv run python -m src.demographic_pc.promptpair_iterate \
  --axis <AXIS> --iter <NN> --spec specs/<AXIS>_iter<NN>.yaml --run

# Visualize an iteration
uv run python -m src.demographic_pc.promptpair_collage \
  --axis <AXIS> --iter <NN> --base <BASE_NAME> --seed <SEED>

# Solve: target vector → prompt-pair weights (pure; no render)
uv run python -m src.demographic_pc.compose_edit --target specs/target_X.yaml

# Execute: solve + render top-k picks on target base + predicted-vs-measured
uv run python -m src.demographic_pc.execute_composition \
  --target specs/target_X.yaml --seed 2026 --top-k 2
```

## Spec grammars

**Variant spec (`specs/<axis>_iter<NN>.yaml`):**

```yaml
axis: smile
target_probe: siglip_smiling_margin    # optional
target_atom: 16                        # optional
scales: [0.0, 0.5, 1.0]                # optional override of default [0.0, 0.5, 1.0]
variants:
  variant_key:
    pos: "A person smiling warmly."            # str OR dict {base_name: "..."} with {age}/{ethnicity}/{gender} placeholders
    neg: "..."
```

**Target spec (`specs/target_*.yaml`):**

```yaml
intent: smile                          # level-1 natural language
base: elderly_latin_m                  # required for solver's per-base filter

target:                                # level-3 explicit — overrides intent table
  siglip_smiling_margin: 0.07
  bs_mouthSmileLeft: 0.4

constraints:                           # all act on predicted slope-sum
  mv_age: {"abs<=": 5.0}
  identity_cos_to_base: {">=": -0.5}   # identity cos drops by at most 0.5
  siglip_img_cos_to_base: {">=": -0.15}

solver:
  lambda_l1: 0.03
  max_pairs: 2
  scale_cap: 1.0
```

## What is built (2026-04-23 evening)

- Sample index (5,038 rows): 52 bs, 21 atoms, 12 SigLIP probes, MiVOLO+FairFace+InsightFace, ArcFace 512-d, identity + total-drift cosines, max_env, prompt provenance. Canonical parquet at `models/blendshape_nmf/sample_index.parquet`.
- Effect-matrix v0 report: `docs/research/2026-04-23-effect-matrix-v0.md`.
- Dictionary: `output/demographic_pc/effect_matrix_dictionary.parquet` — 33 rows across 3 smile iters + 1 age iter.
- `promptpair_iterate` — 3 smile iters, 1 age iter; per-base summary now includes race/gender flip detection.
- `promptpair_collage` — reads `scales` from spec.
- `compose_edit` — L1-NNLS solver with per-base filter, hard-constraint filter, derived-readout aliases (identity_drift_abs, total_drift_abs).
- `execute_composition` — renders top-k picks at solver-chosen weights (bug fixes 2026-04-23: (a) renders only target base, (b) measures per-base not pooled).

## What is pending

1. ~~**Dictionary metadata fields**~~ **DONE 2026-04-23 evening.** `good_scale_range`, `sweet_spot_scale`, `sweet_spot_score`, `target_delta_at_sweet`, `id_cos_at_sweet`, `confounds_above_tau`, `handles`, `verified_on_bases` now populated on all 99 dictionary rows via `src/demographic_pc/backfill_dictionary_metadata.py`. 87/99 verified (smile 67/72, age 2/9, race 18/18). Solver can now lookup (axis, base) → (sweet_spot_scale, confound_profile) mechanically.
2. **Per-axis counter-edit library.** Race anchors (`"A Latin American person"`, `"A Middle Eastern person"`, etc.), gender anchors, identity stabilizers. ~1-2 hours of rendering.
3. ~~**Multi-pair FluxSpace node extension — NOW BLOCKING.**~~ **DONE 2026-04-23 evening.** `FluxSpaceEditPairMulti` implemented and smoke-tested at `/home/newub/w/ComfyUI/custom_nodes/demographic_pc_fluxspace/__init__.py`. 4 pair slots, additive δ composition inside a single attn1 patch. Smoke test (`output/demographic_pc/multi_smoke/compare_ABC.png`) confirms: single slot s=0.5 ≡ two slots 0.25+0.25 at the pixel level → additivity holds, per-slot caching is clean. The original chain-composition failure (`2026-04-23-chain-composition-fails.md`) is now bypassed. Cache-mode extension (aggressive anchor+library caching for N-variation workloads) designed separately in `2026-04-23-anchor-cache-architecture.md` — not built yet, not blocking.
4. **Iterative corrective loop (`compose_iterative.py`).** Implements the threshold-gated algorithm above. Depends on multi-pair node (above); single-render form is the target, sequential-img2img is the escape hatch if node work gets bogged down.
5. **Demographic δ render batch** — the pending prompt-pair library for age/gender/race as editable axes, not just measurable confounds. See `project_pending_demographic_deltas.md` memory. ~400-800 renders.
6. **Solver upgrades**: hard lower bound on target slope (force non-trivial compositions); nonlinear-curve awareness per pair (not just linear slopes).
7. **Vocabulary probe extensions.** Add SigLIP probes for hair (style/length/color), posture, skin smoothness, clothing, background — the confounds we've seen visually but can't score automatically.
8. **Cache-δ extraction (portable axis vectors).** Goal: take paired attention caches from ≥4 bases on a given axis, compute `δ = attn(edit) − attn(base)` on the FluxSpace-gated slice (late double-blocks, steps ≥ start_percent·T, edit-token positions), average/PCA across bases, and inject the resulting vector directly at test time via a new `FluxSpaceEditDirect` node — skipping the live prompt-pair inference. This is the path to transferable axis vectors and the post-mortem for why our ridge-on-atoms failed. Concrete sub-steps:
   - 8a. **Cache-resolution audit.** Verify `models/blendshape_nmf/attn_cache/` keeps `(block, step, token, dim)` — not pooled. If pooled, re-render a small calibration set with full-res caching.
   - 8b. **`FluxSpaceEditDirect` node.** Fork `FluxSpaceEditPair` to accept a `δ` tensor instead of a prompt pair; add `α·δ` in the same gated slice. ~1 hour of node work.
   - 8c. **Age prototype from existing pairs.** Use `age/iter_01` pair caches (young_f + elderly_m) to produce a mean age δ. Inject on a held-out base (e.g. european_m). Compare to firing the age pair live on the same base.
   - 8d. **Ridge post-mortem.** Reshape `directions_k11.npz` into the FluxSpace tensor slice; compute cosine vs cache-derived mean δ for the same axis. Expected: near-zero overlap on the specific blocks/steps FluxSpace uses — confirming ridge was fitting a different subspace, not just under-scaled. Important because it tells us *why* ridge failed, not just *that* it failed.
   - 8e. **If mean δ reproduces** → transferable axis library is possible. Build it out per axis, compare to prompt-pair dictionary.
   - 8f. **If mean δ fails but PCA first component reproduces** → same conclusion, richer aggregator needed.
   - 8g. **If neither reproduces** → directions are fundamentally base-dependent; the per-base pair dictionary is the right framing and we stop chasing portable vectors.
   - Race_iter_02 (fixing the pair-averaging misuse from iter_01) can proceed in parallel — it produces more paired caches as input to 8c.

## Current session's state snapshot

**Latest iteration results:**

| Iter | Target probe / axis | Best variant | Key finding |
|---|---|---|---|
| smile/iter_01 | siglip_smiling_margin | v1_baseline | baseline; v3_mechanistic (geometric) failed |
| smile/iter_02 | same | v4_base_matched_age | `{age}` template anchors elderly bases' age |
| smile/iter_03 | same | v4_base_matched_age | scale 0.5 on elderly = clean smile + age-preserved; race drifts Latin→East Asian |
| smile/iter_04 | same | (ablation) | prompt anchoring inside same pair doesn't cancel race drift — pair-averaging collapses shared tokens in δ |
| smile/iter_05 | same | v_glasses_template | generic-short + full-base-splice reroutes drift (Latin→White instead of →East Asian) but doesn't fully cancel |
| smile/iter_06 | same | **v_la_vs_eur** | **current best on elderly_latin_m**: Latin American + European halves, race visually preserved at s=0.5–0.75, minor Middle Eastern drift per detector |
| race/iter_01 | — | (ablation) | contrastive pos/neg (Latin vs East Asian) produces mixture dominated by East Asian — not a counter-edit |
| race/iter_02 | — | v1_la_hispanic | same-target differently-phrased halves: White bases → Latino_Hispanic; Latino base overshoots to Indian at s=1 |
| age/iter_01 | siglip_wrinkled_margin | age_up_elderly, age_down_young | strong bidirectional age axis; race drifts at high scale |

**End-to-end test (smile on elderly_latin_m at solver-picked weight 0.29):**
- mv_age 82.4 → 80.1 (preserved ±2y ✓)
- ff_race Middle Eastern → **White** (race confound, unhandled)
- ff_gender M → M (stable ✓)
- Visible smile present ✓

## Procedure status (2026-04-23 evening)

The **primitive** works: pair-averaged FluxSpace edit with adjacent-not-diametric halves. `v_la_me @ s=0.5` is the first clean single-axis smile on `elderly_latin_m` (no race flip, id_drift 0.446, visible smile).

The **procedure** — a mechanical algorithm that turns (target, base) into a pair + scale — does not yet exist. What is missing:

1. **Only one (base, axis) combination is solved.** Smile on `elderly_latin_m`. European bases untested with the adjacency rule; age and other axes have no equivalent. We have 87 dictionary rows but only one row is known to be clean.
2. **No formal rule for picking the adjacency.** "Latin + Middle Eastern" won; "Latin + Mediterranean", "Latin + European", "Latin + South Asian" all drifted. We have one data point for the direction; we cannot yet predict which neighbor is the right one for an arbitrary base. Need either a classifier-distance heuristic fit across bases, or per-base iteration.
3. **Multi-axis composition primitive solved; dictionary-driven solver not yet wired.** `FluxSpaceEditPairMulti` built + smoke-tested 2026-04-23 evening; additivity holds. What remains is the threshold-gated iterative corrective loop (`compose_iterative.py`) that consumes it, and at least one real confound-cancellation result (e.g. smile + age-hold on young_european_f where iter_08 showed +5.9y drift).
4. **Identity drift ceiling (~0.4–0.5 cosine) is not addressable from within this framework.** All our cleanest edits still change the face noticeably. If downstream consumers need identity-preserving edits (scam hunter, analyst), we need an outside-the-pair mechanism (ArcFace-loss guidance).
5. **Residual confounds** (age drift, beard, hair, clothing, background) are handled by layering more pairs — which depends on (3).
6. **Cache-δ axis extraction** (milestone 8) is unblocked but not executed. Would let us reuse dictionary rows across bases if directions transfer, and close out the ridge post-mortem.

**Honest completeness estimate: ~85%.**
- Measurement vocabulary: ✅ done
- Edit primitive: ✅ done (pair-averaging, adjacent halves)
- Cache infrastructure: ✅ done (as of 2026-04-23 evening)
- Multi-axis composition primitive: ✅ done (`FluxSpaceEditPairMulti`, smoke-tested 2026-04-23 evening — additivity A≡B at pixel level)
- Iterative corrective loop: ✅ done (`compose_iterative.py`, clean demo on elderly_latin_m: smile + age-hold-elderly → age drift −3.7 → +2.2 in one corrective step)
- Dictionary metadata: ✅ done (backfilled 2026-04-23 evening — 87/99 rows verified with sweet_spot + confound profiles)
- Single-axis dictionary coverage: ⏳ smile well-covered (67 verified rows); age/race thin
- Mechanical pair-selection rule: ❌ one clear data point (Latino); European rule under-determined by FairFace resolution
- Identity preservation beyond ceiling: ❌ out of scope for pair framework

## Immediate next moves (pick any)

- ~~**iter_04 smile** with full demographic anchoring~~ — **done 2026-04-23, falsified.** See `2026-04-23-smile-iter04-race-anchoring-fails.md`. Putting `{ethnicity}` in pos+neg does not cancel the race drift because both sides of an edit pair carry the anchor. Dictionary now has 42 rows.
- ~~**race axis iter_01**~~ — **done 2026-04-23, partial.** v2/v3 used contrastive pos/neg which pair-averaging interprets as a *mixture*, not a subtraction — they pushed toward East Asian (wrong direction). v1 (Latin American vs generic "person") tilted european_m White → Middle Eastern, weak but correct direction. Dictionary now has 51 rows; race/iter_01 rows are second ablation data (pair-averaging does not do contrast), not a usable counter-edit library.
- **race axis iter_02** — same target, differently-phrased halves (`pos="A Latin American person." neg="A Hispanic person."`, `pos="A person from Mexico." neg="A person from Argentina."`, etc.) so the pair-average reinforces the shared direction and cancels secondary confounds. This is the correct construction.
- **dictionary metadata upgrade** — backfill `good_scale_range` + `sweet_spot_scale` for existing 33 rows.
- **multi-pair node extension** — unlocks real composition; ~1 hour node work.
- **compose_iterative v0** — the threshold-gated corrective loop; ~200 lines, works today with sequential rendering (img2img chain or multi-pass FluxSpace calls).

## Invariants and rules of thumb

- `mix_b` is fixed at 0.5 (injection threshold cliff at ~0.45; below is no-op, above saturates).
- Effective `scale` range for identity-preserving smile: ~0.3-0.5. Above that, identity drifts > 50%.
- Prompt anchoring works for the **base** prompt (scale=0 image): `{age}` / `{ethnicity}` / `{gender}` placeholders pin those dimensions in the base. It does **NOT** work for the **edit δ** — putting the same anchor in pos+neg of an edit pair cancels out in the δ, so race/gender drift produced by the edit token attention is untouched (iter_04, 2026-04-23). Anchor the base; compose counter-edits to fix δ-side drift.
- **`FluxSpaceEditPair` averages the two halves' attention — it is NOT a contrastive subtraction.** Both halves should steer toward the *same* target with different secondary phrasing; the shared direction reinforces, the ambient confounds cancel. Building pairs as "positive vs opposite" (e.g. Latin American vs East Asian) produces a mixture of both attentions and pushes toward the mixture — wrong (race_iter_01, 2026-04-23).
- **Pair halves must be ADJACENT on the confound axis, not diametric, to cancel.** `smile_iter_06` (2026-04-23) tested Latin American halves paired with {East Asian, European, generic}. On elderly_latin_m:
  - Latin + East Asian → collapses to East Asian (the model's smile-attention is asymmetric; categorical opposites don't cancel, the stronger attractor wins).
  - Latin + European → stays visually Latino, minor drift to Middle Eastern per FairFace (adjacent buckets: Latin, Middle Eastern, European are neighbors on the race scale). **Best smile construction found so far on elderly_latin_m.**
  - Latin + generic → drifts to White.
  Design rule: pick halves whose classifier-space positions are adjacent (one or two buckets apart) to the base, not on the opposite side of the scale. Glasses/age worked because "short-generic" and "full-base-splice" are adjacent on the age axis (one reads ~30, the other ~50, averaging to ~40), not because they were opposite.
- **Always save attention caches.** `promptpair_iterate` and `execute_composition` now pass `measure_path=<render>.pkl` by default (2026-04-23). Each pkl is ~110 MB; budget ~6 GB per 54-render iteration. Existing pngs from iter_01/iter_03/iter_04 and race_iter_01 have no caches — re-render if we later need their δs for cache-δ axis extraction (milestone 8).
- Cross-base variance is large: dictionary rows must be per-base (solver's `base` filter enforces this).
- Visual inspection catches confounds the scalar metrics miss (initially). Surfaced race/gender into the iteration report; hair/posture/etc still visual-only.
- **Prompt sanitization is the gate, not pair-averaging or scale tuning** (2026-04-24, eye_squint v3 → v3.1). FluxSpace injects whatever semantic neighborhood the edit-prompt vocabulary evokes, not the AU label we attach. v3 used `"alert eyes"` (gaze-side connotation) and `"smiling warmly"` (smile connotation); pair-averaging cannot cancel a confound *both pairs share*, so smile + sidelook rode through. Removing those tokens — keeping the same math, scale, mix_b, base — moved identity-pass at α=1.00 from 6 % to 67 %. Same seeds, same bases, same algorithm; one base started laughing with eyes shut at α=1, the other squints at sunlight. **Audit edit-prompt vocabulary for connotational drift before turning any other knob.** Two checks: (a) name any *non-target* features each phrase suggests (smile? gaze direction? mood? age?); (b) confirm the two pair-halves disagree on those side features so they cancel.
- **Demographic stereotypes leak through prompt connotation, not just direct mention** (2026-04-24, `adult_middle_f`). `"Middle Eastern woman"` produced ~85 % headscarf renders with no veil mention in the prompt — Flux's prior. Same mechanism is why `"alert"` produces sideways gaze. Mitigation: re-anchor with named nationality + explicit feature list (`"a Lebanese woman with long dark wavy hair, olive skin, dark almond eyes, ..."`). Named nationalities have weaker stereotype priors in CLIP/T5 than regional labels because training data has more uncovered editorial photography of nationalities than of generic regions.
- **Identity-pass is bimodal across α, not monotone** (2026-04-24, eye_squint v3.1). Pass curve: 100 → 100 → 100 → 100 → 89 → 72 → 72 → 67 across α=0.00…1.00. There is a cliff between α=0.45–0.75 (gradual geometric drift), then a *plateau* — once the edit stabilizes, additional α changes the picture less. Operational consequence: don't waste samples in the 0.50–0.70 transition unless you specifically need that span; either stay in the safe plateau ≤ 0.45 or push to 0.75–1.0 where the corpus stops degrading.
- **Curator manifests are dominated by row-count, not row-quality** (2026-04-24). Scatter–gather across multiple corpora pulls more cells from whichever corpus rendered more PNGs at that α. v3 (50 cells/α × 8) outweighed partial v3.1 (18 cells/α × 8) in the manifest even though v3.1 had higher per-cell pass rates at α≥0.75. **Rebalance after each corpus-version round**: when a new corpus is strictly better at a given α range, drop the prior corpus from that range rather than averaging quality with quantity. Manifest construction must be quality-then-quantity, not quantity-first.

## Anti-regression flags

- **Atom-ridge injection is dead.** Don't propose re-trying without reading `2026-04-23-atom-inject-visual-failure.md`.
- **Scoring composite `target / (1 + 0.05·|age| + 1.5·|id_drift|)` over-penalizes identity drift.** Visible usable edits can score low. Always cross-check rank against visual collage.
- **"Same person" language did not meaningfully anchor identity.** Don't re-test v5_identity_anchor.
- **Report output must surface race + gender flips.** They aren't slopes — the summary originally hid them.
