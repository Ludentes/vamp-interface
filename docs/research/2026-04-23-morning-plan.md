---
status: live
topic: metrics-and-direction-quality
summary: Morning-of briefing after the overnight Phase-4 broad validation — what to check first, what decisions depend on the result, and what fallback work to queue.
---

# Morning plan — 2026-04-23

## Overnight artefacts (if everything ran)

- `output/demographic_pc/direction_inject_broad/atom{NN}_directions_resid_causal/`
  - 20 PNGs per atom (2 bases × 2 seeds × 5 scales)
  - `blendshapes.json` — MediaPipe-scored
- ~320 renders × 400 KB ≈ 130 MB total, negligible
- Broad run log: `/tmp/vamp_broad.log`

## First thing to run

```bash
uv run python -m src.demographic_pc.morning_dashboard
```

One command, runs: `analyze_direction_validation` → `analyze_ridge_vs_causal`
→ `direction_inject_broad_collage` → regional summary (cos > 0.3 atoms
grouped by facial region via `atom_region_labels`).

If you want the validation stage alone: `uv run python -m
src.demographic_pc.analyze_direction_validation` — saves
`output/demographic_pc/direction_inject_broad/validation_report.json`.

**Reading the output:**

| pattern | meaning |
|---|---|
| cos > +0.5 at s=1–4, monotone in scale | direction WORKS for this atom — Phase 4 success |
| cos ≈ 0, delta_norm grows with scale | injection perturbs off-target (predictor-not-controller) |
| cos flips sign at high scale, delta_norm explodes | breakdown — too much injection |
| cos < 0 | direction points backwards — sign error somewhere |

## OVERNIGHT VERDICT: near-zero wins

1/16 atoms pass cos > 0.3 at max scale (s=4.0). Winner: **#07 eye-blink at +0.323**. Everything else at or below noise; 5 atoms (#1, #12, #13, #17, #18) have *negative* cos at max scale — injection moves blendshapes *away* from target.

| rank | atom | region | cos @ s=4 |
|---|---|---|---|
| 1 | #07 | eye-blink | **+0.323** |
| 2 | #11 | eye-gaze-h | +0.226 |
| 3 | #15 | brow-down | +0.158 |
| 4 | #16 | mouth-stretch | +0.140 |
| 5 | #18 | eye-gaze-v | +0.132 |
| ... | | | |
| 15 | #12 | mouth-stretch | −0.092 |
| 16 | #17 | mouth-smile | −0.190 |

Pattern: eye atoms lean weakly positive, mouth atoms scatter around zero or negative, brow-up actively anti-correlated. **This is an architectural failure, not a fit failure** — even the actual mean-training-delta direction does not cause the atom change at inference, despite perfectly correlating with it in training.

**Conclusion to sit with:** linear directions in Flux attention space are not controllers. Predictor-vs-controller gap isn't closed by taking the causal direction instead of the ridge one. Full report: `output/demographic_pc/direction_inject_broad/validation_report.json`. Per-atom scale sweeps: `output/demographic_pc/direction_inject_broad/_collages/`.

Follow the decision tree under **If validation shows near-zero wins** below.

## Key finding ready for your review

Running `analyze_ridge_vs_causal.py` on the two fitted direction sets shows:
- **100% overlap in top-K site selection** (both methods agree on which (step, block) positions matter for each atom)
- **Cosine similarity ≈ 0.01–0.05** within those shared sites (the actual direction vectors are nearly orthogonal)
- **Magnitude ratio 10–300×** (causal >> ridge) — ridge was heavily L2-shrunk

This is textbook predictor-vs-controller. Ridge picked directions that *correlate* with the atom (minimize prediction loss); causal picked the mean edit *observed* when the atom was high. Same diagnostic sites, opposite directions within them. If the overnight validation shows causal works and ridge doesn't, this diff crystallizes why.

## Context: what we tried last night

1. **Ridge directions** (`directions_resid.npz`) — scale 0.05–50000 produced `normal → blur → noise`, no semantic edit. Max row-norm ~0.1 (1% of attn norm 7.5). Classic predictor-not-controller; the learned weights correlate but point off-manifold.
2. **Causal directions** (`directions_resid_causal.npz`) — contrast-mean of delta_mix at high-atom vs low-atom training samples. Max row-norm 3–38 (~40–500% of attn norm). Smoke at scales 0.05–1.0 showed subtle visible shifts but nothing dramatic; the overnight broad run scores whether blendshapes actually move in the target direction.

## Decision tree

### If validation shows wins (≥6 atoms with cos > 0.3)

- Pick 2–3 best atoms for a polished demo sweep at more seeds / bases.
- Draft blog update covering: per-base residualisation architecture, causal-direction fix, Phase 4 validation results.
- Schedule format-v2 measurement writer (the fp16 `.npz` extension-based version from `2026-04-23-corpus-rebalance-todo.md`).

### If validation shows partial wins (2–5 atoms)

- Investigate atom-level correlates: do the winning atoms share a region (mouth vs brow)? CV R²? cos-sim between ridge and causal direction? Build `analyze_ridge_vs_causal.py` to diff the two sets.
- Consider richer causal fits: instead of top-20 / bottom-20 contrast, use multiple quantile bins and weight by the linear trend (projection pursuit / PLS).

### If validation shows near-zero wins

- The issue is architectural, not fit. Options:
  1. **Per-token direction** — we trained against `mean_d` (per-channel mean over tokens). Refit to predict per-token attention patterns rather than the mean. Costly to fit, but actually on-distribution.
  2. **Text-pair replay** — fall back to FluxSpaceEditPair with edit-prompt pairs chosen per atom by finding the training pair whose delta_mix most closely matches `W_atom[k]`. Uses the mechanism that's already known to work.
  3. **Concept Sliders (ECCV 2024)** — train a per-atom LoRA as described in `memory/project_vamp_measured_baseline.md`. ~10 min training per atom, uses LoRA mechanism that's known to inject semantically.

### If validation didn't even run / perms blocked

- `/tmp/vamp_broad.log` will tell you where it stopped. Re-launch: `nohup uv run python -m src.demographic_pc.direction_inject_broad --run --score > /tmp/vamp_broad.log 2>&1 &`

## Pending admin items

- **Measurement format v2** (fp16 `.npz`, extension-based versioning) — not done overnight. See `2026-04-23-corpus-rebalance-todo.md` § "Phase 4 render-batch format hard rule". Blocks any new high-volume render+measurement run, but overnight was PNG-only so this didn't bite.
- **Kill the stale processes** — `pgrep -f tune_directions_resid` and kill any leftovers; that job was stuck.
- The archived pkls on `/media/newub/Seagate Hub/vamp-interface-archive/` are cold storage; don't mount unless we need a field from raw v1 pkls.
- The `superpowers:code-reviewer` review of the injection node flagged only the perf/fp8 items, which were applied. No follow-ups.

## Context docs

- `memory/feedback_disk_preflight.md` — updated pkl-size estimate (112 MB, not 50 MB)
- `memory/feedback_measurement_format_versioning.md` — extension-based version rule
- `memory/reference_external_pkl_archive.md` — where the archived pkls live
- `memory/reference_comfyui_paths.md` — ComfyUI + custom-node paths
- `docs/archive-locations.md` — external drive inventory
- This doc — overall morning briefing
