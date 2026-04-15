# vamp-interface V2

Current work. V2 is the deliberate rebuild that replaces V1's improvised decisions with a framework-backed design. V1 is at [`../v1/`](../v1/).

---

## Where V2 is

| Step | Status | Location |
|---|---|---|
| **1. Formal modeling** — what vamp-interface wants, mathematically | ✅ done (v0.5) | [framework/](framework/) |
| **2. Evaluation** — score tools / papers / models / spaces against the framework | ⏳ next | `evaluation/` (empty) |
| **3. Rebuild plan** — the actual V2 implementation plan | ⏳ after 2 | `plan/` (empty; V1's early draft at [../v1/plans/rebuild-plan-draft.md](../v1/plans/rebuild-plan-draft.md) predates the framework and is not current) |
| **4. Implementation** | ⏳ after 3 | `../src/` (empty) |

We are in step 1.5 — framework done, starting evaluation.

---

## framework/ is load-bearing

The framework is the single document that governs V2 decisions. Every architectural proposal must cite rubric cells; every candidate must be scored before it enters the comparison. The framework also documents what it does *not* do — the list of questions it explicitly defers, so we don't re-debate them.

Read the framework before touching evaluation or plan: [framework/math-framework.md](framework/math-framework.md).

Framework directory details: [framework/README.md](framework/README.md).

---

## evaluation/ (next)

This will contain scored candidate grids. Each grid applies the framework's three channel rubrics (identity / editorial / drift) plus the top-level `P*` user-task criterion to one candidate or candidate family. Expected format:

- `evaluation/<candidate-slug>/scoring.md` — rubric cells filled in, each tagged `{measured | inherited | assumed}` per the framework's P13 / E13 / D13 metadata discipline
- `evaluation/<candidate-slug>/measurements/` — raw measurement outputs, scripts, reproducibility notes

The first candidate to score is the V1 baseline itself — Flux v3 anchor-bridge — to convert the measured `r=+0.914` into a fully-characterized framework entry (global + local rank preservation, seed stability, Fisher ratio, editorial contribution ablation, drift factor-mismatch breakdown). See framework §5 experiments 1, 6, 7.

---

## plan/ (after evaluation)

The V2 implementation plan comes out of the evaluation grid. It will name one identity candidate, one editorial mechanism, and one drift mechanism as the V2 triple, with measured justification for each and explicit costs (training time, license, inference latency, per-corpus throughput).

No guesses. No "this seems like a good idea." Every choice traceable to a measurement or a cited theorem.

---

## Why this structure

Three independent session passes in April 2026 re-derived the same conclusions about StyleGAN vs. diffusion without citing the prior decision. The fix was to write the framework *before* picking tools, then make the framework the single source of truth for what "better" means. V2's directory layout reflects that: framework first, evaluation second, plan third, implementation fourth — not the other way around.
