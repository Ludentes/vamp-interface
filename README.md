# vamp-interface

An experiment in using photorealistic AI-generated faces as a data visualization medium. Each data point becomes a face. Fraudulent job postings develop uncanny, wrong-feeling faces. Legitimate ones look unremarkable.

The premise: humans process faces pre-cognitively. A grid of 200 faces can be scanned in seconds. A table of 200 rows cannot. If the encoding is right, the odd one out jumps out before the viewer can say why.

---

## Where we are

**V1 shipped and works.** We built it fast on an early version of the idea. It produces faces, they cluster, and high-fraud postings drift visibly. On the 543-job benchmark corpus the measured drift correlation `r(anchor_distance, sus_level)` reached **+0.914** with ArcFace IR101 cluster separation 0.2179 (Flux v3, 2026-04-07). That is a real result — the core hypothesis is not dead.

**V2 is a deliberate rebuild.** V1 is the thing that survived a lot of improvised decisions. We think we can do substantially better if we use the right tools and the right approach — but first we had to stop guessing. That meant building a formal evaluation framework *before* picking models, spaces, or training loops.

**We are at the beginning of V2. Step 1 is done.**

| Step | Status | Artifact |
|---|---|---|
| **1. Formal modeling** — what vamp-interface actually wants, mathematically | ✅ done | [v2/framework/](v2/framework/) |
| **2. Evaluation** — score candidate tools, papers, models, and latent spaces against the framework | ⏳ next | `v2/evaluation/` (empty) |
| **3. Rebuild plan** — the actual V2 implementation plan | ⏳ after step 2 | `v2/plan/` (empty; [v1/plans/rebuild-plan-draft.md](v1/plans/rebuild-plan-draft.md) is an early draft that predates the framework) |
| **4. Implementation** | ⏳ after step 3 | `src/` |

Most of the codebase's activity in the next weeks will be in `v2/evaluation/` — running the framework against specific candidates and producing measurement-backed comparisons, not vibes.

---

## Repo structure

```
README.md                        (this file)

v2/                              current V2 work
  README.md                      V2 state and roadmap
  framework/                     ── LOAD-BEARING ──
    math-framework.md            THE framework (version in header, currently v0.6)
    README.md                    framework directory readme
    sources/                     supporting inputs (theory-constraints, blind-alleys,
                                 math-foundations synthesis + raw research)
  evaluation/                    (empty) candidate scoring grids
  plan/                          (empty) V2 implementation plan

v1/                              shipped V1 artifacts
  README.md                      what V1 is, measured results, why we're rebuilding
  src/                           V1 Python (embedding, clustering, generation, scoring)
  plans/                         V1 phase plans, early rebuild draft, deeper-research queue

src/                             (empty) V2 code will live here
docs/
  design/                        enduring design docs (scenarios, diffusion-approach)
  research/                      exploratory reads — background, not daily reference
  shaping/                       scam-guessr product shaping (cross-version)
  runbooks/                      operational notes
  review/                        literature review outputs

comfyui/workflows/               ComfyUI workflow JSON files (V1 uses these)
output/                          local generation artifacts (gitignored)
```

---

## The framework is the single load-bearing artifact right now

Before the framework existed, every design decision was a debate. Three session passes independently re-derived the same conclusion about StyleGAN vs. diffusion without citing the prior decision. We were vacillating because there was no basis for comparison.

The framework fixes that. It specifies:

- **Three channels:** identity (geometric embedding), editorial (semantic labeling), drift (perceptual salience amplifier).
- **Top-level acceptance criterion:** `P*` — downstream human-task accuracy. All other metrics are explicitly proxies.
- **Identity rubric:** `P4a` rank preservation (global Spearman ρ + local k-NN / trustworthiness / seed stability), `P4` Fisher ratio upper-bounded by the source space via Information Bottleneck, `P6` σ-injectivity.
- **Editorial rubric:** `E1` readability, `E2` information contribution (ablation), `E3` identity leakage, `E4` scalability.
- **Drift rubric:** `D1` factor-wise inconsistency (not raw off-manifold distance), `D2` monotonicity, `D3` identity preservation at y=0, `D4` mismatch signature, `D5` reversibility.

All rubric cells are anchored on specific theorems (Venna & Kaski trustworthiness, Mémoli Gromov-Wasserstein, Tishby Information Bottleneck, Brenier optimal transport, Kätsyri uncanny-as-perceptual-mismatch) rather than on hand-waving.

Read the framework before making any V2 architectural proposal: [v2/framework/math-framework.md](v2/framework/math-framework.md).

---

## What V1 achieved (briefly)

- **543 jobs × 5 generation versions measured** on 2026-04-07. Flux v3 (10 hand-curated work_type archetypes + LoRA drift) is the strongest baseline: `r=+0.914`, ArcFace cluster separation 0.2179.
- **Drift mechanism built and measured.** `Cursed_LoRA_Flux + Eerie_horror` at `sus_factor` strength produces monotone ArcFace anchor-distance with `sus_level`. The drift subsystem is not a future experiment — it exists, it's measured, and it works.
- **Same-archetype collapse is a structural feature, not a bug.** Physical-work postings (cleaning, construction) genuinely cluster in qwen embedding space; the editorial layer (clothing differentiation) is what makes them visually distinguishable. V2 credits this properly.

Full details: [v1/README.md](v1/README.md).

---

## Related projects

- **telejobs** — the data source. Job posting scraper + fraud scorer. vamp-interface consumes its `jobs` table.
- **scam-guessr** — the reveal/engagement game that rides on top of vamp-interface. Product shaping in [docs/shaping/](docs/shaping/).

---

## The name

From Watts' *Blindsight*.
