# Hidden Pattern Detective Experiment

**Date:** 2026-04-15
**Version:** v0.1
**Status:** Active proposal. Validation methodology for face-EDA and prototype for the daily-puzzle product direction (see [../../v2/framework/rationale.md](../../v2/framework/rationale.md) Option B/C).

> ⚠️ **Spoiler warning for the project owner.** This document describes the methodology, not the specific patterns to plant. If you (the project owner) intend to play the Detective role in the first session, stop reading at the end of Phase 1 and have a second person — or a fresh, isolated LLM session — execute pattern design. Reading Phase 1's example pattern categories is OK; reading the actual `ground_truth.sealed.json` is not.

## Purpose

Validate face-EDA as a discovery tool by constructing a synthetic corpus where ground-truth patterns are known to the experimenter and hidden from the user, then measuring pattern recovery.

This validates framework properties P4a (cluster-conditional rank preservation), P10 (co-variation invariance), and §2.5.2 (joint Y-channel constraints) under conditions where ground truth is known. A successful detective run is necessary-but-not-sufficient evidence for framework correctness; a failed run is diagnostic.

## The frame

Classic Agatha Christie structure. The experimenter plants clues; the Detective recovers them. "Data mining as puzzle." Success = number of real patterns discovered within a time budget, weighted by confidence and specificity.

Downstream, the same structure packages as a **Wordle-shaped daily puzzle for everyone** (see Phase 7). The research prototype is 500 items / 60 min / analyst-style. The consumer puzzle is 30–60 items / 5–10 min / daily. Same mechanism, different parameters.

## Roles

| Role | Who | Knowledge |
|---|---|---|
| **Experimenter** | Executes Phases 1–3; does NOT use the discovery system | Full knowledge of planted patterns |
| **Detective** | Executes Phase 4; does NOT read Phase 1 output | Blind to ground truth until reveal |

For iteration 1, suggested roles: **Claude (fresh session, Phase-1-isolated)** plants patterns, **project owner** plays Detective. Alternative: two collaborators split the roles.

**Critical:** the Experimenter and Detective must not share a context. If the same person is both, use different sessions separated by enough time that specific patterns are forgotten, or have an LLM execute Phase 1 with explicit instructions to write only to the sealed file and not summarize.

## Phase 1 — Pattern design (Experimenter, sealed)

This phase is methodology. The specific patterns chosen at execution time are sealed and never shown to the Detective.

### Design principles

1. **Multi-dimensional.** ≥3 independent factors, so a discovery surface exists.
2. **Include at least one interaction.** A cluster that exists only at a specific conjunction of axis values — not readable from any single axis.
3. **Include at least one tight cluster with a surface-readable signature.** A small subgroup with a distinguishing phrase or shared template, so at least one pattern is reliably discoverable.
4. **Include distractors.** Noise dimensions that look like they might matter but don't, so the Detective has to reject hypotheses, not just collect them.
5. **Nameable.** Each pattern must have a short verbal description the Detective could plausibly produce (e.g., "urgency increases over time"), or scoring is impossible.
6. **Tunable subtlety.** Start obvious. Iteration 2+ dials down.

### Pattern type inventory (methodology, not specific content)

| Type | What it is | Role in discovery surface |
|---|---|---|
| Orthogonal axis | A stylistic or content dimension varying independently across the corpus | Base discoverability; recoverable via filter + compare |
| Interaction cell | A small region defined by conjunction of specific axis values | Tests whether user can discover structure not present on any single axis |
| Signature cluster | Tight subgroup with shared phrase / template | Guarantees at least one "aha" moment early |
| Temporal drift | Pattern intensity varies with a synthetic timestamp metadata field | Tests whether user notices and uses the time slider |
| Decoy | Surface-obvious pattern that's actually random | Tests precision; catches over-eager hypothesizers |

### Recommended shape for iteration 1 (research prototype, 500-item / 60-min version)

- **3 orthogonal axes** (3 levels each)
- **1 interaction cell** at a chosen conjunction
- **1 signature cluster** (~20 items with a template phrase)
- **1 temporal drift** on one of the axes
- **1 decoy** (surface-obvious but uncorrelated with anything real)

Total: **7 planted items**, of which 6 are real patterns and 1 is a decoy. Detective should recover the 6 real ones and reject the decoy.

### Domain

Pick a text domain where:
- LLM can generate ~150-word items with controlled stylistic variation
- Natural multivariate structure (multiple plausible axes)
- Synthetic-friendly (no real-world ground truth to conflict with)
- Close enough to scam-detection to be relevant but not literal

**Recommended:** short-form classified ads (generic items for sale). Close to job postings without being identical, multivariate by nature, synthetic-friendly. Alternatives: dating profiles, short product reviews, rental listings.

### Artifacts produced by Phase 1

- `experiment/patterns.sealed.md` — full pattern definitions with examples. Never shown to Detective.
- `experiment/generation_prompts.jsonl` — LLM prompts per item, tagged with ground truth. Never shown to Detective.

## Phase 2 — Corpus generation

### How many texts? — **500 for the research prototype**

Rationale:
- ~10–15 exemplars per cell in a 3×3×3 factorial (27 cells) = 270–405; round up for distractor spread and reserve capacity → 500.
- 500 fits a single paginated face grid at 80×80 thumbnails. Visually scannable.
- Flux batch generation: ~5 s/image at 256×256 with the current v3 pipeline → ~40 min for 500.
- Embedding + projection: ~5 min on existing qwen3-embedding + PCA pipeline.
- Total compute: ~1 hour.

For the Wordle-packaged daily-puzzle version: **30–60 items.** See Phase 7.

### Generation protocol

1. Sample 500 `(axis1, axis2, axis3, distractor1, distractor2, distractor3, synthetic_date)` tuples. Weight slightly toward the interaction cell to ensure the cluster is visible (~30 items).
2. For each tuple, construct an LLM prompt encoding the axis levels as tone/style/content directives. The prompt must NOT name the axes — it must embed them in writing instructions.
3. If the tuple is in the signature-cluster subset (~20 items), inject the template phrase.
4. Call LLM. Recommended: **glm-4.7-flash:latest via local Ollama** (per user preferences) or gpt-oss:latest. Claude Haiku acceptable for speed.
5. Store `{id, text, ground_truth}` tuples.
6. Split into two files:
   - `experiment/corpus.jsonl` — `{id, text, distractor_metadata}` only, shown to Detective.
   - `experiment/ground_truth.sealed.json` — full pattern assignments, held until reveal.

### Quality check before release

- **Spot-read 10 random samples.** Does the experimenter recognize their own planted patterns by eye in the obvious cases? If not, patterns are too subtle or the LLM disobeyed — regenerate.
- **Embedding cluster check.** Compute pairwise cosines within vs. across interaction-cell membership. Within-cell should be meaningfully higher. If not, patterns are not embedding-visible → face-EDA can't work → stop and diagnose before spending compute on Flux.
- **Signature cluster sanity.** Substring-search the template phrase; confirm ~20 hits.
- **Decoy check.** The decoy pattern should have zero embedding signal.

## Phase 3 — Pipeline build

Reuses the measured-baseline v3 pipeline (see memory: Flux v3, r(sus)=+0.914). Four steps.

### 3.1 Embed

- Model: `qwen3-embedding:0.6b` via Ollama at localhost:11434.
- Input: 500 texts → 1024-d vectors.
- Storage: `experiment/embeddings.npy`.

### 3.2 Project

- PCA to ~32 dims, whiten.
- Compose into CLIP-conditioning shape for Flux.
- Reuse existing projection code from v3 baseline.
- Storage: `experiment/projected.npy`.

### 3.3 Render

- Flux img2img with fixed neutral anchor.
- Denoising conditioned on projected vector.
- Same seed per item id → deterministic, byte-stable.
- 500 PNGs at 256×256.
- Storage: `experiment/faces/{id}.png`.

### 3.4 Discovery UI

Minimal viable detective interface. **Streamlit recommended** (fastest path, fits a day of build).

Required operations:
- **Face grid** (paginated if Streamlit struggles at 500 — page size 100).
- **Text substring filter.**
- **Metadata filters:** synthetic-date slider, distractor dropdowns.
- **Hover-for-text** (click-to-expand acceptable if hover is awkward in Streamlit).
- **"More like this"** — nearest neighbor in embedding space, reorders the grid.
- **Free-text hypothesis notepad** — persistent within session.
- **Confidence slider (1–5) per hypothesis.**

Non-requirements for v1: multi-user, auth, styling polish, mobile, replays.

Storage: `experiment/ui/app.py`, session state in `experiment/sessions/`.

## Phase 4 — Detective session

### Setup

- Detective is blind to pattern definitions.
- 60-minute uninterrupted budget.
- Experimenter unavailable (no hinting, no watching).
- Detective may take notes but may not consult external tools.

### Instructions to Detective (verbatim)

> This corpus contains classified ads. Hidden in them are between 3 and 10 patterns — axes along which items vary, interaction cells that form archetypes, signature clusters with shared phrases, and possibly temporal effects. Some apparent patterns may be decoys. Your job is to discover as many real patterns as you can and describe each in one sentence ("I think X varies along Y"). Rate each hypothesis 1–5 confidence. Write down which UI operations led to each hypothesis — this helps us improve the tool regardless of whether you're right. You have 60 minutes.

### Logged during session

- Timestamp of each hypothesis submission
- Hypothesis text (verbatim)
- Confidence (1–5)
- Brief provenance note ("I noticed this after filtering by X and seeing Y cluster")

## Phase 5 — Reveal and scoring

### Unseal

Open `ground_truth.sealed.json`. Walk through with Detective.

### Scoring rubric

For each Detective hypothesis, Experimenter labels:
- **Hit** — matches a planted pattern with correct direction and scope
- **Partial hit** — matches a pattern but gets direction, scope, or cell wrong
- **Miss** — does not correspond to any planted pattern
- **Decoy bite** — matches the planted decoy (negative credit)
- **Unplanted truth** — correct pattern we didn't plant but that emerged from LLM generation artifacts (noteworthy, not scored)

### Metrics

- **Recall** = (hits + 0.5 × partial) / (total planted real patterns). Target ≥0.5 for v1.
- **Precision** = (hits + 0.5 × partial) / (hits + partial + misses + decoy bites). Controls for guess-spam.
- **Time to first hit** — minutes from session start to first correct hypothesis. Proxy for UI ergonomics.
- **Confidence calibration** — mean confidence of hits vs. misses. Well-calibrated detective rates hits higher.
- **Decoy rejection** — did the Detective resist the decoy? Binary.

### Interpretation

| Recall | Precision | Conclusion |
|---|---|---|
| ≥0.7 | ≥0.5 | Discovery loop works. Proceed to real-corpus validation on telejobs; proceed to Phase 7 Wordle packaging. |
| 0.4–0.7 | ≥0.4 | Partial. Analyze which pattern types recovered (axes? interactions? clusters?). Reshape the weakest component. |
| <0.4 | any | Face channel not conveying discoverable structure. Return to framework §5 Exp A–C. Diagnose: embedding, projection, generation, or UI. |

## Phase 6 — Iteration

If iteration 1 succeeds:

1. **Replicate with new patterns.** Experimenter writes a fresh sealed spec. Same Detective tries again. Measures whether session-1 learning transfers to session 2.
2. **Add a second Detective.** Fully independent person. Replicates the result with no shared context.
3. **Dial down subtlety.** Reduce pattern strength and add more distractors until recall starts to collapse. Find the floor of what's discoverable. This is the "difficulty curve" calibration for Phase 7.
4. **Cross-domain replication.** Swap classified ads for a different text domain (e.g., product reviews). Do patterns still recover? Tests generality of the approach.

## Phase 7 — Wordle packaging (consumer puzzle)

If Phases 1–6 validate the mechanism, scale down to a daily-puzzle format.

### Parameter changes

| Parameter | Research prototype | Wordle package |
|---|---|---|
| Corpus size | 500 | 30–60 |
| Time budget | 60 min | 5–10 min |
| Patterns per puzzle | 6–7 | 3–4 |
| Decoys | 1 | 0–1 |
| Shareable result | No | Yes (compact, spoiler-safe) |
| Cadence | Ad hoc | 1 puzzle per day, same for everyone |
| Audience | Researchers / testers | General public |

### The round shape

1. User opens today's puzzle. 30–60 faces rendered in a grid.
2. User has a filter bar (text search, metadata sliders), a hypothesis notepad, and a 10-minute timer.
3. User submits up to 5 hypotheses as one-sentence claims.
4. At submission, each hypothesis is auto-graded against the sealed ground truth. Reveal shows which hit, which missed.
5. User gets a shareable result string.

### Shareable result format

Wordle's viral mechanism was the spoiler-safe emoji grid. Candidates for our analog:

```
Dialect #012 — 3/4 🔍
✅✅✅❌ ⏱ 6:42
vamp.fyi/012
```

- `#012` — daily puzzle number
- `3/4` — hits out of total patterns
- emoji row — hit/miss per pattern in submission order
- `⏱` — solve time
- Short link for the day

Spoiler-safe because it shows count and timing but not what the patterns were.

### Corpus generation for daily puzzles

Must be automated. LLM-driven pattern planting + generation + sealing, running on a cron or pre-generated batch. Recommended: pre-generate 30 days of puzzles at a time, hold in sealed storage, release one per day.

### Open design questions for Phase 7

1. How do we auto-grade free-text hypotheses against ground truth? Options: structured submission (pick from dropdown), semantic similarity match against canonical descriptions, LLM grader with rubric. The LLM grader is easiest but introduces model-dependence.
2. What's the difficulty curve? Wordle keeps all puzzles at roughly the same difficulty; we'd likely do the same, calibrated from Phase 6 Exp 3.
3. How many patterns should be "obvious" vs. "subtle" per puzzle? Wordle analog: most players get it in 4. Most Dialect players should get 2 of 4 easily, 3rd with work, 4th as the kicker.
4. Archetype dictionary. Over many puzzles, the same archetype types may recur (interaction-cell, signature-cluster, temporal-drift). Do we let players learn the meta-vocabulary, or keep each puzzle's patterns totally fresh?

## Artifacts (full tree)

```
experiment/
  patterns.sealed.md            — pattern specs (Experimenter only)
  generation_prompts.jsonl      — per-item generation prompts (Experimenter only)
  corpus.jsonl                  — text + distractor metadata (visible to Detective)
  ground_truth.sealed.json      — sealed answer key (released at Phase 5)
  embeddings.npy                — qwen3 1024-d vectors
  projected.npy                 — PCA / CLIP-conditioning vectors
  faces/{id}.png                — 500 rendered faces
  pipeline/
    embed.py
    project.py
    render.py
  ui/
    app.py                      — Streamlit discovery interface
  sessions/
    YYYY-MM-DD-detective-<name>.md  — session transcript + scoring
```

## Time estimate (iteration 1)

| Phase | Effort |
|---|---|
| 1 — pattern design | 2–3 h |
| 2 — corpus generation | 1 h compute + 1 h QA |
| 3 — pipeline build | 1 day (most is reusing v3 + Streamlit glue) |
| 4 — detective session | 1 h |
| 5 — reveal + scoring | 30 min |
| **Total for iteration 1** | **≈2 days focused work** |

Phase 6 iterations: ~1 day each after the first.

Phase 7 packaging: separate project scope, blocked on Phase 6 success.

## Relationship to framework

This experiment validates:
- **§P4a** (cluster-conditional rank preservation) — if faces in the same cell cluster together, interaction-cell recovery succeeds.
- **§P10** (co-variation invariance) — if filtering by one axis consistently moves one face feature, co-discovery is supported.
- **§2.5.2 Y1 cross-axis orthogonality** — entangled axes will show as Detective confusion ("the smile moves whenever I filter anything").
- **§5 Exp A–C** — this experiment is a complement, not a replacement. A/B/C probe the direction basis and λ-sweep at the pipeline level; this experiment probes recoverability in user hands.

Failure modes map to framework diagnoses:
- Detective recovers axes but not interactions → P4a cluster-conditional not holding → projection or generation issue.
- Detective confused on every axis → Y1 orthogonality violated → FluxSpace direction basis entangled.
- Detective recovers embedding-visible patterns but not Flux-visible ones → loss in projection or rendering → diagnose via Exp B.
- Detective recovers nothing → channel is noise → return to framework §5.

## Changelog

- **v0.1 (2026-04-15):** initial spec. Phases 1–6 as research prototype, Phase 7 as Wordle packaging. Corpus size 500 justified. Reuses v3 pipeline.
