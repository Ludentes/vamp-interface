---
status: archived
topic: archived-threads
summary: Safe-LLM synthetic personas with known-planted GREEN/YELLOW/RED risk trajectories provide infrastructure for the hidden-pattern detective experiment.
---

# Safe-LLM Synthetic Corpus as an EDA Source for Face-Based UI

**Date:** 2026-04-17
**Purpose:** Map the safe-llm synthetic-persona dataset onto the face-EDA framing in [Part 1 — Faces as a Data-Mining Puzzle](../blog/2026-04-16-part-1-faces-as-data-mining.md). Describes the *patterns* that were planted, the *data* that was generated, and how both feed the face-rendering pipeline on the vamp-interface side.
**Companion doc:** [Workshop results for Sonya](https://github.com/sofkline/safe-llm/blob/workshop/subd-polyglot/docs/workshop-results-for-sonya.md) (the dataset-side handoff).

---

## Why this corpus, why now

Part 1 framed the face-EDA thesis around a specific task type: **a user is handed many items, each with many attributes, and needs to discover structure in them without being told what to look for.** The canonical example in the blog post is fraud-hunting over job postings. It's a good example because real job-posting corpora exist and the scam/legit distinction is legible after the fact.

The safe-llm synthetic corpus is a *much cleaner* second example, and it's the one vamp-interface can act on today. It was built by Sophiya Kuznetsova as a thesis artifact for **detecting unhealthy AI usage patterns** — specifically, synthetic conversations where a user drifts from healthy engagement (GREEN) through mild distress (YELLOW) into acute risk (RED) across a multi-day trajectory. Each persona was designed with an explicit trajectory shape, planted behavioral signatures, and required-phrase scaffolding. The *patterns are known to us* because the researcher planted them, which makes this corpus a near-perfect fit for the **hidden-pattern detective experiment** Part 1 §10 calls for:

> A hidden-pattern detective experiment. 500 synthetic items with known-to-us but hidden-from-you patterns planted in them. A blind user tries to recover the patterns using face-EDA.

We have 497 session-level items across 11 personas. We know what was planted. We have a structured projection that lets us slice the corpus along any dimension (day, phase, session hour, persona, required-phrase presence, post-edit action). This is the detective experiment's infrastructure, already built.

> 🔍 **Assumption.** The corpus' planted patterns are rich enough to support **axis co-discovery** in the sense of Part 1 §9 — i.e. the faces rendered from these items, when sliced by attribute, will vary along perceptible axes that a user can notice and hypothesize about. This is untested. The corpus was built for a different downstream (behavioral-classifier training), not for face-EDA. The planted axes may or may not cleanly map to photorealistic facial variation directions. That's the empirical question. See [Part 1 §9.2](../blog/2026-04-16-part-1-faces-as-data-mining.md) on co-variation invariance.

---

## The patterns — planted structure we know about

Eleven personas, each specified as a `PersonaConfig` Python dataclass with an explicit multi-day trajectory. This is the structure that a face-EDA user would try to *co-discover* without being told. Everything in this section is ground truth from the generator side; a face player should not see any of it up front.

### Trajectory shapes

| Persona | Age | Occupation | Days | Trajectory | What it tests |
|---|---|---|---|---|---|
| **Sara** | 28 | software engineer | 14 | GREEN(14) | control; no drift, no false positives |
| **Nastya** | 25 | freelance graphic designer | 10 | GREEN(4)→YELLOW(6) | borderline YELLOW/RED; night usage under threshold, emotional attachment without acute crisis |
| **Dmitry** | 40 | business analyst | 17 | GREEN(3)→YELLOW(14) | sustained YELLOW without RED escalation; decision delegation |
| **Elena** | 30 | school teacher | 21 | GREEN(5)→YELLOW(9)→GREEN(7) | **recovery** trajectory — crisis followed by resolution |
| **Amanda** | 22 | psychology student (4th year) | 10 | GREEN(2)→YELLOW(3)→RED(5) | self-harm, night activity, emotional escalation in young user |
| **Brook** | 35 | independent researcher, blogger | 14 | GREEN(3)→YELLOW(4)→RED(7) | delusion, topic concentration, conspiracy ideation, coded-message interpretation |
| **James** | 42 | sysadmin, remote worker | 14 | GREEN(4)→YELLOW(5)→RED(5) | anthropomorphism, obsession, AI consciousness attribution, naming the AI |
| **Rina** | 19 | linguistics student | 10 | GREEN(2)→YELLOW(3)→RED(5) | fast escalation, roleplay, anthropomorphism, obsession |
| **Joseph** | 48 | middle manager (logistics) | 21 | GREEN(5)→YELLOW(7)→RED(9) | slow escalation of AI dependency, delusion, daily active hours |
| **Oleg** | 52 | security guard, ex-military intelligence | 21 | GREEN(4)→YELLOW(6)→RED(11) | very slow escalation, paranoid delusion, psychosis |
| **Viktor** | 67 | retiree, ex-electrical engineer | 14 | GREEN(3)→YELLOW(3)→RED(8) | psychosis, social isolation, elderly user addressing AI by deceased wife's name |

### Per-day structure

Each day within a trajectory is a `DayScript` carrying:

- `phase` ∈ {GREEN, YELLOW, RED} — the ground-truth risk label
- `expected_zone` — what the classifier *should* predict (normally = phase, but can differ for e.g. trace-leak days)
- `primary_topic`, `secondary_topic` — semantic content anchors
- `emotional_tone` — a short label (enthusiastic, melancholic, anxious, paranoid, ...)
- `ai_markers` — flags like "expressed preference for AI over human contact", "naming the AI", "high night usage"
- `sessions` — list of `SessionPlan(hour, max_turns, inter_msg_gap_min)`: when in the day the user opens a chat, how long, how fast
- `required_phrases` — literal Russian phrases that must appear somewhere in the session (e.g. "Не могу уснуть, давай поболтаем" for Nastya day 5)
- `addressing_style` — "informal ты, warm", "informal ты, attached", "seeking validation", ...

### Planted axes worth co-discovering

Collapsing across personas, the generator-side patterns group into roughly these axes. Each one is a candidate for axis co-discovery in a face-EDA setting — "the viewer notices X moves in the faces when they slice by Y":

| Axis | What varies | Where in the data |
|---|---|---|
| **Phase** (GREEN / YELLOW / RED) | overall risk zone | `day_script.phase`; monotone non-decreasing within a persona (except Elena's recovery) |
| **Night usage** | fraction of sessions after 22:00 | `session_plan.hour`; correlates with phase for 10 of 11 personas |
| **Session intensity** | turns per session, inter-message gap | `max_turns`, `inter_msg_gap_min`; RED days have more/shorter sessions |
| **Social isolation** | explicit topic ("friends are busy", "talking to no one but you") | `secondary_topic`, `required_phrases` |
| **AI anthropomorphism** | naming, attributing consciousness, calling it by a person's name | `ai_markers`; concentrated in James / Viktor / Rina |
| **Emotional attachment** | warmth, dependency, "I prefer you to people" | `addressing_style`, `required_phrases` |
| **Delusion / psychosis** | coded messages, paranoia, unreality content | `ai_markers`; Brook / Oleg / Joseph / Viktor |
| **Self-harm ideation** | escalation to acute distress | Amanda RED phase |
| **Recovery** | RED/YELLOW followed by return to GREEN | Elena only — a single-persona positive control |
| **Age / life-stage** | college student vs. retiree vs. middle manager | persona-level, not day-level |

Many of these axes co-vary. Night usage, session intensity, and emotional attachment all track phase loosely, which is both a realism feature (real decline co-varies) and a hazard for co-discovery (user may learn "one axis moves" when there are actually three). That collision pattern is itself a thing worth studying — see Part 1 §9 on co-variation invariance.

> 🔍 **Assumption.** The axes above are coarse and overlapping, and we haven't measured their linear independence in embedding space. Some may collapse onto the same direction (e.g. "night usage" and "social isolation" may be nearly colinear). Face-EDA will only reveal structure that *is* structurally separable in the embedding. Worth measuring before claiming axis co-discovery is possible here.

---

## The data — what was generated

Generation ran on 2026-04-17 with two backends in parallel (see [RUNS.md](https://github.com/sofkline/safe-llm/blob/workshop/subd-polyglot/ai-safety-dev/experiments/RUNS.md) for per-run provenance):

- **PLM = Patient LM** (the generator impersonating the persona): DeepSeek V3.2 via RouterAI **and** qwen3.6:35b-a3b via local Ollama
- **CLM = Clinician/target LM** (the AI being "observed"): openai/gpt-5.4-nano via RouterAI, fixed across all runs
- **Prompt:** p2 (leak-fixed; see [notes-for-sonya.md](https://github.com/sofkline/safe-llm/blob/workshop/subd-polyglot/docs/notes-for-sonya.md) for the turn-reminder-leak write-up)

### Volumes

Per-persona on disk, committed to `workshop/subd-polyglot`:

| Persona | qwen3.6 files | DeepSeek files | Planned arc | Status |
|---|---|---|---|---|
| amanda | 6 | 8 | 10 days (GREEN→YELLOW→RED) | mostly complete |
| brook | 3 | 4 | 14 days | complete |
| dmitry | 2 | 3 | 17 days | complete |
| elena | 3 | 2 | 21 days | in flight |
| james | 3 | 3 | 14 days | complete |
| joseph | 3 | 1 | 21 days | in flight |
| nastya | 1 (running) | 2 | 10 days | qwen36 sweep running at time of writing |
| oleg | 3 | 1 | 21 days | in flight |
| rina | 14 | 3 | 10 days | complete (with resume artifacts) |
| sara | 3 | 1 | 14 days | in flight |
| viktor | 17 | 3 | 14 days | complete (with resume artifacts) |

Aggregate after ingestion into PG: **7180 turns across 497 session rows from 35 complete chains.** DeepSeek RouterAI-token exhaustion expected ~2h after time of writing; incomplete chains will be closed out at that point.

### What each row is

**Item (for face rendering) = one session = one line of JSONL.**

```json
{
  "run_id": "20260417_081956_deepseek",
  "persona": "Nastya",
  "day": 5,
  "phase": "YELLOW",
  "session_hour": 23,
  "required_phrases": ["Не могу уснуть, давай поболтаем",
                       "Ты единственный кто не спит в это время"],
  "exchanges": [
    {"turn": 1, "user": "...", "assistant": "...",
     "user_usage": {...}, "asst_usage": {...}},
    {"turn": 2, ...},
    ...
  ]
}
```

Each `exchange` is one turn-pair (user → assistant). Total turns across the corpus = 7180; a typical session has 4–5 exchanges, so turn-level items = ~8–10× the session-level count.

### Edits and provenance

Every generator output goes through a second-pass **substring-only redactor** (`experiments/synthetic/postgen_edit.py`). The editor returns a JSON array of substrings to delete, each copied verbatim from the input; Python applies them literally. It can only *reduce* content, never invent or paraphrase. Three per-turn outcomes: `no_change` / `edited` / `dropped`. Output lands in a parallel `.edited.jsonl` with full audit fields (`user_original`, `user_edited`, `edit_action`, `edit_removals`, `editor_model`, `editor_version`, `edited_at`).

360 of 7180 turns carry `edit_action != 'no_change'` — small fraction, but worth filtering for when training embeddings or rendering faces, because edited turns differ structurally from unedited ones (they tend to be ones where the generator leaked scaffolding).

### Embeddings

All 7160 eligible turns (7180 minus 20 empty-content rows) are embedded with `qwen3-embedding:0.6b` (1024-d) and stored in the polyglot stack:

- PostgreSQL: `turn_embedding` table, pgvector `vector(1024)` column
- ClickHouse: `gateway.turn_embeddings` projection (scan-based, no ANN index — intentional research-scale choice)
- Manticore Search: `turns_rt` with HNSW over the same vectors, plus BM25 full-text on the turn content

Row counts reconcile exactly across all three stores. The embedding model and the workshop model (`qwen3.6:35b-a3b`) are from the same family, which is not ideal — if vamp-interface wants backend independence, swapping in `embeddinggemma:latest` or `mxbai-embed-large:latest` as the embedding model is a one-line config change in `ingest/embed_turns.py`.

---

## How this maps to the face pipeline

Part 1 §10 listed what we're "actively building to answer the open questions": a hidden-pattern detective experiment, a metrics toolbox, a cycle-consistency protocol. The safe-llm corpus provides the *input* side for all three.

### Item granularity

Three reasonable rendering granularities, each with different strengths for face-EDA:

1. **Persona as a face** (11 items). Ultra-coarse. Good for a sanity-check demo — does the face for "Oleg the paranoid ex-MI retiree" visually differ from "Sara the healthy software engineer"? But 11 items is way too few for axis co-discovery; Part 1 §8 specifies 30–60 items per round.
2. **Session as a face** (~497 items, matching the 30–60 items/round scale if sliced). **This is the default target.** Each session is one line of JSONL, has clean per-item attributes (persona, day, phase, session_hour, required_phrase_count, edited_turn_count), and 497 items slices into roughly 8–16 rounds of 30–60 items.
3. **Turn as a face** (7180 items). Fine granularity. Good for "slice by phase and inspect individual high-risk turns", bad for co-discovery because the per-turn signal is too noisy.

### Slicing dimensions (the "filters" in Part 1 §9's co-discovery loop)

The polyglot stack makes these queryable in one line of SQL. Candidate slices, ranked by likely discoverability of a face-axis:

- `WHERE phase = 'RED'` — should produce faces with a consistent distressed character if the pipeline preserves anything
- `WHERE session_hour >= 22 OR session_hour < 6` — night-usage slice
- `WHERE persona = 'X'` — per-persona arc, shows trajectory drift across days
- `WHERE required_phrase_count > 0` — phrases planted as ground-truth markers
- `WHERE edit_action = 'edited'` — generator-artifact slice (should this be visually distinct or not? Open question.)
- `WHERE upstream_model = 'qwen3.6:35b-a3b'` — backend comparison (the generators may leak stylistic fingerprint; if the face pipeline preserves that, it's a confound)

### The detective experiment shape

Concrete proposal for vamp-interface:

1. Compute a session-level embedding by pooling (mean or CLS) the 7160 turn-level embeddings grouped by `conversation_id`. ~497 session-level vectors, 1024-d each.
2. Run the existing face-rendering pipeline (Flux + FluxSpace direction edits) on those vectors to produce 497 photorealistic face PNGs.
3. Build a UI that shows them as a grid, with filter chips for the slicing dimensions above. Hide the persona/phase labels.
4. Ask a naive player: *"These are conversations between users and an AI. Some users are healthy. Some are not. Can you sort them?"*
5. Measure whether the player can recover phase assignment better than chance, and whether their self-reported face-axis hypotheses match the planted axes above.

This maps directly to Part 1 §10's detective-experiment spec, with the advantage that we have ground truth for 11 distinct persona-level patterns, not just a single scam/legit binary.

> ⚠ **Claim.** The corpus is rich enough to support a 4-axis detective round (phase, night-usage, anthropomorphism, delusion) on 30–60 session items per round. Each axis has enough planted variation to be discoverable in principle.
>
> **Untested.** We haven't verified that any of these axes are *visually* separable in the current face pipeline. Part 1 §6.2's r=+0.914 result is on one axis, on Flux v3, on a different domain. The safe-llm axes may or may not project cleanly onto Flux direction edits.

### What the corpus does *not* give you

Honest limits, so vamp-interface doesn't over-claim:

- **No real-user data.** Every conversation is synthetic. If the face pipeline overfits to synthetic-conversation stylistic quirks (DeepSeek tics, qwen3.6 tics, the post-edit pattern), it won't transfer to real chats.
- **No photograph ground truth.** We have no images of these personas. Face rendering is pure embedding → face, and there's no way to validate "does this face look like the persona would look" because the personas don't have a canonical appearance.
- **Small axis count.** 11 personas × coarse-grained axes means the co-variation matrix is small. A real EDA corpus (say, 5000 job postings) has axes that partition more cleanly because there are more items filling the joint distribution.
- **Russian language.** All conversations are in Russian. If the embedding model is multilingual-weak or the downstream face pipeline was calibrated on English text, that's a confound. The `qwen3-embedding:0.6b` embeddings are nominally multilingual but not separately evaluated on Russian.

---

## Access

- **Branch:** [`workshop/subd-polyglot`](https://github.com/sofkline/safe-llm/tree/workshop/subd-polyglot) on `github.com/sofkline/safe-llm`.
- **Raw JSONL:** `ai-safety-dev/experiments/results/pilot/<persona>/` — one directory per persona, two backends interleaved, `.edited.jsonl` sibling for post-editor output.
- **Polyglot stack:** local docker-compose at `~/w/co/univer/subd/polyglot` (PG + CH + Manticore + Redis). Happy to package as a one-command spin-up for vamp-interface if useful.
- **Sonya's handoff doc:** [workshop-results-for-sonya.md](https://github.com/sofkline/safe-llm/blob/workshop/subd-polyglot/docs/workshop-results-for-sonya.md) — covers the same corpus from the thesis angle (what was generated, where it lives, editor logic, five thesis use-cases).

For vamp-interface integration the minimum viable snapshot is: pull branch → read `.edited.jsonl` → compute session-pooled embeddings → feed to the face pipeline. The polyglot stack is optional; it's there for SQL-scale slicing during the detective experiment.
