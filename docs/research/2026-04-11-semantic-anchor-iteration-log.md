# Semantic Anchor v3 — Iteration Log

**Date:** 2026-04-11
**Spec:** [`../superpowers/specs/2026-04-11-semantic-anchor-face-generation-v3.md`](../superpowers/specs/2026-04-11-semantic-anchor-face-generation-v3.md)
**Plan:** [`../superpowers/plans/2026-04-11-semantic-anchor-face-generation-v3.md`](../superpowers/plans/2026-04-11-semantic-anchor-face-generation-v3.md)

Design drifts and dead ends during v3 implementation. Captured so future
iterations don't re-walk the same mistakes.

## Pivot 1: wrong embedding model in CLAUDE.md

**Wrong assumption.** CLAUDE.md claimed jobs in postgres were encoded with
`mxbai-embed-large`. This was wrong — they were encoded with
`qwen3-embedding:0.6b`.

**Diagnosed by.** Re-embedding a known job's `raw_content` with both models
and comparing cosines to the stored embedding. Result:
- cos vs qwen3-embedding:0.6b = **1.0000** (identical)
- cos vs mxbai-embed-large = −0.0202 (random)

**Fix.** Updated `CLAUDE.md` and the v3 spec to reference
qwen3-embedding:0.6b. The spec's `mxbai-embed-large` references were all
editorial — no code changes needed. Committed as `e96649b`.

**Lesson.** Always run a round-trip verification before trusting project
documentation on embedding model identity. Dimension alone (both are 1024-d)
won't distinguish them.

## Pivot 2: phrase-list queries vs prose queries

**Initial guess.** 30 Russian candidate anchors written as space-separated
keyword phrases: `курьер пеший доставка еды в городе`. This pattern comes
from search engine query habits.

**Why it failed.** qwen3-embedding is trained on document-shaped Russian
text. Job postings in the corpus are 1-3 paragraphs of prose; encoding
keyword phrase-lists produces vectors systematically further from any
actual job than prose of the same role would.

**Diagnosed by.** Hand-crafting multiple query formulations for "food
courier" and comparing cosines to 5 known courier jobs:

| formulation | avg cos |
|---|---|
| `plain` phrase-list | 0.474 |
| `long_sentence` | 0.544 |
| `jobpost_style` (prose) | **0.588** |

Prose beat phrase-list by +0.11 average cosine on matched jobs.

**Fix.** Rewrote all 34 candidates as 1-2 sentence job-posting prose.

**Lesson.** Match query format to the document format the retrieval model
was trained on. For job-posting corpora, queries should *look like job
postings*, not like web-search queries.

## Pivot 3: missing asymmetric retrieval prefix

**Initial guess.** Encode queries the same way documents were encoded — as
plain text with no special prefix.

**Why it mattered.** qwen3-embedding is an **asymmetric retrieval model**:
query-side text expects an instruction prefix, document-side text does not.
Jobs in postgres were encoded as plain documents (verified via the Pivot 1
round-trip). Queries need the prefix to land in the correct "query region"
of the learned space.

**Diagnosed by.** Same diagnostic as Pivot 2. Adding the prefix `"Инструкция: Найди вакансии, соответствующие роли.\nЗапрос: "` raised avg cos on matched
courier jobs from 0.474 → 0.566 (+0.09).

**Fix.** `encode_anchors.py` prepends the instruction prefix to every query
before calling Ollama. Documents stay unmodified.

**Lesson.** For any retrieval-tuned embedding model, look up the official
query/document format. Using one side's format for both leaks ~5-10% of
cosine everywhere.

## Pivot 4: max_cos coverage was the wrong gate metric

**Initial gate.** `p50(max_cos) ≥ 0.50` and `p10(max_cos) ≥ 0.35`, where
`max_cos(job) = max over anchors of cos(job, anchor)`. Inherited from v2's
cluster-prototype thinking.

**Why it failed.** Even with 34 good prose-prefix queries, p50 max_cos
topped out at 0.445 — 5 points below the gate. Expanding the pool didn't
help.

**Two reasons the gate was wrong:**

1. **Calibration.** qwen3-embedding:0.6b is a small (0.6B param) retrieval
   model; its similarity scores for matched content are systematically
   lower than mxbai-embed-large's. The 0.50 threshold was guessed for
   larger models.
2. **Wrong metric entirely.** For a top-k softmax blend system, "how close
   is the best anchor to each job" is not what matters. What matters is
   "when I blend the top-3 anchors, does the blend concentrate on a few
   anchors or mush over all of them". max_cos tells you neither.

**Fix.** Replaced the gate with three blend-concentration metrics at the
generation-time temperature:

- `p50(top-3 weight sum) ≥ 0.80` — top-3 holds most of the mass
- `p50(effective_N) ≤ 4.0` — blend is concentrated, not diffuse
- per-anchor participation ≥ 1% of corpus — no dead anchors

**Lesson.** Your gate metric should match the downstream use of the data.
For a retrieval system max_cos matters; for a top-k blend system the
weight distribution matters; for a classifier the argmax matters. Don't
inherit metrics from adjacent problems.

## Pivot 5: softmax temperature T=0.1 was off by 5×

**Initial guess.** `T = 0.1` in the v3 spec as "sharp but not hard".

**Why it was broken.** Job-vs-anchor cosines are in [0.3, 0.6]. Dividing by
0.1 gives logits in [3, 6]. Softmax of logits within 3 of each other is
nearly uniform.

**Diagnosed by.** Computing blend metrics at T=0.1 on the full 34-anchor
pool:

| metric | value at T=0.1 |
|---|---|
| top-1 weight p50 | 0.096 |
| top-3 weight sum p50 | **0.23** |
| effective N p50 | **24 of 34** |
| entropy | 4.85 / 5.09 bits (nearly uniform) |

**At T=0.1, the "top-3 blend" was actually averaging ~24 anchors with
near-equal weight.** Every generated face would have been a muddy blend
of every role. This would have silently destroyed generation quality on
the full smoke batch — detectable only by visual inspection.

**Temperature sweep:**

| T | top-1 p50 | top-3 p50 | effective_N |
|---|---|---|---|
| 0.005 | 0.99 | 1.00 | 1.0 (one-hot, hard boundaries) |
| 0.010 | 0.91 | 1.00 | 1.2 |
| **0.020** | **0.63** | **0.89** | **2.3** ✓ |
| 0.030 | 0.41 | 0.72 | 4.1 |
| 0.050 | 0.22 | 0.45 | 10.3 |
| 0.100 | 0.10 | 0.23 | 24.3 (near-uniform) |

**Fix.** T=0.02 is the sweet spot: top-1 clearly dominant, top-3 captures
the blend, effective N≈2 (not one-hot so smoothness preserved).

**Lesson.** Always calibrate softmax temperature on the actual score
distribution you'll feed it. Default temperatures from "similar" problems
can be off by an order of magnitude. Run the sweep before shipping.

## Pivot 6: knee analysis vs game design constraints

**Initial approach.** Use greedy marginal-coverage ordering + knee method
to pick N automatically. N ≈ 10-22 expected.

**Why it was incomplete.** The knee method optimises *information gain*
per added anchor, not *learnability*. The scam-guessr game needs:
- Each anchor to be a *learnable "country"* on the 2D map
- Enough countries for the map to feel populated (not 2-3)
- Few enough that casual players can learn them in a session

**User constraint:** "for 7-10 it is casual fun; for 30+ it is geoguessr
for pros who can guess a country by a postbox". This is a game design
constraint, not a retrieval constraint. The knee method cannot see it.

**Also.** The top-10 greedy picks over-indexed on blue-collar (2× delivery,
2× warehouse/construction, 3× office-ish), leaving age/gender/ethnicity
axes underspread. The greedy optimum is not the editorial optimum.

**Fix.** Use greedy + blend metrics as diagnostics for pool quality, then
**hand-curate** the final 10 to balance the visual/demographic axes of
the game. Document the 10 picks and their rationale in
`data/curated_anchor_names.txt`.

**Lesson.** When the downstream product is editorial (a game map, a
playlist, a curriculum), don't let unsupervised selection pick the final
set. Use analytics to rank and diagnose, then curate.

## Pivot 7: exaggeration vs smoothness

**The temptation.** Once we chose T=0.02 with 10 anchors, the next question
was "can we exaggerate the distinctness further so faces feel more
distinct?" Natural answers: lower T more (T=0.01), or apply post-softmax
power sharpening (k>1).

**Why this would have backfired.** Both of those approaches push toward
hard assignment — approaching Voronoi partitioning, which is exactly what
v3 moved *away* from. Losing smoothness reintroduces the problem v3 was
designed to fix: hard cluster boundaries produce abrupt face jumps on the
2D map, so the player experience becomes "one of 10 faces" rather than a
smooth manifold of blended archetypes.

**Fix.** Keep the softmax math alone. Get distinctness from two orthogonal
levers instead:

1. **Vivid face_records.** Write each archetype's descriptors extreme:
   "ice-blue eyes, very fair Slavic, snub nose" not "Slavic". Distinctness
   inside each anchor, no effect on blend math, smoothness preserved.
2. **Optional conditioning-space amplification (α).** Parameter in
   `semantic_anchor_conditioning_nodes()`, default α=1.0. Pushes the top-3
   anchors' CLIP/T5 conditioning vectors further apart in Flux latent
   space *relative to their mean*. Preserves blend continuity (a smooth
   softmax transition is still smooth) but widens the latent-space gap
   between archetypes. Tuned on smoke batches only if the baseline is
   muddy.

**Lesson.** Before "turning up the knob" in one dimension, check whether
the knob's side-effects undo the very property the design was chosen to
provide. Look for orthogonal levers that preserve the invariant.

## What survived unchanged from the initial v3 design

- **Directional text-query anchors in qwen space** (the core idea) ✓
- **Two-channel encoding** (identity from anchor blend, expression from
  per-sus-band phrases) ✓
- **ConditioningAverage blending** in Flux ✓
- **Ethnicity target** (40% Slavic + even rest) ✓
- **Softened LoRA curve** from v2 (`(sus-25)/75)^1.2`) ✓

## Parameters and their calibrated values

| parameter | calibrated value | source |
|---|---|---|
| Embedding model | `qwen3-embedding:0.6b` | verified round-trip |
| Query format | 1-2 sentence prose | Pivot 2 diagnostic |
| Query prefix | `"Инструкция: Найди вакансии...\nЗапрос: "` | Pivot 3 diagnostic |
| Candidate pool size | 34 | editorial coverage |
| Final N | 10 | Pivot 6 (game constraint) |
| Softmax temperature T | 0.02 | Pivot 5 sweep |
| Power sharpening k | 1.0 (no-op default) | Pivot 7 |
| Conditioning amplification α | 1.0 (no-op default) | Pivot 7 |
| Dedup cosine threshold | 0.85 | carry-over, never triggered |
| Gate: top-3 weight p50 | ≥ 0.80 | Pivot 4 |
| Gate: effective_N p50 | ≤ 4.0 | Pivot 4 |
| Gate: per-anchor participation | ≥ 1% of corpus | Pivot 4 |
