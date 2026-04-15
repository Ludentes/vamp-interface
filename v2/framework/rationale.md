# vamp-interface Rationale & Positioning

**Date:** 2026-04-15
**Version:** v0.1
**Status:** Working document. Records *why* the framework is shaped the way it is. Paired with [math-framework.md](math-framework.md) (the formal spec) but not part of it. When positioning changes, update this doc first, then reconcile the framework.

## Why this document exists

The framework specifies *what* vamp-interface must satisfy. This document specifies *why* we care — the positioning analysis that decides which clauses are load-bearing and which are flex slots. Without a written rationale, every architectural choice gets re-argued from first principles each session. This is the first-principles summary as of 2026-04-15.

## The core question

Is there any task for which rendering text items as photorealistic faces is *strictly better* than existing alternatives — and if so, which task, for which user, in which deployment shape?

## The 5-task comparative table

Five tasks a user might run on a multivariate text corpus. Five alternatives per task.

| Task | Color-coded list | t-SNE/UMAP + hover | Parallel coords / star glyphs | LLM summary & cluster labels | Faces (our approach) |
|---|---|---|---|---|---|
| **1. Single-item verdict** ("is this scam?") | ✅ 1 scalar, instant, 0 training | ➖ overkill | ➖ overkill | ✅ named reason | ❌ friction, no gain |
| **2. Scan 20-card list for top N worst** | ✅ sort + color, done | ➖ loses per-item identity in 2D | ➕ readable for trained user | ✅ "top 5 with reasons" | ➕ only if preattentive edge real |
| **3. Corpus-level cluster discovery** ("what types exist") | ❌ 1D | ✅ industry standard, cheap | ➖ crowds at >50 items | ✅ cheapest, named clusters | ➕ gestalt *character*, not position |
| **4. Axis semantic co-discovery** ("wait, the smile means X") | ❌ one axis at a time | ➖ axes are meaningless projection | ➕ axes already labeled → no discovery | ❌ labels are the LLM's model, not yours | ✅ **unique**: covert multi-axis encoding, cortex decodes gestalt, semantics emerge via co-variation under filter |
| **5. Longitudinal intuition / memory across sessions** | ❌ no identity persistence | ❌ layout changes on rerun | ❌ no per-item persona | ➖ labels drift with model version | ✅ same item → same face, Scott 1992 calibration advantage |

Legend: ✅ best, ➕ competitive, ➖ workable, ❌ loses.

## What the table reveals

Faces lose cleanly on tasks 1 and 2. A colored sorted list is the right answer for card-list browsing and single-item verdict work, and the telejobs UI is already correct on those tasks. **We should stop pretending faces are a card-list upgrade — they are not.**

Faces are competitive-at-best on task 3, where t-SNE + hover is cheap and captures most of the value.

**The only two cells where faces are unique and defensible are 4 and 5** — axis semantic co-discovery and longitudinal calibration. The two are tightly coupled: co-discovery *is* the mechanism by which longitudinal intuition accumulates. This lineage traces to Chernoff's original 1973 argument and Tukey's EDA — the visual cortex as an unsupervised pattern detector for multivariate systems we have no model of.

Everywhere else, an existing alternative is cheaper and roughly as good or better. The LLM-summary baseline is the meanest comparator — it's what most teams would actually build, and it covers tasks 1–3 well. Our argument against it is philosophical: *the LLM's taxonomy is the LLM's model, not yours.* Face-EDA is about the user building their own model of an opaque domain.

## Positioning options

Given faces only win at tasks 4 and 5, where and for whom does that matter?

### Option A — Research / analyst tool

**Target:** professional analysts studying the scam ecosystem; employees of consumer-protection NGOs, labor-rights orgs, fraud investigators.

**Deployment:** desktop exploration UI with filter/slice/neighborhood/more-like-this operations, paired with face grid and raw-text drill-down. Corpus ~hundreds to low thousands.

**Strengths:** the defensible niche is real. Nothing else does task 4 well for a text domain. Calibrated analysts extract genuine intuition about scam taxonomy. Framework v0.9 is roughly aligned here.

**Weaknesses:** small audience. Non-trivial build (query infrastructure + generation pipeline). Hard to sell against a cheap LLM-summary dashboard that gives the same analyst 80% of the answer in 5 minutes.

**Verdict:** defensible but narrow. Keep as fallback if consumer play fails.

### Option B — Daily data-mining puzzle (Wordle-shaped)

**Target:** the general public, via a daily puzzle everyone plays at the same cadence.

**Deployment:** web app, one puzzle per day, everyone gets the same puzzle. Each puzzle is a small (~30–60 items) synthetic corpus with 3–5 planted patterns. User explores via face grid + filter in 5–10 minutes, submits pattern hypotheses, gets scored at reveal. Shareable result format (spoiler-safe, Wordle-emoji-style).

**Why this is the primary candidate:**
- **Daily cadence** creates habit (Wordle lesson #1).
- **Same puzzle for everyone** enables social comparison without breaking spoilers (Wordle lesson #2).
- **Shareable compact result** is the viral engine (Wordle lesson #3).
- **Pure brain machinery** — no training, no reading walls of text, no instruction. Glance at the grid, think, guess, reveal.
- **Public good side effect:** players get better at spotting real patterns in text corpora, which transfers to real-world scam detection.
- **The face is load-bearing in a way it isn't for card-list tasks:** texts don't compose preattentively, so a grid of raw text is unplayable in 5 minutes. The face is the preattentive channel that makes the round shape work. This is the Street-View-analog for text domains.

**Extended mode (GeoGuessr-style variant):** same mechanism, longer sessions, archetype tagging, leaderboards. Kept warm as a variant but not the MVP.

**Weaknesses:**
- Is it actually fun? Unknown. Wordle works partly because letters are familiar; faces are familiar but the "discover what the smile means" mechanic is unproven.
- Streamable meta is culturally contingent and unplannable.
- Pattern authoring cost is ongoing — someone has to generate new puzzles. Can be automated with LLM pattern-planting.
- The face signal still has to actually encode useful structure — the game is distribution, not magic.

**Verdict:** most promising consumer play. Dev path is through Option C (below) as validation prototype.

### Option C — Hidden-pattern detective experiment (development precursor to B)

**Target:** ourselves and a handful of testers, as validation methodology and proof-of-concept for Option B.

**Deployment:** controlled synthetic universe. Experimenter plants patterns, sealed detective recovers them via face-EDA. Measured by pattern recovery rate. Spec'd in [../../docs/design/hidden-pattern-experiment.md](../../docs/design/hidden-pattern-experiment.md).

**Strengths:**
- Ground truth is perfect (we defined the patterns). No dependence on real scam data quality.
- Complexity is tunable. Can start simple, add layers.
- Validates the whole co-discovery loop cheaply — one experimenter-weekend per iteration.
- *Data mining as puzzle* framing is genuinely novel territory. Agatha-Christie-shaped.
- Scales down directly to Option B's daily puzzle with parameter changes (smaller corpus, shorter budget, shareable result).

**Weaknesses:**
- Synthetic patterns may not transfer to real-corpus patterns.
- Requires Phase-1 discipline: whoever plants patterns must not be the detective.

**Verdict:** active next step. Built first because it validates the whole face-EDA approach *and* is the prototype for Option B.

## Current standing

- **Option A** — alive as a fallback. Framework v0.9 clauses apply.
- **Option B** — primary consumer target. Depends on C succeeding first. GeoGuessr-style extended mode is a sub-variant.
- **Option C** — active near-term work. Spec in [hidden-pattern-experiment.md](../../docs/design/hidden-pattern-experiment.md).

## Open questions

1. Does the face channel actually encode recoverable structure in user hands? Framework §5 Exp A–C address direction-basis and λ-sweep; Option C addresses recoverability.
2. Is the puzzle actually fun? Answerable only by playtest after Option C validates the mechanism.
3. If C works, what's the minimum corpus size for a playable daily round? Probably 30–60; to be measured.
4. What's the shareable result format? Candidates: Wordle-style emoji grid showing which patterns were hit, face-thumbnail-strip of "key" items, archetype-discovery streak count.

## Relationship to framework

- When this document commits to a positioning shift, [math-framework.md](math-framework.md) must be reconciled within one cycle.
- Framework clauses that don't map to the committed positioning are tagged "flex" or "deferred," not removed.
- The GeoGuessr/Wordle pivot stays documented here as a candidate. Framework clauses specific to the consumer game (round shape, shareable result format, puzzle-a-day cadence) are gated on Option C succeeding.

## Changelog

- **v0.1 (2026-04-15):** initial rationale, records 5-task stacking, three positioning options (A/B/C), Wordle framing as primary consumer target, detective experiment as dev precursor.
