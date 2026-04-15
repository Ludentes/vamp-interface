# V2 Framework

This directory contains the single load-bearing document for V2 design decisions.

## The framework

**[math-framework.md](math-framework.md)** — the whole framework. Three-channel architecture (identity / editorial / drift), formal decision protocol (§2.7), rubric with hard-constraint floors, `P*` user-task top-level criterion, distortion-budget allocation principle, factor-mismatch drift reframing, math foundations anchored on specific theorems.

Version lives inside the document (current: v0.6) and the changelog tracks every revision. The file name carries no version suffix so git history stays coherent across rewrites.

## Sources

Supporting inputs that produced the framework live in [`sources/`](sources/). Read them only if you need the provenance of a specific framework claim; the framework itself is self-contained and cross-references them where needed.

- `theory-constraints.md` — self-contradiction analysis (HyperFace regime analysis, continuity-vs-readability impossibility, dropped-readability decision)
- `rebuild-blind-alleys.md` — paths tried and killed, with blockers cited
- `math-foundations-synthesis.md` — merge of the two adjacent-literature research passes into a "what we'll actually use" list
- `math-foundations-perplexity.md` — raw Perplexity deep-research output (source citation)
- `math-foundations-tavily-raw.json` — raw Tavily pro-research output (source citation); parse with `python -c "import json; print(json.load(open(...))['content'])"`

## How to use

1. **Read [math-framework.md](math-framework.md) cover to cover.** Do not skip §1.5 (distortion budget, Lipschitz demotion via K-R duality), §2.2 (three-channel architecture), §2.3 (P* + identity rubric), §2.3a (editorial rubric), §2.5 (drift as factor-mismatch), §2.6 (math foundations anchors), §2.7 (decision protocol).
2. **Check `sources/theory-constraints.md`** if you need to know *why* readability was dropped. Without that context, §2.2's empty stage-1.5 looks provisional; it isn't.
3. **Check `sources/rebuild-blind-alleys.md`** before proposing a mechanism. If your proposal is in the blind-alleys list, either overturn the specific blocker or pick a different mechanism.
4. **Then** propose an architecture or score a candidate. Proposals that cite neither rubric cells nor specific theorems are not framework-compliant.

## What belongs here vs. elsewhere

**Belongs here:** the framework itself and its direct sources.

**Does NOT belong here:**
- Exploratory literature reads → `../../docs/research/`
- Product shaping → `../../docs/shaping/`
- V1 artifacts → `../../v1/`
- Candidate scorings → `../evaluation/`
- Rebuild plans → `../plan/`

The discipline: if a document is not the framework and not a source cited by it, it does not belong here.
