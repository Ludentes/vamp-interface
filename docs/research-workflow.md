# Research & Document Lifecycle

How we — Claude + human — run a long-lived research project (vamp-interface, ~6 months of experiments, ~hundreds of dated docs) without drowning in stale notes or losing context across sessions and context-window compactions. This document is a teaching artifact: it captures what the workflow actually is, *why* each part exists, and which failure modes it prevents.

It is opinionated by experience, not aspiration. Every rule below was added because something burned us.

## The three-layer document model

We separate raw evidence from interpretation from index.

**Layer 1 — dated evidence (`docs/research/YYYY-MM-DD-<topic>.md`).** Append-only. Each doc is the writeup of a specific spike, experiment, paper review, or session. It is dated and never edited substantively after it's settled — only superseded. The date encodes "this was what we believed on that day." If a later doc overturns it, we don't go back and rewrite — we mark the old doc `status: superseded` and the new one `supersedes:`. The audit trail matters more than tidiness; future-you needs to be able to ask "why did we believe X in April?" and get an honest answer.

**Layer 2 — topic indexes (`docs/research/_topics/<topic>.md`).** Mutable interpretation layer. One per thread (e.g. `arkit-bridge.md`, `liveportrait-stylized.md`, `lam-chibi-recipe.md`, `photobooth-sweep.md`). Says "here is what we currently believe about this thread, and here is the chain of dated docs that got us there." Read-first on any fresh session. When you add a new dated doc on a thread, you update the topic index in the same commit. This is where current-state lives — the rest is history.

**Layer 3 — memory index (`MEMORY.md`).** One-line pointers, persisted across Claude sessions. Each entry is a status icon + slug + one-sentence hook, linking out to either a topic file or a memory file. Triage tool, not knowledge store. When MEMORY.md grows past ~150 lines or exceeds the loadable byte budget, consolidate.

The split exists because each layer rots at a different rate. Dated docs are eternal. Topic indexes get rewritten every few weeks. The memory index churns daily. Mixing them creates either constant edit churn on stable docs or stale claims in the live index.

## Frontmatter discipline ("two-strikes refactoring")

Every `docs/research/` and `docs/blog/` doc that survives review gets:

```yaml
---
status: live | superseded | archived
topic: <_topics/file-name-without-.md>
supersedes: <dated-doc-name>             # optional
superseded_by: <dated-doc-name>          # optional
---
```

Rule: we don't backfill mechanically. When Claude *substantively touches* a doc lacking frontmatter, add it. That's the "two strikes" — the doc has earned the attention. A dedicated Haiku subagent (`frontmatter-tagger`) generates the block, because the topic field requires reading the doc, and that's exactly the kind of cheap-but-not-free task that bloats the main session's context if done inline.

Why it matters: `supersedes:`/`superseded_by:` is what lets you safely answer "is this still true?" without re-reading every dated doc on the thread. Status icons in MEMORY.md (`🛑 FALSIFIED`, `🚢 SHIPPED`, `🆕 RULE`, `📚 RESEARCH`, `🧭 ACTIVE THREAD`) signal load-bearing-ness at a glance.

## The brainstorm → spec → plan → execute spine

Non-trivial work goes through four phases, each a distinct artifact:

1. **Brainstorm** (via `superpowers:brainstorming` skill) — explore 2–3 approaches, pick one, write the spec. Output: `docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md`. Gate: no implementation skill runs until the spec exists and passes self-review.
2. **Plan** (`superpowers:writing-plans`) — turn the spec into bite-sized, TDD-shaped tasks with exact file paths and code blocks. Output: `docs/superpowers/plans/YYYY-MM-DD-<feature>.md`. Each task is 2–5 minutes for a competent implementer with zero context.
3. **Execute** (`superpowers:subagent-driven-development`) — fresh subagent per task, two-stage review (spec compliance → code quality), automated verification gate, then mark the checkbox. The controller never lets a subagent read the plan file — it extracts the task text and hands over exactly the context needed. Context isolation is the point.
4. **Runbook** (`docs/research/YYYY-MM-DD-<system>-architecture.md` or `docs/runbooks/<system>.md`) — written when the system ships. Pipeline diagram, CLI, failure modes, on-disk layout, cross-references. Future-you-with-no-memory must be able to operate the thing from this doc alone.

The spine is not bureaucracy. It is what makes long-horizon work survive context compactions and personnel changes (including "the personnel" being a different Claude session). Skipping brainstorm to "save time" produces designs that look complete until execution exposes an unexamined assumption.

## Falsification as a first-class outcome

Roughly a third of MEMORY.md is `🛑 FALSIFIED` entries. We keep them. Every one represents a path we will otherwise re-explore in three weeks when we've forgotten why we walked away.

The discipline is:

- When a hypothesis dies, write a dated doc that says explicitly why, with the data that killed it.
- Memory entry leads with the verdict icon, names the doc, and points at the *replacement* approach.
- The old approach's memory entry gets marked superseded — not deleted — and the replacement entry links back.

Example: the chibi-on-splats line of work (`project_chibi_diff_geometry_session`) ran for several days, ended with "deformation too aggressive for a baked Gaussian-splat representation," pivoted to mesh-based chibi via FLAME-UV bake. Both the death certificate and the pivot rationale are searchable in one hop from MEMORY.md.

## Verification gates and the "evidence before assertions" rule

Two corpus-level burns drove this:

- We trained against a corpus whose rotation flag was wrong; loss curves looked normal; output was silently corrupted. The fix was the `feedback_verify_training_data_first` rule: gate any training run on `cos(extracted_target, teacher_inference_target) > 0.95` on a 60-frame slice before launching the full run.
- A baseline's `ratio_mean` was quoted from its own eval_log, which used a different holdout split than the current evaluation. Phantom 2.4× gap to the next iteration that didn't exist. Fix: never quote a baseline from its own logs; re-eval on the current holdout.

Generalised: a metric is only as good as the question it can distinguish (see `feedback_calibration_blind_spots`). Symmetries in your evaluation hide failure modes that downstream loss cannot recover. Verify the verifier before trusting it.

The standing rule from `superpowers:verification-before-completion`: never claim work is complete without running the verification command and reading its output. "Tests pass" without showing the green is not evidence; it is a wish.

## Compaction-safe writing

This is the rule that makes the rest viable. Context windows compact. Sessions end. Claude turnover is real. The corollary: **information that is not on disk does not exist.**

- Save research findings the moment they are produced (`feedback_save_research_findings`). Don't wait until "the session is done" — by then, the load-bearing detail may already be gone from your own context, never mind the next session's.
- Every long-running experiment writes intermediates that fully reconstruct its decision-relevant state: cache dirs, parquet metric tables, `eval_log` snapshots tied to a specific holdout-split fingerprint, configs at the top of every result dir.
- Render scripts enrich a parquet table per run (`feedback_render_enriches_parquet`). The mp4 alone is half a deliverable.
- Generation is resumable (`feedback_resumable_generation`). Skip-if-exists per output file. If a plan needs to change mid-run, append new cells and relaunch — don't kill the run to amend.

The test of a good intermediate is: can the next session, with no memory of this one, pick up the thread by reading the dir? If not, the dir is incomplete.

## External resources, vendored references, and archive paths

The repo is not the universe.

- Live external resources (ComfyUI custom nodes outside the repo, model weights, the Windows shard) are documented in CLAUDE.md and runbooks with grep-friendly absolute paths. Searching for "where does ComfyUI install" returns one canonical answer.
- "Stub" copies inside the repo that mirror external state are flagged as such, with an explicit pointer to the live location. We've burned hours editing the repo copy and wondering why ComfyUI didn't pick up the change.
- Heavy artifacts (≈300 GB of attention pkls) live on a USB archive at a documented path with a `docs/archive-locations.md` mounting guide. Symlinks from `models/` into the archive are documented in the runbook so the next session knows where the weights actually live.

## Feedback memory and the "save the why" rule

Memory entries of type `feedback` (corrections and confirmations) carry their reason and trigger condition:

```
Rule: <the rule>
Why: <the prior incident or strong preference>
How to apply: <where this kicks in>
```

The *why* is what lets future-you judge edge cases. A rule without its reason becomes either cargo-culted blindly or rationalized away the moment it's inconvenient. With the reason, it's a tool. Without, it's noise that gets pruned at the next consolidation.

We save confirmations too, not just corrections. If we silently drift away from validated approaches, we end up over-cautious — saving only "don't do X" makes Claude progressively more timid. Saving "yes, the single bundled PR was right" calibrates equally important judgment.

## Read-order for fresh Claude sessions

When a new session opens on a thread:

1. `MEMORY.md` (always loaded). Scan for the thread's status icon and slug.
2. The topic file linked from that entry — current beliefs.
3. The most recent dated doc(s) referenced by the topic file — load-bearing detail.
4. The runbook, if a shipped artifact is involved.
5. `git log --oneline -20` on the relevant subtree, to catch anything written since the topic file was last updated.

Only then read code. Reading code before reading the topic file is how you propose approaches that were ruled out three months ago.

## What we don't do

- **Don't number sections in markdown.** Iterative edits cause renumbering churn that bloats diffs and breaks cross-references. Refer to sections by name.
- **Don't batch-backfill frontmatter, runbooks, or memory entries.** Lazy/just-in-time; each artifact earns the work when it's touched.
- **Don't quote conclusions from a subagent's summary without reading the primary source for load-bearing claims** (`feedback_shallow_research_risk`). Subagent summaries compress. Architectural commits ride on details that compression discards.
- **Don't trust a recalled memory that names a specific function, flag, or file without re-checking it exists.** A memory is a claim about state at write-time, not now.
- **Don't keep destructive shortcuts as the default.** `--no-verify`, `git reset --hard`, "just delete it and start over" — every one of these has burned in-progress work somewhere in MEMORY.md.

## In one sentence

Treat research like an append-only log with a separate interpretation layer and a tiny live index — write everything down at the moment it's produced, never delete, supersede explicitly, and read the index before the code.
