# Face Visualization — Reference Scenarios

**Date:** 2026-04-06
**Context:** Telejobs corpus (23k Telegram job postings, fraud scored). Face visualization concept: each posting rendered as a face, with encoding drawn from either the factor vector or the raw text embedding.

---

## Scenario A — The Scam Hunter

**Who:** OSINT enthusiast. Spends evenings hunting fraud, screenshots scam posts, reports to Telegram admins and anti-fraud channels. Motivated by anger, not money.

**Current workflow:** Opens telejobs UI, sorts by sus_level descending, reads the top cards one by one. Knows to look for "DM only", vague descriptions, inflated pay. Gets bored after ~30 cards. Misses mid-range scores because it's not worth reading 200 cards a session.

**What they actually want:**
- Process 200 postings in the time it currently takes to read 20
- Spot postings that feel like part of a coordinated operation (same ring, different wording)
- Find the ones their gut flags even when the score doesn't — cases the model undersells

**Session in face-viz:**
1. Opens a 10×20 grid of faces, sorted by sus_level or by embedding cluster
2. Scans the grid in seconds — some faces feel wrong immediately
3. Clicks a suspicious face → original post expands alongside it
4. Notices three faces that look alike — same operation, different channels
5. Flags all three, writes up the pattern

**Success:** Finds a coordinated ring in 10 minutes that would have taken an hour of card reading, or wouldn't have been found at all.

**What they need from the encoding:**
- The face must carry legible signal fast — "wrong" must feel wrong on first look, not after calibration
- Cluster membership must be visually apparent (similar posts → similar faces)
- Score alone is not enough; they want to see *why*

**Verdict on approaches:**

| Approach | Fit | Reason |
|---|---|---|
| Factors → face | Medium | Legible from day one if mapping is intuitive. But blind to what factors miss. |
| Embeddings → face | Medium | Cluster membership is genuine. But "similar" doesn't mean "suspicious" — face gives no verdict. |
| Hybrid | High | Identity from embeddings (cluster membership), expression from factors (verdict). Discordance between the two is a useful signal. |

---

## Scenario B — The Analyst

**Who:** The person maintaining telejobs — tweaking prompts, adjusting factor weights, checking golden dataset coverage. Wants to know where the pipeline is wrong.

**Current workflow:** SQL queries against the DB, pivot tables on factor distributions, manual review of misclassified cases from the golden dataset. Finds it very hard to see *structural* failures — the cases where the whole factor vocabulary is wrong for a new pattern.

**Core frustration:** "I don't know what I don't know." The scoring model can be confidently wrong in ways that won't show up as obviously bad scores — it just silently assigns a medium score to something genuinely novel.

**What they actually want:**
- See where in embedding space the model is uncertain or inconsistent
- Find clusters that have no vocabulary in the current factor set (unmapped fraud patterns)
- Detect drift: is the corpus changing over time while the model stays static?
- Build more golden dataset by batch-labeling homogeneous clusters

**Session in face-viz:**
1. Opens embedding-space view (UMAP-organized grid)
2. Finds a cluster where faces look alike (same embedding neighborhood) but expressions vary wildly (mixed factor scores) — this is a model inconsistency
3. Drills into the cluster, reads 5 postings, realizes they're all the same scam type but worded differently enough that some factors don't fire
4. Updates the factor extraction prompt, re-runs on the cluster, checks whether faces now look consistently suspicious
5. Finds another cluster that looks totally different from anything in the golden dataset — new pattern, no labels

**Success:** Identifies a systematic blind spot in the factor set. Labels 50 new golden examples from one cluster batch.

**What they need from the encoding:**
- Embedding geometry must be preserved — spatial position on screen = semantic distance in text space
- Factor-derived expression layered on top — so discordance (calm face in a suspicious cluster) pops out
- Time dimension useful: same cluster at T vs T+90days

**Verdict on approaches:**

| Approach | Fit | Reason |
|---|---|---|
| Factors → face | Low | Visualizes the model's output, not the underlying reality. Can't see what the model misses. |
| Embeddings → face | High | Raw signal, no theory. Shows clusters the model has no vocabulary for. |
| Hybrid | High | Essential: expression (factors) overlaid on identity (embeddings) makes discordance visible. |

---

## Scenario C — The Bored Student

**Who:** 19-year-old looking for a part-time job, heard about telejobs from a friend, curious about the fraud angle. Not technically sophisticated. May share screenshots.

**Current workflow:** Doesn't really have one for job hunting — scrolls Telegram channels, gets suspicious vibes on some posts, ignores them. Attracted to telejobs because it "tells you if a job is a scam."

**What they actually want:**
- Know fast: is this posting safe or not?
- Make job browsing less tedious and more interesting
- Brag about the cool tool to friends

**Session in face-viz:**
1. Opens the grid filtered to their work type (e.g., warehouse, courier)
2. Immediately sees which faces look sketchy — doesn't need to read anything
3. Clicks a trustworthy-looking face, sees the full posting, applies
4. Avoids the creepy-looking ones without needing to understand why
5. Shows a friend: "look, this one has a weird face, it's probably a scam"

**Success:** Avoids one scam posting they'd have otherwise clicked on. Shares the tool.

**What they need from the encoding:**
- Intuitive on first view — no calibration, no legend
- Trustworthy face must feel trustworthy, suspicious face must feel suspicious
- Must not produce false negatives that send them toward scams — higher cost than false positives

**Verdict on approaches:**

| Approach | Fit | Reason |
|---|---|---|
| Factors → face | High | If the mapping is emotionally intuitive (fraud signals → genuinely unsettling face), legible immediately. |
| Embeddings → face | Low | "Similar postings look similar" carries no verdict. Student can't tell if a cluster is safe or not. |
| Hybrid | Medium | Expression from factors is what the student reads. Embedding-based identity is noise to them. |

---

## Cross-Scenario Summary

| | Scam Hunter | Analyst | Student |
|---|---|---|---|
| Primary need | Pattern / ring detection | Model blind spots | Safe vs unsafe, fast |
| Factors → face | Medium | Low | High |
| Embeddings → face | Medium | High | Low |
| Hybrid | **High** | **High** | Medium |
| Needs calibration? | Tolerates it | Wants to calibrate it | Cannot require it |
| Cares about model internals? | Somewhat | Yes | No |

**The hybrid approach is the only one that serves two of three users well.** The student is best served by factors-only with an emotionally intuitive mapping. The scam hunter and analyst both benefit from the discordance signal that only appears when both layers are present.

**The critical design constraint across all three:** The uncanny valley must be triggered reliably for high-sus faces on first view, without calibration. This is not about mapping specific fraud signals to specific expressions — it is about making the face feel *wrong* in proportion to sus_level. The flavor of wrongness comes from the factor vector; the magnitude comes from the score.
