# Scam Guessr — Game Design

**Date:** 2026-04-08
**Status:** Concept / pre-implementation

---

## Concept

Scam Guessr is the product name for the reveal-game mechanic in vamp-interface.

Each job posting from the telejobs corpus is rendered as an AI-generated face that encodes its fraud score via the uncanny valley effect. The game loop: a face appears → player estimates the fraud score → the real score reveals → repeat.

**The core insight:** this is GeoGuessr for fraud detection. GeoGuessr trained a generation of geo-OSINT analysts accidentally, by making location guessing addictive. Scam Guessr does the same for fraud pattern recognition.

---

## Why It Works

- **Faces, not text.** A quiz with job text snippets is homework. Portrait photos that *feel wrong* are addictive. The uncanny valley effect is the engagement hook.
- **Gradient scoring.** Binary swipe (safe/scam) is too forgiving. A slider scored on delta creates depth — you feel the difference between "I was 8 off" and "I nailed it."
- **Incidental skill-building.** Repeated exposure trains real-world fraud recognition. Players don't realise they're learning.
- **Honest about uncertainty.** The "I disagree" button acknowledges the model can be wrong, which is what makes it trustworthy.

---

## Core Game Loop

```
Face appears
  ↓
Player drags sus slider (0–100)
  [face animates clean→uncanny in real time as slider moves]
  ↓
Player commits guess
  ↓
Reveal moment:
  - True sus score slams in
  - Face morphs to its actual uncanny state (dramatic if delta is large)
  - sus_factors flash up one by one: "Payment upfront. Vague location. No company name."
  ↓
Score: points = f(|guess - actual|), peaks at exact match
  ↓
"Fair enough" → next card
"I disagree" → reason picker (see below)
```

---

## The Commit Moment

The most important UX beat — equivalent to GeoGuessr's map animation flying from pin to answer.

When the real score reveals:
- If player guessed 20, actual is 95: face slides from clean into full uncanny valley in front of the player. The "oh god" moment.
- If player guessed 90, actual is 92: subtle confirmation, face barely changes. Feels like expertise.

The face morphing on reveal is load-bearing. It must be animated, not instant.

---

## Slider Mechanic

- Drag 0–100 on a single axis
- Face updates live as slider moves (lower fidelity preview, not full re-render)
- Commit button locks in the guess
- Optionally: slider snaps to deciles to reduce false precision

---

## Scoring

```
delta = |guess - actual|
score = max(0, 1000 - delta * 10)   // 1000 at delta=0, 0 at delta=100
```

Streak bonus: +50 per consecutive round within ±15 of actual.

---

## The "I Disagree" Button

After reveal, if the player wants to contest the model's verdict:

**Reason picker (radio, single select):**
- "Job text looks legitimate"
- "Company seems real"
- "Criteria don't apply here"
- "Other language / region"

**Data flywheel:**
```
Disagreements collected per face
→ faces with N+ disagreements flagged for analyst review
→ analyst relabels or reweights sus_factors
→ model improves → new face version generated
→ players notice "it got harder" → re-engage
```

**Retroactive scoring:** if a flagged face is later corrected in the direction the player predicted, they receive retroactive points. "You called this 6 weeks ago." Strong retention hook.

This is RLHF disguised as a game. Players don't know they're labeling training data.

---

## Competition Layer (GeoGuessr-inspired)

| Mechanic | Description |
|----------|-------------|
| **Daily 5** | Same 5 faces for all players each day. Global leaderboard resets daily. |
| **Streak mode** | Consecutive rounds within ±20 of actual. One miss breaks the streak. |
| **Duels** | Two players, same face, simultaneous reveal. Closer guess wins the round. Best of 5. |
| **Rank titles** | Rookie → Field Agent → Senior Investigator → Chief → Legend. Based on rolling 30-day accuracy. |
| **Shareable result** | Card showing face thumbnail + guess vs. actual + score. Designed for messaging apps. |

---

## The Fraud Landscape View (Map Metaphor)

Complementary to the game: the full corpus as a navigable 2D space.

The PaCMAP embedding layout IS a fraud geography. Scams cluster because scammers copy language. Legitimate jobs cluster because they share vocabulary. The map reveals that structure.

| Google Maps | Fraud Landscape |
|---|---|
| Satellite view | PaCMAP layout, dots colored by sus_level |
| City boundaries | HDBSCAN cluster hulls |
| Zoom to street level | Semantic zoom → faces appear |
| Street View | The reveal game (ground level, one face) |
| Heat map / traffic | Density contours (fraud concentration) |
| Drill into district | Scatter/Gather (re-project cluster internals) |

The faces get wrong-looking as you approach hot zones. No color legend needed.

---

## Three Entry Points (Persona Routing)

```
Student / casual      → Scam Guessr game → Fraud Landscape (grid mode)
Scam Hunter           → Fraud Landscape (dot map) → cluster → faces → lasso → game
Analyst               → Fraud Landscape (dot map) → cluster sidebar → scatter/gather
```

After the reveal game, route to "explore by job type" (Student) or "explore the full corpus" (Hunter/Analyst).

---

## Pitch to Partners

*"We're not selling a black box. We're building a system that gets better every time someone plays."*

Target partners: universities (security research), job platforms (hh.ru, HeadHunter), labour inspection bodies.

Value proposition:
- Operators get a free annotation pipeline disguised as entertainment
- Players build genuine fraud-spotting skill
- Disagreement data surfaces model blind spots and regional vocabulary gaps
- Daily active users create a continuous labeling stream

---

## What Is Not This

- Not a gamified dashboard (points/badges on BI tools) — no ROI for investigation tools
- Not a citizen science annotation app (Fold.it) — the game comes first, labeling is a side effect
- Not a text quiz — the faces are the hook, text is a reward clue

---

## Open Questions

- Should the slider show the face morphing live, or only on reveal? (live = more engaging, but technically harder)
- Daily 5 requires a backend. For MVP: local random selection with shareable result string.
- Retroactive scoring requires persistent player history. Phase 2.
- Duel mode requires real-time infra. Phase 3.
