---
shaping: true
---

# Scam Guessr — Shaping Doc

## Requirements (R)

| ID | Requirement | Status |
|----|-------------|--------|
| R0 | Give users a compelling reason to return on days they are not actively job-hunting | Core goal |
| R1 | The mechanism must work with continuously fresh content (24/7 scraping) | Must-have |
| R2 | Users must be able to express and improve a skill — repeat play should feel like getting better | Must-have |
| R3 | Catching a model misclassification must feel like a win — "I proved the LLM wrong" | Must-have |
| R4 | Global leaderboard — scores have meaning relative to other players | Must-have |
| R5 | Infinite mode as base; daily challenge as optional social object, not obligation | Must-have |
| R6 | TMA is primary platform — Telegram identity solves auth; viral loop via Telegram sharing | Must-have |
| R7 | Guest sessions for web — first session playable with zero auth, no progress persistence | Must-have |
| R8 | Single entry point — no separate "pro mode." Depth emerges through play, not through a different UI | Must-have |
| R9 | The reveal explanation must be specific and reframeable as discovery, never as failure | Must-have |
| R10 | The result card is a designed social object — face thumbnail + score + one punchy shareable line — not a screenshot | Must-have |
| R11 | Accuracy trend visible after session 5 — "your accuracy grew 12% over 10 games" — the first signal of skill development | Nice-to-have |
| R12 | "You flagged this N days before it was removed" — impact feedback loop. Must be architected Day 1 even if surfaced in Phase 2 | Must-have (arch) |

---

## Core Loop

```
See face → Form intuition → Commit guess → Reveal + specific explanation
→ [Discovery moment: "AI saw what I couldn't" OR "I caught what it missed"]
→ Score + delta → Share / replay
```

The emotional outcome at the hinge drives the loop: **both outcomes motivate the next round.** Being wrong is reframed as discovery. Being right is validated by the explanation. Neither outcome feels like failure.

---

## Core design principle (from discovery)

**We are not finding crusaders. We are training them.**

The "crusader hypothesis" (telejobs pilot) failed because it assumed pre-existing motivated users. GeoGuessr didn't find geography enthusiasts — it created them through repeated play. Scam Guessr does the same for fraud detection.

There is one persona: Аня (casual, entertainment-first). Олег (the expert investigator) is not a separate user type to design for — he is what Аня becomes after 200 sessions. The "professional" features (accuracy tracking, calibration score, impact feedback, "I disagree") are depth layers that reveal themselves as skill accumulates. They do not require a separate entry point.

**Implication for R8:** No dual-mode UI at launch. One game. One entry. Depth unlocks through use.

---

## Shapes

_To be developed after persona and discovery phases._

---

## Key findings from simulated discovery interviews

**Аня:**
- Shame is directed inward ("я дура") — the reveal must never feel like correction, only discovery
- Two shareable moments: "AI was wrong" AND "AI saw what I couldn't" — both work
- Re-entry trigger is a friend sharing her result, not notifications or streaks
- Streaks = Duolingo = she quits. No streak mechanics.
- The explanation quality after a wrong answer is the primary retention lever

**Олег (= future Аня):**
- "I flagged it 4 days before the ban" — named unprompted as most compelling feature
- Wants calibration as professional metric, not gamification framing
- Export/API as eventual requirement once he hits ceiling
- Proposed dual-entry himself — but this is a Phase 2 concern, not Day 1
