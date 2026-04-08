---
shaping: true
---

# Scam Guessr — Engagement Design

**Method:** MDA Framework + Engagement Curve + Flow Theory (from entertainment app methodology)
**Date:** 2026-04-08
**Purpose:** Solve the four engagement gaps found in the R1 epic breakdown

---

## Core Loop (formal)

```
See face → Form intuition → Commit guess → Reveal + specific explanation
→ [Discovery: "AI saw what I couldn't" OR expertise: "I caught what it missed"]
→ Score + delta → Share / replay
```

Every requirement must either **enhance** or **protect** this loop. Nothing else ships.

---

## Player Profile: Аня (Bartle Analysis)

Bartle types applied to fraud detection game context:

| Type | Weight | What this means for Scam Guessr |
|------|--------|--------------------------------|
| **Explorer** | 40% | The explanation IS the game. She wants to understand *why* the face feels wrong. Score is secondary. |
| **Achiever** | 35% | Score, rank title, visible improvement across sessions. "I got better" matters. |
| **Socializer** | 25% | Shares results, friend challenge is the primary return trigger. Competition is incidental, not driven. |
| **Killer** | 0% | Does not compete to dominate — competes to connect. |

**Primary implication:** Аня is an Explorer. The reveal explanation is the dopamine hit, not the score. E2-S3 (explanation copy) is the highest-leverage R1 story. A perfect score with a generic explanation retains nobody. A surprising explanation with a mediocre score creates a share.

**SDT mapping (Self-Determination Theory):**
- **Autonomy:** HIGH — she controls the guess, the pace, whether to disagree. No railroading.
- **Competence:** HIGH — must feel calibration skill is real and improving. Score trajectory matters.
- **Relatedness:** MEDIUM — social by sharing, not by co-playing. Passive social (friend's score as context) not active (team modes).

---

## Engagement Curve: The 5-Round Session

A single session maps to Hook + Onboard + Early Mastery. **Flow state develops across sessions 5–15, not within a single session.** This is intentional — sessions are short (2–5 min), and the return hook is what plants the seed for flow.

```
Engagement
    ^
    |        ●R2 (visceral reveal)
    |       ╱  ╲
    |      ╱    ╲
    |     ╱      ╲ R3
    |    ╱    R4  ╲╱ ╲
    |   ╱               ╲__ Exit Hook
    |  ╱R1                  (planted)
    | ╱
    +─────────────────────────────→ Time
      0s   1m    2m    3m   4m   5m
      Hook Onboard    Early Mastery
```

| Round | Phase | Emotional beat | What must happen |
|-------|-------|---------------|-----------------|
| R1 | Hook | "Something is wrong with this face." Curious, low commitment. | Face loads fast. No tutorial. Intuition fires immediately. |
| R2 | Onboard | First big reveal — the "oh god" face (sus ≥ 80). Surprise + validation of gut feeling. | Explanation is specific and striking. Delta is large enough to surprise. |
| R3 | Early Mastery | "I'm starting to read these." First "Я не согласна" likely here if player was right. | Genuinely ambiguous face. Explanation teaches a new tell. |
| R4 | Early Mastery | Calibration. Player is adjusting strategy. | Subversion: low-sus face that looks suspicious (sus ≤ 20). "I was sure that was a scam." |
| R5 | Exit set-up | Tension: "Can I recover?" Total score building. | Medium-difficulty face. Score accumulates. End of session feels conclusive. |

**Ordering rule (implemented in E5-S1 stratified selection):**
Medium → High-sus → Medium → Low-sus → Medium

The "oh god" moment lands at R2 (not R1 — no context yet) and the subversion at R4 (player has enough calibration to be surprised by the low-sus face).

---

## MDA Analysis: Three Critical Mechanics

### 1. Reveal Explanation (R9 — highest leverage)

**Mechanic (what we build):**
- 1–2 sentences per sus_factor, specific and reframeable
- Two template types per factor: large-delta (discovery framing) + small-delta (validation framing)
- 16 factors × 2–3 templates each = 32–48 copy units (content task, not engineering)

**Predicted dynamics (what will emerge):**
- Players screenshot and share "wild" explanations to friends
- Players start noticing the same cues in real job posts outside the game — real-world skill transfer begins
- Players show others: "look what the AI sees that I didn't"
- Players argue with the AI when they disagree — which seeds the "Я не согласна" usage
- Shared explanations become conversation starters ("seriously, look at this one")

**Target aesthetic (what players feel):**
- **When right:** Validation — "I saw it too, I just didn't have the words."
- **When wrong:** Discovery — "I had no idea this was a tell." Never shame.
- **Both outcomes:** Intelligence. The explanation makes the player feel more perceptive, not more naive.

**Failure mode:** Generic explanation ("высокий уровень мошенничества") → no share, no return. The loop breaks here.

**Success metric:** Share rate correlates with explanation specificity. Proxy: session share rate > 15%.

---

### 2. "Я не согласна" Button (E4)

**Mechanic (what we build):**
- Post-reveal button, always visible (not only on high-delta rounds)
- Bottom sheet: 4 reason options, single select
- "Записано — ты не согласна с AI" confirmation, auto-dismiss 1.5s
- Stored: job_id + player_guess + true_sus + reason + timestamp (R12 schema)

**Predicted dynamics (what will emerge):**
- Players use it primarily when they were RIGHT and the AI scored high — "I know this employer, they're legit"
- Becomes the "I beat the model" moment — the most emotionally powerful interaction in the game
- Players track in their heads how often their flags are vindicated (leading to R3's calibration score feature)
- Олег-mode users (Аня after session 30+) use it analytically, not emotionally

**Target aesthetic (what players feel):**
- Expertise and agency: "I am calibrated. The LLM is not always right."
- Not a complaint form — a challenge issued to the AI
- Over time: "I flagged this 4 days before it was removed" is the payoff moment (R3)

**Implication for copy:** Button label "Я не согласна" is correct — it's personal and assertive. Avoid "Сообщить об ошибке" (report an error) — that's a complaint, not a challenge. Confirmation copy: "Твоё несогласие записано" not "Спасибо за отзыв."

---

### 3. Score System

**Mechanic (what we build):**
- `score = max(0, 1000 − |delta| × 10)`
- delta = |player_guess − true_sus|
- Perfect score (1000 pts): delta ≤ 0 — only possible with slider, not binary

**Predicted dynamics (what will emerge):**
- Session 1: Players anchor on binary correctness — "did I pick the right side of 50?" Score feels like a bonus.
- Session 3+: Players discover precision is rewarded — start aiming for ±10 range
- Perfect session scores (5000 pts) become rare, shareable, identity-forming
- Players compare session scores with friends competitively ("I got 4,200 today")
- End-of-session score becomes a proxy for "how calibrated am I today" — variable enough to create daily variance

**Target aesthetic (what players feel):**
- Calibration mastery: "I'm not just guessing — I'm measuring."
- Precision as skill: "Getting within ±10 is genuinely hard and I got it."
- Daily variance creates reason to return: "I scored 2,800 yesterday; I can do better today."

**Implication for E2-S2 (slider):** The slider is not just a UX upgrade from binary buttons — it enables the precision mechanic. Without slider, perfect scores are impossible and the calibration arc doesn't develop. E2-S2 should ship before E2-S3 if possible (reorder to S1 → S2 → S3).

---

## Gap Solutions

### Gap 1: Session Face Curation (solved in E5-S1)

**Problem:** Random selection cannot guarantee the engagement curve emotional beats.
**Solution:** Stratified sampling in `POST /game/session`:

```python
def select_session_faces(corpus: list[Job]) -> list[Job]:
    high_sus = [j for j in corpus if j.sus_level >= 80]   # "oh god" pool
    low_sus  = [j for j in corpus if j.sus_level <= 20]   # "subversion" pool
    mid_sus  = [j for j in corpus if 30 <= j.sus_level <= 70]  # challenge pool

    r2 = random.choice(high_sus)   # round 2: visceral reveal
    r4 = random.choice(low_sus)    # round 4: subversion
    mid_three = random.sample(mid_sus, 3)

    # Order: medium, high, medium, low, medium
    return [mid_three[0], r2, mid_three[1], r4, mid_three[2]]
```

**Edge case:** If corpus has fewer than 5 faces in a pool (impossible at 543, possible for daily challenge), fall back to nearest-tier faces.

---

### Gap 2: Exit Hook (solved in E3-S1)

**Problem:** Session end screen had no designed return seed.
**Solution:** Three exit hook types, rotated by session count:

| Session count | Hook shown | Copy |
|--------------|-----------|------|
| 1–2 | Score gap | "Лучший результат сегодня: 5 120. Ты: 3 840 — попробуй снова?" |
| 3+ (improving) | Score trajectory | "Сессия 1: 2 840 → Сегодня: 3 840 📈 Точность растёт" |
| 3+ (plateau) | Score gap | Fall back to score gap hook |
| Always | Face teaser | Silhouette of one face from the next session pool: "Кто это?" |

**Rules:**
- No streak language ("продолжи серию", "не прерывай стрик") — Аня's Duolingo trauma
- No obligation framing ("не пропусти завтра") — pull not push
- Score trajectory only shows when session count ≥ 3 and the trend is positive — never show a downward trend

---

### Gap 3: Bartle Profile (solved — documented above)

**Problem:** Personas described behavior but not motivation archetype — easy to accidentally optimize for the wrong emotional outcome.
**Solution:** Explicit Explorer-first framing added to this doc and referenced in E2-S3 priority rationale. Key decision rule: **when in doubt, prioritize explanation quality over score mechanics.**

---

### Gap 4: Flow Channel for Returning Players (deferred to R2, but architected now)

**Problem:** After session 15–20, Аня may feel calibrated enough that random faces feel too easy.
**Solution sketch (R2 scope):**

Track per-player session accuracy rate. If rolling average accuracy (delta < 20) > 70% over last 5 sessions:
- Shift face pool toward sus_level 35–65 (the genuinely ambiguous zone)
- Reduce high-sus faces from 1 to 0 per session (no easy "oh god" reveals)
- Label these sessions "Эксперт-режим" or show no label (invisible, like Candy Crush difficulty scaling)

**R1 implication:** Store per-round delta in `game_rounds` table from day one. This enables R2's adaptive difficulty without a schema migration. Add `delta` column to `game_rounds` AC in E5-S1.

---

## What This Changes in R1 Priority

| Change | Impact | Story affected |
|--------|--------|----------------|
| Stratified session selection | Backend complexity +0.25d | E5-S1 |
| Exit hook (score gap + face teaser) | Frontend complexity +0.5d | E3-S1 |
| `delta` column in `game_rounds` | Backend complexity +0.1d | E5-S1 |
| E2-S2 (slider) before E2-S3 | Reorder only, no effort change | Sprint plan |

**Total R1 effort delta: +0.85 days (~17.85 days)**

Revised sprint order recommendation:
```
E1-S1 → E5-S1 → E2-S1 → E2-S2 → E2-S3 → E3-S1 → E4-S2 → E3-S2
```
(Slider before explanation — enables precision mechanic that makes explanation copy land better during testing)
