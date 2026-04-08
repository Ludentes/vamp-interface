---
shaping: true
---

# Customer Journey Map: Scam Guessr
**Persona:** Anxious Аня (21, student, Екатеринбург)
**Objective:** Identify drop-off points and design moments that drive return visits
**Date:** 2026-04-08

---

## Stages

Adapted from NNGroup framework. "Service" split into three sub-stages to capture the training arc — this is the core of the product.

| | **1. Discovery** | **2. First Open** | **3. First Session** | **4. Return Trigger** | **5. Developing Skill** | **6. Trained Detector** |
|---|---|---|---|---|---|---|
| **Customer Actions** | Sees result card in Telegram chat from friend. Taps link out of curiosity. | TMA opens inside Telegram. Sees a face immediately — no onboarding screen. Decides to tap "Мошенник" or "Нормально" | Plays 5 rounds. Guesses. Sees reveal. Reads explanation. Sees session score. Taps "Поделиться" or exits. | Friend mentions her result. Or she's idle (queue, commute) and remembers. Returns to play more. | Sessions 5–30. Starts forming intuitions ("too-wide smile = bad sign"). Accuracy improves. Uses "Я не согласна" for first time. | Sessions 30+. Catches misclassifications confidently. Accuracy score matters to her. Shares "I beat the AI" moments. |
| **Touchpoints** | Telegram chat (shared result card) → TMA deep link | TMA launch screen → first face card | Face cards × 5 → reveal moments → explanation text → result card → share sheet | Telegram chat (friend's message) OR idle context (no push needed) | Game loop + accuracy tracker appearing + "Я не согласна" button | Calibration score + "flagged N days before removal" signal + leaderboard position |
| **Customer Experience** | *"Что это? Лицо какое-то странное. Ладно, посмотрю."* Curious, low commitment. | *"О, сразу игра, не надо ничего заполнять. Ну давай."* Relieved — no friction. | Round 1-3: *"Хм, интересно."* Round 4 wrong answer: *"Серьёзно?! Почему?"* Reads explanation: *"О, вот оно что."* Result: *"Хочу показать Дашке."* | *"А, та игра с лицами. Дашка написала что у неё 780 очков."* Mild competitive pull. | *"Подождите, у этого лицо нормальное, но что-то не так со взглядом."* Starting to trust own intuition. First "Я не согласна" — feels like agency. | *"Я видела это за два дня до того как его убрали."* Genuine pride. Олег-mode activated. |
| **Drop-off Risks** | Result card isn't visually striking enough to make her tap → **lost before entry** | Any screen before the first face (onboarding, permissions, login prompt) → **immediate exit** | Reveal explanation is generic or condescending → **no share, no return** | No social re-entry signal (friend doesn't share, no opt-in daily) → **forgotten** | Accuracy doesn't visibly improve → no sense of growth → **plateau churn** | No impact feedback ("your flags did something") → Олег-type users disengage |
| **KPIs** | Tap-through rate on shared result cards (target: >25%) | Time-to-first-guess <10s; 0 taps before first face | Rounds-per-session ≥5; share rate ≥15%; D1 return rate | D7 return rate ≥30% | Sessions 5–30 retention curve; first "Я не согласна" usage rate | D30 retention; disagreement accuracy rate; "flagged before ban" events |
| **Business Goals** | Viral coefficient >1 (each player generates >1 new player via sharing) | Zero-friction activation: no account required, first face in <3s | Hook established: reveal quality drives sharing and D1 return | Habit formation: friend-triggered re-entry replaces push notifications | Skill development visible to user → sense of investment → switching cost grows | Data flywheel: trained detectors generate high-quality disagreement signals for model improvement |
| **Owner** | Result card design + share mechanic | TMA onboarding (literally: none) + first face loading speed | Reveal moment quality + explanation copy + result card | Social loop design (result card virality) | Accuracy tracking UI + "Я не согласна" UX | Calibration score + impact feedback ("flagged before ban") |

---

## Emotion Arc

```
Discovery    First Open    First Session    Return    Skill Dev    Trained
   ?  ────────  😌  ──────────  😮  ──────────  🙂  ─────  😏  ──────  😎
 (curious)  (relieved)    (surprised)    (recalled)  (intuiting)  (expert)
```

The critical emotional transition is **Session 1 → Return**. The reveal moment must generate enough surprise/delight to trigger sharing, which generates the social signal that brings her back. Without that bridge, D7 retention collapses.

---

## Critical Path (the sequence that must work perfectly)

```
Friend shares result card
    → Аня taps (card must be visually striking)
    → TMA opens, face appears immediately (zero friction)
    → She guesses, commits
    → Reveal: face morphs, score appears, explanation is SPECIFIC and surprising
    → She thinks "надо показать Дашке"
    → She shares her result card
    → Дашка plays → shares → Аня gets social signal → returns
```

Every link in this chain is a potential break. The weakest links:
1. **Result card quality** — if it looks like a screenshot, not a designed card, tap-through dies
2. **Reveal explanation** — if it says "высокий уровень подозрительности" and nothing else, no share
3. **Share friction** — if Web Share API fails or requires extra taps, she doesn't share

---

## Opportunities by Stage

| Stage | Opportunity |
|-------|-------------|
| Discovery | Design the result card as a *social object*, not a score report. Face thumbnail + score + one punchy line ("Я угадала 4 из 5 — а ты?") |
| First Open | Make the first face load in the TMA launch screen itself — zero separate onboarding step |
| First Session | Write reveal explanations as 1-sentence discoveries, not verdicts. "Асимметрия челюсти, характерная для фотомонтажа заявленной зарплаты" > "Подозрительная вакансия" |
| Return | Daily challenge is opt-in and framed as "Дашка уже сыграла сегодня" not "Не пропусти день" |
| Skill Dev | Show accuracy trend after session 5 — "Твоя точность выросла на 12% за 10 игр" |
| Trained | "Ты отметила эту вакансию за 3 дня до её удаления" — first time this appears, it's a moment |

---

## What This Map Changes in Requirements

- **R9 confirmed critical:** Explanation quality is the hinge between Session 1 and Return. Non-negotiable.
- **R5 confirmed:** Daily challenge must be framed as social ("Дашка уже сыграла") not obligation ("не пропусти стрик"). Opt-in.
- **New gap:** Result card as designed social object not yet in requirements — needs to be R10.
- **New gap:** Accuracy trend visibility after session 5 not in requirements — needs to be R11 (depth layer, not Day 1).
- **New gap:** "Flagged before ban" feedback loop not in requirements yet — R12 (Phase 2, but must be architected Day 1).
