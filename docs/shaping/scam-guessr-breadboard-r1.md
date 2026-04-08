---
shaping: true
---

# Breadboard: Scam Guessr R1 — Game Core

**Method:** Breadboarding (Ryan Singer / Shape Up)
**Scope:** E2-S1 through E4-S2 — full R1 game loop
**Date:** 2026-04-08

**Shaped parts:**
- Face card full viewport + binary guess buttons (E2-S1)
- Commit → reveal with score + explanation (E2-S1, E2-S3)
- 5-round session (E5-S1)
- Disagree signal (E4-S1, E4-S2)
- Session end + share (E3-S1)

**Existing backend (E5-S1):**
- `POST /game/session` → `{session_id, job_ids[5]}`
- `POST /game/round` → records round
- `POST /game/disagree` → records disagreement
- `GET /thumbs/{job_id}.jpg`
- `GET /game/faces/{job_id}` → `{sus_level, sus_factors}`

---

## Places

| # | Place | Description |
|---|-------|-------------|
| P1 | Face Card | Active round — face + binary guess buttons + progress |
| P2 | Reveal Screen | Score animation, delta, explanation; "Далее →" and "Я не согласна" |
| P2.1 | Disagree Sheet (modal) | Bottom sheet — reason picker; subplace of P2 |
| P3 | Session End | Total score, rank title, "Поделиться" + "Сыграть ещё" |
| P4 | Backend (FastAPI :8001) | Session + round + disagree API |

---

## UI Affordances

| # | Place | Component | Affordance | Control | Wires Out | Returns To |
|---|-------|-----------|------------|---------|-----------|------------|
| U1 | P1 | face-card | Face image (full viewport) | render | — | — |
| U2 | P1 | face-card | Round progress ("3/5") | render | — | — |
| U3 | P1 | face-card | "Нормально" button (guess=0) | click | → N2 | — |
| U4 | P1 | face-card | "Мошенник" button (guess=100) | click | → N2 | — |
| U5 | P2 | reveal-screen | Face image (same job_id) | render | — | — |
| U6 | P2 | reveal-screen | True sus score (animates in) | render | — | — |
| U7 | P2 | reveal-screen | Delta display ("Ты: 15 → Реальный: 94") | render | — | — |
| U8 | P2 | reveal-screen | Round score | render | — | — |
| U9 | P2 | reveal-screen | Explanation text (1–2 sentences) | render | — | — |
| U10 | P2 | reveal-screen | "Далее →" button | click | → N5 | — |
| U11 | P2 | reveal-screen | "Я не согласна" button | click | → P2.1 | — |
| U12 | P2.1 | disagree-sheet | Reason picker (4 options, single select) | select | → N6 | — |
| U13 | P3 | session-end | Total score | render | — | — |
| U14 | P3 | session-end | Rank title (one of 5 levels) | render | — | — |
| U15 | P3 | session-end | "Поделиться" button | click | → N7 | — |
| U16 | P3 | session-end | "Сыграть ещё" button | click | → N1 | — |

---

## Code Affordances

| # | Place | Component | Affordance | Control | Wires Out | Returns To |
|---|-------|-----------|------------|---------|-----------|------------|
| N1 | P1 | game-session | `startSession()` | call | → N8 | — |
| N8 | P4 | /game/session | `POST /game/session` | call | — | → N1b |
| N1b | P1 | game-session | session store write `{session_id, job_ids[5], round:0, scores[]}` | write | → N1c | → U2 |
| N1c | P1 | face-card | `loadRound(job_ids[round])` | call | → N9 | — |
| N9 | P4 | /thumbs/ | `GET /thumbs/{job_id}.jpg` | call | — | → U1 |
| N2 | P1 | face-card | `commitGuess(guess: 0\|100)` | call | → N3 | — |
| N3 | P1 | game-session | `guess$` store write | write | → N4 | — |
| N4 | P1 | game-session | navigate to reveal | call | → P2 | — |
| N12 | P2 | reveal-screen | `loadReveal(job_id, player_guess)` — fetches sus_level from manifest | call | → N13 | → U6, U7, U9 |
| N13 | P2 | reveal-screen | `calcScore(delta)` → `max(0, 1000 − \|delta\| × 10)` | call | — | → U8 |
| N5 | P2 | reveal-screen | `advanceRound()` | click | → N10 | — |
| N10 | P4 | /game/round | `POST /game/round {session_id, job_id, player_guess, true_sus, score}` | call | — | → N11 |
| N11 | P2 | game-session | `round++` + branch | call | → P3 if round==5 | → P1 if round<5 |
| N6 | P2.1 | disagree | `POST /game/disagree {job_id, session_id, reason, player_guess}` | call | — | → U12 ("Записано" 1.5s → auto-dismiss) |
| N7 | P3 | share | `Telegram.shareURL(text, url)` | call | — | — |

---

## Data Stores

| # | Place | Store | Description |
|---|-------|-------|-------------|
| S1 | P1 | `session` | `{session_id, job_ids[5], round:0, scores[]}` — persists full session |
| S2 | P1 | `currentGuess` | `{job_id, player_guess}` — cleared when entering P2 |
| S3 | P2 | `revealData` | `{true_sus, delta, round_score, explanation}` — populated by N12/N13 |

---

## Navigation Flow

```
TMA Opens
    → N1: startSession() → POST /game/session
    → P1: Face Card (round 1)

P1: Face Card
    U3/U4 tap → N2: commitGuess(0|100)
        → N3: write currentGuess
        → N4: navigate → P2

P2: Reveal Screen
    N12: loadReveal → sus_level, explanation, delta
    N13: calcScore → round_score
    U11: "Я не согласна" → P2.1 (blocks P2)
        P2.1: Disagree Sheet
            U12: select reason → N6: POST /game/disagree
            → "Записано" 1.5s → dismiss → P2
    U10: "Далее →" → N5: advanceRound
        → N10: POST /game/round
        → N11: round++
            if round < 5 → N1c: loadRound → P1
            if round == 5 → P3

P3: Session End
    U16: "Сыграть ещё" → N1: startSession → P1
    U15: "Поделиться" → N7: Telegram.shareURL
```

---

## Rabbit Holes

| Issue | Resolution |
|-------|-----------|
| `loadReveal` (N12) must not make a live LLM call | Explanation comes from a static template file: `sus_factor → string[]`. Must exist before E2-S3 ships. Content task (Кирилл/Sonya), not engineering. |
| `advanceRound` waits on `POST /game/round` | Optimistic UI acceptable — show "Далее →" immediately, fire POST in background. Round score is client-calculated anyway. |
| P2.1 (disagree sheet) blocks P2 | Correct — user cannot tap "Далее →" while picker is open. After "Записано" or dismiss, returns to P2. |
| `true_sus` for reveal | Fetched from local manifest JSON (client-side), not a network call. Manifest loaded at session start. |
| Rank title calculation | Client-side lookup: score ranges → `{Новичок, Стажёр, Инспектор, Старший инспектор, Главный детектив}`. No API needed. |

---

## E2-S2 Delta (Slider)

When slider replaces binary buttons, P1 changes:

| # | Change | Detail |
|---|--------|--------|
| U3, U4 | **Remove** binary buttons | |
| U3′ | **Add** slider (0–100) | drag → N2′: `sliderValue.next(v)` |
| U4′ | **Add** live value display ("47") | render from S2 store |
| U5′ | **Add** "Подтвердить" button | click → N2 (same wire as before) |

Everything downstream of N2 is unchanged.
