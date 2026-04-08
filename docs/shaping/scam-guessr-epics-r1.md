---
shaping: true
---

# Epic Breakdown: Scam Guessr R1
**Method:** Humanizing Work (Richard Lawrence) — 9 splitting patterns
**Date:** 2026-04-08
**Scope:** Release 1 (MVP) — all 🟥 tasks from story map

---

## E1: Face Asset Pipeline

**Epic:** 543 AI-generated face portraits (v5, Flux + LoRA) served as 192px JPEG thumbnails, each linked to a job_id with sus_level and sus_factors metadata.

**INVEST check:** ✅ Independent (no other R1 epic depends on this being complete to start) | ✅ Valuable (without assets, no game) | ✅ Estimable | ✅ Testable
**Status:** Generation running (~30 min remaining). Thumbnail pipeline already built in `web/public/thumbs/`.

**Pattern applied: None — already small enough (2–3 day task, nearly done)**

**Stories:**

### E1-S1: Generated face assets served as static thumbnails (DONE ~90%)
- As Аня, when I open a game round, I see a real portrait face rendered at full viewport width
- **AC:** Given generation completes, when server serves `/thumbs/{job_id}.jpg`, then 192px JPEG loads in <500ms on mobile
- **AC:** All 543 faces have corresponding manifest entries with sus_level, sus_factors, x, y coordinates
- **Effort:** 0.5 days (monitor + verify completion)

### E1-S2: Metadata index for game serving
- As the game backend, I need to query faces by job_id and retrieve sus_level + sus_factors for reveal
- **AC:** Given a job_id, the API returns sus_level (0–100) and top 3 sus_factors as strings
- **AC:** Response time <50ms (served from DB, no generation)
- **Effort:** 0.5 days

**Split evaluation:** ✅ Reveals low-value work — only 543 faces needed for R1, not full 24k corpus. Full generation is R2 scope.

---

## E2: Game Core

**Epic:** Player sees a face, makes a guess (binary or slider), commits, sees reveal with morphed face and specific explanation, receives a score. Session is 5 rounds.

**INVEST check:** ✅ All pass. High uncertainty on one item: does the face morph on reveal require pre-rendered keyframes or live CSS? → Spike needed before E2-S3.

**Pattern applied: Pattern 7 (Simple/Complex)**
Simplest version that delivers end-to-end value: face + binary guess + reveal + score. Complexity added progressively.

---

### E2-S1: Face → Binary Guess → Reveal → Score (Simplest Complete Slice)
**As** Аня
**I want to** see a face, tap "Нормально" or "Мошенник", and immediately see the real sus score
**so that** I get feedback on my intuition without any friction

**AC:**
- Given a game session is active, when I see a face card, then two buttons appear: "Нормально" (≤50) and "Мошенник" (>50)
- When I tap either button, then the true sus score is revealed with a number (0–100)
- When sus score is revealed, then my round score is shown: 1000 points if within ±10, scaling down to 0 at ±50+
- When a session of 5 rounds completes, then total session score is displayed
- Face thumbnail loads in <1s on 4G mobile

**Why first:** Delivers full game loop end-to-end. Binary tap is fastest to build and test. No animation, no slider.
**Effort:** 2 days

---

### E2-S2: Replace Binary with Slider (0–100)
**As** Аня
**I want to** drag a slider to set my sus estimate before revealing
**so that** my score reflects how precisely I read the face, not just which side of 50 I picked

**AC:**
- Given a face card is shown, then a 0–100 slider replaces the two buttons
- When I drag the slider, then the number updates live
- When I commit, then delta = |my_guess − true_sus| and score = max(0, 1000 − delta × 10)
- Binary tap shortcut still available (tapping left half = 0, right half = 100)

**Why after S1:** S1 proves the loop works. Slider adds precision without changing the architecture.
**Effort:** 1 day

---

### E2-S3: Specific Reveal Explanation (R9 — critical)
**As** Аня
**I want to** read a specific 1–2 sentence explanation of why the AI scored this face high or low
**so that** a wrong answer feels like discovery rather than failure, and I learn something I want to share

**AC:**
- Given a reveal, then 1–2 sentences appear below the score
- Sentences reference specific sus_factors: "Запрос предоплаты за оборудование — классический признак мошенничества с курьерами"
- For large delta (player wrong by >40): explanation reframes as "AI sees what humans can't" — never "you were wrong"
- For small delta (player right within ±15): explanation validates — "Ты почувствовала правильно — потому что..."
- No explanation uses the words "подозрительно", "мошенничество" without specifics

**Note:** Explanation copy requires a dedicated writing pass — one template per sus_factor (16 factors × 2–3 templates each). This is a content task, not engineering.
**Effort:** 1 day engineering + 1 day copy writing

---

### E2-S4: Face Morphs to Uncanny State on Reveal *(Spike first)*
**As** Аня
**I want to** see the face transform to its "true" uncanny state when the score reveals
**so that** a large delta creates a visceral "oh god" moment that makes me want to share

**Spike required first:**
- Q1: Can we cross-fade between two pre-rendered face images (clean anchor vs. actual v5 face) with CSS transition?
- Q2: Or do we need keyframe animation using multiple intermediate images?
- Q3: What is acceptable frame budget on mobile (60fps vs 30fps)?
- Spike: 0.5 days → informs implementation

**AC (post-spike):**
- Given delta > 40, when score reveals, then face transitions from neutral state to actual uncanny portrait over 800ms
- Given delta < 15, when score reveals, then face barely changes (subtle confirmation)
- Transition is smooth on mid-range Android (60fps target, 30fps acceptable)

**Effort:** 1–2 days (post-spike)

---

**E2 split evaluation:**
✅ **Reveals low-value work:** E2-S4 (morph animation) could be cut from R1 if spike reveals high complexity. The loop works without it (S1–S3 are complete). Ship S4 if spike shows it's CSS + 2 images — defer if it needs keyframes.
✅ **Equal-sized stories:** S1=2d, S2=1d, S3=2d, S4=1-2d (post-spike)

---

## E3: Social Loop

**Epic:** At session end, a designed result card is generated and shared to Telegram. A friend taps the card, opens the TMA, and sees a competitive entry context.

**INVEST check:** ✅ All pass. Depends on E2-S1 (session end state exists).

**Pattern applied: Pattern 1 (Workflow Steps — thin end-to-end)**

The workflow: session ends → card generated → player shares → friend opens TMA.
Simplest slice: session end screen with score + share button (no designed card image yet — just text share).

---

### E3-S1: Session End Screen + Text Share (Simplest Complete Slice)
**As** Аня
**I want to** see my total score at the end of 5 rounds and share it to Telegram
**so that** I can challenge a friend with minimal friction

**AC:**
- Given 5 rounds completed, then session end screen shows: total score, rank title, "Поделиться" button
- When I tap "Поделиться", then Telegram share sheet opens with: score, rank title, deep link to TMA
- Deep link opens TMA game entry (not home screen) for the recipient
- Recipient sees: "Аня набрала 3 840 — побей её результат" on entry
- **Exit hook displayed before share button** (one of, rotated):
  - Session 1–2: Best score today: "Лучший результат сегодня: 5 120. Ты: 3 840 — попробуй снова?" (score gap hook)
  - Session 3+: Score trajectory if improving: "Сессия 1: 2 840 → Сегодня: 3 840 📈 Точность растёт" (skill validation hook)
  - Always: silhouette of one face from next session ("Кто это?") — curiosity hook
- No streak language. No "не пропусти завтра" pressure. Exit hook is pull, not push.

**Effort:** 1.5 days

---

### E3-S2: Result Card as Designed Image (R10)
**As** Аня
**I want to** share a visually designed card (not just text) that shows my face thumbnail and score
**so that** the share looks compelling enough in a Telegram chat that friends tap it

**AC:**
- Given session end, then a canvas-rendered card is generated: face thumbnail of round with highest score + total score + rank title + punchy line
- Card exports as JPEG blob, shared as image via Telegram `shareURL` API
- Card renders correctly at Telegram preview dimensions (1200×630px)
- Card includes deep link embedded as text below image

**Effort:** 2 days

---

### E3-S3: Competitive Re-entry Context
**As** a friend who tapped Аня's result card
**I want to** enter the game knowing Аня's score so I'm motivated to beat it
**so that** my first session has a competitive frame, not a blank start

**AC:**
- Given I tap a deep link with referrer_score=3840 and referrer_name=Аня, then first game screen shows: "Аня набрала 3 840 — побей её результат"
- Given I complete my session, then I see both my score and Аня's score on the result screen
- Given my score > Аня's, then "Ты победила! Отправь Ане свой результат" appears

**Effort:** 1 day

---

**E3 split evaluation:**
✅ **Reveals low-value work:** E3-S3 (competitive re-entry) could be deferred — the viral loop works even without it. But it significantly improves click-through on result cards. Include in R1 only if S1+S2 go smoothly.
✅ **Equal-sized stories:** S1=1.5d, S2=2d, S3=1d

---

## E4: Disagree Signal

**Epic:** After any reveal, player can tap "Я не согласна", select a reason, and the disagreement is stored with job_id, player guess, true sus, reason, and timestamp (R12 data model).

**INVEST check:** ✅ All pass. Depends on E5-S1 (backend schema must exist for storage).

**Pattern applied: Pattern 6 (Major Effort)**
The data model + API endpoint is the infrastructure. The UI is simple. Build infrastructure first.

---

### E4-S1: Disagree Data Model + API (Infrastructure)
**As** the system
**I need to** store every disagreement with full context
**so that** R12 impact tracking ("flagged before ban") is possible from day one

**AC (backend):**
- Table `game_disagrees`: id, job_id, player_id, session_id, player_guess, true_sus, reason, flagged_at (timestamp), resolution (nullable — filled when moderation outcome known)
- `POST /game/disagree` accepts: job_id, session_id, reason (enum), player_guess
- Response: `{ "recorded": true, "others_disagreed": N }` (N = count of prior disagreements on this job)
- `flagged_at` stored as UTC timestamp — required for future "flagged N days before ban" calculation

**Why first:** Without this schema in R1, R3's "flagged before ban" becomes a painful retrofit.
**Effort:** 1 day

---

### E4-S2: Disagree UI — Button + Reason Picker
**As** Аня
**I want to** tap "Я не согласна" after a reveal and quickly select a reason
**so that** I feel agency over the AI's verdict without it feeling like form-filling

**AC:**
- Given any reveal screen, then "Я не согласна" button is visible (secondary to "Далее →")
- When I tap it, then a bottom sheet shows 4 options (single select):
  - Вакансия выглядит легитимно
  - Компания реальная
  - Критерии не применимы
  - Другой регион или язык
- When I select a reason, then "Записано" confirmation appears for 1.5s, then auto-advances
- Disagreement is posted to E4-S1 API

**Effort:** 1 day

---

**E4 split evaluation:**
✅ **Reveals low-value work:** The "others_disagreed" counter (E4-S1 response) could be dropped from R1 UI — store the data, surface the count in R2. Reduces scope without losing the data.
✅ **Equal-sized stories:** S1=1d, S2=1d

---

## E5: Backend API

**Epic:** telejobs processing-service (FastAPI, port 8001) gains game session management, round recording, and disagree storage.

**INVEST check:** ✅ All pass. E2, E3, E4 all depend on this.

**Pattern applied: Pattern 1 (Workflow Steps)**
The workflow: session created → rounds played → session completed → disagrees recorded.
Build the thinnest session API first (create + record round), then add disagree, then daily challenge.

---

### E5-S1: Session + Round API (Core Infrastructure)
**As** the game frontend
**I need to** create a session and record each round result
**so that** player scores persist server-side and can feed leaderboards

**AC:**
- `POST /game/session` → returns `{ session_id, job_ids: [5 job_ids] }` — **stratified selection, not random:**
  - 1 face: `sus_level ≥ 80` (visceral reveal — the "oh god" moment)
  - 1 face: `sus_level ≤ 20` (the "I was wrong to judge" subversion)
  - 3 faces: `sus_level 30–70` (genuine ambiguity, challenge)
  - Order served: medium → high → medium → low → medium (shock arrives at round 2, not round 1)
- `POST /game/round` → accepts `{ session_id, job_id, player_guess, true_sus, score }` → stores round
- New tables: `game_sessions` (id, player_id, started_at, completed_at, total_score), `game_rounds` (id, session_id, job_id, player_guess, true_sus, score, **delta**, round_number) — `delta` stored from day 1 to enable R2 adaptive difficulty without schema migration
- Telegram user_id used as player_id (from TMA init data); anonymous UUID for guest sessions
- All endpoints require no auth for guest; optional TMA auth for persistent progress

**Why stratified:** Random selection cannot guarantee the engagement curve — some random sessions would be all-ambiguous (no "wow" moment) or all-obvious (no challenge). Stratification ensures each session hits the required emotional beats regardless of corpus composition.

**Effort:** 2 days

---

### E5-S2: Disagree API + R12 Schema
*(Defined under E4-S1 — same story, owned by backend)*

---

### E5-S3: Daily Challenge Endpoint
**As** the game frontend
**I need to** fetch the same 5 job_ids for all players on a given UTC day
**so that** a daily challenge creates a shared experience and social comparison

**AC:**
- `GET /game/daily?date=2026-04-08` → returns `{ date, job_ids: [5 job_ids] }`, same for all callers
- Job IDs seeded deterministically: `sha256(date_str + salt)[:5]` → indices into sorted corpus
- Result cached in Redis with TTL until next UTC midnight
- Guest players can play daily challenge without auth

**Effort:** 1 day

---

**E5 split evaluation:**
✅ **Reveals low-value work:** E5-S3 (daily challenge) can ship after R1 launch — infinite mode works without it. Daily challenge is R1 if time permits, otherwise first week of R2.
✅ **Equal-sized stories:** S1=2d, S2=1d (under E4), S3=1d

---

## R1 Sprint Plan

| Story | Description | Effort | Depends on |
|-------|-------------|--------|------------|
| E1-S1 | Verify face generation complete + thumbnails served | 0.5d | — |
| E1-S2 | Metadata index API (sus_level + sus_factors per job_id) | 0.5d | E1-S1 |
| E5-S1 | Session + Round API + schema (stratified selection + delta col) | 2d | — |
| E4-S1 | Disagree schema + API (R12 data model) | 1d | E5-S1 |
| E2-S1 | Face → Binary Guess → Reveal → Score | 2d | E1-S1, E5-S1 |
| E2-S2 | Replace binary with slider | 1d | E2-S1 |
| E2-S3 | Reveal explanation copy + UI (R9) | 2d | E2-S2 *(reordered: slider first enables precision mechanic)* |
| E3-S1 | Session end screen + text share + exit hook | 1.5d | E2-S1 |
| E4-S2 | Disagree button + reason picker UI | 1d | E2-S1, E4-S1 |
| E3-S2 | Result card as designed image (R10) | 2d | E3-S1 |
| E3-S3 | Competitive re-entry context | 1d | E3-S2 |
| E2-S4 | Face morph on reveal | 1.5d | Spike + E2-S1 |
| E5-S3 | Daily challenge endpoint | 1d | E5-S1 |

**Critical path:** E1-S1 → E5-S1 → E2-S1 → E2-S2 → E2-S3 → E3-S1 → E3-S2

**Total R1 effort:** ~18 days (1 developer, +1d from engagement design additions)
**Parallelisable:** E5-S1 and E1-S1 can run simultaneously. E4-S1 can run in parallel with E2-S1 once E5-S1 is done.

---

## Pre-Launch Blockers (Not Engineering)

| Blocker | Owner | Notes |
|---------|-------|-------|
| Explanation copy — 16 sus_factors × 2-3 templates each | Кирилл / Sonya | Must exist before E2-S3. Not auto-generatable. |
| E2-S4 spike (face morph feasibility) | Кирилл | 0.5d, gates E2-S4 scope decision |
| TMA `shareURL` behaviour on iOS vs Android | Кирилл | Verify before E3-S1 |
