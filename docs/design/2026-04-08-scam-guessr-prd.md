# PRD: Scam Guessr

**Date:** 2026-04-08  
**Status:** Draft  
**Owner:** Кирилл  
**Context:** Swan song pivot from telejobs. Original product sunsetting due to zero retention.

---

## Problem Statement

Telejobs proved the SUS engine works technically but failed as a product: bounce rate 72%, WAU collapsed from 50 to 7 in one week, zero repeat reports from "crusaders." The root cause is structural — job fraud detection is an episodic, civic-duty task with no intrinsic pull. Users have no reason to return.

Scam Guessr reframes the same underlying data as a skill game. The pull mechanism is not "protect yourself from fraud" (instrumental, forgettable) but "can you beat yesterday's score" (intrinsic, daily). The SUS engine and 24K-posting corpus become the content layer for a mobile-first game, not the product itself.

---

## Goals

1. **Retention:** 7-day return rate ≥ 30% among first-week players (vs. ~0% for telejobs)
2. **Session depth:** Average 8+ rounds per session (vs. 1.5 min bounce on telejobs)
3. **Virality:** ≥ 15% of completed sessions result in a shared result card
4. **Data flywheel:** ≥ 5% of revealed rounds produce a disagree signal, building a labeling stream
5. **Validation:** Demonstrable skill improvement — player accuracy increases measurably across 10+ sessions

---

## Non-Goals

1. **Not building job search UI.** No search bar, filters, or apply buttons in Phase 1. Job text is a reveal reward after the guess — not browsable inventory. If a player reads the reveal and pursues a posting, that's a win, not scope creep.
2. **Not building a moderation pipeline.** Disagree signals feed model retraining, not a human review queue. No moderation UI, no SLA on acting on signals.
3. **Not a real-time pipeline.** The corpus is pre-generated and static. No live scraping or daily fresh postings in Phase 1.
4. **Not multi-platform at launch.** Web-first (Vite + React). TMA (Telegram Mini App) is Phase 2 — the original TMA did not solve retention.
5. **Not localized beyond Russian corpus.** The SUS engine was trained on Russian-language Telegram job channels. English/other language support is out of scope.

---

## User Personas

### Primary: Студент / Casual Browser ("Аня")
Student or young professional, 18–28. Encountered scam job postings personally or knows someone who was defrauded. Plays during commute or breaks. Does not think in data or embeddings. Motivation: entertainment, social sharing, mild civic pride. **This is the retention target.**

### Secondary: Scam Hunter ("Охотник")
Investigator or power user who processes many postings. Needs fast triage. Comes for the corpus map, not the card game. Acceptable churn — single long sessions over repeat daily visits.

### Tertiary: Analyst / Researcher
Model improvement focus. Uses disagreement data to find blind spots. Not a retention metric — engagement is deep but infrequent.

---

## User Stories

### Аня (Casual Browser)

- As Аня, I want to see a face and guess how suspicious it is, so I can test my intuition against the AI
- As Аня, I want the reveal to show me *why* a posting is suspicious (not just a score), so I learn something each round
- As Аня, I want to share my result as an image to a chat, so I can challenge friends
- As Аня, I want a daily streak counter, so I have a reason to open the app tomorrow
- As Аня, I want a rank title that updates as I improve, so I feel progression

### Scam Hunter

- As a Hunter, I want to select a cluster of faces on the map and play them as a batch, so I can process a suspicious region efficiently
- As a Hunter, I want to filter by cohort or sus range before playing, so I focus on high-priority postings

### Analyst

- As an Analyst, I want my disagreement tagged with a reason, so the signal is actionable for model retraining
- As an Analyst, I want to see which faces have the most disagreements, so I can identify model blind spots

---

## Requirements

### P0 — Must Have (Phase 1 MVP)

**Core game loop**
- [ ] Face card appears, filling most of the viewport
- [ ] Sus slider (0–100) below the face; face morphs toward uncanny valley in real time as slider moves
- [ ] Commit button locks the guess
- [ ] Reveal: true sus score animates in; face snaps to its actual uncanny state; delta score displays
- [ ] sus_factors flash up sequentially after reveal ("Предоплата • Нет компании • Адрес неизвестен")
- [ ] Round score = `max(0, 1000 - |guess - actual| * 10)`; cumulative session score displayed
- [ ] "Next →" advances to next card; session ends after 5 rounds

**Disagree mechanic**
- [ ] After reveal: two buttons — "Понятно" (next) and "Не согласен"
- [ ] "Не согласен" opens a 4-option radio: "Вакансия выглядит легитимно" / "Компания реальная" / "Критерии не применимы" / "Другой регион/язык"
- [ ] Disagreement logged with job_id, player's guess, true sus, reason, timestamp

**Sharing**
- [ ] End-of-session result card: cumulative score, rounds played, best round, rank title
- [ ] Result card is a canvas-rendered image, shareable via Web Share API or download
- [ ] Card includes a short URL / QR to play

**Rank system**
- [ ] 5 titles based on rolling session accuracy: Новичок → Стажёр → Инспектор → Старший инспектор → Главный детектив
- [ ] Rank stored in localStorage (no backend required for Phase 1)

**Face assets**
- [ ] All 543 face v5 portraits (Flux + LoRA, uncanny valley encoded) pre-generated and served as static 192px JPEGs
- [ ] Faces are served from `/public/thumbs/`; no runtime generation

**Backend (telejobs processing-service, port 8001)**
- [ ] `POST /game/session` — create session, return 5 job_ids for the round (or daily 5 if flagged)
- [ ] `POST /game/round` — submit result: job_id, player_guess, true_sus, delta, score
- [ ] `POST /game/disagree` — submit disagree: job_id, player_guess, true_sus, reason
- [ ] `GET /game/daily` — today's 5 job_ids, seeded by UTC date in Redis, same for all users
- [ ] `GET /game/leaderboard` — top 10 daily scores (nickname + score, no PII)
- [ ] New DB tables: `game_sessions`, `game_rounds`, `game_disagrees`, `game_players`
- [ ] Anonymous session via UUID in localStorage linked to backend (no auth required for Phase 1)
- [ ] Telegram OIDC auth reused as-is for Phase 2 TMA upgrade

---

### P1 — Nice to Have (Phase 2)

**Competition layer**
- [ ] Daily 5: same 5 face IDs for all players each UTC day (seeded by date)
- [ ] Daily leaderboard (top 10 scores, no auth — nickname entry only)
- [ ] Streak counter: consecutive days with ≥1 completed session
- [ ] Streak displayed on result card

**Telegram Mini App**
- [ ] Wrap Phase 1 web app as TMA
- [ ] TMA-native sharing to Telegram chat
- [ ] Note: only ship if Phase 1 web retention ≥ 30%

**Fraud landscape map**
- [ ] PaCMAP 2D layout of full corpus, rendered with deck.gl
- [ ] Semantic zoom: dots at low zoom, faces at high zoom
- [ ] HDBSCAN cluster hulls colored by mean sus_level
- [ ] Lasso select → "Play these N faces"

---

### P2 — Future Considerations

- **Duels:** Real-time head-to-head (requires WebSocket backend)
- **Retroactive scoring:** Points awarded if disagreement later validated by model correction
- **Skill progression charts:** Per-player accuracy curve across sessions
- **Multi-corpus:** hh.ru, VK-groups (requires SUS engine extension research per Вариант Б2)
- **Backend auth:** Persistent accounts, cross-device progress sync

---

## Success Metrics

### Leading indicators (visible within 2 weeks of launch)

| Metric | Target | Measurement |
|--------|--------|-------------|
| Sessions per DAU | ≥ 1.5 | localStorage event log |
| Rounds per session | ≥ 8 | localStorage event log |
| Share rate | ≥ 15% of sessions | Web Share API callback |
| Disagree rate | ≥ 5% of rounds | Disagree log (local or API) |

### Lagging indicators (4-week mark)

| Metric | Target | Notes |
|--------|--------|-------|
| D7 retention | ≥ 30% | Cohort of Week 1 players returning in Week 2 |
| Accuracy improvement | ≥ 10% delta (session 1 vs. session 10) | Validates skill-building claim |
| Organic shares | ≥ 50 external shares | Social proof, distribution |

**Kill metric:** If D7 retention < 10% after 4 weeks, close the project. No pivot to TMA without evidence the loop works.

---

## Open Questions

| Question | Owner | Urgency |
|----------|-------|---------|
| Does the face morphing on slider drag require pre-rendered keyframes or CSS blend? | Engineering (Кирилл) | P0 — affects Phase 1 scope |
| Daily 5 seed: date-based determinism is enough for Phase 1 without a server? | Engineering | P1 |
| ~~Should disagree data be posted to an API endpoint or stored locally?~~ **Resolved: use telejobs processing-service.** Anonymous UUID session, no auth wall for Phase 1. | — | Closed |
| Rank titles: Russian-language only or also English for potential international demo? | Product | P1 |
| Face v5 generation: are all 543 faces generated and reviewed before launch? | Engineering | P0 blocker |

---

## Timeline Considerations

**Hard constraint:** Minimal engineering bandwidth (Кирилл solo, part-time). All Phase 1 scope must be achievable in ≤ 2 weeks of focused work.

**Stack:** Scam Guessr frontend (Vite + React, existing `web/`) → telejobs processing-service (FastAPI, port 8001) → PostgreSQL + Redis. No new infrastructure.

**Phase 1 (MVP, web):** Core loop + sharing + disagree + rank. Target: 2 weeks.  
**Phase 2 (competition):** Daily 5 + streaks + TMA. Target: only if Phase 1 retention ≥ 30%.  
**Phase 3 (map + duels):** Only if Phase 2 validates continued investment.

**Dependency:** Face v5 batch generation must complete before Phase 1 launch. Generation is running now (~25 min).

---

## Why This Addresses the Original Failure

| telejobs failure mode | Scam Guessr solution |
|---|---|
| No pull mechanism (episodic use) | Daily streak + daily 5 creates a daily habit trigger |
| Crusader hypothesis failed (reporting = civic duty) | Disagreement is a game action, not a report — no civic framing |
| 1.5 min session, bounce 72% | 8-round session target = ~4 min minimum; game loop keeps players |
| Distribution blocked (Pikabu, Habr) | Shareable result card = native viral distribution via messaging apps |
| TMA didn't solve retention | TMA is Phase 2, gated behind retention evidence |
