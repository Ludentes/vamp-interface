---
shaping: true
---

# Scam Guessr — Proto-Personas

**Date:** 2026-04-08
**Status:** Proto (assumption-based) — needs validation
**Source:** Telejobs pilot data (30 students, W10–W13), CEO report 2026-03-27, scenarios.md

---

## Proto-Persona 1: Anxious Аня (Primary)

### Bio & Demographics
- 19–23 years old, Russian-speaking, major Russian city
- Student or recent graduate, seeking part-time or first job
- Telegram-native — discovers everything through channels and group chats
- Active on VK and short-form video; shares things that make her look smart
- Job-hunting is stressful and episodic, not a daily activity

### Quotes
- *"Я видела вакансию с зарплатой 100к за 4 часа в день — явно развод, но всё равно страшно кликать."* [pilot sentiment, ASSUMPTION—VALIDATE exact phrasing]
- *"Покажи друзьям, они тоже ищут."* [sharing behavior observed in pilot]
- *"На hh.ru тоже много мусора, круто было бы фильтровать и там."* [verbatim, Sonya's classmates, CEO report]

### Pains
- Can't tell scam from real without reading the full posting — by which point she's already nervous
- Embarrassed to have nearly applied to an obvious scam; doesn't want to feel naive
- Job browsing is tedious; all postings look like the same wall of text
- [ASSUMPTION—VALIDATE] Doesn't fully trust anonymous aggregators — Telegram channels feel more authentic

### What She's Trying to Accomplish
- Screen job postings faster without reading every word
- Feel confident she's not walking into a scam
- Have something interesting to share that makes her look sharp

### Goals
- Short-term: find a legit part-time job without being defrauded
- Social: demonstrate she's savvy about online scams (peer status)
- [ASSUMPTION—VALIDATE] Latent: would enjoy proving an AI wrong — "even the bot made a mistake"

### Attitudes & Influences
- **Decision-Making Authority:** Free user; no purchase decision
- **Influenced by:** Friends in Telegram group chats, peer recommendations, viral screenshots
- **Beliefs:** Skeptical of "official" products; trusts things spread organically by peers. LLMs feel like black boxes — catching one in a mistake is satisfying.

### R5 signal
Аня does NOT have a daily job-hunting habit. She job-hunts in bursts.
→ Infinite mode with optional daily challenge fits better than daily-or-nothing.
[ASSUMPTION—VALIDATE]

---

## Proto-Persona 2: Outraged Олег (Secondary)

### Bio & Demographics
- 28–40 years old, Russian-speaking, tech-adjacent background
- Sysadmin, developer, or security hobbyist
- Active in anti-fraud Telegram channels; screenshots and shares scam postings as a hobby
- Has personal history with fraud — himself or someone close [ASSUMPTION—VALIDATE]
- Engages evenings and weekends; this is leisure, not work

### Quotes
- *"Они уже 5-й раз постят одну и ту же схему с разных аккаунтов."* [inferred from scenario doc]
- *"Я за 30 секунд вижу, что это развод. А люди всё равно ведутся."*
- [PLACEHOLDER—NEEDS RESEARCH: real quotes from anti-fraud Telegram community members]

### Pains
- Current tools let him sort by score but not *scan* — reading cards one by one is slow
- Can't spot coordinated rings without side-by-side comparison
- Reports to admins but gets no feedback — no sense of impact
- Engagement collapses after novelty wears off (confirmed: pilot W12→W13, 18 reports → 0)

### What He's Trying to Accomplish
- Process 200 postings in the time it takes to read 20
- Identify coordinated rings, not just individual scams
- Feel impact — know that his flagging did something

### Goals
- Expose fraud at scale, not one posting at a time
- [ASSUMPTION—VALIDATE] Competitive satisfaction: be "better than the AI"
- Recognition in his anti-fraud community

### Attitudes & Influences
- **Decision-Making Authority:** Free user; might pay for Pro features [ASSUMPTION—VALIDATE]
- **Influenced by:** Anti-fraud Telegram community, news about scam victims
- **Beliefs:** LLMs miss things obvious to humans. "I disagree" button validates his expertise.

### R5 signal
Олег does sessions, not dailies. He opens the app when a new scam wave hits his channels.
→ Infinite mode with cluster/map navigation fits; daily challenge feels like homework.
[ASSUMPTION—VALIDATE]

---

## R5 Resolution

Both personas point the same direction:
**Infinite mode is the base. Daily challenge is opt-in — a social object for Аня, irrelevant for Олег.**
A daily-or-nothing structure punishes Аня (episodic habit) and bores Олег (session-based).

---

## Assumptions to Validate (Priority Order)

| # | Assumption | Persona | Risk if wrong |
|---|-----------|---------|---------------|
| A1 | Аня finds "proving the AI wrong" intrinsically satisfying [R3] | Аня | Core mechanic loses its hook |
| A2 | Олег would return beyond week 1 if given cluster/ring detection, not just card-by-card [R0] | Олег | Secondary persona has zero retention just like pilot |
| A3 | Аня would use daily challenge if friends are playing the same one [R5, social] | Аня | Daily challenge adds complexity without retention benefit |
| A4 | Олег has personal fraud history (motivation source) | Олег | Misread his motivation; wrong retention mechanics |
| A5 | Аня trusts Telegram-spread products more than "official" ones | Аня | TMA distribution strategy may not reach her |
