---
shaping: true
---

# User Story Map: Scam Guessr

**Segment:** Russian-speaking students and young adults (19–25) on Telegram
**Persona:** Anxious Аня — casual, entertainment-first, Telegram-native
**Narrative:** "Develop fraud detection intuition through play and share moments of insight with friends"
**Date:** 2026-04-08

---

## Backbone (Activities)

```
[1. Discover & Enter] → [2. Play a Session] → [3. Share & Re-enter] → [4. Build Skill] → [5. Contribute]
```

Activities are user behaviors, not features. Left-to-right = chronological journey.

---

## Full Story Map

### Activity 1: Discover & Enter

**Step 1.1 — See result card in Telegram chat**

| Priority | Task |
|----------|------|
| 🟥 R1 | Result card renders as a designed image (canvas), not a plain screenshot |
| 🟥 R1 | Card shows: face thumbnail + score + shareable line ("Аня угадала 4/5 — а ты?") |
| 🟥 R1 | Card includes deep link back to the TMA |
| 🟨 R2 | Card shows sender's rank title ("Инспектор Аня вызывает тебя") |

**Step 1.2 — Open TMA via deep link**

| Priority | Task |
|----------|------|
| 🟥 R1 | Deep link opens TMA in correct context (game entry, not home screen) |
| 🟥 R1 | Telegram identity used automatically — zero login screen |
| 🟥 R1 | Guest session created for web fallback (no Telegram) — no auth wall |
| 🟨 R2 | Friend context shown on entry: "Дашка набрала 720 — попробуй больше" |

**Step 1.3 — See first face immediately**

| Priority | Task |
|----------|------|
| 🟥 R1 | First face is the first screen — no splash, no tutorial, no permissions prompt |
| 🟥 R1 | Time to first interactive face < 3 seconds |
| 🟥 R1 | Slider (0–100) visible immediately below face |
| 🟨 R2 | Face loads at full viewport height on mobile |

---

### Activity 2: Play a Session

**Step 2.1 — Form intuition looking at the face**

| Priority | Task |
|----------|------|
| 🟥 R1 | Face renders at full viewport, no clutter — just face + slider |
| 🟥 R1 | Uncanny valley encoding visible — high-sus faces feel wrong without explanation |
| 🟨 R2 | Face animates subtly (breathing, micro-expression) to reward lingering |

**Step 2.2 — Make a guess**

| Priority | Task |
|----------|------|
| 🟥 R1 | Drag slider 0–100 to set sus estimate |
| 🟥 R1 | Commit button locks the guess and triggers reveal |
| 🟥 R1 | Quick binary tap option (Нормально / Мошенник) for fast play without slider precision |
| 🟨 R2 | Face morphs live as slider moves — previews uncanny state before committing |

**Step 2.3 — Experience the reveal**

| Priority | Task |
|----------|------|
| 🟥 R1 | True sus score animates in with dramatic timing — not instant |
| 🟥 R1 | Face snaps to actual uncanny state (if delta is large, this is the "oh god" moment) |
| 🟥 R1 | Delta shown: "Ты: 15 → Реальный: 94" |
| 🟥 R1 | Round score calculated: max(0, 1000 − |delta| × 10) |

**Step 2.4 — Read the explanation (R9 — critical)**

| Priority | Task |
|----------|------|
| 🟥 R1 | 1–2 sentence explanation, specific and factor-based |
| 🟥 R1 | Framed as discovery: "Асимметрия челюсти характерна для ..." NOT "подозрительная вакансия" |
| 🟥 R1 | For large delta: explanation reframeable as "AI sees what humans can't" |
| 🟥 R1 | For small delta: explanation validates: "Ты это почувствовала правильно — потому что ..." |
| 🟨 R2 | Top 2–3 sus_factors shown as pill tags after explanation |

**Step 2.5 — Complete a session**

| Priority | Task |
|----------|------|
| 🟥 R1 | Session = 5 rounds; progress shown (round 3/5) |
| 🟥 R1 | Cumulative session score shown after each round |
| 🟥 R1 | Session end screen: total score + rank title |
| 🟥 R1 | Rank titles (5 levels): Новичок → Стажёр → Инспектор → Старший инспектор → Главный детектив |
| 🟨 R2 | Personal best comparison: "Твой рекорд: 3 840. Сегодня: 4 120 🎉" |

**Step 2.6 — Disagree with the model**

| Priority | Task |
|----------|------|
| 🟥 R1 | "Я не согласна" button appears on every reveal screen |
| 🟥 R1 | Reason picker: Вакансия легитимна / Компания реальная / Критерии не применимы / Другой регион или язык |
| 🟥 R1 | Confirmation: "Твоё несогласие записано" |
| 🟥 R1 | Disagreement stored: job_id + player guess + true sus + reason + timestamp (R12 data model) |
| 🟨 R2 | "Ещё X человек не согласились с этой оценкой" — social proof signal |

---

### Activity 3: Share & Re-enter

**Step 3.1 — See result card at session end**

| Priority | Task |
|----------|------|
| 🟥 R1 | Result card auto-generated: face thumbnail + session score + shareable line |
| 🟥 R1 | "Поделиться" button prominent; "Сыграть ещё" secondary |
| 🟥 R1 | Card renders as image (canvas → blob) suitable for Telegram share |

**Step 3.2 — Share to Telegram**

| Priority | Task |
|----------|------|
| 🟥 R1 | Telegram native share sheet (TMA `shareURL` API) |
| 🟥 R1 | Deep link embedded so friend's tap opens TMA in game mode |
| 🟥 R1 | Web fallback: Web Share API + clipboard copy |

**Step 3.3 — Friend enters through shared card (viral loop)**

| Priority | Task |
|----------|------|
| 🟥 R1 | Friend taps card → TMA opens → first face immediate (same as 1.3) |
| 🟥 R1 | Entry context: "Дашка набрала 3 200 — побей её результат" |
| 🟨 R2 | After friend's session: nudge back to Аня's chat with result |

---

### Activity 4: Build Skill *(Release 2)*

**Step 4.1 — Return and keep playing**

| Priority | Task |
|----------|------|
| 🟨 R2 | Infinite mode: corpus always has fresh faces (24/7 scraping) |
| 🟨 R2 | Daily challenge: same 5 faces for all players, date-seeded server-side |
| 🟨 R2 | Daily challenge entry point: "Дашка уже сыграла сегодня" — social framing, not obligation |
| 🟩 R3 | Opt-in daily notification (TMA push) — never auto-enabled |

**Step 4.2 — See skill improving (R11)**

| Priority | Task |
|----------|------|
| 🟨 R2 | Accuracy trend shown after session 5: "Твоя точность выросла на 12% за 10 игр" |
| 🟨 R2 | Running accuracy % displayed on profile/home |
| 🟩 R3 | Accuracy breakdown by sus range: "Ты лучше всего угадываешь высокоподозрительные" |

**Step 4.3 — Leaderboard**

| Priority | Task |
|----------|------|
| 🟨 R2 | Global leaderboard (top 20, daily reset) |
| 🟨 R2 | "Ты лучше 73% игроков сегодня" — cohort comparison, more motivating than rank |
| 🟩 R3 | All-time leaderboard |

---

### Activity 5: Contribute *(Release 3)*

**Step 5.1 — Disagree accuracy tracked**

| Priority | Task |
|----------|------|
| 🟩 R3 | Profile shows calibration score: "Точность несогласий: 78%" |
| 🟩 R3 | History of past disagreements with resolution status |

**Step 5.2 — See impact (R12)**

| Priority | Task |
|----------|------|
| 🟩 R3 | "Ты отметила эту вакансию за 3 дня до её удаления" — impact moment notification |
| 🟩 R3 | Weekly summary: "3 из твоих флагов подтвердились на этой неделе" |
| 🟩 R3 | Model update signal: "В марте точность модели на категории X выросла — спасибо за сигналы" |

**Step 5.3 — Recognition**

| Priority | Task |
|----------|------|
| 🟩 R3 | Shareable "Детективный паспорт": accuracy + sessions + confirmed flags |
| 🟩 R3 | Special rank title for high-accuracy disagreers: "Аналитик" |

---

## Release Slices

```
━━━━━━━━━━━━━━━━━━━━━━━━━━ RELEASE 1 (MVP) ━━━━━━━━━━━━━━━━━━━━━━━━━━
All 🟥 tasks above
Covers: Discover → Play → Share → basic Disagree
Goal: Viral loop working. D7 retention ≥ 30%.
Kill metric: D7 < 10% after 4 weeks → close project.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━━━━━━━ RELEASE 2 (Habit) ━━━━━━━━━━━━━━━━━━━━━━━━
All 🟨 tasks above
Covers: Infinite mode + Daily challenge + Accuracy tracking + Leaderboard
Gated on: R1 retention ≥ 30%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━━━━━━━ RELEASE 3 (Flywheel) ━━━━━━━━━━━━━━━━━━━━━
All 🟩 tasks above
Covers: Calibration score + Impact feedback + "Flagged before ban"
Gated on: Sufficient disagreement data volume (est. 1k+ disagreements)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Gaps Identified

| Gap | Implication |
|-----|-------------|
| Explanation copy is a product skill, not an engineering task — needs a dedicated writing pass | Must be done before R1 launch; can't be auto-generated |
| R12 data model (disagreement timestamps + moderation outcome linking) must be built in R1 even though the UI ships in R3 | Architecture decision, not just a feature |
| Face generation pipeline (v5 Flux + LoRA) must be complete before R1 — 543 faces needed | Dependency on running generation job |
| TMA `shareURL` API behaviour on iOS vs Android needs verification | Spike required before R1 |
| "Fresh faces" in infinite mode requires either the full 24k corpus faces (not yet generated) or a rotation strategy for the 543 test set | R2 scope decision |
