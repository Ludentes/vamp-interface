---
shaping: true
---

# Scam Guessr — Build Roadmap

**Date:** 2026-04-08
**Scope:** R1 MVP
**Repository:** telejobs (monorepo with separate app directory)
**Kill metric:** D7 retention < 10% after 4 weeks → close project

---

## Assets Ready (from vamp-interface)

| Asset | Status | Location |
|-------|--------|----------|
| 543 face portraits (v5, Flux + LoRA) | ✅ Done | `output/dataset_faces_flux_v5/` |
| 192px JPEG thumbnails | ✅ Done | `web/public/thumbs_v5/` (543 × ~7KB = 4.1 MB) |
| Game manifest (job_id → sus_level, sus_factors, x, y) | ✅ Done | `web/public/game_manifest.json` (349 KB) |

**Next:** Copy thumbs_v5/ and game_manifest.json → `telejobs/apps/scam-guessr/public/`

---

## Repo Structure (telejobs)

```
telejobs/
  frontend/                  ← existing telejobs UI (sunset, do not modify)
  apps/
    scam-guessr/             ← NEW: TMA app
      public/
        thumbs/              ← 543 × 192px JPEGs (copied from vamp-interface)
        game_manifest.json   ← job metadata for client-side reveal
      src/
        ...
      Dockerfile
      vite.config.ts
      package.json
  services/
    processing-service/
      app/api/routes/
        game.py              ← NEW: game endpoints
      app/models/
        game.py              ← NEW: game tables
      alembic/versions/
        017_create_game_tables.py  ← NEW migration
  pnpm-workspace.yaml        ← NEW: workspace root
  docker-compose.yml         ← add scam-guessr service
```

---

## Sprint Plan (ordered)

### Phase 0 — Scaffold (0.5 day)
- [ ] Add `pnpm-workspace.yaml` to telejobs root
- [ ] Create `apps/scam-guessr/` with Vite + React 19 + Tailwind v4 (copy frontend/ setup)
- [ ] Copy face thumbnails + game_manifest.json into `apps/scam-guessr/public/`
- [ ] Register new Telegram bot for Scam Guessr
- [ ] Add `scam-guessr` service to `docker-compose.yml`

### Phase 1 — Backend Infrastructure (E5-S1 + E4-S1)

**E5-S1: Session + Round API** — 2 days
- Alembic migration 017: `game_sessions`, `game_rounds`, `game_disagrees` tables
- `POST /api/v1/game/session` — stratified face selection (1 high-sus ≥80, 1 low-sus ≤20, 3 mid)
- `POST /api/v1/game/round` — record round with delta column
- TMA auth reuse: `telegram_miniapp.py` pattern (already exists)
- Guest UUID for web fallback

**E4-S1: Disagree schema + API** — 1 day
- `POST /api/v1/game/disagree`
- Timestamp required for R3 "flagged N days before removal"
- Runs in parallel once E5-S1 schema is merged

**E1-S2: Metadata API** — 0.5 day
- `GET /api/v1/game/faces/{job_id}` → `{sus_level, sus_factors[]}` 
- Served from game_manifest.json loaded at startup (no DB query)

### Phase 2 — Game Core Frontend (E2-S1 → S2 → S3)

**E2-S1: Face → Binary Guess → Reveal → Score** — 2 days
- P1 (Face Card): full-viewport face image, "Нормально" / "Мошенник" buttons, round progress
- P2 (Reveal Screen): true sus animates in, delta display, round score (`max(0, 1000 − |delta| × 10)`)
- Session state: 5 rounds, cumulative score, `POST /game/round` on advance
- `true_sus` from `game_manifest.json` (client-side, no API call on reveal)

**E2-S2: Slider (0–100)** — 1 day
- Replace binary buttons with `@radix-ui/react-slider` (already installed in telejobs frontend)
- Live value display, "Подтвердить" commit button
- Everything downstream of `commitGuess()` unchanged

**E2-S3: Reveal explanation** — 2 days (depends on copy being ready)
- Static template lookup: `sus_factor → explanation string[]`
- Large-delta template ("AI sees what I couldn't") vs small-delta ("you felt it right — because...")
- **Blocker:** 16 sus_factors × 2–3 templates = 32–48 copy units. Content task, not engineering.

### Phase 3 — Social Loop (E3-S1 → S2 + E4-S2)

**E3-S1: Session end screen + text share + exit hook** — 1.5 days
- Total score, rank title (5 levels), "Поделиться" button
- Exit hook: score gap (sessions 1–2), trajectory (sessions 3+), face silhouette teaser
- Telegram `shareURL` API for text share
- Deep link: `tg://resolve?domain=scamguessr_bot&startapp=ref_{score}`

**E4-S2: Disagree button UI** — 1 day
- "Я не согласна" on every reveal screen (not just high-delta)
- Bottom sheet: 4 reason options, auto-dismiss with "Записано" confirmation
- Fires `POST /game/disagree` in background (no await)

**E3-S2: Result card as designed image** — 2 days
- Canvas → JPEG blob: face thumbnail + score + rank title + punchy shareable line
- Shared as image via Telegram `shareURL` (1200×630px)

### Phase 4 — Polish (E3-S3 + optional)

**E3-S3: Competitive re-entry context** — 1 day
- Deep link carries `referrer_score` + `referrer_name`
- First face card shows: "Аня набрала 3 840 — побей её результат"

**E2-S4: Face morph on reveal** — 1.5 days (spike first)
- Spike (0.5 day): CSS cross-fade between neutral anchor + actual v5 face? Or keyframes?
- If CSS cross-fade: ship. If keyframes needed: defer to R2.

**E5-S3: Daily challenge endpoint** — 1 day (if time permits before launch)
- `GET /api/v1/game/daily?date=YYYY-MM-DD` — deterministic face selection, cached in Redis

---

## Total R1 Effort

| Phase | Stories | Effort |
|-------|---------|--------|
| 0: Scaffold | — | 0.5d |
| 1: Backend | E5-S1, E4-S1, E1-S2 | 3.5d |
| 2: Game core | E2-S1, S2, S3 | 5d |
| 3: Social | E3-S1, E4-S2, E3-S2 | 4.5d |
| 4: Polish | E3-S3, E2-S4 spike, E5-S3 | 3.5d |
| **Total** | | **~17 days** |

**Critical path:** Phase 0 → E5-S1 → E2-S1 → E2-S2 → E2-S3 → E3-S1 → E3-S2

**Parallelisable:**
- Phase 0 scaffold + E5-S1 backend can start simultaneously
- E4-S1 runs in parallel with E2-S1 once E5-S1 schema is done
- E1-S2 is trivially small (½ day), can slot in any gap

---

## Pre-Launch Blockers (non-engineering)

| Blocker | Owner | Needed before |
|---------|-------|--------------|
| Explanation copy: 16 sus_factors × 2–3 templates (32–48 units) | Кирилл / Sonya | E2-S3 |
| TMA `shareURL` behavior on iOS vs Android | Кирилл | E3-S1 |
| E2-S4 spike: CSS cross-fade feasibility | Кирилл | E2-S4 scope decision |
| New Telegram bot token for Scam Guessr | Кирилл | Phase 0 |

---

## Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Repo structure | Monorepo (telejobs), `apps/scam-guessr/` | Same DB, backend, infra. Cross-repo overhead wasteful for 1-dev experiment. |
| Backend | New router in processing-service | Pattern already established (18 routes). Zero new infra. |
| Frontend | New Vite app (not extending telejobs frontend) | Telejobs UI is desktop-first, sunset. TMA is mobile-only, full-viewport. |
| Face serving | Static files in `apps/scam-guessr/public/thumbs/` | 543 × 7KB = 4.1 MB. No MinIO needed. Immutable, served by nginx. |
| `true_sus` on reveal | Client-side from `game_manifest.json` | Loaded at session start. No per-reveal API call. 349 KB one-time load. |
| Auth | Reuse `telegram_miniapp.py` + guest UUID | Already implemented and tested. Zero new auth work. |
| Slider component | `@radix-ui/react-slider` | Already installed in telejobs frontend. |
| Difficulty scaling | Stratified session selection (not adaptive) | R1 scope. Adaptive difficulty is R2, gated on delta data from `game_rounds`. |

---

## R2 Scope (if D7 ≥ 30%)

- **The Map** — zoomable embedding space explorer (see `scam-guessr-map-shape.md`)
  - Semantic zoom: colored dots → face thumbnails → face detail
  - Cluster hulls + cluster summary panel
  - Sus band filter + RasterFairy grid view toggle
  - Post-session entry: "see where your 5 faces lived"
  - Requires: full corpus generation (24k) + convex hull + grid position pre-computation
- Infinite mode: 24k corpus faces (full generation run feeds The Map too)
- Daily challenge: `E5-S3` if not shipped in R1
- Score trajectory exit hook: visible from session 3
- Accuracy trend: "Your accuracy grew 12% over 10 games" (R11)
- Adaptive difficulty: shift face pool toward 35–65 sus when rolling accuracy > 70%
- Leaderboard: daily reset, cohort comparison

---

## R3 Scope (if D30 ≥ 20%)

- Calibration score: disagreement accuracy over time
- "Flagged N days before removal" — R12 data model already built in R1
- "Детективный паспорт": shareable impact card
