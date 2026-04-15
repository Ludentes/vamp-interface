---
shaping: true
---

# Shape: The Map — Embedding Space Explorer

**Date:** 2026-04-08
**Scope:** R2 feature, post-core-loop
**Depends on:** Full corpus generation (24k faces), game_manifest.json with x,y coords

---

## Problem

After playing, users have a verdict ("7 200 points, rank Детектив") but no context. They don't know where their five faces sit in the space, whether they were easy or hard picks, or what the overall landscape looks like. The corpus contains structural patterns — clusters of warehouse fraud, courier scams, remote-work phishing — that are invisible in the game loop.

The Map is the answer to: *"show me the bigger picture."*

It is also the only place in the product where the embedding geometry is directly visible — the core hypothesis (continuous face function, sus_level → uncanny drift) is provable here by looking at it.

---

## Appetite

2–3 weeks, shipping after R1 stabilizes. This is a standalone view, no changes to game backend.

---

## The Thing We're Building

A zoomable 2D canvas showing all corpus faces arranged by their PaCMAP embedding positions. Fraud score drives color at low zoom; actual face thumbnails appear as you zoom in.

Three zoom levels with qualitatively different content:

```
ZOOM OUT         ZOOM MID         ZOOM IN
(full corpus)    (clusters)       (face detail)

  ●●●●           🟥🟥 🟩🟩         👤👤👤
  ●●●●    →      cluster →         face faces
  ●●●            labels            metadata
```

This is semantic zoom — the content changes, not just its scale.

---

## Places

| # | Place | What it is |
|---|-------|------------|
| P1 | Map Canvas | The zoomable embedding space; main view |
| P2 | Cluster Panel | Slides in when a cluster is selected; cluster summary |
| P3 | Face Card | Slides in when a face is tapped; single face detail |
| P4 | Filter Rail | Horizontal strip above canvas; sus band filter + layout toggle |

Entry points into the Map:
- Standalone tab in app bottom nav (browse mode)
- After session end: "Посмотреть на карте" → Map opens with session faces highlighted

---

## Zoom Levels

### Level 0 — Corpus view (default on open)

All 24k faces as colored circles. Color = sus_level on `RdYlGn` scale reversed (red = fraud, green = safe). Cluster convex hulls drawn as faint outlines. Cluster labels visible at cluster center.

The user sees: a landscape. Red blobs at the edges/top, green mass in the center. The distribution of fraud in the job corpus.

No individual faces visible. Too many.

### Level 1 — Cluster view (zoom in to a region)

Colored circles become 32px face thumbnails. Cluster label pin still visible. User can now distinguish individual faces by their uncanniness.

The transition is the payoff: the red blob resolves into dozens of faces that actually look wrong.

### Level 2 — Detail view (zoom in on a face)

Selected face expands to 96px. Tap opens Face Card (P3). Neighboring faces still visible at 32px.

---

## Core Interaction Flow

```
Open Map
  → P1 at Zoom 0 (colored dots, cluster outlines)
  → Pan + zoom freely

User zooms into a red cluster
  → Zoom 1: thumbnails materialize
  → Taps a cluster hull / cluster label
    → P2 (Cluster Panel) slides in from right
      → shows: cluster name, mean sus, top sus_factors, 3 highest-sus faces
      → "Показать все 47 лиц" → filters Map to this cluster

User taps an individual face
  → P3 (Face Card) slides up from bottom
    → face image (large), sus_level, top sus_factors
    → "Сыграть с этим лицом" (R3 scope — fixed face game)
    → ← back closes P3

After a session game
  → Map opens with 5 session faces highlighted (ring + pulse animation)
  → Camera flies to frame all 5 faces
  → User explores where they were in the space
```

---

## Places × Affordances

### P1 — Map Canvas

| Affordance | Control | Description |
|------------|---------|-------------|
| Colored dot per face | render | Color = sus_level, RdYlGn reversed |
| Cluster convex hull | render | Faint stroke, fill color = mean sus of cluster |
| Cluster label | render | Appears at zoom 0-1, fades at zoom 2 |
| 32px face thumbnail | render | Appears at zoom 1, replaces dot |
| 96px face (selected) | render | Tap to select; zooms slightly |
| Session faces highlight | render | Pulse ring on the 5 faces from last game |
| Pinch to zoom | touch | Standard |
| Two-finger pan | touch | Standard |
| Camera fly-to | auto | Triggered by cluster tap or post-session entry |

### P2 — Cluster Panel

| Affordance | Control | Description |
|------------|---------|-------------|
| Cluster name | render | e.g. "Доставка — высокий sus" |
| Mean sus badge | render | Red/green badge with number |
| Sus factor tags | render | Top 3 sus_factors as chips |
| Preview strip | render | 3 highest-sus faces in cluster, 64px |
| "Показать все" button | tap | Filters map to this cluster, closes panel |
| Close / drag down | gesture | Returns to full corpus view |

### P3 — Face Card (bottom sheet)

| Affordance | Control | Description |
|------------|---------|-------------|
| Face image | render | 192px, full quality |
| Sus score | render | Large numeral, color-coded |
| Sus factor list | render | Top 3 factors with icons |
| Work type chip | render | e.g. "Доставка" |
| "← Назад" | tap | Closes card |

### P4 — Filter Rail

| Affordance | Control | Description |
|------------|---------|-------------|
| Sus band filter | tap chip | All / Safe (≤30) / Ambiguous (31–69) / Fraud (≥70) |
| Grid view toggle | toggle | Switches between PaCMAP scatter and RasterFairy grid |

---

## Semantic Zoom Implementation

deck.gl with two layers, swap on zoom threshold:

```
zoom < 9   → ScatterplotLayer (colored dots, radius=4px)
zoom 9–12  → IconLayer (32px thumbnails via /thumbs/{job_id}.jpg)
zoom > 12  → IconLayer (96px) + tooltip on hover
```

Cluster hulls: `PolygonLayer` (always visible, opacity fades with zoom).

---

## Grid View Toggle

Switching between PaCMAP positions and RasterFairy grid positions via the toggle:
- Spring animation on face positions (smooth transition, ~500ms)
- Grid mode: every face visible, no overlap, neighborhood approximately preserved
- Useful for gallery browse — "just show me all the scam faces"

RasterFairy grid positions pre-computed at build time alongside PaCMAP. Stored in game_manifest.json as `gx, gy` alongside `x, y`.

---

## Filter Behavior

Sus band filter: hides non-matching faces from both ScatterplotLayer and IconLayer. Cluster hulls recalculate to show only visible members (or hide clusters with 0 visible members).

When "Fraud (≥70)" is selected and user is at zoom 0: the green mass disappears, only red dots remain. The spatial distribution of fraud becomes immediately legible.

---

## Post-Session Entry

When the Map is opened from the session end screen ("Посмотреть на карте"):

1. Camera starts at zoom 0 (full corpus)
2. Session faces pulse (ring animation, distinct color from fraud color scale)
3. After 1 second: camera flies to frame all 5 session faces (zoom 1)
4. User can zoom and pan freely from there
5. Session face ring persists while in this mode; a "× Сбросить" chip clears it

This answers the question: "were these faces unusual picks, or are they from a well-known cluster?"

---

## Rabbit Holes

**Don't build:**
- Real-time re-projection (re-run PaCMAP on selection). Compute-heavy backend, destroys session state. Offline build step only.
- Force-directed spreading to eliminate overlap. Destroys PaCMAP geometry — the whole point is that position = semantic similarity.
- Temporal animation (faces appearing over time). Corpus is static offline generation. YAGNI.
- Fisheye distortion lens. Cool but distorts semantic distances. Add only if the overlap problem is severe and the grid toggle doesn't solve it.
- Search by text / job content. Map is visual exploration, not search. That's a different product.

**Scope boundary:**
- "Сыграть с этим лицом" (play with a specific face from the Map) is R3 — it requires a fixed-face game mode in the backend that doesn't exist yet.
- Cluster naming (e.g. "Telegram courier scam") is a content task — needs human review. Ship with auto-labels (work_type + sus band) in R2.

---

## Build Sequence

| Step | What | When |
|------|------|------|
| 1 | Pre-compute convex hulls (Python, scipy) + RasterFairy grid positions → bake into manifest | Before frontend |
| 2 | deck.gl canvas + ScatterplotLayer at zoom 0 | Sprint 1 |
| 3 | Zoom threshold → IconLayer switch (thumbnails at zoom 1) | Sprint 1 |
| 4 | Cluster panel (P2) + hull tap | Sprint 2 |
| 5 | Face card (P3) | Sprint 2 |
| 6 | Filter rail + grid toggle | Sprint 2 |
| 7 | Post-session entry with highlighted faces | Sprint 3 |

---

## What Makes This Worth Building

The game is 5 rounds. The Map is unlimited. It's the place where the core hypothesis becomes visually obvious — high-sus faces cluster spatially, and their cluster looks different from low-sus faces in a way that is legible even before individual faces are readable.

For the scam hunter: the first time they zoom from a red blob into a cloud of faces that genuinely look wrong is the product's real reveal. The game teaches intuition; the Map proves it is real.
