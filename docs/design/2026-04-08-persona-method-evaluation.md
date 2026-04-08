# Persona × Method Evaluation

**Date:** 2026-04-08  
**Input:** `docs/research/2026-04-08-semantic-viz-patterns.md` (9 methods)  
**Personas:** `docs/design/scenarios.md` (Scam Hunter, Analyst, Student)  
**Purpose:** Decide which methods to build, in what order, for whom.

---

## Scoring key

**5** — Directly solves stated need, no friction  
**4** — Strong fit, minor gaps  
**3** — Useful but not primary  
**2** — Works if combined with something else  
**1** — Misfit or actively harmful for this persona  

---

## Method × Persona Matrix

| Method | Scam Hunter | Analyst | Student | Notes |
|--------|:-----------:|:-------:|:-------:|-------|
| Semantic zoom (dots → faces by zoom level) | **5** | **5** | **2** | Student won't zoom; Hunter/Analyst love the overview |
| HDBSCAN cluster hulls + fly-to | **5** | **5** | **3** | Cluster labels help Student only if labels are plain-language |
| Density contours (fraud heat map) | **4** | **5** | **1** | Too abstract for Student; essential for Analyst |
| RasterFairy grid toggle | **4** | **1** | **5** | Analyst: destroys geometry. Student: native mental model |
| Sidebar cluster summary | **3** | **5** | **1** | Analyst's primary panel; Hunter wants faces not stats |
| Scatter/Gather (lasso → sub-layout) | **4** | **5** | **1** | Analyst/Hunter power feature; Student won't use |
| Reveal game (current card-flip) | **2** | **1** | **5** | Student's entire product; others find it too slow |
| Fisheye lens | **3** | **4** | **1** | Niche power-user tool; Student interaction model incompatible |
| Lasso → play selection | **5** | **3** | **2** | Hunter's killer feature; already partially built |

---

## Per-persona breakdown

### Scam Hunter

**Primary need:** Process 200 postings in time it takes to read 20. Spot coordinated rings. Trust gut when score doesn't flag.

The Hunter's session is spatial — they scan, not read. They need:
1. A fraud geography overview where hot zones are visible before any interaction
2. Fast cluster navigation (click a suspicious region → faces appear immediately)
3. A mechanism to act on what they spot (lasso → investigate → flag)

**What works:**  
Semantic zoom with cluster hulls is the core. At full zoom-out, colored dots + cluster hulls colored by mean sus_level show the fraud landscape in one glance. One click flies to a cluster. Faces appear as they zoom in. Density contours add a "fraud heat map" layer that makes hot zones pop even at dot scale. Lasso → play selection is the "act" step — select a suspicious pocket, run it through the reveal game, get confirmation or refutation.

**What doesn't work:**  
The sidebar with stats slows them down — they are scanning visually, not reading numbers. The reveal game as primary interface is too slow (one card at a time vs. scanning a 10×20 grid). Fisheye is secondary at best.

**Their optimal stack:**  
`dot view + fraud color + cluster hulls (entry) → fly-to zoom → faces → lasso selection → play selection`

---

### Analyst

**Primary need:** See where the pipeline is wrong. Find clusters with no vocabulary in the current factor set. Batch-label new golden examples.

The Analyst's session is structural — they are debugging the model, not the postings. They need:
1. Embedding geometry preserved at all zoom levels (position = semantic distance)
2. Factor-derived affect visible as a second channel layered on identity
3. Discordance visible — calm face in a red cluster = model blind spot
4. A way to select a cluster and do something with it (batch label, re-project)

**What works:**  
Every geometric method is high value. Semantic zoom preserves the layout while adding detail on demand. Density contours are essential — sparse regions = novel patterns = model gaps. The sidebar cluster summary is their primary tool: factor distribution per cluster, internal sus variance (high variance = model inconsistency), top suspect faces. Scatter/Gather is their core investigative workflow: select a cluster → re-project its internal structure → find the sub-pattern the model missed. Fisheye is useful for examining a dense cluster's internal variation without fully zooming in.

**What doesn't work:**  
RasterFairy grid destroys the embedding geometry, which is the primary signal for the Analyst. The reveal game is useless for this workflow. Any view that collapses cluster membership into a flat list loses the spatial information they need.

**Their optimal stack:**  
`dot view + multiple color channels (sus, source, work_type) → cluster hulls → sidebar cluster panel → scatter/gather sub-projection → fisheye for dense region inspection`

---

### Student (Browser)

**Primary need:** Know fast if a posting is safe. Make browsing less boring. Brag.

The Student does not think in clusters or embeddings. They think in job categories and vibes. Their session is intuitive — they respond to faces, not data. They need:
1. Faces immediately, not after learning to zoom
2. No calibration required — the signal must be pre-attentive
3. Something shareable

**What works:**  
The reveal game (card-flip) is their entire product — it is already well-scoped, shareable, and requires zero calibration. After the game, the RasterFairy grid filtered by work type is the natural continuation: "show me all warehouse jobs as faces." Every face visible, no overlap, they can visually scan for sketchy ones. Cluster hulls help if labels are plain-language work-type names ("Courier jobs", "Office work") — they function as a category filter. The lasso-to-play-selection is marginally interesting if presented as "play with jobs like these."

**What doesn't work:**  
Dots mean nothing. Density contours mean nothing. The sidebar with factor distributions means nothing. Semantic zoom requires intent to zoom. Fisheye requires knowing what you're looking for. Everything requiring "understanding the encoding" fails this user.

**Their optimal stack:**  
`reveal game (entry) → RasterFairy grid filtered by work type → click face → job text + score → share`

---

## Key insight: three entry points, not one

No single view serves all three users. The methods partition cleanly by persona:

```
Student     → Reveal game → Grid (RasterFairy)
Scam Hunter → Dot map → Cluster hulls → Faces → Lasso
Analyst     → Dot map → Cluster sidebar → Scatter/Gather → Fisheye
```

The Scam Hunter and Analyst share an entry point (the semantic zoom map) but diverge at the interaction layer. The Student is entirely separate — they should never see the map as a primary view.

This suggests a **persona-branching landing** rather than one unified product: after the reveal game, route to "browse jobs by type" (Student) or "explore the full corpus" (Hunter/Analyst). The two corpus views can share the same map implementation but expose different toolbars and default states.

---

## Build priority

Given the Student is already served by the reveal game and the grid is one Python script + toggle away, the biggest gap is the **Scam Hunter / Analyst corpus view**. That view is currently a slow DOM renderer with no semantic zoom, no cluster structure, and no fraud geography overview.

### Priority 1 — Unlocks both Hunter and Analyst
- Semantic zoom: dots → faces (deck.gl `ScatterplotLayer` + `IconLayer`, zoom threshold)
- HDBSCAN cluster hulls colored by mean sus_level (pre-computed convex hulls, deck.gl `PolygonLayer`)
- Fraud color encoding at dot level (visible from full zoom-out)

### Priority 2 — Differentiates Hunter from Analyst
- Lasso → play selection (Hunter) — already partially built, just needs deck.gl picking
- Sidebar cluster summary with sus distribution (Analyst)

### Priority 3 — Power features
- Density contours overlay (Analyst)
- RasterFairy grid toggle + spring animation (Student post-game)
- Scatter/Gather sub-projection (Analyst)
- Fisheye lens (Analyst, niche)

### Skip
- Force-directed overlap resolution (destroys geometry)
- Gamification mechanics (the faces are the hook)
- Temporal animation (YAGNI)

---

## On the overlap problem specifically

The research reveals that **the overlap problem is not solved by any single technique** — it is managed through the combination of:

1. **Semantic zoom** (don't show faces until you're zoomed in enough for them to be legible)
2. **Cluster navigation** (fly to a cluster → faces are now in a smaller viewport, less overlap)
3. **RasterFairy grid** (escape hatch for "I want to see all faces with no overlap")
4. **Filtering** (sus range + preset reduce the count before rendering)

The current approach of showing all 543 faces at full zoom as 96px circles is the wrong default. The right default is dots at full zoom with faces only appearing at cluster zoom level. This alone solves most of the overlap complaint.
