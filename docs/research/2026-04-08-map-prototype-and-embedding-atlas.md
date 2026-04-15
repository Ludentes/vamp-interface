# Map Prototype: Current State and Embedding Atlas Direction

**Date:** 2026-04-08
**Context:** vamp-interface prototype, potential R2 path for Scam Guessr Map feature
**Prototype repo:** `/home/newub/w/vamp-interface/`
**Game repo:** `/home/newub/w/telejobs/` (branch `feature/scam-guessr`)

---

## 1. What the Prototype Does Today

The `/home/newub/w/vamp-interface/web/` app (`/home/newub/w/vamp-interface/web/src/views/MapView.tsx`) is a working interactive map of the 543-face test corpus. It is a canvas built in React without WebGL — all faces are absolutely-positioned DOM elements, rendered as circular image thumbnails, repositioned via CSS transforms on pan/zoom.

### Data model

```typescript
interface Point {
  id: string     // job_id
  x: number      // PaCMAP coordinate [0,1]
  y: number      // PaCMAP coordinate [0,1]
  sus: number    // sus_level 0–100
  cohort: string // work_type × sus_band label
  text: string   // job posting (first 2000 chars)
  face: string   // path to full-size portrait (192px PNG)
  thumb: string  // path to thumbnail (96px JPEG)
}
```

All 543 points are loaded from `/home/newub/w/vamp-interface/web/public/layout.json` at startup (one file, ~7 MB with text). PaCMAP coordinates were computed offline; the UI does not re-project.

### Rendering

No canvas, no WebGL. Each face is a `<div>` with:
- `position: absolute`, `left/top` computed from `(x, y)` → screen pixel via `toScreen()`
- `border-radius: 50%` (circular crop)
- `border: 2px solid` colored by sus_level (green/yellow/orange/red)
- `<img>` inside, `loading="lazy"`, 96px

Culling is done before render: points outside the visible viewport (`sx < -48 || sx > w+48`) are excluded from the React render list. This keeps DOM node count manageable at default zoom.

Thumbnail size scales with zoom: `thumbSize = clamp(THUMB × scale, 48, 192)` — so at 2× zoom, thumbnails are 192px.

### Navigation

| Gesture | Effect |
|---------|--------|
| Mouse wheel | Zoom in/out (anchor at cursor position) |
| Left drag (pan mode) | Pan the canvas |
| Left drag (lasso mode) | Draw selection rectangle |
| Click a face | Open detail panel |
| Click empty space | Deselect |
| `S` key | Toggle pan / lasso mode |
| `Esc` | Clear selection / lasso |

Zoom range: 0.3× – 10×. At default (1×) the 543 faces overlap heavily in dense clusters. At 3–4× individual faces separate clearly.

### Filter controls

Four controls in the header strip:

1. **Preset chips** — `All` / `Legit` (sus < 50) / `Scam` (sus ≥ 50)
2. **Sus range slider** — dual-handle, 0–100; hides faces outside the range
3. **Count badge** — "347 shown" after filtering
4. **Lasso / Select toggle** — switches between pan and selection mode

All filters work client-side against the pre-loaded `points` array; no server requests.

### Detail panel (side panel)

Clicking a face opens a right-side panel (272px wide):
- Full portrait image, `aspect-[3/4]`
- Sus score over a gradient overlay colored by sus band
- Cohort label (e.g. "доставка scam")
- Full job posting text, scrollable
- Coordinates: `x 0.412  y 0.739  sus 87`

### Lasso → Play selection

Dragging in lasso mode draws a white rectangle. On mouse-up, any face whose screen center falls inside the rectangle is highlighted (dimming all others). A "Play N selected →" button appears in the header. Clicking it sends those N faces to the game's reveal loop as a custom subset deck — i.e. the game plays exactly those faces, in the map's spatial order.

This is the core "analyst" feature: select a suspicious cluster from the map → play just those faces to test your intuition on a homogeneous group.

### Axis labels

Faint axis labels on the canvas edges reflect the PaCMAP semantic axes (from the PC-to-axis mapping in the build step):

- X: `← fraud language` / `physical labor →`
- Y: `remote / light ↑` / `↓ heavy / skilled`

These are static text, not computed dynamically.

---

## 2. Current Limitations

| Limitation | Impact |
|------------|--------|
| DOM-based rendering (no WebGL) | Smooth up to ~600 faces. At 5k+ faces, React re-render on every pan event becomes slow. |
| No touch support | Pan and zoom via mouse only; no pinch-to-zoom on mobile. |
| Full data in one JSON | 7 MB load at startup. Works for 543 faces; not for 24k (would be ~300 MB). |
| No cluster structure | No cluster labels, no convex hulls, no "jump to cluster" navigation. |
| No semantic zoom | All faces render at the same level of detail (same DOM element regardless of zoom). |
| Face images served from nested git repo | `/home/newub/w/vamp-interface/web/` has its own `.git`. Thumbs can't be committed to the parent vamp-interface repo. In Scam Guessr, this is solved by copying into `/home/newub/w/telejobs/apps/scam-guessr/public/`. |
| No mobile entry point | The web app is desktop-only, launched with `pnpm dev` from `/home/newub/w/vamp-interface/web/`. Not a TMA. |

---

## 3. Embedding Atlas — What It Is

Apple's **Embedding Atlas** (`apple/embedding-atlas`, MIT license, released 2025) is a Python + JavaScript tool for interactive 2D embedding visualization. It runs as a local web server and serves a WebGPU-accelerated UI.

### What it does

- Accepts a Parquet or Arrow file with `x`, `y` columns and any metadata columns
- Renders points using **WebGPU** (WebGL 2 fallback) — handles millions of points at 60fps
- Computes **kernel density estimation** (KDE) on-the-fly and renders density contours as a background layer
- Clusters the data automatically and shows cluster labels
- Nearest-neighbor search: click a point → "Find similar"
- Multi-column crossfilter: select by one metadata field, see how others change
- Color by any metadata column

### How we already use it

The `/home/newub/w/vamp-interface/docs/runbooks/embedding-viz.md` runbook describes the current use of Embedding Atlas against the full 23k-job corpus:

```bash
# Run from /home/newub/w/vamp-interface/
uv run embedding-atlas output/full_layout.parquet \
  --x x --y y --text text \
  --disable-projection \
  --host 0.0.0.0 --port 8765
```

`/home/newub/w/vamp-interface/output/full_layout.parquet` was built by `/home/newub/w/vamp-interface/src/build_full_layout.py` (PaCMAP on all 23k embeddings, pre-cutoff optional). This is a research/analyst tool — not the game UI.

Current capabilities in use:

| Goal | How in Embedding Atlas |
|------|----------------------|
| See fraud geography | Color by `sus_level` |
| Find coordinated channels | Color by `source_name` |
| Validate work-type clusters | Color by `work_type` |
| Find contact-linked jobs | Color by `contact_telegram` |
| Read a posting | Click any point → tooltip shows `text` |
| Find dense fraud regions | Enable density contours |
| Find nearest neighbours | Click point → "Find similar" |

### What it does not do (yet)

- **No face thumbnail rendering.** Embedding Atlas shows metadata on hover (text field), not images. There is no native `<img>` support in the point renderer.
- **No lasso → play game.** It is a research viewer, not a game UI entry point.
- **No mobile.** Desktop web only.
- **No custom affordances** (cluster panels, filter rails, sus-band chips). It is a general-purpose tool.

---

## 4. Embedding Atlas as a Future Direction

### Why it is relevant

The Map feature described in `/home/newub/w/vamp-interface/docs/shaping/scam-guessr-map-shape.md` targets 24k faces. The prototype's DOM renderer will not scale. The research doc (`/home/newub/w/vamp-interface/docs/research/2026-04-07-embedding-visualization.md`) recommends **deck.gl** as the production rendering substrate (WebGL, GPU-instanced, handles millions of points).

Embedding Atlas is an alternative path: instead of building a custom deck.gl component, extend Embedding Atlas to support face thumbnails, then embed or wrap it in the Scam Guessr TMA.

### What "extending Embedding Atlas" would look like

The Embedding Atlas npm package (`@apple/embedding-atlas`) exports React components. The `EmbeddingView` component is the core scatter renderer. The package uses Mosaic-architecture crossfilter (`@uwdata/mosaic-core`).

A thumbnail overlay would require:
1. A parallel `IconLayer`-style renderer sitting above the WebGPU layer (deck.gl `IconLayer` or plain DOM absolutely positioned), toggled on at a zoom threshold
2. The crossfilter selection state exposed so the thumbnail layer knows which points are visible

This is non-trivial — the zoom state and coordinate transforms are internal to the WebGPU renderer. It would require either forking Embedding Atlas or using their public event API (if one exists; not documented as of April 2026).

**Realistic effort:** 1–2 weeks to build a thumbnail overlay, 1 week to productionize (mobile-friendly, TMA-embedded, sus-band filter). Total ~3 weeks, same as the custom deck.gl approach.

### Recommendation

| Path | Pros | Cons |
|------|------|------|
| **Extend Embedding Atlas** | Free density contours, KDE, cluster labels, nearest-neighbor search | No image support out of the box; internal APIs not stable; desktop-first |
| **Custom deck.gl** (per the shape doc) | Full control; WebGL thumbnail rendering is first-class; mobile-friendly from the start | Build cluster labels and density contours manually |
| **Use Embedding Atlas as analyst tool, deck.gl for game Map** | Best tool for each audience | Two separate UIs to maintain |

**Recommended split:**
- Keep Embedding Atlas as the **analyst / research** interface. It is already working for the full 23k corpus. Run it locally when investigating fraud clusters, building game content, or validating generation quality.
- Build the **game-integrated Map** with deck.gl, following the shape in `/home/newub/w/vamp-interface/docs/shaping/scam-guessr-map-shape.md`. The game Map needs mobile-first touch, TMA integration, post-session face highlights, and sus-band filtering — none of which Embedding Atlas provides.

These are different tools for different users. The analyst tool is already done. The game Map is an R2 build.

---

## 5. Prototype → Production Delta

What the current prototype proves and what still needs to be built:

| Capability | Prototype (`/home/newub/w/vamp-interface/web/`) | Production game Map (`/home/newub/w/telejobs/apps/scam-guessr/`) |
|------------|-------------------|---------------------|
| PaCMAP layout | ✅ Working | ✅ Same coordinates |
| Face thumbnails on map | ✅ DOM-based | ⬜ deck.gl IconLayer |
| Pan + zoom | ✅ Mouse only | ⬜ Touch + mouse |
| Sus-band filter | ✅ Preset chips + range slider | ✅ Reuse pattern |
| Lasso → play game | ✅ Works | ⬜ Adapt for TMA |
| Detail panel | ✅ Full text + sus | ⬜ Adapt (no full text, add sus_factors) |
| Cluster labels / hulls | ⬜ Not implemented | ⬜ Build |
| Semantic zoom (dots → faces) | ⬜ Not implemented | ⬜ Build |
| Grid view (RasterFairy) | ⬜ Not implemented | ⬜ Optional |
| Post-session highlight | ⬜ Not implemented | ⬜ Build |
| Mobile / TMA | ⬜ Desktop only | ⬜ Build |
| 24k faces | ⬜ DOM won't scale | ⬜ WebGL required |
| Axis labels | ✅ Static text | ✅ Reuse |

The prototype validates the core UX: filtering + lasso selection + detail panel work well. The rendering substrate (DOM → WebGL) and the corpus scale (543 → 24k) are the main gaps.
