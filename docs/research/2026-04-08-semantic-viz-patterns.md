# Semantic Visualization UX Patterns for High-Density Image Layouts

**Research question:** What are the best UX patterns, tools, and techniques for navigating high-density 2D semantic layouts where items are represented as image thumbnails (portrait photos)?

**Context:** 543 AI-generated portrait faces on a PaCMAP 2D layout, 96px thumbnails, fraud signal encoded in face expression (uncanny valley), ~40 HDBSCAN clusters, primary variable is continuous fraud score.

**Researched:** 2026-04-08

---

## 1. Executive Summary

Several mature techniques directly address the density-vs-context tradeoff in 2D embedding spaces with image data. The strongest candidates for this project, ranked by fit:

1. **Semantic zoom with LOD switching** — the foundational pattern. At low zoom, show colored dots (fraud score → color). At medium zoom, show small faces. At high zoom, show full faces with cluster labels. This is what PixPlot, Nomic Atlas, and Embedding Atlas all do. Implementation: deck.gl `ScatterplotLayer` + `IconLayer` with zoom-threshold switching.

2. **WebGL-rendered scatter with GPU-accelerated thumbnails** — the rendering substrate. deck.gl or Three.js with instanced rendering handles 543 faces trivially (tools handle millions of points). This is non-negotiable for smooth 60fps pan/zoom.

3. **Cluster overview → drill-down** — the primary navigation pattern. Show HDBSCAN clusters as labeled regions at zoom-out; clicking a cluster zooms and reframes to show all members. This is how Nomic Atlas structures navigation.

4. **Fisheye / focus+context lens** — a secondary interaction for dense regions. D3's fisheye plugin (`d3-plugins/fisheye`) lets users magnify a local region while keeping global layout visible. High impact for the analyst user type.

5. **RasterFairy grid view** — an alternative layout mode for "gallery browse" use cases where users want no-overlap thumbnail grids while still preserving approximate neighborhood structure. PixPlot ships this as a toggle.

Gamified EDA exists as a concept (points, leaderboards for annotation tasks) but has no established library and is low-ROI for this specific use case. Force-directed spreading for images is computationally expensive and destroys PaCMAP geometry; it is not recommended here.

The concrete implementation path with lowest effort and highest quality: **deck.gl + React**, using `ScatterplotLayer` for points-mode and `IconLayer` for face-mode, with zoom-threshold switching. Estimated 2-3 days to working prototype.

---

## 2. Semantic Zoom

### What It Is

Semantic zoom is a visualization paradigm in which the amount and *kind* of information presented changes qualitatively as users change scale — not just geometrically. Unlike geometric zoom (which simply scales all objects uniformly), semantic zoom changes object shape, information density, and sometimes presence/absence of elements based on zoom level.

The key definition from InfoVis:Wiki: "Working on the intrinsic structure of data and incorporating knowledge about its meaning (metadata), semantic zoom adjusts the *contents and density* of information that is shown instead of only changing visual detail and scale." It is a form of "details on demand" — you see what you need at each scale.

### The Classic Progression

A canonical example: a city map at low zoom shows dots; at medium zoom, dots become labeled icons; at high zoom, full address + hours + photos appear. For our case the natural progression is:

- **Zoom level 0 (full corpus view):** Each job posting is a colored circle, color = fraud score. Cluster outlines or convex hulls visible. User sees the overall shape of the fraud landscape.
- **Zoom level 1 (cluster view):** Circles grow into small face thumbnails (32–48px). Cluster label appears in center-of-mass. Users can scan which clusters skew fraudulent.
- **Zoom level 2 (detail view):** Full 96px faces. Hover shows job title, sus_level score, top sus_factors. Click opens full record.

### Implementations

The Gosling genomics framework (`gosling-lang.org`) ships semantic zoom as a first-class API — `visibility` rules tied to zoom level. The pattern is also documented extensively in the Vienna CVAST group's work on time-series data (SemTimeZoom). For image data specifically, every major tool described in Section 4 implements semantic zoom implicitly through their LOD (level of detail) switching.

**For our stack:** semantic zoom is easily implemented in deck.gl by maintaining a `zoom` state variable from the viewport and switching which layer is visible/active at threshold zoom values. No external library needed.

---

## 3. Scatter/Gather

### The InfoVis Pattern

Scatter/Gather was introduced by Marti Hearst, David Karger, and Jan Pedersen at AAAI 1995 as a technique for navigating large document collections. The name describes the cycle: **scatter** (cluster the collection into N groups), then **gather** (select promising groups, merge them into a new subcollection), then scatter again on the subcollection.

The workflow: the system clusters 250 documents into 5 groups and summarizes each group with topical terms. Users identify 1-2 promising clusters, gather those documents (~80 items), scatter again into 5 new sub-clusters, repeat until they have a workable collection of highly relevant items. Hearst's studies showed "relevant documents tend to clump into a few clusters," so 2-3 iterations typically isolate what users need.

### Relevance to Our Case

Scatter/Gather is originally a text-clustering interface, but the conceptual model maps well to our fraud investigation use case, particularly for the **scam hunter** user:

- "Scatter" = HDBSCAN clusters already precomputed on PaCMAP layout
- "Gather" = lasso or click to select a suspicious cluster
- Re-scatter = filter the selected cluster and optionally re-run PaCMAP on just those points to see their internal structure

No existing library ships "Scatter/Gather" as a named widget. The pattern must be implemented as a state machine: selection → filtered subview → re-layout option. The Nomic Atlas Python client (`nomic-ai/nomic` on GitHub) supports programmatic subset selection that can feed back into a new Atlas projection, which is the closest off-the-shelf implementation of this pattern.

---

## 4. PixPlot and Similar Image-Grid Tools

### PixPlot (Yale DHLab)

PixPlot (GitHub: `pleonard212/pix-plot`) is the closest existing tool to what this project needs. It visualizes tens of thousands of images in a 2D projection where similar images cluster together. Key technical decisions:

- **Rendering:** Custom WebGL viewer (not a library — custom shader code). Images are packed into texture atlases and rendered as instanced quads. The parameter `cell_size` controls texture atlas packing; smaller values reduce GPU RAM at the cost of visual fidelity.
- **Layout:** UMAP (2,048-d → 2D), with HDBSCAN for cluster hotspot detection (KMeans fallback). Hotspots are shown as named waypoints users can jump to directly.
- **Density handling:** At full zoom-out, individual images are not distinguishable — the visualization shows the *shape* of the layout (density cloud). As users zoom in, individual thumbnails become legible. This is an emergent semantic zoom effect, not explicitly programmed — it falls out naturally from image size at different zoom levels.
- **Grid mode (RasterFairy):** PixPlot optionally applies RasterFairy to remap the UMAP point cloud into a regular grid while preserving neighborhood structure. This is a "no overlap guaranteed" view where every thumbnail is visible simultaneously. Used as an alternative layout toggle, not the primary view.
- **Navigation:** Hotspot list (cluster waypoints) + free pan/zoom + optional date/metadata-based layout. No fisheye or lasso selection.

PixPlot handles 100k images. For 543 images, the rendering overhead is negligible — any WebGL approach will be fast.

**Verdict for our project:** PixPlot's architecture (WebGL texture atlas + UMAP layout + HDBSCAN hotspots + optional grid view) is the right model to follow. We should not use PixPlot itself (it embeds its own feature extraction pipeline), but replicate its viewer pattern.

### Apple Embedding Atlas

Released 2025, open source (`apple/embedding-atlas` on GitHub). Uses WebGPU (WebGL 2 fallback). Key techniques:
- Kernel density estimation + density contours to distinguish dense vs sparse regions
- Order-independent transparency for overlapping points
- Automatic data clustering + labeling for cluster navigation
- Real-time search + nearest neighbors
- Multi-coordinated views for metadata cross-filtering

Embedding Atlas is primarily designed for text/numeric embeddings, not image thumbnails. The Python package supports CLI, Jupyter widget, and Streamlit component modes. It does not have native image thumbnail rendering in its current documentation, but the `EmbeddingViewMosaic` component (part of the npm package) suggests a Mosaic-architecture crossfilter integration.

**Verdict:** Excellent for the analyst user who wants crossfilter + density contours + nearest-neighbor search. Not directly usable as-is for face thumbnails, but the techniques (density contours, KDE) are worth adding as overlays on top of our deck.gl implementation.

### Renumics Spotlight

Python tool (`renumics-spotlight` on PyPI, `Renumics/spotlight` on GitHub) for exploring unstructured ML datasets. Provides a Similarity Map (UMAP/t-SNE), supports images, audio, video. Key pattern: display image thumbnails in a 2D embedding space, with coordinated histogram/metadata filters in side panels. Used heavily for ML dataset debugging.

Spotlight is closer to a desktop data exploration tool than a web visualization. It runs as a local server. Its similarity map does show image thumbnails at zoom-in — the density problem is handled by just accepting overlap at low zoom and relying on zoom to separate.

**Verdict:** Good reference for the coordinated-views pattern. Not suitable as our primary renderer.

### Nomic Atlas

Web-based (`atlas.nomic.ai`). Projects data into 2D, automatically labels regions with topic text, supports 100M points. Key navigation patterns:
- Colored density regions (topics shown as color blobs with text labels)
- Zoom-in reveals individual points; zoom-out shows topic structure
- Hover tooltips on individual points
- Lasso selection for subset creation
- The Python client (`nomic-ai/nomic`) can push datasets and retrieve selections programmatically

Nomic Atlas supports image embeddings via `nomic-embed-vision-v1.5` (same latent space as text embeddings). The Atlas data map UI is the best existing example of the "cluster overview → individual item" navigation pattern at scale.

**Verdict:** Strong reference UX. The Nomic atlas UI pattern (topic color regions + zoom-to-reveal-individuals) is directly applicable. For 543 faces, could actually use Nomic Atlas directly as a hosted solution — upload faces as image embeddings, get the map for free. Worth evaluating before building custom.

---

## 5. Force-Directed Spreading

Force-directed layout spreads nodes using repulsion forces to prevent overlap. Several libraries support this:

- **D3 force simulation** (`d3-force`): `forceCollide` adds a collision constraint treating each node as a circle. Can be used to spread image thumbnails by setting collision radius = thumbnail_size/2. Works well for small datasets (<1000 nodes), but destroys the PaCMAP geometry — nodes drift significantly from their original embedding positions.
- **cola.js** (Constraint-based Layout): Supports non-overlap constraints more precisely than D3, but same geometry-destruction problem.
- **Cytoscape.js** with `cytoscape-fcose` layout: Similar approach, same tradeoff.
- **boundaryLayout** (Cytoscape app): Boundary-constrained force layout that keeps nodes within custom hull shapes.

**The fundamental problem** with force-directed spreading for our use case: the PaCMAP positions encode semantic meaning (similar jobs → similar positions). Moving faces to eliminate overlap breaks the core property that makes the visualization work. If two fraudulent job postings are positioned close together, that *is* the signal — their faces looking similar confirms cluster membership. Force-spreading would destroy this.

**The only legitimate use case** for force-spreading here: as a temporary "fan out" animation when a user clicks on a dense cluster to inspect its members. Spread the cluster's faces within a bounded region to make them all visible, then spring back to PaCMAP positions when dismissed. This is a UX flourish, not a layout algorithm.

**RasterFairy** (GitHub: `Quasimondo/RasterFairy`, `pip install rasterfairy`) is a better alternative for the "I need to see all faces without overlap" use case. It maps the 2D point cloud to a regular grid via optimal transport / assignment, preserving neighborhood structure far better than force-directed approaches. PixPlot uses it as a layout toggle. The Python 3 compatible fork is `pechyonkin/RasterFairy-Py3`.

---

## 6. Gamified EDA

Gamified EDA as a formal field does not exist with established tooling. What does exist:

- **Gamification of data annotation** (labeling games): Dizeez (gene-disease linking game from Scripps), Fold.it (protein folding). These are citizen science games that extract knowledge through play. The mechanics are: game challenge → user input → aggregation of inputs → scientific insight.
- **Gamification of dashboards** (Microsoft research, 2020+): Points, leaderboards, achievement badges applied to BI tools to increase analyst engagement. Studies show modest engagement increases but no significant insight quality improvement.
- **Explorable Explanations** (Bret Victor tradition): Interactive essays where the reader experiments with parameters to build intuition. These are narratively guided, not open-ended EDA.

For our specific use case, the "wow" factor for the casual browser user (our third user type) is closer to an **interactive art installation** than a gamified dashboard. The faces themselves are the engagement hook. The interaction that makes sense is: hover a face → it animates or glows → the fraud score slides into view. That is aesthetically-driven discovery, not gamification in the formal sense.

**Closest practical implementation:** treat the layout as an interactive poster. Add a "most suspicious" highlight mode (faces ranked by sus_level, animated attention pulse on the top 10%). Add a "cluster story" mode where clicking a cluster plays a short narrative about what type of fraud it represents. These patterns exist in data journalism tools (Flourish, Scrollama) rather than gamification libraries.

**Verdict:** No library to install. Implement custom micro-interactions using CSS animations + Framer Motion.

---

## 7. Fisheye / Focus+Context Lenses

### The Technique

Fisheye distortion magnifies a local region around the cursor while leaving the surrounding context undistorted. The key property: it maintains spatial continuity — you can see both the magnified region and its relationship to the broader layout simultaneously. This is called "focus + context" visualization.

George Furnas introduced "Generalized Fisheye Views" at CHI 1986. Sarkar and Brown formalized the mathematical treatment. The core formula applies a nonlinear mapping to coordinates: points near the focus expand; points far from focus compress.

### D3 Implementation

Mike Bostock's `d3-plugins/fisheye` is the canonical implementation. Two distortion modes:

1. **Circular fisheye** (`d3.fisheye.circular()`): Magnifies a circular region around the cursor. Best for network graphs. Downside: creates curved grid lines, which distort quantitative axes.
2. **Cartesian fisheye** (`d3.fisheye.scale()`): Applies distortion independently to X and Y axes. Maintains straight axis lines. Better for scatterplots where axis positions carry meaning. Works with D3's linear, log, and power scales.

A smoother variant (`duaneatat/d3-fisheye` on GitHub) adds a smoothing parameter controlling the transition radius at the edge of the fisheye region.

The G6 graph visualization framework (`g6.antv.vision`) ships a built-in Fisheye plugin for network graphs. For raw canvas/WebGL, the fisheye transform is typically applied as a coordinate pre-processing step before rendering.

### Fit for Our Case

For 543 faces on a PaCMAP layout, fisheye is most useful in dense cluster regions where faces overlap heavily at medium zoom. A circular fisheye centered on cursor position would spread neighboring faces outward, making each face legible while preserving spatial context.

**Implementation approach with deck.gl:** Apply fisheye coordinate transformation to the viewport data coordinates before passing to the layer, or use a WebGL post-processing effect (texture distortion shader). The former is simpler; the latter is smoother. The deck.gl `project` module handles coordinate transforms, making pre-processing the pragmatic choice.

**Caution:** Fisheye distorts distances, which in this visualization encode semantic similarity. Users need to understand that magnified faces are not "more similar" to each other than the overall layout implies. A clear visual indicator (circular boundary, brief animation) helps calibrate expectations.

---

## 8. Cluster Overview → Drill-Down Navigation

### The Pattern

This is one of the oldest and most validated InfoVis patterns, documented in Ben Shneiderman's 1996 "Visual Information Seeking Mantra": **Overview first, zoom and filter, then details on demand.**

The cluster overview → drill-down variant works as follows:
1. **Overview:** Show all clusters as labeled regions. Individual items not distinguishable. Spatial distribution of fraud visible at a glance.
2. **Cluster selection:** User clicks a cluster label or convex hull. System zooms to frame that cluster.
3. **Cluster detail:** Individual faces now visible within the cluster. Faces sorted or highlighted by sus_level. Hover for metadata.
4. **Item detail:** Click a face → full record panel opens.

Implementations of this pattern:
- **Nomic Atlas:** The canonical modern implementation. Topic color regions at zoom-out, individual points at zoom-in. Navigation via both free zoom and explicit "jump to cluster" UI.
- **Relativity Analytics Cluster Visualization:** Document review tool that renders cluster sets as interactive maps with double-click drill-down to subclusters.
- **Oracle Logging Analytics:** Same concept for log data.

### Breadcrumb Trail

When implementing drill-down, a breadcrumb or "back to overview" mechanism is essential for disorientation recovery. Users who zoom into a cluster need a one-click path back to the full corpus view. This is especially important for the scam hunter user who switches between clusters rapidly.

---

## 9. Concrete Recommendations for Our Case

Ranked by impact/effort ratio. Each item specifies the library and the interaction it addresses.

### Tier 1 — Implement First (highest impact, 1-3 days each)

**R1. deck.gl WebGL renderer with semantic zoom switching**

Library: `deck.gl` (npm: `@deck.gl/react`, `@deck.gl/layers`)

Architecture:
```
zoom < 8  → ScatterplotLayer  (colored dots, fraud_score → color scale)
zoom 8-11 → IconLayer         (face thumbnails, 32px, from /faces/{job_id}.png)
zoom > 11 → IconLayer         (96px faces) + TextLayer (job title on hover)
```

The `onViewStateChange` callback provides current zoom. Switch layers by toggling `visible` prop. `IconLayer` supports per-item image URLs via `getIcon` accessor — pass `{ url: '/faces/${d.job_id}.png', width: 96, height: 96 }`. GPU renders all 543 faces in a single draw call.

Color scale for fraud score: use `d3-scale-chromatic` `interpolateRdYlGn` reversed (green = safe, red = fraud). This provides immediate legibility before the face uncanny valley kicks in at higher zoom.

**R2. HDBSCAN cluster convex hulls with jump-to navigation**

On the overview zoom level, draw convex hulls around each of the 40 clusters using deck.gl `PolygonLayer`. Color hulls by mean sus_level of members. Add cluster labels with `TextLayer`. Clicking a hull triggers `flyTo` animation to frame that cluster.

This gives scam hunters the ability to scan 40 clusters in seconds and immediately navigate to suspicious ones.

Compute convex hulls in Python at build time (scipy `ConvexHull`), bake into GeoJSON, serve as static file.

**R3. Color encode fraud score from the start**

Even before faces are visible at low zoom, the color of the dot should encode fraud score. Use a diverging scale anchored at 50 (neutral). This means the spatial density of red dots is visible from the full-corpus view — clusters of fraud are literally visually hot before a single face is legible.

### Tier 2 — Second Sprint (medium effort, high analyst value)

**R4. Fisheye lens on dense clusters**

Library: `d3-plugins/fisheye` or `duaneatat/d3-fisheye`

Activation: hold Alt + hover, or toggle via toolbar button. Apply circular fisheye transform to face coordinates in the canvas. Distortion radius ~150px, distortion factor ~3x. A visible circle boundary communicates the lens boundary to users.

This is most useful when a cluster has 30+ faces all overlapping. Without fisheye, even at high zoom they stack. With fisheye, the cursor separates them locally.

For deck.gl integration, apply the fisheye coordinate transform as a custom `getPosition` function that transforms data coordinates based on current cursor position. Requires re-rendering on mouse move — acceptable at 543 items.

**R5. Sidebar cluster summary panel**

When a cluster is selected (via hull click or zoom), show a sidebar with:
- Cluster label (e.g., "Remote IT work — Telegram collection")
- Mean fraud score + distribution histogram
- Top 3 faces (highest sus_level) as previews
- Top sus_factors for this cluster

This addresses the analyst user's need to understand cluster semantics. Implement as a React panel driven by selected cluster state.

**R6. RasterFairy grid view toggle**

Library: `rasterfairy` (pip), `pechyonkin/RasterFairy-Py3` for Python 3

Pre-compute grid positions at build time:
```python
from rasterfairy import transformPointCloud2D
grid_xy, (cols, rows) = transformPointCloud2D(pacmap_xy)
```

Store both `pacmap_xy` and `grid_xy` per job. Add a toggle button. Animate positions using `deck.gl`'s built-in `transitions` prop on `IconLayer` (specify `getPosition: { type: 'spring', stiffness: 0.1, damping: 0.15 }` for smooth spring animation between layouts).

Grid view is the "gallery browse" mode for the casual user — every face visible, no overlap, approximate neighborhood preserved.

### Tier 3 — Nice-to-Have (low priority)

**R7. Nomic Atlas as a zero-build alternative**

If time is short, upload the 543 face embeddings + metadata to Nomic Atlas (free tier) using the Python client:
```python
from nomic import atlas
project = atlas.map_embeddings(embeddings=pacmap_embeddings, data=metadata_dicts)
```

Nomic Atlas will automatically generate a hosted interactive map with cluster labels, zoom navigation, hover tooltips, and lasso selection. It won't show face thumbnails by default (it shows text metadata on hover), but it provides a working navigation demo in under an hour.

**R8. Scatter/Gather workflow for power users**

No library needed. Implement as: lasso selection on the deck.gl canvas (use `deck.gl`'s built-in picking + lasso extension) → selected job IDs → re-run PaCMAP on the selected subset → display the sub-layout in a modal. This lets analysts "zoom into" a suspicious cluster's internal structure by re-projecting just that subset.

The lasso extension is `@deck.gl/extensions` with `SelectionLayer`. Re-running PaCMAP on the fly requires a Python backend endpoint; alternatively, pre-compute sub-layouts for each HDBSCAN cluster at build time.

**R9. Density contours overlay (Apple Embedding Atlas style)**

Compute 2D KDE (kernel density estimate) over PaCMAP coordinates. Draw density contours as a background layer using deck.gl `ContourLayer`. This gives the visualization topographic map aesthetics and helps users immediately identify the densest regions.

Tip: compute separate KDE contours for high-fraud subset (sus_level > 70) and overlay them in red — creates a visual "fraud heat map" visible even at full-corpus zoom.

### What Not to Build

- **Force-directed layout for overlap resolution:** Destroys PaCMAP geometry. Not worth it.
- **OpenSeadragon deep-zoom tiles:** Designed for single high-resolution images (microscopy, art scans), not scatter plots. Wrong tool.
- **Gamification mechanics (points/badges):** No established ROI for investigation tools. The faces themselves are the engagement hook.
- **Temporal animation / streaming updates:** YAGNI for pre-generated static corpus.

---

## Sources

- [Semantic Zoom — InfoVis:Wiki](https://infovis-wiki.net/wiki/Semantic_Zoom)
- [Semantic Zoom | Gosling genomics framework](https://gosling-lang.org/docs/semantic-zoom/)
- [PixPlot — Yale DHLab](https://dhlab.yale.edu/projects/pixplot/)
- [PixPlot GitHub (pleonard212)](https://github.com/pleonard212/pix-plot)
- [Nomic Atlas](https://atlas.nomic.ai/)
- [Nomic Atlas — How to Visualize Embeddings](https://docs.nomic.ai/api/embeddings-and-retrieval/guides/how-to-visualize-embeddings)
- [Apple Embedding Atlas — GitHub](https://github.com/apple/embedding-atlas)
- [Apple Embedding Atlas — ML Research Page](https://machinelearning.apple.com/research/embedding-atlas)
- [Embedding Atlas Overview](https://apple.github.io/embedding-atlas/overview.html)
- [Renumics Spotlight — GitHub](https://github.com/Renumics/spotlight)
- [Scatter/Gather — Marti Hearst](https://people.ischool.berkeley.edu/~hearst/research/scattergather.html)
- [Scatter/Gather as a Tool for the Navigation of Retrieval Results — AAAI 1995](https://aaai.org/papers/0011-FS95-03-011-scatter-gather-as-a-tool-for-the-navigation-of-retrieval-results/)
- [D3 Fisheye Distortion — Mike Bostock](https://bost.ocks.org/mike/fisheye/)
- [d3-plugins/fisheye — GitHub](https://github.com/d3/d3-plugins/tree/master/fisheye)
- [duaneatat/d3-fisheye — smooth fisheye variant](https://github.com/duaneatat/d3-fisheye)
- [Fisheye View — InfoVis:Wiki](https://infovis-wiki.net/wiki/Fisheye_View)
- [G6 Fisheye Plugin](https://g6.antv.vision/en/manual/plugin/build-in/fisheye/)
- [RasterFairy — Quasimondo/RasterFairy](https://github.com/Quasimondo/RasterFairy)
- [RasterFairy Python 3 — pechyonkin fork](https://github.com/pechyonkin/RasterFairy-Py3)
- [RasterFairy — PyPI](https://pypi.org/project/rasterfairy/)
- [deck.gl IconLayer](https://deck.gl/docs/api-reference/layers/icon-layer)
- [deck.gl ScatterplotLayer](https://deck.gl/docs/api-reference/layers/scatterplot-layer)
- [Mosaic: Architecture for Scalable Interoperable Data Views (2024)](https://idl.cs.washington.edu/files/2024-Mosaic-TVCG.pdf)
- [Gamification in Data Visualization — Visualitics](https://visualitics.it/gamification-data-visualization/?lang=en)
- [OpenSeadragon](https://openseadragon.github.io/)
- [Cluster Visualization — RelativityOne](https://help.relativity.com/RelativityOne/Content/Relativity/Analytics/Cluster_visualization.htm)
- [WebGL rendering optimization — Medium](https://medium.com/@dhiashakiry/60-to-1500-fps-optimising-a-webgl-visualisation-d79705b33af4)
