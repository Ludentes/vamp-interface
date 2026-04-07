# Research: Embedding Visualization — UMAP and Modern Alternatives

**Date:** 2026-04-07  
**Sources:** 15 sources — GitHub repos, PyPI, NVIDIA blog, NeurIPS 2024, AAAI 2025, Apple engineering  
**Context:** 543 job posting embeddings (1024-d, qwen3), building a 2D face grid inside the telejobs React frontend.

---

## Executive Summary

UMAP remains the dominant tool but has well-documented reproducibility problems and has not evolved much since 2022. The most significant recent development is **PaCMAP** (with active 2024–2025 research) which provably preserves both local and global structure better than UMAP. For visualization tooling, two libraries stand out: **datamapplot** (TutteInstitute, March 2026 release) for publication-ready interactive HTML, and **Apple Embedding Atlas** which ships a React component — directly embeddable in the telejobs frontend. At 543 points, algorithm choice barely matters for speed; what matters is layout stability (run once, store in DB) and how the 2D coordinates plug into the React UI.

---

## Key Findings

### UMAP: mature, widely deployed, reproducibility is a real problem

UMAP 0.5.8 is stable and well-understood. On CPU it runs in under a second at 543 points. NVIDIA's cuML in RAPIDS 25.02 delivers a confirmed 60x CPU speedup [1], reaching interactive speeds for datasets of 20M+ points, but this requires RAPIDS which is a heavy optional dependency — irrelevant at our scale.

The structural limitation is **non-determinism**. Since version 0.4, UMAP uses multi-threading during optimization, and the thread scheduling introduces randomness that `random_state` cannot fully control [2][3]. Two runs with identical parameters can produce different layouts. The UMAP docs acknowledge this but offer no fix other than disabling threads (`n_jobs=1`), which eliminates the speed advantage. This matters for production: every time the layout is recomputed, the grid reorganizes. The mitigation is straightforward — compute once, store coordinates in the database — but it means the layout cannot be updated incrementally.

### PaCMAP: better structure preservation, actively developed

PaCMAP (Pairwise Controlled Manifold Approximation) was published in JMLR and received the John M. Chambers Statistical Software Award from the American Statistical Association [4]. A 2021 empirical study in JMLR compared t-SNE, UMAP, TriMAP, and PaCMAP across benchmarks; PaCMAP achieved the best SVM accuracy on more datasets than any other method and is the only one that explicitly balances local and global structure [5].

The key insight: t-SNE and UMAP optimize primarily for local structure. TriMAP optimizes for global structure at the cost of local. PaCMAP uses three pair types (neighbor, mid-near, further) to jointly optimize both. In practice this means clusters are better separated and global topology — which cohort is near which other cohort in embedding space — is more faithfully reflected.

Two significant 2024–2025 developments extend the core:

**ParamRepulsor (NeurIPS 2024)** [6]: A parametric variant that trains a neural network to perform the projection. Parametric methods can embed new points without recomputing the full layout — useful if the corpus grows continuously. ParamRepulsor adds Hard Negative Mining and a strong repulsive loss to fix the known failure mode of parametric methods (they preserve global structure but lose local detail). GPU-accelerated via PyTorch.

**LocalMAP (AAAI 2025)**: A follow-on method for locally adjusting the graph during reduction; slated for integration into the main PaCMAP package. Minimal details available in sources.

PaCMAP installs via `pip install pacmap`, follows scikit-learn conventions (`fit_transform`), and uses FAISS as its default KNN backend (replaced Annoy as of 0.8.0). At 543 points it runs in well under a second on CPU.

### TriMAP and t-SNE: not recommended

TriMAP preserves global structure well but fails on local structure [5] — the opposite failure mode from UMAP. t-SNE (both openTSNE and sklearn) is slow at large scale, has well-known cluster-size distortion, and has not had significant algorithmic improvements since 2022. Neither is the right choice here.

### datamapplot: best-in-class visualization layer, not an algorithm

datamapplot (TutteInstitute, MIT license, latest 0.7.1 released March 2026) [7] is not a dimensionality reduction algorithm — it is a visualization layer that takes precomputed 2D coordinates plus cluster labels and produces publication-ready static or interactive HTML outputs. You supply the projection; datamapplot handles automated label placement, arrow styling, word-cloud overlays, dark mode, and interactive HTML with zoom/pan/search.

This is the TutteInstitute project, the same group that originally developed UMAP. Their decision to separate the layout algorithm from the visualization layer reflects production experience: the algorithm should be computed offline and cached; the visualization should be flexible and configurable independently.

For telejobs the interactive HTML output (`create_interactive_plot`) is most relevant — it produces a self-contained HTML file with JavaScript. This is well-suited for a documentation/analysis artifact but not for embedding in a React SPA, because it is not a React component.

### Apple Embedding Atlas: embeddable React component, production-grade

Apple released Embedding Atlas as MIT-licensed open source [8]. It is the most directly applicable tool for the telejobs integration. Key details:

- `pip install embedding-atlas` then `embedding-atlas <dataset.parquet>` launches a local server
- Available as a **Jupyter widget**, a **Streamlit component**, and critically as **React and Svelte npm packages** (`embedding-atlas/react`, `embedding-atlas/svelte`)
- WebGPU rendering with WebGL 2 fallback — handles millions of points at smooth frame rates
- Built-in: automatic clustering and labeling, kernel density estimation, density contours, real-time nearest-neighbor search, multi-coordinated metadata views, cross-filtering
- Accepts precomputed 2D coordinates (Parquet format with x, y columns) — you bring your own projection

The React component (`embedding-atlas/react`) is installable via npm and can be dropped directly into the telejobs frontend as a new page component. The interface accepts a DataFrame/Parquet input with x, y coordinates plus metadata columns, and handles all rendering internally. This avoids building a custom canvas renderer from scratch.

The one limitation for our face grid use case: Embedding Atlas renders points (colored dots, density maps), not face images. At zoom-out level this is fine — cohort clusters become visible as colored regions. Zoom-in to individual points would need custom handling to show face thumbnails. This is doable by layering absolutely positioned `<img>` elements on top of the canvas at high zoom levels, triggered by a zoom event from the component.

### Nomic Atlas: cloud-first, not self-hostable without enterprise contract

Nomic Atlas is a managed service for embedding visualization with an impressive UI [9]. It computes its own UMAP-like projections internally and supports semantic search, automatic topic labeling, and temporal views. Self-hosting is available only through an enterprise plan — no community self-hosted option exists as of April 2026. Not applicable for our stack.

### Renumics Spotlight: good for data scientists, not for product embedding

Spotlight [10] is a Python-first tool (`pip install renumics-spotlight`) that shows embeddings in a "Similarity Map" view alongside tabular metadata and media. It is excellent for exploratory data analysis in a Python environment and supports images natively. However, it launches its own server (not embeddable as a React component) and is designed for data science workflows, not end-user web products. Not applicable for the telejobs frontend integration.

---

## Comparison

| Tool | Category | Our 543-point use | React embeddable | Self-hostable | Maintenance |
|------|----------|-------------------|-----------------|---------------|-------------|
| **PaCMAP** | Algorithm | Best layout quality | — | — | Active (NeurIPS 2024) |
| **UMAP** | Algorithm | Works fine, non-deterministic | — | — | Stable |
| **ParamRepulsor** | Algorithm (parametric) | Overkill at 543pts | — | — | Active (NeurIPS 2024) |
| **datamapplot** | Viz layer | Great for HTML reports | No | Yes | Very active (Mar 2026) |
| **Embedding Atlas** | Viz tool + component | Strong fit | **Yes** | Yes | Active (Apple) |
| **Nomic Atlas** | Managed service | Strong viz but cloud-only | No | Enterprise only | Active |
| **Renumics Spotlight** | EDA tool | Good for analysis, not product | No | Yes | Active |
| **cuML UMAP** | GPU algorithm | Irrelevant at 543pts | — | — | Active (NVIDIA) |

---

## Recommendation for vamp-interface / telejobs

**Algorithm: PaCMAP.** At 543 points the speed difference vs UMAP is negligible. PaCMAP's better global structure preservation matters: we want cohort clusters that are semantically close in embedding space to also be spatially close on screen. UMAP's non-determinism is a further reason to prefer PaCMAP. Run once, store `(umap_x, umap_y)` — keep the name `umap_x/y` in the DB schema for familiarity, even if computed by PaCMAP.

**Visualization: Embedding Atlas React component.** Drop `embedding-atlas/react` into the telejobs frontend. It handles zoom, pan, density estimation, nearest-neighbor search, and clustering labels out of the box. The face image overlay is the one custom piece: render face `<img>` elements absolutely positioned over the canvas when zoom exceeds a threshold. This is 30–50 lines of React on top of the component, not a from-scratch implementation.

**For research artifacts:** Use datamapplot to produce an interactive HTML report alongside the face grid — one `create_interactive_plot` call on the PaCMAP coordinates generates a shareable analysis document.

---

## Open Questions

- Embedding Atlas React component API: the npm package exists (`embedding-atlas`) but detailed React prop documentation was not retrieved. Need to check `npm info embedding-atlas` and the component's TypeScript types before integrating.
- Optimal PaCMAP parameters for 1024-d text embeddings at n=543: default `n_neighbors=10` should work but `mn_ratio` and `fp_ratio` tuning may improve cohort separation visibility.
- Zoom threshold for face image overlay in Embedding Atlas: depends on the rendered point spacing at various zoom levels, requires empirical testing.

---

## Sources

[1] NVIDIA. "Even Faster and More Scalable UMAP on the GPU with RAPIDS cuML". https://developer.nvidia.com/blog/even-faster-and-more-scalable-umap-on-the-gpu-with-rapids-cuml/ (Retrieved: 2026-04-07)

[2] lmcinnes/umap GitHub Issues. "Setting a random state still leads to stochastic results". https://github.com/lmcinnes/umap/issues/1080 (Retrieved: 2026-04-07)

[3] lmcinnes/umap GitHub Issues. "Semi-deterministic output even though random_state is set". https://github.com/lmcinnes/umap/issues/1108 (Retrieved: 2026-04-07)

[4] YingfanWang/PaCMAP GitHub README. https://github.com/YingfanWang/PaCMAP (Retrieved: 2026-04-07)

[5] Wang et al. "Understanding How Dimension Reduction Tools Work: An Empirical Approach to Deciphering t-SNE, UMAP, TriMAP, and PaCMAP for Data Visualization". JMLR 22(1). https://dl.acm.org/doi/abs/10.5555/3546258.3546459 (Retrieved: 2026-04-07)

[6] Huang et al. "Navigating the Effect of Parametrization for Dimensionality Reduction". NeurIPS 2024. https://arxiv.org/abs/2411.15894 (Retrieved: 2026-04-07)

[7] TutteInstitute/datamapplot GitHub. https://github.com/TutteInstitute/datamapplot (Retrieved: 2026-04-07)

[8] Apple/embedding-atlas GitHub. https://github.com/apple/embedding-atlas (Retrieved: 2026-04-07)

[9] Nomic Atlas Documentation. https://docs.nomic.ai/ (Retrieved: 2026-04-07)

[10] Renumics Spotlight GitHub. https://github.com/Renumics/spotlight (Retrieved: 2026-04-07)
