# vamp-interface

An experiment in using photorealistic AI-generated faces as a data visualization medium. Each data point becomes a face. Fraudulent job postings develop uncanny, wrong-feeling faces. Legitimate ones look unremarkable.

The premise: humans process faces pre-cognitively. A grid of 200 faces can be scanned in seconds. A table of 200 rows cannot. If the encoding is right, the odd one out jumps out before the viewer can say why.

---

## What This Is

A visualization layer over the [telejobs](../telejobs) fraud detection corpus (23k Telegram job postings, fraud-scored). Each posting is rendered as a face:

- **Identity** (face shape, proportions, character) — driven by the raw text embedding. Similar postings look like the same person. Cluster membership becomes visible.
- **Expression** (eyes, brow, mouth tension) — driven by the fraud factor vector. High-fraud signals produce uncanny, wrong-feeling expressions. Legitimate postings produce open, natural faces.
- **Denoising strength** — scales with `sus_level`. A score of 90 drifts far from the neutral anchor into uncomfortable territory. A score of 10 barely deviates.

The signal is not "this face looks angry." The signal is "something is wrong with this face" — a pre-cognitive reaction that requires no legend, no calibration, no reading.

---

## Status

**Research / pre-prototype.** No runnable code yet. The design is documented; the hypothesis is untested.

Key open question: does the linear projection from job-post embedding space to CLIP conditioning space produce face variation that looks like face variation, or just arbitrary noise? This is the first thing to test.

---

## Documentation

| Document | What it covers |
|---|---|
| [docs/design/scenarios.md](docs/design/scenarios.md) | Three user personas: scam hunter, analyst, bored student. Which encoding works for each. |
| [docs/design/diffusion-approach.md](docs/design/diffusion-approach.md) | Technical design: two-pass img2img, PCA projection, uncanny valley as the signal. |
| [docs/research/2026-04-06-state-of-art.md](docs/research/2026-04-06-state-of-art.md) | Literature review: Chernoff faces, disentangled models, preattentive face processing. |
| [docs/research/2026-04-06-telejobs-data-landscape.md](docs/research/2026-04-06-telejobs-data-landscape.md) | What data is available from telejobs: 16 fraud factors, sus_level, structured metadata, embeddings (planned). |

---

## Related Projects

- **telejobs** — the data source. Telegram job posting scraper + fraud scorer.
- **portrait-to-live2d** — face generation infrastructure this project draws from conceptually. Live2D path remains an option if photorealistic diffusion proves too noisy.

---

## The Name

From Watts' *Blindsight*