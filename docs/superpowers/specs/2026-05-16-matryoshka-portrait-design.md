---
status: live
topic: matryoshka-portrait
---

# Matryoshka Portrait — Design

## Purpose

Given a photo of a person, produce a **single matryoshka (Russian nesting
doll) image bearing their likeness**. v1 is one doll; the full nested set is
deferred to v2. This is a **standalone feature**, not an arm of the
photoreal↔stylized training corpus — success is measured by *recognizability
and delight*, not label-transfer fidelity.

The task is structurally the chibi task: an anchor photo → a stylization that
flattens and abstracts the face → identity preserved → uncanny valley
avoided. The chibi-sweep methodology transfers directly.

## Research outcome

No off-the-shelf matryoshka **style** LoRA exists. "Russian Doll Likeness" /
"Super Russian doll" on Civitai are false friends (a face-aesthetic LoRA and
an outfit LoRA respectively). Closest adjacent style assets are folk-art
wooden-carving LoRAs (Wizard's Wooden Figure Carvings, Wooden characters).
Flux-Krea base knows "matryoshka doll" as a common object; the **flat-painted
face convention** is the part at stylistic risk. Decision: start prompt-only;
train a small matryoshka LoRA only if the painted-face style proves
unconvincing (YAGNI until the baseline shows the need).

## Approach

Rejected: single-pass prompt + PuLID — PuLID injects photoreal identity onto
a form that should be flat-painted, reproducing the partial-stylization
uncanny artifact the chibi sweep diagnosed.

Chosen: a **hero-doll pipeline** reusing the chibi node graph —
Flux-Krea + PuLID (identity) + Canny ControlNet of a doll-silhouette template
(structure) + style prompt/LoRA. For a single doll, ControlNet is a
structure tool (keeps PuLID from drifting the render into a portrait), not a
contract — it is swept, not fixed.

### Pipeline (v1)

1. **Extract** — normalize the input face via the existing insightface crop.
2. **Structure** — Canny edge map of a generic matryoshka silhouette template
   (the bowling-pin form). Optional; prompt-only is the Phase 0 baseline.
3. **Render** — Flux-Krea + PuLID + style prompt → one frontal matryoshka
   doll. This is where all uncanny tuning lives.
4. **Score** — run the metric suite (below).

## Components

- **`scripts/matryoshka_sweep.py`** — sweep runner, near-clone of
  `chibi_highstr_sweep.py` (same queue/wait/download/build_workflow
  plumbing, deterministic manifest, atomic writes, skip-if-exists).
  Grid: identity × PuLID weight × style/LoRA strength × schedule.
- **`scripts/score_matryoshka.py`** — scorer, re-skin of `score_highstr.py`.
- **Doll-silhouette template** — one Canny PNG, a generic matryoshka outline.
- **Workflow template** — reuse `flux_pulid_canny_lora.api.json` unchanged.

## Metrics

| Metric | Definition | Role |
|---|---|---|
| Recognizability | human eyeball rating (1–5): "is this clearly *them*?" | **ground truth** |
| Coarse-attribute match | buffalo_l gender + apparent-age bucket + hair region/colour vs input | auto proxy for recognizability |
| Matryoshka-ness | CLIP/SigLIP score of render vs "Russian matryoshka nesting doll" against distractors ("ceramic figurine", "chibi", "a person") | style fidelity |
| Uncanny proxy | ArcFace **detection rate** — a well-painted doll face should *not* trip ArcFace as a human face (low = committed to the doll read) | headline, reused verbatim from chibi sweep |
| Measurability floor | MediaPipe landmarkability — render is not garbage | sanity gate |

Raw ArcFace identity cosine to the input is **explicitly not used** — a good
flat-painted doll face scores ~0 (established on chibis); it would punish
correct stylization.

## Plan

- **Phase 0 — baseline.** Prompt-only Flux-Krea, single doll, 3–4
  identities, eyeball. Decide whether a matryoshka style LoRA is needed.
- **Phase 1 — hero-doll sweep.** Canny doll template + PuLID + style; sweep
  PuLID weight × schedule × style strength. Score with the full suite.
  Pick the cell that maximises recognizability while staying past the
  ArcFace-detection uncanny threshold.
- **Phase 2 — deferred.** Full nested set (row of N, per-doll low-denoise
  img2img for detail falloff, inter-doll consistency cosine); open-reveal
  variant; matryoshka LoRA training if Phase 0/1 demand it.

## Error handling

- Sweep is resumable (skip-if-exists, atomic PNG writes, up-front manifest) —
  inherited from the chibi runner.
- Workflow node-class assertion at startup (a re-exported/renumbered
  workflow fails loud) — inherited.
- A render where MediaPipe fails to landmark is flagged garbage, not scored.

## Testing

- Phase 0 gate: eyeball 3–4 identities — does prompt-only read as a
  matryoshka at all, and is the face recognizable?
- Phase 1 gate: the metric suite + an eyeball montage (strength × PuLID-weight
  grid), same as the chibi `highstr_id14_grid.png` montage.
