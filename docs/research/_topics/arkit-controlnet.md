# Topic: ARKit-conditioned ControlNet on FLUX

**Status:** live, new thread (opened 2026-05-16)

## Current belief

InfiniteYou's InfuseNet is "a generalization of ControlNet" with two
conditioning slots: an ArcFace identity path (residual injection of 8 tokens)
and an *already-existing* optional spatial-control image (currently a 5-point
keypoint image). Re-purposing that control slot for a render of an ARKit-driven
FLAME mesh gives fine expression control over an ArcFace-steady identity.
The user's "combine ArcFace + ARKit" is "keep ArcFace, add ARKit" — ArcFace is
already the identity path.

**Loss:** train on conditional flow matching (CFM) against real
`(photo, FLAME-render)` pairs. CFM is grounded — it does *not* inherit the
slider thread's foolable-critic / VAE-artifact failure. Only if expression
control-following proves weak (corpus skew) add an auxiliary blendshape critic
loss, gated `t_max ≈ 0.5` per channel via the per-channel R²(t) map. The
slider thread's timestep-gating fix transfers to that auxiliary loss, not the
base loss. Sequence: CFM + SPMS data first (InfiniteYou's own stance — it
rejected an identity loss), auxiliary critic only as fallback.

**Corpus:** `output/reverse_index.parquet` (79,116 rows) already pairs each
image with its 52-d ARKit blendshapes and 512-d ArcFace embedding — the
training triples exist; only the FLAME render of each coefficient vector is
missing.

## Zero-train compose spike — falsified for expression (2026-05-18)

Tested the shortcut *before* Path 1: compose InfiniteYou (identity) +
FluxSpace `FluxSpaceEditPair` (expression, attention editing) in one ComfyUI
graph, zero training. The two modules co-exist with no contention and identity
injection is solid (ArcFace cos 0.60–0.83), but **FluxSpace is not a viable
expression channel** — hard collapse to noise at scale ≥1.5, weak and
identity-dependent smile in the only usable band (clear on 2/5 identities),
and no continuous modulation (s0.5 ≈ s1.0). Strengthens the case for the
*trained* CFM route; FluxSpace is not a substitute for a trained expression
ControlNet. Doc: `docs/research/2026-05-18-arkit-controlnet-spike-verdict.md`.

## Path 1 zero-train spike — falsified for expression (2026-05-18)

Tested Path 1 the cheap way: a MediaPipe 478-vertex face-mesh control image fed
into InfuseNet's spatial-control slot (no FLAME, no training). 3 identities ×
3 axes × strength {0.6, 1.0}, 24/24 rendered. **Negative:** only 2/9
identity×axis cells beat the neutral-control baseline by the ≥0.05 expression
margin; several deltas were strongly negative (dense mesh acting as OOD noise);
str 1.0 destroyed identity for 2/3 identities; the collage shows no
viewer-visible expression change across control columns. Confirms the design's
central risk — the 5-keypoint-trained InfuseNet slot has no expression
bandwidth for a dense tessellation. Identity injection still solid. Both
zero-train expression routes (FluxSpace + InfuseNet mesh slot) are now
falsified; expression must be trained (CFM). Doc:
`docs/research/2026-05-18-arkit-landmark-control-spike-verdict.md`.

## Approach C zero-train spike — falsified for expression (2026-05-18)

Tested the last cheap fallback: a stock FLUX Depth ControlNet (InstantX
ControlNet-Union, `depth` mode) stacked alongside identity-only InfuseNet,
driven by a flat-shaded depth raster of the MediaPipe face tessellation. No
training. 3 identities × 4 axes × strength {0.5, 0.8}, 24/24 rendered.
**Negative, both regimes:** at str 0.8 the ControlNet imprints the literal
low-poly faceted mesh as a 3D plastic mask (11/12 outputs have no detectable
face); at str 0.5 the face is photoreal and identity holds (ArcFace
0.39–0.66) but expression does not transfer at all — surprise and neutral
outputs are indistinguishable, baseline-relative expr deltas average negative
on every axis (smile −0.01, pucker −0.18, surprise −0.33). No usable strength
between. Root cause: a faceted landmark raster is OOD for a depth ControlNet
trained on smooth scene depth — the same OOD gap that killed the other two
routes. All three zero-train expression routes (FluxSpace, InfuseNet mesh
slot, stacked depth CN) are now falsified. **No free cheese — proceed
directly to the CFM run.** Doc:
`docs/research/2026-05-18-arkit-depth-controlnet-spike-verdict.md`.

## Open questions

- SPMS data: mine ArcFace-near / blendshape-far pairs from `reverse_index`, or
  generate synthetic SPMS per InfiniteYou Stage 2.
- Render modality for the *trained* CFM channel — surface normals vs depth vs
  flat-shaded mesh. (Untrained, all modalities are OOD; the question is which
  the CFM model learns from fastest.)
- Position vs FG-Portrait (CVPR 2026) — the closest published prior art.

## Reading list

- `docs/research/2026-05-16-arkit-controlnet-infiniteyou.md` — design analysis,
  loss design, corpus, prior art. **Load-bearing.**
- `docs/papers/infiniteyou-2503.16418.pdf` — InfuseNet architecture, SPMS
  ablation, plug-and-play ControlNet stacking.
- `docs/papers/fg-portrait-2603.23381.pdf` — closest prior art.
- `docs/research/2026-04-30-bs-loss-classifier-fooling.md` — the slider-thread
  latent-loss artifact this design routes around.
- `docs/research/2026-05-03-slider-operational-handbook.md` — the `t_max`
  gating values and the per-channel R²(t) map reused for the auxiliary loss.

## Matryoshka fast-model bake-off (2026-05-18)

Side experiment on the idle Windows 3090: which fast 2026 base model generates
the *generic* matryoshka doll fastest without losing quality (identity is added
downstream by inswapper, so generation needs only Canny + folk-art style).

**Verdict: Z-Image Turbo at 6 steps** — 7.2 s warm, 3.1× faster than the
flux_krea baseline (22.5 s), coherent photoreal single doll despite being
prompt-only (no Canny CN for Z-Image). flux_schnell rejected (only usable at
8 steps, no faster than baseline); sdxl_lightning rejected (3 s but flat
illustration output, unsuitable for face-swap input — and still paints the
black doll-eyes, confirming that artifact is not FLUX-specific).

- `docs/research/2026-05-18-matryoshka-bakeoff-verdict.md` — results + verdict.
- `docs/superpowers/specs/2026-05-18-matryoshka-fast-model-bakeoff-design.md` — design.

## Related threads

- `_topics/arc-distill.md` — ArcFace-in-FLUX-latent distillation; the
  auxiliary identity loss, if used, comes from here.
- `_topics/arkit-bridge.md` — ARKit → PersonaLive bridge; the live-animation
  path this ControlNet would feed as an offline teacher.
