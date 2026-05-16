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

## Open questions

- Path 1 spike result: does a stock normal/depth ControlNet driven by a FLAME
  render steer FLUX expression while ArcFace holds identity, with zero training?
- Render modality — surface normals vs depth vs flat-shaded mesh vs landmark
  overlay — which the InfuseNet control channel reads best.
- SPMS data: mine ArcFace-near / blendshape-far pairs from `reverse_index`, or
  generate synthetic SPMS per InfiniteYou Stage 2.
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

## Related threads

- `_topics/arc-distill.md` — ArcFace-in-FLUX-latent distillation; the
  auxiliary identity loss, if used, comes from here.
- `_topics/arkit-bridge.md` — ARKit → PersonaLive bridge; the live-animation
  path this ControlNet would feed as an offline teacher.
