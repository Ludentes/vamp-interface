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

## CFM render-cache built and verified (2026-05-18)

The training-pair render-cache for the CFM route is built. The corpus already
held `(photo, ArcFace, 52-d ARKit)` triples; the missing FLAME render of each
coefficient vector now has its supporting infrastructure:

- **FFHQ sha-index** — `output/ffhq_index/ffhq_sha_index.parquet` maps each
  `image_sha256` to its `(shard_idx, row_idx)` in the FFHQ parquet shards, so a
  row can be turned back into its photo.
- **MediaPipe pose cache** — `output/flame_pose_cache/pose_cache.parquet`, one
  head pose (rotation matrix + bbox) per FFHQ row. Detection rate **69928 /
  70000**; pose / blendshape detection agreement **1.000** (the two detection
  passes never disagree, so `bs_detected` is a safe proxy for pose).
- **`flame_render.py`** — `deform(basis_coeffs)` deforms the FLAME base mesh by
  an ARKit basis vector; `render(verts, rotation, bbox, modality)` rasterises a
  posed control image (normals / depth / flat).

Verified by eyeball collage (`verify_flame_render.py` → 8 FFHQ rows, calm vs
expressive): pose tracks the photo head, the calm/expressive expression split
is obvious, and the normal-map render is *not* inside-out — no `order` flip
needed. One known minor imperfection: a small vertical offset / scale mismatch
in the bbox→render fit (mesh sits slightly high on some rows); a calibration
nicety, not a correctness failure, and it does not block CFM use.

- `docs/research/2026-05-18-cfm-render-cache-verification.md` — verification, per-row notes, collage.
- `docs/superpowers/specs/2026-05-18-cfm-render-cache-design.md` — design spec.

## Trainable-InfuseNet feasibility spike — GO (2026-05-19)

GATING spike for the CFM training-run plan: prove a bf16 InfuseNet runs
forward+backward through frozen FLUX with non-zero gradient on the RTX 5090.

**Verdict: GO.** One CFM step landed at `loss=447.30`, `grad_norm=8.81`,
`peak_vram=30.7 GB` with gradient checkpointing on the FLUX trunk + InfuseNet.

Key finding: the bf16 weights at
`data/infiniteyou_dl_bf16/infu_flux_v1.0/sim_stage1/InfuseNetModel/` are a
**diffusers `FluxControlNetModel`** — load via `FluxControlNetModel.from_pretrained(...)`,
forward returns `(controlnet_block_samples, controlnet_single_block_samples)`
that FLUX's standard `forward(..., controlnet_block_samples=..., controlnet_single_block_samples=...)`
accepts directly. **No custom pipeline porting required.** The 8 identity
tokens from the InfiniteYou Resampler concatenate to the T5 sequence.

Trainable surface: peft LoRA (r=8) on InfuseNet's 4 double + 10 single
DiT-copy blocks (~4.5 M params) plus the `x_embedder` + `controlnet_x_embedder`
control-image input stems (~0.4 M). Krea-bf16 used as base for the spike;
real run should use FLUX.1-dev bf16 (the trunk InfuseNet was trained against).

- `docs/research/2026-05-19-trainable-infusenet-spike-verdict.md` — full verdict + concrete API.
- `src/arkit_controlnet/cfm/spike_trainable_infusenet.py` — reproducible spike.

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

## Matryoshka one-shot ControlNet — grid sweep (2026-05-18)

Follow-on. The painted matryoshka face is oversized (folk-art proportions) and
does not overlay a realistic face — which is why every refine / composite /
latent-mask approach failed. **Fix: one-shot generation** — Z-Image Turbo
txt2img + `Z-Image-Turbo-Fun-Controlnet-Union` driven by a Canny of the
per-job swap identity, so a correctly-proportioned realistic face is baked in
at generation time; `inswapper` then swaps onto a face it can detect. The CN
loads via `ModelPatchLoader` → `ZImageFunControlnet` (stock `ControlNetLoader`
rejects the `videox_fun` format); both nodes are native to ComfyUI ≥ 0.18.

Grid sweep over 20 identities × CN strength {0.5, 0.7, 0.9} × steps {6, 8} on
the Windows 3090 — 120 cells, 0 failed, 27.9 min. **Verdict: strength 0.90,
6 steps** — 100% SCRFD `default` across all 20 identities, id_cos 0.864 mean
(0.795 worst), fastest. CN strength is the dominant axis (monotone on every
metric; 0.50 is the only setting that drops cells to MediaPipe `forced`);
steps barely matter. Open: id_cos plateaus ~0.86 — the swap-onto-small-painted-
face ceiling, not a CN limit; micro-sweep strength past 0.90 to confirm.

- `docs/research/2026-05-18-matryoshka-cn-grid-sweep.md` — results + verdict.
- `scripts/cn_grid_sweep.py` — harness (remote ComfyUI, `/upload/image` API).

## Face-swapper landscape — inswapper alternatives (2026-05-18)

Options survey for the swap stage (current `inswapper_128` result is fine; this
is a menu, not a pivot). **`inswapper_256/512` are not real weights** —
DeepInsight never released them. Only two newer swappers slot into
`swap_core`'s InSwapper ONNX contract without a loader rewrite:
**HyperSwap-256** (FaceFusion Labs, 2025, alive, drop-in ONNX, 2× res — but
non-commercial research license) and **ReSwapper-256** (AGPL-3.0, clean
redistributable license, fidelity slightly below inswapper). Diffusion-grade:
**REFace** (WACV 2025, heavy SD env, non-commercial). Dead/wrong-task:
GHOST/GHOST-2 (abandoned / head-swap), SimSwap (dormant), DeepFaceLive
(archived), DiffSwap (stale), image-DreamID (no weights).

**Bake-off ran (2026-05-18) → adopted HyperSwap 1c as the default swapper.**
20 identities × 4 backends, swap stage isolated (`scripts/swapper_bakeoff.py`).
id_cos: inswapper_128 **0.864**, hyperswap_1c 0.796, hyperswap_1b 0.790,
hyperswap_1a 0.743 — inswapper wins the *metric* on every identity. But on
visual inspection HyperSwap 1c is the better swap (sharper, 256px = 2×
inswapper, photoreal); the ~0.07 id_cos gap is HyperSwap's accepted skin-tone
averaging, not a broken swap. **Decision: HyperSwap 1c is now
`swap_core.DEFAULT_SWAPPER`**; `load_swapper()` dispatches on filename so
inswapper still loads via an explicit path. The HyperSwap tier (1a→1b→1c)
does not change identity (1c +0.006 over 1b, noise); 1c is picked for visual
quality. ReSwapper-256 falsified (cos ≈0.2 — well-aligned but cannot carry
identity onto the small painted doll face; tested all latent conventions).
The ~0.86 id_cos ceiling is target-side; pushing the *metric* further =
generation-time injection (PuLID / InfiniteYou), not a bigger swapper.

**GHOST falsified (2026-05-19).** Surveyed swapper *training* code (the user
asked whether we could train/fine-tune a swapper on our data). Findings:
HyperSwap ships training code but no resumable `.ckpt` — training = from
scratch on VGGFace2, days-to-weeks, and our ~20 identities are far too few;
GHOST-1 is the cleanest transparent training reference (publishes
generator+discriminator weights, documents fine-tune-vs-scratch loss tuning);
GHOST-2.0 is head-swap, wrong task. Gated GHOST-1 quality before any training
analysis: ran its released AEI_Net generator natively on 5 CN-grid dolls
(`scripts/ghost_gate.py`) → **0.536 mean id_cos**, soft and identity-weak.
Falsified — no swapper-training route is worth pursuing; the ~0.86 ceiling is
target-side and the lever stays generation-time identity injection.

- `docs/research/2026-05-18-face-swapper-landscape.md` — survey + bake-off + decision.
- `scripts/swapper_bakeoff.py` — harness; `scripts/swap_core.py` — `HyperSwap`, `DEFAULT_SWAPPER`.
- `scripts/ghost_gate.py` — GHOST quick gate; `exp_output/ghost_gate/`.

## Related threads

- `_topics/arc-distill.md` — ArcFace-in-FLUX-latent distillation; the
  auxiliary identity loss, if used, comes from here.
- `_topics/arkit-bridge.md` — ARKit → PersonaLive bridge; the live-animation
  path this ControlNet would feed as an offline teacher.
