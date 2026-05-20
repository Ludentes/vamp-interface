---
status: live
topic: arkit-controlnet
supersedes: 2026-05-19-cfm-expression-controlnet-design.md
---

# CFM Expression-ControlNet Training Run — Design

**Date:** 2026-05-20

**Goal.** Train InfuseNet (the InfiniteYou control channel) to follow a FLAME
expression render via conditional flow matching against real
`(photo, FLAME-render)` pairs. The base FLUX DiT and the InfiniteYou identity
path (ArcFace + Resampler) stay frozen; only a LoRA on InfuseNet's DiT-copy
blocks and InfuseNet's two control-image input stems train.

Supersedes
[2026-05-19-cfm-expression-controlnet-design.md](2026-05-19-cfm-expression-controlnet-design.md):
the gating spike now records a concrete trainable API (verdict doc
[2026-05-19-trainable-infusenet-spike-verdict.md](../../research/2026-05-19-trainable-infusenet-spike-verdict.md));
render-fit calibration is done (commit `6ae0a18` — face-region fit, then
landmark-anchored similarity transform); the pose cache now carries all 478
MediaPipe landmarks per row. Two open questions in the prior spec are also
closed: render modality is **A — normals only**; dataset is the **intersection
of `pose_detected` ∩ `bs_detected` on FFHQ = 69,918 rows**.

The user constraint that reshapes this design vs the prior spec: **fail fast,
samples ASAP.** A 70k precompute → 20k training-step pipeline has a multi-hour
blind window before the first generation collage. The design front-loads a
held-out eval set, dumps samples every ~50 steps, and gates the long run on a
visual check at step ~500 (after roughly 30-40 min of wall clock).

## Constraints already fixed

- **Render modality:** A — normals only (`flame_render.render(modality="normals")`).
- **Render alignment:** landmark-anchored similarity transform on 6 MediaPipe-478 ↔
  iBUG-70 pairs (`landmark_align.aligned_pixels()`), not the bbox-fit projection.
  The bbox-fit `_project` stays as a fallback for rows missing landmarks (should
  be 0 in the intersection but keep the safety net).
- **Dataset:** rows where both `reverse_index.bs_detected` and
  `pose_cache.pose_detected` are true ∩ `source=='ffhq'` — 69,918 rows.
- **Base FLUX:** `FLUX.1-dev` bf16 — InfuseNet was trained against dev, not Krea.
  The spike used Krea-bf16 because dev-bf16 wasn't on disk; this run downloads
  dev-bf16 first (~24 GB). One-line config switch; no other code change.
- **Resampler weights** for the 8 identity tokens:
  `data/infiniteyou_dl_bf16/infu_flux_v1.0/sim_stage1/image_proj_model.bin`
  (~50 MB) — downloaded once, then loaded frozen.

## Approaches considered

**A. Sequential precompute → train → eval.** Precompute photo latents + id
tokens for all 69,918 rows, then train, then evaluate. Simple. Blind window:
~2 h precompute + N hours training before any generation collage. Rejected:
violates the "samples ASAP" constraint.

**B. Eval-set-first phased.** (Picked.) Precompute the held-out 1,024 eval rows
first, train a pilot on a 5k slice with sample dumps every 50 steps, gate the
long run on a visual response at step ~500. Then precompute the remaining
~64k rows (in parallel with continued training if disk allows). Optimises for
the first visible signal of life rather than total wall clock — matches the
user's failure mode (find out the conditioning is broken at hour 1, not hour 6).

**C. Online precompute (no caching).** Skip the precompute step entirely; run
VAE encode + ArcFace + Resampler inside the dataloader. Maximally simple. But
the photo-side preprocessing is GPU-bound and would serialize with the training
forward pass — throughput would tank. Rejected.

Picked **B**.

## Architecture

A `src/arkit_controlnet/cfm/` package — five focused modules plus a small
extension to `flame_render.py`. Each module has one concern, one clear
interface, and is independently testable.

```
cfm/
  __init__.py
  model.py           # frozen FLUX + bf16 InfuseNet + LoRA + control stems
  precompute.py      # photo latent + id tokens cache (resumable, atomic)
  text_embeds.py     # one-shot prompt encoder (T5 + CLIP)
  dataset.py         # CfmPairDataset — joins, renders control on the fly
  train.py           # CFM training loop, resumable, periodic eval dump
  eval.py            # held-out generation collage + ArcFace + MP-bs metrics
flame_render.py      # gains `render_landmark_aligned()` helper
```

### `cfm/model.py`

Concrete API (recorded by the gating spike verdict):

```python
@dataclass
class CfmModel:
    flux: FluxTransformer2DModel        # frozen
    infusenet: FluxControlNetModel      # peft-wrapped, LoRA-trainable
    trainable_params: list[Parameter]   # LoRA + x_embedder + controlnet_x_embedder

def build_model(
    flux_dev_bf16_path: str,
    infusenet_dir: str,
    lora_r: int = 8,
    lora_alpha: int = 8,
) -> CfmModel: ...

def velocity(
    model: CfmModel,
    z_t_packed: Tensor,           # (B, 1024, 64) for 512² latents
    sigma: Tensor,                # (B,) FLUX sigma (shift-warped)
    encoder_hidden_states: Tensor,  # (B, 8 + T5_seq, 4096) — id || t5
    pooled: Tensor,                 # (B, 768)
    txt_ids: Tensor,                # (8 + T5_seq, 3) zeros
    img_ids: Tensor,                # (1024, 3)
    control_packed: Tensor,         # (B, 1024, 64) — packed render latent
    guidance: Tensor,               # (B,) — 3.5 during training
) -> Tensor:  # (B, 1024, 64) predicted velocity
    ...
```

Internals mirror the spike (`spike_trainable_infusenet.py`) verbatim — same
LoRA target modules, same control-stem trainable list, same residual injection
into FLUX. Gradient checkpointing on both the FLUX trunk and InfuseNet.

### `cfm/precompute.py`

For each `image_sha256` in the working set:

1. Read FFHQ row → 1024² RGB photo.
2. Center-crop and resize to **512² using the cached MediaPipe face bbox plus
   25% margin** (matches InfiniteYou's training crop policy and aligns photo
   geometry with control geometry).
3. VAE-encode → `(16, 64, 64)` bf16 photo latent.
4. ArcFace via `insightface buffalo_l` → 512-d face embedding.
5. InfiniteYou Resampler over the embedding → `(8, 4096)` identity tokens.
6. Save to disk.

Storage layout (~14 GB total):

```
output/cfm_precompute/
  photo_latents/<sha>.pt   (16, 64, 64) bf16  → ~131 KB × 69,918 ≈ 9.2 GB
  id_tokens/<sha>.pt       (8, 4096) bf16     → ~64 KB × 69,918  ≈ 4.5 GB
  meta.parquet             rows: sha, crop_bbox_used, vae_ok, id_ok
```

Resumability: per-row skip-if-exists; one `meta.parquet` row appended per
completed sha (atomic via `.tmp` rename). Crash → resume reads `meta.parquet`
and skips done shas.

**Phasing:** the script takes a `--shas-file` argument so the Phase 0 call
(eval set only) and the Phase 2 call (everything else) share one entry point.

Test surface: round-trip a known row → assert photo latent shape, id token
shape, and that a second invocation skips the row.

### `cfm/text_embeds.py`

Encodes the single fixed prompt
`"a portrait photo of a person looking at the camera"` once using FLUX's T5-XXL
+ CLIP-L, saves to `output/cfm_precompute/text_embeds.pt` as a dict
`{"t5": (1, 512, 4096) bf16, "pooled": (1, 768) bf16}`. Idempotent — skip if
file exists. The dataset broadcasts a single copy across the batch (the prompt
never varies during training, the variation is in the control image).

Rationale for a fixed prompt: InfiniteYou's training set used one neutral
photographic prompt; the expression signal must come from the control channel,
not the text. A varying prompt would steal supervision the LoRA needs.

### `cfm/dataset.py`

`CfmPairDataset(split="train"|"eval")` — a `torch.utils.data.Dataset`.

Init:

```python
def __init__(self, split: str, *, eval_size: int = 1024, seed: int = 0):
    ri = pd.read_parquet("output/reverse_index/reverse_index.parquet")
    ri = ri[(ri.source == "ffhq") & ri.bs_detected]
    pc = pd.read_parquet("output/flame_pose_cache/pose_cache.parquet")
    pc = pc[pc.pose_detected]
    self.df = ri.merge(pc, on="image_sha256", how="inner")
    # Deterministic train/eval split by sha hash.
    bucket = self.df.image_sha256.apply(_hash_bucket_mod_1000)
    is_eval = bucket < (eval_size * 1000 // len(self.df))
    self.df = self.df[is_eval] if split == "eval" else self.df[~is_eval]
```

Per item (`__getitem__`):

1. Load `photo_latents/<sha>.pt` → `(16, 64, 64)` bf16. This is `z_0`.
2. Load `id_tokens/<sha>.pt` → `(8, 4096)` bf16.
3. Build 52-d ARKit vector from `bs_*` columns (via
   `flame_render.mediapipe_to_basis_vector`).
4. `flame_render.deform(arkit52)` → `(5023, 3)` verts.
5. Read `rotation` (9 floats → (3,3)) and `landmarks_xy` (956 floats → (478,2))
   from the joined row. Multiply landmarks by (W, H) of the rendered canvas
   (chosen to match the photo-latent crop: 512×512).
6. `flame_render.render_landmark_aligned(verts, rotation, landmarks_px, H=512, W=512)`
   → `(512, 512, 3)` uint8 RGB normals.
7. Normalize control RGB to `[-1, 1]` float, VAE-encode → `(16, 64, 64)` bf16
   control latent. (VAE encode of the control runs on GPU in the *trainer*,
   not in `__getitem__` — see "Data flow" below.)
8. Return dict: `{photo_latent, id_tokens, control_rgb_uint8, sha}`. The trainer
   batches and VAE-encodes the control on-device.

Why not cache control latents? The render is cheap (~5 ms) and the modality is
fixed; caching would burn ~9 GB to save a 5 ms op. Render-on-the-fly is the
right trade-off.

Test surface: one item returns the right tuple/shapes; the bs column order
matches `flame_render.BASIS_CHANNEL_NAMES`; a row missing from `pose_cache` is
filtered out; same `(sha, split, seed)` returns the same item across processes.

### `cfm/train.py`

CFM loop. Per step:

```
σ_t   = shift * sigmoid(N(0,1)) / (1 + (shift-1)*sigmoid(N(0,1)))  # shift=3.0
z_t   = (1 - σ) * z_0 + σ * ε                                       # FLUX flow
target = ε - z_0                                                     # CFM velocity
v     = velocity(model, pack(z_t), σ, eh, pooled, txt_ids, img_ids,
                 pack(VAE.encode(control_rgb)), guidance=3.5)
loss  = mse(v.float(), pack(target).float())
loss.backward(); clip_grad_norm(trainable, 1.0); opt.step()
```

Optimizer: `bitsandbytes.AdamW8bit`, LR 1e-4, constant with 200-step warmup,
weight decay 0.01 on LoRA only (no decay on stems).

Batch shape: B=1 with gradient accumulation to effective batch 4 (the spike
measured 30.7 GB at B=1; B=2 is tight). Steps = 20,000 — but two early hard
gates short-circuit the run if the conditioning is broken (see "Fail-fast
gates" below).

Logging: append a row to `exp_output/cfm_train/<run_id>/step_log.csv` per step
— `step, loss, grad_norm, lr, sigma_mean, wall_s`. Per-bucket loss (sigma
binned into 10 bins) every 50 steps.

Checkpointing: `latest.pt` every 500 steps (atomic via `.tmp` rename) carrying
`{lora_state, stem_state, opt_state, step, run_id}`. Resume just reads
`latest.pt`.

Periodic eval call: every 50 steps for the first 1k steps, every 200 after
that, run `cfm.eval.dump_samples(model, step)`.

### `cfm/eval.py`

Two responsibilities — held-out collage and scalar metrics.

**`dump_samples(model, step, n=8)`** — picks 8 fixed eval rows (the same ones
every call so visual progress is comparable across steps). For each row,
performs 25-step sampling with `FlowMatchEulerDiscreteScheduler` from
`z_T = ε ~ N(0,I)`, conditioned on
`(photo's id tokens, fixed prompt embeds, FLAME render at the row's
blendshapes/pose)`. Outputs a collage to
`exp_output/cfm_train/<run_id>/samples/step_{step:06d}.png` with 4 columns per
row: `[target photo | control render | id-only baseline | conditioned generation]`.

The id-only baseline is the same sampling call with `controlnet_conditioning_scale=0.0`
— it isolates whether the LoRA actually uses the control channel.

**`metrics(model, step) -> dict`** — every 500 steps, on the full 1,024-row
eval set:

- `id_cos`: ArcFace cos(generated, target). Target ≥ 0.60 (the zero-train
  InfuseNet baseline measured during the depth-spike).
- `bs_cos`: cos(MediaPipe-on-generated, intended ARKit-52). Target ≥ 0.30 with
  a clear margin over `bs_cos_neutral` (the same metric computed with a
  zero-blendshape control render).
- `bs_cos_neutral`: control sanity — passing the *neutral* render should
  produce a much lower `bs_cos` than passing the intended render. If they're
  equal, the control channel is being ignored.

Metrics appended to `metrics.csv`; collages dumped to disk for eyeball.

## flame_render extension

Add `render_landmark_aligned(verts, rotation, mp_landmarks_px, H, W) -> uint8`:
a sibling of `render()` that uses `landmark_align.aligned_pixels()` for the
per-vertex 2D position and otherwise runs the same painter's-algorithm normals
rasterizer. The existing `_render_aligned` inline helper in
`verify_flame_render.py` is the prototype — promote it.

`render()` (bbox-fit) stays as a fallback for callers without landmarks.

Test surface: a posed neutral mesh, given true MediaPipe landmarks from a
known photo, produces eye/mouth/chin pixel positions within ±3 px of the
target landmarks (the alignment is a least-squares fit on 6 points, so it
won't be exact at any single point — but the residual should be small).

## Fail-fast gates

Two checkpoints before the long run.

**Gate 1 — `step == 500` (≈30 min wall clock).** Open the sample collage at
step 500. If at least 3 of the 8 conditioned generations *visibly differ* from
their id-only baselines in a direction consistent with the control render —
proceed. If not, stop, write a failure report (which path is broken: VAE crop
mismatch? text embed wrong? LoRA target wrong? control latent normalization?),
and replan. Do not let it run another 19,500 steps in the hope it converges.

This gate is **manual** — Claude pings the user with the collage. Cheap and
high-signal.

**Gate 2 — `step == 2000`.** Eval metrics must show `bs_cos > bs_cos_neutral +
0.05` and `id_cos > 0.50`. If not, stop and report. (This gate is automatic.)

## Phasing

| Phase | What | Wall-clock estimate |
|---|---|---|
| 0 | Download dev-bf16 + `image_proj_model.bin` | ~30 min download |
| 1 | Build text embeds (one-shot) | <1 min |
| 2 | Precompute eval set (1,024 rows) | ~15 min |
| 3 | Pilot train: 5k-row subset, 500 steps, sample dumps every 50 | ~30 min |
| 4 | **Gate 1**: eyeball samples at step 500 | manual |
| 5 | If Gate 1 passes: kick off full 64k precompute *in parallel* with continued training | ~2 h precompute, runs alongside |
| 6 | Continue training to 20k steps total | ~12 h total |
| 7 | Final eval metrics + collage dump | ~10 min |

Total wall-clock to first visible signal: ~75 min (vs ~3 h in the sequential
plan).

## Data flow

```
reverse_index.bs_detected ─sha─┐
                               ├─▶ pose_cache (rotation, landmarks_xy)
ffhq_sha_index ─sha─▶ photo ───┤
                               ├─▶ photo_latent (cached)  ─┐
                               └─▶ id_tokens   (cached)   ─┼─▶ CFM step
bs_* ─arkit52─▶ verts ─render_landmark_aligned─▶ ctrl rgb ─VAE─▶ control_latent ─┘
text "a portrait photo..." ─▶ T5+CLIP (cached) ─▶ encoder_hidden_states (8 id || t5)
```

## Error handling

- Missing precomputed latent for a sha → `__getitem__` raises with the sha so
  the failure points at the precompute hole, not a generic shape mismatch.
- Non-finite ARKit vector or degenerate render → `flame_render` already raises
  `ValueError`; dataset logs the sha and substitutes a neutral render rather
  than aborting the epoch. (At most a handful of rows; the loss signal isn't
  hurt.)
- CUDA OOM at start → fail loud with a diagnostic; do not try to reduce batch
  silently. (Spike measured 30.7 GB at the spec'd config; OOM means something
  changed.)
- Resume from `latest.pt` is *required* to work; a smoke test in the plan
  verifies a 5-step run resumes correctly.

## Testing

- `model.py` — `build_model()` returns with LoRA params and exactly two stems
  trainable, all else frozen; `velocity()` output shape matches the input.
- `precompute.py` — round-trip one sha; second invocation skips it; missing
  weights raise with a clear pointer.
- `text_embeds.py` — encodes the prompt to the expected shapes; second
  invocation skips.
- `dataset.py` — one item has the right shapes; bs column order matches
  `BASIS_CHANNEL_NAMES`; rows with `pose_detected=False` are absent; eval and
  train splits are disjoint; deterministic across seeds.
- `flame_render.render_landmark_aligned` — fixture test: landmark residual <
  3 px on a known photo.
- `train.py` — a 5-step run on a 2-row fixture drives the loss finite and
  downward; the resumed run continues from the saved step.
- `eval.py` — `dump_samples` writes a collage of the expected shape; `metrics`
  returns the three keys with finite values.

## Out of scope

- The auxiliary blendshape-critic loss — deferred fallback if Gate 2 fails.
- Distillation into a fast student.
- SPSS / synthetic data augmentation.
- Live animation.
- Render modality B (depth) and C (lit + landmark overlay) — fixed to A unless
  Gate 1 fails on a modality-specific failure.

The deliverable at the end of this plan: a trained InfuseNet-LoRA + control
stems that follow a FLAME expression render at held-out, with logged ArcFace
identity retention and MediaPipe-blendshape control fidelity.
