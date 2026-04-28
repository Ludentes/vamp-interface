# arc_distill — ArcFace identity in Flux latent space

A frozen-backbone student that maps Flux VAE latents `(16, 64, 64)` to
512-d L2-normalised ArcFace identity embeddings, **without rendering**.

The teacher is InsightFace `buffalo_l` (`w600k_r50.onnx`, IResNet50).
The student wraps the same IResNet50 weights frozen — only a swapped stem
(and IResNet50 layer-1) train. This is a transfer-learning shape, not a
from-scratch distillation: the student inherits ArcFace's identity manifold
and only learns to translate VAE-latent input into a stem-output that the
deep backbone already knows how to read.

**Variant shipped here:** `latent_a2_full_native_shallow` (epoch 19).
Trained on 26,108 FFHQ rows VAE-encoded at 512² → `(16, 64, 64)` bf16,
SHA-prefix train/val split (val n=1514).

## Files

| File | Purpose |
|---|---|
| `checkpoint.pt` | trained weights — gitignored, 179 MB. Re-pull from shard via the bat under `scripts/` if missing. |
| `eval.json` | training-time eval on val split (Layer 1.1 metrics). |
| `validation_report.json` | full Layer 1+2 validation report from `validate_as_loss.py`. |

## Headline numbers

| Metric | Value |
|---|---|
| Teacher cosine, mean (val n=1514) | **0.881** |
| Teacher cosine, median | 0.939 |
| Frac > 0.9 | 87.6% |
| p05 | 0.063 |
| min | -0.148 |
| Hard-negative AUC (avg over 5 perturbations) | **≥ 0.9999** |
| Backprop into latent works | ✓ (loss descends 1.08→0.99 in 100 SGD steps) |

Headline limitation: long left tail. p05=0.063 means ~5% of val rows are
catastrophic predictions. Diagnostic ([writeup](../../docs/research/2026-04-30-arcface-frozen-adapter.md))
shows these cluster on faces with **beards, hats, headwear, sunglasses,
costume hats** — i.e. localization failures, not distribution mismatch.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  AdapterStudent("latent_a2_full_native_shallow")                 │
│  total params:  43.6 M   trainable: 0.42 M (≈1%)                 │
└──────────────────────────────────────────────────────────────────┘
   input
   (B, 16, 64, 64)  bf16/fp32 Flux VAE latent at 512² resolution
        │
        ▼  ─────  trainable  ──────────────────────────────────────
  LatentStemFull64Native                       0.27 M params
    Conv2d(16 → 64, k=3, p=1, bias=False)
    BatchNorm2d(64) + PReLU(64)
    ConvTranspose2d(64 → 64, k=8, s=2, p=3)   (B, 64, 128, 128)
    crop center 112×112                        (B, 64, 112, 112)
    BatchNorm2d(64) + PReLU(64)
        │
        ▼  ─────  trainable  ──────────────────────────────────────
  IResNet50.layer1 (3 residual blocks)         0.15 M trainable
    BN running stats stay frozen (eval-mode);
    only affine + conv weights train.
        │
        ▼  ─────  frozen ─────────────────────────────────────────
  IResNet50.layer2 + layer3 + layer4 + head    43.0 M frozen
        │
        ▼
  L2 normalize → (B, 512) ArcFace-compatible embedding
```

Stem replaces the native `Conv_0(3, 64, 3x3, s=1)` head of IResNet50 (which
would expect a 112² RGB image) with a stride-2 ConvTranspose path that maps
the 64×64 latent up to 112×112 with no spatial information thrown away. The
stem's PReLU+BN replaces the IResNet50 native PRelu_1+BN_2 (bypassed via the
graph rewrite in [backbone.py](../../src/arc_distill/backbone.py)).

The "A2 / shallow" prefix means **layer-1 of the IResNet50 (the first three
residual blocks) is also trainable**, with a lower learning rate (1e-4 vs
1e-3 for the stem). BN running statistics in layer-1 are kept in eval mode
so they don't drift on a 24k-row dataset.

## Capabilities

**What this model is for.**
Use as an identity-preservation loss term inside slider/LoRA training, where
the gradient is averaged over batches and only the *direction* needs to track
ArcFace. Layer 1+2 validation (see [`validation_report.json`](validation_report.json)
and [the writeup](../../docs/research/2026-04-30-arcface-frozen-adapter.md))
indicates this is plausible. Layer 3 (slider A/B) is the operating-conditions
test; spec at [2026-04-30-layer3-slider-ab-spec.md](../../docs/research/2026-04-30-layer3-slider-ab-spec.md).

**What this model is *not* validated for.**
- **Inference-time classifier guidance** on a single sample. The 0.119
  off-direction component biases every step; errors don't average across
  a batch. See "Possible uses" section of the writeup.
- **Identity verification on multi-shot benchmarks** (LFW / CFP-FP).
  Validation here uses augmentation pairs on FFHQ singletons; we have no
  multi-shot eval set.
- **Partially-noised latents** at `t > 0` of the diffusion schedule. Trained
  on clean VAE-encoded latents; behaviour off-manifold is untested.
- **Identities the teacher fails on** — the long left tail (5% of val rows).
  Faces with hats / sunglasses / dense beards localize poorly through the
  stem.

## Sample usage

### Loading

```python
import torch
from arc_distill.adapter import AdapterStudent

device = torch.device("cuda")
model = AdapterStudent("latent_a2_full_native_shallow").to(device)
ck = torch.load("models/arc_distill/checkpoint.pt", map_location=device,
                weights_only=False)
model.load_state_dict(ck["model"])
model.eval()
# model expects (B, 16, 64, 64) Flux VAE latents and returns
# (B, 512) L2-normalised embeddings.
```

The constructor needs the buffalo_l ONNX file for the frozen backbone; default
path is `~/.insightface/models/buffalo_l/w600k_r50.onnx`. Override via
`AdapterStudent(variant, onnx_path=...)`.

### As an identity-preservation loss

```python
# z_anchor, z_edited: (B, 16, 64, 64) clean Flux latents
emb_anchor = model(z_anchor.detach())
emb_edited = model(z_edited)            # gradient flows through z_edited
id_loss = 1.0 - (emb_anchor * emb_edited).sum(dim=-1).mean()
total_loss = base_loss + lam * id_loss  # try lam ∈ {0.5, 1.0, 2.0}
total_loss.backward()
```

A common pitfall: `model.eval()` is correct (the frozen backbone's BN running
stats must not drift), but the stem and layer-1 still produce gradients —
torch's `eval()` only changes BN/dropout behaviour, not autograd.

### As a similarity scorer for retrieval

```python
emb = model(latents_batch)              # (N, 512), L2-normalised
sim = emb @ emb.T                       # cosine similarity matrix
top5 = sim.topk(k=6, dim=-1).indices[:, 1:]  # exclude self
```

### Probe what the teacher would say without rendering

Use as a fast no-render proxy for ArcFace cosine. ~1.3 s per 1500 latents on
RTX 3090. Compare two latents:

```python
emb1 = model(z1)
emb2 = model(z2)
cos = (emb1 * emb2).sum(dim=-1)         # (B,) — proxy for arcface_cos(render(z1), render(z2))
# expected residual to true teacher: ~0.119 RMS at 0.881 mean cos
```

## Reproducing & verification

Run the full validation suite:

```bash
python -m arc_distill.validate_as_loss \
    --variant latent_a2_full_native_shallow \
    --checkpoint models/arc_distill/checkpoint.pt \
    --compact <path-to-compact.pt> \
    --face-attrs <path-to-face_attrs.pt> \
    --out-json /tmp/validation_report.json
```

`compact.pt` and `face_attrs.pt` live on shard at
`C:\arc_distill\arc_full_latent\` (4.5 GB combined, not in repo). The shard
runner is [`scripts/run_validate_full_native_a2.bat`](../../scripts/run_validate_full_native_a2.bat).

## Lineage

- Pixel-A baseline (0.96): [src/arc_distill/train_pixel.py](../../src/arc_distill/train_pixel.py)
  trains on 112² RGB FFHQ crops; validates the frozen-backbone shape.
- Latent variants (the journey): see the run-log table in
  [the writeup](../../docs/research/2026-04-30-arcface-frozen-adapter.md).
  TL;DR: pool→0.30, native→0.87, A2-native→0.88, crop→0.53, RoI→0.69,
  similarity-warped 14×14→abandoned (interpolation-bound).
- Lessons: [2026-04-30-arc-distill-lessons.md](../../docs/research/2026-04-30-arc-distill-lessons.md).
