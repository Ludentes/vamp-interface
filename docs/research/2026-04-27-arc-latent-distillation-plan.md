---
status: live
topic: demographic-pc-pipeline
---

# Plan: `arc_latent` distillation

Self-contained build spec. Goal: a small student network that takes Flux
VAE latents and outputs ArcFace R50 (insightface `buffalo_l`,
`w600k_r50.onnx`) embeddings, with cosine R² ≥ 0.95
vs teacher on held-out FFHQ. Once shipped, the entire demographic+glasses
ridge battery (already trained on top of ArcFace) becomes free in the
latent path. Out of scope: the slider trainer integration — that lands in
a separate v9 plan once this gate is passed.

## Why we're doing this

Recap from `2026-04-27-latent-native-classifier-distillation.md`. The
slider trainer's velocity-MSE loss systematically prefers global features
because L2 over a 16,000-cell latent gives global support ~80× more
loss-mass than a local 200-cell eye region. The principled fix is to
replace velocity-MSE with a metric-space loss (ArcFace identity, glasses
score, demographics) that has no spatial mass asymmetry. Doing that
without latent-native classifiers means VAE-decoding every step (PuLID /
DRaFT pattern). `arc_latent` removes the decode cost, makes 4+ metric
heads tractable per step, and unblocks Flavor 2 (anchor-only slider
training without paired corpora).

This plan is the foundational artifact: build the student, validate, ship.

## Decision summary

- **Teacher**: ArcFace R50 as packaged in insightface's `buffalo_l` pack
  (recognition head `w600k_r50.onnx`, ResNet-50 backbone, ArcFace loss,
  WebFace600K training data). This is the model every script in
  `src/demographic_pc/` actually loads via `FaceAnalysis(name="buffalo_l")`
  and the model every cached embedding in
  `output/demographic_pc/classifier_scores.parquet` was produced by.
  Earlier docs and the `project_vamp_measured_baseline` memory called this
  "IR101" — that was documentation drift; no IR101 has ever been loaded.
  Verified 2026-04-27 by inspecting the ONNX (input `(N,3,112,112)`,
  output `(1,512)`, 130 nodes — R50 territory). Frozen, 512-d L2-normed
  output, cosine metric.
- **Input data**: FFHQ 70k images. No new generation; encode through Flux
  VAE once.
- **Student input**: 16-channel Flux VAE latent at 64×64 (from 1024² → 16×64×64
  via Flux's f=8 + 2×2 patch). Confirm exact shape from existing cached
  pkls before training.
- **Student architecture**: ResNet-18 with first conv adapted to 16 input
  channels; output head 512-d, L2-normalized.
- **Loss**: 1 − cos(student(z), teacher(decode(z)).detach()). Optional MSE
  term in raw embedding space if cosine alone underconstrains norm.
- **Gate**: held-out cosine R² ≥ 0.95 against teacher. No partial credit;
  if we miss, diagnose before integrating.

## Pre-flight

Three things to verify before writing any training code:

1. **ArcFace teacher reproducibility.** The canonical loader is
   `InsightFaceClassifier` in `src/demographic_pc/classifiers.py:157`
   wrapping `FaceAnalysis(name="buffalo_l")`. Pipeline (audited
   2026-04-27): SCRFD detect → 5-point landmark similarity transform →
   112×112 BGR crop → `(x − 127.5) / 127.5` → R50 → L2-normalized 512-d.
   Largest detected face is selected; undetected faces return
   `embedding=None` (must be filtered, not zero-padded). **Do not** copy
   the shortcut in `fluxspace_primary_metrics.py:44-48` — it skips
   alignment and uses RGB instead of BGR; embeddings from that path are
   not comparable. Reference fixture exists at
   `tests/fixtures/arc_reference.npz` (10 FFHQ images, 6 detected at the
   canonical `det_thresh=0.5`); rebuild via
   `tests/build_arc_reference_fixture.py` and verify bit-identical
   embeddings on the same SHA-256-keyed images before any training
   code lands.

   **Detection-rate caveat (measured 2026-04-27):** at `det_thresh=0.5`
   (canonical), SCRFD drops 30–40% of FFHQ images with clearly visible
   frontal faces — they are detected at lower confidence and dropped by
   the threshold. Working hypothesis: FFHQ's tight dlib-aligned crops
   put faces at ~55–60% of frame area, outside SCRFD's WIDER-FACE-
   trained anchor priors. The threshold stays at 0.5 (consistency with
   every cached embedding and ridge classifier in the project), but
   **expected effective corpus size is ~42k of FFHQ-70k**, not 70k.
   Plenty for distillation; just budget for it. See
   `2026-04-27-arcface-detection-threshold.md` for the sweep.
2. **Flux VAE encode shape.** Pick one v3 corpus image, encode via the
   existing `comfy_flux.py` path, confirm output `(C, H, W)` exactly.
   Document. Match this in the dataset writer.
3. **Disk preflight.** 70k FFHQ images at 16×64×64 fp16 latent ≈ 130 KB
   per image → ~9 GB for FFHQ-70k latents. Plus original images at 1024²
   ≈ 90 GB if not already on disk. `df -h` first; pick a target volume
   with > 100 GB headroom (per `feedback_disk_preflight` memory).

If any of these surface anomalies, fix before proceeding — distillation
quality cannot exceed teacher fidelity, and ArcFace setup is the most
common silent failure point.

## Build steps

### 1. Distillation corpus

`scripts/build_arc_distill_corpus.py`. Resumable (skip-if-exists, per
`feedback_resumable_generation`). For each FFHQ image:

```python
img_1024 = load(...)
img_112  = arcface_preprocess(img_1024)        # crop+align+normalize
arc_emb  = teacher_arcface(img_112).detach()   # 512-d, fp32
img_512  = resize(img_1024, 512)                # Flux training resolution
z        = flux_vae.encode(img_512).latent.fp16
shard.write(image_id, z, arc_emb)
```

Output: sharded parquet or webdataset under
`datasets/arc_distill/ffhq/{train,val}/shard-NNNNN.tar`. 95/5 train/val
split by image_id hash (deterministic). Estimated wall time on one
RTX 4090: 5–8 hours including ArcFace alignment. Idempotent.

**Sanity gate before training**: pick 1000 random shards, decode `z`
back through VAE, confirm reconstructed image still produces an ArcFace
embedding within 0.95 cosine of the original-image embedding. If
reconstruction destroys ArcFace identity, we have a fundamental problem
(VAE is the bottleneck, not the student) — diagnose before continuing.

### 2. Pixel-input baseline

Before doing the latent-input student, train an *image-input* ResNet-18
to predict ArcFace embeddings from raw 112×112 face crops. This is a
control: if even an image-input student can't hit cosine R² > 0.95, the
task itself is harder than expected and the latent-input version has no
chance.

`scripts/train_arc_pixel_student.py`. Standard image-classification
recipe, just with cosine-similarity loss against teacher embeddings.
Same teacher, same val split.

**Gate**: cosine R² ≥ 0.97 on val. (Image input should outperform latent
input by a comfortable margin.) If this fails, stop and diagnose:
preprocessing mismatch, alignment, normalization. Do not proceed to
latent student.

### 3. Latent-input student

`src/demographic_pc/arc_latent.py`. ResNet-18 with stem replaced:

```python
class ArcLatent(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(16, 64, kernel_size=3, padding=1)  # 16-ch latent
        self.body = resnet18(num_classes=0).layer1...layer4       # untouched
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(512, 512)
    def forward(self, z):
        return F.normalize(self.head(self.pool(self.body(self.stem(z)))).squeeze(-1).squeeze(-1))
```

Note: 64×64 latent input → ResNet-18 ends up at ~2×2 spatial after layer4
(plenty for 512-d output). If latents are 32×32 (patched), output spatial
shrinks to 1×1; still fine.

Training: AdamW, lr 1e-3 cosine→1e-5, batch 256, 50 epochs of
FFHQ-train. Loss = 1 − cos(student(z), teacher_emb.detach()). Augment
latents with small Gaussian noise (σ=0.02) for robustness. No flow-matching
noise corruption at this stage — pure clean-latent distillation.

Wall time estimate: ~12 hours on one RTX 4090. Checkpoint every 2 epochs.

**Gate**: held-out cosine R² ≥ 0.95.

If between 0.85 and 0.95, try: bigger student (ResNet-34), more epochs,
add MSE term to constrain norm. If below 0.85, diagnose: is the model
seeing identity-discriminative features in the latent at all? Probe with
a frozen-feature ID classification task (LFW-style). The pixel baseline
told us the task is well-posed; if latent fails far below pixel, the
information just isn't reachable in the latent without VAE-decoder
features (which would push us toward LPL-style decoder-feature probes).

### 4. Validation battery

`scripts/validate_arc_latent.py` runs once the student passes the cosine
gate:

- **Held-out cosine R²** (FFHQ-val, 3500 images). Target ≥ 0.95.
- **Identity-pair discrimination.** Use existing v3.1 corpus pairs. Score
  same-identity pairs vs different-identity pairs in `arc_latent` space.
  AUC should match teacher within 2 percentage points.
- **Demographic ridge transfer.** Take the trained
  `ridge_glasses(arc_emb)`, `ridge_age(arc_emb)`, etc. Apply to
  `arc_latent(z)` outputs on val set. Pearson r vs ground truth labels
  should match teacher within 0.05.
- **Robustness to flow-matching noise.** For t ∈ {0.0, 0.1, 0.2, 0.3,
  0.5, 0.7}, compute `x_t = (1-t)·z + t·ε` and run student on x_t.
  Cosine R² vs clean teacher embedding will degrade with t. Plot the
  curve. This tells us the safe t-range for slider training. Expectation:
  R² stays above 0.9 up to t ≈ 0.3, drops sharply past 0.5.

The fourth check is the most important for downstream use — it
determines whether v9 trainer can use `arc_latent` at all timesteps or
has to restrict to low t.

### 5. Deliverable

```
models/arc_latent/
  arc_latent_v1.safetensors      # student weights
  arc_latent_v1.json             # arch config, training summary
  validation_v1.json             # cosine R², AUC, ridge transfer, t-curve
  README.md                      # how to load, expected I/O shapes, gotchas
```

Plus a clean inference function in `src/demographic_pc/arc_latent.py`:

```python
def load_arc_latent(path="models/arc_latent/arc_latent_v1.safetensors"): ...
def arc_latent(z: Tensor) -> Tensor:  # (B, 16, H, W) -> (B, 512), L2-normed
```

Done = the validation battery passes and the inference function works
in a fresh notebook from a clean import.

## Anti-goals

Things this plan deliberately does NOT do:

- **No swap to a different ArcFace backbone.** CVLFace IR101
  (`minchul/cvlface_arcface_ir101_webface4m`) was once recommended in
  `2026-04-07-face-recognition-embeddings.md` and accidentally propagated
  into the memory as if it were in production. It never was. Distillation
  targets the actual production teacher (R50 buffalo_l). Switching is a
  separate, expensive thread that re-trains every ridge.

- **No t-conditioned student.** "Diffusion classifier" style classifiers
  that condition on `t` are a research direction; we measure the noise
  curve and decide afterward. Adds complexity and doubles training time.
- **No SigLIP or blendshape distillation.** Same recipe applies but
  separate plans, separate gates.
- **No slider trainer integration.** `arc_latent` ships as a standalone
  artifact with documented contracts. v9 trainer plan consumes it later.
- **No new ArcFace training.** We're distilling the existing R50
  (`buffalo_l`), not finetuning it and not switching to a different
  backbone (CVLFace IR101 was once recommended in
  `2026-04-07-face-recognition-embeddings.md` but never installed; the
  whole ridge battery downstream is R50, so swapping now would invalidate
  it for ~1pp IJB-C gain we don't need). If the teacher's demographic
  biases bother us, that's a separate thread.

## Risks and mitigations

| Risk | Probability | Mitigation |
|---|---|---|
| Latent doesn't carry enough ID info → R² < 0.85 | low–medium | Pixel baseline gates this. If latent fails far below pixel, switch to LPL-style (use VAE decoder's first 1–2 blocks as a feature extractor instead of distilling cold). |
| Flux VAE encoding mismatch (wrong norms, wrong scaling) | medium | Pre-flight #2 catches this. Use the same `flux_vae.encode()` path our slider training uses. |
| ArcFace alignment pipeline differs from existing project usage | medium | Pre-flight #1: re-run on 10 reference images and confirm embeddings match a stored fixture. |
| Disk fills mid-corpus build | medium | Disk preflight before Step 1; resumable script. |
| Student overfits to FFHQ demographic distribution | low | Validation battery includes our v3.1 corpus (different demographics); flag if AUC gap >5pp vs FFHQ-val. |

## Estimated wall time

| Step | Wall time | Blocker? |
|---|---|---|
| Pre-flight (1, 2, 3) | 1 hour | yes — gate everything |
| Build corpus (Step 1) | 5–8 GPU-hours | yes |
| Reconstruction sanity (in Step 1) | 30 min | yes |
| Pixel baseline (Step 2) | 4 GPU-hours | yes (gate) |
| Latent student (Step 3) | 12 GPU-hours | yes |
| Validation battery (Step 4) | 1 hour | yes (gate) |
| Packaging (Step 5) | 1 hour | no |

Total: ~1 GPU-day of work plus a working day of script-writing and
inspection. Independent of any v7/v8 outcome — start whenever a GPU
is free.

## Open questions you'll hit

1. **Which Flux VAE checkpoint exactly?** Krea-dev's VAE specifically. Confirm
   it's the same one used by the slider trainer. If the trainer ever
   switches VAE, `arc_latent` invalidates.
2. **Latent normalization.** Flux applies a scaling factor to latents on
   encode/decode (the 0.3611 / shift_factor business). The student must
   see latents in exactly the same scaled state the diffusion process
   sees them in. Check ai-toolkit's data pipeline; mirror it in the
   distillation corpus writer.
3. **What if FFHQ teacher embeddings are biased?** They will be — ArcFace
   training data underrepresents children, dark skin in some setups. The
   student inherits this. Acceptable for v1; flag in README.

## What this unblocks

Once `arc_latent_v1` ships:

- v9 trainer plan can be written (Flavor 1: pairs + metric loss at low t).
- `siglip_latent` distillation plan can copy this template and run in
  parallel (different teacher, same recipe, same FFHQ corpus reusable).
- The whole demographic+glasses ridge classifier battery (already trained
  on top of teacher ArcFace) becomes available in the latent path with
  zero extra training — we just need to validate transfer quality (Step
  4's third check).
- Anchor-only Flavor 2 trainer becomes feasible to design.

End of plan.
