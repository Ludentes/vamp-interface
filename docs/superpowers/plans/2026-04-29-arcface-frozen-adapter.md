# ArcFace Frozen-Backbone Adapter Implementation Plan

> Supersedes [`2026-04-28-arcface-pixel-baseline.md`](2026-04-28-arcface-pixel-baseline.md).
> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an ArcFace-compatible identity encoder for Flux VAE latents by *reusing* the frozen buffalo_l R50 backbone and training only a small input adapter. Gate: held-out cosine ≥ 0.85 to teacher embedding on a SHA-prefix val split — the realistic ceiling for this transfer-learning approach, calibrated by a pixel sanity spike that verifies the freeze-and-adapt premise.

**Architecture:** Frozen `w600k_r50` ONNX is **IResNet50** (not torchvision ResNet-50). Its native stem is just `Conv(3,64,3×3,stride=1) + PReLU + BatchNorm2d` producing `(64, 112, 112)` (no maxpool, no stride-2 in the stem); downsampling happens inside layer-1's first residual block. The frozen headless backbone consumes `(64, 112, 112)` activations and outputs un-normalised 512-d. We replace the stem with a trainable module that produces `(64, 112, 112)` from one of two inputs: aligned 112² RGB (Pixel-A, sanity) or 14×14×16 Flux VAE latents (Latent-A, the goal). Loss is cosine distance to the precomputed teacher embedding; the rest of IResNet50 stays bit-identical to the teacher.

**Tech Stack:** PyTorch, ONNX→PyTorch port (or onnxruntime forward with autograd-tracked stem), torchvision (for ResNet block defs if porting), pyarrow + Pillow for data, the existing `src/arc_distill/` package (extend, don't replace).

---

## Why this plan replaces the previous one

The 2026-04-28 plan trained a fresh ImageNet-pretrained ResNet-18 to distill ArcFace embeddings from raw FFHQ pixels. It failed the gate (val cos 0.377 unaligned, 0.323 aligned). On square-1 verification ([scripts/check_arc_oracle.py](../../../scripts/check_arc_oracle.py)) all five sample rows showed `cos(teacher, oracle_aligned) = 1.0000`, `bbox_eq=True`, `kps_eq=True` — the pipeline is byte-perfect, so the failure is not a setup bug. It is the wrong test:

- **Question we wanted to answer:** can we transfer ArcFace's days of identity training to a latent-input student?
- **Question the previous plan answered:** can a randomly-initialised-from-ImageNet ResNet-18 learn ArcFace's identity manifold from scratch in 4 minutes? (Answer: ~0.4 cosine.)

The two are very different. We have a frozen 25M-parameter network that already encodes identity; the right move is to keep it frozen and learn only the input mapping. That's what this plan builds.

The headline mistakes in the previous plan, for the record:

1. **Fresh student backbone.** ResNet-18 ImageNet has no face-identity geometry baked in. We threw away the teacher's weights and asked a new network to rediscover them.
2. **Gate threshold pulled from production face-recognition norms.** ArcFace cos > 0.9 is the bar for "is this the same person." It is not the bar a 4-minute distillation can clear, and it is not what we actually need (we need *identity-preserving features*, which a lower cosine still provides).
3. **Pixel-vs-latent comparison was framed as a serial validation gate**, when the actual scientific question is the *delta* between matched-compute pixel and latent students.

This plan replaces all three.

---

## What "transfer ArcFace knowledge" actually means here

Buffalo_l recognition is `w600k_r50.onnx`: IResNet50, input `(3, 112, 112)` BGR with `(x - 127.5) / 127.5` normalisation, output 512-d L2-normalised. We split it conceptually:

```
input → Conv(3×3,s=1,3→64) → PReLU → BN → [layer 1..4] → BN → Dropout → fc → BN → L2-norm
        \____________________________/                                  \________________/
              STEM (replace, ~1.7K params)                                  HEAD (frozen)
```

(Confirmed via `onnx.shape_inference`: stem output is `(64, 112, 112)`; stage 1's first block does the 112→56 downsample.)

The "days of training" live in layers 1–4 + fc. They embed faces into the 512-d identity manifold. The stem (Conv 3×3 + PReLU + BN) is feature pre-shaping. If we can produce a `(64, 112, 112)` activation matching what layer 1's input distribution expects, the rest of the network gives us teacher embeddings unchanged.

The two students this plan trains differ only in their stem:

- **Pixel-A stem:** input `(3, 112, 112)` aligned RGB float, output `(64, 112, 112)`. Fresh `Conv2d(3, 64, 3, stride=1, padding=1) + PReLU(64) + BatchNorm2d(64)`. Trainable. Same shape as the IResNet stem.
- **Latent-A stem:** input `(16, 14, 14)` Flux VAE latent, output `(64, 112, 112)`. Two design candidates:
  - *upsample-and-conv:* `Upsample(112, mode='bilinear') → Conv2d(16, 64, 3, s=1, p=1) → PReLU → BN`.
  - *native-stride:* `Conv2d(16, 64, 3, p=1) → PReLU → BN → ConvTranspose2d(64, 64, 8, s=8) → PReLU → BN` to reach 112×112 directly via factor-8 upsample.
  Both <100K trainable parameters.

Pixel-A exists *only to calibrate the gate*. If freeze-and-adapt is well-formed, Pixel-A should converge to cos≈1.0 quickly — at the limit, a perfectly-trained stem reproduces what conv1 did, and the head reproduces the teacher exactly. If Pixel-A plateaus well below 1.0, our framing of "stages 1–4 + fc are identity-sufficient given any reasonable 64-channel feature map" is wrong and we need to revisit the freeze decision (e.g. unfreeze stage 1).

Latent-A is the actual deliverable. Its plateau tells us how much identity geometry survives the Flux VAE bottleneck. The gap `cos(Pixel-A) − cos(Latent-A)` is the meaningful "VAE identity tax" that step 3 of the latent-native classifier roadmap was always trying to measure.

---

## File Structure

Extend the existing `src/arc_distill/` package — keep `dataset.py` (SHA-prefix split is reusable), retire the old `model.py` / `train_pixel.py` / `eval_pixel.py` paths in favour of new modules. Old files stay on disk for git history but are not referenced by the new training scripts.

```
src/arc_distill/
  __init__.py                 # existing
  dataset.py                  # existing — reuse CompactFFHQDataset + is_held_out
  prepare_compact.py          # existing — keep --align mode (Pixel-A input)
  prepare_compact_latent.py   # NEW — pack latent_bf16 + arcface_fp32 into compact_latent.pt
  r50_backbone.py             # NEW — frozen ArcFace R50 PyTorch port loading from ONNX or .pt
  stems.py                    # NEW — PixelStem (3→64), LatentStemUpsample, LatentStemNative
  model_adapter.py            # NEW — AdapterStudent(stem, frozen_r50) wrapper, F.normalize at output
  train_adapter.py            # NEW — single training script, --variant {pixel_a,latent_a_up,latent_a_native}
  eval_adapter.py             # NEW — gate metrics + per-stem comparison
tests/
  test_arc_distill.py         # extend with stem shape tests, frozen-grad tests, adapter-output L2 tests
scripts/
  run_arc_adapter_pixel.bat   # NEW — Pixel-A run on Windows 3090
  run_arc_adapter_latent.bat  # NEW — Latent-A run on Windows 3090
```

External (remote, gitignored, on Windows 3090):
- `C:\arc_distill\ffhq_parquet\data\train-*.parquet` — present
- `C:\arc_distill\encoded\train-*.pt` — present (latent_bf16 + arcface_fp32)
- `C:\arc_distill\arc_pixel_aligned\compact.pt` — present (Pixel-A input, ~3.2 GB aligned 112² uint8)
- `C:\arc_distill\arc_latent\compact_latent.pt` — NEW (latent_bf16 + arcface_fp32, ~3.5 GB)
- `C:\arc_distill\arc_adapter\{pixel_a,latent_a_up,latent_a_native}\` — output dirs

---

### Task 1: Frozen R50 backbone wrapper

Extract everything from buffalo_l's `w600k_r50.onnx` *except* the input stem, hold it as a frozen PyTorch module that maps `(B, 64, 56, 56)` → `(B, 512)` un-normalised, with sanity tests that match teacher embeddings to floating-point equality on a held-out fixture.

**Files:**
- Create: `src/arc_distill/r50_backbone.py`
- Modify: `tests/test_arc_distill.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_arc_distill.py — append
import numpy as np
import torch
from arc_distill.r50_backbone import FrozenR50Headless, run_full_r50_from_stem_output

def test_headless_output_shape(tmp_path):
    backbone = FrozenR50Headless.from_artifacts(tmp_path / "fixtures")
    x = torch.randn(2, 64, 56, 56)
    z = backbone(x)
    assert z.shape == (2, 512)
    # not yet L2-normalised at this layer
    norms = z.norm(dim=-1)
    assert ((norms - 1.0).abs() > 0.01).any()

def test_headless_no_grad_through_backbone():
    backbone = FrozenR50Headless.from_artifacts(...)
    backbone.eval()
    for p in backbone.parameters():
        assert not p.requires_grad
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/test_arc_distill.py::test_headless_output_shape -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'arc_distill.r50_backbone'`.

- [ ] **Step 3: Implement `FrozenR50Headless`**

Two implementation paths, decide on Day 1:

- *Path A (preferred): port to PyTorch.* Use `torchvision.models.resnet50` block definitions, copy weights from the ONNX initialisers (parse `w600k_r50.onnx` with `onnx.load` + numpy_helper). The ONNX op names map cleanly onto torchvision layer names for the standard ResNet-50 blocks. Strip conv1+bn1+relu+maxpool from the front and `fc` is kept (`embedding` head, 2048→512). Save the converted state dict to `output/arc_distill/r50_headless.pt` so we don't re-parse ONNX every run.
- *Path B (fallback): onnxruntime + cut the graph.* Use `onnx.utils.extract_model` to slice off conv1/bn1/relu/maxpool, run the residual graph in onnxruntime with the input being our trainable stem's output. Backprop through the ONNX graph requires `torch_ort` or running the ONNX as a black box and only optimising the stem — fine for our use case (the backbone is frozen) but locks us out of the Latent-A2 fallback (partial unfreeze).

Pick Path A. Code skeleton:

```python
# src/arc_distill/r50_backbone.py
from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn
from torchvision.models.resnet import Bottleneck

class FrozenR50Headless(nn.Module):
    """ResNet-50 with conv1+bn1+relu+maxpool stripped. Input (B,64,56,56). Output (B,512), un-normalised."""

    def __init__(self):
        super().__init__()
        self.layer1 = self._make_layer(64,  64,  3, stride=1)
        self.layer2 = self._make_layer(256, 128, 4, stride=2)
        self.layer3 = self._make_layer(512, 256, 6, stride=2)
        self.layer4 = self._make_layer(1024,512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, 512)

    @staticmethod
    def _make_layer(in_planes, planes, blocks, stride):
        layers = [Bottleneck(in_planes, planes, stride=stride,
                             downsample=nn.Sequential(
                                 nn.Conv2d(in_planes, planes*Bottleneck.expansion, 1, stride=stride, bias=False),
                                 nn.BatchNorm2d(planes*Bottleneck.expansion)))]
        for _ in range(1, blocks):
            layers.append(Bottleneck(planes*Bottleneck.expansion, planes))
        return nn.Sequential(*layers)

    def forward(self, x):  # (B, 64, 56, 56)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

    @classmethod
    def from_onnx(cls, onnx_path: Path) -> "FrozenR50Headless":
        # Parse w600k_r50.onnx initialisers, build name→tensor dict, copy into a fresh
        # cls() instance via state_dict mapping. ArcFace ONNX uses standard ResNet-50
        # naming so the mapping is mechanical.
        raise NotImplementedError("implement onnx → state_dict mapping")

    @classmethod
    def from_state_dict(cls, ckpt_path: Path) -> "FrozenR50Headless":
        m = cls()
        m.load_state_dict(torch.load(ckpt_path, weights_only=True))
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        return m
```

- [ ] **Step 4: Verify equivalence to ONNX teacher**

Build a fixture: take 16 known FFHQ images, run them through the *original* `app.get(bgr)` pipeline → record `f.normed_embedding`. Then for the same 16 images, manually run conv1+bn1+relu+maxpool (the stem we're stripping out) using the ONNX initialisers, feed result into our `FrozenR50Headless`, L2-normalise. The two embeddings must match to `atol=1e-4`. If they don't, we mis-mapped the ONNX initialisers.

```python
# tests/test_arc_distill.py — append
def test_headless_matches_teacher_on_fixture(arc_oracle_fixture):
    """fixture: list of (image_path, teacher_embedding_512)"""
    backbone = FrozenR50Headless.from_state_dict("output/arc_distill/r50_headless.pt")
    onnx_stem = ONNXStemOnly("path/to/w600k_r50.onnx")  # conv1+bn1+relu+maxpool only
    for img_path, teacher in arc_oracle_fixture:
        bgr_aligned = align_to_112(img_path)             # uses our existing pipeline
        stem_out = onnx_stem(bgr_aligned)                # (1, 64, 56, 56)
        z = backbone(torch.from_numpy(stem_out))
        z = z / z.norm(dim=-1, keepdim=True)
        cos = (z[0] * torch.from_numpy(teacher)).sum().item()
        assert cos > 0.9999, f"backbone mismatch: cos={cos}"
```

This test is the gate for Task 1 — it is the equivalent of the square-1 verification we just ran. If it doesn't hit ≥0.9999 we don't proceed.

- [ ] **Step 5: Commit**

```bash
git add src/arc_distill/r50_backbone.py tests/test_arc_distill.py
git commit -m "feat(arc_distill): frozen headless R50 backbone with ONNX-equivalence test"
```

---

### Task 2: Trainable stems

**Files:**
- Create: `src/arc_distill/stems.py`
- Modify: `tests/test_arc_distill.py`

- [ ] **Step 1: Write the failing test**

```python
import torch
from arc_distill.stems import PixelStem, LatentStemUpsample, LatentStemNative

def test_pixel_stem_shape():
    s = PixelStem()
    y = s(torch.randn(2, 3, 112, 112))
    assert y.shape == (2, 64, 56, 56)

def test_latent_stem_upsample_shape():
    s = LatentStemUpsample()
    y = s(torch.randn(2, 16, 14, 14))
    assert y.shape == (2, 64, 56, 56)

def test_latent_stem_native_shape():
    s = LatentStemNative()
    y = s(torch.randn(2, 16, 14, 14))
    assert y.shape == (2, 64, 56, 56)
```

- [ ] **Step 2: Run, expect FAIL (no module).**

- [ ] **Step 3: Implement stems**

```python
# src/arc_distill/stems.py
import torch.nn as nn
import torch.nn.functional as F

class PixelStem(nn.Module):
    """Identical shape to ResNet-50 conv1+bn1+relu+maxpool, fresh init."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    def forward(self, x):
        return self.maxpool(F.relu(self.bn(self.conv(x)), inplace=True))

class LatentStemUpsample(nn.Module):
    """Bilinear upsample 14→112, then a conv stem identical in shape to PixelStem."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(16, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    def forward(self, x):
        x = F.interpolate(x, size=(112, 112), mode="bilinear", align_corners=False)
        return self.maxpool(F.relu(self.bn(self.conv(x)), inplace=True))

class LatentStemNative(nn.Module):
    """Native-stride stem for 14×14×16 → 56×56×64 without upsampling pixels."""
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(16, 64, 3, padding=1, bias=False)   # (B,64,14,14)
        self.b1 = nn.BatchNorm2d(64)
        self.up = nn.ConvTranspose2d(64, 64, 4, stride=4)        # (B,64,56,56)
        self.b2 = nn.BatchNorm2d(64)
    def forward(self, x):
        x = F.relu(self.b1(self.c1(x)), inplace=True)
        return F.relu(self.b2(self.up(x)), inplace=True)
```

- [ ] **Step 4: Run, expect PASS.**

- [ ] **Step 5: Commit**

```bash
git add src/arc_distill/stems.py tests/test_arc_distill.py
git commit -m "feat(arc_distill): trainable Pixel-A and Latent-A stems"
```

---

### Task 3: Adapter wrapper

**Files:**
- Create: `src/arc_distill/model_adapter.py`
- Modify: `tests/test_arc_distill.py`

- [ ] **Step 1: Failing test**

```python
def test_adapter_only_stem_has_grad():
    from arc_distill.model_adapter import AdapterStudent
    from arc_distill.stems import PixelStem
    backbone = FrozenR50Headless.from_state_dict("...")
    stem = PixelStem()
    m = AdapterStudent(stem, backbone)
    x = torch.randn(2, 3, 112, 112)
    z = m(x)
    assert z.shape == (2, 512)
    assert torch.allclose(z.norm(dim=-1), torch.ones(2), atol=1e-5)
    z.sum().backward()
    for p in stem.parameters():    assert p.grad is not None
    for p in backbone.parameters(): assert p.grad is None
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement**

```python
# src/arc_distill/model_adapter.py
import torch.nn as nn, torch.nn.functional as F

class AdapterStudent(nn.Module):
    def __init__(self, stem: nn.Module, frozen_backbone: nn.Module):
        super().__init__()
        self.stem = stem
        self.backbone = frozen_backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)
    def forward(self, x):
        z = self.backbone(self.stem(x))
        return F.normalize(z, dim=-1)
```

- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit.**

---

### Task 4: Pixel-A training script

**Files:**
- Create: `src/arc_distill/train_adapter.py`
- Create: `scripts/run_arc_adapter_pixel.bat`

Design notes (no separate tests for the script — exercise via Task 6 smoke run):

- CLI: `--variant {pixel_a,latent_a_up,latent_a_native} --compact PATH --backbone-ckpt PATH --out-dir DIR --epochs N --batch-size N --lr 1e-3 --workers 0 --device cuda --smoke`.
- For `pixel_a`: load `compact.pt` (aligned 112² uint8) via existing `CompactFFHQDataset`, normalise with **ArcFace stats** `(x*255 - 127.5) / 127.5` not ImageNet stats — the frozen backbone expects ArcFace-normalised input. **Important:** dataset already returns float32 ÷255; train_adapter undoes the ÷255 then applies ArcFace norm. Update `CompactFFHQDataset` (or wrap it) to skip normalisation when `variant.startswith("pixel_a")` → just `(uint8 → float32 - 127.5) / 127.5`.
- For latent variants: load `compact_latent.pt` (Task 5) which carries `latent_bf16 (N,16,14,14)` plus `arcface_fp32` plus `shas`. No image normalisation — VAE latents are already in their native scale.
- Loss: `1 - cos(student, target)`.
- Optimizer: AdamW on `stem.parameters()` only, lr `1e-3`, wd `1e-4`. Cosine LR schedule.
- 20 epochs (much more than the previous 8 — the head is frozen so the stem can be pushed harder without overfitting to the head).
- Per-epoch: train loss, val cos mean / median / p05 / p95, write `train_log.jsonl`. Save best-by-val to `checkpoint.pt`, last to `last.pt`.
- Resumable from `last.pt` if present.

`run_arc_adapter_pixel.bat`:

```bat
@echo off
cd /d C:\arc_distill
set REPO=C:\arc_distill\repo_assets
set PYTHONPATH=%REPO%\src
set PYTHONIOENCODING=utf-8
mkdir C:\arc_distill\arc_adapter\pixel_a 2>nul

C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m arc_distill.train_adapter ^
    --variant pixel_a ^
    --compact C:\arc_distill\arc_pixel_aligned\compact.pt ^
    --backbone-ckpt C:\arc_distill\r50_headless.pt ^
    --out-dir C:\arc_distill\arc_adapter\pixel_a ^
    --epochs 20 --batch-size 256 --workers 0 ^
    --lr 1e-3 --device cuda > C:\arc_distill\arc_adapter\pixel_a\train.log 2>&1
if errorlevel 1 ( echo train failed >> C:\arc_distill\arc_adapter\pixel_a\train.log & exit /b 1 )

C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m arc_distill.eval_adapter ^
    --variant pixel_a ^
    --checkpoint C:\arc_distill\arc_adapter\pixel_a\checkpoint.pt ^
    --backbone-ckpt C:\arc_distill\r50_headless.pt ^
    --compact C:\arc_distill\arc_pixel_aligned\compact.pt ^
    --out-json C:\arc_distill\arc_adapter\pixel_a\eval.json ^
    --device cuda >> C:\arc_distill\arc_adapter\pixel_a\train.log 2>&1
if errorlevel 1 ( echo eval failed >> C:\arc_distill\arc_adapter\pixel_a\train.log & exit /b 1 )

echo arc_adapter_pixel_a_done > C:\arc_distill\arc_adapter_pixel_a.done
```

- [ ] **Step 1:** Implement `train_adapter.py` (skeleton from old `train_pixel.py`, swap model construction, loss is unchanged).
- [ ] **Step 2:** Implement `eval_adapter.py` (skeleton from old `eval_pixel.py`).
- [ ] **Step 3:** Local smoke run — `--smoke` reads only the first 256 rows, runs 1 epoch, validates the loop. Expected loss decrease over 1 epoch.
- [ ] **Step 4:** Commit.

---

### Task 5: Latent compact prep

**Files:**
- Create: `src/arc_distill/prepare_compact_latent.py`

Pull `latent_bf16 (N,16,14,14)` and `arcface_fp32 (N,512)` from each encoded shard, filter `detected=True`, deal with the resolution mismatch (encoded latents are 64×64 from 512² VAE input, not 14×14 — see ⚠ note below).

⚠ **Resolution decision required before this task starts.** `encode_ffhq.py` encoded each FFHQ image as a 512² → `(16, 64, 64)` latent. The Latent-A stem assumes 14×14 (= 112²/8). Two options:

1. **Re-encode 112² aligned crops to (16, 14, 14) latents.** Cleanest — input to VAE is the same aligned crop the teacher saw, so step 3's gate measures latent-VAE-of-aligned-face vs teacher-of-aligned-face, no other moving parts. Cost: ~30 min on 3090 (already wired in `encode_ffhq.py`, just run with smaller resolution).
2. **Use existing 64×64 latents and rescale stems to 64×64 input.** Saves 30 min but conflates "VAE preserves identity" with "VAE preserves identity through whole-image encoding when face is centered but not aligned." That's a worse experiment.

Pick option 1. Add a Task 5a: write `encode_ffhq_aligned.py` that reuses existing infrastructure but emits `(16, 14, 14)` latents from the aligned 112² crops we already have in `compact.pt`. This lets us reuse the SCRFD-aligned crops we already paid for.

- [ ] **Step 1:** Write `encode_ffhq_aligned.py` — reads `compact.pt` (aligned 112² uint8), VAE-encodes each row to `(16, 14, 14)` bf16, writes `compact_latent.pt` with `{latent_bf16, arcface_fp32, shas, format_version: 1}`.
- [ ] **Step 2:** Smoke-run on first 256 rows, sanity-check shapes and dtypes.
- [ ] **Step 3:** Sync to Windows, run on full corpus (~30 min).
- [ ] **Step 4:** Commit.

---

### Task 6: Pixel-A run + gate calibration

Run Task 4's wrapper end-to-end on the 3090. Pull eval.json. **This is where the plan branches.**

Decision rules:
- **Pixel-A val cos ≥ 0.95 (expected):** freeze-and-adapt is well-formed. Set the Latent-A gate at `Pixel-A_val_cos − 0.10`. Proceed to Task 7.
- **Pixel-A val cos in [0.80, 0.95):** the stem can't fully recover R50's native conv1 distribution. Likely a stem-init / batchnorm-stats issue. Investigate before Task 7 — try (a) initialising stem from R50's actual `conv1.weight` and `bn1.{weight,bias,running_mean,running_var}`, (b) longer training, (c) lower LR. Re-run.
- **Pixel-A val cos < 0.80:** the freeze decision is wrong; stages 1–4 are *not* identity-sufficient given an arbitrary 64-channel feature map. Pivot to Latent-A2 (unfreeze stage 1) and update the plan.

- [ ] **Step 1:** Sync code, launch wrapper, monitor for `arc_adapter_pixel_a.done`.
- [ ] **Step 2:** Pull eval.json, log result, decide branch.
- [ ] **Step 3:** Write `docs/research/2026-04-30-pixel-a-result.md` capturing decision and gate threshold for Latent-A.
- [ ] **Step 4:** Commit.

---

### Task 7: Latent-A run (both stems in parallel)

Two wrappers, two run dirs (`latent_a_up`, `latent_a_native`), same data (`compact_latent.pt`). Expected wall: ~10 min each. Run sequentially on the 3090 (single-GPU, but stems are <1M params so even native-stride training is bandwidth-bound on the frozen backbone forward pass, not compute-bound).

- [ ] **Step 1:** Implement `run_arc_adapter_latent.bat` parameterised on variant.
- [ ] **Step 2:** Run upsample variant.
- [ ] **Step 3:** Run native variant.
- [ ] **Step 4:** Pull both eval.json's, compare against Pixel-A and against the gate from Task 6.
- [ ] **Step 5:** Pick winner, commit, update topic index `docs/research/_topics/demographic-pc-pipeline.md`.

---

### Task 8: Writeup + roadmap update

- [ ] **Step 1:** Write `docs/research/2026-04-30-arcface-frozen-adapter.md` with full results, the Pixel-A vs Latent-A delta (the actual scientific output), and what this implies for downstream uses (latent-native classifier distillation, latent-space identity-preservation tests, etc.).
- [ ] **Step 2:** Update `docs/research/2026-04-27-latent-native-classifier-distillation.md` with the concrete Stage 2 result and the new Stage 3 plan derived from Latent-A's outcome.
- [ ] **Step 3:** Add an entry to memory `project_demographic_pc_stage4_5.md` (or a new memory file) with the Pixel-A and Latent-A cosine numbers — these are load-bearing reference points for any future "is this latent identity-preserving" question.
- [ ] **Step 4:** Commit.

---

## Anti-goals

- Not retraining ArcFace itself — the teacher stays exactly as buffalo_l ships it.
- Not chasing cos>0.95 on Latent-A unconditionally — if VAE bottleneck genuinely loses identity, the *number itself* is the result. Don't game the gate by adding capacity until you've measured the honest delta first.
- Not re-running the previous-plan ResNet-18 baseline. Its result is not a relevant comparison; it answers a different question.
- Not implementing Latent-A2 (partial unfreeze) speculatively. Only build it if Task 6's branch rule says we have to.

## Open risks

1. **ONNX→PyTorch port asymmetry.** ArcFace ONNX may have non-standard ops (e.g. PReLU instead of ReLU in some layers, `Add` ordering different from torchvision). If `test_headless_matches_teacher_on_fixture` fails, fall back to onnxruntime-as-blackbox + train only the stem (loses Latent-A2 path but unblocks Pixel-A and Latent-A).
2. **BatchNorm running stats.** The trainable stem starts with random BN. During val we want eval-mode BN; the frozen backbone is already in eval mode. Standard practice — flag in case some weird interaction surfaces.
3. **Stem capacity.** If LatentStemUpsample > LatentStemNative dramatically, the result is partly about stem expressivity, not VAE identity preservation. Holding stem capacity matched (both <1M params) is the partial mitigation; if results are too divergent, run a third stem with capacity in between as a tiebreaker.
