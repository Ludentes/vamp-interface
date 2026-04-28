> **SUPERSEDED by [`2026-04-29-arcface-frozen-adapter.md`](2026-04-29-arcface-frozen-adapter.md).** This plan executed; the gate failed (val cos 0.377 unaligned, 0.323 aligned). Square-1 verification ([scripts/check_arc_oracle.py](../../../scripts/check_arc_oracle.py)) showed the pipeline is byte-perfect — the failure is that the experiment answered the wrong question (can a fresh ResNet-18 learn ArcFace from scratch) rather than the intended one (can we transfer ArcFace's pretrain to a latent-input student). The successor plan trains an adapter in front of the frozen ArcFace R50 backbone instead.

# ArcFace-Pixel Sanity Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a ResNet-18 student on FFHQ raw pixels to predict the frozen InsightFace `buffalo_l` ArcFace 512-d embedding. Gate: held-out cosine R² > 0.9. This validates that the distillation task itself is well-posed before swapping the input to 16-channel Flux VAE latents (step 3, `arc_latent`).

**Architecture:** Read FFHQ shards (image bytes from parquet) + matching encoded `.pt` files (arcface_fp32 + image_sha256 + detected mask). Train only on `detected=True` rows. Student is `torchvision.resnet18(weights=DEFAULT)` with `fc → Linear(in_features, 512)`; output L2-normalised. Loss is `1 - cos(student, teacher)`. Train on remote Windows 3090, persist checkpoints + held-out metrics to `C:\arc_distill\arc_pixel\`.

**Tech Stack:** PyTorch, torchvision, pyarrow, Pillow; no external trainer (Lightning/HF) — keep loop hand-rolled and small.

**Train/val split:** Deterministic SHA-prefix split — `image_sha256` first hex digit `'f'` is held out (~6.25% ≈ ~1500 detected rows). All other digits train. Computed on `image_sha256` so the split is stable across re-runs and identical to the split arc_latent will use in step 3.

---

## File Structure

- `src/arc_distill/__init__.py` — package marker
- `src/arc_distill/dataset.py` — `FFHQPixelDataset` (joins parquet image bytes with `.pt` arcface targets, filters detected=True, applies SHA-prefix split)
- `src/arc_distill/model.py` — `ArcStudentResNet18` (pretrained backbone, 512-d L2-normalised head)
- `src/arc_distill/train_pixel.py` — training script with CLI: `--shards-dir --encoded-dir --out-dir --epochs --batch-size --lr --resolution 224 --workers --device --smoke`
- `src/arc_distill/eval_pixel.py` — held-out evaluation: per-row cosine, mean cosine, cosine R², histogram dump
- `tests/test_arc_distill.py` — unit tests for dataset join, split function, model output shape + L2 norm, cosine loss
- `scripts/run_arc_pixel_train.bat` — Windows wrapper invoked by scheduled task

External (remote, gitignored):
- `C:\arc_distill\ffhq_parquet\data\train-{i:05d}-of-00190.parquet` — already present
- `C:\arc_distill\encoded\shard-{i:05d}.pt` — already present (arcface_fp32, latent_bf16, detected, image_sha256)
- `C:\arc_distill\arc_pixel\checkpoint.pt` — best-by-val-cosine
- `C:\arc_distill\arc_pixel\eval.json` — final gate metrics

---

### Task 1: Package skeleton + SHA-prefix split helper

**Files:**
- Create: `src/arc_distill/__init__.py`
- Create: `src/arc_distill/dataset.py` (split helper only this task)
- Create: `tests/test_arc_distill.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_arc_distill.py
from arc_distill.dataset import is_held_out

def test_is_held_out_f_prefix_held():
    assert is_held_out("f1234567" + "0"*56) is True

def test_is_held_out_other_prefix_train():
    assert is_held_out("a1234567" + "0"*56) is False
    assert is_held_out("01234567" + "0"*56) is False

def test_is_held_out_uppercase_ok():
    assert is_held_out("F1234567" + "0"*56) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/test_arc_distill.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'arc_distill'`

- [ ] **Step 3: Create package and split helper**

```python
# src/arc_distill/__init__.py
```

```python
# src/arc_distill/dataset.py
"""FFHQ pixel + ArcFace teacher dataset for arc_pixel distillation."""
from __future__ import annotations


def is_held_out(image_sha256: str) -> bool:
    """Deterministic 6.25% held-out split: SHA prefix 'f' is val."""
    return image_sha256[:1].lower() == "f"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/test_arc_distill.py -v`
Expected: PASS (3/3)

- [ ] **Step 5: Commit**

```bash
git add src/arc_distill/__init__.py src/arc_distill/dataset.py tests/test_arc_distill.py
git commit -m "feat(arc_distill): package skeleton + SHA-prefix split helper"
```

---

### Task 2: Dataset class — joins parquet bytes with `.pt` arcface targets

**Files:**
- Modify: `src/arc_distill/dataset.py`
- Modify: `tests/test_arc_distill.py`

The dataset reads one shard at construction (parquet image bytes + matching `.pt`), filters `detected=True`, applies the held-out filter (`split="train"` or `"val"`), and returns `(image_tensor, target_512d)` per `__getitem__`.

- [ ] **Step 1: Write the failing test (uses existing 6-row smoke fixture)**

```python
# tests/test_arc_distill.py — append
import io
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

from arc_distill.dataset import FFHQPixelDataset


def _make_smoke_pt(tmp_path: Path, shas: list[str]) -> Path:
    """Build a fake encoded .pt aligned with the smoke parquet's 6 rows."""
    n = len(shas)
    pt_path = tmp_path / "shard-00000.pt"
    torch.save({
        "image_sha256": shas,
        "arcface_fp32": torch.randn(n, 512, dtype=torch.float32),
        "detected": torch.tensor([True, True, True, True, False, True]),
        "format_version": 1,
    }, pt_path)
    return pt_path


def test_dataset_join_filters_detected_and_split(tmp_path):
    smoke = Path("tests/fixtures/ffhq_smoke.parquet")
    shas = pq.read_table(smoke, columns=["image_sha256"]).column("image_sha256").to_pylist()
    pt = _make_smoke_pt(tmp_path, shas)

    ds_train = FFHQPixelDataset(
        parquet_path=smoke, encoded_pt_path=pt, split="train", resolution=224,
    )
    ds_val = FFHQPixelDataset(
        parquet_path=smoke, encoded_pt_path=pt, split="val", resolution=224,
    )
    # 6 rows, 1 has detected=False → 5 candidates. Split by SHA prefix 'f'.
    n_held = sum(1 for s in shas if s[:1].lower() == "f")
    n_detected = 5
    n_detected_held = sum(
        1 for s, d in zip(shas, [True, True, True, True, False, True])
        if d and s[:1].lower() == "f"
    )
    assert len(ds_val) == n_detected_held
    assert len(ds_train) == n_detected - n_detected_held
    assert len(ds_train) + len(ds_val) == n_detected

    img, tgt = ds_train[0]
    assert img.shape == (3, 224, 224)
    assert img.dtype == torch.float32
    assert tgt.shape == (512,)
    assert tgt.dtype == torch.float32
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/test_arc_distill.py::test_dataset_join_filters_detected_and_split -v`
Expected: FAIL — `ImportError: cannot import name 'FFHQPixelDataset'`

- [ ] **Step 3: Implement dataset**

```python
# src/arc_distill/dataset.py — append
import io
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow.parquet as pq
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FFHQPixelDataset(Dataset):
    """One parquet shard joined with its matching encoded .pt by row order.

    The encoded .pt was produced by encode_ffhq.py in the same row order as
    the parquet shard, so we align positionally and verify image_sha256 lines up.
    Filters to detected=True and applies the SHA-prefix train/val split.
    """

    def __init__(
        self,
        parquet_path: Path,
        encoded_pt_path: Path,
        split: Literal["train", "val"],
        resolution: int = 224,
    ):
        self.parquet_path = Path(parquet_path)
        table = pq.read_table(self.parquet_path, columns=["image", "image_sha256"])
        parquet_shas = table.column("image_sha256").to_pylist()
        image_bytes_col = table.column("image").to_pylist()

        pt = torch.load(encoded_pt_path, map_location="cpu", weights_only=False)
        pt_shas = pt["image_sha256"]
        arcface = pt["arcface_fp32"]
        detected = pt["detected"]

        if list(parquet_shas) != list(pt_shas):
            raise ValueError(
                f"sha mismatch parquet={self.parquet_path} pt={encoded_pt_path}"
            )

        keep = []
        for i, (sha, det) in enumerate(zip(parquet_shas, detected.tolist())):
            if not det:
                continue
            held = sha[:1].lower() == "f"
            if (split == "val") == held:
                keep.append(i)

        self.indices = keep
        self.image_bytes = image_bytes_col
        self.shas = parquet_shas
        self.targets = arcface
        self.transform = T.Compose([
            T.Resize((resolution, resolution), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        rec = self.image_bytes[idx]
        img = Image.open(io.BytesIO(rec["bytes"])).convert("RGB")
        x = self.transform(img)
        y = self.targets[idx].to(torch.float32)
        return x, y
```

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/test_arc_distill.py -v`
Expected: PASS (4/4)

- [ ] **Step 5: Commit**

```bash
git add src/arc_distill/dataset.py tests/test_arc_distill.py
git commit -m "feat(arc_distill): FFHQPixelDataset joining parquet bytes with arcface targets"
```

---

### Task 3: Student model + cosine loss

**Files:**
- Create: `src/arc_distill/model.py`
- Modify: `tests/test_arc_distill.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_arc_distill.py — append
import torch.nn.functional as F

from arc_distill.model import ArcStudentResNet18, cosine_distance_loss


def test_model_output_shape_and_l2_norm():
    m = ArcStudentResNet18(pretrained=False).eval()
    with torch.no_grad():
        out = m(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 512)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_cosine_distance_loss_zero_when_aligned():
    a = F.normalize(torch.randn(4, 512), dim=-1)
    assert cosine_distance_loss(a, a).item() < 1e-6


def test_cosine_distance_loss_one_when_opposite():
    a = F.normalize(torch.randn(4, 512), dim=-1)
    assert abs(cosine_distance_loss(a, -a).item() - 2.0) < 1e-5
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src uv run pytest tests/test_arc_distill.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement model + loss**

```python
# src/arc_distill/model.py
"""ArcFace-pixel student: torchvision ResNet-18 with 512-d L2-normalised head."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


class ArcStudentResNet18(nn.Module):
    """ResNet-18 backbone, fc → 512, output L2-normalised on the unit sphere."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = tvm.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = tvm.resnet18(weights=weights)
        in_f = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_f, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        return F.normalize(z, dim=-1)


def cosine_distance_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - cos(pred, target). Both inputs assumed L2-normalised; we re-normalise
    target defensively (teacher embeddings are stored as raw fp32 from insightface,
    which already L2-normalises buffalo_l outputs, but be safe)."""
    target = F.normalize(target, dim=-1)
    return (1.0 - (pred * target).sum(dim=-1)).mean()
```

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/test_arc_distill.py -v`
Expected: PASS (7/7)

- [ ] **Step 5: Commit**

```bash
git add src/arc_distill/model.py tests/test_arc_distill.py
git commit -m "feat(arc_distill): ResNet-18 student model + cosine distance loss"
```

---

### Task 4: Multi-shard concatenated dataset wrapper

**Files:**
- Modify: `src/arc_distill/dataset.py`
- Modify: `tests/test_arc_distill.py`

Per-shard loading keeps memory bounded (one parquet at a time held by a sub-dataset). For training we need a `ConcatDataset`-style view that lazily holds all 190 sub-datasets. We'll lean on `torch.utils.data.ConcatDataset` and just add a builder.

- [ ] **Step 1: Write failing test**

```python
# tests/test_arc_distill.py — append
from arc_distill.dataset import build_ffhq_concat


def test_build_ffhq_concat_smoke(tmp_path):
    smoke = Path("tests/fixtures/ffhq_smoke.parquet")
    shas = pq.read_table(smoke, columns=["image_sha256"]).column("image_sha256").to_pylist()
    pt = _make_smoke_pt(tmp_path, shas)

    # Point both shards-dir and encoded-dir at directories holding our one fake shard.
    sd = tmp_path / "shards"; sd.mkdir()
    ed = tmp_path / "encoded"; ed.mkdir()
    (sd / "train-00000-of-00190.parquet").symlink_to(smoke.resolve())
    (ed / "shard-00000.pt").symlink_to(pt.resolve())

    train = build_ffhq_concat(sd, ed, split="train", resolution=224)
    val = build_ffhq_concat(sd, ed, split="val", resolution=224)
    assert len(train) + len(val) == 5  # 6 rows, 1 not detected
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src uv run pytest tests/test_arc_distill.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_ffhq_concat'`

- [ ] **Step 3: Implement the builder**

```python
# src/arc_distill/dataset.py — append
import re

from torch.utils.data import ConcatDataset

_SHARD_RE = re.compile(r"train-(\d{5})-of-\d{5}\.parquet$")


def build_ffhq_concat(
    shards_dir: Path,
    encoded_dir: Path,
    split: Literal["train", "val"],
    resolution: int = 224,
) -> ConcatDataset:
    """Build a ConcatDataset over every shard whose .parquet has a matching .pt.

    Skips shards missing their encoded .pt without raising — useful while the
    encode_ffhq.py run is still in flight.
    """
    shards_dir = Path(shards_dir)
    encoded_dir = Path(encoded_dir)
    parts: list[FFHQPixelDataset] = []
    for p in sorted(shards_dir.glob("train-*-of-*.parquet")):
        m = _SHARD_RE.search(p.name)
        if not m:
            continue
        idx = int(m.group(1))
        pt = encoded_dir / f"shard-{idx:05d}.pt"
        if not pt.exists():
            continue
        parts.append(FFHQPixelDataset(p, pt, split=split, resolution=resolution))
    if not parts:
        raise FileNotFoundError(f"no matched shard pairs under {shards_dir} / {encoded_dir}")
    return ConcatDataset(parts)
```

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/test_arc_distill.py -v`
Expected: PASS (8/8)

- [ ] **Step 5: Commit**

```bash
git add src/arc_distill/dataset.py tests/test_arc_distill.py
git commit -m "feat(arc_distill): build_ffhq_concat across all matched shard pairs"
```

---

### Task 5: Training script

**Files:**
- Create: `src/arc_distill/train_pixel.py`

No tests on this module — it's a thin loop wrapping the tested dataset + model. Smoke run in Task 6 is the test.

- [ ] **Step 1: Implement training script**

```python
# src/arc_distill/train_pixel.py
"""Train ArcStudentResNet18 on FFHQ pixels → ArcFace teacher embeddings.

Resumable: writes `checkpoint.pt` (best by val cosine) and `last.pt` every epoch.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from arc_distill.dataset import build_ffhq_concat
from arc_distill.model import ArcStudentResNet18, cosine_distance_loss


def evaluate(model, loader, device) -> dict:
    model.eval()
    cos_sum = 0.0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            p = model(x)
            yn = F.normalize(y, dim=-1)
            cos_sum += (p * yn).sum().item()
            n += x.size(0)
    return {"val_cosine_mean": cos_sum / max(n, 1), "val_n": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards-dir", type=Path, required=True)
    ap.add_argument("--encoded-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--smoke", action="store_true",
                    help="Use a single sub-dataset, 1 epoch — sanity only.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "train_log.jsonl"

    print(f"[{time.strftime('%H:%M:%S')}] building datasets...")
    train_full = build_ffhq_concat(args.shards_dir, args.encoded_dir, "train", args.resolution)
    val = build_ffhq_concat(args.shards_dir, args.encoded_dir, "val", args.resolution)

    if args.smoke:
        # take only the first sub-dataset of train_full's ConcatDataset
        train = train_full.datasets[0]
        epochs = 1
    else:
        train = train_full
        epochs = args.epochs

    print(f"train={len(train)} val={len(val)}")

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model = ArcStudentResNet18(pretrained=True).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val = -1.0
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        loss_sum = 0.0
        n = 0
        for step, (x, y) in enumerate(train_loader):
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)
            p = model(x)
            loss = cosine_distance_loss(p, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * x.size(0)
            n += x.size(0)
            if step % 50 == 0:
                print(f"  epoch={epoch} step={step}/{len(train_loader)} "
                      f"loss={loss.item():.4f}")
        sched.step()
        train_loss = loss_sum / max(n, 1)
        val_metrics = evaluate(model, val_loader, args.device)
        elapsed = time.time() - t0
        rec = {"epoch": epoch, "train_cos_loss": train_loss,
               "elapsed_s": round(elapsed, 1), **val_metrics}
        print(f"[epoch {epoch}] {rec}")
        with log_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "val_cosine_mean": val_metrics["val_cosine_mean"]},
                   args.out_dir / "last.pt")
        if val_metrics["val_cosine_mean"] > best_val:
            best_val = val_metrics["val_cosine_mean"]
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "val_cosine_mean": best_val},
                       args.out_dir / "checkpoint.pt")
            print(f"  saved best (val_cosine_mean={best_val:.4f})")

    print(f"done. best val_cosine_mean={best_val:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it parses**

Run: `PYTHONPATH=src uv run python -c "import arc_distill.train_pixel"`
Expected: silent success.

- [ ] **Step 3: Commit**

```bash
git add src/arc_distill/train_pixel.py
git commit -m "feat(arc_distill): training script for ArcFace-pixel student"
```

---

### Task 6: Smoke run on local shard

**Files:**
- (none new — uses existing `tests/fixtures/ffhq_shard0.parquet` plus a fresh local `.pt`)

The smoke confirms the loop runs end-to-end on one shard locally before paying for a 70k run on the remote 3090. We need the matching encoded `.pt` for shard 0; if it's not present locally, fetch it from Windows.

- [ ] **Step 1: Ensure local encoded shard 0 exists**

Run:
```bash
mkdir -p /tmp/arc_pixel_smoke/{shards,encoded,out}
ln -sf "$(pwd)/tests/fixtures/ffhq_shard0.parquet" \
       /tmp/arc_pixel_smoke/shards/train-00000-of-00190.parquet
# Pull just shard 0's encoded .pt from Windows (~50 MB)
scp videocard@192.168.87.25:/c/arc_distill/encoded/shard-00000.pt \
    /tmp/arc_pixel_smoke/encoded/shard-00000.pt
```
Expected: ~50 MB file present.

- [ ] **Step 2: Run smoke training (1 shard, 1 epoch)**

Run:
```bash
PYTHONPATH=src uv run python -m arc_distill.train_pixel \
  --shards-dir /tmp/arc_pixel_smoke/shards \
  --encoded-dir /tmp/arc_pixel_smoke/encoded \
  --out-dir /tmp/arc_pixel_smoke/out \
  --batch-size 64 --workers 2 --device cuda --smoke
```
Expected: completes in <5 min, prints decreasing `train_cos_loss`, writes
`/tmp/arc_pixel_smoke/out/{checkpoint.pt,train_log.jsonl}`. Final
`val_cosine_mean` should be > 0.0 (sign-of-life — too few val rows for a real gate).

If GPU OOM: drop `--batch-size` to 32. If CPU-only smoke desired: `--device cpu --batch-size 16`.

- [ ] **Step 3: Inspect log**

Run: `cat /tmp/arc_pixel_smoke/out/train_log.jsonl`
Expected: one JSON line with `train_cos_loss`, `val_cosine_mean`, `elapsed_s`. Sanity:
`train_cos_loss` should be in (0.05, 0.5).

- [ ] **Step 4: Commit (no code changes — smoke artifacts not committed)**

If any code was tweaked to make the smoke work, commit those fixes:
```bash
git add -A
git commit -m "fix(arc_distill): smoke-pass tweaks" --allow-empty-message --allow-empty
```
Otherwise skip.

---

### Task 7: Remote launcher (Windows scheduled task)

**Files:**
- Create: `scripts/run_arc_pixel_train.bat`
- Modify: `scripts/sync_extractor_assets_to_windows.sh` (add `src/arc_distill` to the rsync list)

- [ ] **Step 1: Add arc_distill to the sync script**

Locate the rsync/scp section that copies `src/demographic_pc` and add `src/arc_distill` alongside it. Verify with:
```bash
grep -n "demographic_pc\|arc_distill" scripts/sync_extractor_assets_to_windows.sh
```
Expected: both directories appear.

- [ ] **Step 2: Create the Windows wrapper**

```batch
@echo off
REM scripts/run_arc_pixel_train.bat
REM Trains arc_distill ArcFace-pixel student on FFHQ. Invoked by schtasks.

set REPO=C:\arc_distill\repo_assets
set PYTHONPATH=%REPO%\src
cd /d %REPO%

call C:\Users\videocard\ComfyUI\venv\Scripts\activate.bat

python -m arc_distill.train_pixel ^
  --shards-dir C:\arc_distill\ffhq_parquet\data ^
  --encoded-dir C:\arc_distill\encoded ^
  --out-dir C:\arc_distill\arc_pixel ^
  --epochs 8 --batch-size 128 --workers 4 --resolution 224 ^
  --lr 3e-4 --device cuda ^
  > C:\arc_distill\arc_pixel\train.log 2>&1
```

- [ ] **Step 3: Sync repo assets to Windows**

Run: `bash scripts/sync_extractor_assets_to_windows.sh`
Expected: `src/arc_distill/` arrives at `C:\arc_distill\repo_assets\src\arc_distill\`.

- [ ] **Step 4: Register and launch the scheduled task**

Run (from local shell — adjust ssh target/credentials to match your established pattern):
```bash
ssh videocard@192.168.87.25 'mkdir -p /c/arc_distill/arc_pixel && \
  schtasks /create /tn arc_pixel_train /sc once /st 23:59 /sd 12/31/2099 \
  /tr "C:\\arc_distill\\repo_assets\\scripts\\run_arc_pixel_train.bat" /f'
ssh videocard@192.168.87.25 'schtasks /run /tn arc_pixel_train'
```
Expected: `SUCCESS: Attempted to run the scheduled task "arc_pixel_train".`

- [ ] **Step 5: Tail the train log**

Run:
```bash
ssh videocard@192.168.87.25 'tail -f /c/arc_distill/arc_pixel/train.log'
```
Expected: builds dataset (~70k train, ~4k val), then per-step loss prints. ResNet-18 at 224² with bs=128 should hit ~3 min/epoch on a 3090; 8 epochs ≈ 25 min plus dataloading overhead.

- [ ] **Step 6: Commit launcher + sync update**

```bash
git add scripts/run_arc_pixel_train.bat scripts/sync_extractor_assets_to_windows.sh
git commit -m "chore(arc_distill): Windows launcher + sync arc_distill source to remote"
```

---

### Task 8: Held-out evaluation + gate

**Files:**
- Create: `src/arc_distill/eval_pixel.py`

The training loop already records `val_cosine_mean` per epoch, but the gate metric is **cosine R²** (= mean cosine, since both vectors are L2-normalised, equivalently `1 - 0.5 * E[||p - y||²]`). We also dump a histogram of per-row cosines for diagnostics.

- [ ] **Step 1: Implement eval script**

```python
# src/arc_distill/eval_pixel.py
"""Evaluate trained arc_distill checkpoint on FFHQ held-out.

Gate: per-row cosine mean > 0.9 → step 2 passes, proceed to arc_latent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from arc_distill.dataset import build_ffhq_concat
from arc_distill.model import ArcStudentResNet18


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--shards-dir", type=Path, required=True)
    ap.add_argument("--encoded-dir", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    val = build_ffhq_concat(args.shards_dir, args.encoded_dir, "val", args.resolution)
    loader = DataLoader(val, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    model = ArcStudentResNet18(pretrained=False).to(args.device)
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    cosines = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)
            p = model(x)
            yn = F.normalize(y, dim=-1)
            c = (p * yn).sum(dim=-1).cpu().numpy()
            cosines.append(c)
    cosines = np.concatenate(cosines)

    out = {
        "n": int(cosines.size),
        "cosine_mean": float(cosines.mean()),
        "cosine_median": float(np.median(cosines)),
        "cosine_p05": float(np.quantile(cosines, 0.05)),
        "cosine_p95": float(np.quantile(cosines, 0.95)),
        "cosine_min": float(cosines.min()),
        "frac_above_0p9": float((cosines > 0.9).mean()),
        "gate_passed_mean_gt_0p9": bool(cosines.mean() > 0.9),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run remotely on the trained checkpoint**

```bash
ssh videocard@192.168.87.25 \
  'cd /c/arc_distill/repo_assets && set PYTHONPATH=C:\\arc_distill\\repo_assets\\src && \
   python -m arc_distill.eval_pixel \
     --checkpoint C:\\arc_distill\\arc_pixel\\checkpoint.pt \
     --shards-dir C:\\arc_distill\\ffhq_parquet\\data \
     --encoded-dir C:\\arc_distill\\encoded \
     --out-json C:\\arc_distill\\arc_pixel\\eval.json \
     --device cuda'
```
Expected: prints JSON with `cosine_mean`. Gate is `cosine_mean > 0.9`.

- [ ] **Step 3: Pull eval + checkpoint home**

```bash
mkdir -p output/arc_pixel
scp videocard@192.168.87.25:/c/arc_distill/arc_pixel/eval.json output/arc_pixel/
scp videocard@192.168.87.25:/c/arc_distill/arc_pixel/train_log.jsonl output/arc_pixel/
# Skip checkpoint.pt unless gate passes (it'll be ~45 MB and only useful if we want to compare)
```

- [ ] **Step 4: Commit eval script + small artifacts**

```bash
git add src/arc_distill/eval_pixel.py output/arc_pixel/eval.json output/arc_pixel/train_log.jsonl
git commit -m "feat(arc_distill): held-out eval script + step-2 gate metrics"
```

---

### Task 9: Research note + topic-index update

**Files:**
- Create: `docs/research/2026-04-29-arcface-pixel-baseline.md`
- Modify: `docs/research/_topics/demographic-pc-pipeline.md`

- [ ] **Step 1: Write research note**

Required frontmatter + sections: Decision (gate passed? what cosine?), Why (sanity check before arc_latent), How (shards + .pt, ResNet-18, 224², AdamW 3e-4 cosine schedule, train/val SHA-prefix split, N rows per side), Anti-goals (not a deployment model — only validates task is well-posed). Include the eval JSON inline.

```markdown
---
status: live
topic: demographic-pc-pipeline
---

# ArcFace-pixel sanity baseline (2026-04-29)

## Decision

[Gate result: cosine_mean=X.XXX on N val rows. PASS / FAIL vs 0.9 threshold.]

## Why

Step 2 of the latent-native classifier distillation roadmap
(`docs/research/2026-04-27-latent-native-classifier-distillation.md`).
Validates that distilling InsightFace `buffalo_l` ArcFace 512-d embeddings
into a small ResNet-18 student is well-posed *on raw pixels* before we
swap the input to 16-channel Flux VAE latents. If pixel-input fails,
latent-input is hopeless.

## How

- **Data.** 70k FFHQ images (190 parquet shards), joined with our
  `encode_ffhq.py` outputs that hold the matching `arcface_fp32 (N, 512)`
  per row. Filter `detected=True` (~25k rows total at SCRFD det_thresh=0.5).
- **Split.** Deterministic SHA-prefix: rows whose `image_sha256` starts
  with `f` are val (~6.25%), rest are train. Stable across re-runs and
  identical to the split arc_latent will use in step 3.
- **Student.** torchvision `resnet18(weights=DEFAULT)`, fc replaced by
  `Linear(in_features, 512)`, output L2-normalised.
- **Teacher.** Frozen `arcface_fp32` already on disk from
  encode_ffhq.py — no teacher forward at train time.
- **Loss.** `1 - cos(student, target)`.
- **Schedule.** AdamW 3e-4, weight decay 1e-4, batch 128, 8 epochs,
  cosine LR. Resolution 224². Single 3090.

## Result

[paste output/arc_pixel/eval.json contents here]

## Anti-goals

- Not a production model — held-out cosine alone is the gate.
- Not optimising input pipeline (no face-crop alignment) — the latent
  variant in step 3 will see whole-image-equivalent latents, so the
  pixel baseline must too.
- Not retuning ArcFace teacher — we accept its known demographic biases.
```

- [ ] **Step 2: Update topic index**

Open `docs/research/_topics/demographic-pc-pipeline.md` and add a one-line entry under the chronological list pointing at `2026-04-29-arcface-pixel-baseline.md` with the gate outcome.

- [ ] **Step 3: Commit**

```bash
git add docs/research/2026-04-29-arcface-pixel-baseline.md \
        docs/research/_topics/demographic-pc-pipeline.md
git commit -m "docs(arc_distill): step-2 ArcFace-pixel baseline result + topic-index update"
```

---

## Decision point at end of plan

After Task 9:

- **If gate passes (`cosine_mean > 0.9`):** start step 3 (`arc_latent`). The architecture is identical except input is `(B, 16, 64, 64)` — replace the ResNet-18 first conv `(64, 3, 7, 7)` with `(64, 16, 3, 3)` and feed `latent_bf16` cast to fp32 directly. No ImageNet pretraining transfers. Memorialise as a new plan: `2026-04-30-arc-latent-distillation.md`.
- **If gate fails:** diagnose. Likely culprits in order: (1) full-image input vs ArcFace expects aligned 112² crop — try cropping using `bbox` if we backfill it; (2) 8 epochs not enough — check `train_log.jsonl` slope; (3) ResNet-18 too small — try ResNet-50. Document the failure and what was tried before declaring the latent path unreachable.
