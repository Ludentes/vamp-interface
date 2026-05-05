---
status: live
topic: neural-deformation-control
supersedes: 2026-05-05-arkit-poseguider-distill-plan.md
---

# ARKit → PersonaLive bridge v1 — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drive PersonaLive from iPhone Live Link Face's 61-float ARKit stream (no driving RGB) by (a) computing implicit keypoints `k_d` analytically from cached reference geometry plus per-frame ARKit head Euler, and (b) distilling a single MLP student that maps ARKit expression channels (52 blendshapes + 6 eye rotations) to PersonaLive's `motion_encoder` output.

**Architecture:** One closed-form path + one learned path, both per-frame. Closed-form path: `kp_ref` cached once via real `motion_extractor`; per frame `R_d = euler_to_rotmat(yaw,pitch,roll)`; `k_d = (kp_ref @ R_d) * s_ref + t_ref`; real `pose_guider(draw_keypoints(k_d))`. Learned path: 4-layer MLP, `b₆₁_expr → m_f` matched to teacher `motion_encoder(face_crop_224)` via per-frame MSE.

**Tech stack:** Python 3.10 (PersonaLive's `.venv`), PyTorch 2.11+cu128, einops, **diffusers 0.27.0** (vamp-interface's 0.37.1 raises on `MotEncoder`'s deprecated `get_1d_sincos_pos_embed_from_grid` call). All bridge code runs in `~/w/PersonaLive/.venv` — pytest installed there via `uv pip install --python ~/w/PersonaLive/~/w/PersonaLive/.venv/bin/python pytest`. PersonaLive `~/w/PersonaLive` source tree is sys.path-injected.

See also: [`2026-05-05-arkit-bridge-v1-design.md`](2026-05-05-arkit-bridge-v1-design.md) for design rationale, paper-correspondence, and risks.

## Tasks at a glance

1. Mark prior plan superseded; drop dead code; topic index
2. Closed-form keypoint path (`euler_to_rotmat` + `compose_kd`) with TDD
3. PersonaLive teachers (`motion_encoder`, `motion_extractor`, `pose_guider`)
4. `MotEncoderStudent` MLP with TDD
5. Pair-extraction script (`b_expr`, `m_f`) per take
6. Distill loop + smoke overfit
7. Per-channel sensitivity eval
8. Real-corpus extraction + real distill
9. ARKit↔LivePortrait Euler sign-flip calibration (8-way enumeration)
10. Drop-in PersonaLive smoke render
11. Per-component perf benchmark (RTX 5090 fp16)
12. Readout doc + decision

Full code skeletons + commands are inline below. Tasks 1–7 are GPU-light;
Task 8 is the only longish run; Task 9 is one-shot calibration; Task 11
is the perf benchmark whose numbers feed the readout.

---

## Task 1: Mark prior plan superseded, drop dead code, update topic index

**Files:**
- Modify: `docs/research/2026-05-05-arkit-poseguider-distill-plan.md` (frontmatter)
- Modify: `docs/research/_topics/neural-deformation-control.md`
- Delete: `src/arkit_bridge/{encoder,teacher,extractors,dataset,distill,eval}.py`,
  `tests/arkit_bridge/test_{encoder,teacher,extractors}.py`,
  `scripts/{extract_distill_pairs,extract_llf_pairs,train_arkit_distill,eval_arkit_distill}.py`

- [ ] **Step 1: Frontmatter on the superseded plan**

In `docs/research/2026-05-05-arkit-poseguider-distill-plan.md`, top-of-file:

```markdown
---
status: superseded
topic: neural-deformation-control
superseded_by: 2026-05-05-arkit-bridge-v1-plan.md
---
```

- [ ] **Step 2: Topic index pointers**

Append to `docs/research/_topics/neural-deformation-control.md`:

```markdown
- 2026-05-05 bridge v1 design: [`2026-05-05-arkit-bridge-v1-design.md`](../2026-05-05-arkit-bridge-v1-design.md) — paper-faithful (arxiv 2512.11253 §3.1) closed-form keypoints + single distilled MLP for the facial motion embedding.
- 2026-05-05 bridge v1 plan: [`2026-05-05-arkit-bridge-v1-plan.md`](../2026-05-05-arkit-bridge-v1-plan.md) — TDD plan; supersedes the single-student PoseGuider plan.
- 2026-05-05 PersonaLive architecture notes: [`2026-05-05-personalive-architecture-notes.md`](../2026-05-05-personalive-architecture-notes.md) — component-by-component breakdown, extension points, perf budget, training notes, replacement options.
```

- [ ] **Step 3: Drop superseded code**

```bash
cd /home/newub/w/vamp-interface
git rm src/arkit_bridge/encoder.py tests/arkit_bridge/test_encoder.py \
       src/arkit_bridge/teacher.py tests/arkit_bridge/test_teacher.py \
       src/arkit_bridge/extractors.py tests/arkit_bridge/test_extractors.py \
       src/arkit_bridge/dataset.py src/arkit_bridge/distill.py \
       src/arkit_bridge/eval.py \
       scripts/extract_distill_pairs.py scripts/extract_llf_pairs.py \
       scripts/train_arkit_distill.py scripts/eval_arkit_distill.py
```

- [ ] **Step 4: Commit**

```bash
git add docs/research/2026-05-05-arkit-poseguider-distill-plan.md \
        docs/research/2026-05-05-arkit-bridge-v1-design.md \
        docs/research/2026-05-05-arkit-bridge-v1-plan.md \
        docs/research/2026-05-05-personalive-architecture-notes.md \
        docs/research/_topics/neural-deformation-control.md
git commit -m "docs(neural-deformation): bridge v1 (paper-faithful) design + plan + arch notes; supersede prior"
```

---

## Task 2: Closed-form keypoint path

**Files:**
- Create: `src/arkit_bridge/closed_form_pose.py`
- Create: `tests/arkit_bridge/test_closed_form_pose.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/arkit_bridge/test_closed_form_pose.py
import torch
from arkit_bridge.closed_form_pose import euler_to_rotmat, compose_kd


def test_euler_to_rotmat_zero_is_identity():
    R = euler_to_rotmat(torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0))
    assert torch.allclose(R, torch.eye(3), atol=1e-6)


def test_euler_to_rotmat_orthonormal():
    R = euler_to_rotmat(torch.tensor(0.3), torch.tensor(-0.2), torch.tensor(0.1))
    assert torch.allclose(R @ R.transpose(-1, -2), torch.eye(3), atol=1e-5)
    assert abs(torch.det(R).item() - 1.0) < 1e-5


def test_compose_kd_zero_rotation_returns_scaled_translated_kp():
    kp_ref = torch.randn(1, 21, 3)
    t_ref = torch.tensor([[0.1, -0.2, 0.0]])
    s_ref = torch.tensor([[1.5]])
    R = torch.eye(3).unsqueeze(0)
    k_d = compose_kd(kp_ref, R, s_ref, t_ref)
    expected = kp_ref * 1.5
    expected[..., 0:2] = expected[..., 0:2] + t_ref[:, None, 0:2]
    assert torch.allclose(k_d, expected, atol=1e-5)


def test_compose_kd_shape():
    kp_ref = torch.randn(1, 21, 3)
    R = torch.eye(3).unsqueeze(0)
    t_ref = torch.zeros(1, 3); s_ref = torch.ones(1, 1)
    assert compose_kd(kp_ref, R, s_ref, t_ref).shape == (1, 21, 3)
```

Run: expect ImportError.

- [ ] **Step 2: Implement**

```python
# src/arkit_bridge/closed_form_pose.py
"""Closed-form ARKit head Euler -> LivePortrait implicit keypoints `k_d`.

Mirrors `~/w/PersonaLive/src/liveportrait/motion_extractor.py:get_kp`:
    kp_transformed = kp.view(bs, num_kp, 3) @ rot_mat
    kp_transformed *= scale[..., None]
    kp_transformed[:, :, 0:2] += t[:, None, 0:2]   # tx, ty only

Per the paper (arxiv 2512.11253 eqn. 3): k_d = s_d · k_{c,s} · R_d + t_d.
We hold s_d ≈ s_s, t_d ≈ t_s (canonical PersonaLive default
`s_scale=0, t_scale=0.5`); only R_d varies per frame.
"""

import torch


def euler_to_rotmat(yaw: torch.Tensor, pitch: torch.Tensor,
                    roll: torch.Tensor) -> torch.Tensor:
    """3x3 rotation matrix from yaw/pitch/roll (radians).

    Matches PersonaLive's `get_rotation_matrix` (camera.py:31-73): builds
    Rz @ Ry @ Rx then returns the transpose, so `kp @ R` (row-vector
    convention used by motion_extractor.py:72) agrees with PersonaLive.
    """
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cr, sr = torch.cos(roll), torch.sin(roll)

    z0 = torch.zeros_like(cy); o = torch.ones_like(cy)
    Ry = torch.stack([
        torch.stack([cy, z0, sy], dim=-1),
        torch.stack([z0, o,  z0], dim=-1),
        torch.stack([-sy, z0, cy], dim=-1),
    ], dim=-2)
    Rx = torch.stack([
        torch.stack([o, z0, z0], dim=-1),
        torch.stack([z0, cp, -sp], dim=-1),
        torch.stack([z0, sp, cp], dim=-1),
    ], dim=-2)
    Rz = torch.stack([
        torch.stack([cr, -sr, z0], dim=-1),
        torch.stack([sr, cr, z0], dim=-1),
        torch.stack([z0, z0, o], dim=-1),
    ], dim=-2)
    R = Rz @ Ry @ Rx
    return R.transpose(-1, -2)


def compose_kd(kp_ref: torch.Tensor, R: torch.Tensor,
               s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """k_d = (kp_ref @ R) * s + (t_x, t_y, 0)."""
    if R.dim() == 2:
        R = R.unsqueeze(0)
    if s.dim() == 1:
        s = s.unsqueeze(-1)
    if s.dim() == 2 and s.shape[-1] != 1:
        s = s[..., :1]
    if t.dim() == 1:
        t = t.unsqueeze(0)

    k = kp_ref @ R
    k = k * s.unsqueeze(-1)
    k = k.clone()
    k[..., 0:2] = k[..., 0:2] + t[:, None, 0:2]
    return k
```

- [ ] **Step 3: Run + commit**

```bash
PYTHONPATH=src ~/w/PersonaLive/.venv/bin/pytest tests/arkit_bridge/test_closed_form_pose.py -v
git add src/arkit_bridge/closed_form_pose.py tests/arkit_bridge/test_closed_form_pose.py
git commit -m "feat(arkit_bridge): closed-form ARKit Euler -> LivePortrait k_d"
```

---

## Task 3: PersonaLive teachers (frozen)

**Files:**
- Create: `src/arkit_bridge/teacher_personalive.py`
- Create: `tests/arkit_bridge/test_teacher_personalive.py`

Teachers needed:
- `load_motion_encoder(device)` — frozen `MotEncoder` (training teacher)
- `load_motion_extractor(device)` — frozen `MotionExtractor` (used at
  pair-extraction time + at session start in inference; expose detector raw)
- `load_pose_guider(device)` — frozen `PoseGuider` (used at inference and
  during sign-flip calibration)

- [ ] **Step 1: Failing test**

```python
# tests/arkit_bridge/test_teacher_personalive.py
import os, pytest, torch

PERSONA_ME = os.path.expanduser("~/w/PersonaLive/pretrained_weights/personalive/motion_encoder.pth")
PERSONA_MX = os.path.expanduser("~/w/PersonaLive/pretrained_weights/personalive/motion_extractor.pth")
PERSONA_PG = os.path.expanduser("~/w/PersonaLive/pretrained_weights/personalive/pose_guider.pth")


@pytest.mark.skipif(not all(os.path.exists(p) for p in [PERSONA_ME, PERSONA_MX, PERSONA_PG]),
                    reason="PersonaLive weights missing")
def test_motion_encoder_loads_and_runs():
    from arkit_bridge.teacher_personalive import load_motion_encoder
    me = load_motion_encoder(device="cpu")
    assert all(not p.requires_grad for p in me.parameters())
    with torch.no_grad():
        y = me(torch.zeros(1, 3, 1, 224, 224))
    assert y.shape == (1, 1, 32, 16)


@pytest.mark.skipif(not os.path.exists(PERSONA_MX), reason="weights missing")
def test_motion_extractor_loads_and_runs():
    from arkit_bridge.teacher_personalive import load_motion_extractor
    mx = load_motion_extractor(device="cpu")
    with torch.no_grad():
        kp = mx(torch.zeros(1, 3, 256, 256))
    assert kp.shape == (1, 21, 3)


@pytest.mark.skipif(not os.path.exists(PERSONA_PG), reason="weights missing")
def test_pose_guider_loads_and_runs():
    from arkit_bridge.teacher_personalive import load_pose_guider
    pg = load_pose_guider(device="cpu")
    with torch.no_grad():
        y = pg(torch.zeros(1, 3, 1, 512, 512))
    assert y.shape == (1, 320, 1, 64, 64)
```

- [ ] **Step 2: Implement**

```python
# src/arkit_bridge/teacher_personalive.py
import os, sys
import torch

_PERSONA = os.path.expanduser("~/w/PersonaLive")
if _PERSONA not in sys.path:
    sys.path.insert(0, _PERSONA)

PERSONA_ME = os.path.join(_PERSONA, "pretrained_weights/personalive/motion_encoder.pth")
PERSONA_MX = os.path.join(_PERSONA, "pretrained_weights/personalive/motion_extractor.pth")
PERSONA_PG = os.path.join(_PERSONA, "pretrained_weights/personalive/pose_guider.pth")


def _freeze(m):
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def load_motion_encoder(device="cuda"):
    from src.models.motion_encoder.encoder import MotEncoder
    me = MotEncoder()
    me.load_state_dict(torch.load(PERSONA_ME, map_location="cpu"))
    return _freeze(me).to(device)


def load_motion_extractor(device="cuda"):
    from src.liveportrait.motion_extractor import MotionExtractor

    class _Wrapped(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return self.inner(x)              # (B, 21, 3) post-pose

        def detect_raw(self, x):
            return self.inner.detector(x)     # raw kp_info dict

    mx = MotionExtractor(num_kp=21)
    mx.load_state_dict(
        torch.load(PERSONA_MX, map_location="cpu"),
        strict=False,
    )
    return _freeze(_Wrapped(mx)).to(device)


def load_pose_guider(device="cuda"):
    from src.models.pose_guider import PoseGuider
    pg = PoseGuider(conditioning_embedding_channels=320,
                    block_out_channels=(16, 32, 96, 256))
    pg.load_state_dict(torch.load(PERSONA_PG, map_location="cpu"))
    return _freeze(pg).to(device)
```

- [ ] **Step 3: Run + commit**

```bash
PYTHONPATH=src ~/w/PersonaLive/.venv/bin/pytest tests/arkit_bridge/test_teacher_personalive.py -v
git add src/arkit_bridge/teacher_personalive.py tests/arkit_bridge/test_teacher_personalive.py
git commit -m "feat(arkit_bridge): frozen PersonaLive teachers"
```

---

## Task 4: Student model

**Files:** `src/arkit_bridge/student.py`, `tests/arkit_bridge/test_student.py`

- [ ] **Step 1: Test**

```python
# tests/arkit_bridge/test_student.py
import torch
from arkit_bridge.student import MotEncoderStudent

def test_output_shape():
    out = MotEncoderStudent()(torch.zeros(2, 58))
    assert out.shape == (2, 1, 32, 16)

def test_output_finite():
    assert torch.isfinite(MotEncoderStudent()(torch.randn(4, 58))).all()

def test_zero_init_outputs_zero():
    out = MotEncoderStudent()(torch.randn(4, 58))
    assert torch.allclose(out, torch.zeros_like(out))

def test_param_count_under_3M():
    n = sum(p.numel() for p in MotEncoderStudent().parameters())
    assert n < 3_000_000, n
```

- [ ] **Step 2: Implement**

```python
# src/arkit_bridge/student.py
"""MLP student: ARKit expression channels (58) -> motion_encoder feature.

Input layout (58):
    [0:52]   Apple ARKit blendshape coefficients in [0..1]
    [52:55]  LeftEye yaw/pitch/roll (radians, Live Link)
    [55:58]  RightEye yaw/pitch/roll (radians, Live Link)

Output: (B, 1, 32, 16) — matches MotEncoder per-frame slice.
"""

import torch
import torch.nn as nn


class MotEncoderStudent(nn.Module):
    def __init__(self, in_dim: int = 58, out_l: int = 32, out_c: int = 16,
                 hidden: tuple = (256, 256, 256)):
        super().__init__()
        self.out_l = out_l; self.out_c = out_c
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.SiLU(inplace=True)]
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(prev, out_l * out_c)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, b_expr: torch.Tensor) -> torch.Tensor:
        h = self.trunk(b_expr)
        return self.head(h).view(-1, 1, self.out_l, self.out_c)
```

- [ ] **Step 3: Run + commit**

```bash
PYTHONPATH=src ~/w/PersonaLive/.venv/bin/pytest tests/arkit_bridge/test_student.py -v
git add src/arkit_bridge/student.py tests/arkit_bridge/test_student.py
git commit -m "feat(arkit_bridge): MotEncoder MLP student"
```

---

## Task 5: Pair-extraction script

**Files:** `scripts/extract_arkit_pairs.py`

For each take, build (b_expr, m_f) pairs by running real `motion_encoder` on
a 224² face crop. The closed-form path doesn't need training pairs — it's
used directly at inference, calibrated once via Task 9.

- [ ] **Step 1: Implement**

```python
# scripts/extract_arkit_pairs.py
"""Extract (b_expr, m_f) pairs from a Live Link Face take.

For each sampled frame i:
  1. Read MOV frame i.
  2. Loose-crop 224x224 with face roughly centered.
  3. Frozen motion_encoder(crop_224.unsqueeze(2)) -> m_f (1,1,32,16).
  4. b_expr <- CSV[i, 0:52] + CSV[i, 55:61].
  5. Save {b_expr, m_f, frame_idx} pkl.

Resumable (skips frames whose pkl already exists).
"""
import argparse, os, pickle, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkit_bridge.llf_csv import load_llf_b61                       # noqa: E402
from arkit_bridge.teacher_personalive import load_motion_encoder    # noqa: E402


def find_take_files(take_dir):
    movs = sorted(take_dir.glob("*_iPhone.mov"))
    csvs = sorted(take_dir.glob("*_iPhone.csv"))
    if not movs or not csvs:
        raise FileNotFoundError(f"no *_iPhone.mov/csv in {take_dir}")
    return movs[0], csvs[0]


def loose_crop_centered(rgb, target):
    import cv2
    h, w = rgb.shape[:2]
    cx, cy = w // 2, int(h * 0.45)
    side = min(h, w); half = side // 2
    return cv2.resize(rgb[max(0,cy-half):min(h,cy+half),
                          max(0,cx-half):min(w,cx+half)],
                      (target, target), interpolation=cv2.INTER_AREA)


def iter_frames(video_path, stride):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            yield idx, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        idx += 1
    cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--take_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_frames", type=int, default=0)
    args = ap.parse_args()

    take_dir = Path(args.take_dir)
    mov, csv = find_take_files(take_dir)
    b_all = load_llf_b61(csv)

    me = load_motion_encoder(device=args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    n_done = n_skipped = n_kept = 0
    for fi, rgb in iter_frames(mov, args.stride):
        if fi >= len(b_all):
            break
        out_path = os.path.join(args.out_dir, f"frame_{fi:06d}.pkl")
        if os.path.exists(out_path):
            n_skipped += 1; continue
        crop = loose_crop_centered(rgb, 224)
        x = torch.from_numpy(crop).float().permute(2,0,1).unsqueeze(0).unsqueeze(2) / 255.0
        with torch.no_grad():
            m_f = me(x.to(args.device)).squeeze(0).cpu().numpy()
        b_expr = np.concatenate([b_all[fi, :52], b_all[fi, 55:61]], axis=0).astype(np.float32)
        with open(out_path, "wb") as f:
            pickle.dump({"b_expr": b_expr,
                         "m_f": m_f.astype(np.float16),
                         "frame_idx": int(fi)}, f)
        n_done += 1; n_kept += 1
        if n_done % 100 == 0:
            print(f"  {n_done} pairs (skipped={n_skipped})")
        if args.max_frames and n_kept >= args.max_frames:
            break
    print(f"done: kept={n_kept} skipped={n_skipped}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke + sanity-inspect + commit**

```bash
PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  systemd-run --user --scope -p MemoryMax=45G \
  ~/w/PersonaLive/.venv/bin/python scripts/extract_arkit_pairs.py \
    --take_dir data/llf-takes/20260505_MySlate_2 \
    --out_dir data/arkit_bridge_pairs/smoke \
    --stride 4 --max_frames 50

~/w/PersonaLive/.venv/bin/python -c "
import pickle, numpy as np
d = pickle.load(open('data/arkit_bridge_pairs/smoke/frame_000000.pkl','rb'))
print('b_expr', d['b_expr'].shape, 'l1', float(np.abs(d['b_expr']).sum()))
print('m_f', d['m_f'].shape, 'std', float(np.array(d['m_f']).std()))"

echo "data/arkit_bridge_pairs/" >> .gitignore
git add .gitignore scripts/extract_arkit_pairs.py
git commit -m "feat(arkit_bridge): pair extraction (b_expr, m_f) from LLF takes"
```

---

## Task 6: Distill loop

**Files:** `src/arkit_bridge/dataset.py`, `src/arkit_bridge/distill.py`,
`scripts/train_arkit_student.py`

- [ ] **Step 1: Dataset + distill loop + CLI**

```python
# src/arkit_bridge/dataset.py
import pickle
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class PairDataset(Dataset):
    def __init__(self, root):
        self.paths = sorted(Path(root).glob("frame_*.pkl"))
        if not self.paths:
            raise FileNotFoundError(f"no frame pkls under {root}")

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        with open(self.paths[i], "rb") as f:
            d = pickle.load(f)
        return (torch.from_numpy(np.asarray(d["b_expr"], dtype=np.float32)),
                torch.from_numpy(np.asarray(d["m_f"], dtype=np.float32)))
```

```python
# src/arkit_bridge/distill.py
import json, os, time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from arkit_bridge.dataset import PairDataset
from arkit_bridge.student import MotEncoderStudent


def train(pairs_dir, out_dir, *,
          batch_size=64, lr=5e-4, steps=20000,
          log_every=100, ckpt_every=2000, device="cuda"):
    os.makedirs(out_dir, exist_ok=True)
    ds = PairDataset(pairs_dir)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                    num_workers=2, drop_last=True, persistent_workers=True)
    s = MotEncoderStudent().to(device)
    opt = torch.optim.AdamW(s.parameters(), lr=lr)
    log = []; step = 0; t0 = time.time()
    while step < steps:
        for b, m in dl:
            b = b.to(device); m = m.to(device)
            loss = F.mse_loss(s(b), m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step % log_every == 0:
                rate = step / max(1e-6, time.time() - t0)
                print(f"step {step:6d}  loss {loss.item():.5f}  ({rate:.1f}/s)")
                log.append({"step": step, "loss": float(loss)})
            if step % ckpt_every == 0 or step >= steps:
                torch.save(s.state_dict(), Path(out_dir) / f"student_step{step:06d}.pt")
                json.dump(log, open(Path(out_dir) / "log.json", "w"))
            if step >= steps:
                break
    return s
```

```python
# scripts/train_arkit_student.py
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from arkit_bridge.distill import train  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    train(args.pairs_dir, args.out_dir,
          batch_size=args.batch_size, lr=args.lr, steps=args.steps,
          device=args.device)
```

- [ ] **Step 2: Smoke train (overfit on 50 pairs) + ratio check**

```bash
PYTHONPATH=src ~/w/PersonaLive/.venv/bin/python scripts/train_arkit_student.py \
  --pairs_dir data/arkit_bridge_pairs/smoke \
  --out_dir exp_output/arkit_bridge/smoke \
  --batch_size 16 --steps 2000 --lr 1e-3

PYTHONPATH=src ~/w/PersonaLive/.venv/bin/python -c "
import torch
from arkit_bridge.student import MotEncoderStudent
from arkit_bridge.dataset import PairDataset
import torch.nn.functional as F
m = MotEncoderStudent().cuda()
m.load_state_dict(torch.load('exp_output/arkit_bridge/smoke/student_step002000.pt'))
ds = PairDataset('data/arkit_bridge_pairs/smoke')
b, mf = ds[0]
with torch.no_grad():
    pred = m(b.unsqueeze(0).cuda()).cpu()
print('overfit MSE:', F.mse_loss(pred, mf.unsqueeze(0)).item())
print('teacher var:', mf.var().item())
print('ratio:', F.mse_loss(pred, mf.unsqueeze(0)).item() / mf.var().item())"
```

Want ratio < 0.1.

- [ ] **Step 3: Commit**

```bash
git add src/arkit_bridge/dataset.py src/arkit_bridge/distill.py scripts/train_arkit_student.py
git commit -m "feat(arkit_bridge): distill loop + smoke overfit"
```

---

## Task 7: Per-channel sensitivity eval

**Files:** `src/arkit_bridge/eval.py`, `scripts/eval_arkit_student.py`

```python
# src/arkit_bridge/eval.py
import json
from pathlib import Path
import torch
import torch.nn.functional as F

from arkit_bridge.dataset import PairDataset
from arkit_bridge.student import MotEncoderStudent

ARKIT_NAMES = [
    "EyeBlinkLeft", "EyeLookDownLeft", "EyeLookInLeft", "EyeLookOutLeft",
    "EyeLookUpLeft", "EyeSquintLeft", "EyeWideLeft",
    "EyeBlinkRight", "EyeLookDownRight", "EyeLookInRight", "EyeLookOutRight",
    "EyeLookUpRight", "EyeSquintRight", "EyeWideRight",
    "JawForward", "JawRight", "JawLeft", "JawOpen",
    "MouthClose", "MouthFunnel", "MouthPucker", "MouthRight", "MouthLeft",
    "MouthSmileLeft", "MouthSmileRight", "MouthFrownLeft", "MouthFrownRight",
    "MouthDimpleLeft", "MouthDimpleRight", "MouthStretchLeft", "MouthStretchRight",
    "MouthRollLower", "MouthRollUpper", "MouthShrugLower", "MouthShrugUpper",
    "MouthPressLeft", "MouthPressRight", "MouthLowerDownLeft", "MouthLowerDownRight",
    "MouthUpperUpLeft", "MouthUpperUpRight",
    "BrowDownLeft", "BrowDownRight", "BrowInnerUp",
    "BrowOuterUpLeft", "BrowOuterUpRight",
    "CheekPuff", "CheekSquintLeft", "CheekSquintRight",
    "NoseSneerLeft", "NoseSneerRight", "TongueOut",
]
EYE_NAMES = ["LeftEyeYaw", "LeftEyePitch", "LeftEyeRoll",
             "RightEyeYaw", "RightEyePitch", "RightEyeRoll"]


def held_out_mse(student, dataset, device="cuda", batch_size=64):
    student.eval()
    total = 0.0; n = 0
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = [dataset[j] for j in range(i, min(i+batch_size, len(dataset)))]
            b = torch.stack([x[0] for x in batch]).to(device)
            m = torch.stack([x[1] for x in batch]).to(device)
            total += F.mse_loss(student(b), m, reduction="sum").item()
            n += m.numel()
    return total / max(n, 1)


def channel_sensitivity(student, dataset, device="cuda"):
    student.eval()
    bs = torch.stack([dataset[i][0] for i in range(len(dataset))])
    b_neutral = bs.median(dim=0).values.to(device)
    out = {}
    with torch.no_grad():
        ref = student(b_neutral.unsqueeze(0))
        for i in range(52):
            b_hi = b_neutral.clone(); b_hi[i] = 1.0
            out[ARKIT_NAMES[i]] = (student(b_hi.unsqueeze(0)) - ref).pow(2).mean().sqrt().item()
        for i, name in enumerate(EYE_NAMES):
            b_hi = b_neutral.clone(); b_hi[52 + i] = b_neutral[52 + i] + 0.5
            out[name] = (student(b_hi.unsqueeze(0)) - ref).pow(2).mean().sqrt().item()
    return out


def main(ckpt, pairs_dir, out_path, device="cuda"):
    s = MotEncoderStudent().to(device)
    s.load_state_dict(torch.load(ckpt, map_location=device))
    ds = PairDataset(pairs_dir)
    mse = held_out_mse(s, ds, device=device)
    sens = channel_sensitivity(s, ds, device=device)
    ranked = sorted(sens.items(), key=lambda kv: -kv[1])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"ckpt": ckpt, "n": len(ds),
               "held_out_mse": mse,
               "sensitivity": sens, "ranked": ranked},
              open(out_path, "w"), indent=2)
    print(f"held-out MSE: {mse:.5f}\ntop 10:")
    for n, v in ranked[:10]:
        print(f"  {n:24s} {v:.5f}")
    print("bottom 5:")
    for n, v in ranked[-5:]:
        print(f"  {n:24s} {v:.5f}")
```

```python
# scripts/eval_arkit_student.py
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from arkit_bridge.eval import main  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pairs_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    main(args.ckpt, args.pairs_dir, args.out, device=args.device)
```

```bash
PYTHONPATH=src ~/w/PersonaLive/.venv/bin/python scripts/eval_arkit_student.py \
  --ckpt exp_output/arkit_bridge/smoke/student_step002000.pt \
  --pairs_dir data/arkit_bridge_pairs/smoke \
  --out exp_output/arkit_bridge/smoke/eval.json
git add src/arkit_bridge/eval.py scripts/eval_arkit_student.py
git commit -m "feat(arkit_bridge): held-out MSE + per-channel sensitivity"
```

---

## Task 8: Real-corpus extraction + real distill

```bash
cd /home/newub/w/vamp-interface
mkdir -p data/arkit_bridge_pairs/real
for d in data/llf-takes/2026*/; do
  base=$(basename "$d")
  if [[ "$base" == "20260505_MySlate_1" ]]; then echo "skip MHA take $base"; continue; fi
  echo "=== $base ==="
  PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    systemd-run --user --scope -p MemoryMax=45G \
    ~/w/PersonaLive/.venv/bin/python scripts/extract_arkit_pairs.py \
      --take_dir "$d" --out_dir data/arkit_bridge_pairs/real --stride 2
done
ls data/arkit_bridge_pairs/real | wc -l   # expect ≥ 16,000

PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  systemd-run --user --scope -p MemoryMax=45G \
  ~/w/PersonaLive/.venv/bin/python scripts/train_arkit_student.py \
    --pairs_dir data/arkit_bridge_pairs/real \
    --out_dir exp_output/arkit_bridge/real \
    --batch_size 128 --steps 30000 --lr 5e-4

PYTHONPATH=src ~/w/PersonaLive/.venv/bin/python scripts/eval_arkit_student.py \
  --ckpt exp_output/arkit_bridge/real/student_step030000.pt \
  --pairs_dir data/arkit_bridge_pairs/real \
  --out exp_output/arkit_bridge/real/eval.json
echo "exp_output/arkit_bridge/*/student_*.pt" >> .gitignore
git add exp_output/arkit_bridge/real/log.json exp_output/arkit_bridge/real/eval.json .gitignore
git commit -m "chore(arkit_bridge): real distill run logs + eval"
```

Expected: ~30–60 min on RTX 5090.

---

## Task 9: ARKit↔LivePortrait Euler sign-flip calibration

**File:** `scripts/calibrate_euler_signs.py`

8-way enumeration over (yaw, pitch, roll) sign combinations: for held-out
frames, render real `motion_extractor` → `k_d_real`; compute closed-form
`k_d_cf` for each combination using ARKit (yaw, pitch, roll); pick the
combo with smallest mean L2 error.

```python
# scripts/calibrate_euler_signs.py
import argparse, json, sys
from itertools import product
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkit_bridge.closed_form_pose import euler_to_rotmat, compose_kd  # noqa: E402
from arkit_bridge.llf_csv import load_llf_b61                          # noqa: E402
from arkit_bridge.teacher_personalive import load_motion_extractor     # noqa: E402


def loose_crop_centered(rgb, target):
    import cv2
    h, w = rgb.shape[:2]
    cx, cy = w // 2, int(h * 0.45); side = min(h, w); half = side // 2
    return cv2.resize(rgb[max(0,cy-half):min(h,cy+half),
                          max(0,cx-half):min(w,cx+half)],
                      (target, target), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--take_dir", required=True)
    ap.add_argument("--n_frames", type=int, default=20)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import cv2
    take_dir = Path(args.take_dir)
    mov = next(take_dir.glob("*_iPhone.mov"))
    csv = next(take_dir.glob("*_iPhone.csv"))
    b_all = load_llf_b61(csv)
    mx = load_motion_extractor(device=args.device)

    cap = cv2.VideoCapture(str(mov))
    ok, frame = cap.read()
    rgb0 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    crop0 = loose_crop_centered(rgb0, 256)
    x0 = torch.from_numpy(crop0).float().permute(2,0,1).unsqueeze(0).to(args.device) / 255.0
    with torch.no_grad():
        info = mx.detect_raw(x0)
    kp_ref = info["kp"].reshape(1, -1, 3); t_ref = info["t"]; s_ref = info["scale"]

    samples = []
    for k in range(args.n_frames):
        fi = (k + 1) * args.stride
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok or fi >= len(b_all): break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        crop = loose_crop_centered(rgb, 256)
        x = torch.from_numpy(crop).float().permute(2,0,1).unsqueeze(0).to(args.device) / 255.0
        with torch.no_grad():
            k_d_real = mx(x)
        b = b_all[fi]
        samples.append((k_d_real.cpu(),
                        torch.tensor([b[52], b[53], b[54]], dtype=torch.float32)))
    cap.release()

    results = {}
    for s_y, s_p, s_r in product([+1, -1], repeat=3):
        errs = []
        for k_d_real, ypr in samples:
            R = euler_to_rotmat(torch.tensor(s_y * ypr[0]),
                                torch.tensor(s_p * ypr[1]),
                                torch.tensor(s_r * ypr[2]))
            k_d_cf = compose_kd(kp_ref.cpu(), R.unsqueeze(0),
                                s_ref.cpu(), t_ref.cpu())
            errs.append((k_d_cf - k_d_real).pow(2).mean().sqrt().item())
        results[f"({s_y:+d}, {s_p:+d}, {s_r:+d})"] = float(np.mean(errs))
    best = min(results, key=results.get)
    print("Sign combo (yaw,pitch,roll) -> mean keypoint L2 err:")
    for k, v in sorted(results.items(), key=lambda kv: kv[1]):
        marker = " <-- best" if k == best else ""
        print(f"  {k}  err={v:.5f}{marker}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"results": results, "best": best, "n": len(samples)},
              open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
```

```bash
PYTHONPATH=src ~/w/PersonaLive/.venv/bin/python scripts/calibrate_euler_signs.py \
  --take_dir data/llf-takes/20260505_MySlate_2 \
  --n_frames 30 --stride 100 \
  --out exp_output/arkit_bridge/calibration/myslate_2.json

PYTHONPATH=src ~/w/PersonaLive/.venv/bin/python scripts/calibrate_euler_signs.py \
  --take_dir data/llf-takes/20260505_MySlate_5 \
  --n_frames 30 --stride 100 \
  --out exp_output/arkit_bridge/calibration/myslate_5.json

git add scripts/calibrate_euler_signs.py exp_output/arkit_bridge/calibration/
git commit -m "feat(arkit_bridge): Euler sign-convention calibration"
```

Expected: same combo wins on both takes.

---

## Task 10: Drop-in PersonaLive smoke render

**File:** `scripts/apply_bridge_to_personalive.py`

End-to-end: reference photo + 1-second slice of MySlate_2's b₆₁ →
PersonaLive renders 24 frames driven entirely by ARKit.

The script needs the exact wrapper-construction call in
`~/w/PersonaLive/inference_offline.py`. Skeleton only:

```python
# scripts/apply_bridge_to_personalive.py
"""Render PersonaLive frames driven by ARKit b₆₁ alone.

Wires both seams:
  - pose_guider gets a closed-form k_d -> draw_keypoints -> real PoseGuider feature.
  - motion_encoder driving slot gets MotEncoderStudent(b_expr); reference slot
    is the real motion_encoder run on the reference RGB once.
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, os.path.expanduser("~/w/PersonaLive"))

from arkit_bridge.closed_form_pose import euler_to_rotmat, compose_kd
from arkit_bridge.llf_csv import load_llf_b61
from arkit_bridge.student import MotEncoderStudent
from arkit_bridge.teacher_personalive import (
    load_motion_encoder, load_motion_extractor, load_pose_guider,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--take_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--euler_signs", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_frames", type=int, default=24)
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()
    raise NotImplementedError(
        "Wire to ~/w/PersonaLive/inference_offline.py: build wrapper, "
        "monkey-patch wrapper.pose_guider.__call__ and "
        "wrapper.motion_encoder.__call__ with adapter callables. The "
        "reference motion_hidden_state is precomputed once via real "
        "motion_encoder on the reference RGB.")


if __name__ == "__main__":
    main()
```

Inspect first:

```bash
sed -n '1,120p' ~/w/PersonaLive/inference_offline.py
```

Identify the wrapper-construction call. Build the adapters around the
callable seams shown in the design doc. Render. Visual-check identity, head
pose, expression. Commit script + first render.

```bash
git add scripts/apply_bridge_to_personalive.py exp_output/arkit_bridge/render/smoke/
git commit -m "feat(arkit_bridge): PersonaLive drop-in render driven by ARKit only"
```

---

## Task 11: Per-component perf benchmark (RTX 5090 fp16)

**Files:**
- Create: `scripts/bench_personalive_components.py`
- Create: `exp_output/perf/personalive-component-budget.json`

Replace the speculative numbers in
`docs/research/2026-05-05-personalive-architecture-notes.md` with measured
ones. Drives whether the MotEncoder student is worth the trouble (if
the real motion_encoder costs <1 ms/frame at T=4 on 5090 fp16, the
distill is for plumbing convenience, not perf).

- [ ] **Step 1: Write the benchmark script**

```python
"""Per-component latency on a real PersonaLive load. RTX 5090, fp16, T=4.

Reports median ms over 20 reps after 5 warmup reps, with cuda.synchronize
around each timed section. Output: exp_output/perf/personalive-component-budget.json
"""
import json, time, os, sys
from pathlib import Path
import torch

PL = Path(os.path.expanduser("~/w/PersonaLive"))
sys.path.insert(0, str(PL / "src"))

from wrapper import build_pipeline_for_bench  # see Task 3 hooks; or load each module directly

DEVICE = "cuda"
DTYPE = torch.float16
T = 4
H = W = 512
REPS = 20
WARMUP = 5

def timed(fn, *args, **kwargs):
    for _ in range(WARMUP):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    times = []
    for _ in range(REPS):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn(*args, **kwargs)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[REPS // 2]  # median ms

def main():
    pipe = build_pipeline_for_bench(device=DEVICE, dtype=DTYPE)
    out = {}
    # 1. motion_extractor on T driving frames (224x224)
    drv = torch.randn(T, 3, 224, 224, device=DEVICE, dtype=DTYPE)
    out["motion_extractor_T4_ms"] = timed(lambda: pipe.motion_extractor(drv))
    # 2. draw_keypoints (CPU op on numpy; time on host)
    import numpy as np
    kp = np.random.randn(T, 21, 3).astype(np.float32)
    from utils.util import draw_keypoints
    def draw_all():
        for i in range(T):
            draw_keypoints(kp[i:i+1], 512, 512)
    out["draw_keypoints_T4_ms"] = timed(draw_all)
    # 3. pose_guider: (1, 3, T, 64, 64) -> (1, 320, T, 64, 64)
    pg_in = torch.randn(1, 3, T, 64, 64, device=DEVICE, dtype=DTYPE)
    out["pose_guider_T4_ms"] = timed(lambda: pipe.pose_guider(pg_in))
    # 4. motion_encoder: (1, 3, T+1, 224, 224) -> (1, T+1, 32, 16)
    me_in = torch.randn(1, 3, T + 1, 224, 224, device=DEVICE, dtype=DTYPE)
    out["motion_encoder_Tplus1_ms"] = timed(lambda: pipe.motion_encoder(me_in))
    # 5. denoising UNet: per-step. Run a single step with realistic shapes.
    z_t = torch.randn(1, 4, T, 64, 64, device=DEVICE, dtype=DTYPE)
    timestep = torch.tensor([500], device=DEVICE)
    # encoder_hidden_states / pose_fea / motion_emb shapes per pipeline
    enc = torch.randn(1, 1, 768, device=DEVICE, dtype=DTYPE)  # placeholder; use real ref enc
    pose_fea = torch.randn(1, 320, T, 64, 64, device=DEVICE, dtype=DTYPE)
    mot = torch.randn(1, T + 1, 32, 16, device=DEVICE, dtype=DTYPE)
    def unet_step():
        with torch.no_grad():
            pipe.denoising_unet(z_t, timestep, encoder_hidden_states=enc,
                                pose_cond_fea=pose_fea, motion_emb=mot)
    out["denoising_unet_step_ms"] = timed(unet_step)
    out["denoising_unet_4steps_ms"] = out["denoising_unet_step_ms"] * 4
    # 6. VAE decode T frames
    lat = torch.randn(T, 4, 64, 64, device=DEVICE, dtype=DTYPE)
    out["vae_decode_T4_ms"] = timed(lambda: pipe.vae.decode(lat / pipe.vae.config.scaling_factor))
    # End-to-end estimate (single chunk, T=4, 4 steps)
    out["chunk_estimate_ms"] = (
        out["motion_extractor_T4_ms"]
        + out["draw_keypoints_T4_ms"]
        + out["pose_guider_T4_ms"]
        + out["motion_encoder_Tplus1_ms"]
        + out["denoising_unet_4steps_ms"]
        + out["vae_decode_T4_ms"]
    )
    out["chunk_fps"] = 1000.0 * T / out["chunk_estimate_ms"]
    out["env"] = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "dtype": str(DTYPE), "T": T, "reps": REPS, "warmup": WARMUP,
    }
    Path("exp_output/perf").mkdir(parents=True, exist_ok=True)
    Path("exp_output/perf/personalive-component-budget.json").write_text(
        json.dumps(out, indent=2)
    )
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
```

`build_pipeline_for_bench` is a thin loader that constructs the same
modules as `inference_offline.py:175-186` but skips the scheduler and
reference-frame priming — write it next to the existing teachers helper
in Task 3.

- [ ] **Step 2: Run and commit JSON**

```bash
PYTHONPATH=src python scripts/bench_personalive_components.py
git add exp_output/perf/personalive-component-budget.json scripts/bench_personalive_components.py
git commit -m "perf(arkit-bridge): per-component latency budget on RTX 5090 fp16 T=4"
```

- [ ] **Step 3: Replace speculative table in arch notes**

Edit `docs/research/2026-05-05-personalive-architecture-notes.md` —
swap the `(speculative)` perf section for the JSON's measured numbers
and recompute the bottleneck ranking. Cite the JSON path.

**Acceptance:** JSON exists, chunk_estimate_ms is in the 50–150 ms
range we expect, arch notes no longer say `(speculative)` next to the
per-component numbers.

---

## Task 12: Readout + decision

**File:** `docs/research/2026-05-05-arkit-bridge-v1-readout.md`

Capture: corpus size, training steps, final held-out MSE, per-channel
sensitivity ranking, calibration result (chosen Euler signs), qualitative
notes from inference render, channels with collapse, **measured perf
budget from Task 11**, recommendation.

```markdown
---
status: live
topic: neural-deformation-control
---

# ARKit→PersonaLive bridge v1 — readout (date)

Companion to [`2026-05-05-arkit-bridge-v1-plan.md`](2026-05-05-arkit-bridge-v1-plan.md).

## Numbers
- corpus: N pkls from 7 ARKit takes
- steps: K
- final held-out MSE: X
- teacher m_f variance: Y
- variance fraction explained: 1 - X/Y = Z

## Calibration
- chosen Euler signs (yaw, pitch, roll): ...
- mean keypoint L2 error at chosen combo: ...

## Per-channel sensitivity
- top 10 / bottom 5 (paste from eval.json)

## Qualitative inference render
- reference: ...
- driving slice: ...
- failure modes: ...

## Decision
- proceed to "replace" (diffusion-loss) fine-tune? yes/no
- channels needing more corpus diversity: ...
- next iteration plan: ...
```

```bash
git add docs/research/2026-05-05-arkit-bridge-v1-readout.md docs/research/_topics/neural-deformation-control.md
git commit -m "docs(neural-deformation): bridge v1 readout"
```

---

## Self-review

**Spec coverage:** Tasks 1–11 cover supersession, closed-form pose math,
PersonaLive teachers, student model, pair extraction, distill, eval, real
run, sign-flip calibration, drop-in render, readout.

**Placeholder scan:** Task 10 has one explicit TBD — exact PersonaLive
wrapper-construction signature in `inference_offline.py`. Acceptable; same
pattern as in the prior plan.

**Type consistency:** Student output `(B,1,32,16)` ≡ MotEncoder per-frame
slice. `compose_kd` output `(B,21,3)` ≡ MotionExtractor's posed-keypoint
output, shape-matched to `draw_keypoints` input.

**Out-of-scope:** No diffusion-loss "replace" fine-tune, no Stage-2
temporal modules, no iOS/transport, no real-time streaming. All gated on
Task 11 readout.

## Execution handoff

Tasks 1–7 are GPU-light. Task 8 needs ~30–60 min on the 5090. Task 9 is
fast (one-shot calibration). Task 10 is one inference render. Task 11 is
documentation.
