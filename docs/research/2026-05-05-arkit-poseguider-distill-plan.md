---
status: superseded
topic: neural-deformation-control
superseded_by: 2026-05-05-arkit-bridge-v1-plan.md
---

> **Superseded 2026-05-05.** PoseGuider does not consume DWPose; it consumes
> a `draw_keypoints` rendering of LivePortrait 3D implicit keypoints.
> Replaced by the paper-faithful single-student design in
> [`2026-05-05-arkit-bridge-v1-design.md`](2026-05-05-arkit-bridge-v1-design.md)
> + [`2026-05-05-arkit-bridge-v1-plan.md`](2026-05-05-arkit-bridge-v1-plan.md).

# ARKit→PoseGuider distillation — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a parametric encoder that ingests Live Link Face's 52 ARKit blendshapes + 9 rotation floats and produces a 64×64×320 conditioning feature map matching PersonaLive's frozen PoseGuider output, drop-in compatible with PersonaLive's denoising UNet at inference.

**Architecture:** Knowledge-distillation. Teacher = frozen PersonaLive PoseGuider on DWPose images (size `(B,3,1,512,512)` → `(B,320,1,64,64)`). Student = `ARKitParametricPoseGuider` taking 61 floats and producing the same 4-D output via Linear→reshape→ConvTranspose2d stack. Per-frame supervised MSE on `(b₆₁, T)` pairs extracted from RGB face video. Reference frame is canonicalized to fix the canvas; driving frames are loosely cropped only, so head pose is preserved in both teacher target and student input.

**Tech stack:** Python 3.12, PyTorch 2.11+cu128, mediapipe (FaceLandmarker), DWPose (controlnet_aux), Moore-AnimateAnyone PoseGuider class, PersonaLive `pose_guider.pth` weights, vamp-interface `.venv`.

---

## Design summary

Why this shape: PoseGuider's job is to produce a spatial bias added to the noisy latent at canvas resolution. Spatial structure in the conditioning is necessary because the denoising UNet was trained to read it that way. Our student must produce the same shape `(B,320,1,64,64)` from non-spatial input. It does so by learning, from supervised pairs, to reconstruct the teacher's spatial activation given expression + pose floats — exploiting that PersonaLive's reference-frame canonical alignment fixes the canvas coordinate system.

Channel breakdown of the 61-d input:
- `b[0:52]` ARKit blendshapes (expression).
- `b[52:55]` head yaw/pitch/roll (rad).
- `b[55:58]` left eye yaw/pitch/roll (rad).
- `b[58:61]` right eye yaw/pitch/roll (rad).

Why distill before replace: distill is a few-hour MSE problem with no diffusion in the loop. It validates the architectural premise and gives a working pipeline immediately. "Replace" (later, separate plan) fine-tunes through the frozen denoising UNet to recover long-tail AUs that DWPose teacher loses.

Out of scope here:
- Stage-2 / temporal modules (gated, no Moore proxy).
- The downstream replace fine-tune (separate plan, gated on per-channel results from this).
- iOS app or live transport (separate thread).
- Full diffusion-loss training (replace).

## File structure

```
src/arkit_bridge/
  __init__.py
  encoder.py      # ARKitParametricPoseGuider student
  teacher.py      # load + freeze PersonaLive PoseGuider
  extractors.py   # MediaPipe FaceLandmarker → b₆₁; DWPose → image
  dataset.py      # iterate cached pair pkls
  distill.py      # train loop
  eval.py         # per-channel sensitivity sweep
tests/arkit_bridge/
  __init__.py
  test_encoder.py
  test_extractors.py
  test_teacher.py
scripts/
  extract_distill_pairs.py  # video → (b₆₁, T) pkls
  train_arkit_distill.py    # CLI for distill training
  eval_arkit_distill.py     # CLI for per-channel eval
exp_output/arkit_distill/
  ckpt/                     # student .pt files
  logs/                     # train logs
  eval/                     # per-channel readouts + viz
docs/research/
  2026-05-05-arkit-poseguider-distill-plan.md  # this file
  2026-05-05-arkit-poseguider-distill-readout.md  # findings (Task 10)
```

---

## Task 1: Project skeleton + design doc commit

**Files:**
- Create: `src/arkit_bridge/__init__.py` (empty)
- Create: `tests/arkit_bridge/__init__.py` (empty)
- Verify: `docs/research/2026-05-05-arkit-poseguider-distill-plan.md` (this file)
- Verify: `docs/research/_topics/neural-deformation-control.md` mentions this plan

- [ ] **Step 1: Create skeleton dirs**

```bash
mkdir -p /home/newub/w/vamp-interface/src/arkit_bridge \
         /home/newub/w/vamp-interface/tests/arkit_bridge \
         /home/newub/w/vamp-interface/exp_output/arkit_distill/{ckpt,logs,eval}
touch /home/newub/w/vamp-interface/src/arkit_bridge/__init__.py \
      /home/newub/w/vamp-interface/tests/arkit_bridge/__init__.py
```

- [ ] **Step 2: Add pointer in topic index**

Append to `/home/newub/w/vamp-interface/docs/research/_topics/neural-deformation-control.md`:

```markdown
- 2026-05-05 distill plan: [`2026-05-05-arkit-poseguider-distill-plan.md`](../2026-05-05-arkit-poseguider-distill-plan.md) — parametric encoder distilled from frozen PoseGuider on DWPose; (b₆₁) → (320,1,64,64) drop-in.
```

- [ ] **Step 3: Commit skeleton**

```bash
cd /home/newub/w/vamp-interface
git add docs/research/2026-05-05-arkit-poseguider-distill-plan.md \
        docs/research/_topics/neural-deformation-control.md \
        src/arkit_bridge/__init__.py \
        tests/arkit_bridge/__init__.py
git commit -m "docs(neural-deformation): ARKit→PoseGuider distill plan + skeleton"
```

---

## Task 2: Student encoder (`ARKitParametricPoseGuider`)

**Files:**
- Create: `src/arkit_bridge/encoder.py`
- Create: `tests/arkit_bridge/test_encoder.py`

- [ ] **Step 1: Write the failing test**

Write to `tests/arkit_bridge/test_encoder.py`:

```python
import torch
from src.arkit_bridge.encoder import ARKitParametricPoseGuider


def test_output_shape_matches_pose_guider():
    student = ARKitParametricPoseGuider()
    b = torch.zeros(2, 61)
    out = student(b)
    assert out.shape == (2, 320, 1, 64, 64), out.shape


def test_output_is_finite_under_random_input():
    student = ARKitParametricPoseGuider()
    b = torch.randn(4, 61)
    out = student(b)
    assert torch.isfinite(out).all()


def test_zero_input_does_not_crash():
    student = ARKitParametricPoseGuider()
    b = torch.zeros(1, 61)
    out = student(b)
    assert out.shape[1:] == (320, 1, 64, 64)


def test_param_count_under_5M():
    student = ARKitParametricPoseGuider()
    n = sum(p.numel() for p in student.parameters())
    assert n < 5_000_000, f"too big: {n}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/newub/w/vamp-interface && .venv/bin/pytest tests/arkit_bridge/test_encoder.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.arkit_bridge.encoder'`.

- [ ] **Step 3: Implement encoder**

Write to `src/arkit_bridge/encoder.py`:

```python
"""Parametric PoseGuider student: 61 floats -> (B, 320, 1, 64, 64)."""

import torch
import torch.nn as nn


class ARKitParametricPoseGuider(nn.Module):
    """Drop-in replacement for PersonaLive's PoseGuider at inference.

    Input: (B, 61) — 52 ARKit blendshapes + 9 rotation floats
        b[:, 0:52]   ARKit blendshape coefficients (clamped 0..1 by ARKit)
        b[:, 52:55]  head yaw/pitch/roll (radians)
        b[:, 55:58]  leftEye yaw/pitch/roll (radians)
        b[:, 58:61]  rightEye yaw/pitch/roll (radians)
    Output: (B, 320, 1, 64, 64) — matches PoseGuider on (B, 3, 1, 512, 512) input.
    """

    def __init__(
        self,
        in_dim: int = 61,
        out_channels: int = 320,
        seed_size: int = 8,
        hidden_channels: tuple = (128, 128, 96, 64),
    ):
        super().__init__()
        self.seed_size = seed_size
        self.hidden = hidden_channels
        c0 = hidden_channels[0]
        self.proj = nn.Linear(in_dim, c0 * seed_size * seed_size)
        layers = []
        cin = c0
        for cout in hidden_channels[1:]:
            layers += [
                nn.ConvTranspose2d(cin, cout, kernel_size=4, stride=2, padding=1),
                nn.SiLU(inplace=True),
            ]
            cin = cout
        # final 1x1 to out_channels at the post-upsample resolution
        layers.append(nn.Conv2d(cin, out_channels, kernel_size=3, padding=1))
        self.decoder = nn.Sequential(*layers)
        # PoseGuider's conv_out is zero-init; mimic that so untrained student
        # contributes zero bias (safe drop-in even before training)
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, b: torch.Tensor) -> torch.Tensor:
        # (B, in_dim) -> (B, c0, seed, seed)
        h = self.proj(b).view(-1, self.hidden[0], self.seed_size, self.seed_size)
        # 8 -> 16 -> 32 -> 64
        h = self.decoder(h)
        # PoseGuider returns (B, 320, 1, 64, 64); add the singleton frame dim
        return h.unsqueeze(2)
```

Quick check: 8→16→32→64 with three transposed convs (channels 128→128→96→64), final conv to 320. Param count: `61*8192 + 128*128*16 + 128*96*16 + 96*64*16 + 64*320*9 ≈ 0.5M + 0.26M + 0.20M + 0.10M + 0.18M ≈ 1.24M`. Well under 5M.

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/newub/w/vamp-interface && .venv/bin/pytest tests/arkit_bridge/test_encoder.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/arkit_bridge/encoder.py tests/arkit_bridge/test_encoder.py
git commit -m "feat(arkit_bridge): parametric PoseGuider student encoder"
```

---

## Task 3: Teacher wrapper (frozen PersonaLive PoseGuider)

**Files:**
- Create: `src/arkit_bridge/teacher.py`
- Create: `tests/arkit_bridge/test_teacher.py`

- [ ] **Step 1: Write the failing test**

Write to `tests/arkit_bridge/test_teacher.py`:

```python
import os
import sys

import pytest
import torch

MOORE = os.path.expanduser("~/w/Moore-AnimateAnyone")
PERSONA_PG = os.path.expanduser(
    "~/w/PersonaLive/pretrained_weights/personalive/pose_guider.pth"
)

# Allow Moore's `src.models.*` imports
if MOORE not in sys.path:
    sys.path.insert(0, MOORE)


@pytest.mark.skipif(
    not os.path.exists(PERSONA_PG), reason="PersonaLive weights not present"
)
def test_teacher_loads_and_runs():
    from src.arkit_bridge.teacher import load_frozen_teacher

    teacher = load_frozen_teacher(weights_path=PERSONA_PG, device="cpu")
    assert all(not p.requires_grad for p in teacher.parameters())
    x = torch.randn(1, 3, 1, 512, 512)
    with torch.no_grad():
        y = teacher(x)
    assert y.shape == (1, 320, 1, 64, 64), y.shape
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/newub/w/vamp-interface && .venv/bin/pytest tests/arkit_bridge/test_teacher.py -v
```

Expected: FAIL (import error).

- [ ] **Step 3: Implement teacher loader**

Write to `src/arkit_bridge/teacher.py`:

```python
"""Load and freeze PersonaLive's PoseGuider as the distillation teacher."""

import os
import sys

import torch

_MOORE = os.path.expanduser("~/w/Moore-AnimateAnyone")
if _MOORE not in sys.path:
    sys.path.insert(0, _MOORE)

# Imported lazily inside loader to keep test collection cheap when Moore tree
# isn't available.


def load_frozen_teacher(weights_path: str, device: str = "cuda"):
    """Build a Moore PoseGuider matching PersonaLive's config and load weights."""
    from src.models.pose_guider import PoseGuider  # noqa: E402  Moore vendor

    pg = PoseGuider(
        conditioning_embedding_channels=320,
        block_out_channels=(16, 32, 96, 256),
    )
    raw = torch.load(weights_path, map_location="cpu")
    # PersonaLive renames the final conv layer; rename back if needed.
    sd = {k.replace("conv_out_modify", "conv_out"): v for k, v in raw.items()}
    missing, unexpected = pg.load_state_dict(sd, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys when loading teacher: {unexpected}")
    pg.eval()
    for p in pg.parameters():
        p.requires_grad_(False)
    return pg.to(device)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/newub/w/vamp-interface && .venv/bin/pytest tests/arkit_bridge/test_teacher.py -v
```

Expected: 1 passed (or skipped if weights not present, which is fine on CI).

- [ ] **Step 5: Commit**

```bash
git add src/arkit_bridge/teacher.py tests/arkit_bridge/test_teacher.py
git commit -m "feat(arkit_bridge): frozen PoseGuider teacher loader"
```

---

## Task 4: Extractors (MediaPipe ARKit, DWPose, face crop)

**Files:**
- Create: `src/arkit_bridge/extractors.py`
- Create: `tests/arkit_bridge/test_extractors.py`
- Modify: `pyproject.toml` (add `mediapipe`, `controlnet_aux` deps)

- [ ] **Step 1: Install deps**

```bash
cd /home/newub/w/vamp-interface
.venv/bin/pip install mediapipe==0.10.20 controlnet_aux==0.0.9
```

If `controlnet_aux` is already in vendor or vamp-interface .venv, skip.

- [ ] **Step 2: Download MediaPipe FaceLandmarker model**

```bash
mkdir -p /home/newub/w/vamp-interface/models/mediapipe
curl -L -o /home/newub/w/vamp-interface/models/mediapipe/face_landmarker_v2_with_blendshapes.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

Verify `ls -lh models/mediapipe/face_landmarker_v2_with_blendshapes.task` shows ~3.7 MB.

- [ ] **Step 3: Write the failing test**

Write to `tests/arkit_bridge/test_extractors.py`:

```python
import numpy as np
import pytest
from src.arkit_bridge.extractors import (
    extract_arkit_61,
    ARKIT_BLENDSHAPE_NAMES,
)


def _synth_face_image():
    # MediaPipe will not detect a face on noise; we test only the API contract
    # here. Real-image tests live in the smoke test (Task 6).
    return np.zeros((512, 512, 3), dtype=np.uint8)


def test_arkit_names_length():
    assert len(ARKIT_BLENDSHAPE_NAMES) == 52


def test_extractor_returns_61_or_none():
    img = _synth_face_image()
    result = extract_arkit_61(img)
    # MediaPipe should fail to detect on a black image; expect None.
    assert result is None or (
        isinstance(result, np.ndarray) and result.shape == (61,)
    )
```

- [ ] **Step 4: Run test to verify it fails**

```bash
.venv/bin/pytest tests/arkit_bridge/test_extractors.py -v
```

Expected: FAIL (import error).

- [ ] **Step 5: Implement extractors**

Write to `src/arkit_bridge/extractors.py`:

```python
"""Per-frame extractors: MediaPipe ARKit-52 + head/eye rotation; DWPose image.

MediaPipe FaceLandmarker emits ARKit-style blendshapes + a 4x4 facial
transformation matrix per detected face. We pull the 52 blendshapes plus
decompose the matrix into head Euler angles. Eye rotations are computed
from the iris landmarks relative to the eye-corner landmarks (MediaPipe's
own approximation of gaze).

DWPose render: we use controlnet_aux's DWposeDetector to produce the same
RGB stick image PersonaLive's PoseGuider was trained on.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

ARKIT_BLENDSHAPE_NAMES = [
    "_neutral",  # MediaPipe emits 52 with first being _neutral; we drop it
    "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft",
    "eyeLookUpLeft", "eyeSquintLeft", "eyeWideLeft",
    "eyeBlinkRight", "eyeLookDownRight", "eyeLookInRight", "eyeLookOutRight",
    "eyeLookUpRight", "eyeSquintRight", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel",
    "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker",
    "mouthRight", "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight", "tongueOut",
]
# MediaPipe returns 52 (with _neutral first); strip that to get the canonical 52.
ARKIT_BLENDSHAPE_NAMES = ARKIT_BLENDSHAPE_NAMES[1:] + ["tongueOut"]
ARKIT_BLENDSHAPE_NAMES = ARKIT_BLENDSHAPE_NAMES[:52]
assert len(ARKIT_BLENDSHAPE_NAMES) == 52

_MP_MODEL_PATH = Path(
    "/home/newub/w/vamp-interface/models/mediapipe/"
    "face_landmarker_v2_with_blendshapes.task"
)


_landmarker = None


def _get_landmarker():
    global _landmarker
    if _landmarker is None:
        import mediapipe as mp
        from mediapipe.tasks.python import vision

        opts = vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(_MP_MODEL_PATH)),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        _landmarker = vision.FaceLandmarker.create_from_options(opts)
    return _landmarker


def _euler_from_matrix(m: np.ndarray) -> tuple[float, float, float]:
    """ZYX intrinsic Euler from a 3x3 rotation matrix. Returns (yaw, pitch, roll).

    Right-handed; yaw=Y, pitch=X, roll=Z (matches Apple's conventions).
    """
    # Pitch from -m[1,2] in some conventions; we use the common ZYX inverse.
    sy = math.sqrt(m[0, 0] ** 2 + m[1, 0] ** 2)
    if sy > 1e-6:
        roll = math.atan2(m[2, 1], m[2, 2])
        pitch = math.atan2(-m[2, 0], sy)
        yaw = math.atan2(m[1, 0], m[0, 0])
    else:
        roll = math.atan2(-m[1, 2], m[1, 1])
        pitch = math.atan2(-m[2, 0], sy)
        yaw = 0.0
    return yaw, pitch, roll


def _eye_yaw_pitch(blendshapes: dict, side: str) -> tuple[float, float, float]:
    """Approximate eye angles from the eyeLook* blendshapes.

    True Live Link angles aren't available from MediaPipe, but the eyeLook*
    channels carry the same information at lower fidelity. We emit angles in
    radians scaled so blendshape=1.0 maps to ~30deg. Roll is set to 0 here
    (MediaPipe has no eye-roll signal); it will live in the b[:,52+] head
    rotation slots if needed.
    """
    s = side.capitalize()
    up = blendshapes.get(f"eyeLookUp{s}", 0.0)
    down = blendshapes.get(f"eyeLookDown{s}", 0.0)
    in_ = blendshapes.get(f"eyeLookIn{s}", 0.0)
    out = blendshapes.get(f"eyeLookOut{s}", 0.0)
    pitch = math.radians(30.0) * (up - down)
    yaw_sign = +1.0 if side == "right" else -1.0  # in/out are mirrored
    yaw = math.radians(30.0) * yaw_sign * (in_ - out)
    return yaw, pitch, 0.0


def extract_arkit_61(image_rgb: np.ndarray) -> np.ndarray | None:
    """Run MediaPipe FaceLandmarker and produce the 61-float ARKit vector.

    Returns None if no face is detected.
    """
    import mediapipe as mp

    landmarker = _get_landmarker()
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB, data=image_rgb.astype(np.uint8))
    res = landmarker.detect(mp_image)
    if not res.face_blendshapes or not res.facial_transformation_matrixes:
        return None
    bs_list = res.face_blendshapes[0]
    bs = {b.category_name: b.score for b in bs_list}
    out = np.zeros(61, dtype=np.float32)
    for i, name in enumerate(ARKIT_BLENDSHAPE_NAMES):
        out[i] = bs.get(name, 0.0)
    M = np.array(res.facial_transformation_matrixes[0])  # 4x4
    yaw, pitch, roll = _euler_from_matrix(M[:3, :3])
    out[52:55] = (yaw, pitch, roll)
    out[55:58] = _eye_yaw_pitch(bs, "left")
    out[58:61] = _eye_yaw_pitch(bs, "right")
    return out


_dwpose = None


def get_dwpose_image(image_rgb: np.ndarray) -> np.ndarray:
    """Run DWPose and return its rendered RGB stick image at 512x512."""
    global _dwpose
    if _dwpose is None:
        from controlnet_aux import DWposeDetector
        _dwpose = DWposeDetector()
    from PIL import Image
    pil = Image.fromarray(image_rgb)
    out = _dwpose(pil, output_type="np", detect_resolution=512,
                  image_resolution=512)
    return out  # H,W,3 uint8


def loose_face_crop(image_rgb: np.ndarray, target: int = 512) -> np.ndarray | None:
    """Pad/crop image to target² with the detected face roughly centered.

    "Loose" means we keep head pose intact: only translate+scale to put the
    face in frame at consistent scale. No FFHQ-style template warp.
    """
    landmarker = _get_landmarker()
    import mediapipe as mp

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB, data=image_rgb.astype(np.uint8))
    res = landmarker.detect(mp_image)
    if not res.face_landmarks:
        return None
    h, w = image_rgb.shape[:2]
    pts = np.array([(p.x * w, p.y * h) for p in res.face_landmarks[0]])
    cx, cy = pts.mean(axis=0)
    bbox = pts.max(axis=0) - pts.min(axis=0)
    side = float(max(bbox)) * 1.6  # ~1.6x face span
    side = max(side, 64.0)
    half = side / 2.0
    x0 = int(round(cx - half)); y0 = int(round(cy - half))
    x1 = int(round(cx + half)); y1 = int(round(cy + half))
    # Pad if out of frame.
    pad_l = max(0, -x0); pad_t = max(0, -y0)
    pad_r = max(0, x1 - w); pad_b = max(0, y1 - h)
    img2 = np.pad(
        image_rgb,
        ((pad_t, pad_b), (pad_l, pad_r), (0, 0)),
        mode="edge",
    )
    x0 += pad_l; x1 += pad_l; y0 += pad_t; y1 += pad_t
    crop = img2[y0:y1, x0:x1]
    import cv2
    return cv2.resize(crop, (target, target), interpolation=cv2.INTER_AREA)
```

- [ ] **Step 6: Run test to verify it passes**

```bash
.venv/bin/pytest tests/arkit_bridge/test_extractors.py -v
```

Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
git add src/arkit_bridge/extractors.py tests/arkit_bridge/test_extractors.py \
        models/mediapipe/face_landmarker_v2_with_blendshapes.task
# Note: model file may be too large for git; if so, add to .gitignore and
# document the curl URL in the README.
git commit -m "feat(arkit_bridge): MediaPipe ARKit + DWPose extractors"
```

If model file is too large for git, instead:

```bash
echo "models/mediapipe/" >> .gitignore
git add .gitignore src/arkit_bridge/extractors.py tests/arkit_bridge/test_extractors.py
git commit -m "feat(arkit_bridge): MediaPipe ARKit + DWPose extractors"
```

---

## Task 5: Pair extraction script

**Files:**
- Create: `scripts/extract_distill_pairs.py`

- [ ] **Step 1: Pick a source video corpus**

Use any RGB face video. For initial smoke testing a single 1-2 minute clip is enough. Suggested sources:

- A short YouTube face-talking clip downloaded with `yt-dlp`, e.g., a TED talk close-up.
- VFHQ test split (license-permitting) — see `2026-05-03-iphone-pipeline-unified-plan.md`.
- A self-recorded 2-minute webcam clip cycling through expressions and head turns.

Place under `/home/newub/w/vamp-interface/data/arkit_distill_videos/` (gitignored).

- [ ] **Step 2: Implement the extractor script**

Write to `scripts/extract_distill_pairs.py`:

```python
"""Extract (b₆₁, T) distillation pairs from a video.

For each sampled frame:
  1. Loose face crop -> 512x512 RGB
  2. MediaPipe -> b₆₁
  3. DWPose -> stick image (512,512,3)
  4. Frozen PoseGuider(stick image) -> T (320,1,64,64)
  5. Save (b, T) to a single .pkl per frame.

Resumable: skips frames whose output .pkl already exists.
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.arkit_bridge.extractors import (  # noqa: E402
    extract_arkit_61, get_dwpose_image, loose_face_crop,
)
from src.arkit_bridge.teacher import load_frozen_teacher  # noqa: E402

PERSONA_PG = os.path.expanduser(
    "~/w/PersonaLive/pretrained_weights/personalive/pose_guider.pth"
)


def iter_frames(video_path: str, stride: int):
    import cv2
    cap = cv2.VideoCapture(video_path)
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
    ap.add_argument("--video", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--stride", type=int, default=2,
                    help="sample every Nth frame")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_frames", type=int, default=0,
                    help="0 = all")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    teacher = load_frozen_teacher(PERSONA_PG, device=args.device)

    n_done = 0
    n_skipped = 0
    n_kept = 0
    for fi, rgb in iter_frames(args.video, args.stride):
        out_path = os.path.join(args.out_dir, f"frame_{fi:06d}.pkl")
        if os.path.exists(out_path):
            n_skipped += 1
            continue
        crop = loose_face_crop(rgb, target=512)
        if crop is None:
            continue
        b = extract_arkit_61(crop)
        if b is None:
            continue
        try:
            stick = get_dwpose_image(crop)
        except Exception as e:
            print(f"frame {fi}: DWPose failed: {e}")
            continue
        # PoseGuider expects (B, 3, 1, H, W), normalized to [0,1].
        stick_t = torch.from_numpy(stick).float().permute(2, 0, 1) / 255.0
        stick_t = stick_t.unsqueeze(0).unsqueeze(2).to(args.device)
        with torch.no_grad():
            T = teacher(stick_t).squeeze(0).cpu().numpy()  # (320, 1, 64, 64)
        with open(out_path, "wb") as f:
            pickle.dump({"b": b, "T": T.astype(np.float16)}, f)
        n_done += 1; n_kept += 1
        if n_done % 50 == 0:
            print(f"  {n_done} pairs written (skipped existing: {n_skipped})")
        if args.max_frames and n_kept >= args.max_frames:
            break
    print(f"done: kept={n_kept} skipped={n_skipped}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke run on a tiny clip**

```bash
cd /home/newub/w/vamp-interface
mkdir -p data/arkit_distill_pairs/smoke
.venv/bin/python scripts/extract_distill_pairs.py \
  --video data/arkit_distill_videos/sample.mp4 \
  --out_dir data/arkit_distill_pairs/smoke \
  --stride 4 --max_frames 50
```

Expected: console reports `done: kept=50 skipped=0`. `ls data/arkit_distill_pairs/smoke/ | wc -l` shows 50 pkls.

- [ ] **Step 4: Sanity-check one pkl**

```bash
.venv/bin/python -c "
import pickle, numpy as np
with open('data/arkit_distill_pairs/smoke/frame_000000.pkl','rb') as f:
    d = pickle.load(f)
print('b:', d['b'].shape, d['b'].dtype, 'min', d['b'].min(), 'max', d['b'].max())
print('T:', d['T'].shape, d['T'].dtype, 'std', float(np.array(d['T']).std()))
"
```

Expected: `b: (61,) float32`, `T: (320, 1, 64, 64) float16`, T.std > 0.

- [ ] **Step 5: Commit script + .gitignore**

```bash
echo "data/arkit_distill_videos/" >> .gitignore
echo "data/arkit_distill_pairs/" >> .gitignore
git add .gitignore scripts/extract_distill_pairs.py
git commit -m "feat(arkit_bridge): pair extraction script"
```

---

## Task 6: Dataset + smoke training (overfit on 50 frames)

**Files:**
- Create: `src/arkit_bridge/dataset.py`
- Create: `src/arkit_bridge/distill.py`
- Create: `scripts/train_arkit_distill.py`

- [ ] **Step 1: Implement dataset**

Write to `src/arkit_bridge/dataset.py`:

```python
"""Iterate cached (b₆₁, T) pkls as a torch Dataset."""

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class DistillPairDataset(Dataset):
    def __init__(self, root: str | Path):
        self.paths = sorted(Path(root).glob("frame_*.pkl"))
        if not self.paths:
            raise FileNotFoundError(f"no frame pkls under {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with open(self.paths[i], "rb") as f:
            d = pickle.load(f)
        b = torch.from_numpy(np.asarray(d["b"], dtype=np.float32))
        T = torch.from_numpy(np.asarray(d["T"], dtype=np.float32))  # (320,1,64,64)
        return b, T
```

- [ ] **Step 2: Implement distill loop**

Write to `src/arkit_bridge/distill.py`:

```python
"""Distillation training loop. MSE between student(b₆₁) and cached teacher T."""

import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.arkit_bridge.dataset import DistillPairDataset
from src.arkit_bridge.encoder import ARKitParametricPoseGuider


def train(
    pairs_dir: str,
    out_dir: str,
    *,
    batch_size: int = 32,
    lr: float = 1e-3,
    steps: int = 5000,
    log_every: int = 50,
    ckpt_every: int = 1000,
    device: str = "cuda",
):
    os.makedirs(out_dir, exist_ok=True)
    ds = DistillPairDataset(pairs_dir)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                    num_workers=2, drop_last=True)
    student = ARKitParametricPoseGuider().to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=lr)
    log = []
    step = 0
    t0 = time.time()
    while step < steps:
        for b, T in dl:
            b = b.to(device); T = T.to(device)
            pred = student(b)
            loss = F.mse_loss(pred, T)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step % log_every == 0:
                rate = step / max(1e-6, time.time() - t0)
                print(f"step {step:5d}  loss {loss.item():.5f}  "
                      f"({rate:.1f} step/s)")
                log.append({"step": step, "loss": float(loss.item())})
            if step % ckpt_every == 0 or step >= steps:
                ckpt = Path(out_dir) / f"student_step{step:06d}.pt"
                torch.save(student.state_dict(), ckpt)
                with open(Path(out_dir) / "log.json", "w") as f:
                    json.dump(log, f)
            if step >= steps:
                break
    return student
```

- [ ] **Step 3: Implement CLI**

Write to `scripts/train_arkit_distill.py`:

```python
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.arkit_bridge.distill import train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    train(
        pairs_dir=args.pairs_dir,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        steps=args.steps,
        device=args.device,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Smoke train (overfit on 50 frames)**

```bash
cd /home/newub/w/vamp-interface
.venv/bin/python scripts/train_arkit_distill.py \
  --pairs_dir data/arkit_distill_pairs/smoke \
  --out_dir exp_output/arkit_distill/smoke \
  --batch_size 16 --steps 1000 --lr 1e-3
```

Expected: loss starts at ~teacher.std² and drops monotonically. After 1000 steps over 50 samples (>= 300 epochs) the loss should be << initial. If loss stagnates, the architecture or normalization is wrong; debug before scaling.

- [ ] **Step 5: Sanity-check overfit quality**

```bash
.venv/bin/python -c "
import torch, pickle
from src.arkit_bridge.encoder import ARKitParametricPoseGuider
from src.arkit_bridge.dataset import DistillPairDataset

m = ARKitParametricPoseGuider().cuda()
m.load_state_dict(torch.load('exp_output/arkit_distill/smoke/student_step001000.pt'))
ds = DistillPairDataset('data/arkit_distill_pairs/smoke')
b, T = ds[0]
with torch.no_grad():
    pred = m(b.unsqueeze(0).cuda()).cpu()
import torch.nn.functional as F
print('overfit MSE:', F.mse_loss(pred, T.unsqueeze(0)).item())
print('teacher var:', T.var().item())
print('ratio:', F.mse_loss(pred, T.unsqueeze(0)).item() / T.var().item())
"
```

Expected: overfit ratio < 0.1 (student captures > 90% of teacher variance on training set). If ratio is near 1.0, something is structurally broken.

- [ ] **Step 6: Commit**

```bash
git add src/arkit_bridge/dataset.py src/arkit_bridge/distill.py \
        scripts/train_arkit_distill.py
git commit -m "feat(arkit_bridge): distill loop + smoke overfit"
```

---

## Task 7: Per-channel sensitivity sweep (eval harness)

**Files:**
- Create: `src/arkit_bridge/eval.py`
- Create: `scripts/eval_arkit_distill.py`

- [ ] **Step 1: Implement eval**

Write to `src/arkit_bridge/eval.py`:

```python
"""Per-channel sensitivity match between student and teacher.

For each ARKit channel i in [0..51]:
  1. Take a held-out neutral b vector (median over corpus).
  2. Sweep b[i] over [0, 0.5, 1.0]; keep all other channels fixed.
  3. Decode each via student → S_i(s); the cached teacher T_i(s) for the same
     b vector run through DWPose isn't available, so we compare the *direction
     of change* against the dataset's mean: how much does the student's output
     respond to channel i moving 0→1, vs how it responds to other channels?
  4. Report per-channel L2 norm of (student(b_i=1) - student(b_i=0)).

This isn't a teacher-match metric; it's a "is the student responsive to this
channel at all" diagnostic. Combined with held-out MSE it tells us which
channels distill cleanly vs which collapse.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.arkit_bridge.dataset import DistillPairDataset
from src.arkit_bridge.encoder import ARKitParametricPoseGuider
from src.arkit_bridge.extractors import ARKIT_BLENDSHAPE_NAMES


def held_out_mse(student, dataset, device="cuda", batch_size=32):
    student.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = [dataset[j] for j in range(i, min(i+batch_size, len(dataset)))]
            b = torch.stack([x[0] for x in batch]).to(device)
            T = torch.stack([x[1] for x in batch]).to(device)
            pred = student(b)
            total += F.mse_loss(pred, T, reduction="sum").item()
            n += T.numel()
    return total / max(n, 1)


def channel_sensitivity(student, dataset, device="cuda"):
    """Per-channel L2 of d output / d channel value."""
    student.eval()
    # Median b across the dataset as a neutral baseline.
    bs = torch.stack([dataset[i][0] for i in range(len(dataset))])
    b_neutral = bs.median(dim=0).values.to(device)
    out = {}
    with torch.no_grad():
        ref = student(b_neutral.unsqueeze(0))
        for i in range(52):
            b_hi = b_neutral.clone()
            b_hi[i] = 1.0
            r_hi = student(b_hi.unsqueeze(0))
            delta = (r_hi - ref).pow(2).mean().sqrt().item()
            out[ARKIT_BLENDSHAPE_NAMES[i]] = delta
    return out


def main(ckpt: str, pairs_dir: str, out_path: str, device: str = "cuda"):
    student = ARKitParametricPoseGuider().to(device)
    student.load_state_dict(torch.load(ckpt, map_location=device))
    ds = DistillPairDataset(pairs_dir)
    mse = held_out_mse(student, ds, device=device)
    sens = channel_sensitivity(student, ds, device=device)
    report = {
        "ckpt": ckpt,
        "pairs_dir": pairs_dir,
        "n_samples": len(ds),
        "held_out_mse": mse,
        "channel_sensitivity": sens,
        "channel_sensitivity_ranked": sorted(
            sens.items(), key=lambda kv: -kv[1]),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"held-out MSE: {mse:.5f}")
    print("top 10 most-responsive channels:")
    for name, val in report["channel_sensitivity_ranked"][:10]:
        print(f"  {name:24s} {val:.5f}")
    print("bottom 5 least-responsive channels:")
    for name, val in report["channel_sensitivity_ranked"][-5:]:
        print(f"  {name:24s} {val:.5f}")
```

- [ ] **Step 2: Implement CLI**

Write to `scripts/eval_arkit_distill.py`:

```python
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.arkit_bridge.eval import main  # noqa: E402


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pairs_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    main(args.ckpt, args.pairs_dir, args.out, device=args.device)
```

- [ ] **Step 3: Run on smoke checkpoint**

```bash
.venv/bin/python scripts/eval_arkit_distill.py \
  --ckpt exp_output/arkit_distill/smoke/student_step001000.pt \
  --pairs_dir data/arkit_distill_pairs/smoke \
  --out exp_output/arkit_distill/smoke/eval.json
```

Expected: a printed top-10 channels with non-zero sensitivity. If all sensitivities are ~0 the model collapsed.

- [ ] **Step 4: Commit**

```bash
git add src/arkit_bridge/eval.py scripts/eval_arkit_distill.py
git commit -m "feat(arkit_bridge): per-channel sensitivity eval"
```

---

## Task 8: Real distill run (~10K-frame corpus)

**Files:**
- Reuse: `scripts/extract_distill_pairs.py`, `scripts/train_arkit_distill.py`

- [ ] **Step 1: Build the real corpus**

Pull/record ≥ 10 minutes of face video with diverse expressions and head motion. Goal: 10K extracted pairs. Place at `data/arkit_distill_videos/real/*.mp4`.

```bash
cd /home/newub/w/vamp-interface
mkdir -p data/arkit_distill_pairs/real
for v in data/arkit_distill_videos/real/*.mp4; do
  .venv/bin/python scripts/extract_distill_pairs.py \
    --video "$v" \
    --out_dir data/arkit_distill_pairs/real \
    --stride 2
done
ls data/arkit_distill_pairs/real | wc -l
```

Expected: ≥ 10000 pkls. Each pkl is ~2.5 MB (320×1×64×64 fp16 = 2.5 MB), total ~25 GB. Verify free disk via `df -h .` before launching.

- [ ] **Step 2: Real distill training run**

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
systemd-run --user --scope -p MemoryMax=45G \
  .venv/bin/python scripts/train_arkit_distill.py \
    --pairs_dir data/arkit_distill_pairs/real \
    --out_dir exp_output/arkit_distill/real \
    --batch_size 64 --steps 50000 --lr 5e-4
```

Expected runtime: ~1-3 hours on RTX 5090. Loss should decrease monotonically; final value depends on teacher entropy.

- [ ] **Step 3: Run eval on the real ckpt**

```bash
.venv/bin/python scripts/eval_arkit_distill.py \
  --ckpt exp_output/arkit_distill/real/student_step050000.pt \
  --pairs_dir data/arkit_distill_pairs/real \
  --out exp_output/arkit_distill/real/eval.json
```

Capture the held-out MSE and per-channel ranking — these feed the readout doc.

- [ ] **Step 4: Commit logs (not data)**

```bash
git add exp_output/arkit_distill/real/log.json \
        exp_output/arkit_distill/real/eval.json
echo "exp_output/arkit_distill/*/student_*.pt" >> .gitignore
git add .gitignore
git commit -m "chore(arkit_bridge): real distill run logs + eval"
```

---

## Task 9: Drop-in PersonaLive inference test

**Files:**
- Create: `scripts/test_student_in_personalive.py`

- [ ] **Step 1: Implement integration test**

Write to `scripts/test_student_in_personalive.py`:

```python
"""Render a frame through PersonaLive's denoising UNet using the student's
output as PoseGuider features. Visual sanity check only.

Approach: re-use PersonaLive's inference entry point but monkey-patch the
PoseGuider call to substitute student-produced features. The denoising UNet
stays unchanged.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PERSONA = os.path.expanduser("~/w/PersonaLive")
if PERSONA not in sys.path:
    sys.path.insert(0, PERSONA)

from src.arkit_bridge.encoder import ARKitParametricPoseGuider  # noqa: E402


def render_once(reference_image: str, b61: np.ndarray, ckpt: str,
                out_path: str, device="cuda"):
    """Skeleton: load the PersonaLive inference pipeline, replace pose_guider
    with a wrapper that ignores its image input and returns student(b61).
    Persona Live's exact API may differ — adjust the import + call site by
    inspecting `~/w/PersonaLive/inference.py` (or whichever entry point ships
    with the install).
    """
    student = ARKitParametricPoseGuider().to(device)
    student.load_state_dict(torch.load(ckpt, map_location=device))
    student.eval()
    b = torch.from_numpy(b61.astype(np.float32)).unsqueeze(0).to(device)

    # NOTE: integration shim — exact entry point depends on PersonaLive's
    # inference module. The general pattern:
    #
    #   from PersonaLive.something import build_pipeline
    #   pipe = build_pipeline(...)
    #   pipe.pose_guider = _StudentAdapter(student, b)  # see below
    #   img = pipe(reference_image=reference_image, ...)
    #   img.save(out_path)
    raise NotImplementedError(
        "Wire to PersonaLive's actual inference module — "
        "inspect ~/w/PersonaLive/ for the entry point and update this stub."
    )


class _StudentAdapter(torch.nn.Module):
    """Wraps the student so it can replace PoseGuider in PersonaLive's pipeline.

    PoseGuider is called as `pose_guider(image_tensor)`. We ignore the image
    tensor and return the cached student(b) precomputed in __init__.
    """

    def __init__(self, student: ARKitParametricPoseGuider, b: torch.Tensor):
        super().__init__()
        with torch.no_grad():
            self._cached = student(b)  # (1, 320, 1, 64, 64)

    def forward(self, _ignored_image):
        return self._cached


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--b61", required=True, help="path to .npy with shape (61,)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    b = np.load(args.b61)
    render_once(args.reference, b, args.ckpt, args.out)
```

- [ ] **Step 2: Inspect PersonaLive inference entry point**

```bash
ls /home/newub/w/PersonaLive/*.py
grep -rn "pose_guider\|PoseGuider" /home/newub/w/PersonaLive/*.py | head -20
```

Identify the inference entry point and the exact call site of `pose_guider(...)`. Replace the `raise NotImplementedError` block in `render_once` with a concrete monkey-patch matching that API. Commit only after this is wired and a test render succeeds.

- [ ] **Step 3: Generate a test b61 vector**

```bash
.venv/bin/python -c "
import numpy as np
b = np.zeros(61, dtype=np.float32)
# 'mouthSmileLeft' index in our names list
from src.arkit_bridge.extractors import ARKIT_BLENDSHAPE_NAMES
i = ARKIT_BLENDSHAPE_NAMES.index('mouthSmileLeft')
j = ARKIT_BLENDSHAPE_NAMES.index('mouthSmileRight')
b[i] = b[j] = 0.7
np.save('exp_output/arkit_distill/real/test_smile.npy', b)
"
```

- [ ] **Step 4: Render**

```bash
.venv/bin/python scripts/test_student_in_personalive.py \
  --reference data/test_reference_face.png \
  --ckpt exp_output/arkit_distill/real/student_step050000.pt \
  --b61 exp_output/arkit_distill/real/test_smile.npy \
  --out exp_output/arkit_distill/real/test_smile.png
```

Expected: a face that is recognizably the reference, smiling, head roughly centered. If the result is gibberish, the cached PoseGuider feature is at the wrong scale/sign — debug by also rendering with a sweep over a random b vector and the student's output magnitude.

- [ ] **Step 5: Commit**

```bash
git add scripts/test_student_in_personalive.py \
        exp_output/arkit_distill/real/test_smile.png
git commit -m "feat(arkit_bridge): drop-in PersonaLive inference smoke test"
```

---

## Task 10: Readout document + decision

**Files:**
- Create: `docs/research/2026-05-05-arkit-poseguider-distill-readout.md`
- Modify: `docs/research/_topics/neural-deformation-control.md`

- [ ] **Step 1: Write the readout**

Capture: held-out MSE, per-channel sensitivity ranking, qualitative notes from the inference test, list of channels that look broken (low sensitivity OR visible failure modes in renders), and the recommendation for next step (proceed to "replace" plan, or stop here, or pivot).

Template structure:

```markdown
---
status: live
topic: neural-deformation-control
---

# ARKit→PoseGuider distill — readout (2026-05-05)

Companion to [`2026-05-05-arkit-poseguider-distill-plan.md`](2026-05-05-arkit-poseguider-distill-plan.md).

## Numbers
- corpus size: N pairs
- training steps: K
- final held-out MSE: X
- teacher feature variance: Y
- variance fraction explained: 1 - X/Y = Z

## Per-channel sensitivity (top / bottom)
[paste from eval.json]

## Qualitative inference render
- reference: <path>
- b vectors tested: ...
- failure modes seen: ...

## Decision
- proceed to replace? yes/no, why
- channels to flag for LoRA-only path: ...
```

- [ ] **Step 2: Update topic index**

Append to `docs/research/_topics/neural-deformation-control.md`:

```markdown
- 2026-05-05 distill readout: [`2026-05-05-arkit-poseguider-distill-readout.md`](../2026-05-05-arkit-poseguider-distill-readout.md) — N pairs, MSE X, top channels Y, decision Z.
```

- [ ] **Step 3: Commit**

```bash
git add docs/research/2026-05-05-arkit-poseguider-distill-readout.md \
        docs/research/_topics/neural-deformation-control.md
git commit -m "docs(neural-deformation): ARKit distill readout"
```

---

## Self-review checklist

Run before handing this off to a subagent executor:

**Spec coverage:**
- [x] Encoder: Task 2.
- [x] Teacher: Task 3.
- [x] Extractors (MediaPipe + DWPose + face crop): Task 4.
- [x] Pair extraction: Task 5.
- [x] Smoke train: Task 6.
- [x] Per-channel eval: Task 7.
- [x] Real run: Task 8.
- [x] PersonaLive integration test: Task 9.
- [x] Readout + decision: Task 10.

**Placeholder scan:** Task 9 has one acknowledged TBD — the exact PersonaLive inference entry point. The plan instructs the executor to inspect `~/w/PersonaLive/` and complete the integration shim before proceeding. This is a research-process step that depends on a not-yet-known external API surface; flagged explicitly in the task rather than guessed.

**Type consistency:** Encoder forward `(B, 61) -> (B, 320, 1, 64, 64)` matches teacher output shape and PersonaLive's call site (`tgt_pose_img.unsqueeze(2) -> (bs, 3, 1, 512, 512)` becomes our 4D output via `.unsqueeze(2)`). `ARKitParametricPoseGuider` referenced consistently in encoder.py, distill.py, eval.py, and test scripts.

**Out-of-scope reminders:** "Replace" is a separate plan, gated on Task 10's decision. Stage-2 / temporal modules untouched. iOS app + transport untouched.

## Execution handoff

After Task 10's commit lands and the readout doc is reviewed, the next decision is: replace plan vs ship offline pipeline as-is vs pivot. That decision is encoded in the readout doc, not this plan.
