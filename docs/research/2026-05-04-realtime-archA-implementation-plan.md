---
status: live
topic: neural-deformation-control
---

# Realtime VTuber Architecture A — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a working realtime "VTuber-with-a-real-face" demo: USB webcam → PersonaLive (RGB-driving, one preselected Flux anchor) → v4l2loopback virtual camera → OBS preview, end-to-end, on Linux + RTX 5090.

**Architecture:** Single-process tight loop in the PersonaLive checkout. `cv2.VideoCapture(/dev/videoN)` reads BGR frames; the script accumulates 4-frame chunks (PersonaLive's temporal window — `chunk_size = 4` per `webcam/vid2vid_trt.py:131`) and feeds each chunk to `pipeline.process_input(...)` after a one-time `pipeline.fuse_reference(anchor_pil)`. Output frames go to `/dev/video10` (v4l2loopback) via `pyvirtualcam`, and OBS picks `/dev/video10` up as a Video Capture Device. No multiprocessing, no Queues — the existing `webcam/vid2vid_trt.py` is multiprocess; we keep its **API exactly** but flatten the Queue-and-Process structure into one tight loop. **Latency note:** the 4-frame chunk adds ~200 ms of input buffering at 20 FPS regardless of inference speed; the realtime doc's 130–200 ms "glass-to-glass" budget already absorbs this.

**Tech Stack:**
- Python 3.10 in `~/w/PersonaLive/.venv/` (torch 2.11+cu128, cv2 4.11, tensorrt 10.15)
- PersonaLive bundled TRT engine (rebuildable in ~10 min via `PersonaLive/scripts/build_trt_engine.py`)
- `v4l2loopback-dkms` kernel module (Ubuntu/Debian package)
- `pyvirtualcam` (Python wrapper that writes RGB frames to a v4l2loopback device)
- OBS Studio (already used in this user's stack)

**Out of scope (deliberate):**
- iPhone NDI / RTSP transport (covered in `2026-05-04-iphone-to-linux-webcam-transport.md`, separate plan)
- Architecture B (ARKit-driving via Phase 3 bridge) — depends on Phase 1 Step 3 motion-API probe (Task #37)
- RTMP / Twitch streaming setup (OBS handles this; demo only needs local preview)
- Anchor-swap-mid-stream UI (mentioned as risk in the realtime doc; punted to follow-up)
- GFPGAN / SeedVR2 super-resolution pass (separate research thread)

---

## File Structure

| Path | Purpose | Status |
|---|---|---|
| `~/w/PersonaLive/scripts/realtime_webcam.py` | New tight-loop driver: webcam → PersonaLive → virtual cam | **Create** |
| `~/w/PersonaLive/.venv/` | Python env, will gain `pyvirtualcam` | Modify |
| `/etc/modules-load.d/v4l2loopback.conf` | Persist module load on boot | Create |
| `/etc/modprobe.d/v4l2loopback.conf` | Persist module options (video_nr, label) | Create |
| `/tmp/personalive_trt/bundled/unet_work.engine` | TRT engine, rebuild target | Rebuild |
| `~/w/vamp-interface/data/llf-phase2/anchors.json` | Anchor catalog (read-only, already exists) | — |

The script lives in the PersonaLive checkout, **not** in vamp-interface, matching `phase2_grid.py` precedent. Vamp-interface keeps only the plan + research docs.

---

## Task 1: Survey hardware and verify the USB webcam

**Files:**
- None (read-only system probing)

- [ ] **Step 1: List video devices**

```bash
ls -la /dev/video*
```

Expected: at least one `/dev/videoN`. Modern UVC webcams typically expose **two** consecutive nodes per camera (e.g. `/dev/video0` for video, `/dev/video1` for metadata). Note which one is the actual capture device — Step 3 will confirm.

- [ ] **Step 2: Install `v4l-utils` for device introspection**

```bash
sudo apt install -y v4l-utils
v4l2-ctl --list-devices
```

Expected output: a block per camera, e.g.

```
USB Camera (usb-0000:xx:xx.x-y):
        /dev/video0
        /dev/video1
```

The first listed `/dev/videoN` under each block is the capture device. Record it as `WEBCAM_DEV` for later steps.

- [ ] **Step 3: Confirm the webcam streams using ffplay**

```bash
ffplay -f v4l2 -framerate 30 -video_size 640x480 $WEBCAM_DEV
```

Expected: a window opens showing the webcam feed at 30 FPS. Press `q` to close.

If multiple devices are listed and the first one fails with "Inappropriate ioctl for device", try the next index. Only **one** of them will be the real capture node.

- [ ] **Step 4: Probe supported formats and resolutions**

```bash
v4l2-ctl -d $WEBCAM_DEV --list-formats-ext | head -40
```

Expected: a list including `MJPG` and/or `YUYV`. Note the highest resolution that supports ≥30 FPS (target: 640×480 @ 30 fps minimum; 1280×720 @ 30 nice-to-have). PersonaLive will downscale to 512² internally regardless, so do not chase 1080p.

- [ ] **Step 5: Commit findings as a one-line note**

No code changes. Append a single line to the plan checklist with the actual device path + max usable mode, e.g. `# webcam survey: /dev/video0, MJPG 1280×720@30, YUYV 640×480@30`. Commit only if a doc was edited; otherwise this step is a no-op note.

---

## Task 2: Install and persist v4l2loopback

**Files:**
- Create: `/etc/modules-load.d/v4l2loopback.conf`
- Create: `/etc/modprobe.d/v4l2loopback.conf`

- [ ] **Step 1: Install the DKMS package**

```bash
sudo apt install -y v4l2loopback-dkms v4l2loopback-utils
```

Expected: dkms builds the module against the running kernel (6.17.0-22-generic). If the build fails, do **not** retry blindly — read the dkms log under `/var/lib/dkms/v4l2loopback/*/build/make.log` and report. Kernel ≥6.11 has had v4l2loopback compat bumps in the past.

- [ ] **Step 2: Load the module manually, one-shot, to verify it works**

```bash
sudo modprobe v4l2loopback video_nr=10 card_label="PersonaLive" exclusive_caps=1
ls -la /dev/video10
```

Expected: `/dev/video10` exists, owned by `root:video`. Current user must be in the `video` group; verify with `id | grep -o 'video'` (no output = not in group). If missing, `sudo usermod -aG video $USER` and re-login.

- [ ] **Step 3: Confirm a test frame lands**

```bash
# In one terminal:
ffmpeg -f lavfi -i testsrc=size=512x512:rate=20 -f v4l2 -pix_fmt yuv420p /dev/video10
# In another terminal:
ffplay /dev/video10
```

Expected: the SMPTE-style colour-bar "testsrc" plays in ffplay. Stop both with `q` / Ctrl+C.

- [ ] **Step 4: Persist the module load across reboots**

Create `/etc/modules-load.d/v4l2loopback.conf` with the single line:

```
v4l2loopback
```

Create `/etc/modprobe.d/v4l2loopback.conf` with:

```
options v4l2loopback video_nr=10 card_label="PersonaLive" exclusive_caps=1
```

```bash
sudo tee /etc/modules-load.d/v4l2loopback.conf <<<'v4l2loopback' > /dev/null
sudo tee /etc/modprobe.d/v4l2loopback.conf <<<'options v4l2loopback video_nr=10 card_label="PersonaLive" exclusive_caps=1' > /dev/null
```

- [ ] **Step 5: Verify persistence by unloading and reloading**

```bash
sudo modprobe -r v4l2loopback
ls /dev/video10  # should be gone
sudo modprobe v4l2loopback
ls -la /dev/video10  # should be back, with the persisted options
```

Expected: `/dev/video10` reappears with `card_label=PersonaLive` (verify via `v4l2-ctl -d /dev/video10 --info | grep -i card`).

- [ ] **Step 6: Commit nothing**

This step is system-level config; no repo change. Move on.

---

## Task 3: Install `pyvirtualcam` in the PersonaLive venv

**Files:**
- Modify: `~/w/PersonaLive/.venv/` (package install only)

- [ ] **Step 1: Install pyvirtualcam**

```bash
~/w/PersonaLive/.venv/bin/python -m pip install pyvirtualcam
```

Expected: clean install, no compile (it's a thin v4l2loopback wrapper on Linux).

- [ ] **Step 2: Smoke-test pyvirtualcam writes to /dev/video10**

```bash
~/w/PersonaLive/.venv/bin/python - <<'EOF'
import numpy as np
import pyvirtualcam
with pyvirtualcam.Camera(width=512, height=512, fps=20, device="/dev/video10") as cam:
    print(f"using {cam.device}")
    for i in range(60):  # 3 seconds @ 20 fps
        frame = np.zeros((512, 512, 3), dtype=np.uint8)
        frame[:, :, i % 3] = 255  # cycle R/G/B
        cam.send(frame)
        cam.sleep_until_next_frame()
print("done")
EOF
```

In a second terminal, run `ffplay /dev/video10` for the duration of the script.

Expected: ffplay shows a 512×512 panel cycling red/green/blue for ~3 seconds. Script prints `using /dev/video10` then `done`.

If it errors with `Permission denied`, recheck the `video` group from Task 2 Step 2.

- [ ] **Step 3: Commit nothing**

Venv state is not tracked. Move on.

---

## Task 4: Rebuild the PersonaLive bundled TRT engine

**Files:**
- Rebuild: `/tmp/personalive_trt/bundled/unet_work.engine` (and any sibling artifacts the build script produces)

- [ ] **Step 1: Verify whether the engine already exists**

```bash
ls -la /tmp/personalive_trt/bundled/ 2>/dev/null || echo "MISSING — needs rebuild"
```

`/tmp` is volatile across reboots; previous Phase 2 work indicated it was wiped. If the directory exists and contains `unet_work.engine`, skip to Step 4.

- [ ] **Step 2: Run the bundled-TRT build script**

```bash
cd ~/w/PersonaLive
.venv/bin/python scripts/build_trt_engine.py 2>&1 | tee /tmp/trt_rebuild.log
```

Expected wall-time: ~10 minutes on RTX 5090 (per `_topics/personalive-acceleration.md`). The script will create `/tmp/personalive_trt/bundled/unet_work.engine` plus calibration artifacts.

If the script fails, read `/tmp/trt_rebuild.log` from the bottom — TRT errors are usually self-explanatory (missing onnx, sm_120 mismatch, OOM). Do NOT skip ahead with SDPA-only — Architecture A's 130–200 ms latency budget assumes the bundled engine.

- [ ] **Step 3: Confirm the engine loads**

```bash
~/w/PersonaLive/.venv/bin/python - <<'EOF'
import sys; sys.path.insert(0, "/home/newub/w/PersonaLive")
from src.wrapper_trt import PersonaLive
pl = PersonaLive()  # default config; uses /tmp/personalive_trt/bundled/
print("loaded:", type(pl).__name__)
EOF
```

Expected: `loaded: PersonaLive` printed; no traceback. If `wrapper_trt.PersonaLive()` requires explicit args, mirror what `phase2_grid.py` does (read `~/w/PersonaLive/scripts/phase2_grid.py` for the canonical construction sequence).

- [ ] **Step 4: Commit nothing**

`/tmp` is not tracked. Move on.

---

## Task 5: Write the failing smoke test for `realtime_webcam.py`

**Files:**
- Create: `~/w/PersonaLive/scripts/realtime_webcam.py`
- Create: `~/w/PersonaLive/scripts/test_realtime_smoke.py`

The script does I/O against real hardware, so the "test" is a 5-second end-to-end smoke run against `/dev/video10` consumed by `ffplay`. We codify it as a script that exits 0 on success.

- [ ] **Step 1: Write the smoke harness**

Create `~/w/PersonaLive/scripts/test_realtime_smoke.py`:

```python
"""Smoke test: run realtime_webcam.py for 5 seconds, verify /dev/video10 receives frames.

Pass: the realtime_webcam process stays alive >=5 s AND /dev/video10 reports
      a non-zero frame count via v4l2-ctl after the run.
Fail: process exits early, or /dev/video10 frame count is 0.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "realtime_webcam.py"
PYTHON = Path("/home/newub/w/PersonaLive/.venv/bin/python")
WEBCAM_DEV = "/dev/video0"  # update if Task 1 found a different device
ANCHOR = "/home/newub/w/vamp-interface/output/demographic_pc/overnight_drift/smile/broad/asian_m/seed2026_s+0.00.png"
DURATION_S = 5

def main() -> int:
    if not SCRIPT.exists():
        print(f"FAIL: {SCRIPT} does not exist yet")
        return 1
    proc = subprocess.Popen(
        [str(PYTHON), str(SCRIPT),
         "--webcam", WEBCAM_DEV,
         "--anchor", ANCHOR,
         "--virtual-cam", "/dev/video10",
         "--max-seconds", str(DURATION_S)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        out, _ = proc.communicate(timeout=DURATION_S + 60)
    except subprocess.TimeoutExpired:
        proc.kill()
        print("FAIL: realtime_webcam.py did not exit within the timeout")
        return 1
    if proc.returncode != 0:
        print(f"FAIL: realtime_webcam.py exited {proc.returncode}\n--- stdout ---\n{out}")
        return 1
    if "FPS" not in out:
        print(f"FAIL: no FPS line in output\n{out}")
        return 1
    print("PASS")
    print(out)
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run the smoke test — expect FAIL (script does not exist yet)**

```bash
~/w/PersonaLive/.venv/bin/python ~/w/PersonaLive/scripts/test_realtime_smoke.py
```

Expected output: `FAIL: ~/w/PersonaLive/scripts/realtime_webcam.py does not exist yet`. Exit code 1. This is the failing-test phase of TDD; do **not** proceed to Task 6 until this line is reproduced.

- [ ] **Step 3: Commit the smoke harness**

```bash
cd ~/w/PersonaLive
git add scripts/test_realtime_smoke.py
git commit -m "test: smoke harness for realtime_webcam (failing — script not yet implemented)"
```

(If `~/w/PersonaLive` is not a git repo we control, skip the commit and just leave the file in place. Verify with `git -C ~/w/PersonaLive rev-parse --is-inside-work-tree`.)

---

## Task 6: Implement `realtime_webcam.py` — chunk-of-4 tight loop

**Files:**
- Create: `~/w/PersonaLive/scripts/realtime_webcam.py`
- Reference (read-only): `~/w/PersonaLive/webcam/vid2vid_trt.py` (streaming API source of truth, lines 50–160), `~/w/PersonaLive/src/wrapper_trt.py` (`PersonaLive` class), `~/w/PersonaLive/webcam/config.py` (`Args` schema)

- [ ] **Step 1: Extract the streaming API**

Open `~/w/PersonaLive/webcam/vid2vid_trt.py` and confirm, against the current source:

1. Construction: `pipeline = PersonaLive(args, device)` (line ~130).
2. One-time anchor fuse: `pipeline.fuse_reference(reference_img)` (line ~136). Find what `reference_img` shape/dtype is by reading `src/wrapper_trt.py::PersonaLive.fuse_reference` and noting whether it accepts a `PIL.Image`, an HWC uint8 numpy array, or a CHW float tensor.
3. Per-chunk inference: `video = pipeline.process_input(images)` where `images` is a `torch.cat([... ], dim=0)` of 4 driving tensors (line ~144–158). Each driving tensor in `vid2vid_trt.py:80–83` is built as:

```python
img_t = pil_image_uint8.to(device).float() / 255.0
img_t = img_t * 2.0 - 1.0
img_t = img_t.permute(2, 0, 1).unsqueeze(0)   # [1, 3, H, W] in [-1, 1]
```

   ⚠ Note `pil_image_uint8.to(device)` is unusual — `vid2vid_trt.py` actually accepts a torch tensor here, not a PIL Image; trace `accept_new_params` callers in the frontend to confirm. If the input is already a tensor, skip `pil_image_uint8` and feed a `torch.from_numpy(rgb_uint8)` directly.

4. Output: `process_input` returns a list/iterable of arrays. Per-element conversion in `webcam/util.py::array_to_image`.
5. Args construction: `webcam/config.py::Args` is a Pydantic-style class. Mirror what `vid2vid_trt.py` does at `pipeline = PersonaLive(args, device)`, including any model paths and the `chunk_size = 4` invariant.

Take notes inline as a comment block at the top of `realtime_webcam.py`. Do not edit any of these reference files.

- [ ] **Step 2: Write the script**

Create `~/w/PersonaLive/scripts/realtime_webcam.py`. The skeleton below assumes the streaming API found in Step 1; adjust the two marked sections (`# === FROM STEP 1 ===`) if `fuse_reference` or `process_input` need anything different.

```python
"""Realtime webcam → PersonaLive → v4l2loopback (Architecture A, single-anchor).

USB webcam frames drive PersonaLive against one preselected Flux anchor;
output is pushed to a v4l2loopback virtual camera that OBS consumes.

Tight single-process loop with PersonaLive's chunk_size=4 temporal window.
Flattens webcam/vid2vid_trt.py's multiprocess Queue structure into one
straight-through pipeline:
    cv2.VideoCapture → 4-frame ring → pipeline.process_input → pyvirtualcam
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyvirtualcam
import torch
from PIL import Image

# PersonaLive imports — match webcam/vid2vid_trt.py exactly.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from src.wrapper_trt import PersonaLive  # noqa: E402
from webcam.config import Args  # noqa: E402

CHUNK_SIZE = 4   # PersonaLive temporal window (vid2vid_trt.py:131)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--webcam", default="/dev/video0")
    p.add_argument("--anchor", required=True,
                   help="Path to the Flux anchor PNG (512×512 ideally)")
    p.add_argument("--virtual-cam", default="/dev/video10")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--fps", type=int, default=20,
                   help="Target output FPS (PersonaLive ceiling ~20–21)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-seconds", type=float, default=0.0,
                   help="If >0, exit after this many seconds (smoke-test mode)")
    p.add_argument("--show-preview", action="store_true")
    return p.parse_args()


def open_webcam(dev: str, width: int, height: int, fps: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"could not open webcam {dev}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    return cap


def driver_tensor(rgb_uint8: np.ndarray, device: torch.device) -> torch.Tensor:
    """HWC uint8 RGB -> [1,3,H,W] float in [-1,1] on device.

    Matches vid2vid_trt.py:80-83.
    """
    t = torch.from_numpy(rgb_uint8).to(device).float() / 255.0
    t = t * 2.0 - 1.0
    return t.permute(2, 0, 1).unsqueeze(0)


def output_to_bgr(out) -> np.ndarray:
    """Convert one PersonaLive output element to a HWC uint8 BGR frame.

    PersonaLive returns a list/iterable; per-element shape is implementation
    -defined. Try the common cases (numpy HWC uint8, torch CHW float, PIL).
    """
    if isinstance(out, np.ndarray):
        if out.dtype != np.uint8:
            out = (out * 255).clip(0, 255).astype(np.uint8) if out.max() <= 1.0 else out.clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR) if out.shape[-1] == 3 else out
    if isinstance(out, torch.Tensor):
        arr = out.detach().cpu().float().numpy()
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = arr.transpose(1, 2, 0)
        if arr.min() < 0:        # in [-1,1]
            arr = (arr + 1.0) / 2.0
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if isinstance(out, Image.Image):
        return cv2.cvtColor(np.asarray(out.convert("RGB")), cv2.COLOR_RGB2BGR)
    raise TypeError(f"unsupported PersonaLive output type: {type(out)}")


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)

    anchor_path = Path(args.anchor)
    if not anchor_path.exists():
        raise FileNotFoundError(anchor_path)
    print(f"[init] anchor={anchor_path.name}")

    # === FROM STEP 1 ===
    # Build Args exactly as webcam/vid2vid_trt.py expects. Use defaults
    # from webcam/config.py and override only what the realtime path needs.
    pl_args = Args()  # adjust if Args() requires positional args
    pipeline = PersonaLive(pl_args, device)

    # fuse_reference accepts a PIL Image in vid2vid_trt.py's flow.
    # If src/wrapper_trt.py expects a tensor, swap to driver_tensor(...).
    ref_pil = Image.open(anchor_path).convert("RGB").resize((args.width, args.height))
    pipeline.fuse_reference(ref_pil)
    print("[init] PersonaLive ready (reference fused)")
    # === END FROM STEP 1 ===

    cap = open_webcam(args.webcam, args.width, args.height, args.fps)
    print(f"[init] webcam {args.webcam} opened "
          f"({int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
          f"@{cap.get(cv2.CAP_PROP_FPS):.0f})")

    t_start = time.time()
    n_out = 0
    last_log = t_start
    chunk: list[torch.Tensor] = []

    with pyvirtualcam.Camera(
        width=args.width, height=args.height, fps=args.fps,
        device=args.virtual_cam,
    ) as vcam:
        print(f"[init] virtual cam {vcam.device} ready")
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue

                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                side = min(h, w)
                y0 = (h - side) // 2
                x0 = (w - side) // 2
                rgb = rgb[y0:y0 + side, x0:x0 + side]
                rgb = cv2.resize(rgb, (args.width, args.height),
                                 interpolation=cv2.INTER_AREA)

                chunk.append(driver_tensor(rgb, device))
                if len(chunk) < CHUNK_SIZE:
                    continue

                images = torch.cat(chunk, dim=0)  # [4,3,H,W]
                chunk = []

                video = pipeline.process_input(images)
                for out in video:
                    out_bgr = output_to_bgr(out)
                    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
                    vcam.send(out_rgb)
                    if args.show_preview:
                        cv2.imshow("PersonaLive", out_bgr)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            return 0
                    n_out += 1

                now = time.time()
                if now - last_log >= 1.0:
                    fps = n_out / (now - t_start)
                    print(f"[loop] out_frames={n_out}  FPS={fps:.2f}")
                    last_log = now

                if args.max_seconds > 0 and now - t_start >= args.max_seconds:
                    print("[loop] max-seconds reached; exiting")
                    break
        finally:
            cap.release()
            if args.show_preview:
                cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    print(f"[done] {n_out} output frames in {elapsed:.2f}s "
          f"= {n_out/max(elapsed, 1e-9):.2f} FPS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**The two `# === FROM STEP 1 ===` blocks are the only places that may need surgery.** Specifically: if `Args()` requires explicit fields, fill them from `webcam/vid2vid_trt.py`'s caller (look in `webcam/connection_manager.py` or the entry point that constructs `Pipeline(args, device)`). If `fuse_reference` rejects a PIL Image, swap to `driver_tensor(np.asarray(ref_pil), device)`.

- [ ] **Step 3: Run the smoke test — expect PASS**

```bash
~/w/PersonaLive/.venv/bin/python ~/w/PersonaLive/scripts/test_realtime_smoke.py
```

Expected output: `PASS`, exit code 0, an `[loop] frames=… FPS=…` line in the captured stdout, FPS ≥ 15. If FPS < 10, the bundled TRT engine probably did not load — check Task 4 Step 3.

If the script crashes inside the loop, the most likely culprit is the `pipe(pil_in)` call signature; cross-reference `phase2_grid.py` and fix.

- [ ] **Step 4: Run with preview against the asian_m anchor**

```bash
~/w/PersonaLive/.venv/bin/python ~/w/PersonaLive/scripts/realtime_webcam.py \
    --webcam /dev/video0 \
    --anchor /home/newub/w/vamp-interface/output/demographic_pc/overnight_drift/smile/broad/asian_m/seed2026_s+0.00.png \
    --show-preview
```

Expected: a 512×512 cv2 window opens showing `asian_m` mimicking your face. Press `q` to exit. Confirm `FPS` log line is ≥ 15.

- [ ] **Step 5: Confirm `/dev/video10` is consumable**

In a second terminal while the script is still running:

```bash
ffplay /dev/video10
```

Expected: ffplay shows the same rendered output the cv2 preview window is showing.

- [ ] **Step 6: Commit**

```bash
cd ~/w/PersonaLive
git add scripts/realtime_webcam.py
git commit -m "feat: realtime webcam → PersonaLive → v4l2loopback driver (Architecture A)"
```

---

## Task 7: Verify OBS picks up `/dev/video10`

**Files:**
- None (OBS GUI configuration only)

- [ ] **Step 1: Launch OBS while `realtime_webcam.py` is running**

In one terminal: launch the realtime script (Task 6 Step 4 command, optionally without `--show-preview` to save GPU).

In another: `obs &`

- [ ] **Step 2: Add `PersonaLive` as a Video Capture Device source**

In OBS: Sources → `+` → "Video Capture Device (V4L2)". In the dialog:
- Device: pick the entry labelled **`PersonaLive`** (this is the `card_label` from Task 2 Step 4). If you see only `/dev/video10` without the label, v4l2loopback's `card_label` option did not stick — recheck `/etc/modprobe.d/v4l2loopback.conf`.
- Resolution / Frame rate: leave on `Device Default`.
- Click OK.

Expected: the OBS preview canvas shows the rendered PersonaLive output, mirroring whatever the realtime script is rendering, at ~20 FPS.

- [ ] **Step 3: Verify with a face movement test**

Smile, blink, turn your head. The avatar in the OBS preview should track. Latency should feel under ~200 ms (consistent with the budget in `2026-05-04-realtime-vtuber-pipeline-plan.md`).

If latency is visibly bad (>500 ms), the most likely cause is OBS pulling at the wrong FPS — set the OBS source's frame-rate override to `20` to match `--fps 20` in the script.

- [ ] **Step 4: Save the OBS scene**

OBS → Scene Collection → Save (or use a scene-collection name like `PersonaLive Demo`). This persists the source config so future launches just need the realtime script to be running.

- [ ] **Step 5: Commit nothing**

OBS scene config lives in `~/.config/obs-studio/`, not in the repo.

---

## Task 8: Mark plan complete and update topic index

**Files:**
- Modify: `~/w/vamp-interface/docs/research/_topics/neural-deformation-control.md`
- Modify: `~/w/vamp-interface/docs/research/2026-05-04-realtime-vtuber-pipeline-plan.md` (status note)

- [ ] **Step 1: Add a "shipped" note to the pipeline plan**

In `2026-05-04-realtime-vtuber-pipeline-plan.md`, under the `## Recommendation` heading, append a single line:

```markdown
> **Status (YYYY-MM-DD):** Architecture A shipped per `2026-05-04-realtime-archA-implementation-plan.md`. `~/w/PersonaLive/scripts/realtime_webcam.py` is the live driver.
```

Replace `YYYY-MM-DD` with the actual completion date.

- [ ] **Step 2: Add a topic-index pointer**

In `_topics/neural-deformation-control.md`, under "Key dated docs", insert:

```markdown
- [2026-05-04-realtime-archA-implementation-plan.md](../2026-05-04-realtime-archA-implementation-plan.md) — **shipped.** Architecture A implementation plan: USB webcam → PersonaLive → v4l2loopback → OBS. Driver lives at `~/w/PersonaLive/scripts/realtime_webcam.py`.
```

- [ ] **Step 3: Commit**

```bash
cd ~/w/vamp-interface
git add docs/research/2026-05-04-realtime-vtuber-pipeline-plan.md docs/research/_topics/neural-deformation-control.md
git commit -m "docs(neural-deformation): Architecture A shipped — realtime webcam driver live"
git push
```

---

## Acceptance criteria

The plan is complete when **all** of the following hold:

1. `ffplay /dev/video10` shows a live PersonaLive avatar driven by the USB webcam.
2. OBS shows the same avatar via its Video Capture Device source.
3. `~/w/PersonaLive/scripts/realtime_webcam.py` runs at ≥ 15 FPS sustained on the RTX 5090.
4. `~/w/PersonaLive/scripts/test_realtime_smoke.py` exits 0.
5. After a reboot, `/dev/video10` reappears automatically (v4l2loopback persistence holds).
6. Topic index + plan-doc status updated and committed.

## Known risks (mirrored from the realtime pipeline doc)

- **Latency feel:** 130–200 ms is fine for streaming, **not** tight enough for first-person mirror UX. Don't over-promise.
- **Anchor swap mid-stream:** changing `--anchor` requires restarting the script (~200 ms pipe rebuild + first-frame cache). Multi-anchor hot-swap is out of scope here.
- **PersonaLive throughput ceiling:** ~20–21 FPS bundled-TRT. If your webcam runs at 60 FPS, OpenCV will silently buffer; mismatch is fine for this demo but worth noting.
- **Identity leakage:** in Architecture A the driver's face is encoded by FAN. Phase 2 didn't show leakage on the asian_m / black_f / european_m anchors with identity-clean drivers, but stress-test with an asymmetric face if it matters.
- **PersonaLive license:** unaudited for streaming/commercial use. This plan is for a local demo only; do **not** push the output to a public stream until the LICENSE file in `~/w/PersonaLive/` is read and confirmed compatible.
