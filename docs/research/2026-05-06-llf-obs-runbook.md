---
status: live
topic: arkit-bridge
---

# Live Link Face → PersonaLive → OBS runbook

End-to-end operator guide for running the streaming bridge locally on
Linux. V1 ships at batch-of-24 latency ~3.5 s glass-to-OBS — fine for a
vtuber mirror, not for real-time conversation overlay. V2 (cohort-stream
refactor of `Pose2VideoPipeline_Stream.__call__`) is deferred and would
bring this to ~150 ms.

## One-time setup

### Linux box

```bash
sudo apt install v4l2loopback-dkms ffmpeg
```

The daemon needs the PersonaLive venv (project venv has diffusers≥0.37
which is incompatible with PersonaLive's `MotEncoder`):
`/home/newub/w/PersonaLive/.venv/bin/python` is the runtime.

### iPhone

1. Install **Live Link Face** from the App Store (free, by Epic Games).
2. Open the app → gear icon → "Live Link" → "Add Target".
3. Set Host = the Linux box's LAN IP, Port = 11111. Tap OK.
4. Toggle "Live Link" on. The indicator turns green when the daemon's
   receiver is running and the iPhone has a route.

## Per-session startup

### Load v4l2loopback (every reboot)

```bash
sudo modprobe v4l2loopback devices=1 video_nr=10 \
  card_label=PersonaLive exclusive_caps=1
ls -la /dev/video10   # confirm device exists
```

### Start the daemon

```bash
cd ~/w/vamp-interface
PYTHONPATH=src /home/newub/w/PersonaLive/.venv/bin/python \
  scripts/streaming_bridge.py \
  --reference data/llf-phase2/asian_m__06_neutral.midframe.png \
  --ckpt runs/student_v2_120k/student_best.pt \
  --device_path /dev/video10 \
  --port 11111 --batch 24 --fps 25
```

Wait for `daemon ready; waiting for LLF packets`. Cold start is
10-20 s (pipe build) plus one pre-warm batch (~4.5 s on a 5090).

### Start the iPhone stream

Toggle Live Link on in the app. The daemon logs `rendered=… fps=…`
lines every 5 s.

### Wire OBS

- Sources → + → **Video Capture Device**
- Device: `PersonaLive` (the `card_label` set above)
- Resolution: 512×512, 25 FPS

## Smoke test (no iPhone)

To verify the pipeline without an iPhone:

```bash
# Terminal A: launch daemon with --mp4_out (skips v4l2loopback dependency)
PYTHONPATH=src /home/newub/w/PersonaLive/.venv/bin/python \
  scripts/streaming_bridge.py \
  --reference data/llf-phase2/asian_m__06_neutral.midframe.png \
  --ckpt runs/student_v2_120k/student_best.pt \
  --mp4_out /tmp/streaming_bridge.mp4 \
  --port 21112 --batch 24 --fps 25

# Terminal B: send 60 synthetic LLF packets
python /tmp/llf_sender.py 21112 60   # script in tasks/04-task-5 commit msg
```

Expected: daemon logs `rendered=24 fps=… infer_ms=~3000 rx_received=60
rx_dropped=0`, mp4 file is non-empty 512×512 @ 25 FPS.

## RGB-driven smoke test (webcam, no iPhone)

The daemon also accepts a webcam frame source via `--driver rgb`, which
runs PersonaLive in `teacher_full` mode (real motion_encoder + real
pose_encoder consume RGB; bridge seams are NOT installed). Useful when
an iPhone isn't available, or for direct stylization experiments where
ARKit blendshapes aren't the input modality.

```bash
# 1. Verify a webcam exists.
v4l2-ctl --list-devices    # note the cam_index (often 0)

# 2. Launch daemon with --driver rgb. Use a SMALL batch — teacher_full
# does ~1.4× more work than bridge mode, so live-conversation latency
# is better with batch=8 than batch=24.
PYTHONPATH=src /home/newub/w/PersonaLive/.venv/bin/python \
  scripts/streaming_bridge.py \
  --driver rgb --cam_index 0 \
  --reference data/llf-phase2/asian_m__06_neutral.midframe.png \
  --ckpt runs/student_v2_120k/student_best.pt \
  --mp4_out /tmp/streaming_bridge_rgb.mp4 \
  --batch 8 --fps 25

# 3. Move your face in front of the camera. The daemon logs:
#   rendered=8 fps=~5 infer_ms=~1500 cap=… drop=… det_fail=…
# Stop with Ctrl-C. The mp4 will contain the rendered face animated
# by your captured motion.
```

Performance reality (RTX 5090, fp16, 4-step DDIM, measured 2026-05-07):

| batch | mean infer | FPS |
|---|---|---|
| 8 | 1.5 s | 5.2 |
| 16 | 2.3 s | 7.1 |
| 24 | 3.0 s | 8.1 |

`teacher_full` is materially slower than `bridge` mode because the real
PersonaLive `MotEncoder` is heavier than the distilled `MotEncoderStudent`
that the bridge swaps in. **5–8 FPS is the current ceiling for the RGB
driver** — the 18 FPS target from the plan is not achievable on this
hardware in teacher_full mode without a separate distill of the RGB-side
encoders, which is out of scope here.

For live-conversation use, prefer the smaller `--batch 8` so the
glass-to-OBS latency stays around 1.5 s instead of 3 s. The `RGBGrabber`
ring drops oldest frames on overflow, so a slow consumer falls back to
"most recent frames", not "stale frames".

Notes:

- `--driver rgb` is incompatible with `--mode v2` (V1 batch render only).
- Camera missing -> `RuntimeError: cv2.VideoCapture(N) returned no frames`
  at startup (during the `[4/4] starting RGB grabber` step). Fix the
  cam_index and re-run.
- `det_fail` count ticking up means MediaPipe FaceMesh failed to detect
  a face on that frame; the EMA cropper falls back to the last known
  bbox so output stays stable. Expected near 0 in normal lighting.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Permission denied: /dev/video10` | module not loaded or wrong owner | re-run `sudo modprobe …`; user must be in `video` group |
| `daemon ready` but no `rendered=` lines | iPhone can't reach Linux | same LAN, no AP isolation; `sudo tcpdump -ni any port 11111` |
| OBS shows black frame | daemon emits idle anchor before LLF arrives | toggle Live Link on; should animate |
| `rx_dropped` growing | render slower than ingest (expected at 60→25 FPS) | not a problem; ring drops oldest |
| Daemon exits with `output_type=='np'` deprecation error | wrong venv | use `/home/newub/w/PersonaLive/.venv/bin/python`, not `uv run` |
| Subject parsed as `?` in receiver logs | wire-format header drift | cosmetic only; floats still decode (see `reference_llf_udp_wire_format.md`) |

## Latency budget (V1, batch-of-24)

| stage | ms |
|---|---|
| LLF capture + UDP transit | ~20 |
| Batch fill (24 frames @ 60 FPS LLF) | ~400 |
| Pipe inference (24 frames @ ~8 FPS) | ~3000 |
| ffmpeg + v4l2 + OBS pickup | ~80 |
| **total glass-to-OBS** | **~3.5 s** |

Pipe inference is the dominant term. Measured 124 ms/frame on RTX 5090
at fp16 + 4-step DDIM (`scripts/streaming_bridge.py` smoke test:
`rendered=24 infer_ms=2987`).

V2 (cohort-stream refactor) chunk-renders 4 frames per cohort instead of
24, so the first cohort reaches the sink while subsequent cohorts are
still rendering. Total compute is identical to V1; the win is
first-frame latency.

## Latency budget (V2, cohort-stream, batch=24)

Run with `--mode v2`:

```bash
PYTHONPATH=src /home/newub/w/PersonaLive/.venv/bin/python \
  scripts/streaming_bridge.py ... --mode v2
```

| stage | ms |
|---|---|
| LLF capture + UDP transit | ~20 |
| Batch fill (24 frames @ 60 FPS LLF) | ~400 |
| `prepare_v2` (= setup + 3 warmup cohorts internally) | ~1500 |
| First `step_v2` (1 cohort, 4 decoded frames) | ~490 |
| ffmpeg + v4l2 + OBS pickup | ~80 |
| **first-cohort glass-to-OBS** | **~2.5 s** |

Subsequent cohorts within the same prepare_v2 window emit every ~490 ms.
Measured 2026-05-07 on RTX 5090 fp16 + 4-step DDIM (smoke test:
`prepare+6×step infer_ms=2941`, mean 490 ms/cohort, 60 packets in,
0 dropped).

V2 saves ~1 s on first-frame latency vs V1 at the cost of slightly
higher per-cohort scheduling overhead. The bigger structural win — true
≤200 ms steady-state — would require re-prepare-per-cohort or hoisting
warmup outside the per-batch path; both are out of scope for the
current LLF→OBS shipping target.

V1 remains the default (`--mode v1`) because operationally simpler.

## Stopping

Ctrl-C the daemon. Receiver → sink → pipe drain in order. OBS goes to
a black frame; restart the daemon to resume.
