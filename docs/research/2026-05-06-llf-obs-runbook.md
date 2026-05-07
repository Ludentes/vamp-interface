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

V2 (cohort-stream refactor) would chunk-render 4 frames per LLF window
instead of 24, dropping the batch-fill term to ~67 ms and inference to
~500 ms — total ~150 ms. Deferred behind a working V1.

## Stopping

Ctrl-C the daemon. Receiver → sink → pipe drain in order. OBS goes to
a black frame; restart the daemon to resume.
