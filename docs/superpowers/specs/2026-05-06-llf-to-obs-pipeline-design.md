---
status: live
topic: arkit-bridge
---

# Live Link Face → PersonaLive → OBS pipeline design

End-to-end real-time pipeline. iPhone running Live Link Face streams ARKit
b_61 over UDP to a Linux box with RTX 5090; the shipped ARKit bridge
(student_v2_120k + closed_form_pose, `EULER_SIGNS=(+1,-1,+1)`, `F=I`)
drives PersonaLive; the rendered 512×512 portrait frames are written to a
v4l2loopback device that OBS picks up as a regular webcam.

## Regime and product context

This is a **Regime A** (realtime puppeteering) product per
`docs/research/2026-05-06-vtuber-pipeline-priorities.md` — hard FPS
budget, OBS as terminal sink, performer drives a portrait. Specifically
it's the **ARKit-driven PersonaLive vtuber** product. The companion
products that share this scaffolding:

| Product | Driver | Backbone | Status |
|---|---|---|---|
| ARKit-driven PersonaLive vtuber | Live Link Face UDP | PersonaLive bridge mode | **this spec** |
| RGB-driven PersonaLive vtuber | webcam + facemesh | PersonaLive teacher_full | next-product, reuses scaffolding |
| RGB-driven FasterLivePortrait vtuber | webcam | FasterLivePortrait + TRT engines | next-product, reuses scaffolding (Regime A backup if PersonaLive stylized fails) |
| Static-portrait slider authoring | slider params | Flux + LoRA / FluxSpace | Regime B, separate thread |

**Scaffolding reuse**: the units here (`LLFReceiver`, `V4L2Sink`, daemon
shell, `BatchDriver` interface) are designed so the next two Regime-A
products are *new drivers in the same scaffolding*, not new pipelines.
The receiver is swapped for an RGB-camera grabber; the driver is swapped
for either a teacher_full pipeline build or a FasterLivePortrait engine
runner; sink and daemon stay. See `docs/research/2026-05-06-rendering-stack-replacement-options.md`
for the stack survey behind that triage.

**Out of scope here**: stylized / non-human anchors. PersonaLive's
Stage-2 StyleGAN2-FFHQ discriminator collapses non-human refs to generic
FFHQ blonde inside the silhouette — that's a separate Regime-A failure
mode (FasterLivePortrait branch will likely fix it; per-style LoRA or
discriminator swap on PersonaLive is the alternative). This pipeline
ships against photoreal anchors only.

## Constraints and ground state

- **No iPhone camera RGB on the wire.** Bridge mode patches both
  `pose_encoder` and `motion_encoder` seams (`patch_pose=True`,
  `patch_motion=True` in `apply_bridge_to_personalive.py`) — every RGB
  consumer is bypassed and replaced by the closed-form pose path +
  `MotEncoderStudent`. The reference image is the static anchor portrait
  loaded once at startup. **The Larix/SRT/NDI iPhone-camera ingest path
  is not on the critical path for the bridge pipeline** (it's a
  separable thread for if/when we want camera-driven LivePortrait).
- **PersonaLive throughput ceiling**: 18-21 FPS on RTX 5090 with
  torch_tensorrt (per `_topics/personalive-acceleration.md`). Target for
  this pipeline: stable ≥20 FPS to OBS, ≤150 ms glass-to-OBS latency.
- **Pipeline streaming primitive**: `Pose2VideoPipeline_Stream` in
  PersonaLive operates on **windows of `temporal_window_size=4` frames**
  with persistent `motion_bank`. So real-time loop is window-paced: pull
  4 b_61 frames, run one window of denoising, emit 4 RGB frames.
- **LLF source rate** is 60 FPS (iPhone TrueDepth), our render rate is
  ~20-25 FPS, so we will subsample b_61 on the fly. The student already
  takes per-frame b_expr; subsampling is fine.
- **OBS sink**: v4l2loopback device. OBS reads it as "Video Capture
  Device". This is the universally-compatible path on Linux (also works
  with browsers, Zoom, Discord); no NDI license / extra plugins needed.

## Components

The pipeline is four units with narrow interfaces. Each is independently
testable.

### `llf_receiver` — UDP listener with latest-frame cache

A daemon thread that opens a UDP socket on `0.0.0.0:11111`, decodes each
packet via the existing `livelink_probe.decode_packet` decoder, and
maintains a *latest-N* ring buffer (default N=8, ≈133 ms at 60 FPS).
Older packets are dropped silently. Exposes `pop_latest_window(k=4)`
which returns the k most-recent decoded packets in arrival order, or
blocks (with timeout) until k arrive on a fresh stream.

Reuses the `decode_packet` function in `scripts/livelink_probe.py`
verbatim — promote it from a script to `src/arkit_bridge/llf_udp.py`
without changing semantics; the script can re-import.

### `streaming_pipe` — windowed PersonaLive driver

Wraps an already-built `Pose2VideoPipeline_Stream` and runs one
`temporal_window_size=4` window per call. Holds the persistent
`motion_bank` and `latents_buffer` across calls (this is what makes it
*streaming* despite operating on windows).

Public API:

```python
class StreamingDriver:
    def __init__(self, pipe, anchor_image, ref_kp_ref, student, ...): ...
    def step(self, b61_window: list[B61Packet]) -> np.ndarray:
        """Take 4 b_61 packets, return (4, H, W, 3) uint8 RGB."""
```

Internally `step` does, per call: (1) compute closed-form `k_d` for each
of the 4 frames via `compose_kd`; (2) run `MotEncoderStudent` on each
b_expr → 4 motion features; (3) one window of `Pose2VideoPipeline_Stream`
denoising (4 steps, jump=4 with `num_inference_steps=4`); (4) VAE decode
to 512×512; (5) return uint8.

This is a refactor of the run loop currently inlined in
`scripts/apply_bridge_to_personalive.py`. The seam-patching logic moves
into `streaming_pipe` so the script becomes a thin user.

### `v4l2_sink` — v4l2loopback writer

Subprocess wrapper around ffmpeg:

```
ffmpeg -loglevel error -f rawvideo -pix_fmt rgb24 -s 512x512 -r 25
       -i pipe:0 -f v4l2 -pix_fmt yuv420p /dev/video10
```

Python writes 512×512×3 raw RGB bytes to ffmpeg's stdin per frame;
ffmpeg handles colorspace conversion and v4l2 ioctls. On startup the
daemon checks `/dev/video10` exists and is `v4l2loopback`-backed; if not
it prints the `modprobe` invocation and exits.

### `daemon` — orchestration

Top-level entry point:

```bash
python -m vamp_interface.streaming_bridge \
    --anchor data/portraits/anchor_001.png \
    --port 11111 --device /dev/video10 --fps 25
```

Wires the three components: receiver thread, main render thread, ffmpeg
subprocess. Main loop: pull 4 packets → `streaming_pipe.step()` → 4
frames to ffmpeg stdin. Logs realised FPS, queue depth, drop count
every 5 s. Ctrl-C drains, joins, closes ffmpeg cleanly.

## Data flow

```
iPhone                         Linux box (RTX 5090)                       OBS
┌───────────┐  UDP 11111   ┌──────────────┐    ring of    ┌──────────────┐
│ Live Link │ ───────────► │ llf_receiver │ ─ b_61 ───►   │ streaming_   │
│   Face    │   60 FPS     │   (thread)   │   latest 8    │    pipe      │
│  (b_61)   │              └──────────────┘               │ (main loop)  │
└───────────┘                                             │              │
                                                          │ pull 4 → 1   │
                                                          │ window infer │
                                                          │ → 4 RGB      │
                                                          └──────┬───────┘
                                                          rgb24  │
                                                                 ▼
                                                          ┌──────────────┐
                                                          │   ffmpeg     │
                                                          │  → v4l2 -    │
                                                          │ /dev/video10 │
                                                          └──────┬───────┘
                                                                 │ V4L2
                                                                 ▼
                                                          ┌──────────────┐
                                                          │     OBS      │
                                                          │ Video Capture│
                                                          │   Device     │
                                                          └──────────────┘
```

## Approaches considered

**A. Headless single-process daemon → v4l2loopback (chosen).** One
Python process, UDP receiver thread + render thread + ffmpeg subprocess.
Simplest mental model, fewest moving parts, no IPC overhead, easy to
profile. Weakness: a long GC pause or CUDA hiccup causes frame stutter.
Mitigated by ring buffer dropping stale b_61 packets rather than
queueing.

**B. Two-process pipeline with shared memory ring.** LLF receiver in
process A, PersonaLive worker in process B, communicating over
`multiprocessing.shared_memory`. Better isolation, render lag can't
back-pressure the UDP socket. Cost: more code, harder profiling, no
real benefit on a 5090 where Python+CUDA are the only consumers
(the UDP receiver is sub-microsecond work). Reject for v1; revisit if
the receiver actually drops packets in practice.

**C. GStreamer pipeline with `appsrc`/`v4l2sink`.** Idiomatic Linux
real-time video. Lets us push numpy buffers into a real GStreamer graph
and use its sinks for free. Cost: PersonaLive is Python-resident; we'd
either marshal frames over `appsrc` (essentially what ffmpeg-stdin does
already) or write a GST plugin (high effort). The ffmpeg-subprocess
path achieves the same result with one syscall pipe. Reject as
overengineering.

**Rejected sink alternatives**: NDI-out from Linux works (libndi sends
uncompressed fine; the broken decoder is receive-side only) and OBS has
an NDI input plugin, but v4l2loopback is more universal and what the
project's other Linux-vtuber-stack tools assume. NDI-out can be added
later as a second sink without changing the streaming_pipe core.

## Acceptance gates (priorities-doc requirements)

Per `docs/research/2026-05-06-vtuber-pipeline-priorities.md`, the
ARKit-PersonaLive bridge has explicit gates that must clear *before* OBS
plumbing ships. Status as of 2026-05-06:

| Gate | Status | Evidence |
|---|---|---|
| Sign-agreement on yaw/pitch/roll vs ground-truth ARKit | ✅ done | axis-isolated FIXED2 collage; `_topics/arkit-bridge.md` |
| Head-attenuation-at-extremes diagnosis from `render_metrics.parquet` | ❌ pending | Task 0a in plan |
| Bridge inference latency micro-bench (target: ≤2 ms/frame on 5090, scaled from priorities-doc 5 ms/frame on 4080) | ❌ pending | Task 0b in plan |
| End-to-end live: iPhone → ARKit → daemon → PersonaLive → OBS on a short take | ❌ pending | V1 itself (Task 5) |

V1 cannot ship to OBS as a deliverable until the head-attenuation
diagnosis confirms no attenuation regression and the latency bench
confirms the bridge is not the bottleneck.

## Acceptance criteria (output side)

Per the priorities doc's RGB-PersonaLive OBS gate, scaled to 5090:

- Sustained ≥18 FPS effective output rate at 512² (priorities-doc says
  ≥15 FPS on 4080; 5090 is ~30% faster on this workload, so 18 FPS is
  the rescaled gate).
- Identity stable on photoreal refs over a 5-minute take (no drift,
  no temporal flicker beyond background noise).
- No observable rotation / crop bugs (the rotation-flag hazard is
  resolved upstream by `cv2.CAP_PROP_ORIENTATION_AUTO=1` on training,
  and is moot in bridge mode because RGB consumers are bypassed).

## Open questions and smoke tests

- **Does `Pose2VideoPipeline_Stream` actually run correctly with
  `video_length=4` per call, persisting `motion_bank` externally?**
  Currently `apply_bridge_to_personalive.py` calls `pipe()` once on the
  whole sequence. The `Stream` class's window loop assumes a known total
  `video_length`. Smoke test: refactor to pull windows in a Python loop,
  check first 8 frames match the offline single-call rendering bit-for-bit
  on a recorded b_61 take.
- **Realised FPS budget**: target 25 FPS = 40 ms/frame = 160 ms/window
  of 4 frames. Probe B/C lands ~50 ms/frame at non-windowed, so ~200
  ms/window (4×). Realistic ceiling 5-6 windows/s → 20-24 FPS. **Smoke
  test before committing to 25 FPS** — drop to 20 FPS if necessary.
- **First-frame stall**: cold-start CUDA + tensorrt engine load is
  ~10-20 s. Daemon should pre-warm with a synthetic b_61 zero-vector
  window before opening the v4l2 device, so OBS doesn't see a black
  flash.
- **Glass-to-OBS latency**: budgeted ≤150 ms but unmeasured. Smoke rig:
  record iPhone clock-flash via screen mirror, compare to OBS-recorded
  rendered output, count frame offset. Acceptance: <8 frames at 25 FPS.
- **Does v4l2loopback survive ffmpeg restart?** OBS sometimes holds the
  device open. Daemon should `MEMfd`-open before ffmpeg or document
  "restart OBS source after daemon restart."

## Error handling

- **No LLF packets within 2 s of startup**: log warning, render anchor
  portrait passthrough (zero b_expr, identity rotation) so OBS sees a
  signal. Resume animation when packets arrive.
- **LLF packet decode failure**: log once per error class, drop that
  packet, continue.
- **CUDA OOM mid-stream**: not expected on a 5090 with PersonaLive's
  ~10 GB footprint, but if it happens, daemon emits a cooldown frame
  (last-good RGB held) and tries to recover for one window before
  exiting. Don't try to be clever — exit and let systemd restart.
- **ffmpeg subprocess dies** (e.g., user pulled v4l2loopback module):
  daemon detects via stdin write failure, exits, systemd restarts.

## Testing

- **Unit**: `llf_receiver` against synthetic UDP packets (loopback
  socket, send 1000 packets at 60 FPS, assert ring contains last 8,
  arrival order preserved).
- **Unit**: `streaming_pipe.step()` on a recorded 60-frame b_61 CSV;
  assert per-frame RGB matches the offline `apply_bridge_to_personalive`
  baseline within ε (allow small differences from window vs full-clip
  motion_bank initialization).
- **Integration**: 30-second iPhone-tethered run. Capture both the iPhone
  screen (LLF preview with timestamp burned in) and the OBS preview;
  measure latency frame-by-frame.
- **Soak**: 30-minute run on a podcast-length take, check no memory
  growth, no FPS drift, no v4l2 device hang.

## What ships

- `src/arkit_bridge/llf_udp.py` — promoted decoder + receiver class.
- `src/arkit_bridge/streaming_pipe.py` — windowed driver wrapping
  `Pose2VideoPipeline_Stream` + bridge seam patches.
- `src/arkit_bridge/v4l2_sink.py` — ffmpeg subprocess sink.
- `scripts/streaming_bridge.py` — daemon entry point.
- `docs/research/2026-05-06-llf-obs-runbook.md` — operator guide
  (modprobe v4l2loopback, iPhone setup, OBS source picker, latency
  smoke test).

## Deferred

- Larix/SRT iPhone-camera ingest (separate pipeline; only needed for
  LivePortrait/X-Nemo paths that consume RGB).
- NDI-out as second sink.
- Multi-anchor switching (swap reference portrait via a control socket).
- LLF-to-disk recorder for retro-debug; can shell out to
  `livelink_probe.py --record`.
