---
status: live
topic: neural-deformation-control
---

# Realtime VTuber-with-a-real-face pipeline plan (iPhone + Linux + RTX 5090 + OBS)

**Date:** 2026-05-04
**Why this exists:** the Phase 2 default-decision (`2026-05-04-personalive-default-decision.md`) opened up a concrete product question — *"can we build a VTuber-with-a-real-face on this stack?"* This doc records the architectural shape, two viable paths, latency budgets, build effort, and open risks **before** the conversation that produced it gets compacted away.

**Direct answer:** yes, and the cheap path is buildable in ~half a day on top of work we already have.

## Two architectures

### A. RGB-driving (use PersonaLive natively)

The iPhone front camera is the driver; ARKit blendshapes are not used at all. This is the use case PersonaLive was designed for (`webcam/vid2vid_trt.py` already implements the multiprocess Queue version of this loop).

```
iPhone front camera (RGB, 30/60 fps)
    │
    │  WiFi 5GHz or USB tether
    │  ── transport candidate (see iphone-to-linux-webcam-transport doc)
    ↓
Linux 5090 box  /dev/video1 (or RTSP / NDI URL)
    │
    │  Python: cv2.VideoCapture → frame loop
    │  ┌─ Reference image preloaded into PersonaLive once at startup
    │  │  (asian_m, european_m, custom Flux portrait, …)
    │  └─ Per frame: motion_encoder(rgb) → 3D-UNet (4 step) → VAE.decode
    ↓                                    (bundled TRT engine, ~20 FPS, 50 ms/frame)
512² PIL frame  →  resize / pad to 720p
    ↓
PyVirtualCam or v4l2loopback /dev/video10
    ↓
OBS adds /dev/video10 as a Video Capture Device
    ↓
RTMP → Twitch/YouTube  *or*  NVENC → mkv recording
```

### B. ARKit-driving (needs the Phase 3 bridge)

Same topology, smaller motion signal. iPhone streams **Live Link Face OSC** packets (52 blendshapes + head/eye rotation) over WiFi. Linux receives OSC → ARKit-52 vector → bridge MLP → motion latent → PersonaLive UNet, bypassing the FAN motion encoder.

```
iPhone Live Link Face → OSC over WiFi (UDP)
    ↓
Linux: python-osc receiver → 60 Hz ARKit-52 vector
    ↓
ARKit→motion-latent bridge MLP   (~50 LOC, ~10 min training)
    ↓
PersonaLive (motion latent fed directly, no FAN encoder needed)
    ↓
[same downstream as A]
```

## Latency budget (glass-to-glass)

| Stage | Architecture A (RGB) | Architecture B (ARKit) |
|---|---|---|
| iPhone capture + encode | ~16 ms | ~16 ms |
| Network transport | 30–80 ms (RGB H.264/MJPEG) | <2 ms (OSC UDP) |
| FAN motion encode (in PersonaLive) | included in render | bypassed |
| PersonaLive render (bundled TRT, ~20 FPS) | 50 ms | 40 ms (no FAN) |
| v4l2loopback handoff + OBS frame pull | 20–40 ms | 20–40 ms |
| Display | 16 ms | 16 ms |
| **Total** | **130–200 ms** | **~80–130 ms** |

Architecture A is solid VTuber territory (Twitch streamers tolerate 200 ms+). Not tight enough for a first-person mirror UX (~50 ms is the perceptual cliff), but for a *streamed avatar* it's fine.

## Trade-off matrix

| Dimension | RGB-driving (A) | ARKit-driving (B) |
|---|---|---|
| Latency | 130–200 ms | ~80–130 ms |
| Fidelity | High (full visual signal) | Capped by 52-channel ARKit basis |
| iPhone resource | Camera + H.264 encoder | ARKit only (~5% CPU) |
| Network | 5–20 Mbps RGB stream | <100 KB/s OSC |
| Time to ship | **Days** | Weeks |
| User identity leak risk | Yes — the driver's face is encoded into FAN motion features | None — only blendshape coefficients leave the phone |
| Required new code | Webcam-loop adapter + virtual camera plumbing (~30 LOC) | Same + OSC receiver (~50 LOC) + bridge MLP training (~150 LOC) + FAN-bypass injection (~20 LOC) |

## What we already have

- PersonaLive bundled TRT engine (`/tmp/personalive_trt/bundled/unet_work.engine`, rebuildable in ~10 min from `PersonaLive/scripts/build_trt_engine.py`).
- `src.wrapper_trt.PersonaLive` — TRT runtime wrapper.
- `webcam/vid2vid_trt.py` Pipeline — multiprocess Queue-based webcam-to-output. **This is the existing realtime path. We adapt it; we don't build it from scratch.**
- 3 working anchor reference images (asian_m, black_f, european_m, all 512×512 Flux Krea, identity-validated by ArcFace floor 0.43 across 900 frames in Phase 2).

## Build manifest — Architecture A

| Component | Effort | Notes |
|---|---|---|
| iPhone → Linux RGB transport | 1–2 hrs | Researched separately in `2026-05-04-iphone-to-linux-webcam-transport.md`. Reincubate Camo does not support Linux; the practical Linux paths are NDI HX, RTSP-server iOS apps, or DroidCam/EpocCam variants. |
| Adapt `webcam/vid2vid_trt.py` to read `/dev/videoN` | ~30 LOC | Strip the multiprocess Queue shim; tight loop: `cap.read() → render → cv2.imshow / virtual_cam.send()`. |
| `v4l2loopback` module install + permissions | one-time | `apt install v4l2loopback-dkms`; `modprobe v4l2loopback video_nr=10 card_label="PersonaLive"` exclusive_caps=1; persist via `/etc/modules-load.d/`. |
| OBS scene + hotkeys | trivial | Add `/dev/video10` as a Video Capture Device source. |

**Total: ~half-day of work** plus whatever the iPhone transport choice costs.

## Build manifest — Architecture B (additional)

| Component | Effort | Notes |
|---|---|---|
| Live Link Face OSC receiver in Python | ~50 LOC | `pip install python-osc`; subscribe to LLF OSC schema (52 named ARKit channels + headRotation + eyeRotation), buffer to a ring with 60 Hz tick. |
| Phase 3 bridge MLP | ~150 LOC | Train on the 124 s ARKit take in `data/llf-takes/20260505_MySlate_2/`. Input: 52 ARKit floats (+ head/eye rotation). Output: PersonaLive motion-latent shape (TBD — Phase 1 Step 3 motion-API probe is still pending and is the gating dependency). |
| Bypass FAN encoder at inference | ~20 LOC | Inject the bridge-produced latent directly into `denoising_unet`'s motion-conditioning path; skip the FAN feature extractor entirely. |

**Total: ~3–5 days** assuming the bridge MLP trains cleanly.

## Recommendation

**Build A first, ship it, then decide on B.** A is essentially the productisation of work we've already done. B needs the Phase 3 bridge that's still load-bearing for the iPhone-pipeline plan independently — better to validate the realtime path with the lower-risk variant first.

The killer side-benefit: A is a **free PersonaLive demo**. Plug in the iPhone, point at your face, become any of our 3 anchors live in OBS. That kind of artefact validates the architecture more than any internal grid.

## Open risks

- **PersonaLive license** — has not been audited for streaming/commercial use. Must read the LICENSE file before any public broadcast or commercial release.
- **Reference-image swap mid-stream** — switching to a different anchor mid-clip means rebuilding PersonaLive's reference cache (~200 ms freeze, equivalent to the `process_input(...)` first-frame cost). For instant character switching, pre-bake all candidate reference caches and swap pointers; otherwise gate switches behind a "next stream" UI affordance.
- **Reincubate Camo on Linux** — officially Mac/Windows only as of last known check. Alternatives exist (NDI HX, RTSP, DroidCam) but each has its own latency/stability profile. See companion research doc.
- **OSC + Live Link Face** — iPhone normally streams LLF data to Unreal on a Mac. Linux receivers are not first-class; the LLF OSC schema is documented but our parser would be self-written. Smartphone apps like OSCAR can bridge if needed.
- **PersonaLive throughput ceiling** — 20–21 FPS bundled-TRT is the documented top (`_topics/personalive-acceleration.md`). If we want 60 fps display, that's a 3× hardware/software ask we don't currently have a plan for. 30 fps is achievable if we drop denoise steps or VAE quality (TAESD swap is the ladder's cheapest medium-effort lever).
- **Identity leakage in Architecture A** — the driver's face is encoded by FAN and could in theory affect the rendered identity. PersonaLive's design does its best to disentangle, but worst-case a strong asymmetric facial feature on the driver could bleed into the output. Phase 2 didn't show this on the Flux anchors, but we used identity-clean ARKit-take frames as drivers. Worth re-testing with a deliberately asymmetric driver face if this matters.

## Pointers

- Companion research doc: [`2026-05-04-iphone-to-linux-webcam-transport.md`](./2026-05-04-iphone-to-linux-webcam-transport.md) (in flight at write time) — concrete iPhone-to-Linux RGB transport options + recommendation.
- Strategic decision driving this work: [`2026-05-04-personalive-default-decision.md`](./2026-05-04-personalive-default-decision.md).
- Active integration plan: [`2026-05-03-iphone-pipeline-unified-plan.md`](./2026-05-03-iphone-pipeline-unified-plan.md). Phase 3 (bridge) is the load-bearing milestone for Architecture B.
- PersonaLive performance state: [`_topics/personalive-acceleration.md`](./_topics/personalive-acceleration.md).
- Resolution upgrade path: [`2026-05-04-video-super-resolution-survey.md`](./2026-05-04-video-super-resolution-survey.md). GFPGAN v1.4 first, SeedVR2-3B as ceiling.
