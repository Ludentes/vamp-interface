---
status: live
topic: arkit-bridge
---

# LLF → PersonaLive → OBS streaming: pacing and sampling lessons

End-to-end live streaming is shipped and watchable. This doc records what
we learned during the first hands-on session, the bugs we hit, the fixes
we landed, and the residual issues we knowingly left for next round.

## What works

iPhone (Live Link Face) → USB-tether (Personal Hotspot Ethernet) →
UDP:11111 → `LLFReceiver` → evenly-spaced sampler →
`Pose2VideoPipeline_Stream` V2 (cohort-stream prepare/step) →
`PacedSinkWriter` (paced 8 FPS drain, prebuffered) → ffmpeg → v4l2loopback
→ OBS.

Steady-state numbers, RTX 5090, fp16, 4-step DDIM, batch=24:

| metric | value |
|---|---|
| sink write rate | 8.0 fps (= drain rate) |
| fresh-frame rate | 8.0 fps (no last-frame repeats) |
| infer time / batch | 2.95–3.05 s |
| prep / cohort | ~810 ms / ~370 ms (×6 cohorts) |
| ring drops at 60→25 sample stride | expected (rx_dropped grows steadily) |
| cursor_skips after seed | 0 (post-fix) |
| glass-to-OBS latency | ~5 s (3 s prebuffer + 2 s pipe) |

Operational guidance for real-time use is in
`docs/research/2026-05-06-llf-obs-runbook.md`.

## Eight things we learned

### Single-buffered v4l2loopback bursts get clobbered

The naïve loop `for f in rgb: sink.write(f)` writes a 24-frame batch to
v4l2 in milliseconds. v4l2loopback is single-buffered, so OBS only ever
sees the *last* frame of each burst — the previous 23 are clobbered before
its capture loop reads them. From the user's seat: one frame every 3 s.

Fix: a `PacedSinkWriter` background thread owns the sink, drains a
thread-safe deque at a fixed FPS, and repeats the last good frame when
the queue is empty. Render thread `push()`es and never blocks on sink
I/O. Burst pattern eliminated structurally.

### Bisecting with synthetic feeders is cheap and decisive

Two diagnostic scripts each took <30 minutes to write and saved hours
of guessing:

- `scripts/v4l2_test_feeder.py` — moving red dot, no PersonaLive, no LLF.
  Confirmed v4l2 ↔ OBS path is healthy at any FPS.
- `scripts/v4l2_burst_feeder.py` — pre-renders 24 frames and pushes them
  through `PacedSinkWriter` every 3 s. Confirmed the writer architecture
  produces smooth output at 8 fps unique frames.
- `scripts/v4l2_llf_feeder.py` — receives Live Link UDP, draws raw
  blendshape bars + head-pose-in-degrees as an overlay, with a heartbeat
  dot. End-to-end validation of the LLF receive path independent of
  PersonaLive. Doubles as a "is the iPhone sending sensible data right
  now" sanity tool.

Whenever something looks wrong in the full pipeline, prefer "remove the
neural network" or "remove the network transport" over instrumenting the
full stack.

### Wi-Fi UDP jitters; iPhone USB-tether (Personal Hotspot) is wired UDP

LLF transmits at ~60 Hz. Over Wi-Fi we saw ~3 % visible packet
loss/jitter (occasional yellow `age` indicators in the diagnostic
overlay). Enabling Personal Hotspot and plugging USB exposes a wired
Ethernet interface (`enxXXXXXXXXXXXX`, addr typically `172.20.10.2/28`).
Setting LLF target to that IP gives zero loss, sub-50 ms `age`, all
green. Cellular plan not required — the toggle creates the USB-Ethernet
device on attach.

### Objective sink-side metrics > render-side fps

The renderer's `fps=8.0` log line tells you *the rate the renderer
produces frames*, not what OBS sees. In V2 mode (4-frame cohorts every
~490 ms) at 25 fps drain the renderer happily reported `fps=8` while OBS
saw a stuttering pattern of `fresh_fps=8 / repeat_fps=17`: each 4-frame
cohort drained in 160 ms then the writer repeated the last frame for
~330 ms.

We added `PacedSinkWriter.snapshot()` exposing:

- `write_fps` — total writes/sec to ffmpeg
- `fresh_fps` — fraction that were unique frames
- `repeat_fps` — fraction that were last-frame holds (= writer
  starvation)
- `q_mean / q_max` — queue depth distribution
- `dropped` — frames evicted from queue overflow (= writer can't keep up)

`fresh_fps == drain_fps` and `repeat_fps == 0` is the smoothness
condition. Anything else is operator-visible jerk.

### Drain rate must match production rate, not "the FPS we want OBS to see"

If render produces 8 unique fps and the sink drains at 25 fps, OBS sees
8 fresh fps in a noisy pattern of bursts and freezes. There is no
visual gain from 25 fps drain when the underlying render rate is 8 —
you just turn the freezes into more obvious stutter.

Setting `--fps 8` so drain matches production gives evenly-spaced
fresh frames every 125 ms. The frame rate floor is still 8 fps (it's
the model's rate) — but it *feels like 8 fps* instead of 8 fps with
glitches. Genuine 25 fps would require frame interpolation
(linear, RIFE, or IFNet) — out of scope for this round.

### Prebuffer covers render variance at batch boundaries

Even with drain=8 fps and production≈8 fps the daemon stutters because
**each batch starts with `prep=820 ms`** — the prepare pass re-encodes
identity and runs 3 warmup cohorts. During those 820 ms the queue
drains by ~6.5 frames; if there are fewer than 6 in the queue it
starves and the writer repeats.

`PacedSinkWriter(prebuffer=N)` waits for the queue to fill to N frames
on first start before draining. With `prebuffer=24` we hold ~3 s of
headroom — more than the worst observed prep gap. Cost is one-time
glass-to-OBS latency on startup.

The prebuffer fires only on the FIRST fill. Subsequent queue empties
fall back to last-frame-repeat (graceful degradation rather than
re-pause, which would look worse).

### "Most recent N packets" was 7.5× slow-motion in disguise

`LLFReceiver.pop_window(k=24)` pulled the latest 24 packets after each
render. iPhone sends at 60 Hz; render takes ~3 s; so the latest 24 came
from the last **0.4 s** of real time and got played back over 3 wall
seconds at 8 fps. Result: every batch shows 0.4 s of motion stretched
to 3 s ≈ **7.5× slow-mo**, with the previous 2.6 s of motion silently
discarded. The **face appeared to barely move** even when we were doing
big expressions on the iPhone.

Fix: `pop_window_evenly_spaced(k, span_s)` picks k packets whose
`recv_time`s are tiled across `[newest − span_s, newest]`. With
`span_s=3.0` matching the render time, output played at 8 fps over
3 wall seconds shows 3 s of real motion — real-time speed at low
frame rate.

### The cursor-tracking design caused catastrophic mid-stream resets

First version of the even-spaced sampler kept an internal `cursor`
that advanced `+= span_s` per call so successive batches tiled real
time contiguously. A `max_lag_s = 2 × span_s` fail-safe jumped the
cursor forward when render fell too far behind real time.

In practice render time has small variance, the cursor accumulated
slip, and **once per minute or so the fail-safe fired and the daemon
skipped 6 s of motion in a single jump**. Visible to the user as a
"reset" — the face suddenly relocated mid-conversation.

Replaced the design with a stateless rule: every call samples
`[newest − span_s, newest]` directly. No cursor, no skip. Render-time
variance shows up as small overlap at seams (fast batch repeats a
few ms of motion) or small gap at seams (slow batch drops a few ms
of motion) — both invisible to a human viewer. `cursor_skips` is now
just a "had to wait for ring to refill" alarm, no longer affects
playback continuity.

## Known residual issues we did not fix this round

### PersonaLive's temporal memory resets at every batch boundary

Each `prepare_v2()` call re-seeds latent noise (same `seed=42`),
re-encodes the identity reference, and rebuilds the temporal context
from scratch. There is **zero cross-batch state**. With `batch=24`
and current render rate, this is a boundary every ~3 s.

Visible as: at each boundary the model momentarily pulls toward the
neutral reference identity, then reapplies expression over the first
~4-8 frames of the new batch. In V1 single-pass mode (whole video file
in one call) this is invisible because there is exactly one prepare
for the entire clip.

Fixes deferred:

- `--batch 48` doubles the time between boundaries; latency doubles.
- 4-frame overlap-blend at seams: render with a small overlap into the
  previous batch's tail, alpha-blend in pixel space.
- True warm-start: stash final cohort's `z_t` + `_stream_*` lists,
  pass as `prepare()`'s initial state. Architecturally correct,
  requires touching the vendored pipeline interface.

### Head and face decoupled in some moments

User flagged "head positioned correctly but face on it turned" which
went away after the cursor fix. Plausibly was downstream of the
sampling resets. Watch list for next session — if it returns the
suspects are:

- `closed_form_pose` Euler-sign mismatch under streaming (vs offline
  takes the bridge was calibrated on)
- b58 vs ypr coordinate frame mismatch under heavy motion
- Identity drift under high-amplitude conditioning

Diagnostic plan if it recurs: `v4l2_llf_feeder.py` displays raw ypr
in degrees — compare what the iPhone reports against what the rendered
head does. If the *numbers* are sensible the issue is downstream of
LLF; if the numbers themselves are wrong it's wire-format or
convention.

### Buffer balance is approximate

Production rate (24 / render_time) is close to but not exactly equal
to drain rate (8 fps). With `render_time ≈ 2.95 s` we observed the
queue **growing** at ~0.4 frames/batch (q_mean drifted up to 41).
With `render_time ≈ 3.12 s` the queue shrank at the same rate. Static
`--fps` choice can't perfectly match variable render time.

Trivial mitigations:

- `--sample_span_s` slightly larger than median render time (e.g. 3.05)
  to balance.
- Tiny PI-style controller in the writer that nudges `dt` up or down
  to keep `q_mean` near target.

Not urgent — the prebuffer absorbs hours of slow drift in either
direction before it would matter operationally.

## Practical settings that work today

```bash
PYTHONPATH=src /home/newub/w/PersonaLive/.venv/bin/python \
  scripts/streaming_bridge.py \
  --reference data/llf-phase2/asian_m__06_neutral.midframe.png \
  --ckpt runs/student_v2_120k/student_best.pt \
  --device_path /dev/video10 \
  --port 11111 \
  --batch 24 --fps 8 --mode v2 \
  --sample_span_s 3.0 \
  --writer_prebuffer 24
```

Set iPhone Live Link Face target = the USB-Ethernet IP shown by
`ip -brief addr | grep -E 'enx|usb'`, port 11111. OBS source =
"PersonaLive" video-capture device.

Glass-to-OBS latency ≈ 5 s. Output is real-time-paced at 8 fps. No
bursts, no freezes, no mid-stream resets.

## Companion code changes

| File | Change |
|---|---|
| `scripts/streaming_bridge.py` | `PacedSinkWriter` (paced background drain, snapshot metrics, prebuffer); CLI `--writer_prebuffer`, `--sample_span_s` |
| `src/arkit_bridge/llf_udp.py` | `peek_latest()` for non-destructive read; `pop_window_evenly_spaced(k, span_s)` stateless even-stride sampler; `cursor_skips` counter (waited-for-ring) |
| `scripts/v4l2_test_feeder.py` (new) | smooth-feed bisection tool |
| `scripts/v4l2_burst_feeder.py` (new) | burst-pattern bisection tool, exercises `PacedSinkWriter` directly |
| `scripts/v4l2_llf_feeder.py` (new) | end-to-end LLF receive validation with on-screen overlay |

## Companion docs

- `docs/research/2026-05-06-llf-obs-runbook.md` — operator guide
- `docs/research/2026-05-07-llf-udp-protocol-verification.md` — wire format
- `docs/research/_topics/arkit-bridge.md` — topic index pointer
