---
status: live
topic: neural-deformation-control
---

# NDI on Linux for iPhone-as-driver: falsified

**Date:** 2026-05-05
**Decision:** abandon NDI for Linux iPhone-as-driver pipelines. Pivot to RTMP/SRT.

## TL;DR

You can install libndi on Linux. You can receive *uncompressed* NDI streams.
You **cannot** receive iPhone NDI HX Camera (or any NDI HX source) because:

- iPhone NDI apps only transmit **NDI HX**, which is H.264 inside an NDI envelope.
- NDI SDK ships H.264 decoding **only on Windows and macOS** — official quote
  from the NDI docs: "If you require Linux support, please contact NDI SDK support."
- All Python bindings (cyndilib, ndi-python) and gst-plugin-ndi inherit this
  limitation. They can't paper over the missing decoder.

Result: discovery + connection succeed; `Receiver.receive(recv_video, ...)`
returns either no frames (cyndilib 0.0.2 + libndi 5.6.1) or segfaults
(cyndilib 0.1.1 + libndi 6.3.2). Both are the same root cause: nothing to
decode HX with.

## Evidence

- buresu/ndi-python #3 — open ask for "Support the NDI|HX (h264) driver on
  Linux". Canonical thread; remains unresolved.
  https://github.com/buresu/ndi-python/issues/3
- NDI docs, *H.264 Support* — explicitly states H.264 decoding works "without
  any installed plugins" on Windows and macOS, plus "HX decoding is not
  supported in the SDK".
  https://docs.ndi.video/all/developing-with-ndi/advanced-sdk/using-h.264-h.265-and-aac-codecs/h.264-support
- teltek/gst-plugin-ndi #86 — Linux GStreamer plugin users report broken
  HX/HX2 decode pipelines.
  https://github.com/teltek/gst-plugin-ndi/issues/86

Working "iPhone → Linux → Python frame" examples *anywhere on the public web*: zero.
Working examples on macOS/Windows: many (e.g. royshil's cyndilib + OpenCV gist).
Conclusion: the platform delta is the missing decoder, not bugs in our setup.

## Time spent confirming this

Significant. Bugs we hit and fixed correctly along the way (still useful for
future work):

1. avahi was binding to docker bridges on a host with 50 interfaces; fix is
   `allow-interfaces=enp4s0` in `/etc/avahi/avahi-daemon.conf`.
2. iOS Local Network permission must be on for NDI HX Camera, or it can't
   advertise via mDNS.
3. Auto-Lock backgrounds the app within seconds and stops the broadcast;
   set Auto-Lock to Never during testing.
4. Wired-vs-Wi-Fi can be on different L2 segments even at the same /24 if
   the router does AP isolation; joining Linux to the same Wi-Fi as iPhone
   bypasses this.

These fixes do NOT make NDI work — they only get us as far as discovery +
connection. The HX decoder gap is downstream of all of them.

## Pivot

**Larix Broadcaster (iOS) → SRT/RTMP → ffmpeg/PyAV → numpy.**

- Standard H.264 inside a standard transport. ffmpeg/libavcodec decodes natively.
- Same latency budget as NDI HX (~80–150 ms end-to-end on LAN).
- ~30 lines of Python with PyAV or `ffmpeg-python`.
- This is the path real ML/VTuber pipelines on Linux use (LivePortrait etc.).

Alternative pivots if Larix doesn't suit:

- **Moblin** (open-source iOS RTMP/SRT app) — same idea, FOSS.
- **iOS WHIP/WebRTC apps** → aiortc on Linux. More moving parts, lower latency.
- **iPhone over USB** as a UVC class device — works on macOS Continuity
  Camera; on Linux you need a third-party UVC bridge (DroidCam-style),
  which is a different ecosystem.

## What survives from this thread

- libndi 5 + cyndilib 0.0.2 install at `/usr/local/lib/libndi.so.5*`. Could be
  removed if we don't keep NDI as a future option; cheap to leave for now.
- avahi config fix (`/etc/avahi/avahi-daemon.conf`) — orthogonal, keep.
- `~/.NDI/ndi-config.v1.json` — orthogonal, keep.
- `scripts/ndi_discover.py`, `scripts/ndi_grab_one.py` — keep as artefacts of
  the falsification; useful if someone tries this again.

## Sources

- [buresu/ndi-python#3](https://github.com/buresu/ndi-python/issues/3)
- [NDI docs — H.264 Support](https://docs.ndi.video/all/developing-with-ndi/advanced-sdk/using-h.264-h.265-and-aac-codecs/h.264-support)
- [teltek/gst-plugin-ndi#86](https://github.com/teltek/gst-plugin-ndi/issues/86)
