---
status: live
topic: neural-deformation-control
---

# v4l2loopback config for OBS + Python writer workflows

**Date:** 2026-05-04
**Why this exists:** During Architecture A implementation we hit a reproducible
bug where `pyvirtualcam` succeeded once on `/dev/video10` and then failed on every
subsequent run with `RuntimeError: 'v4l2loopback' backend: Device /dev/video10 is
not a video output device`. The root cause is `exclusive_caps=1` — a flag we
copied uncritically from common OBS guides. This doc records the correct config
and the reasoning.

## TL;DR — recommended config

For a workflow where **a Python script (or any native v4l2 writer) opens
`/dev/videoN` repeatedly** and OBS Studio consumes the same device:

`/etc/modprobe.d/v4l2loopback.conf`:
```
options v4l2loopback video_nr=10 card_label="PersonaLive"
```

`/etc/modules-load.d/v4l2loopback.conf`:
```
v4l2loopback
```

`/etc/udev/rules.d/60-v4l2loopback.rules` (so any user can write the device
without being in the `video` group):
```
KERNEL=="video10", MODE="0666"
```

**Do not set `exclusive_caps=1`** unless the only consumer is Chromium-based
WebRTC (Chrome, Zoom desktop, Jitsi, Slack-in-browser). For OBS, ffplay, MPV,
Firefox, and native v4l2 readers it is actively harmful.

## Why `exclusive_caps=1` breaks repeat writers

Upstream README description [1]:

> "[exclusive_caps] mode that only reports CAPTURE/OUTPUT capabilities
> exclusively. The newly created device will announce OUTPUT capabilities only
> (so ordinary webcam applications (including Chrome) won't see it). As soon as
> you have attached a producer to the device, it will start announcing CAPTURE
> capabilities only…"

So the device dynamically flips caps based on producer state. The bug is in the
disconnect path: when the producer closes, the device enters a state with
"neither OUTPUT nor CAPTURE capabilities", and the next producer's `open()`
fails with "not a video output device" until the kernel module is reloaded
[2].

Observed `Device Caps` after a successful pyvirtualcam run, with
`exclusive_caps=1`:

```
$ v4l2-ctl -d /dev/video10 --info | grep Caps
Capabilities     : 0x85200000   # 0x02 (V4L2_CAP_VIDEO_OUTPUT) absent
Device Caps      : 0x05200000
```

Reloading the module restored `Device Caps : 0x05200002` (output bit set), and
pyvirtualcam worked once — then flipped back to `0x05200000` after disconnect.
Cycle repeats indefinitely.

Without `exclusive_caps=1`, the device advertises **both** capture and output
simultaneously and forever; producers and consumers can connect/disconnect
freely. The known limitation is that some Chromium-derived applications refuse
dual-cap v4l2 devices ("ordinary webcam applications won't see it" without
exclusive_caps) — but OBS, Firefox, ffplay, MPV, and `cv2.VideoCapture` all
handle dual-cap devices fine.

## When you actually need `exclusive_caps=1`

| Consumer | Needs `exclusive_caps=1`? |
|---|---|
| OBS Studio (V4L2 source or Virtual Camera) | No |
| ffplay / MPV | No |
| Firefox getUserMedia | No |
| Native v4l2 (`cv2.VideoCapture`, gstreamer v4l2src) | No |
| Chrome / Chromium WebRTC | **Yes** |
| Zoom desktop, Jitsi-meet-desktop | **Yes** |
| Slack-in-browser (Chromium) | **Yes** |

If you need both — i.e. the device must serve OBS *and* Chrome — the practical
options are:
1. Use `exclusive_caps=1` and accept that you reload the kernel module
   (`sudo rmmod v4l2loopback && sudo modprobe v4l2loopback`) every time the
   producer reconnects. Tedious for development.
2. Create two v4l2loopback devices: `video_nr=10,11`, one with `exclusive_caps`
   and one without. Mirror frames into both from the writer.
3. Drop the Chromium consumer requirement.

For our PersonaLive Architecture A pipeline, OBS is the only consumer, so
option-zero (drop `exclusive_caps`) is correct.

## Module-options gotchas

- **`devices=N`** — only useful when creating multiple devices. We need one,
  so omit it.
- **`max_buffers`** — defaults to 2. Some guides bump to 8 for high-fps
  streams. Not relevant to our 20 FPS pipeline.
- **`card_label`** — cosmetic, but OBS shows the label in its source picker.
  Set it.
- **`video_nr`** — pin the device index so persisted OBS scenes keep working
  across reboots. We use 10 to stay clear of physical webcams (typically 0–4).

## udev permission rule rationale

The DKMS package creates `/dev/videoN` as `root:video 0660`. Adding the
current user to the `video` group works but logs you out and back in to take
effect. The udev rule sets `0666` on `/dev/video10` specifically (not all
v4l2 devices), keeping the change scoped and surviving the next module reload.

`KERNEL=="video10", MODE="0666"` matches only our loopback node. Physical
webcams remain at the original ACL.

## Verification commands

```bash
# After modprobe / reboot:
ls -la /dev/video10                     # expect: crw-rw-rw- root video
v4l2-ctl -d /dev/video10 --info | grep -iE "card|caps"
                                        # expect: Card type: PersonaLive
                                        #         Device Caps: 0x05200002

# Smoke test the writer + reader simultaneously:
ffmpeg -f lavfi -i testsrc=size=512x512:rate=20 -f v4l2 -pix_fmt yuv420p /dev/video10 &
ffplay /dev/video10                     # expect: SMPTE colour bars
fg; kill %1                             # cleanup
```

If the smoke succeeds twice in a row without reloading the module, the config
is right.

## Sources

[1] umlaeute/v4l2loopback — README. https://github.com/umlaeute/v4l2loopback
[2] v4l2loopback issue #442 — "exclusive_caps limits loopback device to a
    single producer open". https://github.com/v4l2loopback/v4l2loopback/issues/442
[3] letmaik/pyvirtualcam issue #61 — "No v4l2 loopback device found after
    using pyvirtualcam once with exclusive_caps=1".
    https://github.com/letmaik/pyvirtualcam/issues/61
[4] Arch Wiki — V4l2loopback. https://wiki.archlinux.org/title/V4l2loopback
    (Anubis-protected at fetch time; cited from search excerpt.)

Sources:
- [umlaeute/v4l2loopback](https://github.com/umlaeute/v4l2loopback)
- [v4l2loopback issue #442](https://github.com/v4l2loopback/v4l2loopback/issues/442)
- [pyvirtualcam issue #61](https://github.com/letmaik/pyvirtualcam/issues/61)
