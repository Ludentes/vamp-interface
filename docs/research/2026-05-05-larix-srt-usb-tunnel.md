---
status: live
topic: neural-deformation-control
---

# Larix Broadcaster (iOS) → SRT → Linux receiver, tunneled over USB

**Date:** 2026-05-05
**Context:** Pivot research after `2026-05-05-ndi-linux-falsified.md` showed NDI HX H.264 decoder is not available in the NDI SDK on Linux, so iPhone NDI HX Camera cannot be received on Linux. Goal: re-evaluate whether a Larix-Broadcaster-over-USB pipeline is viable as a replacement for getting iPhone camera frames into a numpy array on Linux.

## TL;DR

Larix Broadcaster on iOS is a fully capable SRT publisher (caller / listener / rendezvous, libsrt 1.5.3, latency / passphrase / streamid / maxbw, H.264 + HEVC, AAC) [1][2]. ffmpeg/PyAV can listen for it on Linux trivially [5][6][7]. **The USB-cable tunnel idea via `iproxy` is almost certainly dead**: `iproxy` only forwards host→device (Linux opens a TCP listener that proxies to a port on the iPhone) [3][4], whereas we need device→host (the iPhone must dial a TCP socket served on Linux). No official reverse-iproxy exists; the closest GitHub feature request (libusbmuxd #99) was closed without a documented resolution [8]. **The viable replacement is iPhone Personal Hotspot over Lightning/USB-C**, which exposes a real Ethernet interface on Linux (`ipheth` kernel driver) and lets Larix dial Linux's IP directly with no tunneling needed [9][10]. That is the pipeline to build.

## Key Findings

### Larix Broadcaster iOS SRT capabilities

Larix supports SRT in Push (Caller), Listener and Rendezvous modes with multipoint listener support, built on libsrt 1.5.3, exposing the full SRT param set: `mode`, `latency`, `maxbw`, `pbkeylen`, `passphrase`, `streamid` [1]. Video is H.264/AVC universally with H.265/HEVC on supported devices; audio is AAC (Opus only on the WebRTC path) [2]. Bitrate is per-resolution from a built-in table (1080p → 3000 kbps, 720p → 2000 kbps, 540p → 1500 kbps) with adaptive-bitrate modes (ascend/descend/hybrid) [2]. Background / lock-screen behaviour is **audio-only**: the only documented background mode is "audio-only capture mode: disable preview, stream from background, no video in output" [2] — i.e., backgrounding kills the video stream. Core SRT/RTMP/RTSP streaming is free; HEVC, NDI HX2, multi-destination (3+), Zixi, WebRTC-WHIP, and SEI metadata are paywalled at $19.99/mo IAP or $10/mo via Larix Tuner [2].

### USB TCP tunneling on Linux for iOS

The libimobiledevice stack (`usbmuxd` daemon + `libusbmuxd` client + `iproxy` CLI) is the canonical way to tunnel TCP over the Lightning cable on Linux [3][4]. Install: `sudo apt-get install usbmuxd libimobiledevice6 libimobiledevice-utils` [3]. Pairing the device once (tap "Trust This Computer" on the unlocked iPhone) is required; for app-level services (not lockdownd), Developer Mode + a paired/unlocked phone may also be required, but plain TCP forwarding to a userland app on the iPhone is not a documented use case at all (see Q4). The man page is unambiguous about direction: `iproxy LOCAL_TCP_PORT DEVICE_TCP_PORT` opens `LOCAL_TCP_PORT` **on the host** and proxies traffic to `DEVICE_TCP_PORT` **on the iPhone** [4]. There are no published bandwidth or latency numbers from libimobiledevice.org for `iproxy`; community reports for SSH-over-USB are "fast enough for a shell," nothing useful for video.

### SRT listener on Linux (ffmpeg / PyAV)

ffmpeg listener URL: `srt://0.0.0.0:9000?mode=listener&latency=80000` (latency is in **microseconds** in ffmpeg, default 120 000 µs = 120 ms) [5][6]. PyAV ≥ 14.1.0 ships with libsrt linked, so `av.open("srt://0.0.0.0:9000?mode=listener", ...)` works directly [7]. For LAN / USB-Ethernet RTT (≤ 1 ms) the SRT spec floors RTT at 20 ms internally and recommends latency = 3–4 × RTT, with the absolute minimum SRT latency being 20 ms [11]. A 60–80 ms latency setting is the realistic floor for a glass-to-numpy LAN scenario.

### Q4 — CRITICAL: Does Larix push to its own loopback work over an iproxy tunnel?

**No documented evidence this works, and one piece of strong indirect evidence it doesn't.** The proposed setup is: iPhone runs `iproxy` *in reverse*, i.e., a TCP listener on the phone at `127.0.0.1:9000` whose connections forward over USB to Linux's `127.0.0.1:9000`, then Larix pushes SRT to `srt://127.0.0.1:9000`. Two independent problems falsify this:

1. **`iproxy` is host→device only.** The man page synopsis is `iproxy LOCAL_TCP_PORT DEVICE_TCP_PORT` and forwards "localhost ports to the device" [4]. A reverse-direction request (libusbmuxd issue #99, "Use iproxy in reverse?") was filed in 2020 specifically to forward iPhone-side ports to the host; the issue is closed but the public conversation contains no resolution, comments or workaround in the rendered page [8]. There is no shipping reverse-iproxy in libimobiledevice. (Even if there were, you'd still need a userland TCP listener running on the iPhone that the in-app SRT client could dial; iOS sandboxing and the lack of a stock TCP-listener app make this a non-starter outside Xcode-sideloaded research code.)
2. **Larix won't dial loopback addresses for SRT listener targets.** Softvelum's own guidance is "Do not use localhost or 127.0.0.1 as a listener address. You must specify your real local IP" [12]. This is for the SRT-listener-on-LAN case, not the loopback-on-device case directly, but it indicates the app's SRT URL handling is intentionally LAN-IP-oriented.

Bottom line: **the USB-tunnel-via-iproxy idea is dead**. We have not found anyone publishing this pipeline working, and the building blocks are wrong-direction. Single-source forum claim level: zero — no forum reports of success **or** failure, because nobody tries it.

### Q5 — Fallback: how do you actually get SRT/RTMP from iOS over USB?

The right primitive is **iPhone Personal Hotspot over the USB cable**, which is a different mechanism from `usbmuxd` entirely. With the cable plugged in and Personal Hotspot toggled on, iOS exposes itself as a CDC-Ethernet device; on Linux the in-tree `ipheth` driver claims it and `NetworkManager` brings up a wired interface (typically 172.20.10.x/28) [9][10]. That interface is bidirectional IP — Larix sees the Linux box on the same /28 subnet and pushes SRT directly to the host's interface IP (`srt://172.20.10.1:9000?mode=caller`). No tunneling required, no developer-mode app, no jailbreak. Required packages on Ubuntu: `ipheth-utils`, `usbmuxd`, `libimobiledevice6` [9]. **Caveat:** community reports since iOS 14 describe intermittent USB-tethering breakage on Linux with the workaround being Wi-Fi hotspot instead [10]. Worth a 30-minute smoke test before committing.

Other fallbacks ranked by viability:

| Fallback | Mechanism | Viability |
|---|---|---|
| Personal Hotspot over USB-Ethernet (`ipheth`) | iOS exposes CDC-Ethernet on cable; routed IP, no tunnel | **Recommended** [9][10] |
| Wi-Fi LAN, same router as Linux | Trivial; just point Larix at Linux IP | Works; not USB-tethered |
| iPhone Wi-Fi hotspot, Linux as client | Linux joins iPhone's Wi-Fi AP, then Larix dials Linux IP | Works; uses radio not cable |
| Reverse `iproxy` via custom code (e.g. `pymobiledevice3`) + sideloaded SRT-listener app on iPhone | Build and ship our own iOS app to bridge | High effort, not justified |
| Larix `inetcat` / `tcprelay.py` | Mentioned only in old SSH-over-USB context | Not applicable to userland iOS apps |

### Q6 — Latency budget (Personal-Hotspot-over-USB path)

No measured numbers exist in any official source for "Larix → SRT-over-USB-ethernet → ffmpeg → numpy" specifically. A defensible budget from primitives:

- Larix capture + H.264 encode on iPhone: ~16–33 ms (1 frame at 30–60 fps, ARKit/AVFoundation pipelines typically add ≤ 1 frame).
- SRT latency setting: 60–80 ms is the practical floor over a low-RTT LAN per Haivision [11]; SRT will not honour anything lower than 20 ms internally.
- Ethernet-over-USB transit: sub-millisecond at ~100 Mbps headline throughput; not load-bearing.
- ffmpeg/libavformat decode + PyAV → numpy: ~5–15 ms per 1080p frame on a modern CPU/GPU.

Realistic glass-to-numpy: **~100–150 ms** end-to-end, dominated by the SRT receive buffer. This is **substantially worse than the NDI numbers we were targeting** (NDI HX claims sub-frame on LAN) but acceptable for the offline animation-driver use case and not catastrophic for live preview if we accept ~5–9 frames of lag. Must be measured, not trusted to spec sheets.

## Open Questions

- Does iOS's `ipheth` USB-Ethernet path actually work reliably on Ubuntu 24.04 / current iOS? The reports of post-iOS-14 breakage [10] are 4+ years old and may or may not still apply. Smoke test before committing.
- What is Larix's behaviour when the iPhone is screen-locked but Personal Hotspot is active and the app is foregrounded with screen-on? The audio-only-in-background documentation [2] only addresses true backgrounding; capture-with-screen-on but locked is undocumented.
- Is there a documented way to keep Larix in the foreground indefinitely under Guided Access for an unattended capture rig? Not researched here.
- What is the actual measured latency of the proposed pipeline? Numbers above are budgeted, not measured. Build a smoke rig (iPhone clock-flash → SRT → ffmpeg timestamp) before any architecture decisions depend on the number.
- Does `pymobiledevice3` or similar offer a real reverse-tunnel that we'd actually want, in case Personal Hotspot path falls over? Not investigated.

## Sources

[1] Softvelum. "SRT support in Softvelum products." https://softvelum.com/srt/ (Retrieved 2026-05-05)
[2] Softvelum. "Larix Broadcaster mobile streaming app for iOS." https://softvelum.com/larix/ios/ (Retrieved 2026-05-05)
[3] libimobiledevice. "libusbmuxd README." https://github.com/libimobiledevice/libusbmuxd (Retrieved 2026-05-05)
[4] Ubuntu Manpages. "iproxy(1) — proxy that enables tcp service access to iPhone/iPod." https://manpages.ubuntu.com/manpages/trusty/man1/iproxy.1.html (Retrieved 2026-05-05)
[5] FFmpeg project. "FFmpeg Protocols Documentation — srt." https://ffmpeg.org/ffmpeg-protocols.html (Retrieved 2026-05-05)
[6] Haivision SRT project. "FFmpeg — SRT CookBook." https://srtlab.github.io/srt-cookbook/apps/ffmpeg.html (Retrieved 2026-05-05)
[7] PyAV. "Add support for the srt protocol." Discussion #1503. https://github.com/PyAV-Org/PyAV/discussions/1503 (Retrieved 2026-05-05; libsrt shipped in PyAV 14.1.0)
[8] libimobiledevice. "Use iproxy in reverse? #99." https://github.com/libimobiledevice/libusbmuxd/issues/99 (Retrieved 2026-05-05; closed, no public resolution)
[9] Ubuntu Manpages. "ipheth — USB Apple iPhone/iPad tethered Ethernet driver." https://manpages.ubuntu.com/manpages/bionic/man4/if_ipheth.4freebsd.html (Retrieved 2026-05-05)
[10] Arch Linux Wiki. "iPhone tethering." https://wiki.archlinux.org/title/IPhone_tethering (Retrieved 2026-05-05)
[11] Haivision. "SRT Latency / Round Trip Time / RTT Multiplier." https://doc.haivision.com/SRT/1.5.4/Haivision/latency, https://doc.haivision.com/SRT/1.5.3/Haivision/round-trip-time (Retrieved 2026-05-05)
[12] Softvelum. "FAQ for Larix Broadcaster, Larix Screencaster and Larix Player" + OBS Wiki re: SRT listener addresses. https://softvelum.com/larix/faq/ ; https://obsproject.com/wiki/Streaming-With-SRT-or-RIST-Protocols (Retrieved 2026-05-05)
