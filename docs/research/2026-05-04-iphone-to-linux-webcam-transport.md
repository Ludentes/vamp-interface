---
status: live
topic: neural-deformation-control
---

# iPhone front-camera → Linux RGB transport options (2026)

## Executive summary

For a realtime PersonaLive pipeline (~20 FPS, glass-to-glass ≤ 200 ms target) on Linux + RTX 5090 with motion sourced from an iPhone front camera, the **recommended first attempt is NDI HX over 5 GHz Wi-Fi (or USB-Lightning Ethernet adapter) using NewTek's official "NDI HX Camera" iOS app, received in OBS Studio via the DistroAV plugin (formerly obs-ndi)** — it is the only path that simultaneously offers (a) a maintained free iOS publisher, (b) a maintained Linux receiver shipping in OBS, (c) sub-frame latency claims with realistic ~100–200 ms reports, and (d) seamless v4l2loopback or direct OBS source [1][2][6][9][10]. **Fallback is RTSP/SRT publishing from Larix Broadcaster (free, iOS) into MediaMTX or go2rtc on Linux**, then `ffmpeg` → v4l2loopback; this is the most flexible plumbing but adds 1–2 transcode hops [3][4][12]. **Reincubate Camo has no Linux client and no published SDK in 2026** — the project ships macOS/Windows/Android only and there is no community Linux port [5].

## Comparison table

| # | Option | License / cost | Reported latency (Wi-Fi) | Max res/fps | Codec | Linux client | Transport | Notable quirks |
|---|--------|----------------|--------------------------|-------------|-------|--------------|-----------|----------------|
| 1 | NDI HX Camera (NewTek) + OBS DistroAV | Free iOS app, OBS+plugin OSS | ~80–200 ms; sub-frame possible on wired LAN [10][1] | 1080p30 (HX) | H.264/HEVC over NDI HX | Mature: DistroAV plugin in OBS, libNDI for Linux [2] | 5 GHz Wi-Fi or Lightning-Ethernet | Avahi/mDNS must work; flatpak needs override [2] |
| 2 | Larix Broadcaster + MediaMTX/go2rtc | Free iOS app; OSS server | ~150–400 ms (RTSP), <250 ms (SRT/WebRTC WHIP) [3][4][12] | 1080p60 | H.264/HEVC, RTSP/SRT/WebRTC/RTMP/NDI | Server runs on Linux; need ffmpeg → v4l2loopback | Wi-Fi or USB tether | SRT latency floor `ping×4`, min 120 ms [3] |
| 3 | Iriun Webcam | Free (paid Pro) | "Near-zero" USB; Wi-Fi laggy per users [13] | 1080p30 free, 4K paid | Proprietary H.264 over UDP | First-party Linux .deb client [13] | USB or Wi-Fi | Wi-Fi reports of stutter; PulseAudio loopback latency 200–300 ms reported [13] |
| 4 | DroidCam (dev47apps) | Free (Pro paid) | Sub-200 ms claimed | 720p (free) / 1080p (Pro) | Proprietary | Maintained Linux client + own v4l2loopback-dc [7] | USB (via usbmuxd) or Wi-Fi | iOS USB needs `usbmuxd` running; kernel ≥6.11 had issues [7] |
| 5 | OctoStream RTSP / "RTSP Stream" iOS app | Free | RTSP same-LAN ~200–400 ms | 1080p | H.264 | Any RTSP client (ffmpeg, mpv, OBS via Media Source) | Wi-Fi only | Standalone, no account; minimal control [11] |
| 6 | iFacialMocap / Live Link Face (ARKit blendshapes only) | Free / paid | <50 ms (blendshape stream, not RGB!) | n/a | OSC / proprietary | Several Linux-side parsers | Wi-Fi | **Not RGB** — only the 52-d blendshape vector. Not directly useful as PersonaLive RGB input but worth noting as parallel signal [14] |
| 7 | Reincubate Camo | Free tier + Pro $5/mo | macOS/Windows only | 1080p+ | Proprietary over USB/Wi-Fi | **None — no Linux client, no public SDK** [5] | n/a | Officially unsupported; iOS+Android publishers exist but no Linux receiver |

Excluded: EpocCam (Corsair discontinued, App Store removal in 2022; community wrapper `ohwgiles/epoccam_linux` unmaintained) [8].

## Per-option detail

### NDI HX Camera + OBS DistroAV (recommended first try)

- **iOS app**: "NDI HX Camera" by NewTek (App Store, free, requires iOS 14+) [1]. NewTek also ship a separate "NDI Camera: Easy Streaming" — both publish HX H.264 streams over mDNS-discovered NDI.
- **Linux receiver**: DistroAV plugin for OBS (formerly obs-ndi), shipping libNDI runtime [2]. Flatpak setup requires:
  ```bash
  flatpak install com.obsproject.Studio com.obsproject.Studio.Plugin.DistroAV
  sudo flatpak override com.obsproject.Studio --system-talk-name=org.freedesktop.Avahi
  ```
  For native install, package may be available via OBS's own repos; libNDI sometimes needs manual install (NewTek terms of service ship it as binary blob).
- **OBS → v4l2loopback**: OBS's built-in virtual camera output writes to v4l2loopback directly (no extra plugin since OBS 27); PersonaLive can read `/dev/video10` (or wherever OBS publishes) as RGB.
- **Latency numbers**: NDI itself can hit sub-frame on SpeedHQ over wired LAN [10]; HX (compressed H.264) typically 80–200 ms in real OBS+iPhone setups [1][10]. Wired Lightning-to-Ethernet adapter strongly recommended over Wi-Fi for jitter [10].
- **Quirks**: iPhone NDI apps drop the stream when backgrounded (iOS background-execution rules); sometimes invisible to Linux receivers if mDNS/Avahi is firewalled or running on a different VLAN [9].

### Larix Broadcaster + MediaMTX / go2rtc (fallback)

- **iOS app**: "Larix Broadcaster" by Softvelum (App Store, free) [3]. Supports SRT, RTMP, RTSP, RTSPS, NDI, WebRTC (WHIP), Zixi, RIST.
- **Linux server**: Either of two production-grade OSS gateways:
  - **MediaMTX** (`bluenviron/mediamtx`) — single-binary Go server: SRT/WebRTC/RTSP/RTMP/HLS publish + read [4].
  - **go2rtc** (`AlexxIT/go2rtc`) — universal converter, RTSP/RTMP/HTTP → WebRTC/HLS/MJPEG with very low latency [12].
- **Pipeline**:
  ```bash
  # On Linux: run MediaMTX
  ./mediamtx
  # iPhone Larix: configure SRT publish to srt://<linux-ip>:8890?streamid=publish:cam
  # Linux: ffmpeg from MediaMTX → v4l2loopback
  ffmpeg -i srt://localhost:8890?streamid=read:cam \
         -f v4l2 -pix_fmt yuv420p /dev/video10
  ```
- **Latency**: SRT formula is `ping × 4`, never below 120 ms [3]. WebRTC WHIP path goes lower (sub-100 ms in LAN). RTSP path typically 200–400 ms.
- **Quirks**: Adds an FFmpeg transcode hop unless v4l2loopback can take H.264 directly (it can't — needs raw frames). One CPU/GPU transcode adds ~10–30 ms.

### Iriun Webcam

- **iOS app**: "Iriun Webcam for PC and Mac" (App Store, free; Pro for >1080p, watermark-free) [13].
- **Linux client**: Official `.deb` from iriun.com — installs a v4l2loopback-backed virtual camera. Auto-discovers iPhone over LAN. USB also supported.
- **Latency**: Users report near-zero on USB [13]. Wi-Fi has stutter complaints; one Manjaro user logged a 200–300 ms PulseAudio loopback latency spike when Iriun launched [13].
- **Quirks**: Closed-source Linux client; sparse release cadence; not packaged in Debian/Ubuntu repos.

### DroidCam (dev47apps)

- **iOS app**: "DroidCam OBS" (App Store, free; Pro for HD) [7]. Originally Android-first; iOS port is functional but historically less polished than Android.
- **Linux client**: `dev47apps/droidcam-linux-client` on GitHub, actively maintained (2.1.x in 2025) [7]. Bundles `v4l2loopback-dc` (forked v4l2loopback module that registers `/dev/video*` as "DroidCam").
- **Transport**: USB (requires `usbmuxd` running for iOS) or Wi-Fi.
- **Quirks**: Kernel-module compat issues on kernel 6.11+ reported in 2025 [7]. iOS Pro features (HD, full sensor) gated behind one-time IAP.

### OctoStream / standalone RTSP server apps

- **iOS apps**: "OctoStream RTSP Server" / "RTSP Stream" (Bundle ID `id6474928937`) [11], "RTMP Broadcaster" (`id6747631361`).
- **Use**: Run app on phone, connect with `ffmpeg -i rtsp://<phone-ip>:8554/live -f v4l2 /dev/video10`.
- **Why not first choice**: Less control over codec params, more app-side instability than Larix; Larix subsumes this category.

### Reincubate Camo — Linux status

- **Officially**: macOS, Windows, Android publisher; no Linux client. Camo Pro 2.6.1 (March 2026) maintains the same platform list [5].
- **SDK**: Reincubate previously offered an iOS Camo SDK for partner integrations; **no public SDK is available in 2026** for third parties to wrap a Linux receiver.
- **Community port**: Searched — none exists. Their wire format is undocumented and the Mac client is signed/notarised; reverse-engineering would be substantial effort.
- **Implication**: cross off entirely for our use case.

### Continuity Camera

- macOS-only API surface; the iOS side uses Apple-private wireless framework. **No cross-platform variant**, no documented protocol. Not viable.

### USB UVC tethering / libimobiledevice / gphoto2

- **iOS does not expose a USB Video Class endpoint.** It exposes PTP for stills via libimobiledevice/gphoto2 but **no live video stream** [8].
- `gphoto2 --capture-movie` works for many DSLRs but not iPhone (the iPhone PTP profile lacks the `EOS_RemoteRelease` analogue and movie capture interface).
- Bottom line: forget USB-UVC — it does not exist on iOS in 2026.

## Lived practice (VTuber / streaming community on Linux)

The 2025 *Awesome-VTubing-on-Linux* index confirms the **dominant Linux + iPhone workflow uses iFacialMocap or Live Link Face for ARKit blendshapes** (not RGB), routed into VSeeFace-compatible receivers [14]. RGB-from-iPhone workflows are explicitly rare on Linux. When VTubers do need RGB:

- Reddit / OBS forum recurring recommendations rank **NDI HX Camera + DistroAV** highest for Linux because OBS support is built-in and wireless setup is one-tap [9][10].
- **DroidCam** is the most-mentioned "I just want a webcam" path because of its first-party Linux `.deb` and USB-via-usbmuxd reliability [7].
- **Larix + MediaMTX** appears in more sophisticated streaming/multistream setups where the operator already runs a media server.
- **Iriun** is the consumer-friendly "it just works" choice; lighter on configuration but closed-source.
- **No one in 2025/2026 reports successfully using Camo on Linux** — the consistent answer is "you can't, use NDI."

For the **PersonaLive pipeline specifically**, the preferred ordering is:
1. NDI HX → OBS → v4l2loopback (because OBS already needs to be in the loop for downstream RTMP/recording).
2. NDI HX → libNDI receiver → direct shm/v4l2 (skip OBS) if OBS adds unwanted latency at 5090 GPU saturation.
3. Larix → MediaMTX → ffmpeg → v4l2loopback (when Wi-Fi mDNS is unreliable, e.g., enterprise Wi-Fi).

## Open questions / could not verify

- Exact glass-to-glass latency of NDI HX Camera (iOS) → DistroAV (Linux) → v4l2loopback → PersonaLive on a 5 GHz Wi-Fi 6E link in 2026: no published benchmark found. Plan to measure with a phone-screen-mirror clock test once the box is set up.
- Whether OBS 31 (current as of 2026) virtual camera output reduces the v4l2loopback hop's CPU cost compared to manual ffmpeg piping.
- Whether NDI HX Camera supports HEVC (NDI HX2 spec) on iPhone 15 Plus — the App Store listing is silent; NDICam (Sienna) explicitly advertises HX2 [1].
- Whether `usbmuxd` over Lightning-to-USB-C on iPhone 15 (USB-C port) materially lowers latency vs. Wi-Fi for DroidCam / Larix RTSP — the 2025 reports are mostly from Lightning-era phones.
- Reincubate's roadmap on Linux: their public release notes through March 2026 show no signal of a Linux client; the question is whether a Pro SDK relicensing is plausible. Treat as no.

## Sources

[1] NDI HX Camera - Stream Capture, Apple App Store. https://apps.apple.com/us/app/ndi-hx-camera-stream-capture/id6502563620
[2] Ultimate Guide to DistroAV (NDI Plugin) Setup for OBS Studio, yostream.io. https://yostream.io/tutorials/obs-ndi-plugin-setup-distroav-multi-device-streaming/
[3] Larix Broadcaster, Softvelum docs/FAQ. https://softvelum.com/larix/ and https://softvelum.com/larix/faq/
[4] MediaMTX, bluenviron/mediamtx (GitHub). https://github.com/bluenviron/mediamtx
[5] Camo release notes (March 2026 build), Reincubate. https://uds.reincubate.com/release-notes/camo/ and https://reincubate.com/camo/downloads/
[6] How to Use IPhone Camera In OBS Studio, operations-lab.com. https://operations-lab.com/blog/use-iphone-camera-in-obs
[7] DroidCam Linux client, dev47apps/droidcam-linux-client (GitHub). https://github.com/dev47apps/droidcam-linux-client and https://www.dev47apps.com/droidcam/linux/
[8] Using the iPhone camera as a Linux webcam with v4l2loopback, Michael Stapelberg (2020 — primary reference for libimobiledevice limits, still accurate on UVC absence). https://michael.stapelberg.ch/posts/2020-06-06-iphone-camera-linux-v4l2loopback/
[9] OBS forum: NDI HX iPhone setup discussions. https://obsproject.com/forum/threads/cant-get-video-from-ndi-iphone-to-display.118757/ and https://obsproject.com/forum/threads/ndi-hx-iphone-app-not-showing-up-as-ndi-source-in-obs.185438/
[10] Evercast: How to diagnose and fix latency when using NDI. https://www.evercast.us/blog/ndi-latency
[11] OctoStream RTSP Server App, App Store. https://apps.apple.com/us/app/rtsp-stream/id6474928937
[12] go2rtc, AlexxIT/go2rtc (GitHub). https://github.com/AlexxIT/go2rtc
[13] Iriun Webcam reviews and Linux notes; Manjaro forum on PulseAudio loopback latency. https://iriun.net/ and https://forum.manjaro.org/t/pulseaudio-loopback-module-gets-unexpected-latency-upon-launching-iriun-webcam-or-droidcam/158632
[14] Awesome-VTubing-on-Linux. https://github.com/VTubing-on-Linux/Awesome-VTubing-on-Linux ; VSeeFace iFacialMocap docs. https://www.vseeface.icu/
[15] EpocCam Linux community wrapper (unmaintained reference). https://github.com/ohwgiles/epoccam_linux
