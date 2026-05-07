---
status: live
topic: arkit-bridge
---

# Research: Live Link Face UDP wire-format verification

**Date:** 2026-05-07
**Subject:** Verify `src/arkit_bridge/llf_udp.py` against authoritative sources for the Live Link Face iPhone-app UDP packet format.

## Executive Summary

**Verdict: DISCREPANCY.** Our `decode_packet` is *functionally correct for the float payload* (because it slices 244 bytes off the **tail** of the datagram, which is robust to prefix size), but the **prefix layout we documented and the subject-name slice we compute are wrong**. The real prefix is 45 bytes (4-byte LE version + 37-byte UUID, where the UUID begins with `$`), not 6 bytes. The 4-byte big-endian name length lives at offset **41**, not **6**. As a result, our `subject` string is decoded from random bytes inside the UUID and almost always falls into the `"?"` fallback path. The float-vector ordering (52 ARKit blendshapes followed by 9 head/eye rotations in `head_yaw, head_pitch, head_roll, leftEye_yaw, leftEye_pitch, leftEye_roll, rightEye_yaw, rightEye_pitch, rightEye_roll`) and big-endian byte order both match three independent sources. Default port 11111 confirmed.

In short: the bridge that consumes `floats[:52] + floats[55:61]` is fine. The "subject name" the receiver displays in the probe and the assumed prefix-layout comment are wrong. Frame metadata is also subtly wrong: it's `(frame: u32, sub_frame: float32, fps: u32, denom: u32, data_length: u8)` for a total of 17 bytes, not `4×u32 = 16 bytes` with `(frame, subframe, denom, fps)` ordering.

## Per-question verdict with citations

### Wire format byte layout — DISCREPANCY

The actual layout, as decoded by `PyLiveLinkFace.decode()` and corroborated by the reverse-engineered `AppleARKitLiveLinkSource.cpp` description on the Epic forum, is:

```
[0:4]            uint32 LE              version
[4:41]           37 bytes UTF-8         UUID string (starts with "$")
[41:45]          int32 BE               subject-name length L
[45:45+L]        L bytes UTF-8          subject name
[45+L:49+L]      uint32 BE              frame number
[49+L:53+L]      float32 BE             sub_frame
[53+L:57+L]      uint32 BE              fps
[57+L:61+L]      uint32 BE              denominator
[61+L:62+L]      uint8                  data_length (must == 61)
[62+L:62+L+244]  61× float32 BE         blendshapes + head/eye rotations
```

Our doc claims `[0:6]` is the version/uuid prefix and the name length sits at `[6:10]`. That is wrong on both counts. The version is little-endian (uniquely so within the packet — every other multi-byte integer is big-endian), the UUID is 37 ASCII bytes starting with `$`, and the name length is at byte 41. (PyLiveLinkFace `decode` source [3]; UE forum reverse-engineered description [4].)

Practical consequence: in `llf_udp.py` line 22, `struct.unpack(">I", data[6:10])[0]` reads bytes from inside the UUID string. With probability near 1 the resulting `name_len` falls outside `0 < name_len < 64` (huge integer from ASCII bytes around offset 6 of a `$`-prefixed UUID), and we silently fall into `subject = "?"`. The float decoding still works because `data[-244:]` is unaffected by prefix mis-parsing.

### Float count and ordering — PASS

61 floats, big-endian, in this order (indices 0..60):

- 0..51: ARKit-52 blendshapes in the order matching our `scripts/livelink_probe.py:ARKIT_52` list (eyeBlinkLeft, eyeLookDownLeft, ..., tongueOut).
- 52: HeadYaw
- 53: HeadPitch
- 54: HeadRoll
- 55: LeftEyeYaw
- 56: LeftEyePitch
- 57: LeftEyeRoll
- 58: RightEyeYaw
- 59: RightEyePitch
- 60: RightEyeRoll

Confirmed by PyLiveLinkFace `FaceBlendShape` enum [3], the `aelzeiny/Animoji` deserialization project [5], and the videvago LiveLink UE MoCap docs which describe the "61 float values" array [1]. Our `b58 = floats[0:52] + floats[55:61]` slice is correct: it skips head ypr (indices 52..54) and keeps LE+RE ypr (55..60).

**Caution on Unity's `FaceBlendShape` enum [6]:** Unity's `Unity.LiveCapture.ARKitFaceCapture.FaceBlendShape` enum is **alphabetically ordered** (BrowDownLeft=0, ..., TongueOut=52) and is *not* the wire-format order. It is a Unity-side convenience enum, not the protocol. Don't cross-reference it for byte ordering; it will mislead. The wire order is the Apple `ARFaceAnchor.BlendShapeLocation` ordering as inherited by Epic's `AppleARKitLiveLinkSource.cpp`, which is what PyLiveLinkFace mirrors and what our `ARKIT_52` list matches.

### Endianness — PASS (with one nit)

All multi-byte integers and floats in the *payload* (name length, frame, sub_frame, fps, denom, all 61 floats) are big-endian / network byte order. Our `>` format strings are correct for everything we currently decode. The `version` field at `[0:4]` is *little-endian* per PyLiveLinkFace `decode` (`'<i'`) [3], but we don't read `version`, so this doesn't bite us — just worth flagging if we ever start.

### Default UDP port — PASS

11111. Confirmed by [1], [2], [7]. The Live Link Face iOS app exposes a "Port" field under **Live Link → Add Target** that defaults to 11111 and is user-configurable, so our `port: int = 11111` default is right and parameterizable as already coded.

### Frame metadata — DISCREPANCY

Our doc says four uint32: `(frame, subframe, denom, fps)`. The actual layout per PyLiveLinkFace `decode` is `'!if2ib'`, i.e. `(frame: u32 BE, sub_frame: float32 BE, fps: u32 BE, denominator: u32 BE, data_length: u8)` — 17 bytes total. Two issues:

- `sub_frame` is a **float**, not a uint32. Sub-frame is interpolation between integer frames (e.g. 0.5 means halfway between frame N and N+1). Decoding it as uint32 will produce a giant integer that is meaningless.
- The `(fps, denominator)` order is (fps, denom), not (denom, fps). The time-base is `fps / denominator` frames per second; Live Link Face emits 60/1 or similar.

We don't currently *use* the frame metadata fields anywhere in `B61Packet` or downstream — `recv_time = time.time()` is the wall-clock used everywhere — so this is documentation drift, not a functional bug. But if we ever start using the iPhone's frame timestamp for jitter analysis, the current comment will mislead.

### Coordinate convention for HeadYaw / HeadPitch / HeadRoll — GAP (already documented)

Authoritative sources do not state the unit, axis assignment, or sign convention of the Head{Yaw,Pitch,Roll} values that Live Link Face writes to the wire. PyLiveLinkFace exposes them as named getters/setters but does not document the convention. Epic's published Live Link Face documentation [2] and Apple's `ARFaceAnchor` documentation [10] both describe the underlying ARKit face frame (right-handed, +X to viewer's right = face's own left, +Y up, +Z toward viewer) but neither states explicitly that Live Link Face's emitted Euler triple is in radians or that yaw/pitch/roll map to rotation about Y/X/Z respectively in that frame.

This open question is the same one we already documented in `2026-05-06-arkit-vs-mediapipe-pose-conventions.md` and resolved empirically: the values are radians, the magnitudes match physical observation, and after empirical sign calibration via `closed_form_pose.py` the bridge produces correctly oriented renders. This research note does not change that finding; it just confirms the citation chain has not improved on the convention question.

## Specific fix list for `src/arkit_bridge/llf_udp.py`

Listed in priority order. **Recommendation only — do not apply in this commit.**

- **Fix the subject-name extraction.** Replace the hardcoded `data[6:10]` slice with a layout-aware parse: read 4 LE bytes for version (or skip), skip 37 UUID bytes, then read 4 BE bytes for `name_length` at offset 41, then `data[45:45+name_length]`. Validate `name_length < 64` and that the name decodes as UTF-8. As a defensive fallback, if the prefix-layout parse fails, retain the current "scan from the tail for 244 float bytes" behavior — that is the load-bearing part of the decoder and works regardless of prefix.
- **Update the module docstring.** The "[0:6] six-byte version/uuid prefix" line is wrong; replace with the correct `[0:4]` LE version + `[4:41]` 37-byte UUID + `[41:45]` BE name-length description.
- **Update the frame-metadata comment.** Document it as `(frame: u32, sub_frame: float32, fps: u32, denominator: u32, data_length: u8)` totaling 17 bytes, plus note that `data_length` must equal 61 (sanity check).
- **Optional: add a `data_length` sanity check.** If we walk the prefix forward instead of just slicing the tail, assert `data_length == 61` and reject the packet otherwise. Currently we silently accept any datagram of the right minimum size.
- **Optional: validate UUID starts with `$`.** A cheap sanity-test that the prefix is a Live Link Face packet and not unrelated UDP traffic on port 11111.
- **Mirror the same docstring fix in `scripts/livelink_probe.py`.** Its protocol comment says "uuid prefix (varies; ignore for probe)" without specifying length; bring it in line with the corrected `llf_udp.py` doc.

None of these affect the bridge's training or inference paths because the consumed slice is `floats[0:52]` and `floats[55:61]`, both of which are decoded correctly today.

## Open Questions

- **Is the LLF v1.6.0 build we use on iOS still on protocol version 6?** PyLiveLinkFace was written against UE5 LiveLink protocol v6 per the forum thread [4]. We have not verified what `version` integer our iPhone is emitting. Adding a `version` log on first packet receipt would close this.
- **Sub-frame semantics.** Sub-frame is a float in [0, 1] interpreting interpolation between frame indices, but Live Link Face's exact emitted values (and whether they're ever non-zero, given that the iPhone sends one packet per ARKit frame) is undocumented. Not blocking.
- **Eye-rotation sign convention.** Same caveat as head: the Apple/Epic docs describe `ARFaceAnchor.leftEyeTransform` and `rightEyeTransform` as 4×4 matrices in the face frame, but the Euler decomposition Live Link Face uses for the wire-protocol LeftEye{Yaw,Pitch,Roll} / RightEye{Yaw,Pitch,Roll} columns is not specified. Our `b58` consumer treats them as radians in the same frame as the head Eulers; this is an empirical choice, not a documented one.

## Sources

[1] videvago. "LiveLink UE MoCap." https://videvago.com/apps/livelink/ — describes the Live Link Face packet as "61 float values" dumped to UDP port 11111. (Retrieved 2026-05-07.)

[2] Epic Games. "Recording Facial Animation from an iOS Device — Unreal Engine documentation." https://docs.unrealengine.com/4.27/en-US/AnimatingObjects/SkeletalMeshAnimation/FacialRecordingiPhone — Live Link Face official user docs; confirms 11111 default port and the iOS-app target/port UI but does not document the wire format. (Retrieved 2026-05-07.)

[3] Jim West. "PyLiveLinkFace — `pylivelinkface.py`." https://github.com/JimWest/PyLiveLinkFace/blob/main/pylivelinkface/pylivelinkface.py — the load-bearing source for our verification. The `decode()` function explicitly parses `version` as `'<i'` at `[0:4]`, UUID at `[4:41]`, name_length as `'!i'` at `[41:45]`, frame metadata as `'!if2ib'` (frame u32, sub_frame f32, fps u32, denom u32, data_length u8), and 61 floats as `'!61f'`. The `FaceBlendShape` enum lists indices 0..60 in the wire-protocol order. (Retrieved 2026-05-07.)

[4] Epic Developer Community Forums. "Unreal Live Link (UDP packet format)." https://forums.unrealengine.com/t/unreal-live-link-udp-packet-format/472878 — independent reverse-engineering thread that cites `AppleARKitLiveLinkSource.cpp` and gives the same prefix layout (uint8 version, uint32 device-ID length, char device-ID, uint32 subject-name length, char subject name, frame metadata, uint8 blendshape count = 61, float[61]). Note: this thread describes UE5 protocol v6; PyLiveLinkFace's layout is the iOS-Live-Link-Face-app variant and matches what our iPhone emits. (Retrieved 2026-05-07.)

[5] aelzeiny. "Live-Link-Face-iPhone-App-Deserialization." https://github.com/aelzeiny/Live-Link-Face-iPhone-App-Deserialization — second independent reverse-engineering of the iOS app. README confirms reverse-engineering from `AppleARKitLiveLinkSource.cpp`. (Retrieved 2026-05-07.)

[6] Unity Technologies. "FaceBlendShape Enum — Unity Live Capture 4.0." https://docs.unity3d.com/Packages/com.unity.live-capture@4.0/api/Unity.LiveCapture.ARKitFaceCapture.FaceBlendShape.html — *Negative example.* Unity's `FaceBlendShape` enum is alphabetical (BrowDownLeft=0, ..., TongueOut=52, with `Invalid=22` inserted) and is **not** the wire-protocol order. Listed here as a documented hazard for future readers who go searching for "ARKit blendshape order" and find this enum first. (Retrieved 2026-05-07.)

[7] Jettelly. "Real-Time ARKit Facial Mocap in Godot." https://jettelly.com/blog/real-time-arkit-facial-mocap-in-godot — third-party tutorial confirming default UDP port 11111 and 52 ARKit blendshapes + head/eye rotations. (Retrieved 2026-05-07.)

[8] aelzeiny. "Animoji — Unreal Engine Livelink App + Apple ARKit Blendshapes." https://github.com/aelzeiny/Animoji — additional cross-reference for the 52-blendshape + 9-trailing-rotation structure. (Retrieved 2026-05-07.)

[9] suchipi. "arkit-face-blendshapes." https://github.com/suchipi/arkit-face-blendshapes — visual reference for the 52 ARKit blendshape names (no wire-format claim). (Retrieved 2026-05-07.)

[10] Apple. "ARFaceAnchor.BlendShapeLocation — Apple Developer Documentation." https://developer.apple.com/documentation/arkit/arfaceanchor/blendshapelocation — canonical 52-blendshape identifier list. Apple does not document a numerical ordering — the integer indices used on the wire are an Epic / Unreal convention layered on top of Apple's named enum. (Retrieved 2026-05-07.)

[11] Local source: `scripts/livelink_probe.py:ARKIT_52` and `:TAIL_NAMES` — match the wire-protocol indices verified above. (In-repo, dated 2026-05-04 import.)

[12] Local source: `src/arkit_bridge/llf_udp.py` — the implementation under review. (In-repo, dated 2026-05-05.)
