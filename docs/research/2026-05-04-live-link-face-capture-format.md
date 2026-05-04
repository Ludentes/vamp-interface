---
status: live
topic: neural-deformation-control
summary: Live Link Face (Epic, free iOS app) records ARKit-52 + RGB locally on TrueDepth iPhones; export is a take folder with a 1080p MOV and a frame_log.csv (timecoded per-frame blendshapes + head/eye rotation). Picked as Phase 2 driver-clip source — also yields Phase 3 bridge training data in the same capture.
---

# Live Link Face — capture format brief

**Date:** 2026-05-04
**Why:** picking the driver-clip source for Phase 2 of the iPhone-driven pipeline plan (`2026-05-03-iphone-pipeline-unified-plan.md`). Considered three options: Expo app, custom webui, public dataset, Live Link Face. Live Link Face wins because the same recording produces both Phase 2 driver clips AND Phase 3 bridge training data (ARKit-52 + RGB pairs), with no code.

## What it captures

- ARKit's 52 blendshape coefficients via TrueDepth front camera.
- Head pose (rotation/translation) and eye gaze (`eyeLook*` channels + per-eye rotation).
- Reference RGB video, simultaneous.
- Default 60 fps on TrueDepth devices. Newer non-TrueDepth modes go to 120 fps RGB-only.

## Recording vs streaming

Both modes; we use the local-record path.

- **Live Link mode**: streams over Wi-Fi to a Mac running Unreal. Not needed for us.
- **Local record**: red Record button stores the take on device. No Mac, no Unreal at capture time. Takes browser keeps them until exported.

## Export — what comes out of one take

A folder (zippable, AirDroppable) containing:

- `*.mov` — front-camera RGB, 1080p, timecode-striped, SMPTE timecode in track. ~30-60 MB/min.
- `frame_log.csv` — per-frame row keyed by timecode, columns are the 52 ARKit blendshape values + head rotation + eye rotation.
- Small metadata files (take/slate info).

MOV and CSV share SMPTE timecode → trivial alignment. This is the format Houdini, MotionBuilder, and the Blender LiveLinkFace add-on all consume directly.

## Workflow

1. Install **Live Link Face** from the App Store (free, Epic Games).
2. Open, grant camera + microphone permissions.
3. (Optional) Settings → Live Link → leave unpaired for offline-only record.
4. Frame face, tap red Record.
5. Stop, give the take a slate.
6. Takes browser → select → Share → AirDrop to a Mac, or use Files app / iTunes File Sharing to pull the folder.

## Compatibility

- Any TrueDepth iPhone (X, XS, XR, 11, 12, 13, 14, 15, 16-series).
- iOS 14+.
- MetaHuman Animator (depth video pipeline) requires iPhone 12+; plain Live Link works on older.

## Gotchas

- **CSV header bug** (known UE forum issue): in some versions the ARKit-mode CSV has wrong column headers. Verify the first take's columns against the documented ARKit blendshape order before trusting downstream parsing.
- **Dropped frames at 30 fps mode**: another reported bug. Record at 60 fps, downsample later if needed.
- **No iCloud auto-sync**: must AirDrop / Files / iTunes manually. Easy to forget on a fresh phone.
- **Storage fills fast**: 1080p MOV at 60 fps is ~50 MB/min. Plan device storage before a long capture.

## Alternatives considered

- **Face Cap (iOS)** — same ARKit data, FBX/BVH export, free tier limits streams to 5 s.
- **iFacialMocap / Facemotion3D** — direct FBX, Blender/MMD-friendly.
- **Rokoko Face Capture** — bundle with body, only worth it if already in Rokoko Studio.
- **MediaPipe FaceLandmarker (open source)** — RGB-only 52-blendshape regressor; pick if no iPhone available, but loses TrueDepth ground truth.
- **Expo app** — falsified for this use case; ARFaceAnchor blendshapes are native ARKit, no first-party Expo module, would require writing a Swift bridge anyway. Not worth the wrapper.
- **Custom webui (getUserMedia)** — would need MediaPipe on the side for pseudo-blendshapes; net-new code; loses Phase 3 ground truth.
- **Public dataset (VFHQ / CelebV-HQ)** — viable for Phase 2 sanity, but no canonical-expression direction (clips are conversational, not "smile peak hold 3s"), and no ARKit-52 paired data for Phase 3.

## Decision

**Live Link Face on a TrueDepth iPhone.**

- Phase 2 driver-clip source: 6 deliberate clips (smile peak, jaw-open peak, blink, brow up, head turn 30°, neutral baseline), ~5 s each, recorded in one session.
- Same takes feed Phase 3 bridge training (`b_arkit, m_personalive`) once the corpus expands. ~10 minutes of cooperative recording is enough to seed the bridge MLP per the unified plan.

## Sources

- [Live Link Face on App Store](https://apps.apple.com/us/app/live-link-face/id1495370836)
- [Epic — Recording face animation on iOS device in Unreal Engine](https://dev.epicgames.com/documentation/en-us/unreal-engine/recording-face-animation-on-ios-device-in-unreal-engine)
- [Epic blog — Live Link Face announcement](https://www.unrealengine.com/en-US/blog/new-live-link-face-ios-app-now-available-for-real-time-facial-capture-with-unreal-engine)
- [SideFX — load CSV from Live Link Face into Houdini](https://www.sidefx.com/tutorials/h19-facial-mocap-load-csv-from-live-link-face-app/)
- [UE community — Importing Live Link Face CSV data into Unreal](https://dev.epicgames.com/community/learning/tutorials/mJvq/importing-live-link-face-csv-data-file-into-unreal-engine)
- [alexdjulin/LiveLinkFace-CSV-Retarget-For-Motionbuilder (GitHub)](https://github.com/alexdjulin/LiveLinkFace-CSV-Retarget-For-Motionbuilder)
- [UE forum — iPhone model compatibility](https://forums.unrealengine.com/t/iphone-models-compatibility-with-live-link-face-and-arkit/268872)
- [UE forum — CSV header bug in ARKit mode](https://forums.unrealengine.com/t/inverted-mouth-jaw-blendshapes-in-live-link-face-livelink-arkit-mode/2518040)
