---
status: live
topic: neural-deformation-control
---

# Research: MetaHuman Animator depth take format and pipeline

**Date:** 2026-05-05
**Sources:** 14 sources. Key: Epic Live Link Face docs, Epic forum thread on `depth_data.bin` decoding, Reallusion CC manual on iPhone depth import, Unreal Take Archive Device docs, Apple AVDepthData docs, ARKitRemap reverse-engineering project.

---

## Executive summary

**Recommendation: recapture in ARKit mode and discard the MHA take for the PersonaLive bridge.** The ARKit CSV is exactly the input format your bridge expects (52 ARKit blendshapes per frame). The MHA take's `depth_data.bin` is undocumented, almost certainly Oodle-compressed AVDepthData frames (16 or 32-bit float disparity / depth, 640×480 from TrueDepth, 30 fps), and Epic's forum response is essentially "use the in-engine pipeline" [1]. There is no public parser, the depth processing plugin ships precompiled-only [2], and your only non-Unreal path is `ooz` (an unmaintained third-party Kraken decoder) plus a guess at the per-frame layout. The depth signal *would* add information beyond ARKit-52 (head pose at sub-mm scale, jaw lateral, lip protrusion, eye-corner geometry — none of which the 52 coefficients fully express [3][4]) but the reverse-engineering cost is weeks, and even Epic's own MHA solver throws away the depth after producing 130+ proprietary `CTRL_expressions` curves that you'd then have to invert back to ARKit-52 [3]. Workflow alternative (a) — pull through Unreal once, export both ARKit-52 and the MHA curves — is feasible but adds a Windows/UE pipeline dependency you don't currently have. The 1.1 GB of one-off MHA data is not worth that. Recapture.

## Key findings

### What's actually inside the MHA take folder

A Live Link Face MHA-mode take produces, at minimum: an RGB MOV (1280×720 in your case, 60 fps), a `depth_data.bin` containing the per-frame depth stream, a `depth_metadata.mhaical` sidecar with camera intrinsics and depth metadata, a `take.json` with device/timecode information, and a `frame_log.csv` mapping video frame indices to timecodes [1][5][6]. The CSV in MHA mode contains only frame-index/timecode rows (V/A/D), *not* the 52 ARKit blendshapes — that's the critical difference from ARKit mode and the reason the take cannot directly feed an ARKit-52 → motion-latent bridge [5][6].

The `.mhaical` extension is undocumented in any public Epic source returned by search; it does not match the medical-imaging MHA format despite the prefix collision. By context (Epic naming convention "MHA i-Cal" — MetaHuman Animator iPhone calibration), it carries the TrueDepth camera intrinsics matrix, lens distortion lookup table, and per-frame depth alignment metadata. The schema is not published; it is consumed only by the Capture Manager / Take Archive Device path inside Unreal [7][8].

### What `depth_data.bin` almost certainly is

Apple's AVDepthData — which is the only API the iPhone TrueDepth sensor exposes — vends per-pixel depth as either 16-bit float or 32-bit float, in disparity or depth (metric) representations [9]. TrueDepth natively computes disparity and converts to depth on demand. ARKit on the front camera vends `capturedDepthData` at 640×480, 15 or 30 fps [9][10]. So `depth_data.bin` is a per-frame stream of 640×480 (or 360×640 in your case — Live Link Face appears to downsample to 640×360 to match the 16:9 RGB) float16/float32 depth or disparity buffers, almost certainly Oodle-compressed: the second forum thread (corrupt-depth ingest failures) attributes errors specifically to Oodle decompression, confirming the format [11]. Epic acquired RAD Game Tools (Oodle's vendor) in 2020 and Oodle Data ships with Unreal Engine [12][13]. Frame count alignment with the 60 fps RGB is approximate: depth runs at the sensor's native rate (30 fps) and Capture Manager interpolates / pairs to RGB timecodes via `frame_log.csv`.

This pixel-format claim is **inferred**, not directly confirmed by Epic docs — no source returned by search states the byte-level layout. The Epic forum thread asking exactly this question (Dec 2023) received no concrete answer [1].

### Decompression options outside Unreal

There are three theoretical paths, all bad. (1) Free Oodle SDK: Epic made Oodle Data available as a free download tier for Unreal licensees in 2023, distributed inside the engine and accessible if you have a UE source build, but the standalone SDK is still gated and not redistributable; you cannot ship a Python tool that links against `oo2core` [13]. (2) Open-source `ooz` decoder by powzix: handles Kraken/Mermaid/Selkie/Leviathan/LZNA/Bitknit, last commit 2019, no license file, marked "not fuzz safe" [14]. It works in practice (people use it for game-asset reverse engineering with `oo2core_7.dll` extracted from Warframe), but it's unmaintained and you would still have to figure out the per-frame container around the Oodle-compressed payloads. (3) Pull the take through Unreal once via Capture Manager / Take Archive Device, let the official MHA pipeline decode and export an `ImgMediaSource` of EXR depth frames plus a calibration JSON; this is the only documented and reliable path [7][8]. No Python or Blender open-source project parses MHA takes end-to-end. The closest community work is `arifyaman/Face-Depth-Frame-Mancer`, which *generates* fake depth from RGB to feed MHA — the inverse direction — and gives no help reading existing `depth_data.bin` [15].

### What the MHA in-engine pipeline does with depth

Capture Manager ingests the take, runs the MetaHuman Animator Depth Processing Plugin (precompiled binary only, not in UE source [2]), which feeds depth + RGB into a 4D solver that fits a personalised MetaHuman Identity (built from 3 frames of video + depth via Mesh-to-MetaHuman) and produces ~130+ proprietary `CTRL_expressions` curves on the MetaHuman face rig [3][16]. The output is not ARKit-52; it's a richer parameterisation. Epic ships `PA_MetaHuman_ARKit_Mapping` (a PoseAsset with weighted-least-squares synthesis weights) to project MHA curves back to ARKit-52 if you need that, and the third-party `Dylanyz/ARKitRemap` project documents that math in detail [3]. Quality: MHA + iPhone 11/12 produces visibly better lip closures, eye geometry, and head stability than ARKit-52 alone, with a noted regression on iPhone 13+ where Apple changed the dot projector pattern from 4 to 3 patterns and reduced TrueDepth fidelity for non-FaceID use [4][17].

### Is the depth track redundant given an ARKit-52 parallel take?

No, not strictly. ARKit-52 is a degraded readout from the same TrueDepth sensor — Apple's neural net regresses 52 coefficients from depth + RGB, and that regression discards information. Specifically, the 52-basis lacks: (a) jaw lateral translation as a continuous quantity (Epic's MHA exposes `CTRL_expressions_jawSideways`-style curves; ARKit only has `jawLeft`/`jawRight` as binary-like extremes), (b) tongue position (no ARKit blendshape covers tongue-tip protrusion, but MHA's solver does because the raw depth captures it), (c) lip protrusion / pucker depth at full expressivity (`mouthFunnel` + `mouthPucker` are coarse), (d) sub-mm head pose stability (depth gives metric distance, RGB+IMU does not), (e) micro-expressions outside the 52-d span [3][4]. **However**, for a PersonaLive bridge whose decoder is already conditioned on ARKit-52 motion latents, none of this extra signal is consumable without retraining the bridge — the encoder side has no slot for it. The marginal value of MHA depth is conditional on you also redesigning the motion-latent input, which is a much bigger move than this experiment justifies.

### Workflow alternatives, ranked

1. **Recapture in ARKit mode (recommended).** Zero new code. The ARKit CSV is exactly what your bridge consumes. 1.1 GB sunk cost.
2. **Pull MHA take through Unreal once, export ARKit-52 via PoseAsset mapping.** Feasible if you already have a Windows + UE 5.4+ workstation with Capture Manager and the MHA plugin set up. The export path is documented [3][7]. Adds a one-time tooling burden plus loss-of-fidelity through the MHA→ARKit-52 projection, which partially defeats the purpose of having captured in MHA mode in the first place.
3. **Reverse-engineer `depth_data.bin` directly with `ooz` + guessed framing.** Weeks of work, no documentation, no community precedent for this specific file, and even on success you'd have raw depth frames that your current pipeline cannot consume. Only worth it if you commit to a depth-aware encoder rebuild — which would itself be a multi-week thread.

## Comparison: ARKit mode vs MHA mode for this pipeline

| Property | ARKit mode | MHA mode (your accidental take) |
|---|---|---|
| CSV content | 52 blendshape coefficients + head/eye rotation per frame | frame-index/timecode rows only (V/A/D) [5][6] |
| RGB | MOV, paired with CSV | MOV 1280×720 @ 60 fps |
| Depth | not recorded | `depth_data.bin`, ~5 MB/sec, Oodle-compressed [11] |
| Calibration | none | `depth_metadata.mhaical` (intrinsics, distortion LUT) |
| Bridge consumability | direct (already 52-d) | requires Unreal pipeline or RE work |
| Fidelity ceiling | ARKit-52 basis | full MHA curves (~130+) post-solve [3] |
| Public parser | yes (CSV) | none [1] |

## What we still don't know

- The exact byte layout of `depth_data.bin`: container framing around Oodle blocks, whether each frame is independently decompressible, presence of confidence or IR companion buffers. The Dec 2023 Epic forum question is unanswered as of search date [1].
- The full schema of `depth_metadata.mhaical`. Likely lives in `Engine/Plugins/MetaHuman/.../MetaHumanCaptureSource/...` of UE source, but not in any public schema directory [7].
- Whether Live Link Face downsamples depth to match the 720p RGB or stores native 640×480; your file size (1.1 GB / 106 s ≈ 10 MB/s combined, MHA depth typically ~5 MB/s) is consistent with native-rate depth at 30 fps but doesn't pin it down.
- Whether Epic's free Oodle Data tier permits redistribution of a standalone decoder for this use case; the licensing language returned by search is ambiguous [13].
- No academic paper compares depth-driven vs ARKit-52-driven facial animation accuracy head-to-head with controlled ground truth; the closest is general blendshape-fitting accuracy work that doesn't isolate the depth contribution [18].

## Sources

[1] Epic Developer Community Forums. "Is there any way to decode depth_data.bin file captured by Live Link Face?" https://forums.unrealengine.com/t/is-there-any-way-to-decode-depth-data-bin-file-captured-by-live-link-face/1535541 (retrieved 2026-05-05)

[2] Epic Developer Community Forums. "Using MetaHumanDepthProcessing FAB plugin with source-built UE 5.7.1." https://forums.unrealengine.com/t/using-metahumandepthprocessing-fab-plugin-with-source-built-ue-5-7-1-perforce/2691035 (retrieved 2026-05-05)

[3] Dylanyz. "ARKitRemap — remap MetaHuman Animator curves to ARKit curves." GitHub. https://github.com/Dylanyz/ARKitRemap (retrieved 2026-05-05)

[4] Radical Variance. "Which is the best iPhone model for use with MetaHuman Animator?" https://face.camera/blogs/resources/best-iphone-for-metahuman-animator (retrieved 2026-05-05)

[5] Epic Games. "Using a Live Link Face Source — MetaHuman Documentation." https://dev.epicgames.com/documentation/en-us/metahuman/using-a-live-link-face-source (retrieved 2026-05-05)

[6] Virtual Filmer. "Face Animation Importer: How to Import 'MetaHuman Animator' Recordings into Unreal Engine & Export CSV." https://virtualfilmer.com/iclone-face-animation-plugin/iclone-plugin-face-animation-importer-mha-csv/ (retrieved 2026-05-05)

[7] Epic Games. "Take Archive Device — Unreal Engine 5.7 Documentation." https://dev.epicgames.com/documentation/en-us/unreal-engine/take-archive-device (retrieved 2026-05-05)

[8] Epic Games. "Capture Manager Quick Start — Unreal Engine 5.7 Documentation." https://dev.epicgames.com/documentation/en-us/unreal-engine/capture-manager-quick-start (retrieved 2026-05-05)

[9] Apple. "AVDepthData — Apple Developer Documentation." https://developer.apple.com/documentation/avfoundation/avdepthdata (retrieved 2026-05-05)

[10] Apple. "capturedDepthData — ARFrame." https://developer.apple.com/documentation/arkit/arframe/2928208-captureddepthdata (retrieved 2026-05-05)

[11] Epic Developer Community Forums. "MetaHuman Animator: Ingest Failed: Corrupt Depth Data." https://forums.unrealengine.com/t/metahuman-animator-ingest-failed-corrupt-depth-data/1739379 (retrieved 2026-05-05)

[12] RAD Game Tools. "Oodle Kraken." https://www.radgametools.com/oodlekraken.htm (retrieved 2026-05-05)

[13] Epic Games. "Oodle Data — Unreal Engine Documentation." https://dev.epicgames.com/documentation/en-us/unreal-engine/oodle-data (retrieved 2026-05-05)

[14] powzix. "ooz — open-source Kraken/Mermaid/Selkie/Leviathan/LZNA/Bitknit decompressor." GitHub. https://github.com/powzix/ooz (retrieved 2026-05-05)

[15] arifyaman. "Face-Depth-Frame-Mancer — generates face depth frames from video without a depth camera." GitHub. https://github.com/arifyaman/Face-Depth-Frame-Mancer (retrieved 2026-05-05)

[16] Epic Games / MetaHuman. "Delivering high-quality facial animation in minutes, MetaHuman Animator is now available." https://www.metahuman.com/news/delivering-high-quality-facial-animation-in-minutes-metahuman-animator-is-now-available (retrieved 2026-05-05)

[17] MoCap Online. "Face Capture for Game Dev: iPhone, ARKit, and MetaHuman." https://mocaponline.com/blogs/mocap-news/face-capture-game-dev-iphone-arkit-live-link-metahuman (retrieved 2026-05-05)

[18] Springer Nature. "Transformation of MetaHumans for a Generic XR Workflow." https://link.springer.com/chapter/10.1007/978-3-031-97778-7_5 (retrieved 2026-05-05)
