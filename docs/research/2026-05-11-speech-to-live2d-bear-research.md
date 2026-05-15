# Audio-only → Live2D Cubism (bear) — Actionable research brief (late 2025 / 2026)

Executive summary — recommended “most boring / proven” stack (short)
- Use Live2D’s built‑in audio-driven mouth (volume → ParamMouthOpenY) for real‑time mouth opening, and use Cubism Editor Motion‑sync (audio→viseme bake) for vowel shapes and timing when higher fidelity or baked keyframes are acceptable. Run in Unity with the Cubism MotionSync components to play live or baked motion data. For glue/automation, use a local audio pipeline that produces an AudioSource (microphone or TTS output) consumed by CubismAudioMouthInput (real‑time) or by the Editor’s Motion‑sync WAV bake (offline). This path is the lowest‑integration, production‑proven approach for audio‑only Live2D driving. [1], [4], [17]

Notes on evidence and scope
- All technical statements below are drawn only from the provided research findings. When a requested topic is not present in the findings (for example, vendor-specific status of NVIDIA Audio2Face or ARKit‑52 ↔ Live2D canonical mappings), an explicit Evidence Gaps section lists what is missing.

Contents
1) Canonical audio→Cubism stacks (real‑time and offline)  
2) What Cubism’s built‑in lip‑sync does (parameters, limitations, options)  
3) Community/commercial extensions that add mouth shapes, eyes, pose from audio  
4) Survey of speech→blendshape / audio→face models (evidence available) and research‑level approaches  
5) ARKit‑52 ↔ Live2D parameter mapping and bridging tools (what is/ isn’t supported in evidence)  
6) Non‑human (bear) rig considerations and recommended mapping subset for a snout face  
7) Production examples and battle‑tested combinations (concrete repos/videos)  
8) Deliverables: (a) recommended stack, (b) two alternatives, (c) compact tradeoff table, (d) concrete pointers (repos, docs, protocols)  
9) Evidence gaps  
10) References (numbered)

---

## 1) Canonical audio → Cubism stacks (real‑time and offline)

Overview
- Two production‑proven patterns emerge from the evidence:
  1. Real‑time, loudness‑driven mouth open: microphone or audio stream → sample audio level → set Cubism mouth parameter (ParamMouthOpenY / MouthOpening) each frame → Live2D renderer. This path is minimal, works live, and is provided as a first‑class capability in the Cubism SDKs and Unity components. [1], [2], [19]
  2. Viseme/vowel mapping via Motion‑sync (offline or runtime): audio file (WAV) → Cubism Editor Motion‑sync analysis (viseme detection, scaling, smoothing) → export .motionsync3.json → play back with CubismMotionSyncController (Unity) or related motion‑sync runtime components. This supports mapping individual vowels (A/I/U/E/O) and a mouth‑deformation track for richer mouth shapes and can be baked into keyframes for accurate timing. [4], [13], [17], [18]

Component breakdown (canonical real‑time, minimal)
- Audio input (Microphone or TTS output) → Unity AudioSource (or equivalent in other SDKs) → CubismAudioMouthInput / CubismMouthController → parameter value applied to target parameter(s) such as ParamMouthOpenY → Cubism model renders per-frame in LateUpdate/OnRenderObject. This is the simplest production stack for live audio‑only driving. Relevant settings: sampling quality, gain, smoothing in CubismAudioMouthInput; MouthController Blend Mode (Multiply/Additive/Override). [2], [19], [15]

Component breakdown (viseme/baked)
- Audio WAV → Cubism Editor Motion‑sync (Motion‑sync settings: viseme selection including vowels, scale, blending ratio, smoothing, sample rate) → export .motionsync3.json → runtime: CubismMotionSyncController picks up .motionsync3.json and applies viseme/mouth deformation/keyframes to parameters. This path is suitable when baked fidelity and explicit visemes are needed. Motion‑sync supports sample rates up to 100 Hz and smoothing settings. [4], [13], [12]

Realtime/interactive considerations
- Built‑in volume sampling and the Unity MotionSync components are designed for real‑time operation (microphone/audio playback support exists); the MotionSync plugin and Motion‑sync runtime components support live audio capture and per‑frame updates. For fully interactive low‑latency research systems, the literature and recent engineering notes report real‑time implementations at latencies on the order of ~120–185 ms for academic/advanced pipelines — indicating interactive latencies are feasible with optimized pipelines. [11], [12]

(Sources used in this section: [1], [2], [4], [13], [17], [18], [19], [11], [12])

---

## 2) What the Live2D Cubism SDK’s built‑in lip‑sync does

Key behaviors and parameters
- Cubism’s simplest built‑in lip sync uses sampled audio loudness to set a mouth‑open parameter (ParamMouthOpenY / MouthOpening) with values scaled in 0.0…1.0, where 1.0 is fully open and 0.0 fully closed. Developers can set or add parameter values via the SDK APIs (C++, TypeScript/Java). Sampling and smoothing settings are exposed in components such as CubismAudioMouthInput. [1], [2], [19]

Motion‑sync vs simple mouth open
- The Cubism Editor’s Motion‑sync system provides a higher‑level viseme mapping: Motion‑sync Basic uses two parameters (“Mouth Deformation” and “Mouth Open/Close”) and maps vowels A/I/O/U as individual viseme blend shapes; Motion‑sync exposes Scale, Blending Ratio, Smoothing, and Sample Rate to shape the result and can bake these to .motionsync3.json for runtime playback. The MotionSync runtime components automatically attach to models containing .motionsync3.json. [4], [13]

Blend mode and smoothing
- CubismMouthController supports Blend Modes (Multiply, Additive, Override) for how the computed mouth value is applied, and CubismAudioMouthInput offers Sampling Quality, Gain, and Smoothing controls for live audio‑driven mouth behavior. These settings control responsiveness, blending with existing animations, and micro‑movement smoothing. [2], [19]

CRI LipSync inclusion
- Live2D Cubism 5.0 includes CRI LipSync as a built‑in feature providing higher‑accuracy, real‑time lip sync driven from audio; CRI LipSync is described as working cross‑platform (Windows, macOS, iOS, Android, web) and producing natural mouth patterns without per‑speaker pretraining. [9]

Engine/timing constraints
- When driving parameters in Unity, Cubism parameters must be modified in the engine’s LateUpdate (or the provided SDK hooks) so they take effect before vertex updates during rendering. (CubismModel.OnRenderObject updates vertices after parameters are set.) This timing constraint is important when integrating an external audio pipeline. [15]

(Sources used in this section: [1], [2], [4], [13], [9], [15], [19])

---

## 3) Community / commercial extensions that extend Cubism lip‑sync

Examples documented in the evidence
- Live2DFrequencyLipSync (DenchiSoft) — extends the SDK’s original volume‑only behavior by analyzing audio frequency bands and producing more detailed mouth shapes (components expect Animator parameters like closed/open/pressed/kiss). The project supports microphone input and aims to produce mouth shapes beyond simple open/close. [10]

- Cubism MotionSync Plugin for Unity (Live2D official repo) — a wrapper/plugin enabling Motion‑sync functionality in Unity, including Microphone sample scenes and audio asset registration for motion‑sync. This provides a production path for Motion‑sync in Unity projects. [17]

- Paraworks “audio‑drive‑live2d‑with‑vits‑support” — repository that wires speech generation (VITS), audio processing and driving Live2D models using OpenGL/OpenCV; includes a command‑line tool and required build environment—an example of community projects combining TTS + audio→Live2D pipelines. [22]

- Live2D‑LLM‑Chat (suzuran0y) — a community integration that sequences ASR → LLM → TTS and drives Live2D lip sync in real time (keypress recording, pipeline orchestration). Demonstrates the practical glue code used by hobbyist/indie real‑time projects. [23]

- VTube Studio plugin ecosystem and MCP Server plugin — VTube Studio exposes a local WebSocket API to plugins; community plugins (e.g., MCP Server) exist that retrieve and manipulate Live2D parameters via the Model Context Protocol, enabling third‑party processes to set parameters remotely. This is relevant for bridging external speech‑to‑parameter systems into a Live2D host. Note: the VTube Studio plugin docs explicitly describe a local WebSocket server (default port 8001) and plugin limitations. [24], [25]

Integration notes
- Community projects show two implementation patterns: (a) direct in‑engine audio sampling to set parameters each frame (low integration overhead), or (b) external STT/TTS + audio processing pipelines that produce WAV or parameter data which is either baked into .motionsync3.json or injected via plugin/remote protocol at runtime. Example toolchains combine TTS (VITS), STT, LLMs, RTC (Agora) and Live2D rendering. [22], [23], [12], [24], [25], [10]

(Sources used in this section: [10], [17], [22], [23], [24], [25], [12])

---

## 4) Survey of speech→blendshape / audio→face models (evidence available)

What the provided evidence contains
- Academic/advanced methods for real‑time lip sync and latent‑pose pipelines have been published and demonstrate low latencies and approaches that decode audio to continuous lip‑pose vectors, which can be linearly mapped (via learned or fitted regression) to model morph targets (e.g., Live2D mouth parameters). One research example describes latent‑pose decoders mapping low‑dim codes via linear mapping to Live2D morph targets fitted by ridge regression. Another (Adobe Research) presented a deep learning LSTM system for streaming audio→viseme with reported processing latency under 200 ms. These sources document architectures and latency categories typical of high‑quality, research or production systems. [11], [13]

What is not present in the supplied evidence
- The provided findings do not include vendor‑specific details or current licensing/model availability for NVIDIA Audio2Face/Audio2Face‑3D, Meta Audio2Photoreal, Microsoft VASA‑1, or ByteDance portrait systems. The findings also do not include downloadable weights, specific output formats (e.g., ARKit‑52), or runtime requirements for those vendor products. Therefore no factual claims about their open/closed status, ability to run outside of specific vendor platforms, or ARKit output support can be made from the supplied material.

Classic/utility tools
- The evidence does not include Rhubarb Lip Sync or similar phoneme‑to‑viseme utilities. Motion‑sync and the CRI LipSync inclusion are the primary Live2D‑documented methods for viseme/vowel mapping in the supplied material. [4], [9]

Implication for Live2D
- Research‑level systems and latent‑pose decoders can produce higher‑dimensional continuous control vectors that are mappable to Live2D morph targets (conceptually useful for non‑human rigs), but implementing such a pipeline requires custom model training and a mapping/regression step to fit decoder outputs to model parameters — evidence shows such mapping has been used in research work. [11], [13]

(Sources used in this section: [11], [13])

---

## 5) ARKit‑52 ↔ Live2D parameter mapping and protocol bridging (what evidence shows)

What is present
- Motion‑sync explicitly supports vowel visemes (A/I/U/E/O) and mouth deformation and writes motion sync exports (.motionsync3.json) that the runtime consumes; runtime controllers are attached automatically for models containing motion sync data. Unity MotionSync components and the MotionSync plugin provide the bridge between audio analysis and Live2D parameters at runtime. Cubism’s mouth controller targets parameters such as ParamMouthOpenY and the SDK provides parameter APIs. [4], [17], [1]

- VTube Studio exposes a local WebSocket plugin API and community plugin(s) exist (e.g., an MCP Server) that let external processes manipulate Live2D parameters remotely; this demonstrates a supported pattern for bridging an external blendshape/parameter generator into a Live2D host via plugin/WebSocket. [24], [25]

What is missing (ARKit‑52 and canonical mappings)
- The supplied evidence set does not include an established, maintained ARKit‑52 → Live2D mapping table or a canonical conversion script/spec. No explicit references to ARKit‑52 blendshapes, Apple Live Link Face, or iFacialMocap/LiveLinkFace protocol specs appear in the findings, and the specific acceptance of ARKit‑52 payloads by VTube Studio, Inochi2D, VBridger, Animaze, or other apps is not present in the evidence. Therefore no factual mapping from ARKit‑52 names to Live2D parameters can be provided from the supplied material.

Practical bridging recipes (evidence‑backed examples)
- Audio → Cubism Editor Motion‑sync (WAV bake) → export .motionsync3.json → Unity/CubismMotionSyncController is a supported path for producing vowel/mouth deformation keyframes that run in a Live2D runtime. [13], [17]
- External TTS/ASR pipelines can drive Live2D by either (a) producing audio consumed directly by CubismAudioMouthInput, or (b) exporting WAV for Motion‑sync baking, or (c) using a Live2D host plugin (VTube Studio MCP or similar) to set parameters remotely. The community examples demonstrate these glue patterns. [12], [22], [23], [24], [25]

(Sources used in this section: [4], [17], [1], [24], [25], [13], [12], [22], [23])

---

## 6) Non‑human (bear) rig considerations

Can a human‑centric speech driver work for a snout/snout‑bearing character?
- The supplied Live2D documentation shows that mouth open/close and vowel visemes (Mouth Deformation; A/I/U/E/O) are first‑order, supported drivers; eye open/close parameters and expression parameters are also supported by the SDK and runtime. Because Live2D parameters are abstract (ParamMouthOpenY, ParamEyeLOpen, etc.), mapping human mouth open and vowel visemes to a stylized snout face is feasible in principle, but the evidence does not include explicit examples or presets for non‑human snouts. Retargeting requires conceptual mapping and tuning rather than a drop‑in ARKit→Live2D conversion (ARKit mapping evidence is missing). [1], [4], [19]

Recommended subset of parameters to drive for a bear (based on parameters present in Live2D docs)
- Load‑bearing: Mouth open/close (ParamMouthOpenY / MouthOpening), Mouth Deformation / vowel visemes (A/I/U/E/O via Motion‑sync), Eye open/close (ParamEyeLOpen/ParamEyeROpen), basic head angles (ParamAngleX/Y/Z) and general expression parameters (expression .exp3.json destinations). These will produce the core readable mouth movements and eye cues. [1], [4], [14]

- Likely irrelevant or potentially harmful (no supporting evidence for mapping): nose wrinkles, cheek squint, tongue‑out specific parameters are not discussed as part of the Motion‑sync basic preset in the supplied material and likely do not map cleanly to snout rigs; the evidence does not provide exact ARKit names to exclude. Because Motion‑sync focuses on mouth deformation and vowels, extra facial detail beyond those should be treated cautiously unless the art has corresponding parameters. [4]

Retargeting rules and smoothing recommendations (supported by the docs)
- Use weighted combinations: map continuous mouth deformation / viseme magnitude into a small number of Live2D mouth parameters (Mouth Deformation + Mouth Open). Fit per‑model weights or use a short calibration mapping (e.g., linear or ridge regression) to determine W and b for p = W * z + b where z is the decoder output — research shows this pattern is used to map latent codes to Live2D morphs. Use smoothing and sample rate settings (Motion‑sync Sample Rate, Smoothing; CubismAudioMouthInput Smoothing) to prevent jitter; choose Blend Mode appropriately (Override to fully replace, Multiply/Additive to blend with existing animation). For Unity integration, update parameters in LateUpdate so values take effect before vertex updates. [11], [2], [4], [15], [19]

Practical mitigations for common pitfalls
- Snout geometry & viseme mismatch: favor larger jaw‑open signals and mouth deformation parameters rather than trying to map fine human lip shapes directly to a snout; prefer broad vowel/mouth‑open amplitude over many small facial blendshapes. Use smoothing and thresholding to avoid micro‑artifacts when amplitude is near silence. [4], [2]

- Missing brows/eyes: if the art lacks brows or has stylized eyes, avoid mapping audio‑driven parameter outputs for those features; instead use simplified expression triggers (fade‑in/out times on expressions) to suggest emphasis. Live2D expressions (.exp3.json) support fade‑in/out and destination parameter lists. [14]

(Sources used in this section: [1], [4], [11], [2], [15], [19], [14])

---

## 7) Production examples and battle‑tested combinations (concrete)

Community and open repos (concrete)
- Live2D voice agent (example): an engineering writeup demonstrates a pipeline: microphone → TEN backend STT → LLM → TTS → audio streamed via Agora → Live2D mouth sync in the browser (PixiJS + Live2D). This shows a shipped‑style voice agent stack combining real‑time RTC and Live2D mouth sync. [12]

- Paraworks “audio‑drive‑live2d‑with‑vits‑support” — repository that generates speech (VITS), processes audio, and drives Live2D models; includes command‑line usage and list of dependencies for local builds (Windows + Visual Studio + CMake + OpenGL/OpenCV + VITS environment). This is an example of a local pipeline producing video of a Live2D character driven from generated speech. [22]

- Live2D‑LLM‑Chat — integrates ASR (SenseVoice), LLM, and TTS (CosyVoice) for real‑time Live2D lip syncing; demonstrates practical orchestration of ASR→LLM→TTS→Live2D in hobby/indie projects. [23]

- Live2DFrequencyLipSync (DenchiSoft) — community project enhancing mouth shapes using frequency analysis of audio, demonstrating a pragmatic approach to go beyond volume‑only lip sync. [10]

Notes about adoption
- Motion‑sync and the MotionSync Unity components are included in Live2D’s official toolchain and are used by projects that require baked viseme timing; CRI LipSync (included in Cubism 5.0 per the referenced blog) is positioned as a higher‑accuracy, cross‑platform option for real‑time lip sync. [17], [9]

(Sources used in this section: [12], [22], [23], [10], [17], [9])

---

## 8) Deliverables: recommended stacks, alternatives, tradeoffs, pointers

Deliverable (a) — Most boring / proven stack (audio‑only bear talking head)
- Components and data flow (real‑time, minimal integration):
  1. Audio source: local microphone or TTS output → Unity AudioSource (or equivalent)  
  2. CubismAudioMouthInput / CubismMouthController reads audio level, applies smoothing/gain → sets MouthOpening (ParamMouthOpenY) each frame.  
  3. Optionally run the same audio through Cubism Editor Motion‑sync (WAV) and export .motionsync3.json to get vowel visemes (A/I/U/E/O) and mouth deformation for higher fidelity; use CubismMotionSyncController at runtime to play those viseme tracks when available.  
  4. Render in Unity (or Web/Native) with Cubism SDK, ensure parameter updates happen in LateUpdate so on‑render vertex updates reflect them.  
- Why this stack: direct support in Live2D SDK and tools, supported runtime components for Unity/Web, minimal external ML dependencies, and community examples that implement similar approaches. [1], [2], [4], [17], [15], [19], [12]

Deliverable (b) — Two to three alternative stacks (with tradeoffs)

Alternative 1 — Higher‑quality, research‑style continuous control (best quality, custom)
- Components: microphone/TTS → researched real‑time audio→latent lip‑pose decoder (transformer/LSTM/UNet decoder) → linear/ridge regression mapping to Live2D morph targets (fit per character) → Live2D parameter injection (direct SDK or host plugin).  
- Tradeoffs: higher perceptual quality and continuous control (research reports sub‑200 ms interactive latency), but requires custom model training, decoder runtime, and mapping/regression fitting for each art style/model. Good candidate when seeking nuanced mouth shapes and willing to invest in ML integration. [11], [13]

Alternative 2 — Fully local open‑source toolchain (moderate quality, offline/command‑line)
- Components: TTS (VITS) + audio processing → Paraworks or similar local tool → drive Live2D via OpenGL/OpenCV or generate baked video.  
- Tradeoffs: permits fully local operation and offline batch production (video creation, not necessarily interactive), requires nontrivial local build environment (Windows/Visual Studio/CMake/ dependencies) and is better suited to offline rendering than low‑latency interactive use. [22]

Alternative 3 — Plugin‑bridge approach (flexible host integration)
- Components: External speech→parameter generator (or audio→WAV) → host plugin (e.g., VTube Studio plugin or MCP Server) → VTube Studio/host manipulates Live2D parameters → Live2D model renders in host.  
- Tradeoffs: flexible and enables remote or multi‑process architectures; depends on host plugin APIs and may require protocol glue (WebSocket/MCP). Plugin hosts provide a convenient bridge for streamers or multi‑process systems. [24], [25]

Deliverable (c) — Compact tradeoff table (3 stacks)  
Columns: Stack | Latency (category) | Output format | Licensing (evidence) | Non‑human friendliness

- Most boring / proven (Cubism AudioMouthInput + Motion‑sync)
  - Latency: interactive (real‑time) — no numeric latency in docs but designed for live use and sampling settings; Motion‑sync supports sampling rates up to 100 Hz.  
  - Output format: Live2D parameters (.motionsync3.json for baked; direct parameter writes for real‑time).  
  - Licensing: Live2D SDK/platform details present but licensing specifics are not provided in the supplied findings.  
  - Non‑human friendliness: moderate — mouth open + vowel visemes map reasonably to stylized snout rigs but require retargeting/tuning. [2], [4], [17], [13]

- High‑quality research continuous control (latent‑pose → linear mapping)
  - Latency: interactive / near‑real‑time reported in research (<200 ms for some systems; research numbers ~120–185 ms).  
  - Output format: continuous control vectors mapped to morph targets (requires per‑model mapping step).  
  - Licensing: not specified for individual research code/models in the supplied findings.  
  - Non‑human friendliness: good — continuous latent codes + regression mapping enable per‑model adaptation for stylized rigs. [11], [13]

- Local open‑source / batch (Paraworks + VITS)
  - Latency: offline / batch (command‑line video generation).  
  - Output format: video or Live2D parameter injection depending on implementation; the referenced repo is a tool that produces videos and drives models via OpenGL.  
  - Licensing: the supplied findings list the repositories but do not state their licenses.  
  - Non‑human friendliness: moderate — workable for offline rendering with manual mapping/tuning. [22]

(Notes: Licensing column flags that the supplied findings do not include explicit license or redistribution terms for Live2D SDK, community repos, or commercial vendors; consult original project pages for definitive licensing.) [1], [22], [11], [13], [17], [2], [4], [19]

Deliverable (d) — Concrete pointers (repos, SDK docs, plugin docs and motion sync pages)
- Live2D LipSync / ParamMouthOpenY docs: https://docs.live2d.com/en/cubism-sdk-manual/lipsync/ [1]  
- Cubism Audio/Mouth controller tutorial (Unity): https://docs.live2d.com/4.2/en/cubism-sdk-tutorials/lipsync/ [2]  
- Cubism Editor Motion‑sync (viseme/bake, vowels mapping): https://docs.live2d.com/en/cubism-editor-manual/motion-sync/ [4]  
- Motion‑sync export / bake instructions: https://docs.live2d.com/en/cubism-editor-manual/motion-sync-bake/ [13]  
- MotionSync Unity components repo: https://github.com/Live2D/CubismUnityMotionSyncComponents [17]  
- MotionSync plugin WebGL notes: https://docs.live2d.com/en/cubism-sdk-manual/about-motionsync-plugin-for-unity-webgl/ [16]  
- Cubism mouth movement & parameter timing: https://docs.live2d.com/4.2/en/cubism-sdk-manual/mouthmovement-unity/ [19]  
- Cubism SDK overview (platforms supported): https://live2d.com/sdk/about [7]  
- CRI LipSync inclusion note (Cubism 5.0): https://blog.criware.com/index.php/2023/11/17/cri-lipsync-in-live2d-cubism-5-0/ [9]  
- Live2DFrequencyLipSync (DenchiSoft community project): https://github.com/DenchiSoft/Live2DFrequencyLipSync/blob/master/README.md [10]  
- Paraworks audio→Live2D with VITS repo (local/offline example): https://github.com/Paraworks/audio-drive-live2d-with-vits-support [22]  
- Live2D‑LLM‑Chat (ASR → LLM → TTS → Live2D integration): https://github.com/suzuran0y/Live2D-LLM-Chat [23]  
- Live2D voice agent engineering writeup (Agora/TEN pipeline): https://aiengineering.beehiiv.com/p/hands-on-build-a-live2d-voice-agent-with-real-time-lip-sync [12]  
- Real‑time lip sync research / techniques and latency discussion: https://emergentmind.com/topics/real-time-lip-sync-for-live-2d-animation [11]  
- Adobe Research paper (real‑time LSTM viseme pipeline; <200 ms reported): https://research.adobe.com/publication/real-time-lip-sync-for-live-2d-animation/ [13]  
- VTube Studio plugin docs (WebSocket plugin API): https://github.com/DenchiSoft/VTubeStudio/wiki/Plugins [24]  
- VTube Studio MCP Server plugin (example of model context protocol integration): https://github.com/hkopenai/vtube-studio-plugin-mcp-server [25]  
- Motion‑sync runtime and usage in scenes (Unity instructions): https://docs.live2d.com/en/cubism-sdk-manual/use-on-scene-motion-sync-unity/ [18]  
- Expression (.exp3.json) format and fade settings: https://docs.live2d.com/en/cubism-sdk-manual/expression-unity/ [14]

(These pointers are a compact set of the most relevant URLs from the supplied evidence. Each URL appears in the References section below with its number.)

---

## 9) Evidence gaps (items requested in the brief but not present in supplied findings)

- NVIDIA Audio2Face / Audio2Face‑3D: current open/closed status (2025), licensing, model availability, weights downloadable status, ability to run outside Omniverse, and whether it outputs ARKit‑52 blendshapes — not present in the supplied evidence.

- Meta Audio2Photoreal, Microsoft VASA‑1, ByteDance audio→portrait systems: no vendor details or technical format (pixel vs parameter) are present in the findings.

- Academic models named in the brief (FaceFormer, MeshTalk, CodeTalker, EmoTalk, DiffPoseTalk, SadTalker): the supplied findings do not contain their repository links, output formats, downloadable weights, or licenses.

- ARKit‑52 → Live2D canonical mapping (official or community): no mapping table or conversion script for ARKit‑52 blendshape names to Live2D parameters appears in the provided evidence.

- Protocol specifications requested (iFacialMocap, LiveLinkFace, VMC protocol, full OSC payloads): the supplied evidence does not include these protocol spec links or definitive acceptance matrices for apps like VTube Studio/Inochi2D/VBridger/Animaze.

- Rhubarb Lip Sync (classic phoneme→viseme tool): not present in the supplied findings.

If those vendor or protocol details are required, the next research step is to retrieve primary vendor pages, GitHub repos and protocol specs for each named product (NVIDIA, Meta, Microsoft, ByteDance), and the protocol docs for iFacialMocap/LiveLinkFace/VMC/OSC.

---

## 10) References (numbered — each unique URL listed once)

[1] https://docs.live2d.com/en/cubism-sdk-manual/lipsync/  
[2] https://docs.live2d.com/4.2/en/cubism-sdk-tutorials/lipsync/  
[3] https://docs.live2d.com/en/cubism-sdk-manual/lipsync-ue/  
[4] https://docs.live2d.com/en/cubism-editor-manual/motion-sync/  
[5] https://docs.live2d.com/en/cubism-sdk-manual/motion-sync-setting-web  
[6] https://docs.live2d.com/en/cubism-sdk-manual/motion-sync-setting-native  
[7] https://live2d.com/sdk/about  
[8] https://docs.live2d.com/en/cubism-editor-manual/generating-scene-from-audio-file/  
[9] https://blog.criware.com/index.php/2023/11/17/cri-lipsync-in-live2d-cubism-5-0/  
[10] https://github.com/DenchiSoft/Live2DFrequencyLipSync/blob/master/README.md  
[11] https://emergentmind.com/topics/real-time-lip-sync-for-live-2d-animation  
[12] https://aiengineering.beehiiv.com/p/hands-on-build-a-live2d-voice-agent-with-real-time-lip-sync  
[13] https://research.adobe.com/publication/real-time-lip-sync-for-live-2d-animation/  
[14] https://docs.live2d.com/en/cubism-sdk-manual/expression-unity/  
[15] https://docs.live2d.com/en/cubism-sdk-manual/cubism-sdk-for-unity-parameter/  
[16] https://docs.live2d.com/en/cubism-sdk-manual/about-motionsync-plugin-for-unity-webgl/  
[17] https://github.com/Live2D/CubismUnityMotionSyncComponents  
[18] https://docs.live2d.com/en/cubism-sdk-manual/use-on-scene-motion-sync-unity/  
[19] https://docs.live2d.com/4.2/en/cubism-sdk-manual/mouthmovement-unity/  
[20] https://docs.live2d.com/en/cubism-editor-manual/motion-sync-setting-ow/  
[21] https://medium.com/@kinoshitayukari18/basic-step-2-making-live2d-characters-speak-with-ai-in-unity-animation-and-facial-expression-85d1afd84b93  
[22] https://github.com/Paraworks/audio-drive-live2d-with-vits-support  
[23] https://github.com/suzuran0y/Live2D-LLM-Chat  
[24] https://github.com/DenchiSoft/VTubeStudio/wiki/Plugins  
[25] https://github.com/hkopenai/vtube-studio-plugin-mcp-server

---

End of report.