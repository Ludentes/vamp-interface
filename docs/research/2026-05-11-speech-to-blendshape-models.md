# Survey: Audio-driven (speech-to-blendshape) models — desktop/consumer-GPU feasibility (evidence-limited to provided sources)

Confirmation of scope: The report covers exactly the models and integration questions listed in the Research Brief, and prioritizes official repos, model-hub entries, official docs and papers from the provided evidence set. Where the evidence is absent or ambiguous, entries are marked "uncertain" and the latest repository/paper link or release referenced is given.

Contents
- Per-model JSON-like records (one per requested model)
- Integration appendix (ARKit52 / Live2D / VTube Studio / VMC)
- Evidence gaps
- Works Cited (numbered URLs referenced in-text)

Notes on citation style: each factual sentence is followed by bracketed source number(s) that map to the numbered Works Cited list at the end.

---

## Per-model JSON-like records

All records use these canonical output_format tags where supported by evidence: "ARKit52", "CustomBlendshapes:<name>", "FLAME", "3DMM", "MeshVertices", "Pixels", or "uncertain".

1) NVIDIA Audio2Face / Audio2Face-3D
{
  "name": "NVIDIA Audio2Face-3D (Audio2Face / Audio2Face-3D)",
  "repo_url": "https://github.com/NVIDIA/Audio2Face-3D",  [3]
  "model_hub_url": "https://huggingface.co/nvidia/Audio2Face-3D-v3.0",  [2]
  "license_url": "HuggingFace card states governed by NVIDIA Open Model License (Audio2Face-3D v3.0)",  [2]
  "output_format": "ambiguous: MeshVertices OR ARKit52 (microservice exposes ARKit blendshape names; repo/SDK supports mesh deformations, joints or blendshape weights)",  [3], [11]
  "runs_standalone": true (Audio2Face-3D repository, SDK and microservice are published and runnable via Dockerized microservice),  [3], [16]
  "consumer_gpu_runable": true (microservice lists CUDA/Docker requirements and benchmarks for consumer GPUs such as RTX‑class; GPU-specific stream counts are reported),  [16], [12]
  "latency_ms_per_frame": "reported 30 inferences per second (requires replay at 30 FPS); microservice performance guidance and RTX‑4090 stream counts provided; average latency guidance warns >100 ms affects responsiveness",  [11], [12]
  "emotion_support": ["not specified as emotion-conditioned in microservice docs; basic facial channels include skin, tongue, jaw, eyeballs according to HF card"],  [2], [11]
  "headpose_support": "microservice docs indicate certain head‑rotation blendshapes remain zero; repo/SDK supports driving head via joint transforms but microservice output leaves some head/tongue/eye values zero",  [11], [3]
  "one_line_verdict": "Official NVIDIA Audio2Face-3D provides production-grade audio→face outputs (mesh/blendshape/joint options) with a Dockerized microservice and RTX-class GPU support, but the exact canonical parameterization (native mesh vs ARKit) is ambiguous across docs.",  [3], [11], [2]
  "citations": ["https://github.com/NVIDIA/Audio2Face-3D","https://huggingface.co/nvidia/Audio2Face-3D-v3.0","https://docs.nvidia.com/ace/audio2face-3d-microservice/latest/index.html","https://docs.nvidia.com/ace/audio2face-3d-microservice/1.0/text/getting-started/quick-start.html"]
}

Notes and evidence synthesis:
- The official GitHub repo and SDK are available and include pretrained models and plugins (Maya/UE) and state the system can drive facial performance via mesh deformations, joints or blendshape weights [3].  
- NVIDIA's Audio2Face-3D microservice documentation explicitly exposes blendshape names in its animation header and uses ARKit-named blendshapes in the microservice outputs, while also noting some blendshapes (eyes/head/tongue) may be always zero and that MouthClose definition differs from standard ARKit [11].  
- HuggingFace model card for Audio2Face-3D v3.0 describes a Transformer+Diffusion architecture, release date (09/24/2025 in the card), array float facial motion output and states the license as NVIDIA Open Model License [2].  
- System prerequisites for the microservice are explicit (Ubuntu 22.04, CUDA 12.1, NVIDIA Container Toolkit, NGC key), and performance guidance lists stream counts for RTX‑class GPUs (RTX 4090 example) and a caution that >100ms latency hurts responsiveness [16], [12].  
- There is an ambiguity/conflict in the community forum noting Audio2Face historically used a different 46-blendshape template, while the microservice/docs and HF release show ARKit-named outputs; both facts are present in the evidence and therefore the parameterization is marked ambiguous [1], [11], [2].

---

2) FaceFormer (EvelynFan/FaceFormer; CVPR 2022)
{
  "name": "FaceFormer",
  "repo_url": "https://github.com/EvelynFan/FaceFormer",  [21]
  "model_hub_url": "n/a (pretrained biwi.pth and vocaset.pth downloadable from repo)",  [21]
  "license_url": "uncertain (no explicit license file referenced in provided facts)",  [21]
  "output_format": "MeshVertices (mesh vertex-based facial motion; repo uses --vertice_dim 15069 in demo)",  [21]
  "runs_standalone": true (official repo provides pretrained models and demo/render commands),  [21]
  "consumer_gpu_runable": uncertain (repo provides demo commands but GPU/driver requirements are not specified in the provided facts),  [21]
  "latency_ms_per_frame": "n/a (no runtime latency or FPS benchmarks provided in provided facts)",  [21]
  "emotion_support": ["explicit emotion predictor/augmentation described in related FLAME-based expressive model notes (paper-level technique referenced), but FaceFormer repo facts emphasize audio→mesh synthesis; explicit per-channel emotion outputs not specified"],  [22], [21]
  "headpose_support": "uncertain (repo shows rendering commands but provided facts do not list produced head-pose channels)",  [21]
  "one_line_verdict": "FaceFormer provides autoregressive transformer-based synthesis of realistic 3D facial motion as per-repo mesh outputs (vertex-based), with downloadable pretrained models but lacking explicit runtime/GPU benchmark data in the provided evidence.",  [21]
  "citations": ["https://github.com/EvelynFan/FaceFormer","https://arxiv.org/html/2301.02008v2"]
}

Notes:
- The FaceFormer repository provides pretrained weights and demo commands; demo uses a vertex_dim of 15069, indicating a mesh-vertex output pipeline rather than declared ARKit blendshapes [21].  
- A separate FLAME paper is present in the evidence set and is cited where FLAME-related methods appear in the literature, but the FaceFormer repo facts do not assert FaceFormer produces FLAME parameters per the provided evidence [22].

---

3) CodeTalker (Doubiiu / CodeTalker; CVPR 2023)
{
  "name": "CodeTalker",
  "repo_url": "https://github.com/doubiiu/codetalker",  [23]
  "model_hub_url": "n/a (pretrained biwi.pth and vocaset.pth provided in repo)",  [23]
  "license_url": "uncertain (no license file referenced in provided facts)",  [23]
  "output_format": "MeshVertices (speech→vivid 3D facial motions; CodeTalker used meshes registered to FLAME topology for VOCASET)",  [23], [39]
  "runs_standalone": true (repo provides code and pretrained models),  [23]
  "consumer_gpu_runable": uncertain (repo/pretrained available; no explicit desktop GPU runtime benchmarks in provided facts),  [23]
  "latency_ms_per_frame": "n/a (no latency benchmarks in provided facts)",  [23], [39]
  "emotion_support": ["not listed as explicit emotion conditioning in provided facts; emphasis is on discrete motion codebook and accurate lip motion"],  [23], [39]
  "headpose_support": "uncertain (paper emphasizes facial motion and lip accuracy; head-pose not explicitly documented in provided facts)",  [23], [39]
  "one_line_verdict": "CodeTalker publishes repo and pretrained weights that synthesize detailed 3D facial motion (meshes, using FLAME-registered VOCASET), but no explicit ARKit export or desktop runtime benchmarks appear in the provided evidence.",  [23], [39]
  "citations": ["https://github.com/doubiiu/codetalker","https://openaccess.thecvf.com/content/ICCV2023/papers/Peng_EmoTalk_Speech-Driven_Emotional_Disentanglement_for_3D_Face_Animation_ICCV_2023_paper.pdf"]
}

Notes:
- The CVPR/VOCASET reporting for CodeTalker shows it operates on FLAME-registered meshes and provides pretrained models in the repo; the provided facts do not show a license file or an ARKit export path [23], [39].

---

4) EmoTalk (ICCV 2023 — psyai-net/EmoTalk_release)
{
  "name": "EmoTalk",
  "repo_url": "https://github.com/psyai-net/EmoTalk_release",  [25]
  "model_hub_url": "n/a (paper + repo release)",  [25], [24]
  "license_url": "Creative Commons Attribution-NonCommercial 4.0 International (per repository facts)",  [25]
  "output_format": "ARKit52 (paper says EmoTalk uses 52 blendshape coefficients for facial animation)",  [24], [25]
  "runs_standalone": true (official release repo available),  [25]
  "consumer_gpu_runable": uncertain (repo exists; no explicit desktop GPU requirement stated in provided facts),  [25]
  "latency_ms_per_frame": "n/a (no runtime latency specified in provided facts)",  [24]
  "emotion_support": ["explicit: emotion disentangling encoder and conditioning on emotion level; EmoTalk is designed to generate emotion-enhanced blendshape coefficients"],  [24], [25]
  "headpose_support": "uncertain (paper emphasizes blendshape coefficients and emotional disentanglement; head‑pose channels not explicitly enumerated in provided facts)",  [24]
  "one_line_verdict": "EmoTalk explicitly outputs 52 blendshape coefficients and includes a learned, controllable emotional conditioning path—usable where ARKit‑52-style coefficients are required, with a noncommercial repo license.",  [24], [25]
  "citations": ["https://github.com/psyai-net/EmoTalk_release","https://openaccess.thecvf.com/content/ICCV2023/papers/Peng_EmoTalk_Speech-Driven_Emotional_Disentanglement_for_3D_Face_Animation_ICCV_2023_paper.pdf","https://ziqiaopeng.github.io/emotalk/"]
}

Notes:
- The EmoTalk paper explicitly states it predicts 52 blendshape coefficients and that emotion conditioning is part of its architecture; the official release repo uses a CC BY-NC 4.0 license [24], [25].

---

5) SadTalker (OpenTalker/SadTalker)
{
  "name": "SadTalker",
  "repo_url": "https://github.com/OpenTalker/SadTalker",  [26]
  "model_hub_url": "n/a (repo release)",  [26]
  "license_url": "repository includes license statements and usage restrictions (repo facts mention compliance and prohibited harmful use)",  [26]
  "output_format": "Pixels (single-image talking-face video outputs; SadTalker focuses on producing controllable talking-face videos), 3D motion coefficients are learned as intermediate representations (paper title and repo state '3D Motion Coefficients'), but exposure of 3DMM outputs is not fully detailed in provided facts",  [26], [41]
  "runs_standalone": true (repository states code runs completely offline),  [26]
  "consumer_gpu_runable": uncertain (repo runs offline; consumer GPU requirement not specified in provided facts),  [26]
  "latency_ms_per_frame": "n/a (no runtime latency or per-frame inference numbers in provided facts)",  [26], [41]
  "emotion_support": ["not specified as emotion-conditioned in provided facts; SadTalker emphasizes realistic motion coefficients and stylized output control"],  [26], [41]
  "headpose_support": ["paper reports learned head motion metrics in evaluations, suggesting head motion is modeled at least as learned coefficients"],  [41]
  "one_line_verdict": "SadTalker is a pixel-space, single-image talking-face system that learns realistic 3D motion coefficients internally but publishes a video/pixel output pipeline rather than native blendshape parameter streams.",  [26], [41]
  "citations": ["https://github.com/OpenTalker/SadTalker","https://ar5iv.labs.arxiv.org/html/2211.12194"]
}

Notes:
- The SadTalker repo explicitly notes offline runnable code and the project framing is learning 3D motion coefficients for stylized talking-face animation; the provided facts do not show a direct ARKit blendshape export or exposed 3DMM outputs beyond internal representation [26], [41].

---

6) DiffPoseTalk (SIGGRAPH 2024)
{
  "name": "DiffPoseTalk",
  "repo_url": "https://github.com/DiffPoseTalk/DiffPoseTalk",  [9]
  "model_hub_url": "https://diffposetalk.github.io/",  [8]
  "license_url": "uncertain (no license file referenced in provided facts)",  [9]
  "output_format": "FLAME (DiffPoseTalk uses the FLAME 3D morphable model and predicts FLAME expression parameter ψ, jaw and global rotation components, with FLAME vertex topology noted as 5023 vertices in evidence)",  [7], [9]
  "runs_standalone": true (code and pretrained models hosted on GitHub),  [9]
  "consumer_gpu_runable": true (paper/release reports runtime on Nvidia 3090 GPU),  [7]
  "latency_ms_per_frame": "reported 30 FPS generation of motion parameters on Intel Xeon Gold + Nvidia 3090; live streaming end-to-end pipeline reported total delay 7.33s (4s window + 3.33s processing) in provided facts",  [7]
  "emotion_support": ["style encoder extracts style features from reference segments (2 ms extraction reported) enabling stylistic/expression conditioning"],  [7]
  "headpose_support": "yes (predicts head pose and global rotations in addition to expression parameters)",  [7]
  "one_line_verdict": "DiffPoseTalk is a FLAME-parameterized diffusion model that produces stylistic expression and head-pose parameters with published 30 FPS generation on a 3090 and a live-streaming pipeline that incurs multi-second latency due to windowing.",  [7], [9]
  "citations": ["https://openreview.net/pdf/b26fb5f77c452aa82489b2a9a62809f2d32654d2.pdf","https://diffposetalk.github.io/","https://github.com/DiffPoseTalk/DiffPoseTalk"]
}

Notes:
- DiffPoseTalk explicitly uses FLAME and reports generation rates and a live-streaming delay budget in the paper and repo facts [7], [9], [8].

---

7) MeshTalk (facebookresearch/meshtalk)
{
  "name": "MeshTalk",
  "repo_url": "https://github.com/facebookresearch/meshtalk",  [10]
  "model_hub_url": "n/a (repo release)",  [10]
  "license_url": "Creative Commons CC-NC 4.0 International (per repository facts)",  [10]
  "output_format": "MeshVertices (MeshTalk outputs OBJ mesh geometry and uses a latent categorical expression code for animation)",  [10]
  "runs_standalone": true (code is published, but requires user data reader implementation for real data),  [10]
  "consumer_gpu_runable": uncertain (repo provides code but consumer GPU runtime specifics not in provided facts),  [10]
  "latency_ms_per_frame": "n/a (no runtime latency or FPS benchmarks provided in provided facts)",  [10]
  "emotion_support": ["uses disentangled latent codes including a categorical latent expression code — suggests expression-level control, but how that maps to ARKit channels is not provided"],  [10]
  "headpose_support": "uncertain (repo facts emphasize mesh geometry and latent expression code; head pose not explicitly enumerated in provided facts)",  [10]
  "one_line_verdict": "MeshTalk is a published mesh-output system (OBJ) with a categorical latent expression code and a CC‑NC license; it supplies mesh geometry outputs rather than ARKit blendshape streams.",  [10]
  "citations": ["https://github.com/facebookresearch/meshtalk"]
}

Notes:
- MeshTalk repo provides mesh outputs and conversion utilities; it requires a data reader for real-world usage and is CC-NC licensed [10].

---

8) Meta Audio2Photoreal (facebookresearch/audio2photoreal)
{
  "name": "audio2photoreal",
  "repo_url": "https://github.com/facebookresearch/audio2photoreal/",  [6]
  "model_hub_url": "n/a (repo + dataset release)",  [6]
  "license_url": "Creative Commons CC-NC 4.0 International (per repository facts)",  [6]
  "output_format": "Pixels / photorealistic Codec Avatars (pixel/video outputs driven by audio)",  [6]
  "runs_standalone": true (code and dataset are published),  [6]
  "consumer_gpu_runable": uncertain (repo existence suggests local runs but provided facts do not state hardware requirements),  [6]
  "latency_ms_per_frame": "n/a (no runtime latency numbers in provided facts)",  [6]
  "emotion_support": "uncertain (paper/repo facts emphasize photorealistic rendering from audio; explicit emotion conditioning not listed in provided facts)",  [6]
  "headpose_support": "uncertain (paper focuses on photorealistic codec avatars; provided facts do not enumerate parameter outputs)",  [6]
  "one_line_verdict": "Meta's audio2photoreal repo provides code/dataset to produce photorealistic, Codec-Avatar-style pixel outputs from audio under a CC‑NC license; it targets pixel/video outputs rather than blendshape parameter streams.",  [6]
  "citations": ["https://github.com/facebookresearch/audio2photoreal/"]
}

Notes:
- The repo release states photorealistic codec avatars and is CC-NC licensed; provided facts do not include parameter export details beyond pixel/avatar output [6].

---

9) Microsoft VASA-1
{
  "name": "VASA-1 (Microsoft)",
  "repo_url": "https://microsoft.com/en-us/research/project/vasa-1/",  [5]
  "model_hub_url": "https://arxiv.org/abs/2404.10667",  [18]
  "license_url": "closed / not released (project page provides no public code repository or license in provided facts)",  [5]
  "output_format": "uncertain (project/paper states 'lifelike audio-driven talking faces in real time' but provided facts do not state whether pixels or parameter vectors are produced)",  [5], [18]
  "runs_standalone": false (no public code or weights in provided facts),  [5]
  "consumer_gpu_runable": uncertain (no release to test),  [5]
  "latency_ms_per_frame": "n/a (no runtime numbers in provided facts beyond 'real time' claim in project page/arXiv)",  [5], [18]
  "emotion_support": "uncertain (not specified in provided facts)",  [5]
  "headpose_support": "uncertain (not specified in provided facts)",  [5]
  "one_line_verdict": "VASA-1 is a Microsoft research project/paper claiming real-time lifelike audio-driven talking faces but no public code/weights or license were provided in the evidence, so practical desktop use is not possible from the provided sources.",  [5], [18]
  "citations": ["https://microsoft.com/en-us/research/project/vasa-1/","https://arxiv.org/abs/2404.10667"]
}

Notes:
- Microsoft project page explicitly lacks a public repository in the provided facts; the arXiv entry confirms the research claim but does not constitute a release [5], [18].

---

10) UniTalker, Learn2Talk, GeneFace, AniTalker (grouped)
- UniTalker
{
  "name": "UniTalker",
  "repo_url": "https://github.com/X-niper/UniTalker",  [27]
  "model_hub_url": "https://x-niper.github.io/projects/UniTalker/",  [28]
  "license_url": "uncertain (no explicit license file referenced in provided facts)",  [27]
  "output_format": "FLAME (requires FLAME2020 generic_model.pkl; resources include flame.pkl),",  [27]
  "runs_standalone": true (pretrained models provided and project page),  [27], [28]
  "consumer_gpu_runable": uncertain (no GPU runtime benchmarks provided in facts),  [27]
  "latency_ms_per_frame": "n/a",  [27]
  "emotion_support": "uncertain (evidence reports scaling and multi-dataset training; explicit emotion channels not listed)",  [27], [28]
  "headpose_support": "uncertain",  [27]
  "one_line_verdict": "UniTalker is a unified FLAME-parameter-oriented model with published pretrained files and FLAME dependencies; no explicit ARKit export is documented in the provided facts.",  [27], [28]
  "citations": ["https://github.com/X-niper/UniTalker","https://x-niper.github.io/projects/UniTalker/"]
}

- Learn2Talk
{
  "name": "Learn2Talk",
  "repo_url": "no evidence in provided sources",  [--]
  "model_hub_url": "no evidence in provided sources",
  "license_url": "no evidence",
  "output_format": "no evidence",
  "runs_standalone": "no evidence",
  "consumer_gpu_runable": "no evidence",
  "latency_ms_per_frame": "n/a",
  "emotion_support": "no evidence",
  "headpose_support": "no evidence",
  "one_line_verdict": "No information in provided sources about Learn2Talk; repository, license and output format could not be located in the evidence.",  [--]
  "citations": []
}

- GeneFace
{
  "name": "GeneFace",
  "repo_url": "https://github.com/yerfor/GeneFace",  [29]
  "model_hub_url": "https://github.com/yerfor/GeneFacePlusPlus (related)",  [30]
  "license_url": "uncertain (no explicit license in provided facts)",  [29]
  "output_format": "uncertain (paper/facts note use of 3D landmarks and radiance-field representation; explicit released output parameterization not enumerated in provided facts)",  [31], [29]
  "runs_standalone": true (repo present),  [29]
  "consumer_gpu_runable": uncertain (no runtime details in provided facts),  [29]
  "latency_ms_per_frame": "n/a",
  "emotion_support": "uncertain",
  "headpose_support": "uncertain",
  "one_line_verdict": "GeneFace has published code; the evidence indicates radiance-field-based methods using 3D landmarks, but the exact exposed parameter outputs for ARKit or FLAME are not provided in the evidence.",  [29], [31]
  "citations": ["https://github.com/yerfor/GeneFace","https://ar5iv.labs.arxiv.org/html/2301.13430"]
}

- AniTalker
{
  "name": "AniTalker",
  "repo_url": "https://github.com/x-lance/anitalker",  [32]
  "model_hub_url": "https://huggingface.co/papers/2405.03121 (paper page)",  [35]
  "license_url": "uncertain (repo includes disclaimer/prohibited use note but no license file referenced in provided facts)",  [32]
  "output_format": "uncertain (paper describes universal motion representation and diffusion for lifelike talking faces; provided facts do not explicitly list ARKit/FLAME/mesh outputs)",  [35], [32]
  "runs_standalone": true (repo and ComfyUI nodes published),  [32], [33]
  "consumer_gpu_runable": true (AniTalker-ComfyUI node states example requirement: RTX 2080Ti with 11GB VRAM and torch 2.3.0+cu121),  [33]
  "latency_ms_per_frame": "n/a (no benchmark numbers in provided facts)",  [32], [33]
  "emotion_support": "uncertain (paper claims capturing subtle expressions and head movements via motion representation but explicit emotion-control outputs not enumerated in provided facts)",  [35]
  "headpose_support": "yes (paper/facts say AniTalker captures head movements)",  [35]
  "one_line_verdict": "AniTalker publishes code and ComfyUI nodes for generating lifelike talking faces with head motion; evidence shows GPU memory/torch requirements but does not show a canonical ARKit/FLAME export in the provided facts.",  [32], [33], [35]
  "citations": ["https://github.com/x-lance/anitalker","https://github.com/AIFSH/AniTalker-ComfyUI","https://huggingface.co/papers/2405.03121"]
}

Notes:
- UniTalker explicitly requires FLAME model files per repo facts, so FLAME is evidenced for UniTalker [27].  
- Learn2Talk had no evidence in the provided sources and thus is marked as missing.  
- AniTalker repository and a ComfyUI node exist with explicit example GPU memory requirement in the ComfyUI node facts [33].  
- GeneFace repo exists and paper facts indicate radiance-field processing using 3D landmarks but the provided facts do not enumerate an ARKit export [29], [31].

---

11) Rhubarb Lip Sync (DanielSWolf/rhubarb-lip-sync)
{
  "name": "Rhubarb Lip Sync",
  "repo_url": "https://github.com/DanielSWolf/rhubarb-lip-sync",  [38]
  "model_hub_url": "n/a (command-line tool repo)",  [38]
  "license_url": "https://github.com/DanielSWolf/rhubarb-lip-sync/blob/master/LICENSE.md (Boost Software License 1.0)",  [37]
  "output_format": "CustomBlendshapes:Rhubarb basic 6 mouth-shapes (plus up to 3 optional extended shapes via --extendedShapes)",  [38]
  "runs_standalone": true (command-line tool for automatic 2D mouth animation),  [38]
  "consumer_gpu_runable": true (tool runs offline on desktop command-line environments; no GPU required),  [38]
  "latency_ms_per_frame": "n/a (tool produces phoneme-to-shape mapping files; no per-frame neural inference runtime reported in provided facts)",  [38]
  "emotion_support": "none (phoneme→mouth-shape mapper only, no emotion or eye/brow channels)",  [38]
  "headpose_support": "no",  [38]
  "one_line_verdict": "Rhubarb is a lightweight, open-source phoneme-to-mouth-shape CLI that outputs a 6-shape (±3 extended) mouth-shape track suitable for 2D/hand-rig pipelines rather than full-face ARKit driving.",  [38], [37]
  "citations": ["https://github.com/DanielSWolf/rhubarb-lip-sync","https://github.com/DanielSWolf/rhubarb-lip-sync/blob/master/LICENSE.md"]
}

Notes:
- The Rhubarb repo clearly documents the basic and extended mouth-shape outputs and license (Boost 1.0) [38], [37].

---

12) NVIDIA Maxine Audio2Face / NVIDIA ACE (distinction)
{
  "name": "NVIDIA Maxine Audio2Face / NVIDIA ACE (distinction summary)",
  "repo_url": "NVIDIA ACE samples: https://github.com/NVIDIA/ACE ; Maya-ACE plugin: https://github.com/NVIDIA/Maya-ACE",  [14], [15]
  "model_hub_url": "Maxine pre-release SDK license PDF (proprietary): https://developer.download.nvidia.com/maxine/Maxine_Pre_Release_SDK_License_25Feb2021.pdf",  [17]
  "license_url": "Maya-ACE plugin released under MIT (per repo facts); Maxine Pre-release SDK is proprietary per provided PDF",  [15], [17]
  "output_format": "Audio2Face component converts audio to blendshapes (ACE lists Audio2Face as an AI component); Audio2Face-3D is a separately published NVIDIA GitHub/HF release with mesh/blendshape/joint outputs",  [14], [3], [11]
  "runs_standalone": "ACE provides microservice integration and samples; Maxine SDK is proprietary and not fully published in provided facts",  [14], [17]
  "consumer_gpu_runable": "ACE/Audio2Face components are documented to run on CUDA GPU stacks and containerized microservice flow; Maxine SDK is a closed/proprietary package per provided facts",  [14], [16], [17]
  "latency_ms_per_frame": "ACE/Audio2Face-3D microservice reports 30 inferences/sec and GPU stream benchmarks; Maxine SDK timing not provided in provided facts",  [11], [12]
  "emotion_support": "uncertain across ACE/Maxine product docs in provided facts",  [14], [17]
  "headpose_support": "ACE integration can include Audio2Face features and Maya-ACE supports tongue articulation and custom blendshape sets; Maxine status unclear in provided facts",  [15], [14]
  "one_line_verdict": "NVIDIA's ACE and Maya-ACE provide integration and plugins for Audio2Face functionality (some open-sourced components exist), while the Maxine SDK is a proprietary offering; Audio2Face-3D as published by NVIDIA is the current open model/SDK offering with a microservice and HF model card in the provided facts.",  [14], [15], [3], [2]
  "citations": ["https://github.com/NVIDIA/ACE","https://github.com/NVIDIA/Maya-ACE","https://developer.download.nvidia.com/maxine/Maxine_Pre_Release_SDK_License_25Feb2021.pdf","https://docs.nvidia.com/ace/audio2face-3d-microservice/latest/index.html","https://github.com/NVIDIA/Audio2Face-3D","https://huggingface.co/nvidia/Audio2Face-3D-v3.0"]
}

Notes:
- ACE samples repo cites Audio2Face as an AI component; Maya-ACE plugin is MIT-licensed per provided facts and includes integration that supports tongue articulation and customizable blendshape sets [14], [15].  
- The Maxine SDK pre-release license is a proprietary PDF in the provided evidence [17].

---

## Integration appendix — ARKit52, Live2D, VTube Studio, VMC / VRM / OSC

ARKit-52 -> Live2D Cubism mapping
- Evidence found: ARKitBlendshapeHelper is a Blender addon that generates ARKit‑compatible blendshapes and requires a posed facial rig as input; it produces ARKit's 52 blendshapes for facial capture usage [36].  
- No evidence was found in the provided sources for any canonical or community-prescribed direct mapping table from ARKit‑52 to Live2D Cubism parameters. The provided facts therefore show only a Blender tool that produces ARKit blendshapes but no mapping to Live2D Cubism in the evidence set [36].  
- Practical practice implied by the evidence: absent an established mapping in the provided sources, typical practice is likely to involve manual mapping or creating an intermediate rig that maps ARKit blendshapes to Live2D deformations (evidence gap: no direct mapping scripts/tables in provided sources) [36].

VTube Studio / LiveLinkFace ingestion (iPhone → desktop → VTube Studio)
- Evidence in the provided sources does not include VTube Studio, LiveLinkFace, iFacialMocap, or the iPhone-to-desktop pipeline docs; therefore no factual statements about whether VTube Studio accepts LiveLinkFace or iFacialMocap natively or via plugin/API can be made from the provided evidence. (Evidence gap: no VTube Studio/LiveLinkFace docs in provided sources.)  
- Because Audio2Face-3D microservice outputs ARKit-named blendshape streams and uses bidirectional gRPC streaming in its architecture, a desktop process that emits ARKit-named blendshape packets is technically plausible from the model side; however whether VTube Studio or other common VTuber host apps will accept those packets depends on protocols and ingestion support for which the provided evidence contains no documentation [11], [3].

VMC protocol / VRM-style OSC
- The provided sources contain no specification of the VMC protocol or VRM/OSC message formats, nor any example payloads, therefore no claim can be made from the provided evidence about whether VMC carries ARKit‑52 directly or which blendshape sets are carried. (Evidence gap.)

Practical injection feasibility (short verdict)
- From the provided evidence: it is feasible to obtain ARKit‑named blendshape streams from at least one published microservice (Audio2Face-3D microservice documents ARKit blendshape names in its animation header), and there are tools to produce ARKit‑compatible blendshapes (ARKitBlendshapeHelper) [11], [36].  
- However, the provided evidence does not include VTube Studio/LiveLinkFace/OS/protocol specs indicating how to inject ARKit streams into typical VTuber tooling; therefore a desktop-to-VTube Studio injection pipeline is conditionally feasible from the model-output side but blocked in the evidence set by missing protocol/ingestion documentation for target VTuber host software. [11], [36]

Engineering blockers (based on available evidence)
- Parameterization ambiguity: For some models (notably NVIDIA Audio2Face-3D) the canonical output parameterization is ambiguous across sources (mesh vs ARKit blendshapes) and must be resolved per release or microservice configuration [1], [11], [3].  
- Runtime/environment: Audio2Face-3D microservice requires Ubuntu 22.04, CUDA 12.1, Docker + NVIDIA Container Toolkit and NGC key per quick-start facts — this is an engineering requirement if using the NVIDIA microservice path [16].  
- Ingestion/protocol: no documented evidence in the provided sources for how to push ARKit‑52 streams into VTube Studio or equivalent hosts (evidence gap) — this is the primary integration unknown.

---

## Evidence gaps (explicit)
- No evidence in the provided sources about VTube Studio, LiveLinkFace, iFacialMocap, or their ingestion/plug-in APIs and protocols. (Blocks a definitive injection workflow description.)  
- No evidence for VMC/VRM/OSC specifications or whether they carry ARKit‑52 blendshapes in the provided sources.  
- Several repos/papers (FaceFormer, CodeTalker, UniTalker, GeneFace, MeshTalk, etc.) do not have explicit license file URLs or desktop runtime/GPU benchmark numbers in the provided facts; these items are marked "uncertain" where that is the case.  
- Learn2Talk: no evidence present in the provided source set.

---

## Short synthesis / recommendations (based on provided evidence)

- If the top requirement is to produce ARKit‑52 blendshape streams from audio on desktop with minimal cloud dependence, EmoTalk (ICCV 2023) is an explicitly relevant candidate because the paper and release state 52 blendshape coefficients and a released repo under CC BY-NC 4.0, so it directly matches ARKit‑52 parameterization needs in the provided facts [24], [25].  
- NVIDIA Audio2Face-3D provides a production microservice and a Hugging Face model card and reports ARKit-named blendshape outputs in its microservice, but the documentation shows ambiguity (historical 46-blendshape mention vs microservice ARKit names) and explicit container/runtime constraints (Ubuntu, CUDA, Docker, NGC key) in the provided evidence; it is the most fully packaged path for server/desktop deployment among the provided sources but requires Docker + NVIDIA Container Toolkit and specific CUDA versions [2], [3], [16], [11].  
- DiffPoseTalk is the clearest published FLAME-parameter solution with reported runtime numbers (30 FPS on RTX 3090) and explicit head-pose outputs in the evidence, making it a strong candidate if the target avatar pipeline accepts FLAME parameters or a FLAME→ARKit conversion is available; DiffPoseTalk's live-streaming design has multi-second delay due to windowing in the provided facts, so additional engineering would be required for low-latency streaming [7], [9].  
- For solutions that operate in pixel/video space (SadTalker, Meta audio2photoreal, AniTalker, GeneFace in some papers), the provided evidence shows these produce pixels/video rather than parameter streams; these are viable for direct video outputs but unsuitable if the goal is to inject ARKit‑52 parameters into a realtime avatar system [26], [6], [32], [29].  
- For lightweight phoneme→mouth-shape requirements (2D VTuber mouth-only) Rhubarb Lip Sync is a simple, open, license-friendly tool (Boost 1.0) producing 6 (+up to 3 extended) mouth shapes that integrate into 2D mouth-rig pipelines [38], [37].

---

## Works Cited (numbered unique URLs)

[1] https://forums.developer.nvidia.com/t/does-audio2face-use-the-arkit-52-blendshapes/349499  
[2] https://huggingface.co/nvidia/Audio2Face-3D-v3.0  
[3] https://github.com/NVIDIA/Audio2Face-3D  
[4] https://docs.nvidia.com/ace/overview/latest/index.html  
[5] https://microsoft.com/en-us/research/project/vasa-1/  
[6] https://github.com/facebookresearch/audio2photoreal/  
[7] https://openreview.net/pdf/b26fb5f77c452aa82489b2a9a62809f2d32654d2.pdf  
[8] https://diffposetalk.github.io/  
[9] https://github.com/DiffPoseTalk/DiffPoseTalk  
[10] https://github.com/facebookresearch/meshtalk  
[11] https://docs.nvidia.com/ace/audio2face-3d-microservice/latest/index.html  
[12] https://docs.nvidia.com/ace/audio2face-3d-microservice/2.0/text/interacting/performance.html  
[13] https://docs.nvidia.com/ace/audio2face-3d-microservice/1.2/text/architecture/audio2face-ms.html  
[14] https://github.com/NVIDIA/ACE  
[15] https://github.com/NVIDIA/Maya-ACE  
[16] https://docs.nvidia.com/ace/audio2face-3d-microservice/1.0/text/getting-started/quick-start.html  
[17] https://developer.download.nvidia.com/maxine/Maxine_Pre_Release_SDK_License_25Feb2021.pdf  
[18] https://arxiv.org/abs/2404.10667  
[19] https://huggingface.co/papers/2310.00434  
[20] https://arxiv.org/abs/2310.00434  
[21] https://github.com/EvelynFan/FaceFormer  
[22] https://arxiv.org/html/2301.02008v2  
[23] https://github.com/doubiiu/codetalker  
[24] https://openaccess.thecvf.com/content/ICCV2023/papers/Peng_EmoTalk_Speech-Driven_Emotional_Disentanglement_for_3D_Face_Animation_ICCV_2023_paper.pdf  
[25] https://ziqiaopeng.github.io/emotalk/  
[26] https://github.com/OpenTalker/SadTalker  
[27] https://github.com/X-niper/UniTalker  
[28] https://x-niper.github.io/projects/UniTalker/  
[29] https://github.com/yerfor/GeneFace  
[30] https://github.com/yerfor/GeneFacePlusPlus  
[31] https://ar5iv.labs.arxiv.org/html/2301.13430  
[32] https://github.com/x-lance/anitalker  
[33] https://github.com/AIFSH/AniTalker-ComfyUI  
[34] https://huggingface.co/papers/2511.23475  
[35] https://huggingface.co/papers/2405.03121  
[36] https://github.com/elijah-atkins/ARKitBlendshapeHelper  
[37] https://github.com/DanielSWolf/rhubarb-lip-sync/blob/master/LICENSE.md  
[38] https://github.com/DanielSWolf/rhubarb-lip-sync  
[39] https://openaccess.thecvf.com/content/CVPR2023/papers/Xing_CodeTalker_Speech-Driven_3D_Facial_Animation_With_Discrete_Motion_Prior_CVPR_2023_paper.pdf  
[40] https://ar5iv.labs.arxiv.org/html/2211.12194

---

End of report.