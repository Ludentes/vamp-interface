---
status: live
topic: lam-bakeoff
---

# LAM Ecosystem Capabilities Inventory

Date: 2026-05-13. Source repos surveyed in full:

- `/home/newub/w/LAM/` — main LAM repo (commit on `feat/windows`-style layout, has Apache-2.0 LICENSE)
- `/home/newub/w/arkit-flame-extract/LAM_WebRender/` — minimal three.js scaffold; the real rendering is in npm `gaussian-splat-renderer-for-lam@0.0.9-alpha.1` (MIT, packed to `/tmp/lam_npm/package/`)
- `/home/newub/w/arkit-flame-extract/LAM_Audio2Expression/` — Apache-2.0 standalone repo
- Paper PDF cached at `/tmp/lam.pdf` (SIGGRAPH 2025, He et al., arXiv:2502.17796), text at `/tmp/lam.txt`

## TL;DR — what LAM solves that we didn't know

1. **The 20K checkpoint is the published sweet spot, NOT the size cap.** Paper Table 4 shows authors trained and benchmarked LAM-5K (705 FPS, PSNR 20.96), LAM-20K (562 FPS, PSNR 21.43), and LAM-80K (281 FPS on A100, PSNR 22.65). 80K is what produces the headline numbers in Tables 1/2 — the public release shipped the smaller, faster 20K variant.  Sources: `/tmp/lam.txt` lines 1042-1064, 1090-1097.
2. **Default inference is photoreal-with-hair-as-Gaussians and bg color 0 (black).** No alpha pass-through, no hair-as-separate-channel; `comp_mask` (alpha) is produced as a side-channel but the rendered RGB is composited against `render_bg_colors` (default 0.0, hardcoded in `lam/runners/infer/lam.py:216`). Hair lives in the same Gaussian cloud as skin — there is no separate hair model. The "stylized" outputs in the paper (Fig 5) are achieved by editing the *input image*, not the avatar.
3. **Teeth ARE wired in code but OFF in the released inference config.** `lam/models/rendering/flame_model/flame.py:289 add_teeth()` procedurally generates 120 teeth vertices from lip rings. Inference YAML `configs/inference/lam-20k-8gpu.yaml:43` has `add_teeth: false`. But `lam/models/modeling_lam.py:122` defaults to `True` if the kwarg is absent — so the released checkpoint was likely trained WITHOUT teeth, hence the explicit `false` in the inference config to match. The `teeth_blendshape.json` + `teeth_jawopen_offset.npy` + `tracked_teeth_bs.npz` plumbing exists for a separately-trained teeth-aware checkpoint that is **not in the public release**.
4. **`shoulder_mesh.obj` is dead code in the inference path.** `gs_renderer.py:455` hardcodes `add_shoulder=False`, ignoring whatever the config says. The asset is shipped but never loaded. To use it you'd patch one line.
5. **A2E emits ARKit blendshapes ONLY, no head pose.** `engines/infer.py:233` returns `{"expression": ..., "headpose": None}`. Audio drives the face, not the neck. Pose has to come from elsewhere (LiveLinkFace, hand-authored, idle loop).
6. **A2E has 12 hardcoded identity style classes.** `num_identity_classes=12` in `models/default.py`; `id_idx=0` is the default speaker style. They are not named/documented anywhere in the repo — you pick by integer 0-11.
7. **A2E vocal-track separation runs by default before audio enters the model.** `engines/infer.py:117` (`extract_vocal_track`) — this is a pre-step that removes background music; not advertised in the README.
8. **The WebRender renderer is a Three.js fork of `@mkkellogg/gaussian-splats-3d`** with morph-target / animation.glb bolt-on. It is **independent of LAM Python** at runtime — once you have the `.zip` bundle, no LAM code is needed.
9. **WebRender state machine is hardcoded in the *demo* code, not the renderer.** Renderer accepts `getChatState()` callback returning one of `"Idle" | "Listening" | "Thinking" | "Responding"` → maps internally to animation names `"idle" | "listen" | "think" | "speak"`. State transitions are entirely owned by the demo (`LAM_WebRender/src/gaussianAvatar.ts:43-55` uses hardcoded `setTimeout`s for showcase). The renderer has no LLM/audio integration on its own.
10. **The "ARKit blendshape basis" (`flame_arkit_bs.npy`) is NOT used by the released LAM model.** `FlameHeadArkit` class in `lam/models/rendering/flame_model/flame_arkit.py` exists, but `modeling_lam.py:106` passes `smpl_type="flame"` (not `"flame_arkit"`), and `gs_renderer.py:451` always instantiates `FlameHeadSubdivided` (the non-ARKit class). The ARKit basis is consumed by the **export pipeline** (`tools/generateARKITGLBWithBlender.py` + Blender) to bake ARKit morph targets into `skin.glb` for the web renderer. LAM Python renders via FLAME PCA (`expr_param_dim: 10`, see ablation point below); the Web bundle renders via ARKit blendshapes.
11. **LAM only uses 10 FLAME PCA expression dims at inference**, not 100. `configs/inference/lam-20k-8gpu.yaml:41 expr_param_dim: 10`. VHAP tracker emits 100 (`configs/vhap_tracking/base_tracking_config.yaml:14 n_expr: 100`) — the first 10 are kept. This is a major capacity ceiling: the LAM transformer was trained to condition on only 10 expression coefficients.
12. **Eyes have full FLAME articulation but no separate iris/gaze model.** FlameMask carves out `left_iris`/`right_iris`/`sclerae`/`left_eyeball`/`right_eyeball`/`left_eyelid`/`right_eyelid` regions (`flame.py:1399-1473`) and tracking emits `eyes_pose` (3-DoF per eye via FLAME's eye joints). But there is no separate sclera shader, no iris detail, no gaze prediction — eyeballs are just colored Gaussians snapped to the FLAME eyeball vertices.
13. **Authors trained on VFHQ only, NOT NeRSemble despite the README claim.** Paper §4.1 (`/tmp/lam.txt:585`): "We utilize the VFHQ dataset [67] for training our [model] … following GAGAvatar [7]." Test sets: VFHQ test split + HDTF. README mentions "VFHQ + NeRSemble" as a future training mix, but the released checkpoint is `step_045500` and matches paper numbers from VFHQ-only training. NeRSemble is in the to-do list, not the released checkpoint.
14. **Tongue cannot move.** Stated limitation: "LAM is not able to model the tongue movement since FLAME does not model the tongue blendshapes" (`/tmp/lam.txt:1137`). This is why `skin.glb` ships 51 morph targets, not 52 — the missing one is `tongueOut`.
15. **No multi-view, no orbit camera, no relighting, no texture editing.** Renderer accepts `render_c2ws` so off-axis is mechanically possible, but the public sample motion seqs are all front-facing; any view rotation will reveal that hair and back-of-head Gaussians are extrapolated from a single front-facing image and look bad. Authors' own canonical camera is `dist_to_center=2.0` (`lam/runners/infer/lam.py:178`).

## Feature catalog

### Render pipeline

| Feature | Where | How to enable | Limits | License |
|---|---|---|---|---|
| Core renderer | `lam/models/rendering/gs_renderer.py` (`GS3DRenderer`, 950 LOC) | Default in `scripts/inference.sh` | Uses `diff-gaussian-rasterization` (`gs_renderer.py:20`) | Apache-2.0 LAM, BSL-style 3DGS rasterizer (note: original 3DGS license is research-only, but the `diff-gaussian-rasterization-wda` fork they prefer at line 20 might differ) |
| Per-frame Gaussian morph via FLAME LBS | `gs_renderer.py:570` `animate_gs_model` → `flame.py:animation_forward` | Automatic when calling `model.infer_single_view` with `flame_params` dict | Only 10 expression dims used at inference | Apache-2.0 |
| Background color | `lam/runners/infer/lam.py:216` `render_bg_colors = torch.ones(...) * 0.` | Hardcoded to 0.0 in inference; the model itself accepts arbitrary per-view bg via `background_color` kwarg in `GS3DRenderer.forward` (`gs_renderer.py:845`) | None | — |
| Alpha output (RGBA) | `gs_renderer.py:564` returns `comp_mask` from rasterizer | Already produced; just needs to be written instead of dropped | None | — |
| Depth output | `gs_renderer.py:565` `comp_depth` | Same — emitted but not saved by default | None | — |
| Mesh export (marching cubes) | `lam.py:246` `infer_mesh` | Set `export_mesh=true` in `inference.sh` | Quality is bad — marching cubes on a Gaussian density field | Apache-2.0 |
| Canonical PLY export | `lam.py:381` `save_ply(..., offset2xyz=True)` writes `_gs_offset.ply` | `save_ply=true save_img=true` | Includes per-Gaussian offsets ("offset.ply" in WebRender bundle is this) | Apache-2.0 |
| Per-view PNG export | `lam.py:370` | `save_img=true` | None | — |
| Textured deformed mesh OBJ | `lam.py:413` `mesh_utils.save_obj(... textures=colors, texture_type="vertex")` | `save_img=true` | Vertex-color only, no UV texture | — |
| Teeth (procedural geometry) | `flame.py:289 add_teeth()` creates 120 teeth verts from lip rings | `add_teeth: true` in YAML | OFF in released inference config; released checkpoint was trained without teeth | — |
| Teeth blendshape (separate jaw-open dial) | `flame.py:273 add_teeth_bs()` reads `teeth_blendshape.json` (4 keys) | `teeth_bs_flag: true` AND requires `tracked_teeth_bs.npz` alongside motion seq | OFF; expects per-frame `expr_teeth` array that VHAP doesn't emit by default | — |
| Shoulder mesh | `flame.py:229-237` appends `shoulder_mesh.obj` (5539 verts) | `add_shoulder=True` in `FlameHead` ctor — but **hardcoded False at `gs_renderer.py:455`** | Patch one line to enable; no expression dependence | — |
| Oral cavity mesh | `flame.py:238-271` loads `oral_jawopen0p5.obj` | `oral_mesh_flag: true` | OFF; gives jaw cavity vertices when mouth opens wide | — |
| Surrounding-view orbit cam | `lam.py:193 _default_render_cameras` + `cam_utils.surrounding_views_linspace` | Only invoked by `infer_video()` path which isn't called by main inference (which uses tracked motion's c2ws) | Useful for cookie-cutter NVS but the asset is single-view trained | — |
| FLAME subdivision | `flame.py:678 FlameHeadSubdivided` with `subdivide_num=1` in inference YAML | Default | One subdivision → ~20K Gaussians (hence "LAM-20K") | — |
| Eyeball/iris/sclera region masks | `flame.py:1399-1473` (`FlameMask`) | Available as `self.mask.v.{eyeballs,irises,sclerae,eyelids,...}` for downstream filtering | No separate shaders; just region indexing | — |
| Gaze direction | FLAME's `eyes_pose` (3-DoF per eye) from VHAP tracker | Emitted automatically by `flame_tracking_single_image.py` → reads iris landmarks (`detect_iris_landmarks=True` in `lam.py:145`) | Eyes rotate but no separate gaze prediction model | — |

### Identity encoder

| Feature | Where | Details |
|---|---|---|
| Backbone | `lam/models/encoders/dinov2_fusion_wrapper.py` (`Dinov2FusionWrapper`) | DINOv2 ViT-L/14 with reg tokens — `encoder_model_name: "dinov2_vitl14_reg"` (`configs/inference/lam-20k-8gpu.yaml:6`) |
| Feature fusion | `DPTHead` over multiple ViT layers (shallow + deep) | Output dim 1024. Paper §3 (`/tmp/lam.txt:334`): "fusing features derived from both shallow and deep layers" |
| Query mechanism | `e2e_flame` — learnable features attached to FLAME canonical vertices, then cross-attention with image feats | `lam/models/modeling_lam.py:88` `PointEmbed` |
| Transformer decoder | 10 layers, 16 heads, 1024 inner dim | `lam/models/transformer.py` (173 LOC), config lines 14-17 |
| One-shot only | Single image input, no multi-frame personalization mode | Paper §1: "given a single image" |
| Reconstruction time | 1.4 s per image (README table) | A100. Includes FLAME tracking preprocessing. |

### Audio2Expression

| Feature | Where | Details |
|---|---|---|
| Architecture | `models/network.py:Audio2Expression` | Wav2Vec2-base-960h (HuggingFace) audio encoder → Linear → identity-conditioned GRU → 3-layer ConvNormRelu decoder → Linear → sigmoid → 52 ARKit blendshapes |
| Output dim | 52 ARKit blendshapes (see `models/utils.py:30-84` for exact names) | Yes, includes `tongueOut` (which LAM can't render — useful only for non-LAM consumers) |
| Output framerate | 30 fps (`engines/infer.py:168` `math.ceil(audio.shape[0]/ssr*30)`) | Locked, not configurable in production |
| Streaming API | `engines/infer.py:159 infer_streaming_audio(audio, ssr, context)` | Uses 64-frame (2.13s) lookback window. Returns `context` to chain. |
| Streaming latency | **Not stated in repo.** Author says "real-time"; no per-chunk ms number documented. The example loop in `inference_streaming_audio.py` feeds 1s chunks. Need to benchmark. | Open question |
| Vocal extraction | `engines/infer.py:117 extract_vocal_track` runs by default (`ex_vol=True`) | Removes BGM before model — uses a separate model under `pretrained_models/` |
| Identity style | `id_idx` int 0..11 (`num_identity_classes=12`) | Not documented which int is which speaker style |
| Head pose | Not produced (`return … "headpose": None`) | Must come from another source |
| Emotion conditioning | None — input is audio + id_idx only | No global emotion vector |
| Post-processing | `apply_expression_postprocessing` does: smooth-mouth, random brow flicker, Savitzky-Golay smoothing, random blinks, symmetrization | Toggleable via `movement_smooth`, `brow_movement` flags in config |
| License | Apache-2.0 (`LAM_Audio2Expression/LICENSE`) | Clear |
| Weights | `lam_audio2exp_streaming.tar` from HuggingFace `3DAIGC/LAM_audio2exp` | — |

### Model variants (paper Table 4)

| Variant | Gaussian count | PSNR (VFHQ) | FPS on A100 | Public weights? |
|---|---|---|---|---|
| LAM-5K | ~5K | 20.96 | 705.63 | No |
| LAM-20K | ~20K | 21.43 | 562.97 | **Yes** (`3DAIGC/LAM-20K` on HF) |
| LAM-80K | ~80K | 22.65 | 280.96 | No |

The README also lists three training-data variants of LAM-20K (VFHQ; VFHQ+NeRSemble; "our large dataset") — only the first **and second** are listed as on HF (table shows the same HF link for both rows of VFHQ-only and VFHQ+NeRSemble, so it is ambiguous which one we actually downloaded; size and step_045500 timestamp match the paper-VFHQ checkpoint).

### Avatar export pipeline (offline → web bundle)

| Step | Tool | Output |
|---|---|---|
| 1. Generate avatar | `app_lam.py` (gradio) or `scripts/inference.sh` | Canonical PLY + per-image PNGs + `framework_img.obj` |
| 2. Inject FLAME shape into FBX template | `tools/generateARKITGLBWithBlender.py:update_flame_shape` (uses FBX SDK; expects template at `assets/sample_oac/template_file.fbx` with hardcoded `Vertices: *60054 {` header) | ASCII FBX with patient-specific verts |
| 3. FBX ASCII → Binary | FBX SDK | Binary FBX |
| 4. FBX → GLB | `tools/generateGLBWithBlender_v2.py` invoked via Blender ≥ 4.0 in background mode | `skin.glb` with **51 morph targets** (52 ARKit minus `tongueOut`) |
| 5. Bundle | Zip `{skin.glb, animation.glb, offset.ply, vertex_order.json}` into `p2-1.zip` | Web-ready bundle (~7 MB) |
| Wall clock | Not benchmarked in repo. App uses `--blender_path` to spawn Blender per export. | Estimated minutes, not seconds |
| Prereqs | FBX SDK 2020.2+ (research/commercial license required), Blender ≥ 4.0 (GPL), `sample_oac.tar` template files from Alibaba OSS | — |

### WebRender (npm `gaussian-splat-renderer-for-lam@0.0.9-alpha.1`)

| Feature | Details |
|---|---|
| Base | Fork of `@mkkellogg/gaussian-splats-3d` (Three.js) — `/tmp/lam_npm/package/README.md` |
| License | MIT (`/tmp/lam_npm/package/package.json:author: "Xiaodan Ye"`) — note the LAM SDK author chose MIT, distinct from Apache-2.0 main LAM repo |
| Asset shape | Single `.zip` with `skin.glb` (animatable mesh, 51 morph targets), `offset.ply` (per-Gaussian offsets from FLAME-template verts), `animation.glb` (skeletal animations), `vertex_order.json` (Gaussian↔FLAME-vertex mapping) |
| Renderer API | `GaussianSplatRenderer.getInstance(div, zipPath, { getChatState, getExpressionData, backgroundColor, alpha })` |
| State machine | Renderer reads `getChatState()` returning `"Idle"`/`"Listening"`/`"Thinking"`/`"Responding"` strings, maps internally to animation names `idle`/`listen`/`think`/`speak` — **state transitions are caller's responsibility**, the renderer just plays the right clip |
| Expression callback | `getExpressionData()` returns `{ [blendshapeName]: weight }` dict at whatever rate it's polled (animation.glb uses 30 fps internally) |
| Named animations in `animation.glb` | The 12 `yumi_h5_a3_*` names you've seen are a default per-character animation pack — these are **not** built into the renderer; they're in the GLB. To author your own, create a GLB with `idle`, `listen`, `think`, `speak` named NLA tracks. |
| Sample bundle | `LAM_WebRender/asset/arkit/p2-1.zip` — 7.1 MB, dated 2025-03-17 |
| Test expression data | `asset/test_expression_1s.json` — 30-frame ARKit weights loop |
| No LLM integration | Renderer is pure render; LLM/ASR/TTS are in the OpenAvatarChat sibling project, not this package |

### Paper-reported numbers (VFHQ test, LAM-80K is "Ours")

Self-reenactment, from `/tmp/lam.txt:670-839`:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | CSIM↑ | AED↓ | APD↓ | AKD↓ |
|---|---|---|---|---|---|---|---|
| GAGAvatar [Chu & Harada 2024] | 21.83 | 0.818 | 0.122 | 0.816 | 0.111 | 0.135 | 3.349 |
| Portrait4D-v2 | 21.34 | 0.791 | 0.144 | 0.803 | 0.117 | 0.187 | 3.749 |
| GPAvatar | 21.04 | 0.807 | 0.150 | 0.772 | 0.132 | 0.189 | 4.226 |
| **Ours (LAM-80K)** | **22.65** | **0.829** | **0.109** | **0.822** | **0.102** | **0.134** | **2.059** |

LAM-80K beats GAGAvatar by 0.82 PSNR and 0.013 LPIPS — a real but not dramatic gap. Released LAM-20K (PSNR 21.43) is **0.40 PSNR worse than GAGAvatar's 21.83**. This is non-trivial. The reason to pick LAM over GAGAvatar at the released checkpoint is FPS (562 vs 9.62 = 58× on A100) and the "no extra network at inference" property, not raw quality.

Cross-platform FPS (paper "Ours on Different Platforms"):

| Platform | A100 | Macbook M1 Pro | iPhone 16 | Xiaomi 14 |
|---|---|---|---|---|
| Animate + Render | 562 fps | 120 | 35 | 26 |

(README mentions "110+ FPS on XiaoMi 14" which doesn't match Table 3's 26 — README is for LAM-20K, paper Table 3 column appears to be LAM-80K; the README number is for the released smaller model.)

INSTA, GaussianAvatars, CAP4D are **not in the comparison tables**. The paper compares against single-shot methods only.

### Stylized inputs in the paper

Fig 5 (referenced at line 664 of paper) does show stylized inputs ("Stylize Editing of Animatable Gaussian Avatar"). But the mechanism is: **edit the input image in 2D first (e.g., via existing 2D editing prior models), then run LAM on the edited image**. There's no eval on cartoon/anime inputs, no quantitative number for non-photo. This squares with our LAM-bakeoff finding (`memory/project_lam_bakeoff_x6_verdict.md`): anime + non-human are out of FLAME morphology; stylized humanoids (orc, demon) work after detector threshold relaxation.

### Stated limitations (verbatim from paper §5)

> "We utilize FLAME parameters to animate our reconstructed 3D Gaussian avatar. It fails to reproduce expressions that FLAME cannot model. For example, LAM is not able to model the tongue movement since FLAME does not model the tongue blendshapes. Since we eliminate the 2D post-processing network for efficient animation and rendering, some expression-dependent details like dynamic wrinkles cannot be fully modeled. The limitation of FLAME expressiveness and the challenge of single input image also limits the expression neutralization capability. Also, since we utilize algorithms to estimate FLAME from video, the inaccurate estimated FLAME parameters will infect the results." (`/tmp/lam.txt:1136-1143`)

Implications:
- No tongue (confirms missing `tongueOut` morph in `skin.glb`)
- No dynamic wrinkles (foreheads stay smooth under expression)
- Expression neutralization is weak — if input image has a smile, baseline canonical face inherits some of it
- Tracking errors compound

### Third-party model dependencies

| Component | Source | License | Path |
|---|---|---|---|
| FLAME 2023 | MPI (`flame.is.tue.mpg.de`) | **Research-only**, non-commercial | `model_zoo/human_parametric_models/flame_assets/flame/flame2023.pkl` |
| FLAME masks (parts, lmk embeddings) | MPI | Research-only (derived from FLAME) | `FLAME_masks.pkl`, `landmark_embedding_with_eyes.npy` |
| FLAME template mesh | MPI | Research-only | `head_template_mesh.obj` |
| DINOv2 ViT-L/14 | Meta | **Apache-2.0** | Loaded via name `"dinov2_vitl14_reg"`, downloaded at runtime |
| Wav2Vec2-base-960h | Facebook | **Apache-2.0** | A2E only |
| VGGHead detector | Xuangeng Chu (`xg.chu@outlook.com`), `external/vgghead_detector/` | TorchScript .trcd — license **unspecified** in repo, original from GAGAvatar source | `vgg_heads_l.trcd` |
| StyleMatte | external/human_matting | Apache-2.0 (typical for StyleMatte) | `stylematte_synth.pt` |
| FaceBoxesV2 | external/landmark_detection | unspecified | `FaceBoxesV2.pth` |
| 68-keypoint landmarks | external/landmark_detection | unspecified | `68_keypoints_model.pkl` |
| diff-gaussian-rasterization (CUDA) | Original from Inria 3DGS or fork by ashawkey (`pip install git+...ashawkey/diff-gaussian-rasterization/`) | **Research/non-commercial** (Gaussian Splatting license — see Inria) | imported at `gs_renderer.py:20-22` |
| pytorch3d | Facebook | BSD-3 | Used for mesh ops |
| FBX SDK | Autodesk | Proprietary, free for limited use | Required only for `tools/generateARKITGLBWithBlender.py` (avatar export) |
| Blender | Blender Foundation | GPL | Avatar export only |
| diff-gaussian-rasterization-wda (preferred fork) | `wda` = Wonder Dynamics? Or `weights diff annotation`? — **provenance unclear from repo** | unspecified | Tried first at `gs_renderer.py:20` |

**Commercial-use net:** the FLAME model + Gaussian-Splatting rasterizer + the un-licensed third-party detectors put the whole stack in research-only territory at the released configuration. LAM's own Apache-2.0 license covers their original code only.

## Specific dimensions: answered

### Render pipeline — what's default vs opt-in

- **Teeth**: code present, OFF in released config; would require re-training to use
- **Shoulders**: code present, hardcoded OFF in `gs_renderer.py:455`; would render as static (no expression dependence)
- **Oral cavity**: code present, OFF; appended Gaussian vertices when jaw is open
- **Eyes**: full FLAME articulation always on (eyes_pose, eyelids, iris region masks); no separate gaze model
- **Background**: hardcoded 0.0 (black) at `lam.py:216`; alpha emitted as side-channel `comp_mask` but not saved to video by default
- **Hair**: no separate handling — hair is just Gaussians in the same field as skin; matting mask is used only during *training* to mask the loss

### A2E deep-dive

- Architecture: Wav2Vec2-base-960h → Linear(768→512) → identity-aware GRU + 3-layer Conv decoder → Linear → sigmoid → 52 dims
- Latency per chunk: **not stated, must benchmark** — example script feeds 1 s windows
- Output rate: 30 fps locked
- Streaming: YES (`infer_streaming_audio`), 64-frame lookback
- Quality vs Audio2Face: **no public comparison**
- License: Apache-2.0 confirmed
- Emits head pose: **NO** (`headpose: None`)
- Emits emotion: NO; only 12-class hardcoded identity-style index

### Identity encoder — failure modes

Paper enumerates expression-neutralization weakness and tracking-error compounding; doesn't explicitly enumerate stylized-input failures. Our bake-off (`project_lam_bakeoff_x6_verdict`) supplies that: anime + non-human fail FLAME tracking entirely; stylized humanoids work after VGGHead threshold relaxation.

No video personalization mode — single image only.

### WebRender capabilities

- Contains: Three.js GS renderer (12.5 MB unpacked), bundles loader for `{skin.glb, offset.ply, animation.glb, vertex_order.json}`
- Independent of LAM Python at runtime: yes, fully — once the zip is built, no Python in the loop
- 12 named animations: live inside `animation.glb`, per-bundle (per-character). To author your own, regenerate that GLB.

### Quiet capabilities checklist

| Capability | Present? |
|---|---|
| Lip sync from phonemes | Indirect — A2E reads raw audio, not phonemes |
| Eye gaze tracking output | Yes — `eyes_pose` from VHAP; no separate gaze net |
| Emotion conditioning beyond ARKit | No |
| Multi-view consistency (orbit) | Mechanically possible (`infer_video`), but trained from single front view; back of head will be poor |
| Texture editing / relighting | No |
| Crowd / multi-avatar scenes | Web renderer is per-instance; nothing prevents stacking, no built-in orchestration |
| Mobile (iPhone) / Vision Pro | Yes for mobile (paper claims 35 fps iPhone 16); no Vision Pro hooks |
| TRT / ONNX exports | No — PyTorch only at inference |

### Omissions worth knowing

- No tongue
- No dynamic wrinkles
- No relighting
- No expression neutralization (input smile leaks into canonical)
- No multi-view input fusion
- No video personalization
- Hair from back of head is fake (front-view extrapolation)
- VHAP FLAME tracker is the bottleneck on stylized inputs
- A2E doesn't produce head pose
- WebRender state machine is in the DEMO, not the renderer — caller-owned
- Released ckpt is LAM-20K, not the 80K used in paper's headline tables → published quality numbers DO NOT apply to what we are deploying
- `flame_arkit_bs.npy` is for export-to-web bake, not LAM inference
- Shoulder mesh is dead code in inference
- Teeth requires a separately-trained checkpoint we don't have

## Open questions (worth a follow-up spike)

1. **Benchmark A2E latency**: what is per-1s-chunk wall clock on RTX 5090? Need a script that times `infer_streaming_audio` over a 30s clip.
2. **Confirm released checkpoint identity**: HF model card for `3DAIGC/LAM-20K` says VFHQ-only or VFHQ+NeRSemble? README table has the same HF link in both rows.
3. **Can we cheaply enable teeth?** Patch `add_teeth: true` in YAML and re-train, or hot-patch and see what the unconditioned teeth Gaussians look like with no teeth-specific training. Quick test.
4. **Shoulder + body shot**: patch `add_shoulder=True` at `gs_renderer.py:455` and render — what happens to the Gaussians on the shoulder mesh? Are they predicted as zeros (since not in training)?
5. **`diff-gaussian-rasterization-wda` provenance**: which fork, who maintains it, what license? Affects commercial story.
6. **80K checkpoint availability**: is there a private way to get LAM-80K from Alibaba? The 0.82 PSNR / 0.013 LPIPS lift over GAGAvatar disappears at LAM-20K.
