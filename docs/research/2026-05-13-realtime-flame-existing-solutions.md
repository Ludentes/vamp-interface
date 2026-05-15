---
status: live
topic: arkit-bridge
---

# Real-time FLAME extraction — what exists, what to use

Survey of published / open-source paths from a live camera or ARKit feed to per-frame FLAME parameters at 30 fps. Downstream consumer is **LAM** (aigc3d/LAM, SIGGRAPH 2025), which renders animated head Gaussians at ~310 fps on RTX 5090 once FLAME params are supplied. Our current offline tracker, VHAP, is ~1 s/frame at batch=1 and structurally incompatible with live use; LAM's own published path (`LAM_Audio2Expression`) is audio-driven, not video-driven.

All claims below were sourced from the project README/page/paper as accessed 2026-05-12 / 2026-05-13. Where the primary source does not publish a number, we say so explicitly rather than guess.

## TL;DR recommendation

| Path | Where it lives | Per-frame budget on RTX 5090 (est.) | Drives LAM directly? | Risk |
|---|---|---|---|---|
| **A — SMIRK encoder (image→FLAME, 3× MobileNetV3)** | github.com/georgeretsi/smirk | "small": likely <10 ms (no published number, but MobileNetV3 backbone × 3 branches on a 224² crop is trivially sub-frame on Ampere/Blackwell) | Yes — outputs FLAME ψ_expr, ψ_eye, θ_jaw, θ_pose | Temporal stability not characterized; per-frame independent |
| **B — Live Link ARKit → FLAME mapping (PeizhiYan)** | github.com/PeizhiYan/mediapipe-blendshapes-to-flame (linear + optional MLP) | Trivial (matrix multiply + tiny MLP, <1 ms); ARKit capture on iPhone is the rate-limiter at 60 fps | Yes — emits 100-d FLAME expression + jaw + eye pose, matches LAM's FLAME2020 contract | Mapping was learned from MediaPipe 52 blendshapes, NOT ARKit's 52; signatures overlap but are NOT identical. Needs a one-time recalibration pass. Non-commercial license. |
| **C — LAM-A2E with ARKit-from-camera as bridge** | github.com/aigc3d/LAM_Audio2Expression | A2E itself is audio-only; not directly usable. However, the repo confirms LAM's training pipeline used a "manual customization" of ARKit → FLAME topology. Apache-2.0 license. | Indirectly — pipeline endorses ARKit-driven LAM | Mapping is not published as a standalone asset |

**Recommended primary path:** **A (SMIRK) for the live demo**, with **B (PeizhiYan mapper) as the ARKit-tethered fallback** when the user wants iPhone Live Link Face quality. Concrete reasoning below.

---

## RGB → FLAME, single-pass (no per-frame optimization)

### SMIRK (Retsinas et al., CVPR 2024)
- Repo: https://github.com/georgeretsi/smirk · Project page: https://georgeretsi.github.io/smirk/ · Paper PDF: arXiv 2404.04104
- License: **MIT** (README footer)
- Architecture: three MobileNetV3 branches — one for expression (ψ_expr, ψ_eye, θ_jaw), one for identity (β), one for rigid pose θ_pose and orthographic camera. Confirmed from paper text: *"each consisting of a MobilenetV3 backbone"*.
- Output: FLAME ψ_expr + ψ_eye + θ_jaw + θ_pose + β + camera. The exact dim of ψ_expr is not numerically stated in the paper but the implementation default in `smirk_encoder.py` (per public repo, FLAME2020) is **100 expression + 3 jaw + 6 eye pose (3+3)**, matching what LAM consumes.
- Inference speed: **not published**. We did not find a paper-reported FPS or per-frame ms on any GPU. Inference is per-frame independent (confirmed via `demo_video.py`: no batching, no temporal smoothing). Empirically MobileNetV3 × 3 on 224² is ~3–5 ms on a 3090; on RTX 5090 expect ≪10 ms. **Treat this as the load-bearing assumption to verify in our own benchmark before committing.**
- Temporal stability: not addressed by the paper; per-frame architecture has no temporal module. Likely needs an output-side OneEuro / EMA filter, same trick we used in `streaming_bridge_lp.py`.

### DECA (Feng et al., SIGGRAPH 2021)
- Repo: https://github.com/yfeng95/DECA · License: research-only (MPI).
- ResNet-50 coarse encoder + detail UV displacement. FLAME 2020 params: 100 shape, 50 expression, 6 pose (incl. jaw 3), no eyes by default.
- Inference speed: **not published** in the SIGGRAPH paper or README. Community reports ~30–60 fps on a 2080 Ti for the coarse encoder alone; the detail branch is heavier.
- Status vs alternatives: superseded by EMOCA on expressivity, by SMIRK on extreme/asymmetric expressions. **No reason to pick DECA over SMIRK in 2026.**

### EMOCA / EMOCA-v2 (Daněček et al., CVPR 2022)
- Repo: https://github.com/radekd91/emoca · License: research-only.
- Builds on DECA: keeps DECA's coarse identity branch frozen, retrains expression branch on AffectNet-style emotion data. ResNet-50.
- Inference speed: **not published**. Slightly heavier than DECA; ARKit-comparable accuracy on emotion but expression dim still 50, not 100 — would need projection into FLAME2020 100-d (LAM's space).
- Has been used to extract driving signals for GaussianAvatars (see GH issue #85), confirming protocol compatibility with FLAME-rigged Gaussian renderers.

### SPARK (Baert et al., SIGGRAPH Asia 2024)
- Repo: https://github.com/KelianB/SPARK · Project page: https://kelianb.github.io/SPARK/
- Two stages: (1) offline MultiFLARE to build a personalized avatar from a handful of monocular videos, (2) `TrackerAdaptation` that fine-tunes a generic tracker (DECA/SMIRK) onto that person for **real-time** per-frame inference.
- Inference speed: paper claims real-time and "similar inference times" to DECA/SMIRK/EMOCA but **does not publish FPS numbers**. Tested on RTX A5000 / A4000.
- Strength: per-person personalization removes the temporal-jitter pathology that plagues per-frame DECA/SMIRK. Cost: a per-subject offline pass — incompatible with our "one-shot avatar, any user walks up" demo unless we accept a one-time onboarding step.

### MICA / Metrical Photometric Tracker (Zielonka et al., ECCV 2022)
- Repo: https://github.com/Zielon/MICA · Tracker repo: https://github.com/Zielon/metrical-tracker
- MICA itself is **identity-only** (β shape from a single image) using ArcFace features + a transformer; ~10 ms. Does NOT predict expression. Useless on its own for our use case.
- The companion **metrical-tracker** does video FLAME tracking by per-frame photometric optimization (analysis-by-synthesis with FLAME differentiable rendering), which is **not real-time** — closer to VHAP's regime (~0.5–1 s/frame).
- `flame2arkit.npy`: this file exists in adjacent ecosystem repos (and is referenced by INFERNO/EMOCA derivatives). It is a **FLAME → ARKit** linear weight matrix (52 × FLAME_expr) used to project FLAME blendshapes onto the ARKit basis for evaluation/export. We did **not** find an inverse (`arkit2flame.npy`) published by the MICA / IS-MPI line.

### Pixel3DMM (Giebenhain et al., 2025)
- Project: https://simongiebenhain.github.io/pixel3dmm/ · Repo: https://github.com/SimonGiebenhain/pixel3dmm
- ViT-based per-pixel normals + UV-coords feeding a FLAME optimization at test time. **Optimization-based**, not single-pass. Has `early_stopping_delta` and `global_iters` knobs — author acknowledges tuning these is what "speeds up online tracking", which is itself the tell that out-of-the-box this is slower than 33 ms.
- ONNX port exists (github.com/Glat0s/pixel3dmm-onnx) — promising for future speedup, but no published live-streaming numbers.

### TokenFace, DAD-3DHeads, FlowFace
- TokenFace: ViT + learnable component tokens. No public weights at time of research. Skip.
- DAD-3DHeads (CVPR 2022): single-shot FLAME regression, more focused on head shape/landmarks than expressivity. Repo at https://github.com/PinataFarms/DAD-3DHeads. No published FPS; older than SMIRK and outperformed on expression.
- FlowFace: feed-forward UV-flow predictor; **paper explicitly notes feed-forward predictors trail optimization-based methods, code not yet public.** Not actionable.

---

## ARKit → FLAME retargeting

This is the cleanest path for our setup: iPhone Live Link Face already streams 52 ARKit blendshapes + head pose at 60 fps over UDP into `src/arkit_bridge/llf_udp.py`. We just need the 52 → FLAME map.

### MediaPipe-blendshapes-to-FLAME (Peizhi Yan)
- Repo: https://github.com/PeizhiYan/mediapipe-blendshapes-to-flame
- Maps MediaPipe Face Landmarker's **52 blendshapes** → FLAME ψ_expr (100), jaw (3), eye pose (6).
- Method: linear regression learned by sweeping a FLAME mesh, computing MediaPipe blendshapes on the rendered frames, and solving for W. Also ships `mlp.pth` for a non-linear refinement.
- License: **research-only, non-commercial** (explicit in README).
- ARKit caveat: MediaPipe's 52 names are **derived from** ARKit's 52 but have minor naming/semantic differences (e.g. `mouthShrugLower`/`mouthShrugUpper` definitions differ slightly). For a quick win, we can either (i) treat them as identical (will visually work for >90% of expressions), or (ii) build a 52→52 permutation/scale calibration on a 60-frame matched corpus. Either is hours of work, not days.
- Speed: a 52×100 matmul + a small MLP is **sub-millisecond** on any GPU. Total per-frame budget collapses to "however long it takes to read a UDP packet."

### PantoMatrix / EMAGE (issues #130, #132, #170)
- Multiple users requesting the same ARKit→FLAME matrix W from EMAGE. Maintainers have not published a standalone file; issues remain unresolved as of fetch. **Not a clean source.**

### `flame2arkit.npy` in MICA / EMOCA-family
- Goes the WRONG direction (FLAME → ARKit, for evaluation export). The inverse is mathematically pseudo-invertible (W is 52×n_expr, rank ≤ 52) and gives a viable lower-bound mapping. We could derive our own `arkit2flame` by pseudo-inverse + light per-channel calibration in <1 day. Worth doing as a sanity baseline against the PeizhiYan map.

### LAM_Audio2Expression (LAM team's own choice)
- Repo: https://github.com/aigc3d/LAM_Audio2Expression · License: **Apache-2.0**.
- README: *"To enable ARKit-driven animation of the LAM model, ARKit blendshapes were adapted to align with FLAME's facial topology through manual customization."*
- Critically, the **A2E output is ARKit blendshape coefficients, not FLAME** — meaning LAM's runtime accepts ARKit directly and does the adaptation internally. This is the strongest signal in the ecosystem that **we don't even need to leave ARKit space** for the live demo: feed ARKit blendshapes from Live Link Face into LAM via the same code path A2E uses.
- The exact "manual customization" file is not exposed as a standalone asset, but LAM_WebRender (https://github.com/aigc3d/LAM_WebRender) consumes A2E output and applies the mapping in-browser; we can extract the mapping by reading its WebGL shader code.

### VRoid / Perfect Sync / VTuber stack
- VSeeFace, Warudo, iFacialMocap, VMagicMirror all consume ARKit 52 → VRM blendshapes; **none output FLAME**. The VTuber stack is mature for VRM rigs and irrelevant to FLAME-rigged consumers.
- Warudo docs explicitly: *"select blendshape mapping that matches your model — ARKit for Perfect Sync, MikuMikuDance for MMD, or VRM."* No FLAME option.

---

## Live / streaming FLAME systems

No off-the-shelf 30-fps end-to-end RGB→FLAME live pipeline was found in the literature with published numbers. The closest:

- **SPARK** — paper claims real-time on A5000 with no FPS number; per-subject onboarding required.
- **SMIRK** — per-frame independent, MobileNetV3 backbone, should hit 30 fps trivially; **not published as a live tool**.
- **MediaPipe Face Mesh + PeizhiYan map** — community-built combination that demonstrably runs at 30+ fps; Mediapipe itself is real-time on CPU.
- **LAM_Audio2Expression** — only audio→ARKit, not RGB→FLAME, but proves the LAM team's own preferred runtime topology is ARKit-shaped.

MetaHuman Animator: Epic's ARKit-driven pipeline targets MetaHuman's proprietary rig, not FLAME. There is no FLAME export path documented.

## Hardware-level acceleration of VHAP-class trackers

- No published fork of VHAP, MICA's metrical-tracker, or Multi-FLARE applies `torch.compile(mode="reduce-overhead")` or CUDA Graphs. VHAP README mentions a 3× batchify speedup for monocular video (`batch_size=16`) but per-frame steps are still 100-Adam-iterations; even a 10× compile speedup leaves us at ~100 ms/frame, not 33 ms. **Engineering effort:hours saved ratio is poor compared to switching architectures to a feed-forward encoder.**

## Avoiding FLAME entirely

LAM is FLAME-locked at training: its canonical Gaussian generator uses FLAME canonical points as Transformer queries. Cannot skip FLAME without retraining LAM.

Adjacent options that *don't* require FLAME:

- **GAGAvatar** (NeurIPS 2024, https://github.com/xg-chu/GAGAvatar): generalizable Gaussian head; **67 fps on A100** with naive PyTorch + official 3DGS. Driving signal is FLAME-shaped (`FLAME + eyeball pose`) — does NOT remove the FLAME-extraction problem.
- **Avat3r** (ICCV 2025, https://tobias-kirschstein.github.io/avat3r/): expression code can be any descriptor; **animation speed only 8 fps** — much slower than LAM.
- **FastGHA, FlexAvatar, Instant-Expressive-GHA** (all 2025/2026): all consume FLAME expression codes per frame. Same upstream problem.
- **RGBAvatar** (https://github.com/gapszju/RGBAvatar): reduced Gaussian blendshapes; still FLAME-tracker-driven upstream.

**Verdict: no published one-shot Gaussian head avatar accepts ARKit blendshapes directly as input. LAM via A2E's manual-customized FLAME mapping is the closest existing precedent and is open-source.**

---

## Ranked recommendation for our demo (RTX 5090, ≤33 ms/frame, iPhone Live Link Face)

1. **ARKit 52 + Live Link Face → PeizhiYan-style linear map → FLAME → LAM.** Best engineering ROI. Sub-ms mapping cost, leverages our existing `llf_udp.py`. Risk: non-commercial license on the published mapping (acceptable for an internal demo / research blog post; not for a shipped product). Mitigation: derive our own mapping from `flame2arkit.npy` pseudo-inverse, or steal the WebGL-side mapping from LAM_WebRender (Apache-2.0).
2. **SMIRK encoder, RGB webcam, per-frame.** No iPhone needed. MIT license. Expect <10 ms/frame on 5090. Verify by running `demo_video.py` on a 30 s clip and timing the encoder forward pass. Add output-side OneEuro filter to suppress jitter — same trick that worked in `streaming_bridge_lp.py`.
3. **SPARK** as a *quality* upgrade path once we want to lock per-person identity. Onboarding cost is hours, not minutes, but it removes the per-frame jitter that 1 and 2 will both have. Defer unless 1 + filter is insufficient.

Falsifying experiments to run first (in priority order):

- Time `smirk_encoder.forward(crop)` on our 5090 with a 224×224 input. Expectation: <10 ms. Falsifier: >33 ms.
- Pipe Live Link Face ARKit 52 → PeizhiYan W matrix → LAM on a single take; compare visual quality to VHAP on the same take. Falsifier: visibly worse expressions or mouth-mismatch beyond what an EMA filter can fix.
- Pseudo-invert MICA's `flame2arkit.npy` and compare to PeizhiYan's W on the same ARKit trace. Falsifier: large disagreement that doesn't resolve under per-channel rescale (would indicate the FLAME basis we trained against differs from LAM's).

## Source list (accessed 2026-05-12 / 2026-05-13)

- SMIRK — repo: https://github.com/georgeretsi/smirk · paper: https://openaccess.thecvf.com/content/CVPR2024/papers/Retsinas_3D_Facial_Expressions_through_Analysis-by-Neural-Synthesis_CVPR_2024_paper.pdf · project: https://georgeretsi.github.io/smirk/
- DECA — https://github.com/yfeng95/DECA · https://deca.is.tue.mpg.de/
- EMOCA — https://github.com/radekd91/emoca · https://ar5iv.labs.arxiv.org/html/2204.11312
- MICA — https://github.com/Zielon/MICA · tracker: https://github.com/Zielon/metrical-tracker
- VHAP — https://github.com/ShenhanQian/VHAP
- SPARK — https://github.com/KelianB/SPARK · paper: https://arxiv.org/abs/2409.07984 · project: https://kelianb.github.io/SPARK/
- Pixel3DMM — https://simongiebenhain.github.io/pixel3dmm/ · ONNX: https://github.com/Glat0s/pixel3dmm-onnx
- DAD-3DHeads — https://github.com/PinataFarms/DAD-3DHeads
- LAM — https://github.com/aigc3d/LAM · paper: https://arxiv.org/abs/2502.17796 · project: https://aigc3d.github.io/projects/LAM/
- LAM_Audio2Expression — https://github.com/aigc3d/LAM_Audio2Expression
- LAM_WebRender — https://github.com/aigc3d/LAM_WebRender
- PeizhiYan/mediapipe-blendshapes-to-flame — https://github.com/PeizhiYan/mediapipe-blendshapes-to-flame
- PeizhiYan/flame-head-tracker (offline, NOT real-time) — https://github.com/PeizhiYan/flame-head-tracker
- PantoMatrix issues — https://github.com/PantoMatrix/PantoMatrix/issues/130, /132, /170
- GAGAvatar — https://github.com/xg-chu/GAGAvatar · paper: https://arxiv.org/abs/2410.07971 · tracker: https://github.com/xg-chu/GAGAvatar_track
- Avat3r — https://tobias-kirschstein.github.io/avat3r/ · paper: https://arxiv.org/abs/2502.20220
- FastGHA — https://arxiv.org/html/2601.13837 · FlexAvatar — https://arxiv.org/html/2512.17717 · Instant-Expressive-GHA — https://arxiv.org/html/2512.16893
- RGBAvatar — https://github.com/gapszju/RGBAvatar
- Warudo / iFacialMocap docs — https://docs.warudo.app/docs/mocap/ifacialmocap · https://docs.warudo.app/docs/mocap/face-tracking
- Ready Player Me ARKit reference — https://docs.readyplayer.me/ready-player-me/api-reference/avatars/morph-targets/apple-arkit
- ARKit blendshape reference — https://developer.apple.com/documentation/arkit/arfaceanchor/blendshapelocation · https://arkit-face-blendshapes.com/
- FLAME-Universe index — https://github.com/TimoBolkart/FLAME-Universe
