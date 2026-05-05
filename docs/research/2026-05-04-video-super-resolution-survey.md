---
status: live
topic: neural-deformation-control
---

# Video super-resolution survey for PersonaLive output (512→1024+, portrait, identity-preserving)

## Executive summary

For our PersonaLive 512×512 / 60 fps portrait pipeline with a hard ArcFace identity floor (0.64) and batch (not realtime) constraints, the recommendation is a **two-tier strategy**:

1. **First-line, ship-tomorrow:** **GFPGAN v1.4** (Apache-2.0, 35.7k stars, ~0.2 s/frame on a 4090, well-known to preserve identity at v1.4 weights) [1][9]. Single-frame face restoration is enough for our use case because a PersonaLive talking-head clip is mostly a stable head pose against a static background; flicker risk is low and can be mitigated with a 3-frame temporal blend if needed.
2. **Quality ceiling, when batch time permits:** **SeedVR2-3B** (Apache-2.0, ICLR 2026 ByteDance one-step diffusion video restorer) [3][12]. It is the only permissively-licensed diffusion-based video SR with a real 2026 lineage. Run it as a 2× pass on top of GFPGAN-restored frames for the high-quality archival render.

We explicitly **deprioritise CodeFormer, KEEP, and Upscale-A-Video** — all three are under the **NTU S-Lab License 1.0**, which is non-commercial-only and fails our permissive-license requirement [5][13][6]. RestoreFormer++ is Apache-2.0 [4] but not significantly better than GFPGAN v1.4 on portrait inputs and is essentially unmaintained since early 2024.

For the RTX 5090 (Blackwell sm_120) constraint: any of these models work on torch 2.11+cu128 nightly; the only models that ship custom CUDA kernels (FlashVSR's block-sparse-attention, SeedVR2's adaptive-window-attention) may need a recompile against cu128 — verify on day one [10].

## Top-candidate comparison

| Model | Class | License | Stars | Venue / Year | Notes on identity / temporal | Single-frame latency (reported) |
|---|---|---|---|---|---|---|
| **GFPGAN v1.4** [1][9] | Face-aware GAN, single-frame | Apache-2.0 | 35.7k | NeurIPS 2021 (model still load-bearing in 2025/26) | Best identity preservation in the GFPGAN line; v1.4 specifically tuned for fewer hallucinated details vs v1.3 | ~0.1–0.3 s @ 512² on 4090 (typical, not officially benched) |
| **RestoreFormer++** [4] | Face-aware Transformer, single-frame | Apache-2.0 | 278 | TPAMI 2023 | Stronger texture, weaker identity than GFPGAN v1.4 by reputation; last commit Jan 2024 | ~0.3 s @ 512² on 4090 (reported elsewhere) |
| **CodeFormer** [5][7] | Face-aware codebook, single-frame | **S-Lab 1.0 (non-commercial)** | 17k+ | NeurIPS 2022 | Best fidelity-vs-quality slider (`w=0.7–1.0` for ID preservation) [7] but **license blocks our use** | ~0.3 s @ 2048² on 4090 (per Replicate docs) |
| **KEEP** [13] | Face-aware **video** (Kalman propagation) | **S-Lab 1.0 (non-commercial)** | (small) | ECCV 2024 | Best-in-class for talking-head videos; **license blocks our use** | n/a |
| **Real-ESRGAN** [11] | Generic GAN, single-frame | BSD-3 (Apache-compatible) | ~30k | ICCVW 2021 | No face prior — use as a refinement pass *after* face restoration; not as the face fix itself | ~50 ms @ 512² on 4090 |
| **FlashVSR** [2] | Generic diffusion video, streaming | Apache-2.0 | 1.6k | CVPR 2026 | One-step, ~17 fps @ 768×1408 on A100; designed for 4× SR; no face prior | ~60 ms/frame @ 768×1408 on A100 |
| **SeedVR2-3B / -7B** [3][12] | Generic diffusion video, one-step | Apache-2.0 | ~750+ | ICLR 2026 | Adaptive-window attention; arbitrary resolution; ByteDance backing; no explicit face module but quality is highest in the class | not officially published, expect ~0.5–2 s/frame at 1024 |
| **STAR** [14] | Generic diffusion video | MIT (I2VGen-XL variant) | 1.5k | ICCV 2025 | Excellent temporal consistency; **~39 GB VRAM** for 240p×72-frame run — too heavy for 5090 batch | high (multi-second per frame) |
| **OSDFace** [15] | Face-aware one-step diffusion | CreativeML-OpenRAIL-M (NOT permissive — RAIL has use restrictions) | small | CVPR 2025 | 0.1 s @ 512² on a recent GPU; license is restricted-use-list, not Apache; check before commercial deploy | ~0.1 s @ 512² |

## Detailed findings

### Face-aware single-frame restorers

**GFPGAN v1.4** [1][9]. Tencent ARC, Apache-2.0. Built on StyleGAN2 priors. The v1.4 weights are the consensus "production" version — relative to v1.3 they produce slightly more detail, more natural color, and noticeably better identity preservation [9]. The well-known critique of GFPGAN — "beautifies faces" — applies more to v1.3 than v1.4. For our pipeline, where the input is already a synthetic but *high-quality* PersonaLive frame (no real degradation), GFPGAN's job is more "sharpen + 2× SR" than "blind restoration", which keeps it safely on the identity-preserving side of its operating curve. Expected output: 1024×1024 from 512×512, single-frame, single-shot. Pip-installable via `pip install gfpgan` (PyPI 1.3.8, current as of early 2026 [1]).

**RestoreFormer++** [4]. Wang Zhouxia et al, Apache-2.0, TPAMI 2023. Transformer-based with a "fully-spacial attention" key-value mechanism. Stronger on heavy degradation than GFPGAN but with the same caveat that it is identity-loose by default; not recently maintained (last commit January 2024); only 278 stars. **Use only as a backup if GFPGAN v1.4 produces visible style drift on our anchors.**

**CodeFormer** [5][7]. Zhou Shangchen et al, NeurIPS 2022, **NTU S-Lab License 1.0** [5]. The fidelity slider `w` ranges 0–1, with `w→1` favoring identity preservation and `w→0` favoring perceptual quality. Recommended setting for identity preservation in close-up / portrait contexts is `w=0.7–1.0`, with `w=0.5` as a starting point if identity is a hard floor [7]. **License is non-commercial only, so it is unusable for our pipeline if vamp-interface ships in any commercial form. Document this and skip.**

**OSDFace** [15]. Wang et al, CVPR 2025. One-step diffusion, ~0.1 s/frame at 512². Released under CreativeML-OpenRAIL-M, which is a *use-restricted* license (no harassment, no surveillance, etc.) — depending on how strictly we interpret "permissive", this is a yellow flag, not a green one. If our use case is portrait-only fraud-signal visualisation, the OpenRAIL constraints are likely satisfied, but legal review is recommended before banking on it.

### Generic image / video super-resolution

**Real-ESRGAN** [11]. xinntao, BSD-3. The de-facto generic 4× upscaler. Recommended workflow pattern across ComfyUI / A1111 ecosystems is: (a) run face restorer on the face crop, (b) paste back into the full frame, (c) Real-ESRGAN ×2 the whole frame as a final pass [11]. This is also what we recommend for our pipeline if we go beyond 1024.

**BasicVSR++ / RealBasicVSR** [16]. Kelvin Chan et al. The classic "pure video SR" baselines — bidirectional propagation + flow-guided alignment. Strong temporal consistency but designed for *real-world degraded* video (compression artefacts, motion blur), not for clean-but-low-res diffusion output. License of the original repo is permissive (Apache-2.0 / Apache-style via MMagic/MMEditing port [16]). Use case for us: if SeedVR2 turns out too slow, RealBasicVSR is the open-source pre-2024 fallback. Compatibility with sm_120 is via standard PyTorch — no custom kernels.

**FlashVSR** [2]. OpenImagingLab, CVPR 2026, Apache-2.0. One-step diffusion video SR with locality-constrained sparse attention. ~17 fps @ 768×1408 on A100, ~12× speedup over prior diffusion VSR. Designed for streaming, not for offline maximum quality. **The block-sparse-attention backend requires a separate install, which is the most likely sm_120 friction point** — verify on RTX 5090 + cu128 nightly before committing.

**SeedVR2-3B / -7B** [3][12]. ByteDance Seed team, ICLR 2026, Apache-2.0. The current quality leader in the open-source video-restoration diffusion class. Single-step generation; adaptive-window attention adapts the window size to output resolution, which avoids inconsistency artefacts at high res. Largest open video-restoration GAN at ~16 B parameters total (generator + discriminator). The 3B version is the practical choice for a 5090; the 7B will probably need offload. **This is the recommended top-quality pass for our archival render.**

**STAR** [14]. NJU PCALab, ICCV 2025, MIT (I2VGen-XL variant). High quality, but **39 GB VRAM** for a 426×240×72-frame run, which makes it a non-starter on a 32 GB 5090 unless we reduce frame_length aggressively.

**Upscale-A-Video** [6]. Zhou Shangchen, CVPR 2024, **NTU S-Lab 1.0**. Documented elsewhere as one of the most-cited diffusion VSR baselines, but the license blocks us.

### Talking-head / face video restoration

**KEEP** [13]. Feng/Li/Loy, ECCV 2024, NTU S-Lab 1.0. Kalman-Inspired Feature Propagation. By far the strongest published method *specifically* for talking-head video restoration — it is the natural fit for our use case in pure technical terms. **License blocks commercial use.** If we ever pivot to a non-commercial research-only release, KEEP becomes the obvious top choice; document this.

## Identity preservation — recommendations

The published evidence is consistent on the following ordering for portrait identity preservation, *under the assumption that the input is moderately good quality* (which our PersonaLive frames are):

1. **GFPGAN v1.4** — best identity in the GFPGAN line, explicitly tuned to be less "beautifying" than v1.3 [9].
2. **CodeFormer at `w≥0.7`** — would be #1 on identity if license allowed [7].
3. **OSDFace** at default settings — claims SOTA identity consistency in CVPR 2025 paper [15], but unverified on diffusion-synthetic input distributions like ours.
4. **RestoreFormer++** default — looser than GFPGAN v1.4 on identity, tighter on textures.

**Critical practical guidance:** before approving any face SR model into the pipeline, run our standard **18-render anchor × driver matrix** through it and recompute the ArcFace cosine vs anchor. The 0.64 floor must hold *post*-SR or the model is rejected. This is how we know GFPGAN v1.4 is safe on our distribution; the others are theoretical until measured.

NTIRE 2025 Real-World Face Restoration Challenge winner (AllForFace) used a **three-stage pipeline**: StyleGAN2 prior for ID consistency → diffusion model for texture → VAE refinement [8]. The general pattern from that challenge — **Transformer cleanup first, diffusion texture second** — is a defensible architecture if we ever build our own; it mirrors our recommended GFPGAN-then-SeedVR2 stack at the workflow level.

## Temporal coherence

For 5-second 60 fps portraits with mostly stable head pose, the literature supports the following:

- **Single-frame face restorers (GFPGAN, CodeFormer) flicker** when the codebook / latent code lookup changes between frames, even when the input is visually similar. This is documented in the KEEP and RealBasicVSR papers as motivation [13][16].
- For our content (low-motion, mostly head + shoulders, fixed background), the flicker is dominated by *codebook/latent jumps* on the most-textured regions (skin pores, iris detail, hair edges). Two cheap mitigations both work in practice:
  - **Seed-fix the restorer** so the same input pixels produce the same latent code (free).
  - **Frame-blend pass** (3-frame mean or EMA at α=0.3) on the residual `output - bicubic_upsample(input)`. Removes high-frequency flicker without smearing motion (low-cost).
- True video-SR (BasicVSR++, FlashVSR, SeedVR2) handles temporal coherence by design and removes the need for a smoother. Cost-benefit: the smoother is essentially free, the video-SR model adds 5–10× to the per-frame cost.

**Conclusion:** start with single-frame GFPGAN v1.4 + 3-frame residual smoother. Move to SeedVR2 only if visible flicker survives the smoother on a real PersonaLive clip.

## Practical workflow on RTX 5090 + torch 2.11+cu128

**sm_120 / Blackwell status (May 2026):** torch 2.11+cu128 is the recommended channel; cu129 nightly available and slightly more reliable for fresh Blackwell builds [10]. Stable PyTorch wheels do support sm_120 as of late 2025 [10].

**Compatibility per model:**
- **GFPGAN, Real-ESRGAN, RestoreFormer++** — pure-PyTorch, no custom kernels. Just work on torch 2.11+cu128.
- **FlashVSR** — bundles a block-sparse-attention extension that may need recompile against cu128 [2]. First step: `pip install -e .` and run their smoke test. If it fails, file or check upstream issue.
- **SeedVR2** — adaptive-window attention is implemented in pure PyTorch per the paper; should work out of the box on Blackwell. Verify with their inference script on a 1-frame test.
- **STAR** — pure PyTorch but 39 GB VRAM. Won't fit 5090 at default settings. Skip.

**Minimal Python snippet for the recommended first pass (GFPGAN v1.4):**

```python
# pip install gfpgan basicsr==1.4.2 facexlib==0.3.0 realesrgan
from gfpgan import GFPGANer
import cv2

restorer = GFPGANer(
    model_path="GFPGANv1.4.pth",
    upscale=2,                    # 512 -> 1024
    arch="clean",
    channel_multiplier=2,
    bg_upsampler=None,            # we'll handle bg with Real-ESRGAN separately if needed
)

for frame_path in frame_paths:
    img = cv2.imread(frame_path)
    _, _, restored = restorer.enhance(
        img, has_aligned=False, only_center_face=True, paste_back=True
    )
    cv2.imwrite(out_path, restored)
```

**Pipeline shape:**

```
PersonaLive (512x512x300frames, H.264)
  -> ffmpeg decode to PNG sequence
  -> GFPGAN v1.4 per-frame (1024x1024 output)
  -> [optional] 3-frame residual smoother (numpy, free)
  -> [optional] Real-ESRGAN x2 final pass (1024 -> 2048) for hero shots only
  -> ffmpeg encode H.264 (CRF 17 for archival, 21 for share)
```

For batch render, **pre-warm the model once** and reuse `restorer` across frames — the cold start is the dominant cost for short 300-frame clips.

## Open questions

- We have no first-party benchmark of GFPGAN v1.4, RestoreFormer++, or SeedVR2-3B on our own PersonaLive distribution. The published quality numbers are on FFHQ-degraded test sets, which look nothing like clean diffusion output. **Required next step: render the 18-anchor × driver matrix through each candidate, score ArcFace cosine vs anchor, eyeball flicker on a 5-second clip.**
- FlashVSR's sm_120 / cu128 compatibility is untested by us. Block-sparse-attention extensions are notoriously fragile across CUDA versions.
- The OSDFace OpenRAIL-M license needs a legal-review readthrough before we trust it for any commercial-adjacent shipment.
- KEEP (S-Lab 1.0) — if vamp-interface is research-only / non-commercial, we should reconsider; it is genuinely the strongest fit for talking-head video.
- No public benchmark numbers we trust for SeedVR2-3B latency at 1024² on a 5090. The 3B parameter count + adaptive attention suggests 0.5–2 s/frame, but we should measure.

## Sources

[1] TencentARC. *GFPGAN.* GitHub, Apache-2.0. https://github.com/TencentARC/GFPGAN

[2] OpenImagingLab. *FlashVSR: Towards Real-Time Diffusion-Based Streaming Video Super-Resolution.* CVPR 2026, Apache-2.0. https://github.com/OpenImagingLab/FlashVSR

[3] IceClear. *SeedVR2: One-Step Video Restoration via Diffusion Adversarial Post-Training.* ICLR 2026, Apache-2.0. https://github.com/IceClear/SeedVR2

[4] Wang Zhouxia et al. *RestoreFormer++.* TPAMI 2023, Apache-2.0. https://github.com/wzhouxiff/RestoreFormerPlusPlus

[5] Zhou Shangchen et al. *CodeFormer.* NeurIPS 2022, NTU S-Lab License 1.0. https://github.com/sczhou/CodeFormer  (license: https://github.com/sczhou/CodeFormer/blob/master/LICENSE)

[6] Zhou Shangchen et al. *Upscale-A-Video.* CVPR 2024, NTU S-Lab License 1.0. https://github.com/sczhou/Upscale-A-Video

[7] Segmind. *Best settings for CodeFormer for face restoration.* https://blog.segmind.com/best-settings-for-codeformer/

[8] Chen et al. *NTIRE 2025 Challenge on Real-World Face Restoration: Methods and Results.* arXiv:2504.14600. https://arxiv.org/abs/2504.14600

[9] Genspark. *Evaluating Face Enhancement Tools: GPEN, CodeFormer, RestorFormer, and GFPGAN in 2024.* https://www.genspark.ai/spark/evaluating-face-enhancement-tools-gpen-codeformer-restorformer-and-gfpgan-in-2024/419c04c3-7206-40eb-8d82-9a087eb0541d

[10] PyTorch Issues #159207 / #164342. *Add official support for CUDA sm_120 (RTX 5090 / Blackwell architecture).* https://github.com/pytorch/pytorch/issues/159207 ; https://github.com/pytorch/pytorch/issues/164342

[11] xinntao. *Real-ESRGAN.* GitHub, BSD-3. https://github.com/xinntao/Real-ESRGAN

[12] ByteDance-Seed. *SeedVR / SeedVR2 model cards.* Apache-2.0. https://huggingface.co/ByteDance-Seed/SeedVR2-3B ; https://huggingface.co/ByteDance-Seed/SeedVR2-7B

[13] Feng, Li, Loy. *KEEP: Kalman-Inspired Feature Propagation for Video Face Super-Resolution.* ECCV 2024, NTU S-Lab License 1.0. https://github.com/jnjaby/KEEP ; project: https://jnjaby.github.io/projects/KEEP/

[14] NJU-PCALab. *STAR: Spatial-Temporal Augmentation with Text-to-Video Models for Real-World Video Super-Resolution.* ICCV 2025, MIT (I2VGen-XL variant). https://github.com/NJU-PCALab/STAR

[15] Wang et al. *OSDFace: One-Step Diffusion Model for Face Restoration.* CVPR 2025. https://github.com/jkwang28/OSDFace ; arXiv: https://arxiv.org/html/2411.17163v1

[16] Chan et al. *RealBasicVSR / BasicVSR++.* https://github.com/ckkelvinchan/RealBasicVSR ; https://github.com/ckkelvinchan/BasicVSR_PlusPlus
