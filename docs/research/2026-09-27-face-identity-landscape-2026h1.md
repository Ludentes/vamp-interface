---
status: live
topic: photobooth-sweep
---

# Research: face identity transfer landscape, April–September 2026

**Date:** 2026-09-27
**Baseline:** `2026-05-18-face-swapper-landscape.md` (swappers) and
`2026-05-16-infiniteyou-asset-report.md` (injection adapters).
**Sources:** 22 retrieved; key ones are FaceFusion releases [1], the AlphaFace
paper and repo [2][3], the VisoMaster-Fusion AlphaFace doc [4], Qwen-Image-2.1
[5][6], FLUX.2 [klein] [7][8], ComfyUI-PuLID-Flux2 [9], WithAnyone [10], UMO
[11], DreamID-V [12], and the persistent-identity benchmark [13].

---

## Executive Summary

No new generation of dedicated face-transfer models has shipped. The one new
open swapper is **AlphaFace** (paper Jan 2026, added to FaceFusion 3.9.0 on
2026-09-03). It is still a 256px GAN-style one-shot swapper, like HyperSwap,
and its gains are pose robustness, not resolution. Identity adapters in the
PuLID / InfiniteYou style have stalled: InfiniteYou is still v1.0 on FLUX.1,
and PuLID reached FLUX.2 only through a community port. **Most of the effort
has moved into general multi-reference edit models.** FLUX.2 [klein], Qwen-Image
2.0 → 2.1, Nano Banana 2, GPT-Image 2.x and Seedream 5 all sell "character
consistency" as a built-in feature, so there's no separate identity module. The
open leader is **Qwen-Image-2.1** (7B, open weights 2026-09-20, research-only
license). The only commercially clean multi-reference option is **FLUX.2 [klein]
4B** (Apache-2.0). Research has shifted from "raise ArcFace similarity" toward
avoiding copy-paste faces, handling multiple identities, keeping the whole
person consistent, and staying stable over repeated edits. A September 2026
benchmark finds identity preservation is still a distinct weakness of the
foundation models.

## Key Findings

### Swappers: one new entrant (AlphaFace), no resolution jump

FaceFusion released nine versions between March and September 2026. Only 3.9.0
(2026-09-03) added a swapper, `alphaface_256`, along with a new landmarker
(`hrffa`) [1]. Every other release was infrastructure, and 3.6.0 (2026-03-16)
added only age-modifier and background-remover models [1]. HyperSwap has had no
new tiers since our May survey.

AlphaFace (Yu et al., arXiv 2601.16429) is a 256×256 real-time swapper. It adds
a cross-adaptive identity-injection module and trains with contrastive losses
against CLIP and VLM text, using InternVL3-14B captions of the faces [2]. On
FF++ its ID retrieval of 98.77 is about the same as FaceDancer's 98.84. Its
advantage is geometry: pose error 1.24 vs 2.04, expression error 2.03, and
24.1 ms per image. On the extreme-pose MPIE set it has the best cosine (0.471)
and the lowest pose and expression errors [2]. The paper does not compare
against inswapper or HyperSwap, so how it ranks against our current default is
unknown. The official repo is MIT, ships training code, has about 73 stars, and
hosts weights on Google Drive [3]. VisoMaster-Fusion already integrates it as a
~529 MB ONNX that uses the same W600K ArcFace encoder as inswapper, which means
no new recognition model is needed. It notes visible "frame popping" on
near-profile faces, caused by its multi-template alignment [4]. The license is
**contested**: the repo says MIT [3], but a FaceFusion guide calls the model
non-commercial [14].

InsightFace kept its better swappers closed (`inswapper_cyn`, `inswapper_dax`,
served through Picsi.ai). Commercial use of the inswapper series still requires
contacting them [15]. The original Roop repo was archived in March 2026 [16].
ComfyUI-ReActor now loads HyperSwap 1a/1b/1c from `models/hyperswap/`, and its
`ReActorSetWeight` node sets swap strength [17]. We found no AlphaFace support
in ReActor.

### Identity adapters: stalled, and ported by the community

InfiniteYou is still at FLUX v1.0 (`aes_stage2` / `sim_stage1`). We found no v2
[18]. PuLID on FLUX.2 exists only as **ComfyUI-PuLID-Flux2** by one developer
(iFayens, MIT, about 124 stars). It is active from 2026-03-16 to 2026-05-21. It
runs old PuLID weights, plus "native" Klein v1/v2 weights marked "in progress",
and its training scripts were pulled for instability [9][19]. The big labs'
newer identity work targets different problems. **WithAnyone** (ICLR 2026,
FLUX-based, code and weights released) introduces a copy-paste metric
(MultiID-Bench). It reports higher similarity to ground truth than PuLID with a
much lower copy-paste score: roughly 0.65 vs 0.62 similarity, −0.15 vs 0.25
copy-paste [10]. **UMO** (ByteDance, CVPR 2026, Apache-2.0, ComfyUI workflows)
is a matching-reward fine-tune that reduces identity confusion across multiple
people in UNO and OmniGen2 [11]. Academic papers from July 2026 either remain on
SD1.5 with ArcFace+CLIP adapters (Diff-ID, which still trails InstantID on face
similarity at 72.7 vs 75.1 [20]) or move to whole-person consistency beyond the
face (DBS with the Pexels-100 benchmark [21]).

### Edit and multi-reference foundation models absorbed the use case

FLUX.2 [klein] (released 2026-01-15) combines generation and editing. It takes
multiple reference images with "strong identity and layout preservation", and
FLUX.2 overall supports up to 10 references [7][22]. **Klein 4B is Apache-2.0.**
Klein 9B and FLUX.2 dev use BFL's non-commercial license [8]. Qwen-Image-Edit-2511
(2025-12-26) improved face identity under pose and style changes [23]. Alibaba
then replaced the 20B edit line with **Qwen-Image 2.0** (7B, unified
generation and editing, 2026-02-10). **Qwen-Image-2.1** followed as open weights
on 2026-09-20: 10 references, a group photo generated from six portrait
references, native RGBA, and native ComfyUI support through Comfy-Org repacks.
It is under the **Qwen Research License** (commercial use needs a separate
agreement), needs about 24 GB+ VRAM, and its example code uses 40 steps [5][6].
On Qwen's own benchmark it scores 60.28: first among open models, ahead of
FLUX.2 Max (55.33), and behind the closed GPT Image 2.5 (67.01) [5]. This is
single-source and self-reported. Z-Image-Edit, which would directly suit our
Z-Image stack, is still "to be released" [24].

The closed models lead. Nano Banana 2 (February 2026) claims resemblance for up
to 4–5 characters from up to 14 references [25]. The persistent-identity
benchmark (arXiv 2609.04151, 2026-09-03) tests GPT-Image-2 and NB2
reference-in-context approaches against LoRA and a dedicated identity layer.
It concludes that identity preservation "remains a distinct limitation of
current generative foundation models" and degrades under repeated edits, small
subjects and multiple subjects [13]. That paper is from the vendor of the
identity-layer product it favours (PHOTA), so its ranking is **single-source
and interested**.

### Video: diffusion swapping matured on Wan

DreamID-V (ByteDance, ECCV 2026 Oral, Apache-2.0) added a faster Wan-1.3B
variant (2026-01-12) and a DWPose option. Two community ComfyUI node packs
exist [12]. Stand-In (WeChatCV, CVPR 2026) and Lynx (ByteDance, CVPR 2026, an
ArcFace resampler plus a reference adapter on a DiT) cover identity-preserving
video generation [26][27]. Diffusion face swapping is real in 2026, but in
*video* models, not as a still-image swapper.

## Comparison

| Option | Kind | New since May? | Res / size | License | ComfyUI |
|---|---|---|---|---|---|
| inswapper_128 | swapper | no | 128 | non-commercial weights | ReActor |
| HyperSwap 1a/1b/1c | swapper (our default) | no | 256 | ResearchRAIL-MS | ReActor |
| **AlphaFace** | swapper | **yes** (FF 3.9.0, Sep 3) | 256, ~529 MB ONNX | MIT (repo) / NC per FF guide — contested | none in ReActor found |
| PuLID-Flux2 (community) | adapter | yes (Mar–May) | FLUX.2 Klein/dev | MIT code | yes |
| InfiniteYou v1.0 | adapter | no | FLUX.1 | — | official |
| WithAnyone | adapter (FLUX) | paper Oct 2025, ICLR 2026 | FLUX.1 | not stated | not found |
| UMO | reward fine-tune | no (Sep 2025) | UNO / OmniGen2 | Apache-2.0 | workflows |
| **FLUX.2 [klein] 4B** | multi-ref edit | Jan 2026 | 4B | **Apache-2.0** | native |
| **Qwen-Image-2.1** | multi-ref edit | **yes** (Sep 20) | 7B + 8B VL enc | Qwen Research | native |
| Z-Image-Edit | edit | still unreleased | 6B | Apache-2.0 (family) | — |
| DreamID-V | video swap | faster variant Jan | Wan 1.3B | Apache-2.0 | community |

## What this means for the photobooth

For us, AlphaFace is the one drop-in candidate. It shares the W600K ArcFace
encoder with inswapper, so it can probably be wrapped in `swap_core` the way
HyperSwap was, and it runs on our existing 20-doll bake-off harness
(`scripts/swapper_bakeoff.py`). Our May conclusion that the ~0.86 id_cos ceiling
comes from the target face (the small painted doll face), not the swapper,
predicts only a small gain. The license question has to be settled before any
product use.

Moving identity into generation now means a multi-reference edit model, not a
PuLID-style adapter. FLUX.2 klein 4B is the license-clean candidate. Qwen-Image-2.1
is the stronger candidate but research-only and far slower than Z-Image Turbo at
6 steps. Either would replace the separate swap stage, which the May survey
could only point toward. None of the sources measure identity on stylized or
painted targets, which is our hardest case (PuLID failed it outright).

## Open Questions

- AlphaFace vs HyperSwap 1c / inswapper on our dolls: no published head-to-head;
  only our own bake-off can answer it.
- The license on AlphaFace's weights: MIT repo vs "non-commercial" in a FaceFusion guide.
- Whether FLUX.2 klein or Qwen-Image-2.1 multi-reference keeps a real person's
  identity when rendering into the matryoshka style. No source tests stylized
  targets.
- Qwen-Image-2.1's benchmark is Qwen's own. No independent identity-specific
  numbers for FLUX.2 or Qwen-2.1 were found.
- The Z-Image-Edit release date.

## Sources

[1] FaceFusion. "Releases". https://github.com/facefusion/facefusion/releases (Retrieved: 2026-09-27)
[2] Yu et al. "AlphaFace: High Fidelity and Real-time Face Swapper Robust to Facial Pose". https://huggingface.co/papers/2601.16429 (Retrieved: 2026-09-27)
[3] andrewyu90. "Alphaface_Official". https://github.com/andrewyu90/Alphaface_Official (Retrieved: 2026-09-27)
[4] VisoMaster-Fusion. "alphaface.md". https://github.com/VisoMasterFusion/VisoMaster-Fusion/blob/main/docs/alphaface.md (Retrieved: 2026-09-27)
[5] MarkTechPost. "Alibaba Qwen Releases Qwen-Image-2.1". https://www.marktechpost.com/2026/09/21/alibaba-qwen-releases-qwen-image-2-1/ (Retrieved: 2026-09-27)
[6] Qwen. "Qwen-Image-2.1" model card. https://huggingface.co/Qwen/Qwen-Image-2.1 (Retrieved: 2026-09-27)
[7] Black Forest Labs. "FLUX.2 [klein]: Towards Interactive Visual Intelligence". https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence (Retrieved: 2026-09-27)
[8] Black Forest Labs. "FLUX.2-klein-4B" model card. https://huggingface.co/black-forest-labs/FLUX.2-klein-4B (Retrieved: 2026-09-27)
[9] iFayens. "ComfyUI-PuLID-Flux2". https://github.com/iFayens/ComfyUI-PuLID-Flux2 (Retrieved: 2026-09-27)
[10] "WithAnyone: Towards Controllable and ID Consistent Image Generation". https://huggingface.co/papers/2510.14975 (Retrieved: 2026-09-27)
[11] ByteDance. "UMO". https://github.com/bytedance/UMO (Retrieved: 2026-09-27)
[12] ByteDance. "DreamID-V". https://github.com/bytedance/DreamID-V (Retrieved: 2026-09-27)
[13] "Persistent Identity Preservation in Generative Image Models: A Benchmark and Evaluation System". https://arxiv.org/abs/2609.04151 (Retrieved: 2026-09-27)
[14] Magic Hour. "FaceFusion 3.9.0 guide". https://magichour.ai/blog/how-to-use-facefusion (Retrieved via search snippet: 2026-09-27)
[15] InsightFace. "Enterprise Face Swap Licensing". https://www.insightface.ai/solutions/face-swapping (Retrieved via search snippet: 2026-09-27)
[16] WaveSpeed. "Open Source Face Swap Software (2026)". https://wavespeed.ai/blog/posts/open-source-face-swap-software/ (Retrieved via search snippet: 2026-09-27)
[17] Gourieff. "ComfyUI-ReActor". https://github.com/Gourieff/ComfyUI-ReActor (Retrieved via search snippet: 2026-09-27)
[18] ByteDance. "InfiniteYou". https://github.com/bytedance/InfiniteYou (Retrieved via search snippet: 2026-09-27)
[19] iFayens. "ComfyUI-PuLID-Flux2 commits". https://github.com/iFayens/ComfyUI-PuLID-Flux2/commits/main (Retrieved: 2026-09-27)
[20] "Diff-ID: Identity Consistent Facial Image Generation and Morphing via Diffusion Models". https://arxiv.org/html/2607.25078v1 (Retrieved: 2026-09-27)
[21] "Beyond Facial Consistency: Personalized Person Image Generation with Holistic Identity Preservation". https://arxiv.org/abs/2607.25622 (Retrieved: 2026-09-27)
[22] Black Forest Labs. "FLUX.2 Image Editing". https://docs.bfl.ml/flux_2/flux2_image_editing (Retrieved via search snippet: 2026-09-27)
[23] Qwen. "Qwen-Image-Edit-2511". https://huggingface.co/Qwen/Qwen-Image-Edit-2511 (Retrieved via search snippet: 2026-09-27)
[24] Tongyi-MAI. "Z-Image". https://github.com/Tongyi-MAI/Z-Image (Retrieved: 2026-09-27)
[25] Sunra. "Nano Banana 2 Review". https://sunra.ai/blog/nano-banana-2-review (Retrieved via search snippet: 2026-09-27)
[26] WeChatCV. "Stand-In". https://github.com/WeChatCV/Stand-In (Retrieved via search snippet: 2026-09-27)
[27] "Lynx: Towards High-Fidelity Personalized Video Generation". https://arxiv.org/html/2509.15496v1 (Retrieved via search snippet: 2026-09-27)
