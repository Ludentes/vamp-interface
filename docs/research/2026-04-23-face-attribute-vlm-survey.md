---
status: live
topic: metrics-and-direction-quality
summary: Survey of VLM / image-encoder options for zero-shot face-attribute drift scoring. Recommends SigLIP-2 So400m/16 as drop-in upgrade over OpenCLIP ViT-L/14; keep FaRL as a face-specific fallback if probes still underperform.
---

# Research: Face-attribute VLM survey (replacement for OpenCLIP ViT-L/14)

**Date:** 2026-04-23
**Sources:** 10 primary, curated from 16 search-result pages. Focus: zero-shot face-attribute scoring in service of drift measurement on Flux DiT portraits.

---

## Executive Summary

OpenCLIP ViT-L/14 is a weak baseline for continuous face-attribute scoring — our own smoke on 6 hand-picked images showed bearded/clean-shaven margins separated by only ~0.05, and one case (`asian_m` at beard scale=1.0 vs `elderly_latin_m` bearded baseline) actually inverted. Two classes of upgrade are available. The drop-in generalist upgrade is **SigLIP-2 So400m/16** — same text-embedding probe interface, ~8-point ImageNet zero-shot improvement over OpenCLIP L/14 (82.5% vs 74.0%) [1][2], and HuggingFace-native. The face-specialist upgrade is **FaRL** — a CLIP-style model pretrained on 20M LAION-Face pairs that outperforms CLIP on CelebA-40 attribute classification while using 20× less data (91.39% vs 90.86% mean accuracy on CelebA, linear-probe evaluation) [3]. VLMs (LLaVA/Qwen-VL/InternVL) are a different tool entirely — they produce generated text for VQA-style queries, which recent editing-eval work identifies as error-prone on subtle attributes and prone to hallucination [4]. DINOv2 has the strongest features but no text interface, so it requires supervised probe training per attribute — an engineering cost we don't want for ad-hoc drift measurement. **Recommendation: swap OpenCLIP ViT-L/14 → SigLIP-2 So400m/16 for the probe scorer; if face-specific probes still underperform after the swap, try FaRL-B as a face-specialist sidecar.**

## Key Findings

### SigLIP-2 is the correct drop-in generalist upgrade

SigLIP-2 was released 2025-02-20 and is integrated into HuggingFace Transformers [5]. It inherits the SigLIP sigmoid loss but adds captioning-based pretraining, self-supervised losses (self-distillation, masked prediction), and online data curation. On ImageNet-1k zero-shot the L/16 variant reaches 82.5% versus OpenCLIP L/14's 74.0%, and So400m/16 reaches 83.4% [1]. The same paper reports COCO text-to-image recall@1 improvements of ~4 points over SigLIP at comparable scale [1]. Available checkpoints span ViT-B/86M, L/303M, So400m/400M, g/1B — the So400m variant is the sweet spot at our scale (single-GPU inference comparable to ViT-L/14) [1][5]. The API is identical in shape to OpenCLIP — both produce normalised text/image embeddings, both support the same logit-margin probe pattern we already use. Integration via `transformers.AutoModel.from_pretrained("google/siglip2-so400m-patch16-384")` is one function call.

MetaCLIP 2 (July 2025) edges SigLIP-2 by +0.7% on ImageNet zero-shot at ViT-H/14 scale [6], but the gap is marginal and MetaCLIP is primarily a data-curation paper, not an architectural advance. SigLIP-2 remains the better integration target because of HuggingFace-native support and explicit NaFlex variants that handle non-square inputs — useful if we later probe on non-standard crops. EVA-CLIP L/14 reports 80.4% ImageNet zero-shot [7], strong but below SigLIP-2 L/16; its main advantage was parameter efficiency at 2023 scale, now superseded.

### FaRL is the face-specialist option, with a small numeric win on CelebA

FaRL (CVPR 2022) extends CLIP pretraining with two face-specific terms — Masked Image Modeling and an ALIGN head that crops/aligns faces to 224×224 before encoding [8]. Trained on 20M LAION-Face pairs, it achieves 91.39% mean accuracy on CelebA-40 attributes with a frozen backbone (91.88% fully fine-tuned) versus CLIP's 90.86% under identical evaluation protocols — both using ViT-B/16 at 87M parameters [3]. The 0.5-point gap is small but directionally consistent with pretraining on face-domain data, and the paper attributes the advantage primarily to the ALIGN component — face-alignment before encoding is what matters most on CelebA [3]. FaRL code and weights are public at github.com/FacePerceiver/FaRL [8].

**Caveat**: the 91.39 number is *linear-probe*, not zero-shot text-probe. For our usage (text-pair margins with no supervised training) FaRL's face-conditioned tokenizer/captions should help, but the published numbers don't bracket that use case directly. A face-pretrained encoder plus our existing text prompts is plausibly better than a generic one, but this is not quantified in the literature.

### VLMs (LLaVA / Qwen-VL / InternVL) are wrong-shape for drift measurement

Modern VLMs can answer "does this person have a beard?" directly, and on common VQA benchmarks Qwen-VL and InternVL achieve state-of-the-art results [9][10]. Two reasons they don't fit our task. First, drift measurement needs a **continuous scalar** per attribute so we can take `Δmargin / Δscale` and integrate across the scale sweep. VLM generation is categorical text — converting it to a scalar requires log-probability extraction on yes/no tokens, which is implementable but noisy and tool-chain-heavy compared to the single `encode_text @ encode_image.T` matmul we already do. Second, recent image-editing evaluation work explicitly flags VLM-VQA for "struggling to discern subtle features, leading to less precise consistency assessments" and susceptibility to hallucinations on attribute-preservation judgments [4]. Edit-induced drift IS subtle-feature change — that's where VQA is weakest. Third, a 7B+ parameter VLM forward pass per probe per image is ~30× the cost of a 400M encoder; with 1,536 PNGs × 12 probes in our current scorer, that matters.

InternVL in embedding-only mode (e.g. `Qwen3-VL-Embedding-8B` [11]) avoids the generation-cost problem but at that parameter budget and with less evaluation on face-specific zero-shot tasks, we're buying complexity for uncertain gain.

### DINOv2 is great features without a text interface

DINOv2 is self-supervised on 142M images and produces the strongest frozen-backbone features in benchmark after benchmark [12]. For our use case its fatal limitation is the lack of any text side — to score "bearded" vs "clean-shaven" we would need to train a linear head per attribute on labeled face data, introducing a supervised-training step per probe. CelebA-40 would be the natural training set; the 2018 facial-attributes literature suggests a linear probe on modern self-supervised features can hit 90%+ on most CelebA attributes [13], but this is per-probe engineering work rather than a one-liner. DINOv2 remains useful as an **identity-drift** encoder (cosine on `[CLS]` embedding between edited-vs-reference), which is a separate drift channel not covered by attribute probes — worth considering alongside ArcFace for that specific job, but not a replacement for the CLIP-style scorer.

### What the editing-eval literature actually uses

Recent image-editing and VLM-guided editing papers (Qwen-Edit+, TBStar-Edit) use a mix: face-embedding constraints (ArcFace-style) for identity preservation, CLIP-style probes for attribute adherence, and VLM-VQA as a final *qualitative* gate but not the continuous signal [4]. This triangulates what we should do: keep CLIP-style probes as the *continuous* drift metric, layer ArcFace for identity, and optionally add a VLM as a categorical sanity-check on a sample.

## Comparison

| Model | Text interface | ImageNet zero-shot | Face-task claim | Params | Drop-in for our probe? |
|---|---|---|---|---|---|
| OpenCLIP ViT-L/14 (current) | yes | 74.0% [2] | none | 304M | — |
| SigLIP-2 So400m/16 | yes | 83.4% [1] | none (generalist) | 400M | yes, one line |
| SigLIP-2 L/16 | yes | 82.5% [1] | none | 303M | yes, one line |
| MetaCLIP 2 ViT-H/14 | yes | ~84% [6] | none | 632M | yes, ~similar |
| EVA-CLIP L/14 | yes | 80.4% [7] | none | 428M | yes |
| FaRL ViT-B/16 | yes | n/a (face-specific) | CelebA linear-probe 91.39% [3] | 87M | yes, face-specialist |
| DINOv2 ViT-L/14 | no | n/a (self-sup) | strongest linear-probe features [12] | 304M | no — needs probe training per attribute |
| InternVL / Qwen-VL / LLaVA | VQA text-gen | n/a | SOTA on VQA benchmarks [9][10] | 7B–26B | no — different tool shape |

## Open Questions

- **No direct SigLIP-2 zero-shot numbers on CelebA**: the SigLIP-2 paper does not report CelebA attribute probes. The 8-point ImageNet win is a strong prior but not a guarantee on face-specific fine-grained attributes.
- **No head-to-head FaRL vs SigLIP-2**: FaRL's published comparisons are against CLIP (ViT-B/16, 2021), not against the SigLIP family. It's plausible that SigLIP-2 L/16 matches or beats FaRL on CelebA attributes simply by parameter count and training-data scale even without face-specific pretraining, but we don't have that number.
- **VLM embedding models as a sidecar**: `Qwen3-VL-Embedding-8B` was published 2025 and is an embedding-only VLM, avoiding the VQA-generation problem. If face-attribute probes still underperform after both SigLIP-2 and FaRL, this is a plausible third stop — but it's a 20× jump in compute and has no published face-attribute benchmark numbers [11].

## Concrete next step (for this project)

Swap `MODEL_NAME = "ViT-L-14"` / `PRETRAINED = "openai"` in `src/demographic_pc/score_clip_probes.py` for SigLIP-2 So400m/16 via HuggingFace Transformers. Re-run the hand-picked smoke set of 6 images and compare margin magnitudes. If the bearded-vs-clean gap widens from ~0.05 to >0.15 (on the same probe pairs, same test set) the switch is justified on smoke evidence alone — proceed to full-corpus scoring. If gains are marginal, add FaRL as a sidecar face-encoder and blend margins.

## Sources

[1] Zhai et al. "SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features". arXiv:2502.14786 (Feb 2025). https://arxiv.org/html/2502.14786v1 (Retrieved: 2026-04-23)
[2] LAION. "Reaching 80% zero-shot accuracy with OpenCLIP". https://laion.ai/blog/giant-openclip/ (Retrieved: 2026-04-23)
[3] Zheng et al. "General Facial Representation Learning in a Visual-Linguistic Manner". CVPR 2022. https://ar5iv.labs.arxiv.org/html/2112.03109 (Retrieved: 2026-04-23)
[4] "Qwen-Edit+: Scaling Image Editing with VLM-Guided Consistency and Aesthetic Preference Distillation". https://www.researchgate.net/publication/403643556 (Retrieved: 2026-04-23)
[5] Google. "SigLIP 2 model card / HuggingFace integration". https://huggingface.co/docs/transformers/model_doc/siglip2 (Retrieved: 2026-04-23)
[6] "Meta CLIP 2: A Worldwide Scaling Recipe". arXiv:2507.22062 (July 2025). https://arxiv.org/pdf/2507.22062 (Retrieved: 2026-04-23)
[7] Sun et al. "EVA-CLIP: Improved Training Techniques for CLIP at Scale". arXiv:2303.15389. https://arxiv.org/pdf/2303.15389 (Retrieved: 2026-04-23)
[8] FacePerceiver team. "FaRL for Facial Representation Learning". https://github.com/FacePerceiver/FaRL (Retrieved: 2026-04-23)
[9] Chen et al. "InternVL: Scaling up Vision Foundation Models". CVPR 2024. https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_InternVL_Scaling_up_Vision_Foundation_Models_and_Aligning_for_Generic_CVPR_2024_paper.pdf (Retrieved: 2026-04-23)
[10] Qwen team. "Qwen-VL". https://huggingface.co/Qwen/Qwen-VL (Retrieved: 2026-04-23)
[11] Qwen team. "Qwen3-VL-Embedding-8B". https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B (Retrieved: 2026-04-23)
[12] Oquab et al. "DINOv2: Learning Robust Visual Features without Supervision". arXiv:2304.07193. https://arxiv.org/html/2304.07193v2 (Retrieved: 2026-04-23)
[13] Terhorst et al. "Facial Attributes: Accuracy and Adversarial Robustness". arXiv:1801.02480. https://arxiv.org/pdf/1801.02480 (Retrieved: 2026-04-23)
