---
status: archived
topic: archived-threads
summary: FaRL-B/16 trained on LAION-Face is the primary face encoder for attribute preservation; ArcFace should be auxiliary only as it structurally discards expression, attire, and gaze.
---

# Attribute-Preserving Face Encoders for the Detective-Puzzle Probe

**Date:** 2026-04-16
**Context:** vamp-interface is pivoting from identity/uncanny to co-discovery (Wordle-shaped daily puzzle). The pipeline is `qwen-text → learned projection P → Flux.1-dev → face image → Ψ → attribute probe → predicted axis labels`. We need Ψ to preserve editorial axes (formal/casual, office/warehouse, warm/hollow smile, direct/evasive gaze), not identity. ArcFace IR101 is our current baseline and is probably wrong for this task.

---

## 1. TL;DR

**Use FaRL-B/16 (LAION-Face) as the primary Ψ, and ensemble with DINOv2-L/14 if compute allows.** FaRL is specifically a CLIP-ViT fine-tuned on 20M face-text pairs and is the only publicly released encoder whose published CelebA mAcc is measured in the exact regime we need (linear probe on frozen features, small training data). ArcFace should stay only as an auxiliary signal, because its margin loss structurally discards expression/pose/attire — the axes we want the puzzle to plant.

---

## 2. Ranked table

| Encoder | Dim | Public weights | CelebA 1%/10%/100% mAcc (linear probe) | Face-specific? | License | Our use |
|---|---|---|---|---|---|---|
| **FaRL-B/16 (LAION-Face)** | 512 (CLS) | HF + GitHub (FacePerceiver/FaRL) | **89.66 / 90.99 / 91.39** | Yes (face-text) | MIT | **Primary Ψ** |
| **CLIP ViT-B/16 (OpenAI)** | 512 | HF (openai/clip-vit-base-patch16) | 89.09 / 90.47 / 90.86 | No | MIT | Fallback |
| **OpenCLIP ViT-H/14 (LAION-2B)** | 1024 | HF (laion/...) | not published on CelebA; ImageNet lin-probe >84% | No | MIT | Heavy alternative |
| **DINOv2 ViT-L/14** | 1024 | HF (facebook/dinov2-large) | no published CelebA lin-probe number; ImageNet lin-probe 86.3% | No | Apache-2.0 | Ensemble member |
| **DINOv3 ViT-L/16** | 1024 | HF (facebook/dinov3-vitl16) released Aug 2025 | not published on CelebA | No | Meta research license (check) | Swap for DINOv2 if license permits |
| **DreamSim (ensemble + LoRA)** | ~2300 concat of CLIP/OpenCLIP/DINO with LoRA | pip `dreamsim` | not benchmarked on CelebA; wins NIGHTS mid-level similarity at 96.2% | No (holistic) | MIT | Stacking layer |
| **FLIP (FLIP-80M, ACM MM 2024)** | 512 (CLIP-base backbone) | GitHub; weights partially released | paper claims SOTA on 10 face tasks; exact CelebA mAcc not surfaced in abstract | Yes | check repo | Candidate successor to FaRL |
| **ArcFace IR101 (WebFace4M)** | 512 | HF minchul/cvlface_... | no published CelebA lin-probe; intra-entropy analysis shows low sensitivity to transient attributes (smile, mouth-open) | Yes (identity) | MIT | Keep as ID sanity signal only |
| **MagFace R100** | 512 | GitHub IrvingMeng/MagFace | magnitude encodes quality, not attribute; CelebA lin-probe not published | Yes (identity) | MIT | Skip |
| **FairFace ResNet-34** | 512 (penultimate) | GitHub joojs/fairface | age/race/gender heads, not general-purpose | Yes (3 attrs) | BSD-4 | Age/race/gender probe only |
| **EmoNet** | ~256 | GitHub face-analysis/emonet | 5/8 emotion classes + valence/arousal; Nature MI 2021 | Yes (emotion) | non-commercial | Expression axis only |

Sources: FaRL numbers from Table 2(c) of arXiv:2112.03109 (paper mAcc delta +0.5 over CLIP at all budgets). DINOv2 ImageNet from arXiv:2304.07193. ArcFace attribute-geometry findings from arXiv:2507.11372 ("Attributes Shape the Embedding Space"). DreamSim from arXiv:2306.09344 / NeurIPS 2023. FLIP-80M from OpenReview FHguB1EYYi / ACM MM 2024.

---

## 3. Per-encoder: what it captures, what it discards

**ArcFace IR101.** Trained with angular-margin identity loss on WebFace4M. By construction it collapses intra-identity variation — exactly the editorial axes we're planting. Empirical analysis (arXiv:2507.11372) shows ArcFace is *most* invariant to "contrast" and "illumination," and *least* invariant to "head angle" and "age" — so pose and age may partially survive, but transient attributes (smile, gaze, attire) are heavily compressed. Attributes "deterministically linked to identity" (gender, baldness) show strong structural influence; "smiling" and "mouth slightly open" leak very little information into the embedding. Verdict: wrong tool for the detective puzzle. Keep it only as a control/identity anchor.

**CLIP ViT-B/16 (OpenAI).** Web-scale image-text contrastive. Its embedding space has been shown to encode editorial/descriptive vocabulary well. On CelebA attribute classification with 100% training data and a linear probe, CLIP hits 90.86 mAcc. It is the strongest general-purpose baseline. Weakness: no face-specific inductive bias — expression nuances may be under-represented.

**OpenCLIP ViT-H/14 (LAION-2B).** Same objective, bigger backbone, bigger data. Expected to dominate CLIP-B on category-like editorial axes (formal/casual, office/warehouse) because these are exactly the kinds of descriptions that appear in web alt-text. Heavier (1024-d, ~632M params). A good ensemble member.

**FaRL-B/16.** CLIP ViT-B/16 further pre-trained on LAION-Face (20M face-text pairs) with image-text contrastive + masked image modeling. On CelebA few-shot linear probe, FaRL beats CLIP at every budget (1%: 89.66 vs 89.09; 10%: 90.99 vs 90.47; 100%: 91.39 vs 90.86). The lift is small in absolute terms but consistent, and it is the only face-specialized encoder with published linear-probe numbers in our exact regime. Also wins LaPa face parsing (F1 92.32 vs CLIP 92.21) and AFLW-19 alignment. Should capture expression and pose well because the masked-image modeling forces it to model face geometry.

**DreamSim.** NeurIPS 2023 spotlight. Ensemble of CLIP + OpenCLIP + DINO with LoRA adaptation on ~20k human triplet judgments. Designed for mid-level similarity (pose, layout, foreground color, shape). Not face-specialized, but the training objective directly rewards holistic "these two images feel similar" judgments — which is close to what our probe needs for drift axes (warm vs hollow smile). No published CelebA number. High dimensionality (~2300 concat). MIT license, `pip install dreamsim`.

**DINOv2 / DINOv3.** Self-supervised ViT. Strong dense features, proven to linear-probe well on general vision tasks. DINOv3 (Aug 2025) improves on DINOv2 on segmentation and depth. No published face-attribute linear-probe number — would be our own experiment. Likely preserves pose, attire, and scene context better than ArcFace and competitively with CLIP. Good ensemble member.

**MagFace.** Same margin family as ArcFace, plus magnitude-encodes face quality. Magnitude does not encode attribute information in any usable way. Skip.

**FairFace.** ResNet-34 trained as age/race/gender classifier. Penultimate-layer features are useful as a *task-specific* probe for those three axes only. If the detective puzzle plants age or ethnicity, FairFace's own prediction head is likely stronger than a probe on ArcFace or even FaRL.

**AffectNet / EmoNet.** Same logic for expression: use the head directly rather than probing a frozen backbone. EmoNet emits 5 or 8 classes plus continuous valence/arousal.

**FLIP-80M (ACM MM 2024 Oral).** 80M face-text pairs — largest face-text corpus to date, roughly 4× LAION-Face. Paper claims SOTA on 10 face tasks including face attribute classification. Weights partially released on GitHub (ydli-ai/FLIP). If weights work as frozen encoder, this is the obvious successor to FaRL. Worth a one-hour reproduction check before committing.

---

## 4. Ensemble / stacking notes

DreamSim's own ablation confirms that CLIP + OpenCLIP + DINO concatenated beats any single one on mid-level similarity. The natural ensemble for our task is:

- **Tier 1 (cheap):** FaRL-B/16 (512-d) + DINOv2-L/14 (1024-d) concatenated = 1536-d. Fits a linear probe comfortably on ~500 items.
- **Tier 2 (richer):** add OpenCLIP ViT-L/14 (768-d) → 2304-d. Use shallow MLP (1 hidden layer, 256 units) with weight decay to avoid overfitting 500 items × 2304 dims.
- **Task-specific heads in parallel:** FairFace age/race/gender and EmoNet expression, feeding their logits as 3+5+8 = 16 extra features. This bypasses the probe entirely for axes where a pretrained classifier already exists.

ArcFace IR101 should be *in* the concat as a control: if the probe accuracy doesn't change when ArcFace is ablated out, that confirms identity features don't carry the axis signal, which is itself a useful empirical claim.

---

## 5. What we'd have to measure ourselves

No published result covers our exact task: recover 3–4 planted editorial axes × 3 levels on ~500 synthetic Flux-generated faces with a linear probe. Minimal experiment:

1. **Fix the corpus:** the detective-corpus stage already produces (axis_label, level, image) triples. Hold out 20% as test; train on 80%.
2. **Extract features** with each of {ArcFace IR101, CLIP-B/16, OpenCLIP-H/14, FaRL-B/16, DINOv2-L/14, DreamSim, FairFace-penult, EmoNet-penult}. One forward pass per encoder per image. ~8 × 500 = 4000 forward passes — a few minutes on one GPU.
3. **Train one linear probe per (encoder, axis)** with 5-fold CV on the 400-item train split. Report per-axis accuracy on the 100-item test split and 95% bootstrap CI.
4. **Ensemble:** concat top-3 encoders; re-run probe. Report lift over best single.
5. **Decision rule:** pick the encoder (or ensemble) that maximizes mean per-axis accuracy with ties broken by simplicity (512-d beats 2304-d at equal accuracy).

Expected outcome based on priors: FaRL beats ArcFace by ≥10 points on editorial axes; DINOv2 is within 3 points of FaRL; ensemble gains another 2–4 points; FairFace/EmoNet heads dominate on their specific axes. If FaRL does *not* beat ArcFace by a clear margin on this corpus, that is itself important — it would mean Flux does not modulate the editorial axes strongly enough for any encoder to pick them up, and the bottleneck is the generator, not the readout.

---

## Sources

- [General Facial Representation Learning in a Visual-Linguistic Manner (FaRL, CVPR 2022)](https://arxiv.org/abs/2112.03109)
- [FaRL ar5iv HTML with Table 2(c)](https://ar5iv.labs.arxiv.org/html/2112.03109)
- [FacePerceiver/FaRL GitHub](https://github.com/FacePerceiver/FaRL)
- [Attributes Shape the Embedding Space of Face Recognition Models (arXiv:2507.11372)](https://arxiv.org/html/2507.11372v1)
- [DreamSim GitHub](https://github.com/ssundaram21/dreamsim)
- [DreamSim NeurIPS 2023 paper](https://arxiv.org/html/2306.09344v3)
- [DINOv2 paper](https://arxiv.org/html/2304.07193v2)
- [DINOv3 paper (2025)](https://arxiv.org/html/2508.10104v1)
- [FLIP-80M OpenReview](https://openreview.net/forum?id=FHguB1EYYi)
- [ydli-ai/FLIP GitHub](https://github.com/ydli-ai/FLIP)
- [Foundation vs Domain-specific Models in Face Recognition (arXiv:2507.03541)](https://arxiv.org/html/2507.03541v2)
- [FairFace WACV 2021](https://openaccess.thecvf.com/content/WACV2021/papers/Karkkainen_FairFace_Face_Attribute_Dataset_for_Balanced_Race_Gender_and_Age_WACV_2021_paper.pdf)
- [EmoNet GitHub](https://github.com/face-analysis/emonet)
- [MagFace paper](https://ar5iv.labs.arxiv.org/html/2103.06627)
- [Arc2Face ECCV 2024](https://arc2face.github.io/)
