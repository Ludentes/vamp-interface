---
status: live
topic: photobooth-sweep
---

# SAM3 vs Grounding DINO for masking a shawl

Quick survey, 2026-10-19. Task: a mask of one garment, the shawl, to drive
masked edits in ComfyUI. The Ming and Qwen editors shift the whole picture
unless the edit is masked.

## Verdict

Default to **SAM3**. It is one model doing text → masks directly. On LVIS it
reports 48.8 zero-shot mask AP vs a previous best of 38.5, and its paper uses
Grounding DINO + SAM1 as a baseline [1]. Both numbers are the SAM3 authors'
own; we found no independent head-to-head [1][2]. Native SAM3 nodes already
ship in both of our ComfyUI installs (`comfy_extras/nodes_sam3.py`). Weights
are 3.45 GB (`sam3.pt` / `sam3.safetensors`). A 1.75 GB SAM 3.1 Multiplex fp16
checkpoint also exists [3][4].

Grounding DINO is the older two-stage route: DINO boxes → SAM/SAM2 masks [5].
Its documented weak spot is exactly our case, fine-grained clothing. A fashion
detection paper reports it is misled by garment wording and confuses similar
garment styles [6]. Nothing we found tests shawls or scarves specifically.

Keep a **human-parsing fallback**. "Scarf" is a labelled class in the ATR and
LIP datasets [7]. ATR-fine-tuned SegFormer clothes models (B0/B2/B3) and the
FASHN SegFormer-B4 parser (scarf = class 10) segment it as a trained class,
not as an open-vocabulary guess [7][8]. A draped shawl is usually labelled as
"scarf" there, which is a closer fit than most text prompts.

## Where DINOv3 fits

DINOv3 (Meta, Aug 2025) is not a Grounding DINO successor. The shared name
covers two lineages: Grounding DINO is IDEA Research's detector built on the
DINO DETR architecture; DINOv3 is Meta's self-supervised image backbone [9].
It gives dense per-patch features, not boxes or masks. There are two ways to
use it for a shawl:

- **Zero-shot via dino.txt.** A text-aligned head for the distilled ViT-L/16
  supports open-vocabulary semantic segmentation [9][10]. It is trained with a
  largely global contrastive objective and predicts per 16-px patch, so masks
  are coarse. Follow-up work such as dinov3.seg adds a dedicated segmentation
  framework on top [11]. Expect it to be weaker than SAM3 for one garment
  instance with clean edges.
- **Few-shot, trained.** Freeze DINOv3, train a small head on ~20–50 labelled
  shawl masks from our own photos, and optionally refine edges with SAM point
  or box prompts. This is the strongest option for one fixed category in a
  fixed photo distribution, and it costs a small labelling job.

License: DINOv3 License, a custom licence. A third-party guide says commercial
use is allowed with the licence attached; unverified against the text [12].

**Order:** SAM3 first. Then the human-parsing scarf class. Then a DINOv3
few-shot head if both confuse the shawl with hair or the top.

## Things to test, not assume

- Prompt wording. Try "shawl", "scarf", "headscarf", "wrap", and use the `:1`
  cap to keep one instance (SAM 3.1 prompts are limited to 32 tokens) [3].
- A shawl over the hair or shoulders that is mistaken for hair or for a top.
  This is the classic human-parsing confusion, and probably SAM3's too.
- Stylized or painted images (doll renders): neither model is evaluated there.

## Sources

[1] Carion et al. "SAM 3: Segment Anything with Concepts". https://arxiv.org/html/2511.16719v2
[2] Pith review of SAM 3. https://pith.science/paper/2511.16719
[3] ComfyUI docs. "SAM 3.1: Segment Anything in ComfyUI". https://docs.comfy.org/tutorials/utility/video-segment-sam3
[4] Hugging Face file listings: https://huggingface.co/facebook/sam3 , https://huggingface.co/1038lab/sam3 (retrieved 2026-10-19)
[5] IDEA-Research. "Grounded-Segment-Anything". https://github.com/IDEA-Research/Grounded-Segment-Anything
[6] "Efficient Fine Tuning for Fashion Object Detection", Sensors 2023. https://pmc.ncbi.nlm.nih.gov/articles/PMC10346465/
[7] FASHN. "fashn-human-parser" and blog on fashion segmentation datasets. https://huggingface.co/fashn-ai/fashn-human-parser , https://fashn.ai/blog/fashion-segmentation-datasets-and-their-common-problems
[8] mattmdjaga. "segformer_b0_clothes". https://huggingface.co/mattmdjaga/segformer_b0_clothes
[9] Meta. "DINOv3" repo and paper. https://github.com/facebookresearch/dinov3 , https://arxiv.org/abs/2508.10104
[10] "DINO Soars: DINOv3 for Open-Vocabulary Semantic Segmentation of Remote Sensing Imagery". https://arxiv.org/html/2605.03175v1
[11] "dinov3.seg: Open-Vocabulary Semantic Segmentation with DINOv3". https://arxiv.org/html/2603.19531v1
[12] Roboflow. "DINOv3 Guide". https://blog.roboflow.com/train-dinov3/
