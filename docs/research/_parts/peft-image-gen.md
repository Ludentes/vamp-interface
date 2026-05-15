# Parameter-Efficient Fine-Tuning for Image Generators: What Carries Style?

Parameter-efficient fine-tuning (PEFT) emerged in NLP — LoRA, BitFit, adapters — as a way to adapt a frozen base model for a new task with a tiny trainable footprint. Image generation has imported these tools wholesale onto diffusion U-Nets and, more cautiously, onto GAN backbones. The literature now offers a fairly sharp answer for diffusion personalization (LoRA on cross-attention, ~1–3% of params, minutes of training), a partial answer for one-shot stylization on GANs (layer-swap and StyleGAN-NADA-style adaptation), and essentially nothing on encoder-warp-decoder talking-head networks like face-vid2vid or LivePortrait. Stitching these results together suggests a concrete minimum-parameter recipe for LP-family models, with the explicit caveat that nobody has published it yet.

## LoRA: the foundation

Hu et al. introduced LoRA at ICLR 2022 as a method to freeze the pre-trained weight matrix `W` and learn a low-rank update `ΔW = BA`, where `B ∈ R^{d×r}` and `A ∈ R^{r×k}` with `r ≪ min(d,k)` [1]. Against GPT-3 175B, LoRA reduced trainable parameters by 10,000× and GPU memory by ~3×, with quality on par with or above full fine-tune. The mechanism is generic: any linear layer can be wrapped. The Stable Diffusion community ported this almost immediately (cloneofsimo's `lora` repo [9]), and SD-LoRAs typically wrap attention `to_q`, `to_k`, `to_v`, `to_out`, sometimes also the feed-forward projections, occasionally the ResNet 1×1 convs. Typical rank is 4–32; trainable-parameter count for SD1.5 is ~1–10 MB versus the 3.4 GB base model.

Recent analysis tempers the "equivalent to full fine-tune" claim. Shuttleworth et al. found that LoRA and full fine-tune produce weight matrices with structurally different spectra: LoRA introduces new high-rank singular vectors ("intruder dimensions") absent from full fine-tune, and these correlate with worse generalization away from the training distribution [3]. The community confirmation on FLUX-DreamBooth is consistent: full fine-tune produces lower overfitting and better feature edits (hair color, beard, etc.), while LoRA bleeds style into adjacent concepts more readily [3]. The practical workaround is to fine-tune fully and then *extract* a LoRA from the delta — better than training LoRA from scratch.

## CustomDiffusion: cross-attention K/V is enough

Kumari et al.'s CustomDiffusion (CVPR 2023) made the load-bearing layer-localization finding for diffusion personalization: fine-tuning *only* the key and value projection matrices of the cross-attention layers — `W_K` and `W_V` that map text embeddings into the attention computation — is sufficient to learn a new concept from 4–20 images [2]. Their reported trainable share is roughly 3% of model parameters; storage per concept drops to ~75 MB; training takes ~6 minutes on 2× A100. The reasoning is geometric: cross-attention is the *only* site where text conditioning enters the U-Nets spatial features, so a new concept word needs only a new K/V mapping for that token. Self-attention, ResNet convs, and time-embedding MLPs all stay frozen.

CustomDiffusion's positive result is doubly informative for our question. It says (a) that style/identity information enters diffusion U-Nets through a small, named subspace, and (b) that *which subspace* depends on what you're trying to adapt. Personalization (a new noun) localizes to cross-attention K/V. Global style shift (an anime-look-everywhere edit) is empirically broader: SD-LoRAs trained for style typically need both attention and feed-forward layers, and pure cross-attention-only style LoRAs underperform.

## DreamBooth full-finetune vs. LoRA

Ruiz et al.'s original DreamBooth (CVPR 2023) was a *full* fine-tune of the U-Net (and sometimes text encoder) on 3–5 subject images with a class-preservation loss to avoid catastrophic forgetting. The community standard rapidly became DreamBooth-LoRA, which substitutes LoRA modules for full updates [4]. Trade-offs are stable across reports: full DreamBooth gives the highest fidelity and best generalization to unseen prompts but takes 10–30 minutes on an A100, occupies a full checkpoint, and risks catastrophic forgetting if regularization is weak. LoRA-DreamBooth is 2–5× faster, ships as ~5 MB, and is composable at inference — but suffers more concept-bleed and worse out-of-distribution prompt fidelity [3,4].

## BitFit and DiffFit: bias-only and bias-plus-scale

Ben-Zaken et al. (BitFit, ACL 2022) showed that on BERT, fine-tuning *only* the bias terms — well under 0.1% of parameters — is competitive with full fine-tune on small-to-medium GLUE tasks [5]. The hypothesis is that adaptation involves rescaling and shifting pre-trained features rather than learning new ones.

Xie et al. extended this to diffusion models as DiffFit (ICCV 2023): tune biases plus a small set of learnable per-block scaling factors `γ` (initialized to 1.0), targeted on DiT [6]. DiffFit reports superior FID to full fine-tune on ImageNet-class-conditional transfer with ~0.12% trainable parameters. The follow-up FineDiffusion combined tiered class embeddings, biases, and normalization layers for 10,000-class fine-grained generation. The general lesson: in a strong pre-trained generator, *re-aiming* learned features via bias/scale is often enough to chase a domain shift, provided the new domain is in the model's representational neighborhood. Photoreal-to-anime is in that neighborhood for SD; photoreal-to-painting-of-Pushkin probably isn't, and that's why the in-repo project_personalive_8step_monkey_patch thread had to push α up to 3 with a decoupled CLIP channel.

## Textual Inversion: embedding-only, style is hard

Gal et al. (ICLR 2023) proposed Textual Inversion: freeze the entire model, learn a *single* new embedding vector `v*` in CLIP's token space, ~768 floats per concept [7]. Strong for object identity, weak for style. The published limitation is exactly that "textual inversion fails to replicate style, possibly due to the limited capacity of the learned embeddings" — style needs both global (color tone) and local (brushstroke) decisions that can't be packed into one token's worth of capacity. Multi-token and per-timestep variants (DreamStyler, P+) extend the budget but still trail LoRA on style benchmarks. Textual Inversion is the right baseline to *rule out*: if a single embedding token suffices, the rest of the PEFT discussion is moot.

## GAN-side: layer-swap and one-shot adaptation

LoRA on StyleGAN is rare in the literature; the dominant paradigm is full fine-tune plus layer-targeted intervention. Pinkney's "network blending" took a StyleGAN fine-tuned on a target domain (e.g., ukiyo-e) and swapped *which* resolution layers came from the base versus the fine-tune [8]. The empirical finding is canonical: low-resolution (4×4–8×8) layers carry pose and broad structure; high-resolution (64×64–1024×1024) layers carry texture and color [8]. Karras et al.'s own style-mixing experiments in StyleGAN1/2 established the same hierarchy [13]. The practical consequence: if you want to stylize while preserving identity geometry, fine-tune the upper layers and freeze the lower ones; if you want to retarget geometry, do the inverse.

JoJoGAN (Chong & Forsyth, ECCV 2022) [11] and AgileGAN (SIGGRAPH 2021) [12] are the one-shot face-stylization references. JoJoGAN inverts a target into latent space, generates a style-mixed paired dataset, and fine-tunes the *full* StyleGAN with a discriminator-feature perceptual loss — 30 seconds of training. AgileGAN uses an inversion-consistent VAE plus full fine-tune. "Mind the Gap" (Zhu et al., ICLR 2022) is the closest to PEFT on GANs: a single-shot domain adaptation that fine-tunes the generator under CLIP-based domain-gap regularizers [14], beating StyleGAN-NADA in user studies. None of these are *LoRA* in the strict sense, but they share the spirit — small updates, targeted layers.

Concept Sliders (Gandikota et al., ECCV 2024) ports LoRA back to diffusion in a more axis-aware way: a LoRA trained from a *pair* of prompts learns a continuously-modulable direction (age, weather, expression, style) [10]. The FLUX implementation is experimental but works. The relevant transferable claim is that low-rank edits in diffusion correspond to interpretable, composable axes — useful as evidence that a small parameter budget *can* carry a style-shift axis if the training pair is well-chosen.

## PEFT on talking-head generators: silence

Face-vid2vid (Wang et al. NVIDIA 2021), the architectural ancestor of LivePortrait, and LivePortrait itself [15] are encoder-warp-decoder pipelines: an appearance feature volume `F(x)`, a flow/warp module `W`, and a SPADE decoder `G` synthesizing the final frame. LivePortrait explicitly uses a SPADE decoder rather than face-vid2vid's original decoder, citing its higher representational power. SPADE (Park et al., CVPR 2019) injects conditioning information through *spatially-adaptive normalization* in every decoder block — `γ(s)·BN(x) + β(s)` where `γ, β` are learned from a semantic map [16]. This is structurally analogous to StyleGAN's AdaIN: *style enters the decoder through normalization*, not the convolutions.

We found no published LoRA / BitFit / adapter results on face-vid2vid, LivePortrait, or close relatives. Stylization in LivePortrait is handled at corpus level — the published training recipe mixes "realistic and stylized still images with existing video data" during stage-1 base training [15]. There is no published post-hoc PEFT recipe; the LivePortrait paper's stage-2 training freezes F, W, and G entirely and tunes only the stitching/retargeting modules, which is the *opposite* of what we'd want for stylization.

## A minimum-parameter recipe for LivePortrait

Putting the literature together: cross-attention K/V is enough for diffusion personalization [2]; AdaIN/SPADE-style normalization is where style enters decoders [16,13]; fine layers carry texture and style in StyleGAN [8]; bias/scale alone can carry domain shift in strong generators [5,6]; LoRA on attention beats LoRA on convs in DiT-class models [6]. Mapped onto LP's F+W+G:

- **W (flow predictor)**: should stay frozen. It encodes motion geometry, not appearance. Touching it risks breaking the driver-source decoupling.
- **F (appearance encoder)**: photoreal-trained on FFHQ-class data. For a stylized anchor it produces an out-of-distribution feature volume, but the volume is still *geometrically* correct (it just has the wrong "skin" of features). A small LoRA on F's later blocks (where texture statistics dominate) could re-aim the features, but a full fine-tune of F risks corrupting the geometry that W consumes.
- **G (SPADE decoder)**: this is where style empirically lives in every related architecture. The SPADE `γ`/`β` modulators are the direct analog of CustomDiffusion's K/V — they're the *only* path by which conditioning shapes the output. A LoRA on G's SPADE projections (the small MLPs that produce `γ`, `β` from the feature volume) plus bias/scale on G's convs is the most theoretically defensible minimum-parameter intervention.

Mapping this to the option set: **Option A (LoRA on G only)** is the highest-leverage starting point, by direct analogy to CustomDiffusion-on-cross-attention and to Pinkney's layer-swap (style in the fine layers). **Option C (LoRA on F + G)** is the realistic fallback if the stylized anchor's feature volume is too far OOD for G alone to compensate — F's last 1–2 blocks get LoRA, the rest stays frozen. **Option D and E** (full F+G fine-tune) inherit DreamBooth's overfitting risk on a small stylized corpus and should be reserved for the case where A and C both visibly fail. **Option B (LoRA on F only)** is the least defensible: it asks the encoder to re-paint texture statistics that G's normalization will partially wash away anyway.

Concrete: start with rank-8 LoRA on G's SPADE projection layers plus bias-only training on G's conv blocks (BitFit-style); freeze F and W entirely; train against a paired (photoreal-driver, stylized-output) corpus with perceptual + identity-preserving loss. Target trainable footprint: well under 1% of LP's parameter count. The literature is silent on whether this works — but every adjacent result says it's where to look first.

## Sources

[1] Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," ICLR 2022. https://arxiv.org/abs/2106.09685
[2] Kumari et al., "Multi-Concept Customization of Text-to-Image Diffusion" (Custom Diffusion), CVPR 2023. https://www.cs.cmu.edu/~custom-diffusion/
[3] Shuttleworth et al., "LoRA vs Full Fine-tuning: An Illusion of Equivalence," 2024. https://arxiv.org/html/2410.21228v3
[4] Ruiz et al., "DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation," CVPR 2023; HuggingFace PEFT DreamBooth-LoRA guide. https://huggingface.co/docs/peft/main/en/task_guides/dreambooth_lora
[5] Ben-Zaken et al., "BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models," ACL 2022. https://aclanthology.org/2022.acl-short.1/
[6] Xie et al., "DiffFit: Unlocking Transferability of Large Diffusion Models via Simple Parameter-Efficient Fine-Tuning," ICCV 2023. https://arxiv.org/abs/2304.06648
[7] Gal et al., "An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion," ICLR 2023. https://textual-inversion.github.io/
[8] Pinkney, "StyleGAN Network Blending," 2020. https://www.justinpinkney.com/stylegan-network-blending/
[9] cloneofsimo, "LoRA for Stable Diffusion," GitHub. https://github.com/cloneofsimo/lora
[10] Gandikota et al., "Concept Sliders: LoRA Adaptors for Precise Control in Diffusion Models," ECCV 2024. https://arxiv.org/abs/2311.12092
[11] Chong & Forsyth, "JoJoGAN: One Shot Face Stylization," ECCV 2022. https://arxiv.org/abs/2112.11641
[12] Song et al., "AgileGAN: Stylizing Portraits by Inversion-Consistent Transfer Learning," SIGGRAPH 2021. https://dl.acm.org/doi/abs/10.1145/3450626.3459771
[13] Karras et al., "A Style-Based Generator Architecture for Generative Adversarial Networks" (StyleGAN), CVPR 2019; style-mixing analysis. https://arxiv.org/abs/1812.04948
[14] Zhu et al., "Mind the Gap: Domain Gap Control for Single Shot Domain Adaptation for GANs," ICLR 2022. https://arxiv.org/abs/2110.08398
[15] Guo et al., "LivePortrait: Efficient Portrait Animation with Stitching and Retargeting Control," 2024. https://arxiv.org/html/2407.03168
[16] Park et al., "Semantic Image Synthesis with Spatially-Adaptive Normalization" (SPADE/GauGAN), CVPR 2019. https://arxiv.org/abs/1903.07291
