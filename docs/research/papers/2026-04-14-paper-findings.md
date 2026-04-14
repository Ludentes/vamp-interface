# Paper Findings Log — 2026-04-14 Session

Persistent record of primary-source verification results from the vamp-interface Tier 1 + Tier 1b deep-read sessions. Each entry captures the paper ID, what we verified, the load-bearing facts extracted, and the verdict for our use case. Do not re-run deep reads on these papers unless new evidence contradicts an entry.

## How to read this log

- **Verified** = agent fetched primary source (arXiv PDF, ar5iv HTML, or local PDF) and extracted the facts directly.
- **Provisional** = agent used a workaround (indexed snippets, abstract pages, secondary sources). Flagged items need re-verification.
- **Killed** = option dropped from our design space. Includes the reason and the minimum bar to reopen.
- Each entry ends with a "safe to skip re-reading unless…" clause.

---

## Arc2Face (arXiv:2403.11641, ECCV 2024 Oral) — VERIFIED

**Read:** 2026-04-14, direct PDF + ar5iv.

**Conditioning architecture.** Prompt template `"photo of a ⟨id⟩ person"` tokenized to 5 tokens `{e1, e2, e3, ŵ, e5}`. Only `ŵ ∈ ℝ^768` carries identity: the 512-d ArcFace vector is **zero-padded to 768-d** and substituted into the `⟨id⟩` slot. No learned projection — direct substitution. Ablated against a 4-layer MLP in Fig. 9; direct substitution + fine-tuned CLIP text encoder wins on identity similarity.

**Text channel is dead post-fine-tuning.** Paper: *"exclusively adheres to ID-embeddings, disregarding its initial language guidance."*

**Training.** Stage 1: ~21M images from WebFace42M (1M identities), super-resolved to 448×448 with GFPGAN v1.4, 5 epochs. Stage 2: FFHQ + CelebA-HQ at 512×512, 15 epochs. 8× A100, batch 4/GPU, AdamW, LR 1e-6. ArcFace extractor frozen IR-100.

**Inference.** DPM-Solver, 25 steps, CFG=3.0. Community ~1-2 s per 512×512 image on consumer GPU. **Corrected by Vox2Face reading below**: on A800 80GB, DPM-Solver++ 25 steps runs ~17-18 s per batch of 4 → ~4.3 s per image. Benchmark on actual target hardware before committing to caching budget.

**Interpolation claim.** Section 4.4 "Averaging ID Features" shows one qualitative interpolation grid with **no quantitative metrics**. PCA analysis (Fig. 8) finds ArcFace space needs 300-400 components for facial fidelity — high intrinsic dimensionality.

**License.** CC-BY-NC. Non-commercial only. Weights: `huggingface.co/FoivosPar/Arc2Face`. Code: `github.com/foivospar/Arc2Face`.

**Verdict for vamp:** Primary backbone. Projection target is unambiguous: `qwen_1024 → ArcFace_512`.

**Safe to skip re-reading unless** we consider a CC-BY-NC alternative or need to benchmark inference latency on a new GPU class.

---

## Vox2Face (MDPI *Information* 17(2):200, Feb 2026) — VERIFIED

**Read:** 2026-04-14, full PDF at `docs/research/papers/vox2face.pdf` (user-provided, Akamai blocked direct fetch).

**Authors:** Qiming Ma, Yizhen Wang, Xiang Sun, Jiadi Liu, Gang Cheng, Jia Feng, Rong Wang, Fanliang Bu. People's Public Security University of China + Henan Police College. Funding: Henan Provincial Dept of Public Security, Henan Police College.

### Architecture

**Speaker encoder:** Audio Spectrogram Transformer (Gong et al., Interspeech 2021), AudioSet-pretrained weights, 128-bin log-Mel spectrograms, 16 kHz, 1.25 s clips with 50% overlap, 25 ms frame length. `v_spk` is the `[CLS]` token output. Stage I: "we train only gϕ and the top layers of the speaker encoder." Stage II: AST status not explicitly stated but implied frozen (only gϕ + LoRA are optimized).

**Mapping network gϕ:** 3-layer MLP, hidden width 1024, SiLU activation, LayerNorm, dropout 0.1. Final linear to 512-d followed by `ℓ²`-normalization to unit hypersphere.

**Backbone:** Arc2Face on SD 1.5, fully frozen. LoRA r=16, scale 16, on attention projections (Q/K/V/O) of **both** the text encoder and the UNet.

### Stage I — Cross-modal identity alignment (L_id)

**Equation 8 — AM-Softmax:**
```
L_AM = −log[ exp(s(cos θ_y − m)) / (exp(s(cos θ_y − m)) + Σ_{j≠y} exp(s cos θ_j)) ]
```
Margin m and scale s **not stated numerically** in the paper body. AM-Softmax classification head attached to `ẑ_arc` (the predicted identity).

**Equation 9 — InfoNCE:**
```
L_InfoNCE = −(1/B) log[ exp(cos(ẑ_arc^(i), z_arc^(i))/τ) / Σ_{j=1..B} exp(cos(ẑ_arc^(i), z_arc^(j))/τ) ]
```
In-batch negatives. Temperature τ and λ_AM/λ_InfoNCE weights **not stated numerically**.

**Equation 10 — Combined:** `L_id = λ_AM L_AM + λ_InfoNCE L_InfoNCE`.

### Stage II — Diffusion self-consistency (L_self)

Pipeline: given `ẑ_arc`, compute `c_gen = P(ẑ_arc)` via the Arc2Face projection head. Sample noise `z_T ∼ N(0,I)`, forward diffuse to `z_t`, denoise with LoRA-augmented UNet to recover `z_0`, decode via VAE to `I_gen`. Extract `z_self = f_arc(I_gen)` via frozen ArcFace, then `c_self = P(z_self)`.

**Equation 14 — Self-consistency loss:**
```
L_self = E_{t,ε}[ w(t) · || ε_θ(z_t, c_gen, t) − ε_θ(z_t, c_self, t) ||² ]
```
`w(t)` is a timestep-dependent weighting function (exact form not stated). **Stop-gradient on `c_self`** and the entire path used to compute it. Gradients from `L_self` flow only through `c_gen` back to `gϕ` and `ψ_txt, ψ_unet` (LoRA params).

**Equation 15 — Total:** `L_total = λ_id L_id + λ_self L_self`.

### Training schedule

- **Stage I (identity pre-alignment):** steps 0 → 10,000. Arc2Face + ArcFace + CLIP text encoder + UNet + VAE all frozen. Train only `gϕ` + top layers of AST. AdamW, LR 1e-4, weight decay 0.01, batch 128, cosine LR decay. **8.4 hours** on single NVIDIA A800 80GB.
- **Stage II (diffusion self-consistency):** steps 10,000 → 30,000 (20k steps). LoRA injected into attention projections of both text encoder and UNet. Optimize `gϕ + ψ_txt + ψ_unet`. AdamW, LR 1e-6, batch 16. Timestep truncation, gradient clipping, weight warm-up. **34.2 hours** on A800.
- **Total wall-clock:** ~42.6 hours on single A800 80GB.
- **Inference:** DPM-Solver++ 25 steps, 17-18 s per batch of 4 images → ~4.3 s per image on A800.

### Dataset

**HQ-VoxCeleb (SF2F [31]).** 3638 identities, 8028 HQ face images (~505×505), 609,700 speech segments. Split by identity: 2890 train / 370 val / 378 test. Each training sample: speech segment + frontal, neutral-expression face image.

### Results (Table 2, headline + full)

| Method | Cosine↑ | L2↓ | R@5↑ | R@10↑ | VFS↑ |
|---|---|---|---|---|---|
| Speech2face [13] | 0.213 | 1.20 | 4.6 | 8.7 | 13.43 |
| AST_165 [13] | 0.216 | 1.17 | 5.7 | 10.5 | 13.78 |
| Voice2Face [11] | 0.259 | 1.26 | 8.7 | 12.3 | 15.03 |
| SF2F [31] | 0.295 | 1.14 | 14.6 | 29.8 | 18.82 |
| **Vox2Face** | **0.322** | **1.09** | **17.3** | **32.1** | **23.21** |

### LoRA rank ablation (Table 3)

| r | Cosine | L2 | R@5 | R@10 | VFS |
|---|---|---|---|---|---|
| 4 | 0.314 | 1.14 | 15.5 | 27.3 | 22.92 |
| 8 | 0.317 | 1.12 | 16.6 | 30.2 | 23.09 |
| **16** | **0.322** | **1.09** | **17.3** | **32.1** | **23.21** |
| 32 | 0.323 | 1.09 | 17.0 | 31.7 | 23.23 |

Saturates at r=16.

### Component ablation (Table 4) — CRITICAL for vamp-interface

| Setting | Cosine | L2 | R@5 | R@10 | VFS |
|---|---|---|---|---|---|
| w/o ArcFace | 0.242 | 1.17 | 13.4 | 25.6 | 19.65 |
| w/CosReg | 0.317 | 1.10 | 12.9 | 22.5 | 21.64 |
| w/o DSC | 0.296 | 1.12 | 16.2 | 30.4 | 21.87 |
| **full** | **0.322** | **1.09** | **17.3** | **32.1** | **23.21** |

- **w/o ArcFace**: no ArcFace distillation at all. Shallow MLP maps speech → Arc2Face conditioning space, trained with "simple L2 regression and reconstruction loss." Tanks to 0.242 cosine. This is the FLOOR when you skip ArcFace alignment.
- **w/CosReg**: replace AM-Softmax + InfoNCE with plain cosine regression between `ẑ_arc` and `z_arc`. Cosine similarity stays close (0.317) but **retrieval tanks** (R@10: 22.5 vs 32.1). Pointwise regression approximates the target on average but fails to enforce class boundaries and batch-level pairing structure.
- **w/o DSC**: Stage I only, no Stage II. Cosine 0.296, R@10 30.4, VFS 21.87. Modest drops; Stage I alone works reasonably well.
- **No ablation of "Stage II alone without Stage I"** — the paper does NOT report what SDS does when training from scratch without the supervised alignment anchor. **This matters for vamp-interface:** my earlier plan assumed Stage II could be used in isolation. The paper provides no evidence for this.

### No metrics reported (gaps we should know about)

- **No FID, no LPIPS, no perceptual diversity metric.** The paper explicitly flags LPIPS diversity analysis as future work.
- **No interpolation / continuity experiments.** Smoothness between identity vectors is not measured.
- **No identity retention under perturbation** (e.g., sweep gϕ output by ε, measure degradation).

### Limitations (Figure 4 failure taxonomy)

(i) Age mismatch, (ii) hallucinated / inconsistent sunglasses, (iii) illumination/color-cast errors. Conservative fallback to neutral face when speech is short or noisy.

### Data/code availability

**Data Availability Statement** (verbatim): "The data presented in this study are openly available in the VoxCeleb repository and the HQ-VoxCeleb dataset, with documentation and access information provided at https://www.robots.ox.ac.uk/~vgg/data/voxceleb/ and https://github.com/BAI-Yeqi/SF2F_PyTorch."

**No Vox2Face code is released.** No GitHub repo, no HuggingFace model. The DAS lists only the *datasets*. Implementation is "based on PyTorch 3.10 and HuggingFace diffusers" — that's tools, not their code.

### Verdict for vamp-interface

**Architectural template: partially transferable.**

- **Stage I as-is is NOT transferable** to vamp-interface. AM-Softmax needs identity labels. InfoNCE needs positive pairs. We have neither — no ground-truth face per job posting. The `w/o ArcFace` ablation (0.242 cosine with direct L2 regression) shows what happens when ArcFace distillation is skipped entirely: severe degradation.
- **Stage II SDS self-consistency recipe is transferable in principle**, but the paper does NOT run a Stage-II-alone ablation. Whether SDS trains stably from a random gϕ init without Stage I's supervised anchor is an open empirical question. My earlier "Stage II is the transferable piece" claim was too strong.
- **LoRA recipe transfers cleanly:** r=16, scale 16, Q/K/V/O on text encoder + UNet, LR 1e-6, AdamW.
- **Inference cost is ~4.3 s per image on A800**, substantially higher than the ~1-2 s community number. Update caching budget.
- **No code or weights released.** Must re-implement from the paper.

**Safe to skip re-reading unless** a code release appears at the author's page or we need to check a specific equation detail.

---

## HyperFace (arXiv:2411.08470, Boutros et al., Idiap, late 2024 / 2025) — VERIFIED

**Read:** 2026-04-14 via ar5iv HTML (primary source), two independent extractions.

### The packing objective — exact form

Primary objective: `max min_{i≠j} d(x_ref,i, x_ref,j)` subject to `‖x_ref,k‖₂ = 1`, with `d(·,·) = cosine distance`, `x_ref ∈ ℝ^512`. This is the well-known Tammes / spherical-code / Fejes Tóth max-min-angle objective on the unit hypersphere.

**BUT the actual trained loss is REGULARIZED:**
```
L = − d(x_ref,i*, x_ref,j*) + α · (1/n_id) · Σ_k min_{x_g ∈ gallery} d(x_ref,k, x_g)
```
where `(i*, j*)` is the current-step closest pair, and the second term is a pull-toward-face-manifold regularizer against a gallery of real-or-synthetic face embeddings. **Default α = 0.5. The regularizer is load-bearing** — without it, points drift off the face manifold and downstream Arc2Face generation quality collapses.

### What is trainable

The reference vectors are **directly optimized as free parameters**. Algorithm 1: initialize `X_ref` from StyleGAN-face-encoded vectors, then iteratively: find closest pair → compute L → `X_ref ← X_ref − Adam(∇L, λ)` → renormalize to sphere. **No generator or MLP is trained during packing. Packing is a pure pre-compute.**

### Reference set vs. regularization gallery

- **Reference vectors being optimized: fully synthetic.** Initialized from embeddings of random StyleGAN-generated faces, then gradient-updated. Targets are not tied to real identities.
- **Regularization gallery: separate, can be real OR synthetic.** Paper uses StyleGAN, LDM, or real BUPT images. Ablation: real BUPT > StyleGAN ≈ LDM for downstream FR accuracy. For vamp-interface, synthetic gallery is fine (we don't need SOTA FR accuracy).

### Downstream use

Packed `x_ref` vectors are fed into **Arc2Face**. Intra-class variation comes from Arc2Face's sampling noise (`z`, `v`), not from HyperFace. Default 64 images/identity, 10k identities → 640k images, ~174 GPU-hours on a 3090.

### Training cost

Single NVIDIA 3090. **10k IDs: ~6 hours. 20k: ~11 h. 30k: ~23 h. 50k: ~84 h.** O(n²) per step from pairwise distance matrix. For vamp at ~1k-10k IDs, trivial (minutes to a few hours).

### Code and license

Project page: `https://www.idiap.ch/paper/hyperface`. Code and generated datasets linked. Idiap typical license is research-only — verify before productizing. Method is simple enough (~50 lines PyTorch) to re-implement from Algorithm 1.

### THE KEY FINDING FOR VAMP-INTERFACE — CONTINUITY CONFLICT

**HyperFace's objective is explicitly separation-maximizing. This is the OPPOSITE of vamp-interface's continuity hypothesis `d(face_A, face_B) ≈ C · d(emb_A, emb_B)`.**

The packing objective maximizes the minimum pairwise angle — it actively *pushes neighbors apart*. If we assign similar qwen embeddings to adjacent packed vectors, adjacency in packed-space is not adjacency in any semantically meaningful sense. The packing actively destroys local structure.

**My earlier composition claim "HyperFace Stage I + Vox2Face Stage II" solves the pairing problem but sacrifices continuity.** This is the single most important finding from the HyperFace deep read, and it invalidates the Step 2 Option (d) recommendation from the earlier session.

**Possible reconciliation:** Use HyperFace only to pick `~N` well-spread *anchor* faces (like a codebook), then have `qwen → ArcFace` produce points via interpolation among the k-nearest anchors (not regression to a single packed target). This gives anchor coverage from HyperFace and local smoothness from the interpolation step. But this is a new architectural decision, not a direct HyperFace application.

### Other caveats

1. **Assignment problem.** HyperFace doesn't tell you which packed vector corresponds to which semantic concept. Nearest-unused-target, Hungarian, or OT assignment needs a design.
2. **N must be chosen up-front.** Adding a new identity later requires re-running.
3. **ArcFace-512 lock-in.** Swapping to ElasticFace/AdaFace/CurricularFace requires re-packing.
4. **Gallery dimension match.** Gallery must also be in ArcFace-512 (not qwen space).
5. **Initialization matters.** Starting from Gaussian noise on the sphere converges slower than from StyleGAN-face embeddings.

### Verdict for vamp-interface (REVISED — see theory constraints doc)

**Initial verdict (too strong):** "Option (d) killed because HyperFace is separation-maximizing and incompatible with continuity."

**Corrected verdict after the regime analysis (Statement 1 in [docs/research/2026-04-14-vamp-theory-constraints.md](../2026-04-14-vamp-theory-constraints.md)):** HyperFace is not a path killer. The earlier argument conflated "HyperFace's property" (max-min-angle packing) with "the regime in which HyperFace's output is used" (discrete exact regression vs. interpolation vs. soft assignment). Only Regime A (discrete exact regression) produces piecewise-constant behavior, and even then the behavior is *monotonic* which is all the theory pivot (Statement 2) actually requires. Regime B (interpolation) is smooth but doesn't use HyperFace's max-separation property. Regime C is intermediate.

**Under the identifier-not-readout reframe** (Statement 2), the identity channel only needs monotonic similarity (similar qwen → similar face, different qwen → different face, direction-preserving not distance-preserving). Regime A satisfies this trivially: within-cluster = 0 distance, between-cluster ≥ min-pairwise-angle. HyperFace is therefore a **viable Step 2 option** for the vamp-interface rebuild, essentially generalizing the original fixed-anchor thesis from one anchor to N-anchors-deterministically-packed.

**Residual problems to analyze post-compact:** cluster granularity, within-cluster information budget, gallery choice, license, assignment function design. See blind-alleys entry 9 for the full list.

**Safe to skip re-reading unless** we commit to HyperFace implementation and need specific details on Algorithm 1, the α-regularizer weighting, or the gallery ablation numbers.

---

## Boundary Diffusion (arXiv:2302.08357, Zhu et al., NeurIPS 2023) — VERIFIED

**Read:** 2026-04-14, primary source.

**Venue:** NeurIPS 2023 (**NOT CVPR 2023** as earlier drafts claimed).

**Method:** Linear SVM on `h_t` activations at a searched mixing timestep `t_m`. Edit direction = unit normal vector of the hyperplane: `d(n, x_tm) = nᵀ x_tm`. Edit applied as **symmetric one-shot shift**: `x'_tm = x_tm + ζ · d(n, x_tm)`, *once*, at `t_m`. Pre-`t_m`: DDIM inversion. Post-`t_m`: stochastic sampling resumes.

**NOT class-mean difference.** NOT Asyrp-style asymmetric per-step P_t modification. These are architecturally different from what earlier drafts claimed.

**Supervision:** Attribute-labeled images (e.g., CelebA-HQ smile/gender/age). Binary labels per training image for SVM fit. "Training-free" means no *network* training; SVM fit still needs labels.

**Backbones validated:** DDPM, iDDPM on CelebA-64, CelebA-HQ-256, LSUN-church, LSUN-bedroom, AFHQ-Dog-256. **Never Stable Diffusion. Never CFG. Never identity-fine-tuned backbones.** Paper: *"the ability for unseen domain image editing is relatively limited compared to other learning-based methods."*

**Code:** `github.com/L-YeZhu/BoundaryDiffusion`. License needs confirmation.

**Verdict for vamp:** NOT a drop-in for Step 4A on Arc2Face. A faithful port is a research bet (Step 4C). Mean-difference heuristic applied via Asyrp asymmetric formulation is an alternative that has no Boundary-Diffusion pedigree — it's just the cheapest h-space thing worth a 2-hour spike.

**Safe to skip re-reading unless** we commit to the faithful port.

---

## RigFace (arXiv:2502.02465 v3 Nov 2025, Wei et al.) — VERIFIED

**Read:** 2026-04-14, primary source + URL verification.

**Architecture:** Identity Encoder is a full second SD 1.5 UNet, initialized from SD weights, **fully fine-tuned** (no LoRA, no frozen layers). Identity injected as **per-layer spatial feature maps** via FaceFusion: concat along width → joint self-attention → slice off denoising half → continue. Applied at every transformer block. **Not a vector — a feature pyramid.**

**Input:** requires a pixel-space reference image, VAE-encoded, passed through the Identity UNet. No embedding input port. Spatial Attribute Provider (SAP) adds DECA Lambertian renderings for lighting+pose and Deep3DRecon ψ coefficients for expression via an Attribute Rigger (8-channel 4×4 stride-2 conv).

**Training:** ~30k image pairs from Aff-Wild, 100k steps, batch 4/GPU × 2 GPU, 12 wall-clock hours on 2× MI250X (≈24 GPU-hours).

**Quality:** 59.3% of RigFace expression edits rated more realistic than ground truth in human perceptual study.

**Availability (verified via URL check 2026-04-14):**
- `github.com/weimengting/RigFace` **returns 404** (not a real repo).
- `huggingface.co/mengtingwei/rigface` **exists**: 8.82 GB safetensors, Apache-2.0, README is a stub pointing to the broken GitHub URL.
- **arXiv v3 (Nov 2025) advertises neither repo.** The author appears to have silently uploaded weights to HF without referencing them in the paper.
- **No inference code, no training code, no demo, no project page.** Adopters must reverse-engineer the loader from the paper.

**Verdict for vamp:** NOT a viable fallback.
1. No embedding input port — requires a pixel reference image, which is the Arc2Face-shaped problem we're trying to solve.
2. Per-layer spatial feature conditioning is worst-case for local Lipschitz behavior.
3. Weights-only availability makes it a real training-port project, not a swap-in.

**Safe to skip re-reading unless** RigFace releases inference code or a compatible derivative appears.

---

## NoiseCLR (arXiv:2312.05390, Dalva & Yanardag, CVPR 2024) — VERIFIED

**Read:** 2026-04-14, primary source.

**Contrastive objective (Eq. 5):** InfoNCE on `Δε_k^n = ε_θ(x_t^n, d_k) − ε_θ(x_t^n, φ)`. Positives: same direction across different images. Negatives: different directions on same image. Operates at **UNet predicted-noise output (ε-space), not h-space.**

**Where directions live:** **Pseudo-tokens in cross-attention text-conditioning slot.** Plugged into CFG at inference: `ε̄_θ(x_t, c, d_e) = ε̃_θ(x_t, c) + λ_e (ε_θ(x_t, d_e) − ε_θ(x_t, φ))`. Textual-inversion-adjacent.

**Backbone:** **Vanilla SD 1.5 only.** No DreamBooth, no LoRA, no Arc2Face, no identity fine-tunes. Zero fine-tuned-backbone validation.

**Cost:** ~7 h for 100 face directions on 1× L40 (48 GB). N=100 unlabeled images default. FFHQ / AFHQ-Cats / Stanford Cars.

**Code:** `github.com/gemlab-vt/NoiseCLR`. **Apache-2.0.** Project page: `noiseclr.github.io`.

**Verdict for vamp:** **Not a viable alternative to paired h-space direction finding on Arc2Face.**
1. Wrong slot — learns pseudo-tokens in a CLIP space that Arc2Face has semantically repurposed for ArcFace identity.
2. Wrong geometry — ε-output via CFG, not h-space.
3. Never validated on fine-tuned / identity-locked backbones.
4. Unsupervised discovery — no guarantee "uncanny" is among the K=100 directions (which are typically age, race, glasses, lipstick, hair on FFHQ).

**Safe to skip re-reading unless** a fine-tune-validated variant appears.

---

## Asyrp (arXiv:2210.10960, Kwon et al., ICLR 2023 Oral) — VERIFIED

**Read:** 2026-04-14, primary source (ar5iv).

**h-space:** UNet bottleneck layer 8 — *"the bridge of the U-Net, not influenced by any skip connection, has the smallest spatial dimension with compressed information."* SD 1.5: approximately 8×8×1280 ≈ 80k dim.

**Method:** Train small 1×1 conv `f_θ` with timestep conditioning via **CLIP directional loss** (Gal et al., StyleGAN-NADA):
```
L^(t) = λ_CLIP · L_direction(P_t^edit, y_ref; P_t^source, y_source) + λ_recon · |P_t^edit − P_t^source|
```
Training: **1000 samples, S=40 timesteps, 1 epoch, ~20 min on 3× RTX 3090.**

**Asymmetric reverse (Theorem 1):** modify only predicted-x0 term `P_t`, leave noise term `D_t` unchanged. Symmetric edits destroy each other.

**Editing interval:** `[T, t_edit]` where `t_edit` is chosen by `LPIPS(x, P_t_edit) = 0.33`.

**Claimed properties (and where they're validated — this is the critical 2026-04-14 correction):**
- Homogeneity: *"same Δh leads to same effect on different samples"*. Validated on unconditional DDPM.
- Linearity: *"linearly scaling Δh controls the magnitude of attribute change, even with negative scales"*. Fig. 7 on unconditional DDPM.
- Timestep consistency: *"Δh_t is **roughly consistent** across different timesteps"* (Section 5.3). **The hedge is load-bearing** — support is that a time-invariant global Δh_t^global "also yields similar results", NOT strict equivalence.

**Architectures tested:** DDPM++, iDDPM, ADM. **All unconditional pixel-space diffusion or class-conditional ADM.** Datasets: CelebA-HQ, AFHQ-Dog, LSUN-church, LSUN-bedroom, MetFaces.

**Stable Diffusion / latent diffusion / CFG: NOT tested.** Paper explicitly lists *"h-space in latent diffusion models"* as **future work**.

**Validated edits on face models:** Smiling, sad, angry, tanned, disgusted, makeup, curly hair, bald, young, **zombie**, identity swap, style transfer. *Glasses are absent* — the 8×8 bottleneck is too coarse for small localized features.

**Prerequisites:** DDIM deterministic sampling (η=0). SD default DPM-Solver requires switching.

**Verdict for vamp:** **Primary candidate for Step 4A with mandatory α-sweep gate.** Linearity/homogeneity are NOT established on any SD-family or CFG backbone. "Zombie" is exactly the direction we want, but on an unconditional DDPM, not a CFG'd identity-locked SD 1.5 fine-tune. Run α-sweep spike first.

**Safe to skip re-reading unless** a published replication on latent diffusion appears.

---

## HAAD (arXiv:2507.17554, July 2025) — VERIFIED

**Read:** 2026-04-14, primary source.

**What it is:** Adversarial-attack paper. PGD-based adversarial perturbations on input images whose gradients are restricted to the h-space reconstruction loss. Purpose: poison downstream few-shot personalization (LoRA + DreamBooth, Custom Diffusion) to disrupt unauthorized face customization.

**Backbone:** Stable Diffusion 1.5 (vanilla).

**Datasets:** CelebA-HQ (10 IDs × 20 img), WikiArt.

**CRITICAL CORRECTION TO EARLIER PROJECT NOTE:** HAAD is **NOT** a semantic-direction-transfer study. Earlier project memory claimed it "implicitly confirms h-space survives few-shot fine-tuning" — that overstates what the paper shows. HAAD demonstrates only that h-space gradients remain informative enough to *attack* fine-tuned SD derivatives. It does **not** demonstrate Asyrp-style homogeneity/linearity transfer.

**Verdict for vamp:** Does not support the Asyrp-on-SD transfer claim. Remove from "supporting evidence" column.

**Safe to skip re-reading unless** we need the adversarial-defense angle specifically.

---

## PhotoMaker (arXiv:2312.04461, TencentARC) — VERIFIED

**Read:** 2026-04-14, primary source (ar5iv).

**Input:** *"a few ID images to be customized"* — reference images, N ≥ 1.

**Encoder:** CLIP ViT-L/14 + additional projection layer → CLIP-image token per reference.

**Conditioning:** N encoded tokens **"concatenated along the length dimension to form the stacked id embedding"**. Not a single vector. Identity is co-determined by text class words ("a man/woman") alongside identity tokens.

**Backbone:** SDXL + LoRA residuals on attention layers. Backbone not fully fine-tuned.

**Verdict for vamp:** **NOT viable as continuous-identity fallback.**
1. Input space is a token stack, not a vector.
2. Designed to average out multi-image noise — a single synthetic token is off-manifold.
3. Identity is not fully determined by the identity tokens — text class word steers it. Breaks continuous-identity property.

**Safe to skip re-reading unless** a single-vector variant is released.

---

## InstantID (arXiv:2401.07519) — VERIFIED

**Read:** 2026-04-14, primary source (ar5iv).

**Architecture (three components):**
1. ID embedding projection (ArcFace → tokens).
2. Decoupled cross-attention module (IP-Adapter-style image prompt branch).
3. IdentityNet: ControlNet variant taking **spatial landmark map** (five facial keypoints) as image input, ArcFace embedding replacing text tokens in its cross-attention.

**Backbone:** SDXL, frozen. IdentityNet + decoupled cross-attention trained.

**Interpolation:** Fig. 11 in appendix shows ArcFace-space slerp between two *real extracted* embeddings, with all three branches active. Not evidence of local Lipschitz behavior on synthetic vectors through the identity branch alone.

**Verdict for vamp:** **NOT viable as continuous-identity fallback.**
1. IdentityNet needs *both* an ArcFace vector *and* a 2D landmark spatial map at inference.
2. Decoupled cross-attention needs an IP-Adapter CLIP-image token from a real reference.
3. Cannot isolate "just the ID channel" with a projected qwen vector.

**Safe to skip re-reading unless** a decoupled variant without landmark+image requirements appears.

---

## StyleGAN3 FFHQ W-space (reference + agent A analysis) — VERIFIED

**Read:** 2026-04-14, agent analysis of e4e (2102.02766), pSp (2008.00951), ReStyle (2104.02699) literature.

**Property:** FFHQ-trained StyleGAN2/3 have the smoothest known face latent space. W-space interpolation is locally Lipschitz by construction — this is the exact property our Step 5 pre-flight needs.

**The encoder-inversion literature's reconstruction penalty (e4e / pSp / ReStyle distortion-editability tradeoff) applies to *real-image inversion*, not to our use case.** We are learning `qwen → W` from scratch on synthetic training pairs; we're not inverting a specific real image. The reconstruction penalty is a symptom of a different problem.

**Critical choice:** target **native W** (not extended w+). Native W codes stay on the trained StyleGAN manifold and inherit smoothness for free. w+ gives more expressive capacity but drifts off manifold.

**Verdict for vamp:** **Viable as primary fallback or even primary.** Locally-Lipschitz latent by construction, single deterministic forward pass (no sampler stochasticity), no ArcFace dependency, no CC-BY-NC license issue (StyleGAN3 is Nvidia Source Code License — non-commercial research OK, verify commercial terms). Caveats: dated photorealism vs SDXL/FLUX, fixed square portrait crop.

**Safe to skip re-reading unless** we commit to StyleGAN3 and need to evaluate specific encoders.

---

## InfiniteYou (arXiv:2503.16418, ByteDance, Mar 2025) — PROVISIONAL

**Read:** 2026-04-14 via agent summary. **Not directly verified against PDF.**

**Claim:** FLUX DiT backbone + InfuseNet residual injection of ArcFace-derived identity features. **Code and weights released** at `github.com/bytedance/InfiniteYou`.

**Caveats (from agent):** InfuseNet residual injection into DiT blocks is a different conditioning shape than Arc2Face's cross-attention token substitution. Continuity properties are not characterized in the paper.

**UNVERIFIED:**
- Exact InfuseNet architecture (is it really residual injection? what layers?)
- License terms (may or may not be commercial-friendly)
- Whether ArcFace embeddings can be supplied directly, or if reference images are required
- Local Lipschitz behavior

**Verdict for vamp:** **Newer-backbone upgrade candidate, but needs first-hand read before committing.** Queue as a T1b.10 residual.

**Safe to skip re-reading — FALSE.** Read this before adopting as a primary alternative.

---

## FEM / Realistic Face Reconstruction (arXiv:2602.13168, AAAI 2026) — PROVISIONAL

**Read:** 2026-04-14 via agent summary.

**Claim:** Uses Kolmogorov-Arnold Networks (KAN) to map arbitrary face-recognition embeddings into Arc2Face's native identity space. Reported MSE loss, Adam 1e-3, 20 epochs.

**Needs ground-truth (source_embedding, target_face) pairs** via matching FR pass. Same pairing problem as Vox2Face.

**Verdict for vamp:** Methodologically adjacent to our Step 2 MLP question — worth reading for the KAN-vs-MLP design choice. Not a direct pipeline template.

**Safe to skip re-reading unless** we adopt KAN over MLP.

---

## 2026 successor scan — VERIFIED (negative result)

**Scope:** arXiv 2601-2604.

**Finding:** **No 2026 paper deprecates Arc2Face or releases a new identity-conditioned foundation model** with code+weights on a modern backbone. 2026 activity is dominated by face-swapping (AlphaFace, APPLE, DreamID-V), forensics, sampling-time tricks (AdaptDiff AAAI student abstract), and downstream applications of existing Arc2Face / PuLID / InstantID / InfiniteYou.

**Three 2026 papers build on top of Arc2Face rather than replacing it:**
- Arc2Morph (arXiv:2602.16569) — morphing attack tool
- FEM (arXiv:2602.13168) — FR embedding → Arc2Face reconstruction
- IdGlow (arXiv:2603.00607) — multi-subject extension

**No ArcFace embedding replacement in 2026.** DINOv3 (arXiv:2508.10104, Aug 2025) is used for general diffusion and as a secondary appearance encoder (WithAnyone arXiv:2510.14975, Oct 2025) but **alongside** ArcFace, not replacing it.

**No 2026 paper reports quantitative LPIPS sweep across identity interpolations — this is a genuine research gap our Step 5 pre-flight would fill.**

**Verdict:** Arc2Face is still the reference anchor in early 2026. The rebuild is not anchoring on deprecated work. If SD 1.5 ceiling becomes limiting, InfiniteYou is the upgrade candidate.

**Coverage caveat:** `arxiv.org/list/cs.CV/2603` returned 404 in the agent's attempt; coverage is search-driven. A low-visibility paper could have escaped.

**Safe to skip re-reading unless** a new month of arXiv publications (2605+) changes the picture.

---

## LoRA weight-space editing (arXiv:2406.09413, 2024) — NOT YET READ

Flagged in the Asyrp-alternatives agent response but not read first-hand. Key claim to verify: fitting PCA / linear classifiers on ~65k LoRA fine-tune weights to extract semantic directions. Alternative latent for Step 4A if h-space fails.

**Queue:** T1b.9.

---

## Changelog

- **2026-04-14** — Initial log created late in session after user directive "save results of research every time". Entries for Arc2Face, Vox2Face (full PDF), HyperFace (full), Boundary Diffusion, RigFace, NoiseCLR, Asyrp, HAAD, PhotoMaker, InstantID, StyleGAN3 W, InfiniteYou (provisional), FEM (provisional), 2026 scan.
